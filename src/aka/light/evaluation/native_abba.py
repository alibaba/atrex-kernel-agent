"""Native Agate Eval ABBA: exact A/B/B/A evidence, no Dev driver fallback."""
from __future__ import annotations

import json
import math
import os
import statistics
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from aka.legacy.evaluation.abba import verification_schedule
from aka.light.evaluation.result_conversion import compile_failures
from aka.legacy.evaluation.shapes import batch_shape_ids, shape_batch_request
from aka.light.evaluation.evaluation_feedback import evaluation_feedback
from aka.light.evaluation.policy import BATCH_WORKERS, validate_abba_repeats
from aka.light.evaluation.public_result import correctness_summary
from aka.light.supervisor.abba_checkpoints import AbbaBatchStore, validate_batch
from aka.light.supervisor.measurement_records import JOB_ROOT_ENV, EvidenceUnavailable


def _summarize(raw: dict, shapes: list[str]) -> dict:
    """Same per-Shape median (ms -> us) and GeoMean as Runtime's ABBA ledger."""
    from aka.light.supervisor.gateway_jobs import complete_evaluation
    if not complete_evaluation(raw, {"reference": {"shapes": dict.fromkeys(shapes)}}):
        raise ValueError("Native ABBA run has incomplete evaluator evidence")
    passed = raw.get("passed") or {}
    correctness = raw.get("correctness") or {}
    shape_results = correctness.get("shapes") or {}
    statuses = passed.get("correctness") or {}
    correct = (not raw.get("error") and not compile_failures(passed.get("compile"), shapes)
               and all(statuses.get(sid, {}).get("status") == "passed" and sid in shape_results for sid in shapes))
    latencies = {}
    performance = (raw.get("performance") or {}).get("shapes") or {}
    for sid in shapes:
        shape = performance.get(sid) or {}
        values = [sample["end_to_end_time_ms"] for sample in shape.get("samples", [])
                  if isinstance(sample, dict) and type(sample.get("end_to_end_time_ms")) in (int, float)
                  and math.isfinite(sample["end_to_end_time_ms"]) and sample["end_to_end_time_ms"] > 0]
        if values and not shape.get("error"):
            latencies[sid] = statistics.median(values) * 1000
    correct = bool(correct and len(latencies) == len(shapes) and shapes)
    result = {"all_pass": correct, "latency_us_by_shape": latencies,
              "latency_us_geomean": math.exp(statistics.mean(math.log(v) for v in latencies.values())) if correct else None,
              "error": None if correct else "compile, correctness, or performance did not pass"}
    result["correctness"] = correctness_summary(raw, passed=correct)
    result.update(evaluation_feedback(raw, shapes))
    return result


def parse_native_abba(job: dict, repeats: int, shapes: list[str]) -> dict:
    validate_abba_repeats(repeats)
    if not shapes or job.get("status") != "succeeded" or job.get("error"):
        raise ValueError("Native ABBA job did not complete")
    result = job.get("result") or {}
    blocks = (result.get("abba") or {}).get("sdk_results")
    if not isinstance(blocks, list) or len(blocks) != repeats // 2:
        raise ValueError("Native ABBA result lacks complete SDK blocks")
    expected = [("baseline", "A", 0), ("candidate", "B", 0),
                ("candidate", "B", 1), ("baseline", "A", 1)]
    runs = []
    for block_index, block in enumerate(blocks):
        rows = (block.get("abba") or {}).get("runs") if isinstance(block, dict) else None
        if not isinstance(rows, list) or len(rows) != 4:
            raise ValueError("Native ABBA block lacks four runs")
        for index, (row, (revision, label, repeat)) in enumerate(zip(rows, expected, strict=True)):
            if (not isinstance(row, dict) or any(row.get(k) != v for k, v in {
                    "index": index, "revision": revision, "label": label, "repeat": repeat}.items())
                    or not isinstance(row.get("result"), dict)):
                raise ValueError("Native ABBA run order or evaluator evidence differs from the request")
            runs.append({"revision": "incumbent" if revision == "baseline" else "candidate",
                         "repeat": block_index * 2 + repeat, "exit_code": 0,
                         "result": _summarize(row["result"], shapes)})
    payload = {"schema_version": 1, "transport": "agate_native_eval_abba", "runs": runs, "error": None}
    return validate_batch(payload, verification_schedule(repeats), shapes)


def compare(gateway, args, workspace: Path, queue_wait_grace: int) -> int:
    from aka.light.supervisor.operations import validate_comparison, comparison_run_timeout
    from aka.light.long_horizon.verifier import merge_abba_batch_payloads
    from aka.light.supervisor.projection import abba, emit_gateway_error
    from aka.light.supervisor.gateway_jobs import eval_retry_kind, payload_identity

    validate_comparison(args)
    if args.ssh or (workspace / "workload.jsonl").exists():
        raise ValueError("Native Eval ABBA requires an Agate Atrex-Bench input/Shape contract; SOL/SSH Dev fallback is disabled")
    from aka.light.workspace.sources import snapshot as source_snapshot, unpack, policy_at
    tree = policy_at(workspace) is not None
    baseline_source = source_snapshot(workspace / args.baseline_path) if tree else None
    baseline = (unpack(baseline_source)[1]["kernel.py"].decode() if tree else
                gateway.read_workspace_override(workspace, args.baseline_path, field="baseline-path", max_bytes=16 * 1024 * 1024))
    command = ["python3", "test_kernel.py"]
    for flag, value in (("--multi-seed", args.multi_seed), ("--timed-runs", args.timed_runs)):
        if value is not None:
            command += [flag, str(value)]
    limitation = gateway._typed_workspace_limitation(workspace, command, "run")
    if limitation:
        raise ValueError(f"Native ABBA cannot represent this workspace: {limitation}; no Dev fallback was submitted")
    request = gateway.build_typed_request(
        workspace, args.hardware, args.timeout, args.env, command, "run", evaluation_mode="full",
        requirements=args.requirement or (), deps_mode=args.deps_mode,
    )
    if not baseline.strip() or not request["candidate"]:
        raise ValueError("Native ABBA requires two non-empty Kernel sources")
    request["abba"] = {"baseline": baseline, "repeats": args.comparison_repeats // 2}
    if tree:
        from .source_transport import attach
        attach(request, workspace, side="baseline", source=baseline_source)
    request["runner_overrides"]["perf_timeout_s"] = comparison_run_timeout(args)
    shapes = sorted(request["reference"]["shapes"], key=gateway.shape_id_sort_key)
    if not shapes:
        raise ValueError("Native ABBA requires at least one Shape")
    if args.dry_run:
        print(json.dumps({"kind": "same_allocation_abba", "transport": "agate_native_eval_abba",
                          "shape_count": len(shapes), "comparison_repeats": args.comparison_repeats}))
        return 0
    executable = gateway.find_agate()
    url = args.url
    if not executable and not url:
        raise ValueError("Native ABBA requires a resolvable Agate Gateway URL; no Dev fallback was submitted")
    schedule = verification_schedule(args.comparison_repeats)
    store = AbbaBatchStore(Path(os.environ[JOB_ROOT_ENV])) if os.environ.get(JOB_ROOT_ENV) else None

    def run_batch(ids):
        batch = shape_batch_request(request, ids)
        identity = {"transport": "agate_native_eval_abba", "request": payload_identity(batch),
                    "schedule": schedule, "shape_ids": ids}
        if store and (saved := store.load(identity)) is not None:
            return saved["payload"], None
        if tree:
            from .source_transport import run as run_tree
            process = run_tree(gateway, args, batch, "run", queue_wait_grace)
        elif executable:
            # Use native eval CLI flags, retaining configured profile credentials.
            # This never uploads/runs a Dev harness. --abba-repeats counts blocks.
            with tempfile.TemporaryDirectory(prefix="aka-native-abba-") as directory:
                sidecars = Path(directory)
                reference = sidecars / "reference"
                reference.mkdir()
                for field, filename in (("reference_py", "reference.py"), ("input_py", "input.py"),
                                        ("shapes", "shapes.json"), ("metadata", "metadata.json"), ("roofline", "roofline.json")):
                    if field in batch["reference"]:
                        content = batch["reference"][field]
                        (reference / filename).write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
                command = gateway.build_typed_agate_command(executable, args, workspace, "run", batch, queue_wait_grace,
                                                           reference_dir=reference, request_sidecar_dir=sidecars)
                process = gateway.run_agate_with_cancel_retry(
                    agate=command, executable=executable, url=url, gateway_profile=args.gateway_profile,
                    command_timeout=args.timeout, wait_budget=args.timeout + queue_wait_grace,
                    request_identity={"kind": "eval", "request": batch},
                    max_device_retries=10, device_retry_delay=60,
                )
        else:
            process = gateway.run_direct_job(url=url, kind="eval", payload=batch, timeout=args.timeout,
                                            queue_wait_grace=queue_wait_grace,
                                            max_device_retries=10, device_retry_delay=60)
        job = gateway.parse_job_response(process.stdout or "")
        if not job or job.get("status") != "succeeded":
            return None, process
        try:
            payload = parse_native_abba(job, args.comparison_repeats, ids)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise EvidenceUnavailable("Native ABBA returned incomplete evidence; inspect the saved Job, not the Kernel") from exc
        if store:
            store.save(identity, payload, stdout=process.stdout, stderr=process.stderr or "")
        return payload, None

    batches = batch_shape_ids(shapes, args.shape_batch_size)
    with ThreadPoolExecutor(max_workers=min(BATCH_WORKERS, len(batches))) as pool:
        results = list(pool.map(run_batch, batches))
    failures = [process for _, process in results if process is not None]
    if failures:
        infrastructure = False
        for process in failures:
            emit_gateway_error(process, "same_allocation_abba")
            job = gateway.parse_job_response(process.stdout or "")
            infrastructure |= job is None or eval_retry_kind(job) is not None or job.get("status") in {"cancelled", "canceled"}
        if infrastructure:
            raise EvidenceUnavailable("Native ABBA did not produce a measurement; saved batches are retained for recovery")
        return 1
    payload = merge_abba_batch_payloads([value for value, _ in results], schedule, shapes)
    value = abba(payload, schedule, shapes, args.comparison_repeats)
    print("[sandbox] ABBA_JSON=" + json.dumps(value))
    return 0 if value["correct"] else 1
