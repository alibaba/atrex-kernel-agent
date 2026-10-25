"""Native Agate Eval ABBA: exact A/B/B/A evidence, no Dev driver fallback."""
from __future__ import annotations

import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from aka.legacy.evaluation.abba import verification_schedule
from aka.light.evaluation.result_conversion import compile_failures
from aka.legacy.evaluation.shapes import batch_shape_ids, shape_batch_request
from aka.light.evaluation.evaluation_feedback import evaluation_feedback
from aka.light.evaluation.policy import BATCH_WORKERS, validate_abba_repeats
from aka.light.evaluation.public_result import correctness_summary
from aka.light.experiment.abba_checkpoints import AbbaBatchStore, validate_batch
from aka.light.experiment.store import (
    ABBA_BATCH_ROOT_ENV,
    BATCH_ROOT_ENV,
    JOB_ROOT_ENV,
    EvidenceUnavailable,
)


ABBA_WALL_CLOCK_TIMEOUT_REASON = "abba_run_wall_clock_timeout"
_ABBA_WALL_CLOCK_TIMEOUT = re.compile(
    r"^candidate exceeded [0-9]+(?:\.[0-9]+)?s wall-clock timeout$",
    re.IGNORECASE,
)


def _abba_wall_clock_timeout_failures(job: object) -> list[dict]:
    """Return exact native-ABBA run/Shape wall-clock failures.

    Agate's inner Eval calls every submitted side ``candidate``.  Keep that
    original message, but also attach the ABBA revision/label so callers do not
    mistake a timed-out A-side baseline for the optimization candidate.
    """
    if not isinstance(job, dict):
        return []
    result = job.get("result")
    abba = result.get("abba") if isinstance(result, dict) else None
    blocks = abba.get("sdk_results") if isinstance(abba, dict) else None
    if not isinstance(blocks, list):
        return []
    failures = []
    for block_index, block in enumerate(blocks):
        nested = block.get("abba") if isinstance(block, dict) else None
        runs = nested.get("runs") if isinstance(nested, dict) else None
        if not isinstance(runs, list):
            continue
        for run in runs:
            if not isinstance(run, dict) or not isinstance(run.get("result"), dict):
                continue
            raw = run["result"]
            performance = raw.get("performance")
            shapes = performance.get("shapes") if isinstance(performance, dict) else None
            verdicts = raw.get("passed")
            verdicts = verdicts.get("performance") if isinstance(verdicts, dict) else None
            shape_ids = set(shapes or {}) | set(verdicts or {})
            for shape_id in sorted(shape_ids, key=lambda item: str(item)):
                shape = shapes.get(shape_id) if isinstance(shapes, dict) else None
                verdict = verdicts.get(shape_id) if isinstance(verdicts, dict) else None
                messages = []
                if isinstance(shape, dict):
                    messages.append(shape.get("error"))
                if isinstance(verdict, dict):
                    messages.append(verdict.get("reason"))
                message = next(
                    (
                        value
                        for value in messages
                        if isinstance(value, str) and _ABBA_WALL_CLOCK_TIMEOUT.fullmatch(value.strip())
                    ),
                    None,
                )
                if message is None:
                    continue
                failures.append(
                    {
                        "block": block_index,
                        "run_index": run.get("index"),
                        "side": run.get("label"),
                        "revision": run.get("revision"),
                        "repeat": run.get("repeat"),
                        "shape_id": str(shape_id),
                        "message": message,
                    }
                )
    return failures


def _terminal_abba_timeout_error(attempts: list[list[dict]]) -> dict:
    failures = [
        dict(failure, attempt=index + 1)
        for index, rows in enumerate(attempts)
        for failure in rows
    ]
    message = failures[-1]["message"]
    retained = failures[:64]
    return {
        "error_class": "timeout",
        "reason": ABBA_WALL_CLOCK_TIMEOUT_REASON,
        "message": message,
        "details": {
            "attempts": len(attempts),
            "failures": retained,
            **(
                {"omitted_failures": len(failures) - len(retained)}
                if len(failures) > len(retained)
                else {}
            ),
        },
    }


def is_terminal_abba_timeout(value: object) -> bool:
    """Whether an Agent result records the exhausted one-retry policy."""
    if not isinstance(value, dict):
        return False
    error = value.get("error")
    details = error.get("details") if isinstance(error, dict) else None
    return bool(
        isinstance(error, dict)
        and error.get("reason") == ABBA_WALL_CLOCK_TIMEOUT_REASON
        and isinstance(details, dict)
        and details.get("attempts") == 2
    )


def _job_deadline_error(value: object) -> dict | None:
    """Return Agate's original job-deadline error from nested ABBA evidence.

    A native ABBA request can finish as a successful outer Eval job while one
    of its four evaluator runs terminates at the service deadline.  The outer
    envelope then has no ``error`` and the failed run is discovered only while
    parsing its SDK evidence.  Preserve the service's structured error instead
    of replacing it with the generic ``evidence_unavailable`` diagnostic.
    """
    if isinstance(value, str):
        return (
            {"reason": "timeout", "message": value, "error_class": "timeout"}
            if "job deadline exceeded" in value.casefold()
            else None
        )
    if isinstance(value, list):
        for item in value:
            if error := _job_deadline_error(item):
                return error
        return None
    if not isinstance(value, dict):
        return None
    message = value.get("message")
    if isinstance(message, str) and "job deadline exceeded" in message.casefold():
        error = {
            key: value[key]
            for key in ("reason", "message", "trace_id", "error_class", "details")
            if key in value
        }
        error.setdefault("reason", "timeout")
        error.setdefault("error_class", "timeout")
        return error
    # Traverse only evaluator result/error containers.  Do not inspect source,
    # request metadata, stdout or stderr while constructing Agent feedback.
    for key in (
        "error", "result", "abba", "sdk_results", "runs", "baseline",
        "candidate", "passed", "compile", "correctness", "performance",
        "shapes", "cases",
    ):
        if error := _job_deadline_error(value.get(key)):
            return error
    return None


def _job_deadline_process(
    job: dict, process: subprocess.CompletedProcess[str]
) -> subprocess.CompletedProcess[str] | None:
    """Project a nested deadline as a terminal Gateway error for the Agent."""
    error = _job_deadline_error(job.get("result"))
    if error is None:
        return None
    failed = dict(job)
    failed["status"] = "failed"
    failed["error"] = error
    return subprocess.CompletedProcess(
        process.args,
        process.returncode or 1,
        json.dumps(failed),
        process.stderr,
    )


def _summarize(raw: dict, shapes: list[str]) -> dict:
    """Same per-Shape median (ms -> us) and GeoMean as Runtime's ABBA ledger."""
    from aka.light.execution.gateway_jobs import complete_evaluation
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


def _failed_summary(raw: dict, shapes: list[str], error: dict | str) -> dict:
    """Retain completed diagnostics without inventing missing latency evidence."""
    result = {
        "all_pass": False,
        "latency_us_by_shape": {},
        "latency_us_geomean": None,
        "error": error,
        "failures": [error.get("message", str(error)) if isinstance(error, dict) else str(error)],
    }
    result["correctness"] = correctness_summary(raw, passed=False)
    result.update(evaluation_feedback(raw, shapes))
    return result


def _timed_out_abba_payload(
    job: dict, repeats: int, shapes: list[str], error: dict
) -> dict:
    """Create a completed negative ABBA result after the one retry is exhausted."""
    validate_abba_repeats(repeats)
    result = job.get("result") or {}
    blocks = (result.get("abba") or {}).get("sdk_results")
    if not isinstance(blocks, list) or len(blocks) != repeats // 2:
        raise ValueError("Native ABBA timeout result lacks complete SDK blocks")
    expected = [
        ("baseline", "A", 0),
        ("candidate", "B", 0),
        ("candidate", "B", 1),
        ("baseline", "A", 1),
    ]
    runs = []
    for block_index, block in enumerate(blocks):
        rows = (block.get("abba") or {}).get("runs") if isinstance(block, dict) else None
        if not isinstance(rows, list) or len(rows) != 4:
            raise ValueError("Native ABBA timeout result lacks four runs")
        for index, (row, (revision, label, repeat)) in enumerate(
            zip(rows, expected, strict=True)
        ):
            if (
                not isinstance(row, dict)
                or any(
                    row.get(key) != value
                    for key, value in {
                        "index": index,
                        "revision": revision,
                        "label": label,
                        "repeat": repeat,
                    }.items()
                )
                or not isinstance(row.get("result"), dict)
            ):
                raise ValueError("Native ABBA timeout run order differs from the request")
            raw = row["result"]
            try:
                summary = _summarize(raw, shapes)
                exit_code = 0
            except (ValueError, TypeError, KeyError, AttributeError):
                summary = _failed_summary(raw, shapes, error)
                exit_code = 1
            runs.append(
                {
                    "revision": "incumbent" if revision == "baseline" else "candidate",
                    "repeat": block_index * 2 + repeat,
                    "exit_code": exit_code,
                    "result": summary,
                }
            )
    payload = {
        "schema_version": 1,
        "transport": "agate_native_eval_abba",
        "runs": runs,
        "error": None,
        "terminal_failure": error,
    }
    return validate_batch(payload, verification_schedule(repeats), shapes)


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
    from aka.light.evaluation.operations import validate_comparison, comparison_run_timeout
    from aka.light.acceptance.verifier import merge_abba_batch_payloads
    from aka.light.evaluation.projection import abba, emit_gateway_error
    from aka.light.execution.gateway_jobs import eval_retry_kind, payload_identity

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
    # Recorded acceptance retries create a fresh Gateway Record after an
    # incomplete result.  Its per-record JOB_ROOT_ENV changes, so checkpoints
    # stored there cannot suppress already-completed Shape submissions.  The
    # Runtime supplies a Campaign-stable private root for recorded ABBA; retain
    # JOB_ROOT_ENV as the standalone/cooperative CLI fallback.
    checkpoint_root = (
        os.environ.get(BATCH_ROOT_ENV)
        or os.environ.get(ABBA_BATCH_ROOT_ENV)
        or os.environ.get(JOB_ROOT_ENV)
    )
    store = AbbaBatchStore(Path(checkpoint_root)) if checkpoint_root else None
    job_root = os.environ.get(JOB_ROOT_ENV)
    record_store = AbbaBatchStore(Path(job_root)) if job_root else None

    def retain_in_record(identity, saved):
        """Keep a replacement Record self-contained without another GPU run."""
        retained = saved
        if record_store is not None and (store is None or record_store.root != store.root):
            retained = record_store.save(
                identity,
                saved["payload"],
                stdout=saved["stdout"],
                stderr=saved["stderr"],
                returncode=saved["returncode"],
            )
        if record_store is not None:
            record_store.retain_terminal_job(identity, retained)

    def run_batch(ids):
        batch = shape_batch_request(request, ids)
        identity = {"transport": "agate_native_eval_abba", "request": payload_identity(batch),
                    "schedule": schedule, "shape_ids": ids}
        if store and (saved := store.load(identity)) is not None:
            retain_in_record(identity, saved)
            return saved["payload"], None

        def dispatch(attempt: int):
            retry_identity = (
                {"abba_wall_clock_retry": attempt} if attempt else None
            )
            if tree:
                from .source_transport import run as run_tree
                return run_tree(
                    gateway,
                    args,
                    batch,
                    "run",
                    queue_wait_grace,
                    identity_extra=retry_identity,
                )
            if executable:
                # Use native eval CLI flags, retaining configured profile credentials.
                # This never uploads/runs a Dev harness. --abba-repeats counts blocks.
                with tempfile.TemporaryDirectory(prefix="aka-native-abba-") as directory:
                    sidecars = Path(directory)
                    reference = sidecars / "reference"
                    reference.mkdir()
                    for field, filename in (
                        ("reference_py", "reference.py"),
                        ("input_py", "input.py"),
                        ("shapes", "shapes.json"),
                        ("metadata", "metadata.json"),
                        ("roofline", "roofline.json"),
                    ):
                        if field in batch["reference"]:
                            content = batch["reference"][field]
                            (reference / filename).write_text(
                                content if isinstance(content, str) else json.dumps(content),
                                encoding="utf-8",
                            )
                    command = gateway.build_typed_agate_command(
                        executable,
                        args,
                        workspace,
                        "run",
                        batch,
                        queue_wait_grace,
                        reference_dir=reference,
                        request_sidecar_dir=sidecars,
                    )
                    request_identity = {"kind": "eval", "request": batch}
                    if retry_identity:
                        request_identity.update(retry_identity)
                    return gateway.run_agate_with_cancel_retry(
                        agate=command,
                        executable=executable,
                        url=url,
                        gateway_profile=args.gateway_profile,
                        command_timeout=args.timeout,
                        wait_budget=args.timeout + queue_wait_grace,
                        request_identity=request_identity,
                        max_device_retries=10,
                        device_retry_delay=60,
                    )
            return gateway.run_direct_job(
                url=url,
                kind="eval",
                payload=batch,
                timeout=args.timeout,
                queue_wait_grace=queue_wait_grace,
                max_device_retries=10,
                device_retry_delay=60,
                identity_extra=retry_identity,
            )

        timeout_attempts = []
        for attempt in range(2):
            process = dispatch(attempt)
            job = gateway.parse_job_response(process.stdout or "")
            if not job or job.get("status") != "succeeded":
                return None, process
            timeout_failures = _abba_wall_clock_timeout_failures(job)
            if timeout_failures:
                timeout_attempts.append(timeout_failures)
                if attempt == 0:
                    print(
                        "[sandbox] Native ABBA run exceeded its wall-clock limit; "
                        "retrying the complete ABBA batch once.",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                error = _terminal_abba_timeout_error(timeout_attempts)
                try:
                    payload = _timed_out_abba_payload(
                        job, args.comparison_repeats, ids, error
                    )
                except (ValueError, TypeError, KeyError, AttributeError) as exc:
                    raise EvidenceUnavailable(
                        "Native ABBA timed out twice but returned malformed terminal evidence; "
                        "inspect the saved Agate Job"
                    ) from exc
                return payload, None
            try:
                payload = parse_native_abba(job, args.comparison_repeats, ids)
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                if deadline := _job_deadline_process(job, process):
                    return None, deadline
                raise EvidenceUnavailable(
                    "Native ABBA returned incomplete evidence; inspect the saved Job, not the Kernel"
                ) from exc
            if store:
                saved = store.save(
                    identity,
                    payload,
                    stdout=process.stdout,
                    stderr=process.stderr or "",
                    returncode=process.returncode,
                )
                retain_in_record(identity, saved)
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
            raise EvidenceUnavailable(
                "Native ABBA did not produce a measurement; completed Shape batches "
                "are retained and only incomplete Shapes will be retried"
            )
        return 1
    payload = merge_abba_batch_payloads([value for value, _ in results], schedule, shapes)
    value = abba(payload, schedule, shapes, args.comparison_repeats)
    print("[sandbox] ABBA_JSON=" + json.dumps(value))
    return 0 if value["correct"] else 1
