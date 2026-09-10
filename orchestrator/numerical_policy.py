"""Independent, evidence-bound numerical safety gate for production promotion."""
from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Event
from urllib.parse import urlparse
from pathlib import Path
import shutil
import subprocess
import tempfile
import uuid

from long_horizon.remote_numerical import PREFIX, validate_suite, validation_schedule
from .numerical_suite import resolve_suite
from .infrastructure_retry import InfrastructureUnavailable, check_review_service, check_transport, retry_infrastructure
from long_horizon.store import VERIFY_DIR

CHECKS = {"input_domain", "precision_and_reductions", "nonlinear_and_quantization",
          "routing_and_boundaries", "distribution_coverage"}
DRIVER = Path(__file__).resolve().parents[1] / "long_horizon" / "remote_numerical.py"
PROMPT = Path(__file__).with_name("prompts") / "numerical_review.md"
HARNESS = Path(__file__).resolve().parents[1] / "reference" / "atrex_bench_test_kernel.py"


def remote_agate(campaign):
    if campaign.sandbox_ssh:
        return False
    if campaign.sandbox_profile:
        return True
    endpoint = campaign.sandbox_url or os.environ.get("AGATE_URL", "")
    if not endpoint:
        # Respect agate's configured loopback endpoint without exposing credentials.
        config = Path.home() / ".atrex" / "config.json"
        if config.is_file():
            endpoint = json.loads(config.read_text()).get("url", "")
    return urlparse(endpoint).hostname not in {"localhost", "127.0.0.1", "::1"}


def gate_mode(campaign):
    selected = getattr(campaign, "numerical_gate", "auto")
    if selected == "auto":
        return "thorough" if remote_agate(campaign) else "light"
    if selected not in {"light", "thorough"}:
        raise ValueError("numerical gate must be auto, light or thorough")
    return selected


def evidence_files(workspace, private, suite_path):
    files = {"candidate/kernel.py": workspace / "kernel.py", "evaluator.py": workspace / "test_kernel.py",
             "numerical_suite.json": suite_path, "driver.py": DRIVER, "transport.py": HARNESS,
             "numerical_review.md": PROMPT}
    for name in ("input.py", "reference.py", "shapes.json", "metadata.json"):
        path = private / name if (private / name).is_file() else workspace / name
        if name != "metadata.json" or path.is_file():
            files["trusted/" + name] = path
    for name in ("agent_problem.json", "solution.json", "README.md"):
        if (workspace / name).is_file():
            files[("candidate/" if name == "solution.json" else "trusted/") + name] = workspace / name
    suite = json.loads(suite_path.read_text())
    for index, name in enumerate(suite.get("evaluator_files", [])):
        path = (private / name).resolve()
        files[f"trusted/evaluator/{index}-{path.name}"] = path
    return files


def evidence_digest(files):
    digest = hashlib.sha256(b"numerical-safety-v2\0")
    for name, path in sorted(files.items()):
        digest.update(name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def validate_evaluation(payload, suite, shapes, schedule):
    if not isinstance(payload, dict) or payload.get("schema_version") != 1 or payload.get("all_pass") is not True or payload.get("error"):
        raise ValueError("numerical distribution evaluation failed")
    rows = payload.get("runs", [])
    if [r.get("case_id") for r in rows] != [p["case_id"] for p in schedule]:
        raise ValueError("numerical evaluation omitted, duplicated or reordered distribution cases")
    for row, plan in zip(rows, schedule):
        expected_probes = len({json.dumps(shapes[sid].get("input_kwargs") or {}, sort_keys=True)
                               for sid in plan["shape_ids"]}) * len(plan["seeds"]) * suite["world_size"]
        if (row.get("passed") is not True or row.get("exit_code") != 0
            or not isinstance(row.get("result"), dict) or row["result"].get("all_pass") is not True
            or row.get("expected_probes") != expected_probes or row.get("observed_probes") != expected_probes
            or row.get("shape_count") != len(plan["shape_ids"]) or row.get("seeds") != plan["seeds"]
            or row.get("selection_digest") != plan["selection_digest"]
            or row.get("world_size") != suite["world_size"]):
            raise ValueError(f"incomplete numerical evidence for {row.get('case_id')}")


def validate_review(payload, digest, supplied_files=None):
    supplied_files = supplied_files or {"candidate/kernel.py", "trusted/reference.py", "evaluation.json", "numerical_suite.json"}
    if not isinstance(payload, dict) or payload.get("schema_version") != 1 or payload.get("evidence_digest") != digest:
        raise ValueError("numerical review is missing or bound to different evidence")
    items = payload.get("checks", [])
    if not isinstance(items, list) or not all(isinstance(i, dict) for i in items) or len(items) != len(CHECKS) or {i.get("id") for i in items} != CHECKS:
        raise ValueError("numerical review omitted or duplicated required checks")
    errors = []
    for item in items:
        evidence = item.get("evidence", [])
        if (item.get("decision") not in {"allow", "reject"} or not isinstance(item.get("reason"), str) or not item["reason"].strip()
            or not isinstance(evidence, list) or not evidence or not all(isinstance(e, str) for e in evidence)
            or any(e.split(":", 1)[0] not in supplied_files or ":" not in e for e in evidence)
            or not any(e.startswith("candidate/kernel.py:") for e in evidence)
            or not any(e.startswith(("trusted/", "evaluation.json:", "numerical_suite.json:")) for e in evidence)):
            raise ValueError(f"numerical check lacks source and contract/evaluation evidence: {item.get('id')}")
        if item["decision"] == "reject":
            errors.append(f"numerical safety {item['id']}: {item['reason']}")
    if payload.get("verdict") != ("reject" if errors else "allow"):
        raise ValueError("numerical review verdict disagrees with checks")
    return errors


def numerical_violations(campaign, workspace):
    # Imported here to preserve the campaign/session module initialization order.
    from .session_io import _sandbox_command, run_session
    private = Path(campaign.private_reference_dir or workspace)
    record = {"schema_version": 1, "accepted": False}
    directory = workspace / VERIFY_DIR / ("numerical-" + uuid.uuid4().hex)
    directory.mkdir(parents=True)
    try:
        suite_path = resolve_suite(campaign, workspace, private)
        suite = validate_suite(json.loads(suite_path.read_text()))
        files = evidence_files(workspace, private, suite_path)
        mode = gate_mode(campaign)
        source_digest = evidence_digest(files)
        digest = hashlib.sha256((source_digest + ":" + mode).encode()).hexdigest()
        record["evidence_digest"] = digest
        # Cache only within this supervisor process, after both gates passed.
        cache = getattr(campaign, "_numerical_review_cache", set())
        if digest in cache:
            record.update(accepted=True, cached=True)
            return []
        shapes = json.loads(files["trusted/shapes.json"].read_text())
        schedule = validation_schedule(suite, shapes, digest, mode)
        coverage = {"mode": mode, "coverage": suite.get("coverage", "compact"), "total_shapes": len(shapes),
                    "planned_rank_probes": sum(len(p["shape_ids"]) * len(p["seeds"]) * suite["world_size"] for p in schedule),
                    "selected_shapes_per_case": [len(p["shape_ids"]) for p in schedule]}
        record["coverage"] = coverage
        driver = directory / "test_kernel.py"
        shutil.copy2(DRIVER, driver)
        if "atrex-bench/run_eval" in (workspace / "test_kernel.py").read_text():
            snapshot = directory / "snapshots" / "evaluator.py"
            snapshot.parent.mkdir()
            shutil.copy2(HARNESS, snapshot)
        # Keep each allocation within the gateway's 600-second command limit.
        # Correctness probes never enter the ABBA timing aggregate.
        evaluation = {"schema_version": 1, "runs": [], "all_pass": True}
        record["evaluation"] = evaluation
        specs = []
        for index, case in enumerate(suite["cases"]):
            request = directory / f"request-{index:04d}.json"
            request.write_text(json.dumps({"suite": suite, "case_ids": [case["id"]],
                                           "rotation": digest, "mode": mode, "per_case_timeout": 540}))
            specs.append((index, case, request))
        is_remote = remote_agate(campaign)
        # Concurrent submissions can wait for admission without consuming their
        # worker execution budget. Match the sandbox's existing queue allowance.
        from tools.sandbox import DEFAULT_QUEUE_WAIT_GRACE
        queue_grace = int(os.environ.get("ATREX_SANDBOX_QUEUE_WAIT_GRACE", str(DEFAULT_QUEUE_WAIT_GRACE)))
        wall_timeout = 600 + 240 + queue_grace if is_remote else None
        cancel = Event()
        def evaluate_case(spec):
            index, case, request = spec
            if cancel.is_set():
                raise subprocess.SubprocessError("numerical batch cancelled")
            print(f"[numerical-safety] mode={mode} case={case['id']} ({index + 1}/{len(specs)})", flush=True)
            try:
                process = _sandbox_command(
                    workspace, campaign.sandbox_hardware, campaign.sandbox_profile, campaign.sandbox_url,
                    600, ["python3", str(driver.relative_to(workspace)), str(request.relative_to(workspace))],
                    ssh=campaign.sandbox_ssh, ssh_init=campaign.sandbox_ssh_init,
                    health_command=campaign.sandbox_health_command, gateway_kind="dev",
                    private_reference_dir=campaign.private_reference_dir, cancel_event=cancel,
                    wall_timeout=wall_timeout)
            except subprocess.TimeoutExpired as exc:
                raise InfrastructureUnavailable("GPU transport wait deadline exceeded") from exc
            batch = None
            for line in process.stdout.splitlines():
                if line.startswith(PREFIX):
                    batch = json.loads(line[len(PREFIX):])
            if process.returncode or not isinstance(batch, dict):
                if not isinstance(batch, dict):
                    check_transport(process)
                raise ValueError(f"numerical evaluator produced no result (exit={process.returncode}): {process.stderr[-1000:]}")
            return index, batch

        def run_case(spec):
            return retry_infrastructure(
                workspace, f"numerical:{digest}:{spec[1]['id']}",
                lambda: evaluate_case(spec), cancel=cancel)

        # Each agate case owns a separate allocation. Submit all cases immediately;
        # gateway quotas/queueing, rather than a local worker cap, govern admission.
        workers = len(specs) if is_remote else 1
        coverage["concurrent_jobs"] = workers
        batches = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_case, spec) for spec in specs]
            try:
                for future in as_completed(futures):
                    index, batch = future.result()
                    batches[index] = batch
                    evaluation["runs"] = [row for i in sorted(batches) for row in batches[i].get("runs", [])]
                    if batch.get("all_pass") is not True or batch.get("error"):
                        evaluation["all_pass"] = False
                        if batch.get("error"):
                            evaluation["error"] = batch["error"]
                        raise ValueError(f"numerical distribution {specs[index][1]['id']} failed; see numerical_result.json")
            except BaseException:
                cancel.set()
                for future in futures:
                    future.cancel()
                raise
        validate_evaluation(evaluation, suite, shapes, schedule)
        with tempfile.TemporaryDirectory(prefix="atrex-numerical-review-") as temporary:
            review_root = Path(temporary)
            review_files = {name: path for name, path in files.items()
                            if name not in {"trusted/shapes.json", "trusted/metadata.json"}}
            visible_digest = evidence_digest(review_files)
            for name, path in review_files.items():
                target = review_root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
            (review_root / "evaluation.json").write_text(json.dumps(evaluation, indent=2))
            (review_root / "review_request.json").write_text(json.dumps({"evidence_digest": digest, "coverage": coverage}))
            def review_once():
                (review_root / "numerical_review.json").unlink(missing_ok=True)
                result = run_session(review_root, PROMPT.read_text(), timeout=getattr(campaign, "numerical_review_timeout", 600),
                                     agent_cli=campaign.agent_cli, reasoning_effort="high", agent_plugins=False)
                campaign._account(result, "independent numerical safety review")
                check_review_service(result)
                return result
            retry_infrastructure(workspace, f"numerical-review:{digest}", review_once)
            if evidence_digest({name: review_root / name for name in review_files}) != visible_digest:
                raise ValueError("numerical reviewer modified supplied evidence")
            if json.loads((review_root / "evaluation.json").read_text()) != evaluation:
                raise ValueError("numerical reviewer modified evaluation evidence")
            review = json.loads((review_root / "numerical_review.json").read_text())
            record["review"] = review
            errors = validate_review(review, digest, set(review_files) | {"evaluation.json", "numerical_suite.json"})
        if evidence_digest(files) != source_digest:
            raise ValueError("candidate or numerical contract changed during validation")
        record["errors"] = errors
        record["accepted"] = not errors
        if not errors:
            cache.add(digest)
            campaign._numerical_review_cache = cache
        return errors
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as exc:
        record["errors"] = [f"production numerical validation blocked: {exc}"]
        return record["errors"]
    finally:
        (directory / "numerical_result.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
