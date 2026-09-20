"""Turn numerical review suggestions into probes and coding-agent feedback.

A reviewer requests bounded experiments, not an admission verdict. Only measured
failures ask the coding agent to repair the candidate. Incomplete experiments stay
pending validation and never become correctness failures or successful receipts.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import uuid

from long_horizon.remote_numerical import PREFIX, validate_suite, validation_schedule
from long_horizon.store import VERIFY_DIR
from .durable_state import durable_write_json
from .infrastructure_retry import (
    check_review_service, check_transport, retry_infrastructure, retry_review,
)

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "long_horizon" / "remote_numerical.py"
PROMPT = Path(__file__).with_name("prompts") / "numerical_review.md"
HARNESS = ROOT / "reference" / "atrex_bench_test_kernel.py"


def _digest(files):
    digest = hashlib.sha256()
    for name, path in sorted(files.items()):
        digest.update(name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def _sources(campaign, workspace):
    from .session_io import _production_review_candidate_paths

    private = Path(campaign.private_reference_dir or workspace)
    files = {"instructions.md": PROMPT, "driver.py": DRIVER, "transport.py": HARNESS}
    for source in _production_review_candidate_paths(workspace):
        files["candidate/" + source.relative_to(workspace).as_posix()] = source
    for name in ("input.py", "reference.py", "agent_problem.json", "README.md"):
        path = private / name if (private / name).is_file() else workspace / name
        if path.is_file():
            files["trusted/" + name] = path
    # Private workload parameters are used only by the supervisor/remote evaluator.
    shapes = private / "shapes.json"
    return files, shapes


def _validate_review(value, digest):
    if not isinstance(value, dict) or value.get("schema_version") != 1 or value.get("evidence_digest") != digest:
        raise ValueError("supplemental review is bound to different evidence")
    if value.get("action") not in {"complete", "probe"}:
        raise ValueError("numerical reviewers must propose probes, not reject candidates")
    if not isinstance(value.get("summary"), str) or not value["summary"].strip():
        raise ValueError("supplemental review requires an evidence-based summary")
    suite = value.get("suite")
    if value["action"] == "probe":
        validate_suite(suite)
        for case in suite["cases"]:
            if case.get("shape_ids") is not None:
                raise ValueError("reviewers select public input constraints, not private shape IDs")
            evidence = case.get("evidence", [])
            if (not any(str(e).startswith("candidate/") for e in evidence)
                    or not any(str(e).startswith("trusted/") for e in evidence)):
                raise ValueError("each probe needs candidate and input-contract evidence")
    elif suite is not None:
        raise ValueError("complete reviews must not leave unexecuted probes")
    return value


def _request_review(campaign, workspace, files, digest, previous=None):
    from .session_io import run_session

    def review_once():
        with tempfile.TemporaryDirectory(prefix="atrex-numerical-advice-") as temporary:
            root = Path(temporary)
            for name, source in files.items():
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            request = {"evidence_digest": digest, "previous_validation": previous}
            (root / "review_request.json").write_text(json.dumps(request, indent=2))
            result = run_session(
                root, PROMPT.read_text(), timeout=campaign.production_review_timeout,
                agent_cli=campaign.agent_cli, reasoning_effort="high", agent_plugins=False,
            )
            campaign._account(result, "numerical supplemental-test planning")
            check_review_service(result)
            if _digest({name: root / name for name in files}) != digest:
                raise ValueError("numerical reviewer modified supplied evidence")
            value = json.loads((root / "numerical_review.json").read_text())
            return _validate_review(value, digest)

    return retry_review(workspace, f"numerical-advice:{digest}", review_once)


def _probe_status(batch, plan, suite, shapes):
    """A complete receipt is required before labeling a result a counterexample."""
    rows = batch.get("runs", [])
    if batch.get("error") or len(rows) != 1:
        return "needs_validation"
    row = rows[0]
    expected = len({json.dumps(shapes[sid].get("input_kwargs") or {}, sort_keys=True)
                    for sid in plan["shape_ids"]}) * len(plan["seeds"]) * suite["world_size"]
    if (row.get("case_id") != plan["case_id"]
            or row.get("selection_digest") != plan["selection_digest"]
            or row.get("shape_count") != len(plan["shape_ids"])
            or row.get("seeds") != plan["seeds"]
            or row.get("world_size") != suite["world_size"]
            or row.get("expected_probes") != expected
            or type(row.get("observed_probes")) is not int
            or not 0 <= row["observed_probes"] <= expected
            or not isinstance(row.get("result"), dict)):
        return "needs_validation"
    result = row["result"]
    if row.get("input_error"):
        return "needs_validation"
    # A single fully observed workload/seed can disprove correctness. Passing
    # still requires every requested workload, seed and rank to finish.
    if (0 < row.get("failed_probes", 0) <= row["observed_probes"]
            and result.get("all_pass") is False):
        return "needs_repair"
    if row["observed_probes"] != expected:
        return "needs_validation"
    if row.get("passed") is True and row.get("exit_code") == 0 and result.get("all_pass") is True:
        return "passed"
    if result.get("all_pass") is False:
        return "needs_repair"
    return "needs_validation"


def _run_probes(campaign, workspace, suite, shapes_path, digest, directory):
    from .session_io import _sandbox_command

    shapes = json.loads(shapes_path.read_text())
    schedule = validation_schedule(suite, shapes, digest, "thorough")
    driver = directory / "test_kernel.py"
    shutil.copy2(DRIVER, driver)
    if "atrex-bench/run_eval" in (workspace / "test_kernel.py").read_text():
        snapshot = directory / "snapshots" / "evaluator.py"
        snapshot.parent.mkdir()
        shutil.copy2(HARNESS, snapshot)
    results = []
    for index, plan in enumerate(schedule):
        if plan.get("status") == "unsupported":
            results.append({"case_id": plan["case_id"], "status": "unsupported",
                            "diagnosis": plan["diagnosis"]})
            continue
        request = directory / f"request-{index:04d}.json"
        durable_write_json(request, {
            "suite": suite, "case_ids": [plan["case_id"]], "rotation": digest,
            "mode": "thorough", "per_case_timeout": 540,
        })

        def execute():
            process = _sandbox_command(
                workspace, campaign.sandbox_hardware, campaign.sandbox_profile,
                campaign.sandbox_url, 600,
                ["python3", str(driver.relative_to(workspace)), str(request.relative_to(workspace))],
                ssh=campaign.sandbox_ssh, ssh_init=campaign.sandbox_ssh_init,
                health_command=campaign.sandbox_health_command, gateway_kind="dev",
                private_reference_dir=campaign.private_reference_dir,
            )
            check_transport(process)
            for line in process.stdout.splitlines():
                if line.startswith(PREFIX):
                    batch = json.loads(line[len(PREFIX):])
                    if process.returncode:
                        batch["error"] = f"probe transport exited {process.returncode}"
                    return batch
            raise ValueError(f"numerical probe produced no receipt (exit={process.returncode})")

        batch = retry_infrastructure(workspace, f"numerical-probe:{digest}:{plan['case_id']}", execute)
        status = _probe_status(batch, plan, suite, shapes)
        # Preserve comparison metrics and actual input-generation diagnostics, but
        # never surface private workload kwargs or raw evaluator output to the agent.
        rows = batch.get("runs", [])
        result = (rows[0].get("result") or {}) if rows else {}
        results.append({
            "case_id": plan["case_id"], "status": status,
            "metrics": result.get("numerical_metrics", {}),
            "expected_probes": rows[0].get("expected_probes") if rows else None,
            "observed_probes": rows[0].get("observed_probes") if rows else None,
            "failed_probes": rows[0].get("failed_probes", 0) if rows else 0,
            "exit_code": rows[0].get("exit_code") if rows else None,
            "diagnosis": batch.get("error") or (rows[0].get("input_error", "") if rows else ""),
        })
    status = ("needs_repair" if any(r["status"] == "needs_repair" for r in results)
              else "needs_validation" if any(r["status"] == "needs_validation" for r in results)
              else "advisory" if any(r["status"] == "unsupported" for r in results)
              else "passed")
    return {"status": status, "probes": results}


def supplemental_feedback(campaign, workspace):
    """Return repair/pending feedback, or an empty string when advice is resolved."""
    if campaign.optimization_mode != "production":
        return ""
    workspace = Path(workspace)
    files, shapes = _sources(campaign, workspace)
    digest = _digest(files)
    validation_digest = _digest({**files, "shapes.json": shapes, "evaluator.py": workspace / "test_kernel.py"})
    cache = getattr(campaign, "_supplemental_results", {})
    key = (str(workspace.resolve()), validation_digest)
    if key in cache:
        return cache[key]

    directory = workspace / VERIFY_DIR / ("supplemental-" + uuid.uuid4().hex)
    directory.mkdir(parents=True)
    # The plan stays in supervisor memory during agent repairs. Public copies are
    # evidence only; edits by a coding agent cannot weaken the pending probes.
    plans = getattr(campaign, "_supplemental_plans", {})
    plan_key = (str(workspace.resolve()), _digest({name: path for name, path in files.items()
                                                if not name.startswith("candidate/")}))
    review = plans.get(plan_key)
    plan_root = (Path(campaign.private_reference_dir) / ".atrex_numerical_advice"
                 if campaign.private_reference_dir
                 else Path(campaign.workspace) / ".atrex_long_horizon" / "numerical_advice")
    plan_path = plan_root / (hashlib.sha256(repr(plan_key).encode()).hexdigest() + ".json")
    if review is None and plan_path.is_file():
        review = json.loads(plan_path.read_text())
        validate_suite(review["suite"])
    record = {"schema_version": 1, "evidence_digest": validation_digest}
    try:
        if review is None:
            review = _request_review(campaign, workspace, files, digest)
        record["review"] = review
        if review["action"] == "complete":
            record["status"] = "passed"
        else:
            plans[plan_key] = review
            campaign._supplemental_plans = plans
            durable_write_json(plan_path, review, indent=2)
            for attempt in range(2):
                trial = directory / f"probe-{attempt}"
                trial.mkdir()
                try:
                    evaluation = _run_probes(campaign, workspace, review["suite"], shapes, digest, trial)
                except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as exc:
                    evaluation = {"status": "needs_validation", "diagnosis": str(exc)}
                record.setdefault("evaluations", []).append(evaluation)
                record["status"] = evaluation["status"]
                if evaluation["status"] != "needs_validation" or attempt == 1:
                    break
                # Input ABI/scheduling failures belong to the probe planner, not
                # the optimization agent. Keep the original risk in the request.
                replacement = _request_review(campaign, workspace, files, digest,
                                              {"review": review, "evaluation": evaluation})
                if replacement["action"] == "complete":
                    # A failed experiment is never silently reclassified as passed.
                    break
                if ({case["id"] for case in replacement["suite"]["cases"]}
                        != {case["id"] for case in review["suite"]["cases"]}):
                    raise ValueError("probe-plan repair must preserve the requested risk cases")
                review = replacement
                plans[plan_key] = review
                record["review"] = review
                durable_write_json(plan_path, review, indent=2)
        if _digest({**files, "shapes.json": shapes, "evaluator.py": workspace / "test_kernel.py"}) != validation_digest:
            raise ValueError("candidate or contract changed during supplemental validation")
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as exc:
        record.update(status="needs_validation", diagnosis=str(exc))

    durable_write_json(directory / "numerical_result.json", record, indent=2, ensure_ascii=False)
    feedback_path = workspace / VERIFY_DIR / "numerical_feedback.json"
    durable_write_json(feedback_path, record, indent=2, ensure_ascii=False)
    status = record["status"]
    print(f"[numerical-supplement] {status}; evidence={feedback_path}", flush=True)
    if status in {"passed", "advisory"}:
        # Keep the closed experiment as a regression probe for subsequent edits
        # and process restarts. Recertify against the new candidate; never reopen
        # the same advisory merely because a reviewer session is fresh.
        feedback = ""
    elif status == "needs_repair":
        feedback = (
            f"Supplemental numerical probes found a measured failure. Read {feedback_path.relative_to(workspace)} "
            "for the requested distributions and results; repair the candidate using the immutable reference, "
            "then rerun the usual evaluator and hand off the updated candidate. Do not edit the probes, "
            "evaluator or tolerances. The supervisor reruns the same requested probes after repair; "
            "passing them closes the suggestion without another numerical-review veto."
        )
    else:
        feedback = (
            f"Supplemental validation is incomplete; this is not a measured correctness failure. "
            f"See {feedback_path.relative_to(workspace)}. Preserve the candidate and report the validation "
            "blocker if it cannot be resolved within the public contract; do not modify the trusted harness."
        )
    # Pending infrastructure/evaluator results are retryable even for unchanged code.
    if status != "needs_validation":
        cache[key] = feedback
        campaign._supplemental_results = cache
    return feedback
