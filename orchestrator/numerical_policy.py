"""Turn numerical review suggestions into probes and coding-agent feedback.

A reviewer requests bounded experiments, not an admission verdict. Only measured
failures ask the coding agent to repair the candidate. Planner timeouts may skip
additional testing after standard correctness passes; they are never probe passes.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import uuid
import threading
import time
from concurrent.futures import CancelledError

from supervisor.workspace import read_input, publish
from .session_tail import read_regular_bytes

from long_horizon.remote_numerical import MAX_REL_L2, PREFIX, validate_suite, validation_schedule
from reference.atrex_bench_test_kernel import _fp4_correctness_max_rel_l2
from .constants import SUPPLEMENTAL_PENDING_PREFIX, SUPPLEMENTAL_REPAIR_PREFIX
from .durable_state import durable_write_json
from .infrastructure_retry import (
    check_review_service, retry_infrastructure,
)

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "long_horizon" / "remote_numerical.py"
PROMPT = Path(__file__).with_name("prompts") / "numerical_review.md"
HARNESS = ROOT / "reference" / "atrex_bench_test_kernel.py"


class NumericalPlanningTimeout(RuntimeError):
    """The numerical planner timed out without a usable plan."""


class NumericalCancellation:
    """Outage waits and nested GPU requests do not outlive their report Session."""

    def __init__(self, runtime, parent=None):
        self.runtime, self.parent = runtime, parent

    def is_set(self):
        return self.runtime.closed or (self.parent is not None and (
            not self.runtime._live(self.parent)
            or self.parent.deadline is not None and time.monotonic() >= self.parent.deadline))

    def check(self):
        if self.is_set():
            raise CancelledError("Numerical validation cancelled; report remains unaccepted")

    def wait(self, seconds):
        deadline = time.monotonic() + seconds
        while not self.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            threading.Event().wait(min(1, remaining))
        return True

    def timeout(self, seconds):
        self.check()
        if self.parent is not None and self.parent.deadline is not None:
            return max(1, min(seconds, int(self.parent.deadline - time.monotonic())))
        return seconds


def validate_candidate(campaign, workspace, *, source=None, parent=None):
    """Freeze inputs outside the Agent workspace before planning or execution.

    The caller must have verified ordinary correctness of these candidate bytes.
    Kernel-only Episodes retain their pinned manifest; this migration does not
    reopen Agent Git access or permit manifest edits in candidate commits.
    """
    if campaign.optimization_mode != "production":
        return ""
    runtime = campaign._supervisor_runtime
    workspace = Path(workspace)
    key = hashlib.sha256(str(workspace.resolve()).encode()).hexdigest()
    state_root = (runtime.audit_root or runtime.root) / "numerical" / key
    # Report requests and recovered handoffs may reach the same validator.
    # Serialize plan updates without holding the Runtime's global auth lock.
    with runtime.lock:
        locks = getattr(runtime, "numerical_locks", {})
        lock = locks.setdefault(key, threading.RLock())
        runtime.numerical_locks = locks
    cancel = NumericalCancellation(runtime, parent)
    cancel.check()
    with lock:
        cancel.check()
        state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with tempfile.TemporaryDirectory(prefix="candidate-", dir=state_root) as temporary:
            frozen = Path(temporary)
            publish(frozen, "kernel.py", source if source is not None else read_input(workspace, "kernel.py"))
            private = Path(campaign.private_reference_dir) if campaign.private_reference_dir else None
            for name in ("input.py", "reference.py", "shapes.json", "metadata.json", "agent_problem.json",
                         "solution.json", "config.json", "definition.json", "workload.jsonl"):
                if private is not None and name != "solution.json" and (private / name).is_file():
                    data = read_input(private, name)
                else:
                    # Do not trust mutable native-workspace evaluator inputs.
                    result = subprocess.run(["git", "show", f"HEAD:{name}"], cwd=workspace,
                                            capture_output=True, check=False)
                    if result.returncode:
                        continue
                    data = result.stdout
                publish(frozen, name, data)
            if not all((frozen / name).is_file() for name in ("input.py", "shapes.json", "reference.py")):
                # The declarative probes require the native input ABI. SOL's
                # standard full-workload gate remains mandatory, not replaced.
                if (frozen / "workload.jsonl").is_file():
                    durable_write_json(state_root / "numerical_feedback.json", {
                        "status": "advisory", "reason": "supplemental input ABI unsupported for SOL",
                        "kernel_sha256": hashlib.sha256(read_input(frozen, "kernel.py")).hexdigest(),
                    })
                    return ""
                return f"{SUPPLEMENTAL_PENDING_PREFIX}: trusted input.py/reference.py/shapes.json unavailable"
            publish(frozen, "test_kernel.py", HARNESS.read_bytes())
            return supplemental_feedback(campaign, frozen, state_root=state_root,
                                         standard_correctness_passed=True, cancel=cancel)


def _digest(files):
    digest = hashlib.sha256()
    for name, path in sorted(files.items()):
        digest.update(name.encode() + b"\0" + read_input(path.parent, path.name) + b"\0")
    return digest.hexdigest()


def _sources(campaign, workspace):
    from .session_io import _production_review_candidate_paths

    private = workspace
    files = {"instructions.md": PROMPT, "driver.py": DRIVER, "transport.py": HARNESS}
    for source in _production_review_candidate_paths(workspace):
        files["candidate/" + source.relative_to(workspace).as_posix()] = source
    for name in ("input.py", "reference.py", "agent_problem.json"):
        path = private / name if (private / name).is_file() else workspace / name
        if path.is_file():
            files["trusted/" + name] = path
    # Private workload parameters are used only by the supervisor/remote evaluator.
    shapes = private / "shapes.json"
    return files, shapes


def _world_size(workspace):
    from supervisor.gateway import _distributed_evaluation_world_size
    path = workspace / "metadata.json"
    return _distributed_evaluation_world_size(json.loads(read_input(workspace, path.name)) if path.is_file() else {})


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


def _request_review(campaign, workspace, files, digest, previous=None, *, state_root, cancel):
    from .session_io import run_session
    from .supervisor_runtime import scrub_environment

    def review_once():
        cancel.check()
        timeout = campaign.production_review_timeout
        with tempfile.TemporaryDirectory(prefix="atrex-numerical-advice-") as temporary:
            root = Path(temporary).resolve()
            for name, source in files.items():
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            request = {"evidence_digest": digest, "previous_validation": previous,
                       "world_size": _world_size(workspace)}
            (root / "review_request.json").write_text(json.dumps(request, indent=2))
            result = run_session(
                root, PROMPT.read_text(), timeout=cancel.timeout(timeout),
                agent_cli=campaign.agent_cli, reasoning_effort="high", agent_plugins=False,
                extra_environment={**{key: "" for key in os.environ.keys() - scrub_environment(dict(os.environ)).keys()},
                    **campaign.agent_boundary_environment("numerical-review"),
                    "ATREX_SESSION_CAPTURE_DIR": str(state_root / "sessions" / uuid.uuid4().hex)},
            )
            campaign._account(result, "numerical supplemental-test planning")
            cancel.check()
            response = root / "numerical_review.json"
            record = {"evidence_digest": digest, "session_id": result.session_id,
                      "timeout_s": timeout, "exit_status": result.exit_status,
                      "timed_out": result.timed_out, "response_written": response.is_file()}
            record_path = state_root / f"numerical_planning-{uuid.uuid4().hex}.json"
            durable_write_json(record_path, record, indent=2)
            if _digest({name: root / name for name in files}) != digest:
                raise ValueError("numerical reviewer modified supplied evidence")
            # A timed-out CLI may already have written the requested plan before
            # hanging on its final response. Validate that artifact rather than
            # discarding it with the temporary session directory.
            if not result.timed_out:
                check_review_service(result)
            try:
                value = json.loads(read_regular_bytes(response, limit=128 * 1024 + 1))
                if response.stat().st_size > 128 * 1024:
                    raise ValueError("numerical plan exceeds 128 KiB")
                record["response"] = value
                _validate_review(value, digest)
                if value["action"] == "probe" and value["suite"]["world_size"] != request["world_size"]:
                    raise ValueError("probe world_size must match the Supervisor's trusted contract")
            except (OSError, ValueError, TypeError, KeyError) as exc:
                record["validation_error"] = str(exc)
                if result.timed_out:
                    raise NumericalPlanningTimeout(
                        "numerical supplemental-test planner timed out without a usable plan"
                    ) from exc
                check_review_service(result)
                raise
            else:
                record["validated"] = True
                if result.timed_out:
                    print("[numerical-supplement] recovered valid plan from timed-out reviewer", flush=True)
                return value
            finally:
                durable_write_json(record_path, record, indent=2, ensure_ascii=False)

    context = hashlib.sha256(json.dumps(previous, sort_keys=True).encode()).hexdigest()
    stage = f"numerical-advice:{digest}:{campaign.agent_cli}:{campaign.production_review_timeout}:{context}"
    return retry_infrastructure(campaign.workspace, stage, review_once, cancel=cancel)


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
    if row.get("input_error") or {"reference", "unknown"}.intersection(result.get("nonfinite_outputs", [])):
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


def _run_probes(campaign, workspace, suite, shapes_path, digest, directory, *, cancel):
    if suite["world_size"] != _world_size(workspace):
        raise ValueError("retained probe world_size differs from the trusted contract")
    shapes = json.loads(shapes_path.read_text())
    schedule = validation_schedule(suite, shapes, digest, "thorough")
    results = []
    for index, plan in enumerate(schedule):
        cancel.check()
        if plan.get("status") == "unsupported":
            results.append({"case_id": plan["case_id"], "status": "unsupported",
                            "diagnosis": plan["diagnosis"]})
            continue
        request = directory / f"request-{index:04d}.json"
        durable_write_json(request, {
            "suite": suite, "case_ids": [plan["case_id"]], "rotation": digest,
            "mode": "thorough", "per_case_timeout": 540,
            "max_rel_l2": _fp4_correctness_max_rel_l2(workspace) or MAX_REL_L2,
        })

        # Only a private, bounded input snapshot enters the normal Dev record
        # pipeline. Stable filenames preserve exact-request reuse across restarts.
        with tempfile.TemporaryDirectory(prefix="probe-", dir=directory) as temporary:
            probe = Path(temporary)
            for name in ("kernel.py", "solution.json", "input.py", "reference.py", "shapes.json",
                         "metadata.json", "agent_problem.json", "config.json", "definition.json", "workload.jsonl"):
                if (workspace / name).is_file():
                    publish(probe, name, read_input(workspace, name))
            publish(probe, "numerical_probe.py", DRIVER.read_bytes())
            publish(probe, "probe-request.json", request.read_bytes())
            response = retry_infrastructure(campaign.workspace,
                f"numerical-probe:{digest}:{hashlib.sha256(request.read_bytes()).hexdigest()}",
                lambda: campaign.measure_for_acceptance(probe, [
                    "--kind", "dev", "--no-sync", "--",
                    "python3", "numerical_probe.py", "probe-request.json",
                ], execution_timeout=600, parent=cancel.parent, reference_snapshot=probe), cancel=cancel)
        durable_write_json(directory / f"response-{index:04d}.json", response)
        batch, record_id = None, None
        for line in response["stdout"].splitlines():
            if line.startswith(PREFIX):
                batch = json.loads(line[len(PREFIX):])
            elif line.startswith("[sandbox] RECORD_JSON="):
                record_id = json.loads(line.split("=", 1)[1]).get("gateway_record_id")
        if not isinstance(batch, dict):
            raise ValueError("Recorded numerical probe produced no valid receipt")
        if response["exit_code"]:
            batch["error"] = "probe transport did not complete"
        status = _probe_status(batch, plan, suite, shapes)
        # Preserve comparison metrics and actual input-generation diagnostics, but
        # never surface private workload kwargs or raw evaluator output to the agent.
        rows = batch.get("runs", [])
        result = (rows[0].get("result") or {}) if rows else {}
        results.append({
            "case_id": plan["case_id"], "status": status, "gateway_record_id": record_id,
            "metrics": result.get("numerical_metrics", {}),
            "nonfinite_outputs": result.get("nonfinite_outputs", []),
            "expected_probes": rows[0].get("expected_probes") if rows else None,
            "observed_probes": rows[0].get("observed_probes") if rows else None,
            "failed_probes": rows[0].get("failed_probes", 0) if rows else 0,
            "exit_code": rows[0].get("exit_code") if rows else None,
            "diagnosis": (
                "non-finite output diagnostic has unknown roles; repair or verify the probe evaluator"
                if "unknown" in result.get("nonfinite_outputs", []) else
                "reference output is non-finite under the requested probe inputs; "
                "repair the input distribution or packed encoding, preserving the original risk case"
                if "reference" in result.get("nonfinite_outputs", []) else
                "candidate output is non-finite while the reference output is finite"
                if "candidate" in result.get("nonfinite_outputs", []) else
                batch.get("error") or (rows[0].get("input_error", "") if rows else "")
            ),
        })
    # Repair invalid reference inputs before asking the coding agent to act on
    # any other failing case. All retained cases are measured again afterwards.
    status = ("needs_validation" if any({"reference", "unknown"}.intersection(r.get("nonfinite_outputs", [])) for r in results)
              else "needs_repair" if any(r["status"] == "needs_repair" for r in results)
              else "needs_validation" if any(r["status"] == "needs_validation" for r in results)
              else "advisory" if any(r["status"] == "unsupported" for r in results)
              else "passed")
    return {"status": status, "probes": results}


def supplemental_feedback(campaign, workspace, *, state_root, standard_correctness_passed=False, cancel):
    """Return repair/pending feedback, or an empty string when advice is resolved."""
    if campaign.optimization_mode != "production":
        return ""
    workspace = Path(workspace)
    files, shapes = _sources(campaign, workspace)
    digest = _digest(files)
    validation_files = {**files, "shapes.json": shapes, "evaluator.py": workspace / "test_kernel.py"}
    validation_files.update({name: workspace / name for name in ("metadata.json", "config.json", "definition.json")
                             if (workspace / name).is_file()})
    validation_digest = _digest(validation_files)
    cache = getattr(campaign, "_supplemental_results", {})
    key = (str(state_root), validation_digest, standard_correctness_passed)
    if key in cache:
        return cache[key]

    directory = state_root / ("supplemental-" + uuid.uuid4().hex)
    directory.mkdir(parents=True)
    # Private plans survive repairs and restarts; public feedback is never state.
    plans = getattr(campaign, "_supplemental_plans", {})
    plan_key = (str(state_root), _digest({name: path for name, path in files.items()
                                                if not name.startswith("candidate/")}))
    review = plans.get(plan_key)
    # Persist outside agent worktrees, just like the private reference corpus.
    # Workspace feedback copies are never accepted as supervisor state.
    plan_root = state_root / "plans"
    plan_path = plan_root / (hashlib.sha256(repr(plan_key).encode()).hexdigest() + ".json")
    record = {"schema_version": 1, "evidence_digest": validation_digest,
              "comparison": {"metric": "relative_l2", "max_rel_l2":
                  _fp4_correctness_max_rel_l2(workspace) or MAX_REL_L2}}
    try:
        previous_path = state_root / "numerical_feedback.json"
        if previous_path.is_file():
            previous = json.loads(read_regular_bytes(previous_path, limit=1024 * 1024 + 1))
            if previous.get("evidence_digest") == validation_digest:
                record["known_candidate_failure"] = previous.get("known_candidate_failure", False)
        if plan_root.resolve().is_relative_to(workspace.resolve()):
            raise ValueError("supplemental plans require supervisor state outside the agent workspace")
        if review is None and plan_path.is_file():
            review = json.loads(read_regular_bytes(plan_path, limit=128 * 1024 + 1))
            if not isinstance(review, dict) or review.get("action") != "probe":
                raise ValueError("persisted supplemental plans must require probes")
            # Candidate edits are expected on repair, so preserve the original
            # evidence digest while revalidating the complete review structure.
            original_digest = review.get("evidence_digest")
            if not isinstance(original_digest, str) or len(original_digest) != 64:
                raise ValueError("persisted supplemental plan has no evidence digest")
            _validate_review(review, original_digest)
        if review is None:
            review = _request_review(campaign, workspace, files, digest, state_root=state_root, cancel=cancel)
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
                    evaluation = _run_probes(campaign, workspace, review["suite"], shapes, digest, trial, cancel=cancel)
                except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as exc:
                    # Evaluations also enter the planner's repair request. Keep
                    # raw exception text (paths, commands and input details) in
                    # a separate private record, never in either Agent's context.
                    record.setdefault("probe_errors", []).append({
                        "attempt": attempt, "type": type(exc).__name__, "message": str(exc),
                    })
                    evaluation = {
                        "status": "needs_validation",
                        "diagnosis": "Supplemental probe could not complete. No Kernel failure was established. "
                                     "Check the existing plan against the public input contract; if it remains valid, "
                                     "ask the operator to inspect the private numerical validation record.",
                    }
                record.setdefault("evaluations", []).append(evaluation)
                record["status"] = evaluation["status"]
                if evaluation["status"] != "needs_validation" or attempt == 1:
                    break
                # Input ABI, scheduling and reference-output failures belong to
                # the probe planner. Keep the original risk in the request.
                replacement = _request_review(campaign, workspace, files, digest,
                                              {"review": review, "evaluation": evaluation}, state_root=state_root, cancel=cancel)
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
        if _digest(validation_files) != validation_digest:
            raise ValueError("candidate or contract changed during supplemental validation")
    except NumericalPlanningTimeout as exc:
        try:
            # Only the controller's just-verified sealed evidence permits a skip;
            # Agent-writable legacy evaluation logs are never a trust source.
            standard_passed = standard_correctness_passed
            measured_failure = record.get("known_candidate_failure", False) or any(
                evaluation.get("status") == "needs_repair"
                or any(probe.get("status") == "needs_repair"
                       for probe in evaluation.get("probes", []))
                for evaluation in record.get("evaluations", [])
            )
            unchanged = _digest(validation_files) == validation_digest
            record.update(
                status=("skipped_planner_timeout" if standard_passed and unchanged and not measured_failure
                        else "needs_validation"),
                diagnosis=str(exc), standard_correctness_passed=bool(standard_passed),
            )
        except (OSError, ValueError, TypeError, KeyError) as evidence_error:
            record.update(
                status="needs_validation",
                diagnosis=f"{exc}; timeout skip evidence unavailable: {evidence_error}",
            )
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as exc:
        record.update(status="needs_validation", diagnosis=str(exc))

    cancel.check()
    record["known_candidate_failure"] = record["status"] != "passed" and (
        record.get("known_candidate_failure", False) or any(
            item.get("status") == "needs_repair" or any(probe.get("status") == "needs_repair"
                for probe in item.get("probes", [])) for item in record.get("evaluations", [])))
    durable_write_json(directory / "numerical_result.json", record, indent=2, ensure_ascii=False)
    feedback_path = state_root / "numerical_feedback.json"
    durable_write_json(feedback_path, record, indent=2, ensure_ascii=False)
    status = record["status"]
    print(f"[numerical-supplement] {status}; evidence={feedback_path}", flush=True)
    # Return a purpose-built projection, never planner transcripts, private paths,
    # shape parameters, raw subprocess diagnostics or the complete internal record.
    public = {
        "status": status, "comparison": record["comparison"],
        "cases": [{key: case[key] for key in ("id", "purpose", "fields", "input_constraints") if key in case}
                  for case in (record.get("review", {}).get("suite") or {}).get("cases", [])],
        "evaluations": [{"status": item["status"], "probes": item.get("probes", [])}
                        for item in record.get("evaluations", [])],
    }
    # Planner exceptions can contain private data; diagnostics are retained only
    # in numerical_result.json. Public probes already use the receipt allowlist.
    rendered = json.dumps(public, ensure_ascii=False, allow_nan=False)
    if status in {"passed", "advisory", "skipped_planner_timeout"}:
        feedback = ""
    elif status == "needs_repair":
        feedback = (
            f"{SUPPLEMENTAL_REPAIR_PREFIX}: {rendered}\n"
            "Repair kernel.py, measure the changed candidate, record its Experiment and resubmit episode-report. "
            "The Supervisor reruns the retained probes. Do not edit the harness, probe plan or tolerances."
        )
    else:
        feedback = (
            f"{SUPPLEMENTAL_PENDING_PREFIX}: {rendered}\n"
            "This is not a measured Kernel failure. Preserve the candidate; ask the operator to inspect "
            "the private numerical validation record, or submit an evidence-backed blocked report."
        )
    # Pending infrastructure/evaluator results are retryable even for unchanged code.
    if status != "needs_validation":
        cache[key] = feedback
        campaign._supplemental_results = cache
    return feedback
