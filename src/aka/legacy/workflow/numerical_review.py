"""Turn numerical review suggestions into probes and coding-agent feedback.

A reviewer requests bounded experiments, not an admission verdict. Only measured
failures ask the coding agent to repair the candidate. Planner timeouts may skip
additional testing after standard correctness passes; they are never probe passes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from .resources import PROMPTS_DIR
import shutil
import subprocess
import tempfile
import uuid

from aka.legacy.evaluation.drivers.remote_numerical import MAX_REL_L2, PREFIX, validate_suite, validation_schedule
from aka.legacy.workflow.store import VERIFY_DIR
from aka.legacy.task.problem.native_harness import _fp4_correctness_max_rel_l2
from aka.legacy.workflow.constants import SUPPLEMENTAL_PENDING_PREFIX, SUPPLEMENTAL_REPAIR_PREFIX
from aka.legacy.task.durable import durable_write_json
from aka.legacy.agent.availability import check_review_service
from aka.legacy.execution.transport import check_transport
from aka.legacy.execution.gateway.protocol import classify_gateway_outcome
from aka.legacy.experiment.retry import retry_infrastructure
from aka.legacy.workflow.review_trace import ReviewTrace

import aka.legacy.evaluation.drivers.remote_numerical as remote_numerical
from aka.legacy.evaluation.numerical import _validate_review, _probe_status
DRIVER = Path(remote_numerical.__file__)
from aka.legacy.evaluation.drivers import numerical_inputs as probe_algorithms
PROBE_ALGORITHMS = Path(probe_algorithms.__file__)
PROMPT = PROMPTS_DIR / "numerical_review.md"


class NumericalPlanningTimeout(RuntimeError):
    """The numerical planner timed out without a usable plan."""


class NumericalProbeFailure(RuntimeError):
    """A supplemental probe failed at a supervisor-owned execution stage."""

    def __init__(self, stage, cause):
        super().__init__(f"{stage}: {cause}")
        self.stage = stage
        self.cause = cause


_PROBE_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    subprocess.SubprocessError,
)


def _at_probe_stage(stage, operation, *args, **kwargs):
    try:
        return operation(*args, **kwargs)
    except NumericalProbeFailure:
        raise
    except _PROBE_EXCEPTIONS as exc:
        raise NumericalProbeFailure(stage, exc) from exc


def _public_failure_fact(stage, exc):
    """Return a bounded diagnostic without private inputs, paths or messages."""
    if isinstance(exc, json.JSONDecodeError):
        exception_type, category = "JSONDecodeError", "invalid_json"
        summary, retryable = "validation JSON is invalid", False
    elif isinstance(exc, FileNotFoundError):
        exception_type, category = "FileNotFoundError", "missing_file"
        summary, retryable = "required validation file is missing", False
    elif isinstance(exc, subprocess.TimeoutExpired):
        exception_type, category = "TimeoutExpired", "timeout"
        summary, retryable = "validation subprocess timed out", True
    elif isinstance(exc, subprocess.CalledProcessError):
        exception_type, category = "CalledProcessError", "subprocess_error"
        summary, retryable = "validation subprocess failed", True
    elif isinstance(exc, KeyError):
        exception_type, category = "KeyError", "missing_key"
        summary, retryable = "validation data is missing a required field", False
    elif isinstance(exc, TypeError):
        exception_type, category = "TypeError", "type_error"
        summary, retryable = "validation data has an incompatible type", False
    elif isinstance(exc, ValueError):
        exception_type, category = "ValueError", "value_error"
        summary, retryable = "validation data was rejected", False
    elif isinstance(exc, OSError):
        exception_type, category = "OSError", "io_error"
        summary, retryable = "validation I/O failed", True
    elif isinstance(exc, subprocess.SubprocessError):
        exception_type, category = "SubprocessError", "subprocess_error"
        summary, retryable = "validation subprocess failed", True
    elif isinstance(exc, NumericalPlanningTimeout):
        exception_type, category = "NumericalPlanningTimeout", "timeout"
        summary, retryable = "numerical planner timed out", True
    else:
        exception_type, category = "Exception", "internal_error"
        summary, retryable = "supplemental validation failed", False
    return {
        "stage": stage,
        "exception_type": exception_type,
        "code": f"{stage}.{category}",
        "retryable": retryable,
        "summary": summary,
    }


def _load_shapes(path):
    return json.loads(path.read_text())


def _uses_native_harness(path):
    return "atrex-bench/run_eval" in path.read_text()


def _write_probe_request(request, suite, case_id, digest, private):
    durable_write_json(request, {
        "suite": suite, "case_ids": [case_id], "rotation": digest,
        "mode": "thorough", "per_case_timeout": 540,
        "max_rel_l2": _fp4_correctness_max_rel_l2(private) or MAX_REL_L2,
    })


def _digest(files):
    digest = hashlib.sha256()
    for name, path in sorted(files.items()):
        digest.update(name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def _sources(campaign, workspace, *, harness=None):
    from aka.legacy.acceptance.review import _production_review_candidate_paths

    private = Path(campaign.private_reference_dir or workspace)
    if harness is None:
        raise ValueError('numerical validation requires the frozen Task harness')
    files = {"instructions.md": PROMPT, "driver.py": DRIVER, "numerical_inputs.py": PROBE_ALGORITHMS, "transport.py": harness}
    for source in _production_review_candidate_paths(workspace):
        files["candidate/" + source.relative_to(workspace).as_posix()] = source
    for name in ("input.py", "reference.py", "agent_problem.json", "metadata.json", "README.md"):
        path = private / name if (private / name).is_file() else workspace / name
        if path.is_file():
            files["trusted/" + name] = path
    # Private workload parameters are used only by the supervisor/remote evaluator.
    shapes = private / "shapes.json"
    return files, shapes


def _request_review(campaign, workspace, files, digest, previous=None):
    from aka.legacy.agent.session import run_session

    def review_once():
        timeout = campaign.production_review_timeout
        trace = ReviewTrace(
            workspace,
            "numerical-planner",
            getattr(campaign, "review_trace_mode", "errors"),
        )
        try:
            with tempfile.TemporaryDirectory(prefix="atrex-numerical-advice-") as temporary:
                root = Path(temporary)
                for name, source in files.items():
                    target = root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                request = {"evidence_digest": digest, "previous_validation": previous}
                (root / "review_request.json").write_text(json.dumps(request, indent=2))
                trace.write_private_json("review-request.json", request)
                result = run_session(
                    root, PROMPT.read_text(), timeout=timeout,
                    agent_cli=campaign.agent_cli, reasoning_effort="high", agent_plugins=False,
                    runtime=campaign.agent_runtime,
                    observation_directory=trace.observation_directory,
                )
                trace.record_session(result)
                campaign._account(result, "numerical supplemental-test planning")
                response = root / "numerical_review.json"
                record = {"evidence_digest": digest, "session_id": result.session_id,
                          "timeout_s": timeout, "exit_status": result.exit_status,
                          "timed_out": result.timed_out, "response_written": response.is_file()}
                record_path = (
                    workspace / VERIFY_DIR / f"numerical_planning-{trace.attempt_id}.json"
                )
                durable_write_json(record_path, record, indent=2)
                if _digest({name: root / name for name in files}) != digest:
                    raise ValueError("numerical reviewer modified supplied evidence")
                # A timed-out CLI may already have written the requested plan before
                # hanging on its final response. Validate that artifact rather than
                # discarding it with the temporary session directory.
                if not result.timed_out:
                    check_review_service(result)
                try:
                    value = json.loads(response.read_text())
                    record["response"] = value
                    _validate_review(value, digest)
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
                    retain_diagnostics = bool(
                        result.timed_out
                        or result.exit_status
                        or result.observation_errors
                    )
                    trace.finish(
                        "recovered_after_timeout" if result.timed_out else "validated",
                        failed=retain_diagnostics,
                        response_written=True,
                        recovered_after_timeout=bool(result.timed_out),
                    )
                    if result.timed_out:
                        print("[numerical-supplement] recovered valid plan from timed-out reviewer", flush=True)
                    return value
                finally:
                    durable_write_json(record_path, record, indent=2, ensure_ascii=False)
        except BaseException as exc:
            trace.finish(
                "timed_out" if isinstance(exc, NumericalPlanningTimeout) else "failed",
                failed=True,
                failure_type=type(exc).__name__,
            )
            raise

    context = hashlib.sha256(json.dumps(previous, sort_keys=True).encode()).hexdigest()
    stage = f"numerical-advice:{digest}:{campaign.agent_cli}:{campaign.production_review_timeout}:{context}"
    return retry_infrastructure(workspace, stage, review_once)


def _run_probes(campaign, workspace, suite, shapes_path, digest, directory, *, execute=None, harness=None):

    shapes = _at_probe_stage("load_shapes", _load_shapes, shapes_path)
    schedule = _at_probe_stage(
        "schedule", validation_schedule, suite, shapes, digest, "thorough"
    )
    driver = directory / "test_kernel.py"
    _at_probe_stage("prepare_driver", shutil.copy2, DRIVER, driver)
    _at_probe_stage("prepare_driver", shutil.copy2, PROBE_ALGORITHMS, directory / "numerical_inputs.py")
    native_harness = _at_probe_stage(
        "prepare_harness", _uses_native_harness, workspace / "test_kernel.py",
    )
    if native_harness:
        snapshot = directory / "snapshots" / "evaluator.py"
        _at_probe_stage("prepare_harness", snapshot.parent.mkdir)
        if harness is None:
            raise NumericalProbeFailure(
                "prepare_harness",
                ValueError('numerical validation requires the frozen Task harness'),
            )
        _at_probe_stage("prepare_harness", shutil.copy2, harness, snapshot)
    results = []
    for index, plan in enumerate(schedule):
        if plan.get("status") == "unsupported":
            results.append({"case_id": plan["case_id"], "status": "unsupported",
                            "diagnosis": plan["diagnosis"]})
            continue
        request = directory / f"request-{index:04d}.json"
        _at_probe_stage(
            "prepare_request",
            _write_probe_request,
            request,
            suite,
            plan["case_id"],
            digest,
            Path(campaign.private_reference_dir or workspace),
        )

        transport_attempts = []

        def probe_once(request=request, transport_attempts=transport_attempts):
            from aka.contracts.execution import CommandSpec
            if execute is None:
                execute_command = campaign.services.numerical_command
            else:
                execute_command = execute
            command = CommandSpec(workspace,
                ("python3", str(driver.relative_to(workspace)), str(request.relative_to(workspace))),
                timeout=600, operation="dev")
            process = _at_probe_stage(
                "launch_probe", execute_command, command,
            )

            def parse_receipt():
                receipt_status = "missing"
                batch = {"runs": []}
                for line in process.stdout.splitlines():
                    if line.startswith(PREFIX):
                        try:
                            batch = json.loads(line[len(PREFIX):])
                        except json.JSONDecodeError:
                            receipt_status = "invalid_json"
                            batch = {"runs": "invalid"}
                        else:
                            if isinstance(batch, dict):
                                receipt_status = "valid"
                            else:
                                receipt_status = "invalid_schema"
                                batch = {"runs": "invalid"}
                        break
                return batch, receipt_status

            batch, receipt_status = _at_probe_stage("parse_receipt", parse_receipt)
            transport_attempts.append(_at_probe_stage(
                "record_transport",
                _probe_transport_fact,
                process,
                receipt_status=receipt_status,
            ))
            check_transport(process)
            if process.returncode:
                batch["error"] = f"probe transport exited {process.returncode}"
            return batch

        batch = retry_infrastructure(workspace, f"numerical-probe:{digest}:{plan['case_id']}", probe_once)
        verdict = _at_probe_stage(
            "classify_result", _probe_status, batch, plan, suite, shapes
        )
        status = verdict["status"]
        # Preserve comparison metrics and actual input-generation diagnostics, but
        # never surface private workload kwargs or raw evaluator output to the agent.
        rows = batch.get("runs", [])
        result = (rows[0].get("result") or {}) if rows else {}
        results.append({
            "case_id": plan["case_id"], "status": status,
            "reason": verdict["reason"],
            "metrics": result.get("numerical_metrics", {}),
            "nonfinite_outputs": result.get("nonfinite_outputs", []),
            "expected_probes": verdict["expected_probes"],
            "observed_probes": verdict["observed_probes"],
            "failed_probes": verdict["failed_probes"],
            "transport_attempts": transport_attempts,
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
    reason = next(
        (r.get("reason") for r in results if r["status"] == status and r.get("reason")),
        "unsupported" if status == "advisory" else status,
    )
    counted = [r for r in results if r["status"] != "unsupported"]
    return {
        "status": status,
        "reason": reason,
        "expected_probes": sum(r.get("expected_probes") or 0 for r in counted),
        "observed_probes": sum(r.get("observed_probes") or 0 for r in counted),
        "failed_probes": sum(r.get("failed_probes") or 0 for r in counted),
        "probes": results,
    }


def _probe_transport_fact(process, *, receipt_status):
    """Keep bounded gateway/receipt facts without retaining raw transport output."""
    if receipt_status not in {"valid", "missing", "invalid_json", "invalid_schema"}:
        raise ValueError("unsupported numerical receipt status")
    fact = {
        "return_code": int(process.returncode),
        "receipt_status": receipt_status,
    }
    diagnostic = classify_gateway_outcome(process).failure_record() or {}
    mapping = {
        "failure_class": "failure_class",
        "reason": "failure_reason",
        "http_status": "http_status",
        "job_id": "job_id",
    }
    for source, destination in mapping.items():
        value = diagnostic.get(source)
        if isinstance(value, str) or (source == "http_status" and type(value) is int):
            fact[destination] = value
    return fact


def _public_numerical_supplement(record):
    """Project private numerical feedback into one bounded public diagnostic."""
    evaluations = record.get("evaluations") if isinstance(record, dict) else None
    evaluations = evaluations if isinstance(evaluations, list) else []
    latest = evaluations[-1] if evaluations and isinstance(evaluations[-1], dict) else {}
    probes = latest.get("probes") if isinstance(latest.get("probes"), list) else []

    def count(name):
        value = latest.get(name)
        if type(value) is int and value >= 0:
            return value
        return sum(
            probe.get(name, 0)
            for probe in probes
            if isinstance(probe, dict)
            and type(probe.get(name, 0)) is int
            and probe.get(name, 0) >= 0
        )

    status = record.get("status") if isinstance(record, dict) else None
    status = status if isinstance(status, str) and status else "needs_validation"
    reason = latest.get("reason") or record.get("reason")
    if not isinstance(reason, str) or not reason:
        reason = "passed" if status in {"passed", "advisory", "skipped_planner_timeout"} else "probe_exception"
    transport_attempts = []
    retry_count = 0
    for probe in probes:
        attempts = (
            probe.get("transport_attempts", [])
            if isinstance(probe, dict)
            and isinstance(probe.get("transport_attempts"), list)
            else []
        )
        retry_count += max(0, len(attempts) - 1)
        for attempt in attempts:
            if isinstance(attempt, dict) and len(transport_attempts) < 32:
                transport_attempts.append(dict(attempt))
    failure = latest.get("failure") or record.get("failure")
    if not isinstance(failure, dict):
        failure = None
    return {
        "schema_version": 3,
        "status": status,
        "reason": reason,
        "attempts": len(evaluations),
        "expected_probes": count("expected_probes"),
        "observed_probes": count("observed_probes"),
        "failed_probes": count("failed_probes"),
        "transport_retry_count": retry_count,
        "transport_attempts": transport_attempts,
        "failure": failure,
    }


def _numerical_advice_root(campaign, workspace):
    """Keep mutable probe plans beside, never inside, immutable Problem snapshots."""
    snapshot = getattr(getattr(campaign, "problem", None), "snapshot_directory", None)
    if snapshot is not None:
        root = Path(snapshot).parent / ".atrex_numerical_advice"
    else:
        root = (
            Path.home()
            / ".local"
            / "state"
            / "atrex-kernel-agent"
            / "numerical_advice"
        )
    if root.resolve().is_relative_to(Path(workspace).resolve()):
        raise ValueError("supplemental plans require supervisor state outside the agent workspace")
    return root


def supplemental_feedback(campaign, workspace, *, standard_correctness_passed=False, execute=None, harness=None):
    """Return repair/pending feedback, or an empty string when advice is resolved."""
    if campaign.optimization_mode != "production":
        return ""
    workspace = Path(workspace)
    files, shapes = _sources(campaign, workspace, harness=harness)
    digest = _digest(files)
    validation_digest = _digest({**files, "shapes.json": shapes, "evaluator.py": workspace / "test_kernel.py"})
    cache = getattr(campaign, "_supplemental_results", {})
    key = (str(workspace.resolve()), validation_digest, standard_correctness_passed)
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
    # Persist outside agent worktrees, just like the private reference corpus.
    # Workspace feedback copies are never accepted as supervisor state.
    plan_root = _numerical_advice_root(campaign, workspace)
    plan_path = plan_root / (hashlib.sha256(repr(plan_key).encode()).hexdigest() + ".json")
    record = {"schema_version": 1, "evidence_digest": validation_digest,
              "comparison": {"metric": "relative_l2", "max_rel_l2":
                  _fp4_correctness_max_rel_l2(Path(campaign.private_reference_dir or workspace)) or MAX_REL_L2}}
    try:
        if review is None and plan_path.is_file():
            review = json.loads(plan_path.read_text())
            if not isinstance(review, dict) or review.get("action") != "probe":
                raise ValueError("persisted supplemental plans must require probes")
            # Candidate edits are expected on repair, so preserve the original
            # evidence digest while revalidating the complete review structure.
            original_digest = review.get("evidence_digest")
            if not isinstance(original_digest, str) or len(original_digest) != 64:
                raise ValueError("persisted supplemental plan has no evidence digest")
            _validate_review(review, original_digest)
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
                    evaluation = _run_probes(campaign, workspace, review["suite"], shapes, digest, trial,
                                             execute=execute, harness=harness)
                except NumericalProbeFailure as exc:
                    evaluation = {
                        "status": "needs_validation",
                        "reason": "probe_exception",
                        "expected_probes": 0,
                        "observed_probes": 0,
                        "failed_probes": 0,
                        "diagnosis": str(exc.cause),
                        "failure": _public_failure_fact(exc.stage, exc.cause),
                    }
                except _PROBE_EXCEPTIONS as exc:
                    evaluation = {
                        "status": "needs_validation",
                        "reason": "probe_exception",
                        "expected_probes": 0,
                        "observed_probes": 0,
                        "failed_probes": 0,
                        "diagnosis": str(exc),
                        "failure": _public_failure_fact("supplemental_validation", exc),
                    }
                record.setdefault("evaluations", []).append(evaluation)
                record["status"] = evaluation["status"]
                if evaluation["status"] != "needs_validation" or attempt == 1:
                    break
                # Input ABI, scheduling and reference-output failures belong to
                # the probe planner. Keep the original risk in the request.
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
    except NumericalPlanningTimeout as exc:
        from aka.legacy.evaluation.results import _latest_complete_episode_performance

        try:
            # Episode receipts bind the full standard workload result to the current
            # kernel/manifest. Baselines supply their just-completed standard gate.
            standard_passed = standard_correctness_passed or (
                _latest_complete_episode_performance(
                    workspace, expected_shape_ids=set(json.loads(shapes.read_text()))
                ) is not None
            )
            measured_failure = any(
                evaluation.get("status") == "needs_repair"
                or any(probe.get("status") == "needs_repair"
                       for probe in evaluation.get("probes", []))
                for evaluation in record.get("evaluations", [])
            )
            unchanged = _digest({**files, "shapes.json": shapes,
                                 "evaluator.py": workspace / "test_kernel.py"}) == validation_digest
            record.update(
                status=("skipped_planner_timeout" if standard_passed and unchanged and not measured_failure
                        else "needs_validation"),
                reason="probe_exception",
                diagnosis=str(exc), standard_correctness_passed=bool(standard_passed),
                failure=_public_failure_fact("planner", exc),
            )
        except (OSError, ValueError, TypeError, KeyError) as evidence_error:
            record.update(
                status="needs_validation",
                reason="probe_exception",
                diagnosis=f"{exc}; timeout skip evidence unavailable: {evidence_error}",
                failure=_public_failure_fact("timeout_evidence", evidence_error),
            )
    except _PROBE_EXCEPTIONS as exc:
        record.update(
            status="needs_validation",
            reason="probe_exception",
            diagnosis=str(exc),
            failure=_public_failure_fact("supplemental_validation", exc),
        )

    durable_write_json(directory / "numerical_result.json", record, indent=2, ensure_ascii=False)
    feedback_path = workspace / VERIFY_DIR / "numerical_feedback.json"
    durable_write_json(feedback_path, record, indent=2, ensure_ascii=False)
    public = _public_numerical_supplement(record)
    public_path = workspace / VERIFY_DIR / "numerical-supplement.json"
    durable_write_json(public_path, public, indent=2, ensure_ascii=False)
    status = record["status"]
    print(
        "[numerical-supplement] "
        f"status={public['status']} reason={public['reason']} attempts={public['attempts']} "
        f"expected={public['expected_probes']} observed={public['observed_probes']} "
        f"failed={public['failed_probes']}",
        flush=True,
    )
    if status in {"passed", "advisory", "skipped_planner_timeout"}:
        # A timeout skips only this candidate's expansion; it is not a probe PASS.
        # Retained probe plans still run against subsequent candidate edits.
        feedback = ""
    elif status == "needs_repair":
        feedback = (
            f"{SUPPLEMENTAL_REPAIR_PREFIX} found a measured failure. Read {feedback_path.relative_to(workspace)} "
            "for the requested distributions and results; repair the candidate using the immutable reference, "
            "then rerun the usual evaluator and hand off the updated candidate. Do not edit the probes, "
            "evaluator or tolerances. The supervisor reruns the same requested probes after repair; "
            "passing them closes the suggestion without another numerical-review veto."
        )
    else:
        feedback = (
            f"{SUPPLEMENTAL_PENDING_PREFIX}; this is not a measured correctness failure. "
            f"See {feedback_path.relative_to(workspace)}. Preserve the candidate and report the validation "
            "blocker if it cannot be resolved within the public contract; do not modify the trusted harness."
        )
    # Pending infrastructure/evaluator results are retryable even for unchanged code.
    if status != "needs_validation":
        cache[key] = feedback
        campaign._supplemental_results = cache
    return feedback
