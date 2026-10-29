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
import stat
import tempfile
import uuid

from aka.legacy.workflow.store import VERIFY_DIR
from aka.legacy.workflow.constants import SUPPLEMENTAL_PENDING_PREFIX, SUPPLEMENTAL_REPAIR_PREFIX
from aka.core._filesystem import durable_write_json, durable_write_text
from aka.agent.availability import check_review_service
from aka.experiment.retry import retry_infrastructure
from aka.contracts.execution import ExecutionPending
from aka.contracts.evaluation import EvaluationRequest, EvaluationUnavailable, NumericalProbeRequest, PreparationFile
from aka.contracts.experiment import ExperimentExecutionError, ExperimentRecordError
from aka.contracts.experiment_identity import ExperimentRef
from aka.legacy.workflow.review_trace import ReviewTrace

from aka.evaluation.numerical import validate_numerical_review
PROMPT = PROMPTS_DIR / "numerical_review.md"


class NumericalPlanningTimeout(RuntimeError):
    """The numerical planner timed out without a usable plan."""


class NumericalProbeFailure(RuntimeError):
    """A supplemental probe failed at a supervisor-owned execution stage."""

    def __init__(self, stage, cause, *, continuation_pending=False, experiment_ref=None):
        super().__init__(f"{stage}: {cause}")
        self.stage = stage
        self.cause = cause
        self.continuation_pending = continuation_pending
        self.experiment_ref = experiment_ref


_PROBE_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    subprocess.SubprocessError,
    ExecutionPending,
    ExperimentExecutionError,
    ExperimentRecordError,
    EvaluationUnavailable,
    LookupError,
)


def _private_failure_chain(exc):
    facts, seen = [], set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        facts.append({"exception_type": type(exc).__name__, "message": str(exc)})
        exc = exc.__cause__
    return facts


def _public_failure_fact(stage, exc):
    """Return a bounded diagnostic without private inputs, paths or messages."""
    cause = exc
    seen = set()
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        if isinstance(cause, ExecutionPending):
            return {"stage": stage, "exception_type": "ExecutionPending", "code": f"{stage}.execution_pending",
                    "retryable": True, "summary": "original numerical execution requires continuation"}
        cause = cause.__cause__
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
    driver = campaign.plugin_registry.resource_result().resource("evaluation-numerical-driver").source
    files = {"instructions.md": PROMPT, "driver.py": driver, "transport.py": harness}
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
    from aka.legacy.workflow.agent_session import run_session

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
                access_options = {}
                if campaign.agent.capabilities.filesystem_isolation:
                    from .agent_access import for_workspace
                    access_options["access"] = for_workspace(campaign, root, resources=False,
                        read_only=tuple(root / name for name in (*files, "review_request.json")))
                result = run_session(
                    root, PROMPT.read_text(), timeout=timeout,
                    agent_cli=campaign.agent_cli, reasoning_effort="high", agent_plugins=False,
                    runtime=campaign.agent_runtime,
                    observation_directory=trace.observation_directory,
                    **access_options,
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
                    validate_numerical_review(value, digest)
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


def _candidate_snapshot(files):
    candidates = tuple(
        PreparationFile(name.removeprefix("candidate/"), path.read_bytes(),
                        stat.S_IMODE(path.stat().st_mode) & 0o777)
        for name, path in sorted(files.items()) if name.startswith("candidate/")
    )
    rows = [{"path": item.path, "mode": item.mode,
             "sha256": hashlib.sha256(item.content).hexdigest()} for item in candidates]
    identity = hashlib.sha256(json.dumps(
        rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()
    return candidates, identity


def _pending_path(plan_root, workspace):
    # A plan/candidate change must not hide a possibly admitted original Job.
    identity = hashlib.sha256(str(workspace.resolve()).encode("utf-8")).hexdigest()
    return plan_root / ("pending-" + identity + ".json")


def _load_pending(path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("schema") != "aka.workflow.numerical-continuation.v1":
        raise ValueError("invalid numerical continuation record")
    for name in ("request_id", "candidate_identity", "suite_json", "rotation"):
        if not isinstance(value.get(name), str) or not value[name]:
            raise ValueError("incomplete numerical continuation identity")
    ref = value.get("experiment_ref")
    if ref is not None and (not isinstance(ref, dict) or set(ref) != {"run_id", "experiment_id"}
                            or any(not isinstance(item, str) or not item for item in ref.values())):
        raise ValueError("invalid numerical continuation reference")
    return value


def _continuation_record(experiment, pending):
    ref = pending.get("experiment_ref")
    record = (experiment.get(ExperimentRef(**ref)) if ref is not None
              else experiment.find(pending["request_id"]))
    if record.request_id != pending["request_id"]:
        raise ValueError("numerical continuation references a different experiment request")
    return record


def _numerical_artifact(experiment, record):
    if record.lifecycle != "terminal" or record.phase != "complete":
        raise ValueError("original numerical experiment is not fully collected")
    if record.evaluation is None or not record.evaluation.artifact:
        raise ValueError("numerical experiment has no interpreted artifact")
    artifacts = [item for item in record.artifacts
                 if item.artifact_id == record.evaluation.artifact]
    if len(artifacts) != 1:
        raise ValueError("numerical interpreted artifact identity is unavailable")
    content = experiment.read_artifact(artifacts[0])
    value = json.loads(content)
    if (not isinstance(value, dict)
            or set(value) != {"schema", "summary", "probes", "measurements"}
            or value["schema"] != "aka.evaluation.numerical-probe.v1"):
        raise ValueError("invalid numerical interpreted artifact schema")
    summary = value["summary"]
    if (not isinstance(summary, dict)
            or set(summary) != {"status", "reason", "expected_probes", "observed_probes", "failed_probes"}
            or summary["status"] not in {"passed", "advisory", "needs_repair", "needs_validation"}
            or not isinstance(summary["reason"], str) or not summary["reason"]
            or any(summary[name] is not None and (type(summary[name]) is not int or summary[name] < 0)
                   for name in ("expected_probes", "observed_probes", "failed_probes"))
            or not isinstance(value["probes"], list) or not isinstance(value["measurements"], list)):
        raise ValueError("invalid numerical interpreted summary")
    return content, value


def _not_dispatched(record):
    return record.lifecycle == "terminal" and record.phase != "complete" and all(
        item.disposition == "not_submitted" and (item.result is None or not item.result.job_references)
        for item in record.attempts
    )


def _guard_changed_continuation(experiment, pending_path, *, request_id=None):
    if not pending_path.is_file():
        return None
    pending = None
    try:
        pending = _load_pending(pending_path)
        if request_id == pending["request_id"]:
            return pending
        record = _continuation_record(experiment, pending)
        pending["experiment_ref"] = record.ref.as_dict()
        durable_write_json(pending_path, pending, ensure_ascii=False)
        if not _not_dispatched(record):
            _numerical_artifact(experiment, record)
        pending_path.unlink()
        return None
    except _PROBE_EXCEPTIONS as exc:
        ref = pending.get("experiment_ref") if pending else None
        raise NumericalProbeFailure("resume_original", exc, continuation_pending=True,
                                    experiment_ref=ref) from exc


def _run_probes(campaign, workspace, suite, digest, directory, *, experiment, files, pending_path):
    try:
        candidates, candidate_identity = _candidate_snapshot(files)
        if _digest(files) != digest:
            raise ValueError("review evidence changed before validation")
    except _PROBE_EXCEPTIONS as exc:
        raise NumericalProbeFailure("prepare_candidate", exc,
                                    continuation_pending=pending_path.is_file()) from exc
    suite_json = json.dumps(suite, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    binding = {"rotation": digest, "candidate_identity": candidate_identity,
               "suite_sha256": hashlib.sha256(suite_json.encode("utf-8")).hexdigest()}
    request_id = "numerical-probe:" + hashlib.sha256(json.dumps(
        binding, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()
    request = EvaluationRequest(workspace=workspace, mode="numerical",
        numerical_probe=NumericalProbeRequest(suite_json, digest, candidate_identity, candidates))
    pending = _guard_changed_continuation(experiment, pending_path, request_id=request_id)
    try:
        if pending is None:
            try:
                registered = experiment.find(request_id)
            except LookupError:
                registered = None
            pending = {"schema": "aka.workflow.numerical-continuation.v1", "request_id": request_id,
                       "candidate_identity": candidate_identity, "suite_json": suite_json,
                       "rotation": digest, "experiment_ref": registered.ref.as_dict() if registered else None}
            # Persist intent before dispatch: a crash must never silently create a new Job.
            durable_write_json(pending_path, pending, ensure_ascii=False)
            if registered is None:
                record = experiment.run_recorded(request, request_id=request_id)
            else:
                record = experiment.resume_recorded(request, request_id=request_id,
                    recovery_timeout=600, max_remeasurements=0)
        else:
            # Lookup failure after a retained intent is uncertainty, never fresh work.
            _continuation_record(experiment, pending)
            record = experiment.resume_recorded(request, request_id=request_id,
                recovery_timeout=600, max_remeasurements=0)
        pending["experiment_ref"] = record.ref.as_dict()
        durable_write_json(pending_path, pending, ensure_ascii=False)
        content, interpreted = _numerical_artifact(experiment, record)
        durable_write_text(directory / "numerical_result.json", content.decode("utf-8"))
        result = {**interpreted["summary"], "probes": interpreted["probes"],
                  "interpreted_result": interpreted, "experiment_ref": record.ref.as_dict(),
                  "request_id": request_id, "execution_attempts": [item.as_dict() for item in record.attempts]}
        pending_path.unlink()
        return result
    except _PROBE_EXCEPTIONS as exc:
        if pending is None:
            raise NumericalProbeFailure("find_original", exc, continuation_pending=True) from exc
        ref = getattr(exc, "experiment_ref", None)
        if ref is not None:
            pending["experiment_ref"] = ref.as_dict()
            durable_write_json(pending_path, pending, ensure_ascii=False)
        continuation_pending = True
        try:
            failed_record = _continuation_record(experiment, pending)
            if _not_dispatched(failed_record):
                pending["experiment_ref"] = failed_record.ref.as_dict()
                pending_path.unlink()
                continuation_pending = False
        except _PROBE_EXCEPTIONS:
            pass
        raise NumericalProbeFailure("recorded_probe", exc, continuation_pending=continuation_pending,
                                    experiment_ref=pending.get("experiment_ref")) from exc


def _public_numerical_supplement(record):
    """Project private numerical feedback into one bounded public diagnostic."""
    evaluations = record.get("evaluations") if isinstance(record, dict) else None
    evaluations = evaluations if isinstance(evaluations, list) else []
    latest = evaluations[-1] if evaluations and isinstance(evaluations[-1], dict) else {}
    probes = latest.get("probes") if isinstance(latest.get("probes"), list) else []

    def count(name):
        if name in latest and latest[name] is None:
            # The existing bounded diagnostic uses integer counts. Full evidence
            # retains Evaluation's unknown value without inventing completeness.
            return 0
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


def supplemental_feedback(campaign, workspace, *, standard_correctness_passed=False, experiment=None, harness=None):
    """Return repair/pending feedback, or an empty string when advice is resolved."""
    if campaign.optimization_mode != "production":
        return ""
    workspace = Path(workspace)
    files, shapes = _sources(campaign, workspace, harness=harness)
    digest = _digest(files)
    validation_digest = _digest({**files, "shapes.json": shapes, "evaluator.py": workspace / "test_kernel.py"})
    experiment = experiment if experiment is not None else campaign.services.baseline_experiment
    plan_root = _numerical_advice_root(campaign, workspace)
    pending_path = _pending_path(plan_root, workspace)
    cache = getattr(campaign, "_supplemental_results", {})
    key = (str(workspace.resolve()), validation_digest, standard_correctness_passed)
    if key in cache and not pending_path.is_file():
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
    record = {"schema_version": 1, "evidence_digest": validation_digest}
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
            validate_numerical_review(review, original_digest)
        if review is None:
            review = _request_review(campaign, workspace, files, digest)
        record["review"] = review
        if review["action"] == "complete":
            _guard_changed_continuation(experiment, pending_path)
            record["status"] = "passed"
        else:
            plans[plan_key] = review
            campaign._supplemental_plans = plans
            durable_write_json(plan_path, review, indent=2)
            for attempt in range(2):
                trial = directory / f"probe-{attempt}"
                trial.mkdir()
                try:
                    evaluation = _run_probes(campaign, workspace, review["suite"], digest, trial,
                                             experiment=experiment, files=files, pending_path=pending_path)
                except NumericalProbeFailure as exc:
                    evaluation = {
                        "status": "needs_validation",
                        "reason": "probe_exception",
                        "expected_probes": 0,
                        "observed_probes": 0,
                        "failed_probes": 0,
                        "diagnosis": str(exc.cause),
                        "cause_chain": _private_failure_chain(exc.cause),
                        "failure": _public_failure_fact(exc.stage, exc.cause),
                        "continuation_pending": exc.continuation_pending,
                        "experiment_ref": exc.experiment_ref,
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
                if (evaluation["status"] != "needs_validation" or attempt == 1
                        or evaluation.get("continuation_pending")):
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
        from aka.evaluation.records import latest_complete_episode_performance

        try:
            # Episode receipts bind the full standard workload result to the current
            # kernel/manifest. Baselines supply their just-completed standard gate.
            standard_passed = standard_correctness_passed or (
                latest_complete_episode_performance(
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
                status=("skipped_planner_timeout" if standard_passed and unchanged and not measured_failure and not pending_path.is_file()
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
    except NumericalProbeFailure as exc:
        record.update(status="needs_validation", reason="probe_exception",
            diagnosis=str(exc.cause), cause_chain=_private_failure_chain(exc.cause),
            failure=_public_failure_fact(exc.stage, exc.cause), continuation_pending=exc.continuation_pending, experiment_ref=exc.experiment_ref)
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
