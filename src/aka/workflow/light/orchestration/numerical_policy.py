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
import traceback
from concurrent.futures import CancelledError

from aka.evaluation.drivers.native_harness import _fp4_correctness_max_rel_l2
from aka.evaluation.resources import native_harness_script
from aka.evaluation.drivers import numerical_inputs as probe_algorithms
from aka.workflow.light.host.workspace import read_input
from aka.resources.publication import publish
from aka.workflow.light.host.projection import bounded_text, credential_values, redact_diagnostics
from aka.core.filesystem import read_regular_bytes

from aka.evaluation.drivers.remote_numerical import MAX_REL_L2, validate_suite
from aka.contracts.evaluation import EvaluationRequest, NumericalProbeRequest, PreparationFile
from aka.workflow.light.orchestration.constants import SUPPLEMENTAL_PENDING_PREFIX, SUPPLEMENTAL_REPAIR_PREFIX
from aka.core.filesystem import durable_write_json
from aka.workflow.light.orchestration.infrastructure_retry import (check_review_service, retry_infrastructure)

from importlib.resources import files
from aka.resources.bundles import resource_path

ROOT = Path(str(files("aka.workflow.light"))).resolve()
from aka.evaluation.drivers import remote_numerical
DRIVER = Path(remote_numerical.__file__).resolve(strict=True)
# Trusted package assets may be imported through a virtualenv's lib64 -> lib
# alias. Canonicalize them here, not in _digest/read_input: candidate and
# reviewer-owned evidence must still reject symlinks in every path component.
PROBE_ALGORITHMS = Path(probe_algorithms.__file__).resolve(strict=True)
PROMPT = resource_path("prompts/research-episodes/numerical_review.md")
HARNESS = native_harness_script().resolve(strict=True)


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


def validate_candidate(campaign, workspace, *, source=None, parent=None, flow="episode"):
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
            owner = campaign.candidate_workspace
            owner.install_source(frozen, source if source is not None else owner.read_source(workspace))
            private = Path(campaign.private_reference_dir) if campaign.private_reference_dir else None
            names = ("input.py", "reference.py", "shapes.json", "metadata.json", "agent_problem.json",
                     "solution.json", "config.json", "definition.json", "workload.jsonl")
            revision = owner.current_revision(workspace)
            committed = owner.read_revision(workspace, revision, list(names)).files if revision else {}
            for name in names:
                if private is not None and name != "solution.json" and (private / name).is_file():
                    data = read_input(private, name)
                else:
                    # Do not trust mutable native-workspace evaluator inputs.
                    data = committed.get(name)
                    if data is None:
                        continue
                publish(frozen, name, data)
            if not all((frozen / name).is_file() for name in ("input.py", "shapes.json", "reference.py")):
                # The declarative probes require the native input ABI. SOL's
                # standard full-workload gate remains mandatory, not replaced.
                if (frozen / "workload.jsonl").is_file():
                    durable_write_json(state_root / "numerical_feedback.json", {
                        "status": "advisory", "reason": "supplemental input ABI unsupported for SOL",
                        "kernel_sha256": owner.source_identity(owner.read_source(frozen)),
                    })
                    return ""
                return f"{SUPPLEMENTAL_PENDING_PREFIX}: trusted input.py/reference.py/shapes.json unavailable"
            publish(frozen, "test_kernel.py", HARNESS.read_bytes())
            frozen_files = {path.relative_to(frozen).as_posix(): path for path in frozen.rglob("*") if path.is_file()}
            content_identity = _digest(frozen_files)
            retained = state_root / "candidates" / content_identity
            if not retained.exists():
                retained.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(frozen, retained)
            if _digest({name: retained / name for name in frozen_files}) != content_identity:
                raise ValueError("retained numerical candidate snapshot changed")
            return supplemental_feedback(campaign, retained, state_root=state_root,
                                         standard_correctness_passed=True, cancel=cancel, flow=flow)


def _digest(files):
    digest = hashlib.sha256()
    for name, path in sorted(files.items()):
        digest.update(name.encode() + b"\0" + read_input(path.parent, path.name) + b"\0")
    return digest.hexdigest()


def _sources(campaign, workspace):
    from aka.workflow.light.orchestration.session_io import _production_review_candidate_paths

    files = {"instructions.md": PROMPT, "driver.py": DRIVER, "numerical_inputs.py": PROBE_ALGORITHMS, "transport.py": HARNESS}
    for source in _production_review_candidate_paths(workspace, workspace_provider=campaign.candidate_workspace):
        files["candidate/" + source.relative_to(workspace).as_posix()] = source
    # Metadata/configuration are part of the pinned public input contract and
    # must be visible as evidence if the reviewer reasons about their values.
    for name in ("input.py", "reference.py", "agent_problem.json", "metadata.json", "config.json", "definition.json"):
        path = workspace / name
        if path.is_file():
            files["trusted/" + name] = path
    # Private workload parameters are used only by the supervisor/remote evaluator.
    shapes = workspace / "shapes.json"
    return files, shapes


def _world_size(workspace):
    from aka.evaluation.metadata import distributed_evaluation_world_size
    path = workspace / "metadata.json"
    return distributed_evaluation_world_size(json.loads(read_input(workspace, path.name)) if path.is_file() else {})


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
    def review_once():
        cancel.check()
        timeout = campaign.production_review_timeout
        review_id = uuid.uuid4().hex
        observation_directory = state_root / "sessions" / review_id
        with tempfile.TemporaryDirectory(prefix="atrex-numerical-advice-") as temporary:
            root = Path(temporary).resolve()
            for name, source in files.items():
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            request = {"evidence_digest": digest, "previous_validation": previous,
                       "world_size": _world_size(workspace)}
            (root / "review_request.json").write_text(json.dumps(request, indent=2))
            result = campaign._run_session(
                root, PROMPT.read_text(), timeout=cancel.timeout(timeout),
                agent_cli=campaign.agent_cli, reasoning_effort="high", agent_plugins=False,
                extra_environment=campaign.agent_boundary_environment("numerical-review"),
                observation_directory=observation_directory,
            )
            campaign._account(result, "numerical supplemental-test planning")
            cancel.check()
            response = root / "numerical_review.json"
            record = {"evidence_digest": digest, "session_id": result.session_id,
                      "timeout_s": timeout, "exit_status": result.exit_status,
                      "timed_out": result.timed_out, "response_written": response.is_file(),
                      "observations": observation_directory.relative_to(state_root).as_posix()}
            record_path = state_root / f"numerical_planning-{review_id}.json"
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


def _run_probes(campaign, workspace, suite, shapes_path, digest, directory, *, cancel):
    if suite["world_size"] != _world_size(workspace):
        raise ValueError("retained probe world_size differs from the trusted contract")
    cancel.check()
    source = campaign.candidate_workspace.read_source(workspace)
    candidates = tuple(PreparationFile(name, content) for name, content in sorted(source.files.items())
                       if content is not None)
    rows = [{"path": item.path, "mode": item.mode, "sha256": hashlib.sha256(item.content).hexdigest()}
            for item in candidates]
    identity = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    probe = NumericalProbeRequest(json.dumps(suite, sort_keys=True, separators=(",", ":")),
                                  digest, identity, candidates)
    request = EvaluationRequest(workspace, mode="numerical", numerical_probe=probe, cancel_event=cancel)
    request_id = "numerical:" + hashlib.sha256(json.dumps({
        "candidate": identity, "rotation": digest, "suite": probe.suite_json, "timeout": probe.timeout,
    }, sort_keys=True).encode()).hexdigest()
    result = campaign._supervisor_runtime.measurements.numerical(request, request_id=request_id)
    durable_write_json(directory / "evaluation.json", result)
    return result


def supplemental_feedback(campaign, workspace, *, state_root, standard_correctness_passed=False,
                          cancel, flow="episode"):
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
    research_records = campaign.research_records
    key = (str(state_root), validation_digest, standard_correctness_passed, flow, research_records)
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
                    # Preserve the original exception privately and give both
                    # Agents the same error, with only secrets/host paths scrubbed.
                    record.setdefault("probe_errors", []).append({
                        "attempt": attempt, "type": type(exc).__name__, "message": str(exc),
                        "traceback": traceback.format_exc(),
                    })
                    evaluation = {
                        "status": "needs_validation",
                        "diagnosis": bounded_text(redact_diagnostics(
                            f"{type(exc).__name__}: {exc}",
                            private_paths=(str(state_root), str(workspace), str(ROOT)),
                            private_values=credential_values(os.environ),
                        ), limit=8192),
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
    # Return the cases, observations and bounded Gateway diagnostics, not planner
    # transcripts, private request snapshots or the complete internal record.
    public = {
        "status": status, "comparison": record["comparison"],
        "cases": [{key: case[key] for key in ("id", "purpose", "fields", "input_constraints") if key in case}
                  for case in (record.get("review", {}).get("suite") or {}).get("cases", [])],
        "evaluations": [{"status": item["status"], "probes": item.get("probes", []),
                         **({"diagnosis": item["diagnosis"]} if item.get("diagnosis") else {})}
                        for item in record.get("evaluations", [])],
    }
    # Probe errors already passed through the Gateway or the redaction above.
    # Other planner exceptions remain in the private numerical_result.json.
    rendered = json.dumps(public, ensure_ascii=False, allow_nan=False)
    if status in {"passed", "advisory", "skipped_planner_timeout"}:
        feedback = ""
    elif status == "needs_repair":
        if flow == "framework_baseline":
            next_step = (
                "Repair the candidate sources, rerun the bounded V1 smoke command, then stop. "
                "Do not submit episode-report. "
            )
        elif research_records:
            next_step = "Repair the candidate within its editable source scope, measure it, record its Experiment and resubmit episode-report. "
        else:
            next_step = (
                "Repair the candidate within its editable source scope, measure it, then resubmit episode-report "
                "with the new measurement's gateway_record_ids. "
            )
        feedback = (
            f"{SUPPLEMENTAL_REPAIR_PREFIX}: {rendered}\n"
            + next_step +
            "The Supervisor reruns the retained probes. Do not edit the harness, probe plan or tolerances."
        )
    else:
        pending_step = (
            "Preserve the V1 candidate and ask the operator to inspect the private numerical validation record."
            if flow == "framework_baseline" else
            "Preserve the candidate; ask the operator to inspect the private numerical validation record, "
            "or submit an evidence-backed blocked report."
        )
        feedback = (
            f"{SUPPLEMENTAL_PENDING_PREFIX}: {rendered}\n"
            "This is not a measured Kernel failure. " + pending_step
        )
    # Pending infrastructure/evaluator results are retryable even for unchanged code.
    if status != "needs_validation":
        cache[key] = feedback
        campaign._supplemental_results = cache
    return feedback
