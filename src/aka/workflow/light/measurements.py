"""Project shared Experiment facts for Light's tools and research Journal.

This Workflow view owns no ledger, source store, Job recovery or evaluator codec.
It borrows the selected modules for every operation in one Campaign namespace.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import re
import uuid

from aka.contracts.evaluation import EvaluationRequest
from aka.contracts.content import WorkspaceFile
from aka.contracts.execution import CommandSpec
from aka.contracts.experiment import ExperimentRecordError
from aka.contracts.workspace import RevisionContent

RECORD_ID = re.compile(r"gateway-[0-9a-f]{64}")
KERNEL_ID = re.compile(r"kernel-[0-9a-f]{64}")
QUERY_KINDS = frozenset({"record-read", "kernel-read", "kernel-records", "kernel-pareto-frontier"})
_PREFIXES = {"run": "[test_kernel] RESULT_JSON=", "same_allocation_abba": "[sandbox] ABBA_JSON=",
             "profile": "[sandbox] PROFILE_JSON=", "check": "[sandbox] CHECK_JSON=",
             "disassemble": "[sandbox] DISASSEMBLE_JSON="}


class RecordedMeasurements:
    """Read recorded owner DTOs without opening an execution session."""
    def __init__(self, *, workspace_provider, evaluation_input=None):
        self.workspace_provider = workspace_provider
        self.evaluation_input = evaluation_input

    def project(self, records, record):
        return self._evidence(records, record)

    def candidate_source(self, records, record):
        return self._source(records, record)

    @staticmethod
    def _identities(record):
        return tuple(step.measurement_identity for step in record.steps if step.measurement_identity is not None)

    @staticmethod
    def _measurements(record):
        latest = {attempt.task_id: attempt for attempt in record.attempts}
        measurements = []
        for step in record.steps:
            if step.measurement_identity is None:
                continue
            attempt = latest.get(step.task_id)
            if (attempt is None or attempt.disposition != "terminal" or attempt.result is None
                    or attempt.result.facts.disposition != "terminal"
                    or attempt.result.transport_status not in {"succeeded", "failed"}
                    or attempt.assessment is None
                    or attempt.assessment.identity.digest != step.measurement_identity.digest):
                return ()
            measurements.append(attempt.assessment)
        return tuple(measurements)

    def kernel(self, source):
        if not isinstance(source, RevisionContent):
            raise TypeError("Kernel identity requires the selected Workspace RevisionContent")
        identity = self.workspace_provider.source_identity(source)
        return {"kernel_id": "kernel-" + identity, "kernel_artifact_digest": "sha256:" + identity,
                "sha256": identity}

    def _source(self, experiment, record):
        artifacts = [artifact for artifact in record.artifacts if artifact.role == "candidate-source"]
        if not artifacts:
            return None
        sources = [experiment.read_source(artifact) for artifact in artifacts]
        identities = {self.workspace_provider.source_identity(source) for source in sources}
        if len(identities) != 1:
            raise ExperimentRecordError("source_binding_mismatch", record.ref)
        artifact_digests = {artifact.sha256 for artifact in artifacts}
        for identity in self._identities(record):
            expected = identity.parameters.get("candidate_source_sha256")
            artifact_digest = identity.parameters.get("candidate_source_artifact_sha256")
            source_identity = identity.parameters.get("candidate_source_identity")
            if (expected not in artifact_digests or artifact_digest not in artifact_digests
                    or source_identity is not None and source_identity not in identities):
                raise ExperimentRecordError("source_binding_mismatch", record.ref)
        return sources[0]

    def _evidence(self, experiment, record):
        identities = self._identities(record)
        measurements = self._measurements(record)
        source = self._source(experiment, record)
        selected = record.evaluation
        known = bool(identities) and len(measurements) == len(identities) and all(
            (measurement.validity == "invalid" or measurement.validity == "valid" and measurement.completeness == "complete")
            and measurement.comparability == "comparable" for measurement in measurements) and len({
                measurement.environment_identity for measurement in measurements}) == 1
        valid = known and all(measurement.validity == "valid" for measurement in measurements)
        trusted_legacy = False
        modes = {identity.parameters.get("mode") for identity in identities}
        mode = next(iter(modes)) if len(modes) == 1 else None
        operation = (record.steps[0].operation if record.steps else str(record.request.get("operation", "dev")))
        if operation == "auto":
            operation = "run" if identities else "dev"
        if identities and identities[0].method == "atrex-bench/abba":
            operation = "same_allocation_abba"
        elif identities and identities[0].method in {"atrex-bench/eval", "legacy-sol/eval", "legacy-eval/command"}:
            operation = "run"
        elif identities and identities[0].method in {"atrex-bench/profile", "legacy-profile/1"}:
            operation = "profile"
        elif identities and identities[0].method.startswith("source-tree/"):
            operation = identities[0].method.split("/", 1)[1]
        settled = record.lifecycle == "terminal" and record.phase == "complete"
        if identities and any(identity.method == "legacy-sol/eval" for identity in identities):
            from aka.evaluation.sol_evidence import matches_task
            # Old projections alone cannot establish which evaluator actually ran.
            trusted = all(matches_task(identity, self.evaluation_input) for identity in identities)
            trusted_legacy = trusted
            known = (trusted and settled and selected is not None and selected.gate in {"PASS", "FAIL"}
                and not selected.error and len(measurements) == len(identities)
                and all(measurement.validity == "invalid" or
                        measurement.validity == "valid" and measurement.completeness == "complete"
                        for measurement in measurements)
                and (selected.gate != "PASS" or all(measurement.validity == "valid" for measurement in measurements)))
            valid = known and selected.gate == "PASS"
        if any(measurement.identity.digest != identity.digest for measurement, identity in zip(measurements, identities)):
            known = valid = False
        if source is None:
            known = valid = False
        if selected is not None and selected.gate not in {"PASS", "FAIL"}:
            known = valid = False
        passed = selected.gate == "PASS" and valid if selected is not None and known and settled else None
        feedback = {}
        if record.kind == "command" and record.phase == "complete" and operation != "same_allocation_abba":
            from aka.evaluation.tool_results import canonical_result, with_execution_error
            receipt = experiment.command_result(record.ref)
            operation_name = "evaluate" if operation == "run" else operation
            public = canonical_result(with_execution_error({"exit_code": receipt.returncode,
                "stdout": receipt.stdout, "stderr": receipt.stderr}, receipt), {"operation": operation_name})["result"]
            feedback = {key: public[key] for key in ("failures", "actionable_diagnostics", "diagnostics",
                "feedback_truncation", "diagnostic_truncation", "error", "next_action", "outcome") if key in public}
        result = {**feedback, "all_pass": passed, "correct": passed, "mode": mode,
                  "error": feedback.get("error") or (selected.error if selected is not None else None),
                  "latency_us_by_shape": dict(selected.candidate_latency_us_by_shape) if selected is not None else {},
                  "latency_us_geomean": selected.candidate_latency_us if selected is not None else None,
                  "performance_score": selected.candidate_performance_score if selected is not None else None}
        if selected is not None and selected.failure is not None:
            result["failure"] = asdict(selected.failure)
        native_incomplete = False
        if operation == "same_allocation_abba" and selected is not None:
            from aka.evaluation.native_abba import recorded_comparison
            if selected.gate in {"PASS", "FAIL"}:
                try:
                    result.update(recorded_comparison(selected, identities))
                except ValueError as error:
                    known = False
                    native_incomplete = True
                    result["error"] = result.get("error") or str(error)
            else:
                known = False
                native_incomplete = True
            result.update(complete=bool(known and settled), correct=passed if known else None,
                          all_pass=passed if known else None)
        # Explain the final applicable verdict, after legacy policy and source /
        # native-comparison checks. A decisive invalid assessment need not be complete.
        if not known or operation in {"run", "same_allocation_abba"} and result["correct"] is None:
            qualification = []
            if not identities:
                qualification.append("measurement-identity-missing")
            if len(measurements) != len(identities):
                latest = {attempt.task_id: attempt for attempt in record.attempts}
                mismatched = any(step.measurement_identity is not None
                    and (attempt := latest.get(step.task_id)) is not None
                    and attempt.assessment is not None
                    and attempt.assessment.identity.digest != step.measurement_identity.digest
                    for step in record.steps)
                qualification.append("measurement-identity-mismatch" if mismatched else "measurement-evidence-unsettled")
            for measurement in measurements:
                if measurement.validity != "invalid" and measurement.completeness != "complete":
                    qualification.append("measurement-incomplete")
                if measurement.validity == "unknown":
                    qualification.append("measurement-validity-unknown")
                if not trusted_legacy and measurement.comparability != "comparable":
                    if not measurement.environment_identity:
                        qualification.append("environment-identity-missing")
                    if measurement.identity.dependency_policy == "unknown":
                        qualification.append("dependency-policy-unknown")
                    if measurement.environment_identity and measurement.identity.dependency_policy != "unknown":
                        qualification.append("measurement-not-comparable")
            if not trusted_legacy and len({item.environment_identity for item in measurements}) > 1:
                qualification.append("environment-identity-mismatch")
            if any(identity.method == "legacy-sol/eval" for identity in identities) and not trusted_legacy:
                qualification.append("legacy-evaluator-untrusted")
            if source is None:
                qualification.append("source-binding-missing")
            if not settled or selected is None:
                qualification.append("evaluation-unsettled")
            elif native_incomplete:
                qualification.append("native-abba-incomplete")
            elif selected.gate == "ERROR":
                qualification.append("evaluation-error")
            reasons = list(dict.fromkeys(qualification)) or ["measurement-not-comparable"]
            result["qualification"] = {"status": "unknown", "reasons": reasons}
            diagnostics = result.get("diagnostics", ())
            generated = [{"category": "qualification", "code": code} for code in reasons]
            result["diagnostics"] = [*diagnostics,
                *(item for item in generated if item not in diagnostics)]
        binding = self.kernel(source) if source is not None else {}
        attempts = [attempt for attempt in record.attempts if attempt.result is not None]
        exit_code = attempts[-1].result.returncode if attempts else None
        completed = settled and (known if operation in {"run", "same_allocation_abba"} else bool(attempts) and all(
            attempt.result.transport_status in {"succeeded", "failed"} for attempt in attempts))
        failed = settled and bool(attempts) and attempts[-1].result.returncode != 0 and (
            attempts[-1].result.facts.disposition in {"terminal", "not_submitted"})
        full_task = bool(identities) and all(identity.method in {"atrex-bench/eval", "legacy-sol/eval"}
            and identity.parameters.get("workload_scope") == "full-task" and identity.parameters.get("mode") == "full"
            for identity in identities)
        return {"gateway_record_id": "gateway-" + record.ref.experiment_id, "experiment_ref": record.ref.as_dict(),
                "gateway_kind": operation, **binding, "source_sha256": binding.get("sha256"), "result": result,
                "execution_status": "completed" if completed else "failed" if failed else "unknown", "exit_code": exit_code,
                "standard_full_evaluation": full_task and known,
                "typed_six_case_correctness": known and bool(identities) and all(
                    identity.method == "atrex-bench/eval" and identity.parameters.get("correctness_reuse_policy") == "typed-six-case/1"
                    for identity in identities), "created_at": record.created_at}


class SharedMeasurements(RecordedMeasurements):
    def __init__(self, *, factory, executor, evaluator, namespace_workspace, evaluation_input,
                 workspace_provider, command_evaluator, environment_identity=None, query_request=None):
        self.factory = factory
        self.executor = executor
        self.evaluator = evaluator
        self.namespace_workspace = Path(namespace_workspace)
        self.evaluation_input = evaluation_input
        self.workspace_provider = workspace_provider
        self.command_evaluator = command_evaluator
        # The host supplies this only after a selected Executor confirms it.
        # A recorded old environment never becomes current availability evidence.
        self.environment_identity = environment_identity
        self.query_request = query_request

    @contextmanager
    def _open(self):
        from aka.workflow.budgets import bound_executor
        with self.factory.open(workspace=self.namespace_workspace, purpose="development", executor=bound_executor(self.executor),
                               evaluator=self.evaluator, evaluation_input=self.evaluation_input,
                               command_evaluator=self.command_evaluator) as experiment:
            yield experiment

    def numerical(self, request, *, request_id):
        """Run or resume the original public numerical experiment, then read its verdict."""
        from aka.evaluation.numerical import decode_numerical_artifact
        with self._open() as experiment:
            try:
                experiment.find(request_id)
            except LookupError:
                record = experiment.run_recorded(request, request_id=request_id)
            else:
                record = experiment.resume_recorded(request, request_id=request_id,
                    recovery_timeout=request.numerical_probe.timeout, max_remeasurements=0)
            if record.lifecycle != "terminal" or record.phase != "complete" or record.evaluation is None:
                raise ValueError("original numerical experiment is pending")
            artifacts = [item for item in record.artifacts if item.artifact_id == record.evaluation.artifact]
            if len(artifacts) != 1:
                raise ValueError("numerical experiment has no unique interpreted artifact")
            value = decode_numerical_artifact(experiment.read_artifact(artifacts[0]))
            return {**value['summary'], 'probes': value['probes'], 'experiment_ref': record.ref.as_dict()}

    def _record(self, experiment, record_id):
        if not isinstance(record_id, str) or RECORD_ID.fullmatch(record_id) is None:
            raise ValueError("Record ID must be the full gateway-... ID returned by this Campaign")
        # The namespace is selected by the host, never by an Agent argument.
        records = tuple(experiment.iter_records())
        found = [record for record in records if record.ref.experiment_id == record_id[len("gateway-"):]]
        if len(found) != 1:
            raise LookupError("Record is not available in this Campaign")
        return found[0]


    def read(self, record_id):
        """Return Workflow facts from public owner DTOs, not a file record envelope."""
        with self._open() as experiment:
            return self._evidence(experiment, self._record(experiment, record_id))

    evidence = read

    def records(self):
        with self._open() as experiment:
            return tuple(self._evidence(experiment, record) for record in experiment.iter_records())

    def read_kernel(self, kernel_id):
        if not isinstance(kernel_id, str) or KERNEL_ID.fullmatch(kernel_id) is None:
            raise ValueError("Kernel ID must be the full kernel-... ID returned by this Campaign")
        with self._open() as experiment:
            for record in experiment.iter_records():
                source = self._source(experiment, record)
                if source is not None and self.kernel(source)["kernel_id"] == kernel_id:
                    return source
        raise LookupError("Kernel is not available in this Campaign")

    source = read_kernel

    def _response(self, experiment, record, *, reused=False):
        evidence = self._evidence(experiment, record)
        payload = dict(evidence["result"], gateway_record_id=evidence["gateway_record_id"])
        if evidence.get("kernel_id"):
            payload["kernel_id"] = evidence["kernel_id"]
        prefix = _PREFIXES.get(evidence["gateway_kind"])
        returned = experiment.command_result(record.ref) if record.kind == "command" and record.phase == "complete" else None
        if returned is not None and evidence["gateway_kind"] != "same_allocation_abba":
            # The owner supplies its interpreted final command receipt. This
            # preserves all metrics, diagnostics and outputs for Campaign
            # memory without asking Workflow to know an artifact codec.
            stdout, stderr = returned.stdout, returned.stderr
        elif record.evaluation is not None and prefix:
            stdout = prefix + json.dumps(payload, allow_nan=False) + "\n"
            stderr = ""
        else:
            latest = next((attempt.result for attempt in reversed(record.attempts) if attempt.result is not None), None)
            stdout, stderr = (latest.stdout, latest.stderr) if latest is not None else ("", "")
        if stdout and not stdout.endswith("\n"):
            stdout += "\n"
        stdout += "[sandbox] RECORD_JSON=" + json.dumps({"gateway_record_id": evidence["gateway_record_id"],
            **({"kernel_id": evidence["kernel_id"]} if evidence.get("kernel_id") else {})}) + "\n"
        response = {"exit_code": returned.returncode if returned is not None else evidence["exit_code"] if evidence["exit_code"] is not None else 2,
                    "stdout": stdout, "stderr": stderr, "reused": reused}
        from aka.evaluation.tool_results import canonical_result, with_execution_error
        if returned is not None:
            response = with_execution_error(response, returned)
        operation = "evaluate" if evidence["gateway_kind"] == "run" else evidence["gateway_kind"]
        agent = canonical_result(response, {"operation": operation, "options": {"evaluation_mode": payload.get("mode")}},
            record_id=evidence["gateway_record_id"])
        if operation in {"evaluate", "same_allocation_abba"} or evidence["execution_status"] == "unknown":
            agent["status"] = evidence["execution_status"]
        agent["identity"] = {key: evidence[key] for key in ("gateway_record_id", "kernel_id") if key in evidence}
        for key in ("gateway_record_id", "kernel_id", "baseline_kernel_id"):
            agent.pop(key, None)
        if operation in {"evaluate", "same_allocation_abba"}:
            agent["result"]["correct"] = evidence["result"]["correct"]
            if "qualification" in evidence["result"]:
                agent["result"]["qualification"] = evidence["result"]["qualification"]
                agent["result"]["diagnostics"] = [*agent["result"].get("diagnostics", ()),
                    *(item for item in evidence["result"].get("diagnostics", ())
                      if item.get("category") == "qualification" and item not in agent["result"].get("diagnostics", ()))]
            if evidence["result"]["correct"] is None:
                agent["result"]["correctness"] = None
        return {**response, "agent_result": agent}

    def execute(self, request, *, request_id=None, reuse_completed=False, reuse_correctness=False):
        if not isinstance(request, (CommandSpec, EvaluationRequest)):
            raise TypeError("Measurement requires public CommandSpec or EvaluationRequest")
        identities = ()
        identity = request_id or uuid.uuid4().hex
        with self._open() as experiment:
            try:
                existing = experiment.find(identity)
            except LookupError:
                existing = None
            if existing is not None:
                if existing.phase != "complete" or existing.lifecycle != "terminal":
                    if self.environment_identity:
                        prepared = (self.command_evaluator if isinstance(request, CommandSpec) else self.evaluator).prepare(request)
                        replacement = experiment.replacement_abba(existing.ref, prepared.measurement_identities,
                            environment_identity=self.environment_identity)
                        if replacement is not None:
                            return self._response(experiment, replacement, reused=True)
                    record = experiment.resume_recorded(request, request_id=identity, max_remeasurements=0)
                    return self._response(experiment, record)
                else:
                    from .host.errors import AgentRequestError
                    record_id = "gateway-" + existing.ref.experiment_id
                    if not (reuse_completed or reuse_correctness):
                        raise AgentRequestError("This request_id already has a completed record.",
                            code="duplicate_request_id", gateway_record_id=record_id,
                            next_action="Read the existing record, or use a new request_id for a new measurement.")
                    record = experiment.run_recorded(request, request_id=identity)
                    if (self._evidence(experiment, record)["execution_status"] != "completed"
                            or self.environment_identity and any(measurement.environment_identity != self.environment_identity
                                for measurement in self._measurements(record))):
                        raise AgentRequestError("The stored evidence cannot qualify this measurement.",
                            code="record_not_reusable", gateway_record_id=record_id,
                            next_action="Read the existing record; use a new request_id for a fresh measurement if needed.")
                    return self._response(experiment, record, reused=True)
            if (reuse_correctness or reuse_completed) and self.environment_identity:
                prepared = (self.command_evaluator if isinstance(request, CommandSpec) else self.evaluator).prepare(request)
                identities = prepared.measurement_identities
            if reuse_correctness and self.environment_identity:
                reused = experiment.correctness_reuse(identities, environment_identity=self.environment_identity)
                if reused is not None:
                    return self._response(experiment, reused, reused=True)
            if reuse_completed and self.environment_identity and identities:
                for record in reversed(tuple(experiment.iter_records())):
                    if (experiment.reusable(record.ref, identities) and all(
                            measurement.environment_identity == self.environment_identity for measurement in self._measurements(record))):
                        return self._response(experiment, record, reused=True)
            record = experiment.run_recorded(request, request_id=identity)
            return self._response(experiment, record)

    def query(self, args, workspace):
        if args.kind not in QUERY_KINDS or getattr(args, "command", ()) or getattr(args, "input", ()) or getattr(args, "baseline_path", None):
            raise ValueError("Record queries take IDs, not commands or GPU inputs")
        with self._open() as experiment:
            if args.kind == "record-read":
                return self._response(experiment, self._record(experiment, args.record_id), reused=True)
        if args.kind == "kernel-read":
            path = Path(args.output_path or "")
            if path.is_absolute() or ".." in path.parts or len(path.parts) < 2 or path.parts[0] != "scratch":
                raise ValueError("Kernel output must be inside scratch/")
            source = self.read_kernel(args.kernel_id)
            # The Workspace owner decides file/tree installation and path policy.
            destination = Path(workspace) / path
            if len(source.files) > 1:
                if destination.exists():
                    if self.workspace_provider.source_identity(self.workspace_provider.read_source(destination)) != self.workspace_provider.source_identity(source):
                        raise ValueError("Source output directory contains different content")
                else:
                    self.workspace_provider.install_source(destination, source)
            else:
                self.workspace_provider.install_files(Path(workspace), (WorkspaceFile(path.as_posix(), text=source.files["kernel.py"].decode()),), immutable=True)
            result = {"ok": True, "kernel_id": args.kernel_id, "file": path.as_posix()}
        elif args.kind == "kernel-records":
            result = {"kernel_id": args.kernel_id, "gateway_records": [record for record in self.records() if record.get("kernel_id") == args.kernel_id]}
        else:
            if self.query_request is None:
                raise ValueError("Per-Shape history requires the host's declared full Task measurement")
            prepared = self.command_evaluator.prepare(self.query_request)
            with self._open() as experiment:
                minima = experiment.shape_minima(prepared.measurement_identities, environment_identity=self.environment_identity)
                best = {item.shape_id: {"kernel_id": self.kernel(experiment.read_source(item.source))["kernel_id"],
                    "gateway_record_id": "gateway-" + item.experiment.experiment_id, "latency_us": item.latency_us} for item in minima}
            result = {"best_by_shape": best, "shape_count": len(best), "winner_kernel_ids": sorted({value["kernel_id"] for value in best.values()})}
        return {"exit_code": 0, "stdout": json.dumps(result, allow_nan=False) + "\n", "stderr": ""}
