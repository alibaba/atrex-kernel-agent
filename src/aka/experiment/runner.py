"""Run prepared batches without owning measurement algorithms or host bindings."""
from __future__ import annotations

from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from contextvars import copy_context
from pathlib import Path
from dataclasses import asdict, replace
import subprocess
import json
from threading import Event
from typing import TYPE_CHECKING, Callable

from aka.contracts.execution import CommandSpec, Executor, JobResult
from aka.contracts.experiment import ExperimentExecutionError, ExperimentRecordError
from aka.contracts.evaluation import (
    EvaluationRequest, EvaluationUnavailable, StepEvaluation, VerificationResult,
)
from .file_records import _atomic

if TYPE_CHECKING:
    from aka.contracts.evaluation import EvaluationRequest, Evaluator, VerificationResult


class LegacyExperiment:
    def __init__(
        self, executor: Executor, evaluator: Evaluator | None = None, *,
        retry: Callable | None = None,
        check_result: Callable[[JobResult], None] | None = None,
        stage: Callable[[EvaluationRequest, int, CommandSpec], str] | None = None,
        records=None,
        purpose: str = "development",
        command_evaluator=None,
    ):
        self.executor = executor
        self.evaluator = evaluator
        self.retry = retry
        self.check_result = check_result
        self.stage = stage
        self.records = records
        self.purpose = purpose
        if command_evaluator is None:
            from aka.evaluation.gateway_command import GatewayCommandEvaluator
            command_evaluator = GatewayCommandEvaluator()
        self.command_evaluator = command_evaluator
        if purpose not in {"development", "formal"}:
            raise ValueError("unsupported experiment purpose")
        if retry is not None and stage is None:
            raise ValueError("an injected retry requires a stage identity")

    def run_command(self, command: CommandSpec) -> JobResult:
        if self.records is not None:
            import uuid
            return self._recorded(command, uuid.uuid4().hex)[1]
        return self._run_command(command)

    def run(self, request: EvaluationRequest) -> VerificationResult:
        if self.records is not None:
            import uuid
            return self._recorded(request, uuid.uuid4().hex)[1]
        return self._run(request)

    def get(self, ref):
        return self._records().get(ref)

    def find(self, request_id):
        return self._records().find(request_id)

    def unresolved(self):
        return self._records().unresolved()

    def read_artifact(self, artifact):
        return self._records().read_artifact(artifact)

    def _records(self):
        if self.records is None:
            raise ExperimentRecordError("recording_not_configured")
        return self.records

    def run_recorded(self, request, *, request_id):
        return self._recorded(request, request_id)[0]

    def resume_recorded(self, request, *, request_id, recovery_timeout=None, max_remeasurements=0):
        if type(max_remeasurements) is not int or max_remeasurements < 0:
            raise ValueError("measurement retry limit must be nonnegative")
        if recovery_timeout is not None and (type(recovery_timeout) not in (int, float) or recovery_timeout <= 0):
            raise ValueError("recovery timeout must be positive")
        return self._recorded(request, request_id, resume={"timeout": recovery_timeout,
                              "remeasurements": max_remeasurements})[0]

    def reusable(self, ref, identities):
        """Experiment qualifies stored evidence without an Evaluation approval call."""
        record = self.get(ref)
        measurements = record.evaluation.measurements if record.evaluation else ()
        if record.phase != "complete" or record.lifecycle != "terminal" or not measurements:
            return False
        return tuple(item.identity.digest for item in measurements) == tuple(item.digest for item in identities) and all(
            item.completeness == "complete" and item.validity == "valid" and item.comparability == "comparable"
            for item in measurements) and len({item.environment_identity for item in measurements}) == 1

    def _request_data(self, request):
        if isinstance(request, CommandSpec):
            import hashlib
            import json
            from .command_identity import handoff_identity
            return {"workspace": str(request.workspace.resolve()), "argv": list(request.command),
                    "operation": request.operation, "timeout": request.timeout, "hardware": request.hardware, "num_gpus": request.num_gpus,
                    "languages": list(request.languages), "requirements": list(request.requirements),
                    "artifacts": list(request.artifacts), "environment_keys": sorted(request.environment),
                    "environment_digest": hashlib.sha256(json.dumps(dict(request.environment),
                        sort_keys=True).encode()).hexdigest(), "cancellation": request.cancel_event is not None, "submission_sha256":
                    hashlib.sha256(request.submission).hexdigest() if request.submission is not None else None,
                    **handoff_identity(request)}
        data = asdict(request)
        data.pop("experiment_ref", None)
        for name in ("workspace", "canonical_workspace"):
            data[name] = str(data[name].resolve()) if data[name] is not None else None
        data["changed_paths"] = list(data["changed_paths"])
        data["expected_shape_ids"] = sorted(data["expected_shape_ids"]) if data["expected_shape_ids"] is not None else None
        return data

    def _recorded(self, request, request_id, *, resume=None):
        records = self._records()
        if not isinstance(request, (EvaluationRequest, CommandSpec)):
            raise TypeError("recorded experiment requires EvaluationRequest or CommandSpec")
        if isinstance(request, EvaluationRequest) and self.evaluator is None:
            raise ValueError("evaluation requires an injected Evaluator")
        with records.operation(request_id) as ref:
            record, created = records.register(ref, request_id, self._request_data(request),
                kind="command" if isinstance(request, CommandSpec) else "evaluation", purpose=self.purpose)
            if not created:
                if record.phase == "complete" and record.outcome is not None and (resume is None or record.lifecycle == "terminal"):
                    return record, None
                if resume is None:
                    raise ExperimentRecordError("experiment_incomplete", ref)
                if not record.plan_digest:
                    raise ExperimentRecordError("experiment_not_prepared", ref)
            try:
                if isinstance(request, CommandSpec):
                    result = self._run_command(request, records=records, ref=ref, resume=resume if not created else None)
                    for relative in request.artifacts:
                        artifact_path = Path(relative)
                        if artifact_path.is_absolute() or ".." in artifact_path.parts:
                            raise ValueError("experiment artifacts must be workspace-relative files")
                        from aka.contracts.filesystem_access import require_path_access
                        path = require_path_access(request.filesystem_access,
                            request.workspace.absolute() / artifact_path)
                        if request.filesystem_access is None and not path.resolve().is_relative_to(request.workspace.resolve()):
                            raise ValueError("experiment artifact escapes its workspace")
                        if path.is_file():
                            records.publish(ref, path, "command")
                        elif path.is_dir() and any(item.path.startswith(relative.rstrip("/") + "/") for item in result.files):
                            pass
                        else:
                            records.artifact_missing(ref)
                    from aka.evaluation.evaluator import result_from_stdout
                    evaluation = None
                    try:
                        payload = result_from_stdout(result.stdout)
                    except (RuntimeError, json.JSONDecodeError):
                        pass
                    else:
                        evaluation = VerificationResult("PASS" if payload.get("all_pass") else "FAIL",
                            payload.get("latency_us_geomean"), None, None, experiment_ref=ref).as_dict()
                        records.publish_content(ref, json.dumps(payload, allow_nan=False).encode(), "evaluation")
                    record = records.completed(ref, evaluation)
                    return record, result
                if request.experiment_ref is not None and request.experiment_ref != ref:
                    raise ValueError("evaluation experiment reference mismatch")
                request = replace(request, experiment_ref=ref)
                result = self._run(request, records=records, ref=ref, resume=resume if not created else None)
                if result.experiment_ref is not None and result.experiment_ref != ref:
                    raise ValueError("evaluation returned another experiment reference")
                result = replace(result, experiment_ref=ref)
                value = result.as_dict()
                if result.artifacts:
                    primary = None
                    for output in result.artifacts:
                        artifact = records.publish_content(ref, output.content, output.role)
                        if output.name == Path(result.artifact).name:
                            primary = artifact.artifact_id
                    value["artifact"] = primary
                elif value["artifact"]:
                    artifact_path = Path(value["artifact"])
                    if not artifact_path.is_absolute():
                        artifact_path = request.workspace / artifact_path
                    if artifact_path.is_file():
                        artifact = records.publish(ref, artifact_path, "evaluation")
                        value["artifact"] = artifact.artifact_id
                    else:
                        records.artifact_missing(ref)
                        value["artifact"] = None
                return records.completed(ref, value), result
            except BaseException as error:
                try:
                    records.failed(ref, error)
                except ExperimentRecordError as storage_error:
                    if hasattr(error, "add_note"):
                        error.add_note(str(storage_error))
                # Keep existing exception types, with a durable identity for readers.
                try:
                    error.experiment_ref = ref
                except (AttributeError, TypeError):
                    pass
                raise

    def _resume_results(self, records, ref, commands, evaluator, prepared, options):
        latest = {}
        history = {}
        for attempt in records.get(ref).attempts:
            latest[attempt.task_id] = attempt
            history.setdefault(attempt.task_id, []).append(attempt)
        cached = {}
        for index, command in enumerate(commands):
            attempt = latest.get(f"step-{index}")
            if attempt is None:
                continue
            result = attempt.result
            if attempt.disposition in {"intent", "unknown"}:
                query = getattr(self.executor, "query_job", None)
                references = result.job_references if result else ()
                if not callable(query) or len(references) != 1:
                    raise ExperimentRecordError("experiment_pending", ref)
                observed = query(references[0], timeout=options["timeout"] or command.timeout)
                if observed.job_references != references:
                    raise ExperimentRecordError("job_identity_mismatch", ref)
                if result is not None:
                    observed = replace(observed, attempts=(*result.attempts, *observed.attempts),
                                       causes=(*result.causes, *observed.causes),
                                       budgets=observed.budgets or result.budgets,
                                       environment_identity=observed.environment_identity or result.environment_identity,
                                       environment_snapshot=observed.environment_snapshot or result.environment_snapshot,
                                       environment_support=observed.environment_support or result.environment_support)
                records.observed(ref, attempt.attempt_id, result=observed)
                for item in observed.files:
                    records.publish_content(ref, item.content, "execution_output")
                result = observed
                if result.facts.disposition == "unknown":
                    raise ExperimentRecordError("experiment_pending", ref)
            assessment = getattr(evaluator, "diagnose_step", lambda *args: None)(prepared, index, result) if result is not None else None
            if assessment is not None:
                records.assessed(ref, attempt.attempt_id, assessment)
            retryable = result is not None and result.transport_status in {"infrastructure", "timed_out"}
            if assessment is not None and (assessment.validity == "invalid" or assessment.completeness == "complete"):
                retryable = False
            if retryable and options["remeasurements"] and len(history[attempt.task_id]) <= options["remeasurements"]:
                if result.facts.disposition == "not_submitted" or result.facts.remote_terminated is True:
                    continue  # Experiment explicitly schedules one missing measurement.
            if result is None or result.facts.disposition == "unknown":
                raise ExperimentRecordError("experiment_pending", ref)
            if result.transport_status in {"timed_out", "cancelled", "interrupted", "infrastructure"}:
                result.require_transport()
            cached[index] = records.result_for(ref, attempt.attempt_id)
        return cached

    @staticmethod
    def _observed_command(command, records, ref, attempt_id):
        if records is None:
            return command
        def observe(value):
            records.observed(ref, attempt_id, result=value)
            if command.on_observation is not None:
                command.on_observation(value)
        return replace(command, on_observation=observe)

    def _run_command(self, command, *, records=None, ref=None, resume=None):
        from .measurement_files import record_result_lines
        evaluator = self.command_evaluator
        prepared = evaluator.prepare(command)
        capabilities = getattr(self.executor, "capabilities", None)
        if capabilities is not None:
            for step in prepared.steps:
                capabilities.require(step)
        commands = (*prepared.steps, *prepared.fallback_steps)
        cached = {}
        if records is not None:
            records.prepared(ref, commands, identities=prepared.measurement_identities, resume=resume is not None)
            if resume is not None:
                cached = self._resume_results(records, ref, commands, evaluator, prepared, resume)
            for step in prepared.steps:
                if step.submission is not None:
                    records.publish_content(ref, step.submission, "evaluation_input")
                for item in step.input_files:
                    records.publish_content(ref, item.content, "execution_input")
        cancel = Event()
        def attempt(index):
            if index in cached:
                result = cached[index]
                if not isinstance(prepared.context, dict) or prepared.context.get("kind") != "profile" or prepared.context.get("measurement") is None:
                    self._publish_files(commands[index], result.files, records, ref)
                return result
            if cancel.is_set():
                from concurrent.futures import CancelledError
                raise CancelledError("experiment batch cancelled before submission")
            attempt_id = records.intent(ref, index) if records is not None else None
            try:
                capabilities = getattr(self.executor, "capabilities", None)
                if capabilities is not None:
                    capabilities.require(commands[index])
                command = commands[index]
                command = self._observed_command(command, records, ref, attempt_id)
                result = self.executor.run(command)
            except BaseException as error:
                if records is not None:
                    records.observed(ref, attempt_id, error=error)
                raise
            if records is not None:
                records.observed(ref, attempt_id, result=result)
                assessment = getattr(evaluator, "diagnose_step", lambda *args: None)(prepared, index, result)
                if assessment is not None:
                    records.assessed(ref, attempt_id, assessment)
                for output in evaluator.observe_step(prepared, index, result).artifacts:
                    records.publish_content(ref, output.content, output.role)
            if not isinstance(prepared.context, dict) or prepared.context.get("kind") != "profile" or prepared.context.get("measurement") is None:
                self._publish_files(command, result.files, records, ref)
            elif records is not None:
                for item in result.files:
                    records.publish_content(ref, item.content, "execution_output")
            try:
                result.require_transport()
            except BaseException:
                cancel.set()
                raise
            return result
        if prepared.max_parallelism == 1:
            results = tuple(copy_context().run(attempt, index) for index in range(len(prepared.steps)))
        else:
            with ThreadPoolExecutor(max_workers=prepared.max_parallelism) as pool:
                futures = [pool.submit(copy_context().run, attempt, index) for index in range(len(prepared.steps))]
                results = tuple(future.result() for future in futures)
        if evaluator.fallback(prepared, results):
            results = tuple(attempt(len(prepared.steps) + index) for index in range(len(prepared.fallback_steps)))
        result = evaluator.interpret(prepared, results)
        if isinstance(prepared.context, dict) and prepared.context.get("measurement") is not None:
            self._publish_files(command, result.files, records, ref)
        if isinstance(prepared.context, dict) and prepared.context.get("evaluator"):
            job_ids = ",".join(str(json.loads(item.stdout).get("job_id", "")) for item in results
                if item.stdout.lstrip().startswith("{") and prepared.context["measurement"] is not None)
            record_result_lines(command.workspace, result.stdout, gateway_kind=prepared.context["kind"], job_id=job_ids)
        return result

    def _publish_files(self, request, files, records, ref):
        from aka.contracts.evaluation import PreparedEvaluation, PreparationFile
        if not files:
            return
        prepared = PreparedEvaluation((), 1, None, inputs=tuple(
            PreparationFile(item.path, item.content, item.mode) for item in files))
        self._write_inputs(request, prepared, None, ref)
        if records is not None:
            for item in files:
                records.publish_content(ref, item.content, "execution_output")

    def _run(self, request: EvaluationRequest, *, records=None, ref=None, resume=None) -> VerificationResult:
        if self.evaluator is None:
            raise ValueError("evaluation requires an injected Evaluator")
        prepared = self.evaluator.prepare(request)
        steps = prepared.steps
        if steps and prepared.max_parallelism < 1:
            raise ValueError("evaluation parallelism must be positive")
        cached = {}
        if records is not None and resume is not None:
            records.prepared(ref, (*steps, *prepared.fallback_steps), prepared.inputs,
                             prepared.measurement_identities, resume=True)
            cached = self._resume_results(records, ref, (*steps, *prepared.fallback_steps), self.evaluator, prepared, resume)
        self._write_inputs(request, prepared, records, ref)
        if records is not None:
            for command in steps:
                if command.submission is not None:
                    records.publish_content(ref, command.submission, "evaluation_input")
        if records is not None and resume is None:
            records.prepared(ref, (*steps, *prepared.fallback_steps), prepared.inputs, prepared.measurement_identities)
        if not steps:
            if records is not None:
                records.phase(ref, "interpret")
            return self._interpret(request, prepared, (), ref)
        cancel = Event()

        def run_step(index: int) -> JobResult:
            command = steps[index]
            if index in cached:
                result = cached[index]
                self._publish_files(request, result.files, records, ref)
                if self.check_result is not None:
                    self.check_result(result)
                observed = self.evaluator.observe_step(prepared, index, result)
                self._write_outputs(request, prepared, observed.artifacts, ref)
                return result

            def attempt():
                attempt_id = records.intent(ref, index) if records is not None else None
                try:
                    observed_command = self._observed_command(command, records, ref, attempt_id)
                    result = self.executor.run(observed_command)
                except BaseException as error:
                    if records is not None:
                        records.observed(ref, attempt_id, error=error)
                    raise
                if records is not None:
                    records.observed(ref, attempt_id, result=result)
                    assessment = getattr(self.evaluator, "diagnose_step", lambda *args: None)(prepared, index, result)
                    if assessment is not None:
                        records.assessed(ref, attempt_id, assessment)
                self._publish_files(request, result.files, records, ref)
                if records is not None and command.submission is not None:
                    records.publish_content(ref, result.stdout.encode("utf-8"), "measurement_receipt")
                result.require_transport()
                if self.check_result is not None:
                    self.check_result(result)
                observed = self.evaluator.observe_step(prepared, index, result)
                if not isinstance(observed, StepEvaluation):
                    raise TypeError("Evaluator.observe_step must return StepEvaluation")
                self._write_outputs(request, prepared, observed.artifacts, ref)
                if records is not None:
                    for output in observed.artifacts:
                        records.publish_content(ref, output.content, output.role)
                return result

            if self.retry is None:
                return attempt()
            stage = prepared.retry_keys[index] if prepared.retry_keys else self.stage(request, index, command)
            return self.retry(request.workspace, stage, attempt, cancel=cancel)

        pool = (ThreadPoolExecutor(max_workers=min(prepared.max_parallelism, len(steps)))
                if prepared.max_parallelism != 1 else None)
        futures = []
        try:
            if pool is None:
                results = tuple(copy_context().run(run_step, i) for i in range(len(steps)))
            else:
                futures = [pool.submit(copy_context().run, run_step, i) for i in range(len(steps))]
                completed, _ = wait(futures, return_when=FIRST_EXCEPTION)
                for future in completed:
                    future.result()
                results = tuple(future.result() for future in futures)
            fallback = getattr(self.evaluator, "fallback", lambda plan, receipts: ())(prepared, results)
            if fallback:
                initial_count = len(steps)
                steps = (*steps, *prepared.fallback_steps)
                results = tuple(run_step(initial_count + index) for index in range(len(fallback)))
        except BaseException as error:
            cancel.set()
            for future in futures:
                future.cancel()
            if isinstance(error, EvaluationUnavailable):
                raise
            if isinstance(error, ExperimentRecordError):
                raise
            if isinstance(error, (subprocess.SubprocessError, RuntimeError, ValueError, KeyError, TypeError)):
                raise ExperimentExecutionError(str(error)) from error
            raise
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
        try:
            if records is not None:
                records.phase(ref, "interpret")
            return self._interpret(request, prepared, results, ref)
        except ExperimentRecordError:
            raise
        except EvaluationUnavailable:
            raise
        except (subprocess.SubprocessError, RuntimeError, ValueError, KeyError, TypeError) as error:
            raise ExperimentExecutionError(str(error)) from error

    def _write_inputs(self, request, prepared, records, ref):
        """Preflight all inputs, then publish bytes before any backend submission."""
        from aka.contracts.filesystem_access import require_path_access
        access = getattr(request, "filesystem_access", None)
        workspace = request.workspace.absolute() if access is not None else request.workspace.resolve()
        targets = {}
        for item in prepared.inputs:
            relative = Path(item.path)
            if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
                raise ValueError("preparation inputs must be workspace-relative files")
            path = require_path_access(access, workspace / relative, write=True)
            if access is not None:
                from .scoped_publication import validate_target
                validate_target(path)
            if (access is None and not path.resolve().is_relative_to(workspace) or path.is_symlink()
                    or (path.exists() and not path.is_file())):
                raise ValueError("preparation input escapes workspace or is not a regular file")
            if path in targets:
                raise ValueError("duplicate preparation input path")
            if any(parent.is_symlink() or (parent.exists() and not parent.is_dir())
                   for parent in path.parents if parent.is_relative_to(workspace)):
                raise ValueError("preparation input traverses a symlink or file")
            targets[path] = item
        if any(parent in targets for path in targets for parent in path.parents):
            raise ValueError("preparation input paths conflict as file and directory")
        manifest = []
        for path, item in targets.items():
            try:
                if access is not None:
                    from .scoped_publication import publish
                    publish(path, item.content, mode=item.mode)
                else:
                    _atomic(path, item.content, mode=item.mode)
            except (OSError, ValueError) as error:
                raise ExperimentRecordError("storage_unavailable", ref) from error
            if records is not None:
                artifact = records.publish_content(ref, item.content, "evaluation_input")
                manifest.append({"path": item.path, "artifact_id": artifact.artifact_id,
                                 "sha256": artifact.sha256, "mode": item.mode,
                                 "content_type": item.content_type})
        if records is not None and manifest:
            records.publish_content(ref, json.dumps({"schema_version": 1, "files": manifest},
                sort_keys=True).encode("utf-8"), "evaluation_inputs")

    def _write_outputs(self, request, prepared, outputs, ref):
        """Retain optional workspace mirrors; all result writes belong here."""
        if prepared.output_directory is None:
            return {}
        relative = Path(prepared.output_directory)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("evaluation output directory must be workspace-relative")
        workspace = request.workspace.resolve()
        directory = workspace / relative
        if not directory.resolve().is_relative_to(workspace):
            raise ValueError("evaluation outputs escape the workspace")
        paths = {}
        for output in outputs:
            if output.name in paths:
                raise ValueError("duplicate measurement output name")
            path = directory / output.name
            if path.is_symlink():
                raise ValueError("evaluation output cannot replace a symlink")
            try:
                _atomic(path, output.content)
            except (OSError, ValueError) as error:
                raise ExperimentRecordError("storage_unavailable", ref) from error
            paths[output.name] = str(path)
        return paths

    def _interpret(self, request, prepared, results, ref):
        result = self.evaluator.interpret(prepared, results)
        from .measurement_files import record_episode_evaluation
        if request.mode in {"v0", "framework"}:
            for run in result.runs:
                if run.result is not None:
                    record_episode_evaluation(request.workspace, run.result, gateway_kind="run")
        if not isinstance(result, VerificationResult):
            raise TypeError("Evaluator.interpret must return VerificationResult")
        names = [output.name for output in result.artifacts]
        if len(names) != len(set(names)) or (
            names and result.artifact and result.artifact not in names
        ):
            raise ValueError("evaluation primary artifact must name a unique returned output")
        paths = self._write_outputs(request, prepared, result.artifacts, ref)
        if result.artifact in paths:
            result = replace(result, artifact=paths[result.artifact])
        return result


def abba_stage(repeats: int, shape_batch_size: int):
    """Retain the original durable retry identity for each prepared batch."""
    def stage(request, index, command):
        return (
            f"abba:{request.base_commit}:{request.candidate_commit}:"
            f"{max(1, repeats)}:{shape_batch_size}:{Path(command.command[-2]).name}"
        )
    return stage


def evaluation_stage(request, index, command):
    """Stable retry identity for an arbitrary selected Evaluator command."""
    import hashlib
    import json
    digest = hashlib.sha256(json.dumps({
        "command": command.command, "hardware": command.hardware, "operation": command.operation,
        "timeout": command.timeout, "artifacts": command.artifacts,
    }, sort_keys=True).encode()).hexdigest()
    return f"evaluation:{request.mode}:{request.base_commit}:{request.candidate_commit}:{index}:{digest}"
