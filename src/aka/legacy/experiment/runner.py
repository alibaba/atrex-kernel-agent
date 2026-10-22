"""Run prepared batches without owning measurement algorithms or host bindings."""
from __future__ import annotations

from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from contextvars import copy_context
from pathlib import Path
from dataclasses import asdict, replace
import subprocess
from threading import Event
from typing import TYPE_CHECKING, Callable

from aka.contracts.execution import CommandSpec, Executor, JobResult
from aka.contracts.experiment import ExperimentExecutionError, ExperimentRecordError
from aka.contracts.evaluation import EvaluationRequest, EvaluationUnavailable

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
    ):
        self.executor = executor
        self.evaluator = evaluator
        self.retry = retry
        self.check_result = check_result
        self.stage = stage
        self.records = records
        self.purpose = purpose
        if purpose not in {"development", "formal"}:
            raise ValueError("unsupported experiment purpose")
        if retry is not None and stage is None:
            raise ValueError("an injected retry requires a stage identity")

    def run_command(self, command: CommandSpec) -> JobResult:
        if self.records is not None:
            import uuid
            return self._recorded(command, uuid.uuid4().hex)[1]
        return self.executor.run(command)

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

    def _request_data(self, request):
        if isinstance(request, CommandSpec):
            import hashlib
            import json
            return {"workspace": str(request.workspace.resolve()), "argv": list(request.command),
                    "operation": request.operation, "timeout": request.timeout, "hardware": request.hardware,
                    "artifacts": list(request.artifacts), "environment_keys": sorted(request.environment),
                    "environment_digest": hashlib.sha256(json.dumps(dict(request.environment),
                        sort_keys=True).encode()).hexdigest(), "cancellation": request.cancel_event is not None}
        data = asdict(request)
        data.pop("experiment_ref", None)
        for name in ("workspace", "canonical_workspace"):
            data[name] = str(data[name].resolve()) if data[name] is not None else None
        data["changed_paths"] = list(data["changed_paths"])
        data["expected_shape_ids"] = sorted(data["expected_shape_ids"]) if data["expected_shape_ids"] is not None else None
        return data

    def _recorded(self, request, request_id):
        records = self._records()
        if not isinstance(request, (EvaluationRequest, CommandSpec)):
            raise TypeError("recorded experiment requires EvaluationRequest or CommandSpec")
        if isinstance(request, EvaluationRequest) and self.evaluator is None:
            raise ValueError("evaluation requires an injected Evaluator")
        with records.operation(request_id) as ref:
            record, created = records.register(ref, request_id, self._request_data(request),
                kind="command" if isinstance(request, CommandSpec) else "evaluation", purpose=self.purpose)
            if not created:
                if record.phase == "complete" and record.outcome is not None:
                    return record, None
                raise ExperimentRecordError("experiment_incomplete", ref)
            try:
                if isinstance(request, CommandSpec):
                    records.prepared(ref, (request,))
                    attempt_id = records.intent(ref, 0)
                    try:
                        result = self.executor.run(request)
                    except BaseException as error:
                        records.observed(ref, attempt_id, error=error)
                        raise
                    records.observed(ref, attempt_id, result=result)
                    for relative in request.artifacts:
                        artifact_path = Path(relative)
                        if artifact_path.is_absolute() or ".." in artifact_path.parts:
                            raise ValueError("experiment artifacts must be workspace-relative files")
                        path = request.workspace / artifact_path
                        if not path.resolve().is_relative_to(request.workspace.resolve()):
                            raise ValueError("experiment artifact escapes its workspace")
                        if path.is_file():
                            records.publish(ref, path, "command")
                        else:
                            records.artifact_missing(ref)
                    record = records.completed(ref)
                    return record, result
                if request.experiment_ref is not None and request.experiment_ref != ref:
                    raise ValueError("evaluation experiment reference mismatch")
                request = replace(request, experiment_ref=ref)
                result = self._run(request, records=records, ref=ref)
                if result.experiment_ref is not None and result.experiment_ref != ref:
                    raise ValueError("evaluation returned another experiment reference")
                result = replace(result, experiment_ref=ref)
                value = result.as_dict()
                if value["artifact"]:
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

    def _run(self, request: EvaluationRequest, *, records=None, ref=None) -> VerificationResult:
        if self.evaluator is None:
            raise ValueError("evaluation requires an injected Evaluator")
        prepared = self.evaluator.prepare(request)
        steps = prepared.steps
        if records is not None:
            records.prepared(ref, steps)
        if not steps:
            if records is not None:
                records.phase(ref, "interpret")
            return self.evaluator.interpret(prepared, ())
        if prepared.max_parallelism < 1:
            raise ValueError("evaluation parallelism must be positive")
        cancel = Event()

        def run_step(index: int) -> JobResult:
            command = steps[index]

            def attempt():
                attempt_id = records.intent(ref, index) if records is not None else None
                try:
                    result = self.executor.run(command)
                except BaseException as error:
                    if records is not None:
                        records.observed(ref, attempt_id, error=error)
                    raise
                if records is not None:
                    records.observed(ref, attempt_id, result=result)
                result.require_transport()
                if self.check_result is not None:
                    self.check_result(result)
                self.evaluator.record_step(prepared, index, result)
                if records is not None:
                    artifacts = getattr(self.evaluator, "artifacts", None)
                    if callable(artifacts):
                        for artifact in artifacts(prepared, index):
                            records.publish(ref, artifact.path, artifact.role)
                return result

            if self.retry is None:
                return attempt()
            stage = prepared.retry_keys[index] if prepared.retry_keys else self.stage(request, index, command)
            return self.retry(request.workspace, stage, attempt, cancel=cancel)

        pool = ThreadPoolExecutor(max_workers=min(prepared.max_parallelism, len(steps)))
        futures = []
        try:
            futures = [pool.submit(copy_context().run, run_step, i) for i in range(len(steps))]
            completed, _ = wait(futures, return_when=FIRST_EXCEPTION)
            for future in completed:
                future.result()
            results = tuple(future.result() for future in futures)
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
            pool.shutdown(wait=True, cancel_futures=True)
        try:
            if records is not None:
                records.phase(ref, "interpret")
            return self.evaluator.interpret(prepared, results)
        except EvaluationUnavailable:
            raise
        except (subprocess.SubprocessError, RuntimeError, ValueError, KeyError, TypeError) as error:
            raise ExperimentExecutionError(str(error)) from error


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
