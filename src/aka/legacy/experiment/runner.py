"""Run prepared batches without owning measurement algorithms or host bindings."""
from __future__ import annotations

from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from contextvars import copy_context
from pathlib import Path
import subprocess
from threading import Event
from typing import TYPE_CHECKING, Callable

from aka.contracts.execution import CommandSpec, Executor, JobResult
from aka.contracts.experiment import ExperimentExecutionError
from aka.contracts.evaluation import EvaluationUnavailable

if TYPE_CHECKING:
    from aka.contracts.evaluation import EvaluationRequest, Evaluator, VerificationResult


class LegacyExperiment:
    def __init__(
        self, executor: Executor, evaluator: Evaluator | None = None, *,
        retry: Callable | None = None,
        check_result: Callable[[JobResult], None] | None = None,
        stage: Callable[[EvaluationRequest, int, CommandSpec], str] | None = None,
    ):
        self.executor = executor
        self.evaluator = evaluator
        self.retry = retry
        self.check_result = check_result
        self.stage = stage
        if retry is not None and stage is None:
            raise ValueError("an injected retry requires a stage identity")

    def run_command(self, command: CommandSpec) -> JobResult:
        return self.executor.run(command)

    def run(self, request: EvaluationRequest) -> VerificationResult:
        if self.evaluator is None:
            raise ValueError("evaluation requires an injected Evaluator")
        prepared = self.evaluator.prepare(request)
        steps = prepared.steps
        if not steps:
            return self.evaluator.interpret(prepared, ())
        if prepared.max_parallelism < 1:
            raise ValueError("evaluation parallelism must be positive")
        cancel = Event()

        def run_step(index: int) -> JobResult:
            command = steps[index]

            def attempt():
                result = self.executor.run(command)
                if self.check_result is not None:
                    self.check_result(result)
                self.evaluator.record_step(prepared, index, result)
                return result

            if self.retry is None:
                return attempt()
            stage = self.stage(request, index, command)
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
            if isinstance(error, (subprocess.SubprocessError, RuntimeError, ValueError, KeyError, TypeError)):
                raise ExperimentExecutionError(str(error)) from error
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        try:
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
