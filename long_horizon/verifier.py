"""Compatibility verifier coordinating original executions through Evaluation."""

from __future__ import annotations
import subprocess
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from contextvars import copy_context
from pathlib import Path
from threading import Event

from orchestrator._bootstrap import task_modules

task_modules()
from orchestrator.infrastructure_retry import (
    InfrastructureUnavailable,
    check_transport,
    retry_infrastructure,
)
from aka.contracts.acceptance import AcceptanceRequirements
from aka.contracts.evaluation import (
    EvaluationRequest,
    VerificationResult,
    VerificationRun,
)
from aka.contracts.execution import JobResult
from aka.legacy.acceptance import LegacyAcceptancePolicy
from aka.legacy.evaluation import EvaluationConfiguration, LegacyEvaluator
from aka.legacy.evaluation.abba import (
    ABBA_RESULT_PREFIX,
    DEFAULT_SHAPE_BATCH_SIZE,
    DEFAULT_SHAPE_BATCH_WORKERS,
    _geomean,
    _merge_batch_results,
    _merge_batch_payloads,
    _payload_from_stdout,
    _result_latency,
    _result_performance_score,
    _verification_shape_batches,
    interpret_verification_payload,
    verification_schedule,
)
from . import main_adapter


def score_verification_payload(
    payload, *, schedule, repeats, min_improvement_pct, artifact=""
):
    result = interpret_verification_payload(
        payload, schedule=schedule, repeats=repeats, artifact=artifact
    )
    return (
        LegacyAcceptancePolicy()
        .decide(result, AcceptanceRequirements(min_improvement_pct))
        .verification
    )


def _snapshot_revision(
    workspace, revision, changed_paths, destination, label, *, provider=None
):
    from aka.legacy.application.workspace import candidate_workspace

    with candidate_workspace(provider=provider) as provider:
        return provider.write_revision(
            workspace, revision, changed_paths, destination, label
        )


class GatewayABBAValidator:
    def __init__(
        self,
        *,
        hardware: str,
        profile: str = "",
        url: str = "",
        ssh: str = "",
        ssh_init: str = "",
        health_command: str = "",
        timeout: int = 600,
        repeats: int = 2,
        per_run_timeout: int = 120,
        min_improvement_pct: float = 0.0,
        queue_wait_grace: int = 14_400,
        private_reference_dir: Path | None = None,
        shape_batch_size: int = DEFAULT_SHAPE_BATCH_SIZE,
        workspace_provider=None,
    ):
        self.workspace_provider = workspace_provider
        self.hardware = hardware
        self.profile = profile
        self.url = url
        self.ssh = ssh
        self.ssh_init = ssh_init
        self.health_command = health_command
        self.timeout = timeout
        self.repeats = max(1, repeats)
        self.per_run_timeout = per_run_timeout
        self.min_improvement_pct = min_improvement_pct
        self.queue_wait_grace = queue_wait_grace
        self.private_reference_dir = private_reference_dir
        self.shape_batch_size = shape_batch_size

    def verify(self, workspace, *, base_commit, candidate_commit, changed_paths):
        from aka.legacy.application.workspace import candidate_workspace

        with candidate_workspace(provider=self.workspace_provider) as provider:
            evaluator = LegacyEvaluator(
                EvaluationConfiguration(
                    timeout=self.timeout,
                    repeats=self.repeats,
                    per_run_timeout=self.per_run_timeout,
                    shape_batch_size=self.shape_batch_size,
                    max_parallelism=1 if self.ssh else DEFAULT_SHAPE_BATCH_WORKERS,
                    private_reference_dir=self.private_reference_dir,
                ),
                candidate_workspace=provider,
            )
            prepared = evaluator.prepare(
                EvaluationRequest(
                    workspace,
                    base_commit=base_commit,
                    candidate_commit=candidate_commit,
                    changed_paths=tuple(changed_paths),
                )
            )
        if not prepared.steps:
            return evaluator.interpret(prepared, ())
        cancel = Event()

        def run_batch(index):
            spec = prepared.steps[index]

            def evaluate_batch():
                try:
                    process = main_adapter.run_sandbox(
                        spec.workspace,
                        self.hardware,
                        self.profile,
                        self.url,
                        spec.timeout,
                        list(spec.command),
                        ssh=self.ssh,
                        ssh_init=self.ssh_init,
                        health_command=self.health_command,
                        sync=(),
                        wall_timeout=self.timeout + self.queue_wait_grace + 120,
                        gateway_kind=spec.operation,
                        private_reference_dir=self.private_reference_dir,
                    )
                except subprocess.TimeoutExpired as exc:
                    raise InfrastructureUnavailable(
                        "GPU transport wait deadline exceeded"
                    ) from exc
                result = JobResult(process.returncode, process.stdout, process.stderr)
                if process.returncode == 0:
                    evaluator.record_step(prepared, index, result)
                    return result
                check_transport(process)
                raise RuntimeError(
                    f"gateway ABBA command exited {process.returncode}: "
                    + (process.stdout + "\n" + process.stderr)[-3000:]
                )

            return retry_infrastructure(
                workspace,
                f"abba:{base_commit}:{candidate_commit}:{self.repeats}:{self.shape_batch_size}:"
                + Path(spec.command[-2]).name,
                evaluate_batch,
                cancel=cancel,
            )

        try:
            executor = ThreadPoolExecutor(max_workers=prepared.max_parallelism)
            futures = [
                executor.submit(copy_context().run, run_batch, index)
                for index in range(len(prepared.steps))
            ]
            try:
                completed, _ = wait(futures, return_when=FIRST_EXCEPTION)
                for future in completed:
                    future.result()
                results = [future.result() for future in futures]
            except BaseException:
                cancel.set()
                for future in futures:
                    future.cancel()
                raise
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
            result = evaluator.interpret(prepared, results)
        except (
            subprocess.SubprocessError,
            RuntimeError,
            ValueError,
            KeyError,
            TypeError,
        ) as exc:
            return VerificationResult(
                "ERROR",
                None,
                None,
                None,
                error=f"gateway ABBA verification failed: {exc}",
            )
        return (
            LegacyAcceptancePolicy()
            .decide(result, AcceptanceRequirements(self.min_improvement_pct))
            .verification
        )
