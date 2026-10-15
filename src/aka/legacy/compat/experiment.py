"""Original verifier request and result projection over injected Experiment."""
from __future__ import annotations

from aka.contracts.experiment import ExperimentExecutionError
from aka.legacy.evaluation.abba import DEFAULT_SHAPE_BATCH_SIZE
from aka.legacy.experiment import LegacyExperiment
from aka.legacy.experiment.runner import abba_stage
from aka.legacy.experiment.transport import ABBAExecutor, check_abba_result


class GatewayABBAValidator:
    def __init__(
        self, *, hardware: str, profile="", url="", ssh="", ssh_init="",
        health_command="", timeout=600, repeats=2, per_run_timeout=120,
        min_improvement_pct=0.0, queue_wait_grace=14_400,
        private_reference_dir=None, shape_batch_size=DEFAULT_SHAPE_BATCH_SIZE, workspace_provider=None,
        experiment=None,
    ):
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
        self.workspace_provider = workspace_provider
        self.experiment = experiment

    def verify(self, workspace, *, base_commit, candidate_commit, changed_paths):
        from aka.contracts.evaluation import EvaluationRequest, VerificationResult
        from aka.contracts.acceptance import AcceptanceRequirements
        from aka.legacy.acceptance import LegacyAcceptancePolicy
        from aka.legacy.application.execution import application_execution
        from aka.legacy.application.experiment_executor import sandbox_executor
        from aka.legacy.application.workspace import candidate_workspace
        from aka.legacy.evaluation import EvaluationConfiguration, LegacyEvaluator
        from aka.legacy.experiment.retry import retry_infrastructure

        request = EvaluationRequest(
            workspace, base_commit=base_commit, candidate_commit=candidate_commit,
            changed_paths=tuple(changed_paths),
        )

        def run(experiment):
            try:
                result = experiment.run(request)
            except ExperimentExecutionError as exc:
                return VerificationResult(
                    "ERROR", None, None, None,
                    error=f"gateway ABBA verification failed: {exc}",
                )
            return LegacyAcceptancePolicy().decide(
                result, AcceptanceRequirements(min_improvement_pct=self.min_improvement_pct),
            ).verification

        if self.experiment is not None:
            return run(self.experiment)
        # The compatibility constructor is the composition root. The runner
        # itself never selects providers or knows sandbox host configuration.
        with application_execution() as executor, candidate_workspace(provider=self.workspace_provider) as provider:
            evaluator = LegacyEvaluator(EvaluationConfiguration(
                timeout=self.timeout, repeats=self.repeats,
                per_run_timeout=self.per_run_timeout, shape_batch_size=self.shape_batch_size,
                max_parallelism=1 if self.ssh else 4,
                private_reference_dir=self.private_reference_dir,
            ), candidate_workspace=provider)
            bound = sandbox_executor(
                executor, hardware=self.hardware, profile=self.profile, url=self.url,
                ssh=self.ssh, ssh_init=self.ssh_init, health_command=self.health_command,
                wall_timeout=self.timeout + self.queue_wait_grace + 120,
                private_reference_dir=self.private_reference_dir,
            )
            return run(LegacyExperiment(
                ABBAExecutor(bound), evaluator, retry=retry_infrastructure,
                check_result=check_abba_result,
                stage=abba_stage(self.repeats, self.shape_batch_size),
            ))


def baseline_command(workspace, hardware, profile, url, timeout, command, *,
                     ssh="", ssh_init="", health_command="",
                     gateway_kind="run", private_reference_dir=None):
    """Execute the original V0/framework request once; retain its local projection."""
    from aka.contracts.execution import CommandSpec
    from aka.legacy.application.execution import application_execution, _completed_process
    from aka.legacy.application.experiment_executor import sandbox_executor

    with application_execution() as executor:
        bound = sandbox_executor(
            executor, hardware=hardware, profile=profile, url=url, ssh=ssh,
            ssh_init=ssh_init, health_command=health_command,
            private_reference_dir=private_reference_dir,
        )
        spec = CommandSpec(workspace, tuple(command), timeout=timeout, operation=gateway_kind)
        result = LegacyExperiment(bound).run_command(spec)
        if bound.completed_process is not None:
            return bound.completed_process
        return _completed_process(spec, result)
