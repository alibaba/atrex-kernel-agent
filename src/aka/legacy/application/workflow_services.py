"""Assemble per-Campaign dependencies; business consumers borrow public contracts."""

from contextlib import contextmanager

from aka.legacy.evaluation import LegacyEvaluator, EvaluationConfiguration
from aka.legacy.evaluation.baseline import (
    BaselineInputs,
    BaselineResources,
    BaselinePreparation,
)
from aka.legacy.evaluation import records
from aka.legacy.experiment import LegacyExperiment
from aka.legacy.experiment.runner import abba_stage
from aka.legacy.experiment.transport import ABBAExecutor, check_abba_result
from aka.legacy.acceptance import LegacyAcceptancePolicy
from .execution import application_execution
from .experiment_executor import sandbox_executor


class BorrowedExperiment:
    """Enter the selected execution scope at the host boundary for each operation."""

    def __init__(self, services, *, formal=False):
        self.services = services
        self.formal = formal

    def run(self, request):
        with self.services.experiment(formal=self.formal) as experiment:
            return experiment.run(request)

    def run_command(self, command):
        with self.services.experiment() as experiment:
            return experiment.run_command(command)


class BaselineEvaluator:
    """Print the retained transport output before interpreting baseline results."""

    def __init__(self, evaluator, bound):
        self.evaluator = evaluator
        self.bound = bound

    def prepare(self, request):
        return self.evaluator.prepare(request)

    def record_step(self, prepared, index, result):
        # V0/framework historically print immediately after their single command.
        # This hook avoids making the consumer inspect executor-private state.
        process = self.bound.completed_process
        if process is None:
            from subprocess import CompletedProcess

            process = CompletedProcess(
                (), result.transport_returncode, result.stdout, result.stderr
            )
        BaselinePreparation.print_v0_evaluator_output(process)
        return self.evaluator.record_step(prepared, index, result)

    def interpret(self, prepared, results):
        from dataclasses import replace
        from .execution import _completed_process

        # Preserve the old baseline's command-code projection while the original
        # JobResult returned by the executor remains intact.
        projected = tuple(
            replace(
                result,
                transport_returncode=_completed_process(command, result).returncode,
            )
            for command, result in zip(prepared.steps, results)
        )
        return self.evaluator.interpret(prepared, projected)


class BaselineExperiment(LegacyExperiment):
    """Retain the baseline's synchronous single-attempt and exception boundary."""

    def run(self, request):
        prepared = self.evaluator.prepare(request)
        result = self.run_command(prepared.steps[0])
        self.evaluator.record_step(prepared, 0, result)
        try:
            return self.evaluator.interpret(prepared, (result,))
        except (RuntimeError, ValueError, KeyError, TypeError) as error:
            from aka.contracts.experiment import ExperimentExecutionError

            raise ExperimentExecutionError(str(error)) from error


class WorkflowServices:
    def __init__(self, settings, campaign):
        if sum(bool(v) for v in (settings.ssh, settings.url, settings.profile)) > 1:
            raise ValueError(
                "sandbox_ssh, sandbox_url, and sandbox_profile are mutually exclusive"
            )
        self.settings = settings
        self.campaign = campaign
        self.acceptance = LegacyAcceptancePolicy(
            candidate_workspace=campaign.candidate_workspace,
            production_policy=campaign._production_kernel_violations,
        )
        self.evaluator = LegacyEvaluator(EvaluationConfiguration(shape_batch_size=settings.shape_batch_size))
        self.records = records
        self.baseline_experiment = BorrowedExperiment(self)
        self.formal_experiment = BorrowedExperiment(self, formal=True)

    def preparation(self, include_problem=False):
        from orchestrator.constants import ATREX_BENCH_HARNESS, PROFILE_DRIVER

        c = self.campaign
        native = bool(c.atrex_bench_root)
        inputs = BaselineInputs(
            c.workspace,
            native=native,
            problem=c.problem if include_problem and not native else None,
            private_reference_dir=c.private_reference_dir,
            arch=c.arch,
            campaign_name=c.campaign_name,
            framework=c.framework,
            notes=c.notes,
            optimization_mode=c.optimization_mode,
            platform=c.platform,
            target_util=c.target_util,
        )
        return BaselinePreparation(
            inputs, BaselineResources(ATREX_BENCH_HARNESS, PROFILE_DRIVER)
        )

    @contextmanager
    def experiment(self, *, formal=False):
        from orchestrator.infrastructure_retry import retry_infrastructure

        s = self.settings
        with application_execution() as selected:
            bound = sandbox_executor(
                selected,
                hardware=s.hardware,
                profile=s.profile,
                url=s.url,
                ssh=s.ssh,
                ssh_init=s.ssh_init,
                health_command=s.health_command,
                wall_timeout=s.timeout + 14_400 + 120 if formal else None,
                private_reference_dir=self.campaign.private_reference_dir,
            )
            evaluator = LegacyEvaluator(
                EvaluationConfiguration(
                    timeout=s.timeout,
                    repeats=s.verify_repeats,
                    per_run_timeout=s.verify_run_timeout,
                    shape_batch_size=s.shape_batch_size,
                    max_parallelism=1 if s.ssh else 4,
                    private_reference_dir=self.campaign.private_reference_dir,
                    native=bool(self.campaign.atrex_bench_root),
                ),
                candidate_workspace=self.campaign.candidate_workspace,
            )
            if formal:
                yield LegacyExperiment(
                    ABBAExecutor(bound),
                    evaluator,
                    retry=retry_infrastructure,
                    check_result=check_abba_result,
                    stage=abba_stage(s.verify_repeats, s.shape_batch_size),
                )
            else:
                yield BaselineExperiment(bound, BaselineEvaluator(evaluator, bound))

    def sandbox_command(self, command):
        import shlex

        s = self.settings
        argv = ["python", "tools/sandbox.py", "--kind", "run"]
        if s.hardware:
            argv += ["--hardware", s.hardware]
        if s.ssh:
            argv += ["--ssh", s.ssh]
        elif s.url:
            argv += ["--url", s.url]
        elif s.profile:
            argv += ["--gateway-profile", s.profile]
        return shlex.join([*argv, "--no-sync", "--", *command])

    def numerical_feedback(self, workspace, *, standard_correctness_passed=False):
        from orchestrator.numerical_policy import supplemental_feedback

        return supplemental_feedback(
            self.campaign,
            workspace,
            standard_correctness_passed=standard_correctness_passed,
            execute=self.numerical_command,
        )

    def numerical_command(self, command):
        from .execution import _completed_process

        result = self.baseline_experiment.run_command(command)
        return _completed_process(command, result)

    def baseline(self):
        from aka.legacy.workflow.baseline import WorkflowBaseline
        from orchestrator.constants import SOL_SEED

        return WorkflowBaseline(
            self.campaign,
            preparation=self.preparation,
            experiment=self.baseline_experiment,
            acceptance=self.acceptance,
            command=self.sandbox_command,
            sol_seed=SOL_SEED,
        )
