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
from aka.legacy.execution.selection import execution_scope
from aka.legacy.execution.binding import sandbox_executor


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
        from aka.legacy.execution.selection import _completed_process

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
    def __init__(self, settings, campaign, *, evaluator_factory=None,
                 workflow_observer=None):
        self.settings = settings
        self.campaign = campaign
        self.problem_provider = campaign.problem_provider
        if evaluator_factory is None:
            from aka.legacy.evaluation.plugin import LegacyEvaluatorFactory
            evaluator_factory = LegacyEvaluatorFactory()
        if not callable(getattr(evaluator_factory, "create", None)):
            raise TypeError("provider does not implement EvaluatorFactory")
        self.evaluator_factory = evaluator_factory
        if workflow_observer is not None:
            try:
                workflow_observer.campaign_started(campaign.request)
            except Exception as error:
                import sys
                print(f"[workflow] auxiliary observer failed: {type(error).__name__}", file=sys.stderr)
        self.acceptance = LegacyAcceptancePolicy(
            candidate_workspace=campaign.candidate_workspace,
            production_policy=campaign._production_kernel_violations,
        )
        self.evaluator = self.create_evaluator()
        self.records = records
        self.baseline_experiment = BorrowedExperiment(self)
        self.formal_experiment = BorrowedExperiment(self, formal=True)

    def evaluation_input(self):
        """Acquire validated frozen Task roles for every evaluation route."""
        return self.problem_provider.content(self.campaign.problem, self.campaign.workspace).evaluation

    def task_environment(self):
        from aka.legacy.execution.binding import task_environment
        return task_environment(self.evaluation_input().worker_requirements)

    def create_evaluator(self):
        s = self.settings
        from aka.legacy.execution.configuration import selected_ssh
        evaluator = self.evaluator_factory.create(
            EvaluationConfiguration(
                timeout=s.timeout, repeats=s.verify_repeats,
                per_run_timeout=s.verify_run_timeout,
                shape_batch_size=s.shape_batch_size,
                max_parallelism=1 if selected_ssh() else 4,
                private_reference_dir=self.campaign.private_reference_dir,
                native=bool(self.campaign.atrex_bench_root),
            ),
            candidate_workspace=self.campaign.candidate_workspace,
            evaluation_input=self.evaluation_input(),
        )
        if any(not callable(getattr(evaluator, method, None))
               for method in ("prepare", "record_step", "interpret")):
            raise TypeError("factory did not return an Evaluator")
        return evaluator

    def preparation(self, include_problem=False):
        from aka.legacy.resources.paths import PROFILE_DRIVER

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
            inputs, BaselineResources(self.evaluation_input().harness, PROFILE_DRIVER)
        )

    @contextmanager
    def experiment(self, *, formal=False):
        from aka.legacy.experiment.retry import retry_infrastructure

        s = self.settings
        with execution_scope() as selected:
            bound = sandbox_executor(
                selected,
                hardware=s.hardware,
                timeout=s.timeout, formal=formal,
                private_reference_dir=self.campaign.private_reference_dir,
                worker_requirements=self.evaluation_input().worker_requirements,
            )
            evaluator = self.create_evaluator()
            if not isinstance(evaluator, LegacyEvaluator):
                if formal:
                    from aka.legacy.experiment.runner import evaluation_stage
                    yield LegacyExperiment(bound, evaluator, retry=retry_infrastructure, stage=evaluation_stage)
                else:
                    yield LegacyExperiment(bound, evaluator)
            elif formal:
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
        argv = ["python", "tools/sandbox.py", "--kind", "run"]
        if self.settings.hardware:
            argv += ["--hardware", self.settings.hardware]
        return shlex.join([*argv, "--no-sync", "--", *command])

    def numerical_feedback(self, workspace, *, standard_correctness_passed=False):
        from aka.legacy.workflow.numerical_review import supplemental_feedback

        return supplemental_feedback(
            self.campaign,
            workspace,
            standard_correctness_passed=standard_correctness_passed,
            execute=self.numerical_command,
            harness=(self.evaluation_input().harness.source
                     if self.evaluation_input().harness is not None else None),
        )

    def numerical_command(self, command):
        from aka.legacy.execution.selection import _completed_process

        result = self.baseline_experiment.run_command(command)
        return _completed_process(command, result)

    def baseline(self):
        from aka.legacy.workflow.baseline import WorkflowBaseline

        return WorkflowBaseline(
            self.campaign,
            preparation=self.preparation,
            experiment=self.baseline_experiment,
            acceptance=self.acceptance,
            command=self.sandbox_command,
        )
