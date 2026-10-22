"""Assemble per-Campaign dependencies; business consumers borrow public contracts."""

from contextlib import contextmanager

from aka.legacy.evaluation import EvaluationConfiguration
from aka.legacy.evaluation.baseline import (
    BaselineInputs,
    BaselineResources,
    BaselinePreparation,
)
from aka.legacy.evaluation import records
from aka.legacy.experiment import LegacyExperiment, FileExperimentRecords
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

    def run_recorded(self, request, *, request_id):
        with self.services.experiment(formal=self.formal) as experiment:
            return experiment.run_recorded(request, request_id=request_id)

    def get(self, ref):
        return self.services.experiment_records.get(ref)

    def find(self, request_id):
        return self.services.experiment_records.find(request_id)

    def unresolved(self):
        return self.services.experiment_records.unresolved()

    def read_artifact(self, artifact):
        return self.services.experiment_records.read_artifact(artifact)


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
        self._evaluation_input = self.problem_provider.content(
            self.campaign.problem, self.campaign.workspace
        ).evaluation
        self.evaluator = self.create_evaluator()
        # A sibling root survives candidate workspace reset/clean operations.
        from pathlib import Path
        workspace = Path(campaign.workspace).resolve()
        self.experiment_records = FileExperimentRecords(
            workspace.parent / ".aka-experiments", str(workspace))
        self.records = records
        self.baseline_experiment = BorrowedExperiment(self)
        self.formal_experiment = BorrowedExperiment(self, formal=True)

    def evaluation_input(self):
        """Return the once-validated frozen Task roles for every evaluation route."""
        return self._evaluation_input

    def task_environment(self):
        from aka.legacy.execution.binding import task_environment
        return task_environment(self.evaluation_input().worker_requirements)

    def _measurement_environment(self):
        import os
        environment = {**os.environ, **self.task_environment()}
        if self.campaign.private_reference_dir is not None:
            environment["ATREX_PRIVATE_REFERENCE_DIR"] = str(self.campaign.private_reference_dir)
        return environment

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
                hardware=s.hardware, environment=self._measurement_environment(),
            ),
            candidate_workspace=self.campaign.candidate_workspace,
            evaluation_input=self.evaluation_input(),
        )
        if any(not callable(getattr(evaluator, method, None))
               for method in ("prepare", "observe_step", "interpret")):
            raise TypeError("factory did not return an Evaluator")
        return evaluator

    def preparation(self, include_problem=False):
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
            inputs, BaselineResources(self.evaluation_input().harness,
                c.plugin_registry.resource_result().resource("execution-profile-driver").source)
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
            from aka.legacy.experiment.runner import evaluation_stage
            yield LegacyExperiment(bound, evaluator,
                retry=retry_infrastructure if formal else None, stage=evaluation_stage,
                records=self.experiment_records, purpose="formal" if formal else "development")

    def sandbox_command(self, command):
        import shlex
        import sys
        argv = [sys.executable, "-m", "aka.legacy.experiment.cli", "--kind", "run"]
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
        return self.baseline_experiment.run_command(command)

    def baseline(self):
        from aka.legacy.workflow.baseline import WorkflowBaseline

        return WorkflowBaseline(
            self.campaign,
            preparation=self.preparation,
            experiment=self.baseline_experiment,
            acceptance=self.acceptance,
            command=self.sandbox_command,
        )
