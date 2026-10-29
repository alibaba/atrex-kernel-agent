"""Assemble per-Campaign dependencies; business consumers borrow public contracts."""

from contextlib import contextmanager

from aka.evaluation import EvaluationConfiguration
from aka.evaluation.baseline import (
    BaselineInputs,
    BaselineResources,
    BaselinePreparation,
)
from aka.evaluation import records
from aka.contracts.execution import ExecutionContext, validate_executor
from aka.contracts.experiment import EpisodeEvidenceScope
from aka.execution.binding import SandboxExecutor


class BorrowedExperiment:
    """Borrow the selected Factory for operations in one canonical namespace."""

    def __init__(self, services, *, formal=False):
        self.services = services
        self.formal = formal

    def run(self, request):
        with self.services.experiment(formal=self.formal, workspace=request.workspace) as experiment:
            return experiment.run(request)

    def run_command(self, command):
        with self.services.experiment(formal=self.formal, workspace=command.workspace) as experiment:
            return experiment.run_command(command)

    def run_recorded(self, request, *, request_id):
        with self.services.experiment(formal=self.formal, workspace=request.workspace) as experiment:
            self._recorded(experiment, "run_recorded")
            return experiment.run_recorded(request, request_id=request_id)

    @staticmethod
    def _recorded(experiment, method):
        if not callable(getattr(experiment, method, None)):
            raise TypeError("selected Experiment does not implement " + method)

    def resume_recorded(self, request, *, request_id, recovery_timeout=None, max_remeasurements=0):
        with self.services.experiment(formal=self.formal, workspace=request.workspace) as experiment:
            self._recorded(experiment, "resume_recorded")
            return experiment.resume_recorded(request, request_id=request_id,
                recovery_timeout=recovery_timeout, max_remeasurements=max_remeasurements)

    def get(self, ref):
        with self.services.experiment(formal=self.formal) as experiment:
            self._recorded(experiment, "get")
            return experiment.get(ref)

    def find(self, request_id):
        with self.services.experiment(formal=self.formal) as experiment:
            self._recorded(experiment, "find")
            return experiment.find(request_id)

    def unresolved(self):
        with self.services.experiment(formal=self.formal) as experiment:
            self._recorded(experiment, "unresolved")
            return experiment.unresolved()

    def read_artifact(self, artifact):
        with self.services.experiment(formal=self.formal) as experiment:
            self._recorded(experiment, "read_artifact")
            return experiment.read_artifact(artifact)


class WorkflowServices:
    def __init__(self, settings, campaign, *, executor, experiment_factory,
                 evaluator_factory, execution_environment=None, measurement_parallelism=4,
                 canonical_measurement_reader, workflow_observer=None, rule_set=None,
                 experiment_episode_evidence=None, evaluation_episode_interpreter=None):
        from .rule_sets import DEFAULT_RULE_SET, select_rule_set
        self.rule_set = select_rule_set({"rule_set": DEFAULT_RULE_SET}) if rule_set is None else rule_set
        self.settings = settings
        self.campaign = campaign
        self.problem_provider = campaign.problem_provider
        validate_executor(executor)
        if not callable(getattr(experiment_factory, "open", None)):
            raise TypeError("provider does not implement ExperimentFactory")
        if not callable(getattr(evaluator_factory, "create", None)):
            raise TypeError("provider does not implement EvaluatorFactory")
        self.evaluator_factory = evaluator_factory
        if type(measurement_parallelism) is not int or measurement_parallelism < 1:
            raise ValueError("measurement_parallelism must be positive")
        self.executor = executor
        self.experiment_factory = experiment_factory
        self.experiment_episode_evidence = experiment_episode_evidence
        self.evaluation_episode_interpreter = evaluation_episode_interpreter
        self.execution_environment = dict(execution_environment or {})
        self.measurement_parallelism = measurement_parallelism
        if any(not callable(getattr(canonical_measurement_reader, method, None))
               for method in ("baseline", "measurements")):
            raise TypeError("provider does not implement CanonicalMeasurementReader")
        self.canonical_measurement_reader = canonical_measurement_reader
        self._closed = False
        from threading import Lock
        self._command_lock = Lock()
        self._command_host = None
        if workflow_observer is not None:
            try:
                workflow_observer.campaign_started(campaign.request)
            except Exception as error:
                import sys
                print(f"[workflow] auxiliary observer failed: {type(error).__name__}", file=sys.stderr)
        self.acceptance = self.rule_set.create_acceptance(
            candidate_workspace=campaign.candidate_workspace,
            production_policy=campaign._production_kernel_violations,
        )
        self._evaluation_input = self.problem_provider.content(
            self.campaign.problem, self.campaign.workspace
        ).evaluation
        self.records = records
        self.baseline_experiment = BorrowedExperiment(self)
        self.formal_experiment = BorrowedExperiment(self, formal=True)

    def evaluation_input(self):
        """Return the once-validated frozen Task roles for every evaluation route."""
        return self._evaluation_input

    def task_environment(self):
        from aka.execution.binding import task_environment
        return task_environment(self.evaluation_input().worker_requirements)

    def _measurement_environment(self):
        import os
        environment = dict(os.environ)
        for key, value in self.execution_environment.items():
            if value is None:
                environment.pop(key, None)
            else:
                environment[key] = value
        environment.update(self.task_environment())
        if self.campaign.private_reference_dir is not None:
            environment["ATREX_PRIVATE_REFERENCE_DIR"] = str(self.campaign.private_reference_dir)
        return environment

    def measurement_reader(self):
        return self.canonical_measurement_reader

    def create_evaluator(self, *, execution_capabilities=None):
        s = self.settings
        evaluator = self.evaluator_factory.create(
            EvaluationConfiguration(
                timeout=s.timeout, repeats=s.verify_repeats,
                per_run_timeout=s.verify_run_timeout,
                shape_batch_size=s.shape_batch_size,
                max_parallelism=self.measurement_parallelism,
                private_reference_dir=self.campaign.private_reference_dir,
                native=bool(self.campaign.atrex_bench_root),
                hardware=s.hardware, environment=self._measurement_environment(),
                measurement_reader=self.measurement_reader(),
                execution_capabilities=execution_capabilities,
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
            measurement_reader=self.measurement_reader(),
        )
        return BaselinePreparation(
            inputs, BaselineResources(self.evaluation_input().harness,
                c.plugin_registry.resource_result().resource("execution-profile-driver").source)
        )

    @contextmanager
    def experiment(self, *, formal=False, workspace=None):
        if self._closed:
            raise RuntimeError("Workflow services have been closed")
        s = self.settings
        bound = SandboxExecutor(self.executor, ExecutionContext(
            self._measurement_environment(), s.hardware, s.timeout, formal))
        evaluator = self.create_evaluator(execution_capabilities=bound.capabilities)
        with self.experiment_factory.open(workspace=self.campaign.workspace,
                purpose="formal" if formal else "development", executor=bound, evaluator=evaluator,
                evaluation_input=self.evaluation_input(),
                episode_evidence=self._episode_evidence(workspace)) as experiment:
            if any(not callable(getattr(experiment, method, None)) for method in ("run", "run_command")):
                raise TypeError("factory did not return an Experiment")
            yield experiment

    def _episode_evidence(self, workspace):
        if workspace is None:
            return None
        from pathlib import Path
        from .store import RUNTIME_DIR
        actual = Path(workspace)
        if not (actual / RUNTIME_DIR / "journal.json").is_file():
            return None
        return EpisodeEvidenceScope(actual, self.campaign.candidate_workspace)

    def close(self):
        with self._command_lock:
            if self._closed:
                return
            self._closed = True
            host = self._command_host
        if host is not None:
            host.close()

    @contextmanager
    def agent_experiment(self, workspace, access):
        """Borrow the selected services with the current Agent's exact scope."""
        if self._closed:
            raise RuntimeError("Workflow services have been closed")
        bound = SandboxExecutor(self.executor, ExecutionContext(
            {}, self.settings.hardware, self.settings.timeout, False, access))
        with self.experiment_factory.open(workspace=self.campaign.workspace,
                purpose="development", executor=bound,
                evaluator=self.create_evaluator(execution_capabilities=bound.capabilities),
                evaluation_input=self.evaluation_input(),
                episode_evidence=self._episode_evidence(workspace)) as experiment:
            if any(not callable(getattr(experiment, method, None)) for method in ("run", "run_command")):
                raise TypeError("factory did not return an Experiment")
            yield experiment

    @contextmanager
    def command_session(self, workspace, access):
        """Register command authorization for one Agent session and revoke it on return."""
        with self._command_lock:
            if self._closed:
                raise RuntimeError("Workflow services have been closed")
            if self._command_host is None:
                from aka.experiment.command_host import ExperimentCommandHost
                self._command_host = ExperimentCommandHost(self.agent_experiment,
                    self.executor.capabilities, hardware=self.settings.hardware)
            host = self._command_host
        with host.session(workspace, access) as registration:
            yield registration

    def sandbox_command(self, command):
        import shlex
        import sys
        argv = [sys.executable, "-m", "aka.experiment.command_client", "--kind", "run"]
        if self.settings.hardware:
            argv += ["--hardware", self.settings.hardware]
        return shlex.join([*argv, "--no-sync", "--", *command])

    def numerical_feedback(self, workspace, *, standard_correctness_passed=False):
        from aka.legacy.workflow.numerical_review import supplemental_feedback

        return supplemental_feedback(
            self.campaign,
            workspace,
            standard_correctness_passed=standard_correctness_passed,
            experiment=self.baseline_experiment,
            harness=(self.evaluation_input().harness.source
                     if self.evaluation_input().harness is not None else None),
        )

    def baseline(self):
        from aka.legacy.workflow.baseline import WorkflowBaseline

        return WorkflowBaseline(
            self.campaign,
            preparation=self.preparation,
            experiment=self.baseline_experiment,
            acceptance=self.acceptance,
            command=self.sandbox_command,
        )
