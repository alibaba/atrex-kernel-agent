"""Compose the production Workflow from explicit module dependencies."""
from .assembly import LegacyWorkflow
from .dependencies import CampaignDependencies


def compose_workflow(*, agent=None, task=None, optimization_mode="leaderboard",
                     evaluator_factory=None, executor=None, experiment_factory=None,
                     execution_environment=None, measurement_parallelism=None,
                     canonical_measurement_reader=None, workflow_observer=None):
    from contextlib import ExitStack
    from .agent_host import create_agent
    from aka.legacy.task.problem.legacy import LegacyProblemProvider
    from aka.legacy.task.source.legacy import LegacySourceProvider
    from aka.legacy.task.candidate_workspace.provider import GitCandidateWorkspace
    from aka.contracts.execution import validate_executor

    owned = ExitStack()
    try:
        if executor is None:
            import os
            from aka.legacy.execution.assembly import compose_execution
            selected = compose_execution()
            owned.callback(selected.dispose)
            executor = selected.executor
            if execution_environment is None:
                execution_environment = dict(os.environ)
        validate_executor(executor)
        if measurement_parallelism is None:
            measurement_parallelism = 1 if (execution_environment or {}).get("ATREX_SANDBOX_SSH") else 4
        if evaluator_factory is None:
            from aka.legacy.evaluation.plugin import LegacyEvaluatorFactory
            evaluator_factory = LegacyEvaluatorFactory()
        if canonical_measurement_reader is None:
            from .records import WorkflowMeasurementReader
            canonical_measurement_reader = WorkflowMeasurementReader()
        if experiment_factory is None:
            from aka.legacy.experiment.factory import LegacyExperimentFactory
            experiment_factory = LegacyExperimentFactory(measurement_reader=canonical_measurement_reader)
            owned.callback(experiment_factory.close)
        if not callable(getattr(experiment_factory, "open", None)):
            raise TypeError("provider does not implement ExperimentFactory")
        if not callable(getattr(evaluator_factory, "create", None)):
            raise TypeError("provider does not implement EvaluatorFactory")
        if agent is None:
            agent = create_agent()
            owned.callback(agent.close)
        task = task if task is not None else dict(
            problem=LegacyProblemProvider(optimization_mode),
            source=LegacySourceProvider(), workspace=GitCandidateWorkspace(),
        )
        # Validate all task keys while owned module resources are still guarded.
        problem, source, candidate = task["problem"], task["source"], task["workspace"]
    except BaseException:
        owned.close()
        raise

    opened_services = []
    def services(campaign):
        from aka.legacy.workflow.services import WorkflowServices
        selected = WorkflowServices(campaign.request.runtime, campaign,
                                executor=executor, experiment_factory=experiment_factory,
                                evaluator_factory=evaluator_factory,
                                execution_environment=execution_environment,
                                measurement_parallelism=measurement_parallelism,
                                canonical_measurement_reader=canonical_measurement_reader,
                                workflow_observer=workflow_observer)
        opened_services.append(selected)
        return selected

    def execution(campaign):
        from aka.legacy.workflow.campaign_execution import CampaignExecution
        return CampaignExecution(campaign.services.settings, environment=execution_environment)

    try:
        dependencies = CampaignDependencies(
            agent, problem, source, candidate,
            lambda campaign: campaign.services.baseline(),
            run_campaign, execution,
            services=services,
        )
    except BaseException:
        owned.close()
        raise

    class AssembledWorkflow(LegacyWorkflow):
        def close(self):
            if self._closed:
                return
            super().close()
            try:
                for selected in reversed(opened_services):
                    selected.close()
            finally:
                owned.close()

    try:
        return AssembledWorkflow(dependencies)
    except BaseException:
        owned.close()
        raise


def run_campaign(campaign):
    from .episode import LongHorizonCampaign
    from .session import LongSessionRunner, budget_tokens
    from .store import CampaignStore
    from .evaluation import FormalVerification, EpisodeMeasurements
    from .environment import CampaignEnvironment

    services = campaign.services
    settings = services.settings
    CampaignStore.ensure_excluded(campaign.workspace)
    engine = LongHorizonCampaign(
        campaign,
        verifier=FormalVerification(services.formal_experiment, services.acceptance,
                                    settings.min_improvement_pct),
        session_runner=LongSessionRunner(
            agent=campaign.agent, agent_cli=campaign.agent_cli,
            usage_policy=lambda turn: budget_tokens(turn, campaign.agent_cli),
        ),
        evaluation=None, acceptance=services.acceptance,
        environment=CampaignEnvironment(campaign),
        max_version=campaign.max_iters, fast_episodes=campaign.fast_episodes,
        fast_trials=campaign.fast_trials, token_budget=campaign.token_budget,
        handoff_resumes=campaign.handoff_resumes, max_stall=campaign.max_stall,
        conversion_budget_action=campaign.request.settings.conversion_budget_action,
    )
    engine.evaluation = EpisodeMeasurements(
        campaign.workspace, campaign.private_reference_dir, settings.min_improvement_pct,
        experiment=services.baseline_experiment, acceptance=services.acceptance,
        records=services.records, command=services.sandbox_command,
    )
    return campaign._finish(engine.run())
