"""Compose the production Workflow from explicit module dependencies."""
from .assembly import LegacyWorkflow
from .dependencies import CampaignDependencies


def compose_workflow(*, agent=None, task=None, optimization_mode="leaderboard",
                     prepare_request=lambda request, problem: request,
                     evaluator_factory=None,
                     workflow_observer=None):
    from aka.legacy.agent.selection import LegacyAgent
    from aka.legacy.task.problem.legacy import LegacyProblemProvider
    from aka.legacy.task.source.legacy import LegacySourceProvider
    from aka.legacy.task.candidate_workspace.provider import GitCandidateWorkspace

    owned_agent = agent is None
    agent = agent if agent is not None else LegacyAgent()
    task = task if task is not None else dict(
        problem=LegacyProblemProvider(optimization_mode),
        source=LegacySourceProvider(), workspace=GitCandidateWorkspace(),
    )

    def services(campaign):
        from aka.legacy.workflow.services import WorkflowServices
        return WorkflowServices(campaign.request.runtime, campaign,
                                evaluator_factory=evaluator_factory,
                                workflow_observer=workflow_observer)

    def execution(campaign):
        from aka.legacy.execution.campaign_runtime import CampaignExecution
        return CampaignExecution(campaign.services.settings)

    dependencies = CampaignDependencies(
        agent, task["problem"], task["source"], task["workspace"],
        lambda campaign: campaign.services.baseline(),
        run_campaign, execution,
        prepare_request=prepare_request, services=services,
    )

    class AssembledWorkflow(LegacyWorkflow):
        def close(self):
            if self._closed:
                return
            super().close()
            if owned_agent:
                agent.close()

    return AssembledWorkflow(dependencies)


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
        evaluator=services.evaluator, acceptance=services.acceptance,
        records=services.records, command=services.sandbox_command,
    )
    return campaign._finish(engine.run())
