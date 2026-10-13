"""Assemble the retained Task/evaluation implementation outside Workflow."""

from .workflow import bind_evaluation


def default_workflow(
    *,
    agent=None,
    task=None,
    optimization_mode="leaderboard",
    evaluator_factory=None,
    baseline_evaluation=None,
    episode_evaluation=None
):
    from aka.legacy.agent.selection import LegacyAgent
    from aka.legacy.task.problem.legacy import LegacyProblemProvider
    from aka.legacy.task.source.legacy import LegacySourceProvider
    from aka.legacy.task.candidate_workspace.provider import GitCandidateWorkspace
    from aka.legacy.workflow.dependencies import CampaignDependencies
    from aka.legacy.workflow.assembly import LegacyWorkflow

    owned_agent = agent is None
    agent = agent if agent is not None else LegacyAgent()
    task = (
        task
        if task is not None
        else dict(
            problem=LegacyProblemProvider(optimization_mode),
            source=LegacySourceProvider(),
            workspace=GitCandidateWorkspace(),
        )
    )
    dependencies = CampaignDependencies(
        agent,
        task["problem"],
        task["source"],
        task["workspace"],
        lambda c: baseline_adapter(c, baseline_evaluation),
        lambda c: run_campaign(
            c,
            evaluator_factory=evaluator_factory,
            episode_evaluation=episode_evaluation,
        ),
        lambda c: execution_adapter(c.evaluation.context),
    )

    class AssembledWorkflow(LegacyWorkflow):
        def close(self):
            super().close()
            if owned_agent:
                agent.close()

    return AssembledWorkflow(dependencies)


def baseline_adapter(campaign, implementation=None):
    from long_horizon.legacy_evaluation.baseline import LegacyBaselineEvaluation

    return bind_evaluation(implementation or LegacyBaselineEvaluation(), campaign)


def execution_adapter(context):
    from .execution import CampaignExecution

    return CampaignExecution(context)


def run_campaign(campaign, *, evaluator_factory=None, episode_evaluation=None):
    from aka.legacy.workflow.episode import LongHorizonCampaign
    from aka.legacy.workflow.session import LongSessionRunner
    from aka.legacy.workflow.store import CampaignStore
    from long_horizon.verifier import GatewayABBAValidator
    from long_horizon.legacy_evaluation.episode import LegacyEpisodeEvaluation
    from .session import legacy_budget_tokens

    context = campaign.evaluation.context
    CampaignStore.ensure_excluded(campaign.workspace)
    factory = evaluator_factory or GatewayABBAValidator
    verifier = factory(
        workspace_provider=campaign.candidate_workspace,
        hardware=context.sandbox_hardware,
        profile=context.sandbox_profile,
        url=context.sandbox_url,
        ssh=context.sandbox_ssh,
        ssh_init=context.sandbox_ssh_init,
        health_command=context.sandbox_health_command,
        timeout=context.sandbox_timeout,
        repeats=context.verify_repeats,
        per_run_timeout=context.verify_run_timeout,
        min_improvement_pct=context.min_improvement_pct,
        private_reference_dir=campaign.private_reference_dir,
    )
    engine = LongHorizonCampaign(
        campaign,
        verifier=verifier,
        session_runner=LongSessionRunner(
            agent=campaign.agent,
            agent_cli=campaign.agent_cli,
            usage_policy=lambda turn: legacy_budget_tokens(turn, campaign.agent_cli),
        ),
        evaluation=None,
        environment=CampaignEnvironment(context),
        max_version=campaign.max_iters,
        fast_episodes=campaign.fast_episodes,
        fast_trials=campaign.fast_trials,
        token_budget=campaign.token_budget,
        handoff_resumes=campaign.handoff_resumes,
        max_stall=campaign.max_stall,
    )
    engine.evaluation = bind_evaluation(
        episode_evaluation or LegacyEpisodeEvaluation(),
        engine,
        settings=context.settings,
        base=context,
    )
    return campaign._finish(engine.run())


class CampaignEnvironment:
    def __init__(self, context):
        self.context = context

    def __getattr__(self, name):
        from . import main_adapter

        operation = getattr(main_adapter, name)
        if name in {
            "prepare_campaign",
            "link_episode_runtime",
            "episode_directives",
            "candidate_policy_violations",
            "conversion_required",
        }:
            return lambda campaign, *args, **kwargs: operation(
                self.context, *args, **kwargs
            )
        return operation
