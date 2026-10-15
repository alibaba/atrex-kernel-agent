"""Assemble the retained Task/evaluation implementation outside Workflow."""

from .workflow import bind_evaluation


def default_workflow(
    *,
    agent=None,
    task=None,
    optimization_mode="leaderboard",
    evaluator_factory=None,
    baseline_evaluation=None,
    episode_evaluation=None,
    acceptance_policy=None,
):
    from aka.legacy.workflow.composition import compose_workflow
    from .startup import prepare_request

    return compose_workflow(
        agent=agent, task=task, optimization_mode=optimization_mode,
        prepare_request=prepare_request,
        baseline_factory=lambda campaign: baseline_adapter(campaign, baseline_evaluation),
        campaign_runner=lambda campaign: run_campaign(
            campaign, evaluator_factory=evaluator_factory,
            episode_evaluation=episode_evaluation, acceptance_policy=acceptance_policy),
    )


def baseline_adapter(campaign, implementation=None):
    if implementation is None:
        return campaign.services.baseline()
    return bind_evaluation(implementation, campaign)


def execution_adapter(context):
    from .execution import CampaignExecution

    return CampaignExecution(context)


def run_campaign(
    campaign, *, evaluator_factory=None, episode_evaluation=None, acceptance_policy=None
):
    from aka.legacy.workflow.episode import LongHorizonCampaign
    from aka.legacy.workflow.session import LongSessionRunner
    from aka.legacy.workflow.store import CampaignStore
    from .session import legacy_budget_tokens

    from .evaluation import episode_adapter

    from aka.legacy.workflow.evaluation import FormalVerification, EpisodeMeasurements
    services = campaign.services
    context = services.settings
    CampaignStore.ensure_excluded(campaign.workspace)
    policy = acceptance_policy if acceptance_policy is not None else services.acceptance
    if evaluator_factory is not None:
        verifier = evaluator_factory(
            workspace_provider=campaign.candidate_workspace, hardware=context.hardware,
            profile=context.profile, url=context.url, ssh=context.ssh, ssh_init=context.ssh_init,
            health_command=context.health_command, timeout=context.timeout,
            repeats=context.verify_repeats, per_run_timeout=context.verify_run_timeout,
            min_improvement_pct=context.min_improvement_pct,
            private_reference_dir=campaign.private_reference_dir,
        )
    else:
        verifier = FormalVerification(services.formal_experiment, policy, context.min_improvement_pct)
    engine = LongHorizonCampaign(
        campaign,
        verifier=verifier,
        session_runner=LongSessionRunner(
            agent=campaign.agent,
            agent_cli=campaign.agent_cli,
            usage_policy=lambda turn: legacy_budget_tokens(turn, campaign.agent_cli),
        ),
        evaluation=None,
        acceptance=policy,
        environment=CampaignEnvironment(campaign),
        max_version=campaign.max_iters,
        fast_episodes=campaign.fast_episodes,
        fast_trials=campaign.fast_trials,
        token_budget=campaign.token_budget,
        handoff_resumes=campaign.handoff_resumes,
        max_stall=campaign.max_stall,
    )
    if episode_evaluation is not None:
        engine.evaluation = episode_adapter(episode_evaluation, engine,
            settings=context, base=campaign)
    else:
        engine.evaluation = EpisodeMeasurements(campaign.workspace, campaign.private_reference_dir,
            context.min_improvement_pct, evaluator=services.evaluator, acceptance=policy,
            records=services.records, command=services.sandbox_command)
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
