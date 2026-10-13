"""Construct the legacy Workflow using borrowed Agent, Task and production evaluation."""


class LegacyWorkflow:
    def __init__(
        self,
        agent,
        *,
        evaluator_factory=None,
        baseline_evaluation=None,
        episode_evaluation=None
    ):
        self.agent = agent
        self.evaluator_factory = evaluator_factory
        self.baseline_evaluation = baseline_evaluation
        self.episode_evaluation = episode_evaluation
        self._closed = False

    def close(self):
        self._closed = True

    def _check_open(self):
        if self._closed:
            raise RuntimeError("Workflow selection has been closed")

    def campaign(self, **configuration):
        self._check_open()
        from .campaign import Campaign

        configuration["workflow"] = self
        return Campaign(**configuration)

    def run(self, campaign):
        self._check_open()
        from .episode import LongHorizonCampaign
        from .session import LongSessionRunner
        from .store import CampaignStore
        from long_horizon.verifier import GatewayABBAValidator

        CampaignStore.ensure_excluded(campaign.workspace)
        factory = self.evaluator_factory or GatewayABBAValidator
        verifier = factory(
            workspace_provider=campaign.candidate_workspace,
            hardware=campaign.sandbox_hardware,
            profile=campaign.sandbox_profile,
            url=campaign.sandbox_url,
            ssh=campaign.sandbox_ssh,
            ssh_init=campaign.sandbox_ssh_init,
            health_command=campaign.sandbox_health_command,
            timeout=campaign.sandbox_timeout,
            repeats=campaign.verify_repeats,
            per_run_timeout=campaign.verify_run_timeout,
            min_improvement_pct=campaign.min_improvement_pct,
            private_reference_dir=campaign.private_reference_dir,
        )
        engine = LongHorizonCampaign(
            base_campaign=campaign,
            max_version=campaign.max_iters,
            fast_episodes=campaign.fast_episodes,
            fast_trials=campaign.fast_trials,
            token_budget=campaign.token_budget,
            handoff_resumes=campaign.handoff_resumes,
            max_stall=campaign.max_stall,
            verifier=verifier,
            session_runner=LongSessionRunner(
                agent_cli=campaign.agent_cli, agent=self.agent
            ),
            evaluation=self.episode_evaluation,
        )
        return campaign._finish(engine.run())


def default_workflow():
    """Compatibility construction for direct Campaign users outside a selected host."""
    from aka.legacy.agent.selection import LegacyAgent

    return LegacyWorkflow(LegacyAgent())
