"""Original episode construction translated to explicitly borrowed dependencies."""

from aka.legacy.workflow.episode import *
from aka.legacy.workflow.episode import LongHorizonCampaign as EpisodeCampaign
from . import main_adapter
from .session import LongSessionRunner
from .workflow import bind_evaluation
from .assembly import CampaignEnvironment


class LongHorizonCampaign(EpisodeCampaign):
    def __init__(
        self,
        base_campaign,
        *,
        verifier=None,
        session_runner=None,
        evaluation=None,
        acceptance_policy=None,
        **options
    ):
        from aka.legacy.compat.verifier import GatewayABBAValidator
        from aka.legacy.compat.legacy_evaluation.episode import LegacyEpisodeEvaluation

        if not hasattr(base_campaign, "evaluation") and hasattr(
            base_campaign, "_supplemental_numerical_feedback"
        ):
            from types import SimpleNamespace

            base_campaign.evaluation = SimpleNamespace(
                supplemental_numerical_feedback=base_campaign._supplemental_numerical_feedback
            )
        base_context = getattr(
            getattr(base_campaign, "evaluation", None), "context", base_campaign
        )
        if verifier is None:
            verifier = GatewayABBAValidator(
                workspace_provider=getattr(base_campaign, "candidate_workspace", None),
                hardware=getattr(base_context, "sandbox_hardware", ""),
                profile=getattr(base_context, "sandbox_profile", ""),
                url=getattr(base_context, "sandbox_url", ""),
                ssh=getattr(base_context, "sandbox_ssh", ""),
                ssh_init=getattr(base_context, "sandbox_ssh_init", ""),
                health_command=getattr(base_context, "sandbox_health_command", ""),
                timeout=getattr(base_context, "sandbox_timeout", 600),
                private_reference_dir=base_campaign.private_reference_dir,
            )
        if session_runner is None:
            session_runner = LongSessionRunner(
                agent_cli=getattr(base_campaign, "agent_cli", "claude"),
                agent=getattr(base_campaign, "agent", None),
            )
        from aka.legacy.acceptance import LegacyAcceptancePolicy
        from .evaluation import episode_adapter

        options["acceptance"] = (
            options.get("acceptance") or acceptance_policy or LegacyAcceptancePolicy()
        )
        super().__init__(
            base_campaign,
            verifier,
            session_runner,
            None,
            CampaignEnvironment(base_context),
            **options
        )
        self.evaluation = episode_adapter(
            evaluation,
            self,
            settings=getattr(base_context, "settings", None),
            base=base_context,
        )

    def _expected_shape_ids(self) -> set[str] | None:
        return self.evaluation.expected_shape_ids()

    def _fast_evaluator_command(self, version: int) -> str:
        return self.evaluation.fast_evaluator_command(version)

    def _fast_verification_result(
        self, episode_workspace: Path, *, memory_version: int
    ) -> VerificationResult:
        return self.evaluation.fast_verification_result(
            episode_workspace, memory_version=memory_version
        )


def __getattr__(name):
    from aka.legacy.workflow import episode

    return getattr(episode, name)
