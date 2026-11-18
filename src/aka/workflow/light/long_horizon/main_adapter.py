"""Light orchestration uses public owners without a measured-V0 prelude."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from aka.workflow.light.initial_source import prepare_campaign

if TYPE_CHECKING:
    from aka.workflow.light.orchestration.campaign import Campaign


def link_episode_runtime(campaign: Campaign, workspace: Path) -> None:
    from aka.workflow.light.runtime_resources import install_runtime_resources
    from aka.workflow.light.orchestration.optimization_policy import install_workspace_policy

    install_runtime_resources(campaign, workspace)
    install_workspace_policy(
        workspace, campaign.optimization_mode, campaign.framework,
        update_tracked_files=False,
        source_checks_enabled=campaign.source_checks_enabled,
        specify_dsl=campaign.specify_dsl,
        expose_evaluator_shapes=campaign.expose_evaluator_shapes,
    )


def episode_directives(campaign: Campaign) -> dict[str, str]:
    return {"plugins": campaign.plugin_directive("episode")}


def session_execution_policy(campaign: Campaign) -> dict[str, object]:
    from aka.workflow.light.orchestration.constants import DEFAULT_OUTPUT_LIMIT_RECOVERY_RETRIES

    return {
        "reasoning_effort": "max",
        "output_limit_recovery_retries": getattr(
            campaign.module_options, "output_limit_recovery_retries",
            DEFAULT_OUTPUT_LIMIT_RECOVERY_RETRIES),
        "dependency_policy_recovery_retries": getattr(
            campaign.module_options, "dependency_policy_recovery_retries", 2),
    }
