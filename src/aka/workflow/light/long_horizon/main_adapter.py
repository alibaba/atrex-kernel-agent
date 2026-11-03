from __future__ import annotations

from pathlib import Path

from aka.workflow.light.orchestration import constants as _constants
from aka.workflow.light.orchestration import workspace_state as _workspace_state
from aka.workflow.light.orchestration.campaign import Campaign
from aka.workflow.light.orchestration.constants import DEFAULT_CONVERT_AFTER
from aka.workflow.light.orchestration.hardware import (
    hardware_directive,
    hardware_vendor,
    head_kernel_is_gluon,
    should_convert_to_gluon,
)
from aka.workflow.light.orchestration.optimization_policy import install_workspace_policy
from aka.workflow.light.runtime_resources import (
    agent_runtime_directive as _agent_runtime_directive,
    install_runtime_resources,
)
from aka.workflow.light.orchestration.workspace_state import (
    git_head,
    head_kernel_is_initial_baseline,
    latest_version,
    read_stall,
    reconstruct_stall,
    write_stall,
)


def prepare_campaign(campaign: Campaign) -> None:
    """Run current main's complete setup/resume prelude, excluding only its loop."""
    if latest_version(campaign.workspace) < 0:
        campaign.setup_baseline()
    else:
        if not git_head(campaign.workspace):
            raise RuntimeError("existing campaign workspace has no Git HEAD")
        print(
            f"[orchestrator] resuming: latest = v{latest_version(campaign.workspace)}",
            flush=True,
        )
        campaign._link_runtime()  # Compatibility seam intentionally isolated in this module.
    campaign.ensure_framework_baseline()
    if (
        campaign.optimization_mode == "production"
        and latest_version(campaign.workspace) > 0
    ):
        violations = campaign._production_kernel_violations(
            require_gluon=head_kernel_is_gluon(campaign.workspace)
        )
        if violations and not head_kernel_is_initial_baseline(campaign.workspace):
            raise RuntimeError(
                "cannot resume a non-compliant production HEAD: "
                + "; ".join(violations)
            )
        if violations:
            print(
                "[orchestrator] production resume: HEAD kernel is still the original "
                "V0 baseline; continuing until a framework-compliant candidate is accepted",
                flush=True,
            )


def link_episode_runtime(campaign: Campaign, workspace: Path) -> None:
    install_runtime_resources(campaign, workspace)
    install_workspace_policy(
        workspace, campaign.optimization_mode, campaign.framework,
        update_tracked_files=False,
        source_checks_enabled=campaign.source_checks_enabled,
    )


def episode_directives(campaign: Campaign, version: int) -> dict[str, str]:
    """Public operator guidance for the single optimization workflow."""
    return {
        "plugins": campaign.plugin_directive("episode"),
        "hardware": hardware_directive(campaign.platform, campaign.arch),
        "sandbox": campaign._sandbox_directive(),
        "evaluator": campaign._evaluator_directive(),
        "mode_policy": campaign._mode_directive(),
        "agent_runtime": _agent_runtime_directive(
            campaign.agent_cli, is_ppu=hardware_vendor(campaign.platform, campaign.arch) == "ppu",
            wiki=campaign.tool_runtime.wiki,
            pause_agent_during_requests=campaign.tool_runtime.pause_agent_during_requests,
            knowledge_skills=campaign.plugin_registry.knowledge_skills() if campaign.tool_runtime.wiki else (),
        ),
    }


def candidate_policy_violations(
    campaign: Campaign,
    workspace: Path,
    *,
    require_gluon: bool = False,
) -> list[str]:
    if campaign.optimization_mode != "production":
        return []
    return campaign._production_kernel_violations(
        workspace,
        require_gluon=require_gluon,
    )


def conversion_required(campaign: Campaign, stall: int, workspace: Path) -> bool:
    if not campaign.source_checks_enabled:
        return False
    return should_convert_to_gluon(
        campaign.framework,
        stall,
        int(getattr(campaign, "convert_after", DEFAULT_CONVERT_AFTER)),
        head_is_gluon=head_kernel_is_gluon(workspace),
    )


def candidate_is_gluon(workspace: Path) -> bool:
    return head_kernel_is_gluon(workspace)


def restored_stall(workspace: Path) -> int:
    value = read_stall(workspace)
    return int(value if value is not None else reconstruct_stall(workspace))


def save_stall(workspace: Path, value: int) -> None:
    write_stall(workspace, value)


# Re-exported for the rest of long_horizon, which reaches orchestrator only through this adapter.
peak_util = _workspace_state.peak_util
read_memory = _workspace_state.read_memory
speedup_vs_reference = _workspace_state.speedup_vs_reference
CONVERT_PERF_TOL = _constants.CONVERT_PERF_TOL
IMMUTABLE_BASELINE_PATHS = _constants.IMMUTABLE_BASELINE_PATHS
STALL_STATE_FILE = _constants.STALL_STATE_FILE
