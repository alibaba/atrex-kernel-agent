from __future__ import annotations

from pathlib import Path

from orchestrator import constants as _constants
from orchestrator import workspace_state as _workspace_state
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aka.legacy.workflow.campaign import Campaign
from orchestrator.constants import DEFAULT_CONVERT_AFTER
from orchestrator.hardware import (
    hardware_directive,
    hardware_vendor,
    head_kernel_is_gluon,
    should_convert_to_gluon,
)
from orchestrator.optimization_policy import install_workspace_policy
from orchestrator.session_io import _sandbox_command
from orchestrator.workspace_runtime import (
    _agent_runtime_directive,
    _plan_generator_directive,
    link_runtime,
)
from orchestrator.workspace_state import (
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
        campaign._link_runtime()
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
    native = Path(campaign.atrex_bench_root) if campaign.atrex_bench_root else None
    link_runtime(
        workspace,
        native,
        plugin_registry=campaign.plugin_registry,
        is_ppu=hardware_vendor(campaign.platform, campaign.arch) == "ppu",
    )
    install_workspace_policy(
        workspace,
        campaign.optimization_mode,
        campaign.framework,
        update_tracked_files=False,
        source_checks_enabled=campaign.source_checks_enabled,
    )


def episode_directives(
    campaign: Campaign, version: int, *, fast: bool = False
) -> dict[str, str]:
    agent_cli = getattr(campaign, "agent_cli", "claude")
    return {
        "plugins": campaign.plugin_directive("fast_episode" if fast else "episode"),
        "hardware": hardware_directive(campaign.platform, campaign.arch),
        "sandbox": (
            campaign._fast_sandbox_directive()
            if fast
            else campaign._sandbox_directive()
        ),
        "evaluator": campaign.evaluation.evaluator_directive(),
        "mode_policy": campaign._mode_directive(),
        "agent_runtime": _agent_runtime_directive(
            agent_cli, is_ppu=hardware_vendor(campaign.platform, campaign.arch) == "ppu"
        ),
        "plan_generator": _plan_generator_directive(agent_cli, version),
    }


def run_sandbox(
    workspace: Path,
    hardware: str,
    profile: str,
    url: str,
    timeout: int,
    command: list[str],
    *,
    ssh: str = "",
    ssh_init: str = "",
    health_command: str = "",
    sync: tuple[str, ...] = (),
    wall_timeout: int | None = None,
    gateway_kind: str = "auto",
    private_reference_dir: Path | None = None,
):
    """Use main's sandbox command builder and queue/timeout semantics verbatim."""
    return _sandbox_command(
        workspace,
        hardware,
        profile,
        url,
        timeout,
        command,
        ssh=ssh,
        ssh_init=ssh_init,
        health_command=health_command,
        sync=sync,
        wall_timeout=wall_timeout,
        gateway_kind=gateway_kind,
        private_reference_dir=private_reference_dir,
    )


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



class CampaignEnvironment:
    """Bind Workflow-owned environment operations to one campaign."""
    CONVERT_PERF_TOL = CONVERT_PERF_TOL
    candidate_is_gluon = staticmethod(candidate_is_gluon)
    restored_stall = staticmethod(restored_stall)
    save_stall = staticmethod(save_stall)
    read_memory = staticmethod(read_memory)
    latest_version = staticmethod(latest_version)
    speedup_vs_reference = staticmethod(speedup_vs_reference)
    peak_util = staticmethod(peak_util)

    def __init__(self, campaign):
        self.campaign = campaign

    def prepare_campaign(self, campaign):
        return prepare_campaign(self.campaign)

    def link_episode_runtime(self, campaign, workspace):
        return link_episode_runtime(self.campaign, workspace)

    def episode_directives(self, campaign, version, *, fast=False):
        return episode_directives(self.campaign, version, fast=fast)

    def candidate_policy_violations(self, campaign, workspace, *, require_gluon=False):
        return candidate_policy_violations(self.campaign, workspace, require_gluon=require_gluon)

    def conversion_required(self, campaign, stall, workspace):
        return conversion_required(self.campaign, stall, workspace)
