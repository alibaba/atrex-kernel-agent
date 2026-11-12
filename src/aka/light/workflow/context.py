from __future__ import annotations

from pathlib import Path

import aka.light.workflow.constants as _constants
import aka.light.workspace.state as _workspace_state
import aka.light.agent.runtime.process as _agent_process
from aka.light.workflow.bootstrap import Campaign
from aka.light.evaluation.hardware import (
    hardware_vendor,
    head_kernel_is_gluon,
)
from aka.light.acceptance.source_policy import install_workspace_policy
from aka.light.workflow.session_io import _sandbox_command
from aka.light.workspace.state import (
    git_head,
    head_kernel_is_initial_baseline,
    latest_version,
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
    # Light starts the persistent optimization Session directly from measured
    # V0. Bring-up, diagnostics, optimization and optional commit review all
    # happen in the same Agent conversation and trajectory Git repository.
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
    campaign.candidate_workspace.install_runtime(
        workspace,
        native,
        plugin_registry=campaign.plugin_registry,
        research_records=campaign.research_records,
        research_modules=campaign.research_modules,
        is_ppu=hardware_vendor(campaign.platform, campaign.arch) == "ppu",
    )
    install_workspace_policy(
        workspace, campaign.optimization_mode, campaign.framework,
        update_tracked_files=False,
        source_checks_enabled=campaign.source_checks_enabled,
        specify_dsl=campaign.specify_dsl,
        numerical_validation_enabled=campaign.module_options.numerical_validation,
        expose_evaluator_shapes=campaign.expose_evaluator_shapes,
    )


def episode_directives(campaign: Campaign) -> dict[str, str]:
    """Return only optional task-specific additions to the Episode prompt."""
    return {
        "plugins": campaign.plugin_directive("episode"),
    }


def session_execution_policy(campaign: Campaign) -> dict[str, object]:
    """Agent-session policy shared by every resumed Episode segment.

    Stage goals and terminal instructions belong in their prompts. Resource
    budgeting, reasoning effort and automatic recovery must not drift between
    the two workflows.
    """
    return {
        "episode_timeout_seconds": getattr(
            campaign.module_options, "episode_timeout_seconds", None
        ),
        "reasoning_effort": "max",
        "output_limit_recovery_retries": getattr(
            campaign.module_options,
            "output_limit_recovery_retries",
            _constants.DEFAULT_OUTPUT_LIMIT_RECOVERY_RETRIES,
        ),
        "dependency_policy_recovery_retries": getattr(
            campaign.module_options, "dependency_policy_recovery_retries", 2
        ),
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


# Re-exported for the rest of long_horizon, which reaches orchestrator only through this adapter.
run_bounded = _agent_process.run_bounded
peak_util = _workspace_state.peak_util
read_memory = _workspace_state.read_memory
speedup_vs_reference = _workspace_state.speedup_vs_reference
IMMUTABLE_BASELINE_PATHS = _constants.IMMUTABLE_BASELINE_PATHS
STALL_STATE_FILE = _constants.STALL_STATE_FILE
