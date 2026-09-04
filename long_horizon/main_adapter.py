from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from orchestrator import constants as _constants
from orchestrator import workspace_state as _workspace_state
from orchestrator.agent_runtime import process as _agent_process
from orchestrator.agent_runtime.adapter import DEFAULT_BACKEND_REGISTRY
from orchestrator.agent_runtime.codex_ledger import (
    CodexSessionLedgerObserver,
    codex_thread_id_from_stream,
    observe_codex_usage,
)
from orchestrator.agent_runtime.model import (
    AgentRuntimeCapabilities,
    NormalizedAgentEvent,
    TokenUsage,
)
from orchestrator.agent_runtime.runtime import (
    build_session_command,
    build_session_environment,
    terminal_usage_from_stream,
    token_usage_from_stream,
)
from orchestrator.campaign import Campaign
from orchestrator.constants import DEFAULT_CONVERT_AFTER
from orchestrator.hardware import (
    hardware_directive,
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
    native = Path(campaign.atrex_bench_root) if campaign.atrex_bench_root else None
    link_runtime(workspace, native)
    install_workspace_policy(workspace, campaign.optimization_mode, campaign.framework)


def episode_directives(
    campaign: Campaign, version: int, *, fast: bool = False
) -> dict[str, str]:
    agent_cli = getattr(campaign, "agent_cli", "claude")
    return {
        "hardware": hardware_directive(campaign.platform, campaign.arch),
        "sandbox": (
            campaign._fast_sandbox_directive()
            if fast
            else campaign._sandbox_directive()
        ),
        "evaluator": campaign._evaluator_directive(),
        "mode_policy": campaign._mode_directive(),
        "agent_runtime": _agent_runtime_directive(agent_cli),
        "plan_generator": _plan_generator_directive(agent_cli, version),
    }


def fresh_session_command(
    prompt: str,
    session_id: str,
    reasoning_effort: str,
    agent_cli: str = "claude",
) -> list[str]:
    command = build_session_command(agent_cli, prompt, session_id, reasoning_effort)
    if agent_cli == "codex":
        if command[:2] != ["codex", "exec"] or command[-1] != prompt:
            raise RuntimeError("current main Codex command has no compatible exec seam")
        # Long Horizon recovery must keep the native Codex thread persistent.
        # The current adapter already omits this flag; retain the compatibility
        # guard in case the base command policy changes independently.
        try:
            command.remove("--ephemeral")
        except ValueError:
            pass
    return command


def resume_session_command(
    prompt: str,
    session_id: str,
    reasoning_effort: str,
    agent_cli: str = "claude",
) -> list[str]:
    if agent_cli == "codex":
        command = fresh_session_command(prompt, session_id, reasoning_effort, agent_cli)
        if command[:2] != ["codex", "exec"] or command[-1] != prompt:
            raise RuntimeError("current main Codex command has no compatible exec seam")
        command = command[:-1]
        try:
            color_index = command.index("--color")
        except ValueError:
            pass
        else:
            del command[color_index : color_index + 2]
        command.insert(2, "resume")
        command.extend([session_id, prompt])
        return command
    if agent_cli != "claude":
        raise RuntimeError(
            f"same-session handoff recovery is not supported by current main for {agent_cli}"
        )
    command = fresh_session_command(prompt, session_id, reasoning_effort, agent_cli)
    try:
        index = command.index("--session-id")
    except ValueError as exc:
        raise RuntimeError(
            "current main Claude command has no --session-id compatibility seam"
        ) from exc
    command[index : index + 2] = ["--resume", session_id]
    return command


def session_environment(agent_cli: str = "claude") -> dict[str, str]:
    return build_session_environment(agent_cli)


def supports_same_session_resume(agent_cli: str) -> bool:
    return agent_cli in {"claude", "codex"}


def session_id_from_stream(
    agent_cli: str, stdout: str, requested_session_id: str
) -> str:
    """Return the persistent CLI session id observed in one stream.

    Claude accepts the supervisor-provided id. Codex creates its own thread id
    and reports it in the initial ``thread.started`` JSONL event, so recovery
    must use that observed id rather than the supervisor's bookkeeping UUID.
    """
    if agent_cli != "codex":
        return requested_session_id
    return codex_thread_id_from_stream(stdout)


def tokens_from_stream(stdout: str) -> int:
    return token_usage_from_stream(stdout)


def normalize_stream(
    agent_cli: str,
    stdout: str,
    *,
    session_id: str = "",
    codex_observer: CodexSessionLedgerObserver | None = None,
) -> tuple[
    tuple[NormalizedAgentEvent, ...],
    TokenUsage,
    AgentRuntimeCapabilities,
    tuple[str, ...],
]:
    """Observe one long-session invocation through main's backend adapter."""
    adapter = DEFAULT_BACKEND_REGISTRY.create(agent_cli)
    observation_errors: tuple[str, ...] = ()
    try:
        events, terminal_usage = adapter.normalize_stream(stdout)
    except Exception as exc:
        events = ()
        terminal_usage = terminal_usage_from_stream(stdout)
        observation_errors = (f"stream_normalization_failed:{type(exc).__name__}",)
    capabilities = replace(
        adapter.capabilities,
        usage_delta_observed=any(event.kind == "usage_delta" for event in events),
    )
    if agent_cli == "codex" and codex_observer is not None and session_id:
        try:
            (
                events,
                terminal_usage,
                capabilities,
                ledger_errors,
            ) = observe_codex_usage(codex_observer, session_id, terminal_usage)
            observation_errors += ledger_errors
        except Exception as exc:
            observation_errors += (f"codex_ledger_unavailable:{type(exc).__name__}",)
    return events, terminal_usage, capabilities, observation_errors


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
run_bounded = _agent_process.run_bounded
peak_util = _workspace_state.peak_util
read_memory = _workspace_state.read_memory
speedup_vs_reference = _workspace_state.speedup_vs_reference
CONVERT_PERF_TOL = _constants.CONVERT_PERF_TOL
IMMUTABLE_BASELINE_PATHS = _constants.IMMUTABLE_BASELINE_PATHS
STALL_STATE_FILE = _constants.STALL_STATE_FILE
