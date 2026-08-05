from __future__ import annotations

import json
from pathlib import Path

from orchestrator import optimize as base
from orchestrator.optimization_policy import install_workspace_policy


def prepare_campaign(campaign: base.Campaign) -> None:
    """Run current main's complete setup/resume prelude, excluding only its loop."""
    if base.latest_version(campaign.workspace) < 0:
        campaign.setup_baseline()
    else:
        if not base.git_head(campaign.workspace):
            raise RuntimeError("existing campaign workspace has no Git HEAD")
        print(
            f"[orchestrator] resuming: latest = v{base.latest_version(campaign.workspace)}",
            flush=True,
        )
        base.preserve_interrupted_tracked_changes(
            campaign.workspace, f"resume {campaign.campaign_name}"
        )
        campaign._link_runtime()  # Compatibility seam intentionally isolated in this module.
    campaign.ensure_framework_baseline()
    if campaign.optimization_mode == "production" and base.latest_version(campaign.workspace) > 0:
        violations = base.production_kernel_violations(
            campaign.workspace, campaign.framework
        )
        if violations and not base.head_kernel_is_initial_baseline(campaign.workspace):
            raise RuntimeError(
                "cannot resume a non-compliant production HEAD: " + "; ".join(violations)
            )
        if violations:
            print(
                "[orchestrator] production resume: HEAD kernel is still the original "
                "V0 baseline; continuing until a framework-compliant candidate is accepted",
                flush=True,
            )


def link_episode_runtime(campaign: base.Campaign, workspace: Path) -> None:
    native = Path(campaign.atrex_bench_root) if campaign.atrex_bench_root else None
    base.link_runtime(workspace, native)
    install_workspace_policy(workspace, campaign.optimization_mode, campaign.framework)


def episode_directives(campaign: base.Campaign) -> dict[str, str]:
    return {
        "hardware": base.hardware_directive(campaign.platform, campaign.arch),
        "sandbox": campaign._sandbox_directive(),
        "evaluator": campaign._evaluator_directive(),
        "mode_policy": campaign._mode_directive(),
    }


def iteration_playbook(campaign: base.Campaign, workspace: Path, version: int) -> str:
    """Render main's current iteration playbook as the episode's inherited workflow."""
    agent_cli = getattr(campaign, "agent_cli", "claude")
    return base._render(
        base.PROMPTS_DIR / "iteration.md",
        WORKSPACE=str(workspace),
        N=version,
        PREV=version - 1,
        PLATFORM=campaign.platform,
        NOTES=campaign.notes,
        AGENT_RUNTIME=base._agent_runtime_directive(agent_cli),
        PLAN_GENERATOR=base._plan_generator_directive(agent_cli, version),
        HARDWARE=base.hardware_directive(campaign.platform, campaign.arch),
        SANDBOX=campaign._sandbox_directive(),
        EVALUATOR=campaign._evaluator_directive(),
        MODE_POLICY=campaign._mode_directive(),
    )


def fresh_session_command(
    prompt: str,
    session_id: str,
    reasoning_effort: str,
    agent_cli: str = "claude",
) -> list[str]:
    command = base._session_command(agent_cli, prompt, session_id, reasoning_effort)
    if agent_cli == "codex":
        if command[:2] != ["codex", "exec"] or command[-1] != prompt:
            raise RuntimeError("current main Codex command has no compatible exec seam")
        # Main deliberately starts disposable Codex iterations. Long Horizon owns a
        # different lifecycle: its bounded recovery turns must resume the same thread.
        # Adapt that policy here instead of changing the main orchestrator command.
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
        raise RuntimeError("current main Claude command has no --session-id compatibility seam") from exc
    command[index : index + 2] = ["--resume", session_id]
    return command


def session_environment(agent_cli: str = "claude") -> dict[str, str]:
    return base._session_env(agent_cli)


def supports_same_session_resume(agent_cli: str) -> bool:
    return agent_cli in {"claude", "codex"}


def session_id_from_stream(agent_cli: str, stdout: str, requested_session_id: str) -> str:
    """Return the persistent CLI session id observed in one stream.

    Claude accepts the supervisor-provided id. Codex creates its own thread id
    and reports it in the initial ``thread.started`` JSONL event, so recovery
    must use that observed id rather than the supervisor's bookkeeping UUID.
    """
    if agent_cli != "codex":
        return requested_session_id
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("type") != "thread.started":
            continue
        thread_id = event.get("thread_id") or event.get("threadId")
        if isinstance(thread_id, str) and thread_id.strip():
            return thread_id.strip()
    return ""


def run_bounded(
    command: list[str], workspace: Path, timeout: int, environment: dict[str, str]
) -> tuple[str, str, int, bool]:
    return base._run_bounded(command, workspace, timeout, environment)


def tokens_from_stream(stdout: str) -> int:
    return base._tokens_from_stream(stdout)


def latest_version(workspace: Path) -> int:
    return base.latest_version(workspace)


def initial_aggregation_padding_target(campaign: base.Campaign) -> int | None:
    """Return main's bootstrap barrier only for coordinator-owned bucket campaigns."""
    if not callable(getattr(campaign, "on_improvement", None)):
        return None
    if not callable(getattr(campaign, "on_iteration", None)):
        return None
    return int(base.INITIAL_AGGREGATION_MIN_ITERATIONS)


def has_accepted_bucket_kernel(campaign: base.Campaign) -> bool:
    """Use main's kernel-provenance predicate instead of trusting counters or memory."""
    return not base.head_kernel_is_initial_baseline(campaign.workspace)


def run_sandbox(
    workspace: Path,
    hardware: str,
    profile: str,
    url: str,
    timeout: int,
    command: list[str],
    *,
    sync: tuple[str, ...] = (),
    wall_timeout: int | None = None,
    gateway_kind: str = "auto",
):
    """Use main's sandbox command builder and queue/timeout semantics verbatim."""
    return base._sandbox_command(
        workspace,
        hardware,
        profile,
        url,
        timeout,
        command,
        sync=sync,
        wall_timeout=wall_timeout,
        gateway_kind=gateway_kind,
    )


def candidate_policy_violations(campaign: base.Campaign, workspace: Path) -> list[str]:
    if campaign.optimization_mode != "production":
        return []
    return base.production_kernel_violations(workspace, campaign.framework)


def notify_iteration(
    campaign: base.Campaign,
    version: int,
    memory: dict | None,
    won: bool,
    previous_latency: float | None = None,
) -> None:
    if won and hasattr(campaign, "_notify_improvement"):
        campaign._notify_improvement(version, memory, previous_latency)
    if hasattr(campaign, "_notify_iteration"):
        campaign._notify_iteration(version, memory, won)


def peak_util(memory: dict | None) -> float:
    return base.peak_util(memory)


def conversion_required(campaign: base.Campaign, stall: int, workspace: Path) -> bool:
    return base.should_convert_to_gluon(
        campaign.framework,
        stall,
        int(getattr(campaign, "convert_after", base.DEFAULT_CONVERT_AFTER)),
        head_is_gluon=base.head_kernel_is_gluon(workspace),
    )


def candidate_is_gluon(workspace: Path) -> bool:
    return base.head_kernel_is_gluon(workspace)


def restored_stall(workspace: Path) -> int:
    value = base.read_stall(workspace)
    return int(value if value is not None else base.reconstruct_stall(workspace))


def save_stall(workspace: Path, value: int) -> None:
    base.write_stall(workspace, value)


Campaign = base.Campaign
IMMUTABLE_BASELINE_PATHS = base.IMMUTABLE_BASELINE_PATHS
CONVERT_PERF_TOL = base.CONVERT_PERF_TOL
STALL_STATE_FILE = base.STALL_STATE_FILE
