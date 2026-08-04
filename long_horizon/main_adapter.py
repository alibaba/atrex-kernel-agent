from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from orchestrator import optimize as base
from orchestrator.optimization_policy import install_workspace_policy


@dataclass(frozen=True)
class OperatorInfo:
    name: str
    reference: str
    op_dir: str
    atrex_bench_root: str = ""


def resolve_operator(op_dir: str) -> OperatorInfo:
    directory = Path(op_dir).resolve()
    if not directory.is_dir():
        raise ValueError(f"operator directory not found: {directory}")
    reference = directory / "reference.py"
    if not reference.is_file():
        raise ValueError(f"operator directory has no reference.py: {directory}")
    atrex_root = ""
    if not base.is_sol_op(directory) and (directory / "shapes.json").is_file():
        root = base.find_atrex_bench_root(directory)
        if root is None:
            raise ValueError(
                "native Atrex-Bench operator requires scripts/run_eval.py and src/atrex_bench"
            )
        atrex_root = str(root)
    return OperatorInfo(
        name=directory.name,
        reference=str(reference),
        op_dir=str(directory),
        atrex_bench_root=atrex_root,
    )


def prepare_campaign(campaign: base.Campaign) -> None:
    """Use current main for baseline setup, never for the optimization loop."""
    if base.latest_version(campaign.workspace) < 0:
        campaign.setup_baseline()
    else:
        if not base.git_head(campaign.workspace):
            raise RuntimeError("existing campaign workspace has no Git HEAD")
        campaign._link_runtime()  # Compatibility seam intentionally isolated in this module.
    campaign.ensure_framework_baseline()


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


def fresh_session_command(prompt: str, session_id: str, reasoning_effort: str) -> list[str]:
    return base._session_command("claude", prompt, session_id, reasoning_effort)


def resume_session_command(prompt: str, session_id: str, reasoning_effort: str) -> list[str]:
    command = fresh_session_command(prompt, session_id, reasoning_effort)
    try:
        index = command.index("--session-id")
    except ValueError as exc:
        raise RuntimeError("current main Claude command has no --session-id compatibility seam") from exc
    command[index : index + 2] = ["--resume", session_id]
    return command


def session_environment() -> dict[str, str]:
    return base._session_env("claude")


def run_bounded(
    command: list[str], workspace: Path, timeout: int, environment: dict[str, str]
) -> tuple[str, str, int, bool]:
    return base._run_bounded(command, workspace, timeout, environment)


def tokens_from_stream(stdout: str) -> int:
    return base._tokens_from_stream(stdout)


def latest_version(workspace: Path) -> int:
    return base.latest_version(workspace)


def ensure_submodules() -> None:
    base.ensure_submodules()


def detect_arch(hardware: str, profile: str, url: str) -> str:
    return base.detect_arch(hardware, profile, url)


def framework_workspace_suffix(framework: str, platform: str, mode: str) -> str:
    return base.framework_workspace_suffix(framework, platform, mode)


Campaign = base.Campaign
AGENT_CLI_CHOICES = ("claude",)
OPTIMIZATION_MODE_CHOICES = base.OPTIMIZATION_MODE_CHOICES
FRAMEWORK_BASELINE_MODES = base.FRAMEWORK_BASELINE_MODES
DEFAULT_SANDBOX_TIMEOUT = base.DEFAULT_SANDBOX_TIMEOUT
MAX_SANDBOX_TIMEOUT = base.MAX_SANDBOX_TIMEOUT
