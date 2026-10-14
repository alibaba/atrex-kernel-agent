"""Workspace runtime wiring: references, plugin resources, skills and directives."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .constants import REPO_ROOT, STALL_STATE_FILE
from .plugins import PluginRegistry


def _agent_runtime_directive(agent_cli: str, *, is_ppu: bool = False) -> str:
    ppu_skill = "`ppu-acu-joint-profile`, " if is_ppu else ""
    if agent_cli in {"codex", "pi"}:
        syntax = (
            "Codex's `$skill-name` syntax"
            if agent_cli == "codex"
            else "Pi's `/skill:name` syntax"
        )
        return (
            f"- `.agents/skills/` — repository-local {agent_cli} skills, including "
            "`gpu-kernel-baseline`, `gpu-kernel-episode-loop`, "
            f"`autonomous-gpu-kernel-timeline`, {ppu_skill}`ncu-report-skill`, "
            f"`gen-plan` and discovered plugin skills. Invoke a named skill with {syntax}."
        )
    runtime_root = ".qoder" if agent_cli == "qodercli" else ".claude"
    return (
        f"- `{runtime_root}/skills/` — repository-local runtime skills, including `gen-plan`, "
        f"`autonomous-gpu-kernel-timeline`, {ppu_skill}`ncu-report-skill`, "
        "and discovered plugin skills."
    )


def _baseline_driver_directive(agent_cli: str) -> str:
    if agent_cli == "codex":
        return (
            "Use the `$gpu-kernel-baseline` skill and complete its baseline workflow in this "
            "session. If Codex collaboration/sub-agent tools are available, delegate that bounded "
            "implementation task and wait for it; otherwise execute the skill directly yourself"
        )
    if agent_cli == "pi":
        return (
            "Use the `/skill:gpu-kernel-baseline` skill and complete its workflow directly in "
            "this Pi session. Pi has no built-in subagent requirement here; do not launch a "
            "nested coding-agent process"
        )
    if agent_cli == "qodercli":
        return (
            "Complete the baseline workflow directly in this Qoder session. Do not launch an "
            "Agent/subagent. Treat the current working directory as the only writable "
            "workspace and use relative paths for every campaign file"
        )
    return (
        "Launch the `gpu-kernel-baseline` subagent (by name). You may spawn it in the "
        "background, but **you MUST wait for it to complete before you exit**"
    )


def _plan_generator_directive(agent_cli: str, version: int) -> str:
    draft = f"plans/v{version}_draft.md"
    plan = f"plans/v{version}_plan.md"
    if agent_cli == "codex":
        return (
            f"Invoke the `$gen-plan` skill with `{draft}` as input and `{plan}` as "
            "output. Use direct/no-discussion mode for this optimization plan. "
            "The skill is repository-local under `.agents/skills/`; freeze its candidate proposal "
            "and Codex review in this current session before reading the independent Qoder review, "
            "then synthesize."
        )
    if agent_cli == "pi":
        return (
            f"Invoke `/skill:gen-plan` in this Pi session with `{draft}` as input and "
            f"`{plan}` as output. Use direct/no-discussion mode and wait for the plan file before "
            "continuing."
        )
    if agent_cli == "qodercli":
        return (
            f"Read `skills/gen-plan/SKILL.md` and execute it directly with `{draft}` as input and "
            f"`{plan}` as output in direct/no-discussion mode. Do not launch a planning subagent; "
            "freeze the skill's candidate proposal and Qoder review in this current session before "
            "reading the independent Codex review, then synthesize and wait for the plan file before "
            "continuing."
        )
    return f"```text\n/gen-plan --input {draft} --output {plan} --direct\n```"


def _install_atrex_bench_runtime(workspace: Path, atrex_bench_root: Path) -> None:
    from ._bootstrap import task_modules
    task_modules()
    from aka.legacy.task.candidate_workspace.runtime import _install_atrex_bench_runtime as install
    install(workspace, atrex_bench_root)


def link_runtime(
    workspace: Path,
    atrex_bench_root: Optional[Path] = None,
    *,
    is_ppu: bool = False,
    plugin_registry: PluginRegistry | None = None,
    provider=None,
) -> None:
    from ._bootstrap import task_modules
    task_modules()
    from aka.legacy.application.workspace import candidate_workspace
    with candidate_workspace(provider=provider) as provider:
        provider.install_runtime(workspace, atrex_bench_root,
                                 is_ppu=is_ppu,
                                 plugin_registry=plugin_registry or PluginRegistry(),
                                 repo_root=REPO_ROOT, stall_state_file=STALL_STATE_FILE)
