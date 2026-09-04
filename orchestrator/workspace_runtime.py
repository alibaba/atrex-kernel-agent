"""Workspace runtime wiring: reference/gpu-wiki links, agent skills, session directives."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

from .constants import REPO_ROOT, STALL_STATE_FILE


def _agent_runtime_directive(agent_cli: str) -> str:
    if agent_cli in {"codex", "pi"}:
        syntax = (
            "Codex's `$skill-name` syntax"
            if agent_cli == "codex"
            else "Pi's `/skill:name` syntax"
        )
        return (
            f"- `.agents/skills/` — repository-local {agent_cli} skills, including "
            "`gpu-kernel-baseline`, `gpu-kernel-episode-loop`, "
            "`autonomous-gpu-kernel-timeline`, `ppu-acu-joint-profile`, `ncu-report-skill`, "
            f"`KernelWiki`, and `gen-plan`. Invoke a named skill with {syntax}."
        )
    runtime_root = ".qoder" if agent_cli == "qodercli" else ".claude"
    return (
        f"- `{runtime_root}/skills/` — repository-local runtime skills, including `gen-plan`, "
        "`autonomous-gpu-kernel-timeline`, `ppu-acu-joint-profile`, `ncu-report-skill`, "
        "and `KernelWiki`."
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
            "output. Use direct/no-discussion mode for this single-action optimization plan. "
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
    """Copy evaluator code without exposing the checkout's data directory."""
    evaluator = atrex_bench_root / "scripts" / "run_eval.py"
    package = atrex_bench_root / "src" / "atrex_bench"
    if not evaluator.is_file() or not package.is_dir():
        raise FileNotFoundError(
            f"invalid Atrex-Bench runtime root (missing run_eval.py/src): {atrex_bench_root}"
        )

    runtime_dir = workspace / "atrex-bench"
    if runtime_dir.is_symlink() or runtime_dir.is_file():
        runtime_dir.unlink()
    elif runtime_dir.exists():
        shutil.rmtree(runtime_dir)
    (runtime_dir / "scripts").mkdir(parents=True)
    shutil.copy2(evaluator, runtime_dir / "scripts" / "run_eval.py")
    shutil.copytree(
        package,
        runtime_dir / "src" / "atrex_bench",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )


def link_runtime(workspace: Path, atrex_bench_root: Optional[Path] = None) -> None:
    """Link repository runtime assets into a campaign workspace.

    The gpu-kernel-* skills reference ``tools/``, ``reference/``, ``skills/``,
    ``reference-projects/``, and ``gpu-wiki/`` by relative path. Sessions run with
    ``cwd=workspace``, so symlink them in using absolute targets. Atrex-Bench evaluator code is
    copied from its checkout without linking the checkout's private ``data/`` tree.

    Also installs the same skills and agent definitions into ``.claude/`` and ``.qoder/``, and
    repository-local Codex/Pi skills into ``.agents/skills/``.

    Every backend receives the repository-native ``gen-plan`` skill through its project-local
    discovery root; plan generation does not require an external plugin or global installation.
    """
    for sub in ("tools", "reference", "skills", "reference-projects", "gpu-wiki"):
        src, dst = REPO_ROOT / sub, workspace / sub
        if src.exists() and not dst.exists():
            os.symlink(src, dst)
    if atrex_bench_root is not None:
        _install_atrex_bench_runtime(workspace, atrex_bench_root)
    # Claude and Qoder use parallel project-local discovery roots. Keep their contents identical
    # so selecting a different --agent-cli does not change the available optimization knowledge.
    ncu_src = REPO_ROOT / "3rdparty" / "ncu-report-skill"
    kw_src = REPO_ROOT / "gpu-wiki" / "3rdparty" / "KernelWiki"
    agents_src = REPO_ROOT / "agents"
    project_skills = REPO_ROOT / "skills"
    plan_skill_src = project_skills / "gen-plan"
    timeline_skill_src = project_skills / "autonomous-gpu-kernel-timeline"
    ppu_profile_skill_src = project_skills / "ppu-acu-joint-profile"
    for runtime_dir_name in (".claude", ".qoder"):
        runtime_dir = workspace / runtime_dir_name
        runtime_skills_dir = runtime_dir / "skills"
        runtime_agents_dir = runtime_dir / "agents"
        runtime_skills_dir.mkdir(parents=True, exist_ok=True)
        for src, name in ((ncu_src, "ncu-report-skill"), (kw_src, "KernelWiki")):
            dst = runtime_skills_dir / name
            if src.exists() and not dst.exists():
                os.symlink(src, dst)
        plan_skill_dst = runtime_skills_dir / "gen-plan"
        if (plan_skill_src / "SKILL.md").is_file() and not plan_skill_dst.exists():
            os.symlink(plan_skill_src, plan_skill_dst)
        timeline_skill_dst = runtime_skills_dir / "autonomous-gpu-kernel-timeline"
        if (timeline_skill_src / "SKILL.md").is_file() and not timeline_skill_dst.exists():
            os.symlink(timeline_skill_src, timeline_skill_dst)
        ppu_profile_skill_dst = runtime_skills_dir / "ppu-acu-joint-profile"
        if (
            ppu_profile_skill_src / "SKILL.md"
        ).is_file() and not ppu_profile_skill_dst.exists():
            os.symlink(ppu_profile_skill_src, ppu_profile_skill_dst)
        # Claude/Qoder setup prompts can launch the baseline agent by name.
        if agents_src.exists() and not runtime_agents_dir.exists():
            os.symlink(agents_src, runtime_agents_dir)

    # Codex and Pi discover repository-scoped skills from .agents/skills. Keep these local to
    # the campaign so selecting either runtime neither requires nor mutates user-global state.
    agent_skills_dir = workspace / ".agents" / "skills"
    agent_skills_dir.mkdir(parents=True, exist_ok=True)
    # Remove runtime copies created by releases that hydrated the external plan plugin. These
    # paths are orchestrator-owned and ignored by campaign Git; leaving them in a resumed
    # workspace would expose duplicate or broken plan skills after the migration.
    for legacy_name in (
        "humanize",
        "humanize-gen-plan",
        "humanize-refine-plan",
        "humanize-rlcr",
    ):
        legacy_path = agent_skills_dir / legacy_name
        if legacy_path.is_symlink() or legacy_path.is_file():
            legacy_path.unlink()
        elif legacy_path.is_dir():
            shutil.rmtree(legacy_path)
    if project_skills.is_dir():
        for source in project_skills.iterdir():
            if not (source / "SKILL.md").is_file():
                continue
            destination = agent_skills_dir / source.name
            if not destination.exists():
                os.symlink(source, destination)
    for source, name in ((ncu_src, "ncu-report-skill"), (kw_src, "KernelWiki")):
        destination = agent_skills_dir / name
        if source.exists() and not destination.exists():
            os.symlink(source, destination)
    gi = workspace / ".gitignore"
    existing = gi.read_text(encoding="utf-8") if gi.exists() else ""
    existing_lines = set(existing.splitlines())
    add = ""
    runtime_ignores = [
        "/tools",
        "/reference",
        "/skills",
        "/reference-projects",
        "/gpu-wiki",
    ]
    missing_runtime_ignores = [
        entry for entry in runtime_ignores if entry not in existing_lines
    ]
    if missing_runtime_ignores:
        if (
            "# orchestrator runtime symlinks (not part of the workspace)"
            not in existing_lines
        ):
            add += "\n# orchestrator runtime symlinks (not part of the workspace)\n"
        add += "".join(f"{entry}\n" for entry in missing_runtime_ignores)
    if "/.claude" not in existing:
        add += "/.claude\n"
    if "/.qoder" not in existing:
        add += "/.qoder\n"
    if "/.agents" not in existing:
        add += "/.agents\n"
    if atrex_bench_root is not None and "/atrex-bench" not in existing:
        add += "/atrex-bench\n"
    if "/" + STALL_STATE_FILE not in existing:
        add += (
            "\n# orchestrator live stall counter (rebuilt on restart; never committed)\n"
            "/" + STALL_STATE_FILE + "\n"
        )
    if add:
        with gi.open("a", encoding="utf-8") as fh:
            fh.write(add)
