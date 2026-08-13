"""Workspace runtime wiring: reference/gpu-wiki links, agent skills, session directives."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .constants import HUMANIZE_DIR, REPO_ROOT, STALL_STATE_FILE


def _install_agent_humanize_skill(skills_dir: Path) -> None:
    """Install a workspace-local, hydrated Humanize subset for Codex and Pi.

    Humanize's upstream Codex installer also changes user-global hooks and configuration. The
    orchestrator must not mutate global state, so this mirrors only the skill/runtime hydration
    into the campaign's repository-scoped ``.agents/skills`` directory.
    """
    source_skills = HUMANIZE_DIR / "skills"
    if not (source_skills / "humanize-gen-plan" / "SKILL.md").is_file():
        return

    skill_names = (
        "humanize",
        "humanize-gen-plan",
        "humanize-refine-plan",
        "humanize-rlcr",
    )
    runtime_root = skills_dir / "humanize"
    for skill_name in skill_names:
        source = source_skills / skill_name / "SKILL.md"
        if not source.is_file():
            continue
        destination_dir = skills_dir / skill_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        text = source.read_text(encoding="utf-8").replace(
            "{{HUMANIZE_RUNTIME_ROOT}}", str(runtime_root)
        )
        # These Claude plugin frontmatter keys are stripped by Humanize's own Codex installer.
        text = "\n".join(
            line for line in text.splitlines()
            if not line.startswith((
                "user-invocable:",
                "disable-model-invocation:",
                "hide-from-slash-command-tool:",
            ))
        ) + "\n"
        destination = destination_dir / "SKILL.md"
        if not destination.exists() or destination.read_text(encoding="utf-8") != text:
            destination.write_text(text, encoding="utf-8")

    for component in ("scripts", "hooks", "prompt-template", "templates", "config", "agents"):
        source = HUMANIZE_DIR / component
        destination = runtime_root / component
        if source.exists() and not destination.exists():
            os.symlink(source, destination)


def _agent_runtime_directive(agent_cli: str) -> str:
    if agent_cli in {"codex", "pi"}:
        syntax = "Codex's `$skill-name` syntax" if agent_cli == "codex" else "Pi's `/skill:name` syntax"
        return (
            f"- `.agents/skills/` — repository-local {agent_cli} skills, including "
            "`gpu-kernel-baseline`, `gpu-kernel-episode-loop`, `ncu-report-skill`, "
            f"`KernelWiki`, and `humanize-gen-plan`. Invoke a named skill with {syntax}."
        )
    return (
        "- `.claude/skills/ncu-report-skill/` — NVIDIA profiling skill.\n"
        "- `.claude/skills/KernelWiki/` — kernel optimization knowledge base."
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


def _plan_generator_directive(
    agent_cli: str,
    version: int,
    discussion: bool = False,
) -> str:
    draft = f"plans/v{version}_draft.md"
    plan = f"plans/v{version}_plan.md"
    if not discussion:
        return (
            f"Read `{draft}` and generate `{plan}` yourself in this same coding-agent session. "
            "Do not invoke Humanize, ask-codex, a slash command, the Skill tool, or a planning "
            "subagent. Write a complete plan containing the evidence-to-action chain, exactly "
            "one optimization category, concrete file changes, correctness/performance "
            "validation steps, and measurable acceptance criteria. Preserve the draft's Search "
            "Log and constraints."
        )
    if agent_cli == "codex":
        return (
            f"Invoke the `$humanize-gen-plan` skill with `{draft}` as input and `{plan}` as "
            "output. Use discussion/convergence mode with iterative review before finalizing. "
            "The skill is repository-local under `.agents/skills/`; do not look for a slash "
            "command or Claude plugin."
        )
    if agent_cli == "pi":
        return (
            f"Invoke `/skill:humanize-gen-plan` in this Pi session with `{draft}` as input and "
            f"`{plan}` as output. Use discussion/convergence mode and wait for the plan file before "
            "continuing."
        )
    if agent_cli == "qodercli":
        return (
            f"Read `{draft}` and generate `{plan}` yourself in this Qoder session. Do not invoke "
            "Humanize, a slash command, the Skill tool, or a planning subagent. Write a complete "
            "discussion-style plan with iterative self-review, containing the evidence-to-action "
            "chain, exactly one optimization "
            "category, concrete file changes, correctness/performance validation steps, and "
            "measurable acceptance criteria. Preserve the draft's Search Log and constraints."
        )
    return (
        "```text\n"
        f"/humanize:gen-plan --input {draft} --output {plan} --discussion\n"
        "```"
    )


def link_runtime(workspace: Path, atrex_bench_root: Optional[Path] = None) -> None:
    """Make the skill's `tools/`, `reference/`, `skills/`, `reference-projects/`, `gpu-wiki/` resolvable from cwd=workspace.

    The gpu-kernel-* skills reference these by relative path; sessions run with cwd=workspace,
    so symlink them in (absolute targets, so the workspace can live anywhere). Idempotent.

    Also installs the same skills and agent definitions into ``.claude/`` and ``.qoder/``, and
    repository-local Codex/Pi skills into ``.agents/skills/``.

    Claude loads Humanize via ``--plugin-dir`` after the orchestrator provisions ``jq``. Qoder
    owns plan generation directly and does not load Humanize. Codex and Pi receive a
    repository-scoped, hydrated Humanize skill without changing global user state.
    """
    for sub in ("tools", "reference", "skills", "reference-projects", "gpu-wiki"):
        src, dst = REPO_ROOT / sub, workspace / sub
        if src.exists() and not dst.exists():
            os.symlink(src, dst)
    if atrex_bench_root is not None:
        evaluator = atrex_bench_root / "scripts" / "run_eval.py"
        package = atrex_bench_root / "src" / "atrex_bench"
        if not evaluator.is_file() or not package.is_dir():
            raise FileNotFoundError(
                f"invalid Atrex-Bench runtime root (missing run_eval.py/src): {atrex_bench_root}"
            )
        runtime_link = workspace / "atrex-bench"
        if runtime_link.is_symlink():
            if runtime_link.resolve() != atrex_bench_root.resolve():
                raise RuntimeError(
                    f"workspace Atrex-Bench runtime points at {runtime_link.resolve()}, "
                    f"expected {atrex_bench_root.resolve()}"
                )
        elif runtime_link.exists():
            raise RuntimeError(
                f"workspace path blocks the Atrex-Bench runtime link: {runtime_link}"
            )
        else:
            os.symlink(atrex_bench_root.resolve(), runtime_link)
    # Claude and Qoder use parallel project-local discovery roots. Keep their contents identical
    # so selecting a different --agent-cli does not change the available optimization knowledge.
    ncu_src = REPO_ROOT / "3rdparty" / "ncu-report-skill"
    kw_src = REPO_ROOT / "gpu-wiki" / "3rdparty" / "KernelWiki"
    agents_src = REPO_ROOT / "agents"
    for runtime_dir_name in (".claude", ".qoder"):
        runtime_dir = workspace / runtime_dir_name
        runtime_skills_dir = runtime_dir / "skills"
        runtime_agents_dir = runtime_dir / "agents"
        runtime_skills_dir.mkdir(parents=True, exist_ok=True)
        for src, name in ((ncu_src, "ncu-report-skill"), (kw_src, "KernelWiki")):
            dst = runtime_skills_dir / name
            if src.exists() and not dst.exists():
                os.symlink(src, dst)
        # Claude/Qoder setup prompts can launch the baseline agent by name.
        if agents_src.exists() and not runtime_agents_dir.exists():
            os.symlink(agents_src, runtime_agents_dir)

    # Codex and Pi discover repository-scoped skills from .agents/skills. Keep these local to
    # the campaign so selecting either runtime neither requires nor mutates user-global state.
    # The project-native optimization skills can remain symlinks; Humanize needs a
    # hydrated SKILL.md, so it is materialized by the helper above.
    agent_skills_dir = workspace / ".agents" / "skills"
    agent_skills_dir.mkdir(parents=True, exist_ok=True)
    project_skills = REPO_ROOT / "skills"
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
    _install_agent_humanize_skill(agent_skills_dir)
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
        if "# orchestrator runtime symlinks (not part of the workspace)" not in existing_lines:
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
        add += ("\n# orchestrator live stall counter (rebuilt on restart; never committed)\n"
                "/" + STALL_STATE_FILE + "\n")
    if add:
        with gi.open("a", encoding="utf-8") as fh:
            fh.write(add)
