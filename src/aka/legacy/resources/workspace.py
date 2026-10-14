"""Install workspace runtime files using explicitly supplied repository resources."""
from __future__ import annotations
import os
import shutil
from pathlib import Path
from typing import Optional

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


def install_workspace_resources(workspace: Path, resources, *, plugin_registry, stall_state_file: str):
    """Install a selected contribution; preserve original workspace semantics."""
    registry = plugin_registry
    registry.check_lock(workspace)
    for src, destination in resources.runtime_links:
        dst = workspace / destination
        if not dst.exists():
            os.symlink(src, dst)
    if resources.evaluator_root is not None:
        _install_atrex_bench_runtime(workspace, resources.evaluator_root)
    # Claude and Qoder use parallel project-local discovery roots. Keep their contents identical
    # so selecting a different --agent-cli does not change the available optimization knowledge.
    ncu_src = resources.ncu_skill
    agents_src = resources.agents
    if not resources.ppu_enabled:
        for root in (".claude", ".qoder", ".agents"):
            stale = workspace / root / "skills" / "ppu-acu-joint-profile"
            if stale.is_symlink():
                stale.unlink()
    for runtime_dir_name in (".claude", ".qoder"):
        runtime_dir = workspace / runtime_dir_name
        runtime_skills_dir = runtime_dir / "skills"
        runtime_agents_dir = runtime_dir / "agents"
        runtime_skills_dir.mkdir(parents=True, exist_ok=True)
        for src, name in ((ncu_src, "ncu-report-skill"),):
            dst = runtime_skills_dir / name
            if src.exists() and not dst.exists():
                os.symlink(src, dst)
        for source, name in resources.backend_skills:
            destination = runtime_skills_dir / name
            if not destination.exists():
                os.symlink(source, destination)
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
    for source in resources.project_skills:
        destination = agent_skills_dir / source.name
        if not destination.exists():
            os.symlink(source, destination)
    for source, name in ((ncu_src, "ncu-report-skill"),):
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
    if resources.evaluator_root is not None and "/atrex-bench" not in existing:
        add += "/atrex-bench\n"
    if "/" + stall_state_file not in existing:
        add += (
            "\n# orchestrator live stall counter (rebuilt on restart; never committed)\n"
            "/" + stall_state_file + "\n"
        )
    if add:
        with gi.open("a", encoding="utf-8") as fh:
            fh.write(add)

    # One-time adoption of pre-plugin workspaces: remove only the exact symlinks
    # installed by the previous runtime when their owning plugin is disabled.
    legacy_mounts = dict(resources.legacy_mounts)
    desired = registry.mounts()
    for name, source in legacy_mounts.items():
        destination = workspace / name
        if (
            name not in desired
            and destination.is_symlink()
            and destination.resolve() == source.resolve()
        ):
            destination.unlink()
    return registry.install(workspace)


def link_runtime(workspace: Path, atrex_bench_root: Optional[Path] = None, *, is_ppu=False,
                 plugin_registry, repo_root: Path, stall_state_file: str):
    """Compatibility entry: select legacy resources, then install the contribution."""
    from .contributions import select_workspace_resources
    resources = select_workspace_resources(repo_root, is_ppu=is_ppu, atrex_bench_root=atrex_bench_root)
    return install_workspace_resources(workspace, resources, plugin_registry=plugin_registry,
                                       stall_state_file=stall_state_file)
