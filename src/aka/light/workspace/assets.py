"""Install the small Agent-facing tool/knowledge view, not the Supervisor source tree."""

from __future__ import annotations

import logging
import os
import shutil
import stat
import uuid
from pathlib import Path

from aka.light.workspace.skill_manifest import SKILL_MANIFEST
from aka.light.workspace.resource_catalog import resource
from aka.light.workflow.constants import REPO_ROOT

RETIRED_SKILL_NAMES = frozenset({
    "humanize", "humanize-gen-plan", "humanize-refine-plan", "humanize-rlcr",
    "gen-plan", "gpu-kernel-baseline", "gpu-kernel-episode-loop", "ncu-report-skill",
})


def _agent_runtime_directive(
    agent_cli: str,
    *,
    is_ppu: bool = False,
    wiki: bool = False,
    pause_agent_during_requests: bool = True,
    knowledge_skills=("KernelWiki",),
) -> str:
    root = ".agents" if agent_cli in {"codex", "pi", "dsh"} else ".qoder" if agent_cli == "qodercli" else ".claude"
    extra = ", PPU profiling" if is_ppu else ""
    knowledge = "".join(f", `{name}`" for name in knowledge_skills) if wiki else ""
    suspension = (
        "Authenticated Runtime requests pause this Agent and its subagents; that wait is excluded "
        "from the Agent active-time budget."
        if pause_agent_during_requests
        else "Runtime request suspension is disabled; request time remains part of the Agent "
        "active-time budget. Keep each request in the foreground and wait for it to finish."
    )
    return (
        f"- `skills/` (also discoverable under `{root}/skills/`): `gpu-measurement`, "
        f"`runtime-records`{knowledge}, and optional timeline diagnostics{extra}. "
        "Read the relevant Skill for request schemas/examples. "
        "Use `python3 tools/plugin.py list` for additional operator-enabled tools and Skills. "
        + suspension
    )


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


def _remove_legacy_link(path: Path) -> None:
    """Unlink only known repository assets, never user directories or external targets."""
    if path.is_symlink():
        target = path.resolve()
        if not target.is_relative_to(REPO_ROOT):
            raise ValueError(f"Unrecognized runtime asset link: {path}")
        path.unlink()


def _archive_skill_entry(workspace: Path, path: Path) -> None:
    """Move a reserved Skill entry out of discovery without reading/deleting its contents."""
    from aka.light.agent.home import open_private_directory
    from aka.light.runtime.paths import supervisor_campaign_root

    relative = path.relative_to(workspace)
    with open_private_directory(path.parent) as parent:
        try:
            mode = os.stat(path.name, dir_fd=parent, follow_symlinks=False).st_mode
        except FileNotFoundError:
            return
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode) or stat.S_ISLNK(mode)):
            raise ValueError(f"Invalid Skill entry {path}; stop the Supervisor and move this special file out of discovery")
        backup = supervisor_campaign_root(workspace) / "skill-migrations" / uuid.uuid4().hex / relative
        try:
            with open_private_directory(backup.parent) as destination:
                # Rename the entry itself, including a dangling/external symlink;
                # never follow it or recursively copy/remove an Agent-owned tree.
                os.rename(path.name, backup.name, src_dir_fd=parent, dst_dir_fd=destination)
        except OSError as error:
            raise RuntimeError(
                f"Cannot migrate Skill {path} to {backup}; original entry was not removed. "
                "Stop the Supervisor, repair directory permissions or move this entry to an operator backup, then retry"
            ) from error
    logging.getLogger(__name__).warning("Archived legacy/conflicting Skill %s to %s", path, backup)


def link_custom_skills(workspace: Path, reserved: tuple[str, ...]) -> None:
    """Expose custom Skills to CLI discovery without copying or following user links."""
    from aka.light.workspace.managed_assets import custom_entries, is_reserved, MAX_ENTRIES
    from aka.light.agent.home import open_private_directory
    # The bounded no-follow scan also validates custom tools before Session start.
    names = {Path(relative).parts[1] for relative, data, _ in custom_entries(workspace, reserved)
             if data is not None and len(Path(relative).parts) == 3
             and Path(relative).parts[0] == "skills" and Path(relative).name == "SKILL.md"}
    for backend in (".claude", ".qoder", ".agents"):
        root = workspace / backend / "skills"
        with open_private_directory(root) as directory:
            with os.scandir(directory) as entries:
                for index, entry in enumerate(entries):
                    if index >= MAX_ENTRIES:
                        raise ValueError("Skill discovery exceeds 4096 entries")
                    if is_reserved(f"skills/{entry.name}", reserved):
                        continue
                    try:
                        target = os.readlink(entry.name, dir_fd=directory)
                    except OSError:
                        continue
                    expected = workspace / "skills" / entry.name
                    if Path(os.path.abspath(root / target)) == expected and entry.name not in names:
                        os.unlink(entry.name, dir_fd=directory)
            for name in sorted(names):
                try:
                    os.stat(name, dir_fd=directory, follow_symlinks=False)
                except FileNotFoundError:
                    os.symlink(workspace / "skills" / name, name, dir_fd=directory)


def link_runtime(workspace: Path, atrex_bench_root: Path | None = None, *, is_ppu: bool = False,
                 plugin_registry=None, research_records: bool = False,
                 research_modules=None) -> None:
    # Install the same frozen module contract enforced by the Runtime.  A Skill
    # that advertises a disabled operation is not harmless documentation drift:
    # it causes the Agent to submit requests the Runtime must reject.
    modules = tuple(
        ("directions", "experiments") if research_records and research_modules is None
        else research_modules or ()
    )
    if len(set(modules)) != len(modules) or set(modules) - {"directions", "experiments"}:
        raise ValueError("research_modules may contain only unique directions and experiments")
    if research_records != bool(modules):
        raise ValueError("research_records must agree with whether research_modules are enabled")
    from aka.light.agent.home import open_private_directory
    from aka.light.workspace.files import publish

    # Upgrade only controller-created links. Source assets are copied with the same
    # bounded, no-follow publisher used by Episode input projection.
    for name in ("tools", "skills", "reference", "reference-projects", "gpu-wiki"):
        _remove_legacy_link(workspace / name)
    publish(workspace, "tools/sandbox.py", resource("tools/sandbox.py").read_bytes())
    # Gateway Records are ordinary projected JSON files.  Remove the retired
    # wrapper when upgrading an existing workspace so there is only one access
    # contract: normal read-only file operations under records/.
    with open_private_directory(workspace / "tools") as tools:
        try:
            os.unlink("record.py", dir_fd=tools)
        except FileNotFoundError:
            pass
        except IsADirectoryError as error:
            raise ValueError(
                "Reserved runtime tool path tools/record.py must not be a directory"
            ) from error
    publish(workspace, "tools/plugin.py", resource("tools/plugin.py").read_bytes())
    if plugin_registry is None:
        from aka.light.runtime.catalog import PluginRegistry
        plugin_registry = PluginRegistry()
    names = ["gpu-measurement", "runtime-records", "autonomous-gpu-kernel-timeline"]
    if any(plugin.id == "gpu-wiki" for plugin in plugin_registry.plugins):
        names.append("KernelWiki")
    if is_ppu:
        names.append("ppu-acu-joint-profile")
    inactive = RETIRED_SKILL_NAMES | (set(SKILL_MANIFEST) - set(names))
    for name in sorted(inactive):
        _archive_skill_entry(workspace, workspace / "skills" / name)
    for name in names:
        manifest = SKILL_MANIFEST[name]
        for filename in manifest.files:
            source = resource(f"{manifest.root}/{filename}")
            if name == "runtime-records":
                if not research_records and filename != "references/records.md":
                    source = resource(f"skills/runtime-report/{filename}")
                elif set(modules) in ({"directions"}, {"experiments"}) and filename in {
                    "SKILL.md", "references/journal.md"
                }:
                    variant = (
                        "directions-only.md"
                        if set(modules) == {"directions"}
                        else "experiments-only.md"
                    )
                    source = resource(
                        f"skills/runtime-records/{variant}"
                        if filename == "SKILL.md"
                        else f"skills/runtime-records/references/{variant}"
                    )
            data = source.read_bytes()
            publish(workspace, f"skills/{name}/{filename}", data)
    names.extend(plugin_registry.install_agent_skills(workspace))
    for backend in (".claude", ".qoder", ".agents"):
        _remove_legacy_link(workspace / backend / "agents")
        root = workspace / backend / "skills"
        with open_private_directory(root) as directory:
            for name in sorted(inactive):
                _archive_skill_entry(workspace, root / name)
            # Only reserved runtime names are migrated. Preserve unrelated user
            # directories and links, including names beginning with "humanize".
            for name in names:
                target = workspace / "skills" / name
                try:
                    existing = os.readlink(name, dir_fd=directory)
                except OSError:
                    existing = None
                if existing is not None and Path(os.path.abspath(root / existing)) == target.absolute():
                    continue
                _archive_skill_entry(workspace, root / name)
                os.symlink(target, name, dir_fd=directory)
    if atrex_bench_root is not None:
        _install_atrex_bench_runtime(workspace, atrex_bench_root)
    # Runtime assets are covered by the tracked policy installed when the
    # workspace is created, plus .git/info/exclude for resumed old workspaces.
    # Never mutate an already tracked .gitignore while installing a Session.
