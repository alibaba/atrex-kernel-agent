"""Install the small Agent-facing tool/knowledge view, not the Supervisor source tree."""

from __future__ import annotations

import logging
import os
import shutil
import stat
import uuid
from pathlib import Path

from .agent_skill_manifest import SKILL_MANIFEST
from .constants import REPO_ROOT, STALL_STATE_FILE

RETIRED_SKILL_NAMES = frozenset({
    "humanize", "humanize-gen-plan", "humanize-refine-plan", "humanize-rlcr",
    "gen-plan", "gpu-kernel-baseline", "gpu-kernel-episode-loop", "ncu-report-skill",
})


def _agent_runtime_directive(agent_cli: str, *, is_ppu: bool = False) -> str:
    root = ".agents" if agent_cli in {"codex", "pi"} else ".qoder" if agent_cli == "qodercli" else ".claude"
    extra = ", PPU profiling" if is_ppu else ""
    return (
        f"- `skills/` (also discoverable under `{root}/skills/`): `gpu-measurement`, "
        f"`runtime-records`, `KernelWiki`, and optional timeline diagnostics{extra}. "
        "Read the relevant Skill for request schemas/examples. "
        "Use `python3 tools/plugin.py list` for additional operator-enabled tools and Skills."
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
    from .agent_home import open_private_directory
    from .supervisor_runtime import supervisor_campaign_root

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


def link_runtime(workspace: Path, atrex_bench_root: Path | None = None, *, is_ppu: bool = False,
                 plugin_registry=None) -> None:
    from .agent_home import open_private_directory
    from supervisor.workspace import publish

    # Upgrade only controller-created links. Source assets are copied with the same
    # bounded, no-follow publisher used by Episode input projection.
    for name in ("tools", "skills", "reference", "reference-projects", "gpu-wiki"):
        _remove_legacy_link(workspace / name)
    publish(workspace, "tools/sandbox.py", (REPO_ROOT / "tools/sandbox.py").read_bytes())
    publish(workspace, "tools/plugin.py", (REPO_ROOT / "tools/plugin.py").read_bytes())
    names = ["gpu-measurement", "runtime-records", "KernelWiki", "autonomous-gpu-kernel-timeline"]
    if is_ppu:
        names.append("ppu-acu-joint-profile")
    inactive = RETIRED_SKILL_NAMES | (set(SKILL_MANIFEST) - set(names))
    for name in sorted(inactive):
        _archive_skill_entry(workspace, workspace / "skills" / name)
    for name in names:
        manifest = SKILL_MANIFEST[name]
        for filename in manifest.files:
            source = REPO_ROOT / manifest.root / filename
            publish(workspace, f"skills/{name}/{filename}", source.read_bytes())
    if plugin_registry is None:
        from .plugins import PluginRegistry
        plugin_registry = PluginRegistry()
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
    gi = workspace / ".gitignore"
    from .session_tail import read_regular_bytes
    try:
        existing = read_regular_bytes(gi, limit=1024 * 1024).decode()
    except FileNotFoundError:
        existing = ""
    ignores = ("/tools", "/skills", "/reference", "/reference-projects", "/gpu-wiki", "/.claude", "/.qoder", "/.agents", "/" + STALL_STATE_FILE)
    # Preserve the committed ignore file of resumed SOL workspaces, which never
    # installed Atrex-Bench. Episode-boundary dirty checks must remain strict.
    if atrex_bench_root is not None:
        ignores += ("/atrex-bench",)
    missing = [name for name in ignores if name not in existing.splitlines()]
    if missing:
        publish(workspace, ".gitignore", (existing.rstrip() + "\n" + "\n".join(missing) + "\n").encode())
