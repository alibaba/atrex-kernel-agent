"""Install the small Agent-facing tool/knowledge view, not the Supervisor source tree."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from .agent_skill_manifest import SKILL_MANIFEST
from .constants import REPO_ROOT, STALL_STATE_FILE


def _agent_runtime_directive(agent_cli: str, *, is_ppu: bool = False) -> str:
    root = ".agents" if agent_cli in {"codex", "pi"} else ".qoder" if agent_cli == "qodercli" else ".claude"
    extra = ", PPU profiling" if is_ppu else ""
    return (
        f"- `skills/` (also discoverable under `{root}/skills/`): `gpu-measurement`, "
        f"`runtime-records`, `KernelWiki`, and optional timeline diagnostics{extra}. "
        "Read the relevant Skill for request schemas/examples."
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


def link_runtime(workspace: Path, atrex_bench_root: Path | None = None, *, is_ppu: bool = False) -> None:
    from .agent_home import open_private_directory
    from supervisor.workspace import publish

    # Upgrade only controller-created links. Source assets are copied with the same
    # bounded, no-follow publisher used by Episode input projection.
    for name in ("tools", "skills", "reference", "reference-projects", "gpu-wiki"):
        _remove_legacy_link(workspace / name)
    publish(workspace, "tools/sandbox.py", (REPO_ROOT / "tools/sandbox.py").read_bytes())
    names = ["gpu-measurement", "runtime-records", "KernelWiki", "autonomous-gpu-kernel-timeline"]
    if is_ppu:
        names.append("ppu-acu-joint-profile")
    for name in names:
        manifest = SKILL_MANIFEST[name]
        for filename in manifest.files:
            source = REPO_ROOT / manifest.root / filename
            publish(workspace, f"skills/{name}/{filename}", source.read_bytes())
    for backend in (".claude", ".qoder", ".agents"):
        _remove_legacy_link(workspace / backend / "agents")
        root = workspace / backend / "skills"
        with open_private_directory(root):
            pass
        for path in root.iterdir():
            if path.is_symlink() and path.resolve() == (workspace / "skills" / path.name).resolve():
                continue
            _remove_legacy_link(path)
        for name in names:
            path = root / name
            target = workspace / "skills" / name
            if path.is_symlink() and path.resolve() == target.resolve():
                continue
            if path.exists() or path.is_symlink():
                raise ValueError(f"Skill discovery path is not a managed link: {path}")
            os.symlink(target, path)
    if atrex_bench_root is not None:
        _install_atrex_bench_runtime(workspace, atrex_bench_root)
    gi = workspace / ".gitignore"
    from .session_tail import read_regular_bytes
    try:
        existing = read_regular_bytes(gi, limit=1024 * 1024).decode()
    except FileNotFoundError:
        existing = ""
    ignores = ("/tools", "/skills", "/reference", "/reference-projects", "/gpu-wiki", "/.claude", "/.qoder", "/.agents", "/atrex-bench", "/" + STALL_STATE_FILE)
    missing = [name for name in ignores if name not in existing.splitlines()]
    if missing:
        publish(workspace, ".gitignore", (existing.rstrip() + "\n" + "\n".join(missing) + "\n").encode())
