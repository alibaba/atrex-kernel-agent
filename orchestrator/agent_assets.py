"""Minimal Agent assets, sealed outside the writable workspace."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

HTTP_CLIENT_FILES = ("sandbox.py",)
SKILL_PATHS = {
    "gpu-measurement": "orchestrator/agent_skills/gpu-measurement",
    "runtime-records": "orchestrator/agent_skills/runtime-records",
    "KernelWiki": "skills/KernelWiki",
    "ncu-report-skill": "3rdparty/ncu-report-skill",
    "autonomous-gpu-kernel-timeline": "skills/autonomous-gpu-kernel-timeline",
    "ppu-acu-joint-profile": "skills/ppu-acu-joint-profile",
}
REQUIRED_AGENT_SKILLS = ("gpu-measurement", "runtime-records", "KernelWiki")
DEFAULT_AGENT_SKILLS = (*REQUIRED_AGENT_SKILLS, "autonomous-gpu-kernel-timeline")
BACKEND_SKILL_ROOTS = (".claude", ".qoder", ".agents")
SKILL_BOUNDARY = """## Runtime execution contract

Use this Skill for domain knowledge and optional diagnostics, not as a replacement for the
Episode workflow. The injected hardware, DSL, and operator contract are authoritative; examples
are not task inputs. All GPU execution must go through `python3 tools/sandbox.py`, using typed
operations or Dev with explicit inputs. Never run a local GPU command from an example.
Keep temporary files in `scratch/`; record plans and analysis with Journal tools. Repository-specific
layouts, direct commands, and mandatory standalone reports below are reference examples only.
This contract takes precedence over conflicting Skill instructions.

"""


def resolve_agent_skills(names: tuple[str, ...]) -> tuple[str, ...]:
    """Validate the selection and retain mandatory Skills even when it is empty."""
    unknown = set(names) - SKILL_PATHS.keys()
    if unknown:
        raise ValueError(f"Unknown Agent skills: {sorted(unknown)}")
    return tuple(dict.fromkeys((*REQUIRED_AGENT_SKILLS, *names)))


def materialize_agent_assets(workspace: Path, repository: Path, names: tuple[str, ...]) -> Path:
    """Publish a content-keyed view. Existing sessions retain their immutable view."""
    from .supervisor_runtime import supervisor_campaign_root

    names = resolve_agent_skills(names)
    files = {
        f"tools/{name}": (repository / "tools" / name).read_bytes() for name in HTTP_CLIENT_FILES
    }
    executable = set()
    for name in names:
        source = repository / SKILL_PATHS[name]
        if not (source / "SKILL.md").is_file():
            raise RuntimeError(f"Selected Skill is not installed: {name} ({source})")
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source)
            if any(part in {".git", "__pycache__", ".pytest_cache"} for part in relative.parts):
                continue
            if path.is_symlink():
                raise RuntimeError(f"Skill asset cannot be a symlink: {path}")
            if not path.is_file():
                continue
            content = path.read_bytes()
            if relative.as_posix() == "SKILL.md":
                value = content.decode("utf-8")
                end = (
                    value.index("\n---\n", 4) + 5
                    if (value.startswith("---\n") and "\n---\n" in value[4:])
                    else 0
                )
                content = (value[:end] + "\n" + SKILL_BOUNDARY + value[end:]).encode()
            files[f"skills/{name}/{relative.as_posix()}"] = content
            if path.stat().st_mode & 0o111:
                executable.add(f"skills/{name}/{relative.as_posix()}")
    digest = hashlib.sha256(json.dumps(names).encode())
    for path, content in sorted(files.items()):
        digest.update(
            path.encode() + b"\0" + hashlib.sha256(content).digest() + bytes([path in executable])
        )
    root = supervisor_campaign_root(workspace) / "agent-assets"
    destination = root / digest.hexdigest()
    if destination.is_dir():
        return destination
    root.mkdir(parents=True, mode=0o700, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".building-", dir=root))
    try:
        (staging / "skills").mkdir()
        for path, content in files.items():
            target = staging / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            target.chmod(0o755 if path in executable else 0o644)
        try:
            staging.rename(destination)
        except OSError:
            if not destination.is_dir():
                raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination


def initialize_writable_tools(workspace: Path, assets: Path) -> Path:
    """Seed an Episode-local copy once; never write through shared asset links."""
    destination = workspace / "tools"
    if destination.is_symlink():
        previous = destination.resolve()
        if previous.name != "tools" or previous.parent.parent != assets.parent:
            raise RuntimeError("Agent tools link is not a managed asset; migrate it explicitly")
    elif destination.is_dir():
        return destination  # Keep additions, modifications, and deletions on resume.
    elif destination.exists():
        raise RuntimeError("Agent tools path must be a directory")

    staging = Path(tempfile.mkdtemp(prefix=".tools-", dir=workspace))
    try:
        shutil.copytree(assets / "tools", staging, dirs_exist_ok=True)
        # Asset seeds are immutable by mount policy, not by copying that mount.
        staging.chmod(0o700)
        for path in staging.rglob("*"):
            path.chmod(path.stat().st_mode | 0o600 | (0o100 if path.is_dir() else 0))
        if destination.is_symlink():
            destination.unlink()  # Remove only the old link, never its shared target.
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination
