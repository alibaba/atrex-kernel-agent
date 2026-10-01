"""Bounded snapshots and no-follow publication across the HTTP trust boundary."""
from __future__ import annotations

import os
import stat
import uuid
from pathlib import Path

from orchestrator.agent_home import open_private_directory
from orchestrator.session_tail import read_regular_bytes

MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
SKIP = {".git", ".claude", ".codex", ".qoder", ".qodersec", ".pi", ".agents",
        ".atrex_long_horizon", ".atrex_plugins", ".gpu_wiki_profile", ".atrex_environment",
        "atrex-bench", "tools", "skills", "reference", "reference-projects",
        "gpu-wiki", "__pycache__", ".venv", "node_modules", "plans"}


def relative_path(value: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ValueError("Use a non-empty workspace-relative path without '..'")
    if path.parts[0] in {".git", ".claude", ".codex", ".qoder", ".agents",
                          ".atrex_long_horizon", ".atrex_plugins", ".atrex_environment", "atrex-bench"}:
        raise ValueError("Private/control paths cannot be used as request files")
    return path


class InputSizeLimitError(ValueError):
    """A regular workspace file exceeded its bounded read budget."""


def read_input(workspace: Path, value: str, *, limit: int = MAX_FILE_BYTES) -> bytes:
    path = relative_path(value)
    data = read_regular_bytes(workspace / path, limit=limit + 1)
    if len(data) > limit:
        raise InputSizeLimitError("Request input file exceeds the size limit")
    return data


def snapshot(source: Path, destination: Path) -> None:
    """Copy regular candidate files only; no links to arbitrary host paths."""
    count = total = 0
    for current, dirs, files in os.walk(source, followlinks=False):
        dirs[:] = [name for name in dirs if name not in SKIP and not (Path(current) / name).is_symlink()]
        for name in files:
            path = Path(current) / name
            relative = path.relative_to(source)
            if relative.parts[0] in SKIP or path.suffix in {".pyc", ".ncu-rep", ".sqlite", ".jsonl"}:
                # SOL workload is a required public execution contract.
                if relative != Path("workload.jsonl"):
                    continue
            if relative.parts[0] == "profiles" and path.suffix not in {".py", ".sh", ".json", ".cu", ".cuh"}:
                continue
            if not stat.S_ISREG(path.lstat().st_mode):
                continue
            data = read_input(source, relative.as_posix())
            count, total = count + 1, total + len(data)
            if count > 4096 or total > MAX_TOTAL_BYTES:
                raise ValueError("GPU input snapshot exceeds 4096 files / 64 MiB; remove temporary files")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)


def publish(workspace: Path, relative: str, data: bytes, *, append: bool = False) -> None:
    """Write using pinned directory descriptors, not Agent-controlled symlinks."""
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Unsafe publication path")
    with open_private_directory(workspace / path.parent) as parent:
        if append:
            descriptor = os.open(path.name, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW,
                                 0o600, dir_fd=parent)
            with os.fdopen(descriptor, "ab") as output:
                output.write(data)
            return
        temporary = ".runtime-" + uuid.uuid4().hex
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=parent)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(data)
            os.replace(temporary, path.name, src_dir_fd=parent, dst_dir_fd=parent)
        finally:
            try:
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass
