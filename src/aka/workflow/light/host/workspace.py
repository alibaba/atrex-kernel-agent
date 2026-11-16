"""Bounded snapshots and no-follow publication across the HTTP trust boundary."""
from __future__ import annotations

import os
import stat
import uuid
from pathlib import Path

from aka.agent.execution.home import open_private_directory
from aka.agent.session_tail import read_regular_bytes

MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
SKIP = {".git", ".claude", ".codex", ".qoder", ".qodersec", ".pi", ".agents",
        ".aka-agent-observations",
        ".atrex_versions", ".atrex_plugins", ".gpu_wiki_profile", ".atrex_environment",
        "atrex-bench", "tools", "skills", "reference", "reference-projects", "references",
        "gpu-wiki", "records", "research-library", ".atrex_request", ".aka_plugins", "__pycache__", ".venv", "node_modules", "plans"}


def relative_path(value: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ValueError("Use a non-empty workspace-relative path without '..'")
    if path.parts[0] in {".git", ".claude", ".codex", ".qoder", ".agents",
                          ".aka-agent-observations",
                          ".atrex_versions", ".atrex_plugins", ".atrex_environment", "atrex-bench"}:
        raise ValueError("Private/control paths cannot be used as request files")
    if path.parts[0] == "references":
        raise ValueError("Reference knowledge cannot be used as a GPU request file")
    return path


class InputSizeLimitError(ValueError):
    """A regular workspace file exceeded its bounded read budget."""


def read_input(workspace: Path, value: str, *, limit: int = MAX_FILE_BYTES) -> bytes:
    path = relative_path(value)
    data = read_regular_bytes(workspace / path, limit=limit + 1)
    if len(data) > limit:
        raise InputSizeLimitError("Request input file exceeds the size limit")
    return data


def snapshot(source: Path, destination: Path, *, custom_tools: bool = False, filesystem_access=None) -> None:
    """Copy regular candidate files only; no links to arbitrary host paths."""
    count = total = 0
    skipped = SKIP - {"tools"} if custom_tools else SKIP
    for current, dirs, files in os.walk(source, followlinks=False):
        dirs[:] = [name for name in dirs if name not in skipped and not (Path(current) / name).is_symlink()]
        for name in files:
            path = Path(current) / name
            relative = path.relative_to(source)
            if relative.as_posix() in {"tools/sandbox.py", "tools/plugin.py"}:
                continue
            if relative.parts[0] in skipped or path.suffix in {".pyc", ".ncu-rep", ".sqlite", ".jsonl"}:
                # SOL workload is a required public execution contract.
                if relative != Path("workload.jsonl"):
                    continue
            if relative.parts[0] == "profiles" and path.suffix not in {".py", ".sh", ".json", ".cu", ".cuh"}:
                continue
            if not stat.S_ISREG(path.lstat().st_mode):
                continue
            from aka.contracts.filesystem_access import require_path_access
            require_path_access(filesystem_access, path)
            data = read_input(source, relative.as_posix())
            count, total = count + 1, total + len(data)
            if count > 4096 or total > MAX_TOTAL_BYTES:
                raise ValueError("GPU input snapshot exceeds 4096 files / 64 MiB; remove temporary files")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            if relative.parts[0] == "tools" and path.stat(follow_symlinks=False).st_mode & 0o111:
                target.chmod(0o700)


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


def publish_if_changed(workspace: Path, relative: str, data: bytes) -> bool:
    """Atomically publish immutable projection bytes only when they changed.

    Agent-visible research history can contain thousands of immutable records
    and large archived references.  Avoid replacing every file at every
    Episode bind while still healing deleted, modified, linked, or malformed
    projections from the authoritative private stores.
    """
    try:
        current = read_regular_bytes(workspace / relative_path(relative), limit=len(data) + 1)
    except (FileNotFoundError, OSError, ValueError):
        current = None
    if current == data:
        return False
    publish(workspace, relative, data)
    return True


def publish_pinned_if_changed(workspace: Path, relative: str, data: bytes) -> bool:
    """Refresh a file without replacing its inode when one already exists.

    Bubblewrap can overlay a single Agent-visible file read-only, but a file bind
    mount pins the existing inode.  Runtime projections such as ``PROGRESS.md``
    must therefore retain that inode so Supervisor updates remain visible inside
    a long-lived Agent namespace.  Missing, linked, or otherwise invalid paths
    are still healed with the normal atomic publisher before the next sandbox
    launch; an established regular projection is updated in place.
    """
    path = relative_path(relative)
    try:
        current = read_regular_bytes(workspace / path, limit=len(data) + 1)
    except (FileNotFoundError, OSError, ValueError):
        current = None
    if current == data:
        return False

    with open_private_directory(workspace / path.parent) as parent:
        try:
            descriptor = os.open(
                path.name,
                os.O_WRONLY | os.O_NOFOLLOW,
                dir_fd=parent,
            )
        except OSError:
            # Creation and recovery from an Agent-created link/special path use
            # the existing bounded atomic publication path.  Once Bubblewrap is
            # active the read-only mount prevents either condition from arising.
            publish(workspace, relative, data)
            return True
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                os.close(descriptor)
                descriptor = -1
                publish(workspace, relative, data)
                return True
            os.ftruncate(descriptor, 0)
            with os.fdopen(descriptor, "wb") as output:
                descriptor = -1
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    return True
