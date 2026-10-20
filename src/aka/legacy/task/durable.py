"""Crash-durable filesystem transitions for recovery protocol state."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path


def fsync_directory(directory: Path) -> None:
    """Persist prior directory-entry changes on supported POSIX filesystems."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def ensure_private_directory(directory: Path) -> None:
    """Create a private directory tree and persist every new directory entry."""
    missing: list[Path] = []
    current = directory
    while not current.exists():
        missing.append(current)
        current = current.parent
    directory.mkdir(parents=True, exist_ok=True)
    for created in reversed(missing):
        created.chmod(0o700)
        fsync_directory(created)
        fsync_directory(created.parent)
    directory.chmod(0o700)


def durable_write_text(path: Path, value: str, *, mode: int = 0o600) -> None:
    """Atomically replace a text file and persist its data and directory entry."""
    ensure_private_directory(path.parent)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            stream.write(value)
            stream.flush()
            os.fchmod(stream.fileno(), mode)
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def durable_write_json(
    path: Path,
    value: object,
    *,
    indent: int | None = None,
    ensure_ascii: bool = True,
) -> None:
    separators: tuple[str, str] | None = None
    if indent is None:
        separators = (",", ":")
    payload = json.dumps(
        value,
        indent=indent,
        ensure_ascii=ensure_ascii,
        separators=separators,
        sort_keys=indent is None,
    )
    durable_write_text(path, payload + "\n")


def durable_replace(source: Path, destination: Path) -> None:
    """Atomically move a file and persist both affected directories."""
    source_parent = source.parent
    destination_parent = destination.parent
    os.replace(source, destination)
    fsync_directory(destination_parent)
    if source_parent != destination_parent:
        fsync_directory(source_parent)


def durable_unlink(path: Path, *, missing_ok: bool = False) -> bool:
    """Remove a file and persist the removal. Return whether a file was removed."""
    try:
        path.unlink()
    except FileNotFoundError:
        if missing_ok:
            return False
        raise
    fsync_directory(path.parent)
    return True


def durable_rmdir(path: Path) -> None:
    """Remove an empty directory and persist the removal."""
    path.rmdir()
    fsync_directory(path.parent)
