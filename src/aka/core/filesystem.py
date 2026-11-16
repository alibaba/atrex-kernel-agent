"""Public filesystem primitives with caller-owned publication and size policies."""
from __future__ import annotations

from contextlib import contextmanager, suppress
import json
import os
from pathlib import Path
import stat
import uuid


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


@contextmanager
def open_private_directory(path):
    """Open or create a directory chain without following links."""
    path = Path(os.path.abspath(path))
    descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for name in path.parts[1:]:
            with suppress(FileExistsError):
                os.mkdir(name, mode=0o700, dir_fd=descriptor)
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


def _open_regular(path):
    path = Path(path).absolute()
    directory = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    finally:
        os.close(directory)
    stream = os.fdopen(descriptor, "rb")
    try:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("Filesystem input must be a regular file")
    except BaseException:
        stream.close()
        raise
    return stream


def read_regular_bytes(path, limit=-1):
    """Read a bounded prefix; callers detect oversize by requesting limit + 1.

    A limit of -1 reads the whole file. This performs no size-policy decision
    and never follows links in the file or its parent components.
    """
    with _open_regular(path) as stream:
        return stream.read(limit)
