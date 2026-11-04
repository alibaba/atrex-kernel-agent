"""No-follow file reads used by Workflow, Workspace and Journal consumers."""

from __future__ import annotations

import os
import stat
from pathlib import Path


def _open_regular(path: Path):
    """Open a regular transcript without following symlinks in any component."""
    path = path.absolute()
    directory = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            child = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory,
            )
            os.close(directory)
            directory = child
        fd = os.open(
            path.name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=directory,
        )
    finally:
        os.close(directory)
    stream = os.fdopen(fd, "rb")
    try:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError(f"Not a regular transcript: {path}")
    except BaseException:
        stream.close()
        raise
    return stream


def read_regular_bytes(path: Path, limit: int = -1) -> bytes:
    """Read at most limit bytes from a regular file, refusing symlinks in its path.

    A negative limit reads the whole file. Size validation belongs to the caller:
    read one byte beyond the allowed size to detect an oversized input.
    """
    with _open_regular(path) as stream:
        return stream.read(limit)
