"""Bounded no-follow reads, independent of session capture."""
from __future__ import annotations

import os
import stat
from pathlib import Path


def read_regular_bytes(path: Path, limit: int) -> bytes:
    if limit <= 0:
        raise ValueError("A positive read limit is required")
    path = Path(os.path.abspath(path))
    directory = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                             dir_fd=directory)
    finally:
        os.close(directory)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError(f"Not a regular file: {path.name}")
        return stream.read(limit)
