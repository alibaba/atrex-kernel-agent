"""Bounded, regular-file-only reads for operator-owned plugin dependencies."""
from __future__ import annotations

import hashlib
import os
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_ENTRIES = 16_384
MAX_DEPTH = 64
MAX_DOCUMENT_BYTES = 1024 * 1024
CHUNK_BYTES = 64 * 1024
IGNORED_DIRECTORIES = frozenset({".git", "__pycache__", ".pytest_cache"})


@dataclass
class FileBudget:
    """One budget for the entire catalog, including overlapping dependencies."""

    total_bytes: int = 0
    entries: int = 0

    def visit(self) -> None:
        self.entries += 1
        if self.entries > MAX_ENTRIES:
            raise ValueError(
                f"Plugin scan exceeds {MAX_ENTRIES} entries/metadata reads; "
                "narrow declared resources"
            )


@contextmanager
def open_regular(path: Path):
    """Pin each path component; O_NONBLOCK prevents FIFO opens from waiting."""
    path = path.absolute()
    directory = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(
            path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
        )
    finally:
        os.close(directory)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"Plugin input is not a regular file: {path}")
        yield descriptor
    finally:
        os.close(descriptor)


def _chunks(descriptor: int, budget: FileBudget, limit: int):
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("Plugin input is not a regular file")
    if before.st_size > limit:
        raise ValueError(f"Plugin file exceeds {limit} bytes; split or narrow the dependency")
    if before.st_size > MAX_TOTAL_BYTES - budget.total_bytes:
        raise ValueError(
            f"Plugin scan exceeds {MAX_TOTAL_BYTES} total bytes; narrow declared resources"
        )
    used = 0
    while True:
        remaining = min(limit - used, MAX_TOTAL_BYTES - budget.total_bytes)
        chunk = os.read(descriptor, min(CHUNK_BYTES, remaining + 1))
        if not chunk:
            break
        if len(chunk) > remaining:
            raise ValueError("Plugin input grew beyond its file/total byte budget during the read")
        used += len(chunk)
        budget.total_bytes += len(chunk)
        yield chunk
    after = os.fstat(descriptor)
    if (used != before.st_size or
            (before.st_size, before.st_mtime_ns, before.st_ctime_ns) !=
            (after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
        raise ValueError("Plugin input changed during the read; stop updates and retry")


def read_text(path: Path, *, budget: FileBudget | None = None) -> str:
    budget = budget if budget is not None else FileBudget()
    budget.visit()
    with open_regular(path) as descriptor:
        return b"".join(_chunks(descriptor, budget, MAX_DOCUMENT_BYTES)).decode("utf-8")


def _file_digest(descriptor: int, budget: FileBudget) -> bytes:
    digest = hashlib.sha256()
    for chunk in _chunks(descriptor, budget, MAX_FILE_BYTES):
        digest.update(chunk)
    return digest.digest()


def _directory_digest(descriptor: int, budget: FileBudget, depth: int) -> bytes:
    if depth > MAX_DEPTH:
        raise ValueError(f"Plugin tree exceeds {MAX_DEPTH} directory levels")
    # Count while enumerating, before sorting/allocating a whole large directory.
    entries = []
    with os.scandir(descriptor) as scan:
        for entry in scan:
            budget.visit()
            if entry.name in IGNORED_DIRECTORIES or entry.name.endswith((".pyc", ".pyo")):
                continue
            entries.append((entry.name, entry.stat(follow_symlinks=False).st_mode))
    digest = hashlib.sha256(b"atrex-plugin-tree-v2\0")
    for name, mode in sorted(entries):
        encoded = os.fsencode(name)
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        if stat.S_ISDIR(mode):
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            try:
                digest.update(b"d" + _directory_digest(child, budget, depth + 1))
            finally:
                os.close(child)
        elif stat.S_ISREG(mode):
            child = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
            try:
                digest.update(b"f" + _file_digest(child, budget))
            finally:
                os.close(child)
        else:
            raise ValueError(
                f"Plugin dependency must be a regular file/directory, "
                f"not a symlink or special file: {name}"
            )
    return digest.digest()


def tree_digest(root: Path, *, budget: FileBudget | None = None) -> str:
    """Stream content hashes with a shared budget; never use an mtime-only cache.

    Declared roots may be operator symlinks; entries inside those trees may not.
    Missing optional roots retain a stable sentinel. Unreadable trees fail closed.
    """
    budget = budget if budget is not None else FileBudget()
    budget.visit()
    root = root.resolve()
    try:
        mode = root.lstat().st_mode
    except FileNotFoundError:
        return "missing"
    if stat.S_ISREG(mode):
        with open_regular(root) as descriptor:
            return _file_digest(descriptor, budget).hex()
    if not stat.S_ISDIR(mode):
        raise ValueError(f"Plugin dependency is not a regular file/directory: {root}")
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        return _directory_digest(descriptor, budget, 0).hex()
    finally:
        os.close(descriptor)
