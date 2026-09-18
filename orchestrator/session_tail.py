"""Bounded, no-follow incremental reads of Agent-owned JSONL transcripts."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
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


@dataclass(frozen=True)
class CaptureLimits:
    file_bytes: int = 64 * 1024 * 1024
    total_bytes: int = 128 * 1024 * 1024
    line_bytes: int = 2 * 1024 * 1024
    files: int = 4096
    records: int = 200_000

    def __post_init__(self):
        if min(self.file_bytes, self.total_bytes, self.line_bytes, self.files, self.records) <= 0:
            raise ValueError("Capture limits must be positive")


class CaptureBudget:
    def __init__(self, limits: CaptureLimits, warn):
        self.limits = limits
        self.warn = warn
        self.used = 0
        self.records = 0

    def record(self) -> bool:
        if self.records >= self.limits.records:
            self.warning("session_records_exceeded")
            return False
        self.records += 1
        return True

    def warning(self, reason: str):
        self.warn("capture_limit", ValueError(reason))

    def retain(self, size: int, *, file_used: int) -> bool:
        if size > self.limits.file_bytes - file_used:
            self.warning("per_file_bytes_exceeded")
            return False
        if size > self.limits.total_bytes - self.used:
            self.warning("session_bytes_exceeded")
            return False
        self.used += size
        return True


class TranscriptTail:
    def __init__(self):
        # Absolute file position and invocation-local capture usage are distinct.
        self.offset = 0
        self.captured_bytes = 0
        self.identity = None
        self.pending = bytearray()
        self.dropping_line = False
        self.incomplete = False
        self.stopped = False
        self.invalidated = False
        self.history_incomplete = False

    def resume_history(self, path: Path, budget: CaptureBudget, previous_size: int):
        """Parse a bounded prefix without charging or stopping the live tail."""
        if self.offset or self.captured_bytes or self.identity is not None:
            raise ValueError("Resume history must be initialized before live capture")
        # Even if the scan is capped or fails, never replay old bytes as new output.
        self.offset = previous_size
        history = TranscriptTail()
        try:
            yield from history.read(path, budget, through=previous_size, final=True)
        except (OSError, ValueError):
            self.stopped = True
            raise
        finally:
            self.identity = history.identity
            self.invalidated = history.invalidated
            self.history_incomplete = history.stopped or history.incomplete or self.stopped
            self.dropping_line = history.dropping_line
            self.stopped = self.stopped or history.invalidated

    def read(
        self, path: Path, budget: CaptureBudget, *, through: int | None = None, final: bool = False
    ):
        """Yield complete lines, reading only newly appended bytes. Never rewind."""
        if self.stopped:
            return
        with _open_regular(path) as stream:
            stat = os.fstat(stream.fileno())
            identity = (stat.st_dev, stat.st_ino)
            if self.identity not in (None, identity) or stat.st_size < self.offset:
                self.stopped = True
                self.invalidated = True
                budget.warn(
                    "native_capture", ValueError(f"transcript replaced or truncated: {path.name}")
                )
                return
            self.identity = identity
            end = stat.st_size if through is None else min(through, stat.st_size)
            if through is not None and through > stat.st_size:
                self.stopped = True
                self.invalidated = True
                budget.warn(
                    "native_capture", ValueError(f"resume transcript truncated: {path.name}")
                )
                return
            stream.seek(self.offset)
            while self.offset < end:
                available = min(
                    budget.limits.file_bytes - self.captured_bytes,
                    budget.limits.total_bytes - budget.used,
                )
                if available <= 0:
                    budget.warning(
                        "per_file_bytes_exceeded"
                        if self.captured_bytes >= budget.limits.file_bytes
                        else "session_bytes_exceeded"
                    )
                    self.stopped = True
                    self.pending.clear()
                    return
                chunk = stream.read(min(256 * 1024, available, end - self.offset))
                if not chunk:
                    self.stopped = True
                    self.invalidated = True
                    budget.warn(
                        "native_capture", ValueError(f"transcript shrank during read: {path.name}")
                    )
                    break
                self.offset += len(chunk)
                self.captured_bytes += len(chunk)
                budget.used += len(chunk)
                self.pending.extend(chunk)
                consumed = 0
                while True:
                    newline = self.pending.find(b"\n", consumed)
                    if self.dropping_line:
                        if newline < 0:
                            consumed = len(self.pending)
                            break
                        self.dropping_line = False
                        consumed = newline + 1
                        continue
                    if newline < 0:
                        if len(self.pending) - consumed > budget.limits.line_bytes:
                            self.dropping_line = True
                            self.incomplete = True
                            budget.warning("native_line_bytes_exceeded")
                            consumed = len(self.pending)
                        break
                    if newline + 1 - consumed > budget.limits.line_bytes:
                        self.incomplete = True
                        budget.warning("native_line_bytes_exceeded")
                        consumed = newline + 1
                        continue
                    if not budget.record():
                        self.stopped = True
                        self.pending.clear()
                        return
                    yield bytes(self.pending[consumed : newline + 1])
                    consumed = newline + 1
                if consumed:
                    del self.pending[:consumed]
            if final and self.pending:
                if budget.record():
                    yield bytes(self.pending)
                else:
                    self.stopped = True
                self.pending.clear()
