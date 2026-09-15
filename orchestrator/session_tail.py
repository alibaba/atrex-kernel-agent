"""Bounded, no-follow incremental reads of Agent-owned JSONL transcripts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .agent_workspace import _open_regular


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
        self.offset = 0
        self.identity = None
        self.pending = bytearray()
        self.stopped = False

    def read(self, path: Path, budget: CaptureBudget, *, through: int | None = None,
             final: bool = False):
        """Yield complete lines, reading only newly appended bytes. Never rewind."""
        if self.stopped:
            return
        with _open_regular(path) as stream:
            stat = os.fstat(stream.fileno())
            identity = (stat.st_dev, stat.st_ino)
            if self.identity not in (None, identity) or stat.st_size < self.offset:
                self.stopped = True
                budget.warn("native_capture", ValueError(f"transcript replaced or truncated: {path.name}"))
                return
            self.identity = identity
            end = stat.st_size if through is None else min(through, stat.st_size)
            if through is not None and through > stat.st_size:
                self.stopped = True
                budget.warn("native_capture", ValueError(f"resume transcript truncated: {path.name}"))
                return
            stream.seek(self.offset)
            while self.offset < end:
                available = min(budget.limits.file_bytes - self.offset,
                                budget.limits.total_bytes - budget.used)
                if available <= 0:
                    budget.warning("per_file_bytes_exceeded" if self.offset >= budget.limits.file_bytes
                                   else "session_bytes_exceeded")
                    self.stopped = True
                    self.pending.clear()
                    return
                chunk = stream.read(min(256 * 1024, available, end - self.offset))
                if not chunk:
                    self.stopped = True
                    budget.warn("native_capture", ValueError(f"transcript shrank during read: {path.name}"))
                    break
                self.offset += len(chunk)
                budget.used += len(chunk)
                self.pending.extend(chunk)
                consumed = 0
                while True:
                    newline = self.pending.find(b"\n", consumed)
                    if newline < 0:
                        break
                    if newline + 1 - consumed > budget.limits.line_bytes:
                        break
                    if not budget.record():
                        self.stopped = True
                        self.pending.clear()
                        return
                    yield bytes(self.pending[consumed:newline + 1])
                    consumed = newline + 1
                if consumed:
                    del self.pending[:consumed]
                if len(self.pending) > budget.limits.line_bytes:
                    self.stopped = True
                    self.pending.clear()
                    budget.warning("native_line_bytes_exceeded")
                    return
            if final and self.pending:
                if budget.record():
                    yield bytes(self.pending)
                self.pending.clear()
