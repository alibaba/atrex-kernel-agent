"""Versioned read-only projection for archive and CI consumers.

Version 1 exposes retained memory documents and completed episode facts. It does
not authorize accepting candidates, modifying checkpoints or resuming execution.
Documents preserve the existing memory schema; consumers must reject unsupported
schema versions rather than infer source paths from a private runtime layout.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class VersionSnapshot:
    version: int
    memory: dict[str, Any] | None
    committed_revision: str


@dataclass(frozen=True)
class WorkflowExportSnapshot:
    schema_version: int
    framework: str
    versions: tuple[VersionSnapshot, ...]
    journals: tuple[dict[str, Any], ...]
    attempts: tuple[dict[str, Any], ...]
    baseline: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[dict[str, Any], ...] = ()
    plans: dict[int, str] = field(default_factory=dict)
    sessions: tuple[dict[str, Any], ...] = ()
    progress: dict[str, Any] = field(default_factory=dict)
    checkpoint_ready: bool = False


class WorkflowExportReader(Protocol):
    def read(self, workspace: Path) -> WorkflowExportSnapshot:
        """Return one detached snapshot; incomplete current writes may have no memory."""
        ...
