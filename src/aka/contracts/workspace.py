"""Source preparation and candidate operations independent of the application."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Any

from .content import SourceContent, WorkspaceFile
from .problem import InitialSource


@dataclass(frozen=True)
class SourceSnapshot:
    paths: tuple[str, ...]
    sha256: str


@dataclass(frozen=True)
class SourceRequest:
    kind: str = "kernel"
    kernel_demo: str = ""
    operator_dir: str = ""
    name: str = ""
    framework: str = ""
    platform: str = ""
    gpu_wiki: str = ""
    definition: dict | None = None
    initial: InitialSource | None = None


class SourceProvider(Protocol):
    def prepare(self, request: SourceRequest | None = None) -> SourceContent:
        """Describe initial source files without writing an agent workspace."""
        ...


@dataclass(frozen=True)
class CandidateHandle:
    episode: int
    base_commit: str
    branch: str
    path: Path


@dataclass(frozen=True)
class CandidateWorkspaceSnapshot:
    """Opaque stopped Task workspace/Git checkpoint; only original paths restore."""
    directory: Path
    identity: str


@dataclass(frozen=True)
class CandidateCheckpointState:
    workspace: Path
    revision: str
    candidates: tuple[CandidateHandle, ...] = ()
    candidate_revisions: tuple[str, ...] = ()


class CandidateWorkspace(Protocol):
    def export_checkpoint(self, workspace: Path, destination: Path, *,
                          candidates: tuple[CandidateHandle, ...] = ()) -> CandidateWorkspaceSnapshot: ...
    def validate_checkpoint(self, snapshot: CandidateWorkspaceSnapshot, *,
                            check_destinations: bool = False) -> CandidateCheckpointState:
        """Read-only validation of stopped Git/worktree bytes and original path bindings."""
        ...
    def checkpoint_files(self, snapshot: CandidateWorkspaceSnapshot, paths: tuple[str, ...]) -> dict[str, bytes]:
        """Read validated canonical workspace files; reject traversal, Git metadata and symlinks."""
        ...
    def restore_checkpoint(self, snapshot: CandidateWorkspaceSnapshot) -> CandidateCheckpointState:
        """Restore only original paths after all modules pass the caller's validation phase."""
        ...
    def install_files(self, workspace: Path, files: tuple[WorkspaceFile, ...], *,
                      immutable: bool = False) -> SourceSnapshot:
        """Copy selected files; immutable preflights all targets and rejects changed bytes/modes."""
        ...
    def initialize_kernel(self, workspace: Path, content: SourceContent, *,
                          runtime_files: tuple[WorkspaceFile, ...]) -> int:
        """Initialize explicit source and constraints; validate unchanged V0 on retry."""
        ...
    def prepare_sol(self, workspace: Path, content: SourceContent, *,
                    problem_files: tuple[WorkspaceFile, ...], harness: WorkspaceFile,
                    runtime_files: tuple[WorkspaceFile, ...]) -> str:
        """Install and commit V0, or validate unchanged V0 on measurement retry."""
        ...
    def baseline_changes(self, workspace: Path, baseline_commit: str) -> list[str]:
        """Inspect immutable baseline content without modifying the workspace."""
        ...
    def restore_baseline(self, workspace: Path, baseline_commit: str) -> list[str]:
        """Restore baseline content on request; return only successfully restored paths."""
        ...
    def plan(self, workspace: Path, episode: int, base_commit: str, root: Path | None = None) -> CandidateHandle: ...
    def materialize(self, handle: CandidateHandle, workspace: Path) -> None: ...
    def validate_candidate(self, handle: CandidateHandle, candidate_commit: str) -> tuple[str, list[str]]: ...
    def archive(self, handle: CandidateHandle, destination: Path, candidate_commit: str = "HEAD") -> Path: ...
    def remove(self, handle: CandidateHandle, workspace: Path) -> None: ...
    def promote(self, workspace: Path, *, base_commit: str, candidate_commit: str, episode: int,
                evidence: dict[str, Any], memory_version: int, memory_record: dict[str, Any]) -> str: ...
    def write_revision(self, workspace: Path, revision: str, changed_paths: list[str],
                       destination: Path, label: str) -> dict[str, str | None]: ...

