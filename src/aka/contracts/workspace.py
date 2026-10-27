"""Source preparation and candidate operations independent of the application."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Any, Mapping
from types import MappingProxyType

from .content import SourceContent, WorkspaceFile
from .problem import InitialSource
from .filesystem import FilesystemAccess


@dataclass(frozen=True)
class RevisionContent:
    """Committed path bytes; None represents a path absent at that revision."""
    files: Mapping[str, bytes | None]

    def __post_init__(self):
        files = dict(self.files)
        if any(value is not None and not isinstance(value, bytes) for value in files.values()):
            raise TypeError("revision content must contain immutable bytes or None")
        object.__setattr__(self, "files", MappingProxyType(files))


def _content_identities(values):
    identities = dict(values)
    if any(not isinstance(path, str) or (identity is not None and not isinstance(identity, str))
           for path, identity in identities.items()):
        raise TypeError("content identities require string paths and opaque strings or None")
    return MappingProxyType(identities)


@dataclass(frozen=True)
class RevisionFacts:
    revision: str
    parents: tuple[str, ...]
    message: str
    committed_at: int
    content_identities: Mapping[str, str | None]

    def __post_init__(self):
        if not isinstance(self.revision, str) or not self.revision:
            raise ValueError("revision identity must be a nonempty opaque string")
        parents = tuple(self.parents)
        if any(not isinstance(parent, str) or not parent for parent in parents):
            raise ValueError("parent identities must be nonempty opaque strings")
        if not isinstance(self.message, str) or type(self.committed_at) is not int:
            raise TypeError("revision facts require a message and integer commit time")
        object.__setattr__(self, "parents", parents)
        object.__setattr__(self, "content_identities", _content_identities(self.content_identities))


@dataclass(frozen=True)
class WorkingFacts:
    changed_paths: tuple[str, ...]
    content_identities: Mapping[str, str | None]

    def __post_init__(self):
        paths = tuple(self.changed_paths)
        if any(not isinstance(path, str) for path in paths):
            raise TypeError("working changed paths must be strings")
        object.__setattr__(self, "changed_paths", paths)
        object.__setattr__(self, "content_identities", _content_identities(self.content_identities))


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
class CandidateBindingFacts:
    registered: bool
    branch: str | None
    revision: str | None


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
    def current_revision(self, workspace: Path) -> str:
        """Return the opaque current revision, or an empty string before initialization."""
        ...
    def read_working(self, workspace: Path, paths: tuple[str, ...]) -> RevisionContent:
        """Read selected working bytes without following links or interpreting content."""
        ...
    def revision_history(self, workspace: Path, *, revision: str | None = None,
                         paths: tuple[str, ...] = (),
                         changed_path: str | None = None) -> tuple[RevisionFacts, ...]:
        """Return reachable history in stable newest-first order, with explicit path identities."""
        ...
    def working_facts(self, workspace: Path, *, paths: tuple[str, ...] = ()) -> WorkingFacts:
        """Read index/worktree changes and explicit working content identities without mutation."""
        ...
    def restore_revision(self, workspace: Path, *, expected_revision: str,
                         revision: str, retain_working_files: bool) -> str:
        """Restore an explicit revision with an incumbent check and transactional rollback.

        Retaining working files restores only revision/index; otherwise the
        implementation restores tracked files using its original untracked rules.
        """
        ...
    def record(self, workspace: Path, *, expected_revision: str,
               files: tuple[WorkspaceFile, ...], message: str) -> str:
        """Record explicit owner files while preserving unrelated working/index content.

        Precommit failure restores partial writes; a committed result survives a
        lost return and remains inspectable through the public revision methods.
        """
        ...
    def filesystem_access(self, workspace: Path, *, include_git: bool = False) -> FilesystemAccess:
        """Describe immutable inputs; reject Git grants unsupported by this implementation."""
        ...
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
                          runtime_files: tuple[WorkspaceFile, ...],
                          context_directories: tuple[str, ...] = ()) -> int:
        """Initialize explicit source and constraints; validate unchanged V0 on retry."""
        ...
    def prepare_sol(self, workspace: Path, content: SourceContent, *,
                    problem_files: tuple[WorkspaceFile, ...], harness: WorkspaceFile,
                    runtime_files: tuple[WorkspaceFile, ...],
                    context_directories: tuple[str, ...] = ()) -> str:
        """Install and commit V0, or validate unchanged V0 on measurement retry."""
        ...
    def baseline_changes(self, workspace: Path, baseline_commit: str, *,
                         required_paths: tuple[str, ...] = ()) -> list[str]:
        """Inspect immutable content; required paths must have baseline identities."""
        ...
    def restore_baseline(self, workspace: Path, baseline_commit: str) -> list[str]:
        """Restore baseline content on request; return only successfully restored paths."""
        ...
    def plan(self, workspace: Path, episode: int, base_commit: str, root: Path | None = None) -> CandidateHandle: ...
    def materialize(self, handle: CandidateHandle, workspace: Path) -> None: ...
    def inspect_candidate(self, workspace: Path, handle: CandidateHandle) -> CandidateBindingFacts:
        """Read actual registration, branch and revision; damaged bindings raise."""
        ...
    def validate_candidate(self, handle: CandidateHandle, candidate_commit: str) -> tuple[str, list[str]]: ...
    def archive(self, handle: CandidateHandle, destination: Path, candidate_commit: str = "HEAD") -> Path: ...
    def remove(self, handle: CandidateHandle, workspace: Path) -> None: ...
    def promote(self, workspace: Path, *, base_commit: str, candidate_commit: str, episode: int,
                record_files: tuple[WorkspaceFile, ...] = (),
                evidence: dict[str, Any] | None = None, memory_version: int | None = None,
                memory_record: dict[str, Any] | None = None) -> str:
        """Commit candidate and opaque owner-prepared files in one transaction.

        Legacy record keywords are retained only by the compatibility adapter.
        """
        ...
    def read_revision(self, workspace: Path, revision: str,
                      changed_paths: list[str]) -> RevisionContent:
        """Read committed bytes without creating snapshot files or modifying Git."""
        ...

