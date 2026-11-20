"""Source preparation and candidate operations independent of the application."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol, Any, Mapping
from types import MappingProxyType

from .content import SourceContent, WorkspaceFile
from .problem import EvaluationInput, InitialSource
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
    status_text: str | None = None

    def __post_init__(self):
        if self.status_text is not None and not isinstance(self.status_text, str):
            raise TypeError("working status text must be a string or None")
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


@dataclass(frozen=True)
class CandidateManifestRequest:
    """Frozen native candidate inputs; the host decides whether to install them."""
    evaluation_input: EvaluationInput
    campaign_name: str
    framework: str
    reference_source: bytes | None


class SourceProvider(Protocol):
    def prepare(self, request: SourceRequest | None = None) -> SourceContent:
        """Describe initial source files without writing an agent workspace."""
        ...

    def prepare_candidate_manifest(self, request: CandidateManifestRequest) -> WorkspaceFile:
        """Optional pure candidate-format contribution; never writes a workspace."""
        ...


@dataclass(frozen=True)
class WorkspaceExclusion:
    """One literal workspace-relative runtime path, with an explicit directory scope."""
    path: str
    subtree: bool = False

    def __post_init__(self):
        if not isinstance(self.path, str):
            raise TypeError("workspace exclusion path must be a string")
        path = PurePosixPath(self.path)
        if (not self.path or path.is_absolute() or path.as_posix() != self.path
                or path == PurePosixPath(".") or ".." in path.parts
                or any(char in self.path for char in "\\*?[]")
                or any(ord(char) < 32 or ord(char) == 127 for char in self.path)
                or any(part.casefold() == ".git" for part in path.parts)):
            raise ValueError("workspace exclusion must be a safe literal relative path")
        if type(self.subtree) is not bool:
            raise TypeError("workspace exclusion subtree must be a boolean")


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
class WorkspaceFilesSnapshot:
    """Opaque, validated closure of explicitly selected mutable directories."""
    directory: Path
    identity: str


@dataclass(frozen=True)
class CandidateCheckpointState:
    workspace: Path
    revision: str
    candidates: tuple[CandidateHandle, ...] = ()
    candidate_revisions: tuple[str, ...] = ()


class CandidateWorkspace(Protocol):
    def bind_source_contract(self, namespace: Path, *, file_system: str, contract: dict | None) -> None:
        """Bind host-persisted source structure for repositories in a namespace."""
        ...

    def initialize_agent(self, workspace: Path, content: SourceContent) -> str:
        """Create the initial Agent Git root once; preserve an existing repository.

        Only source and .gitignore enter the initial commit. This operation
        neither evaluates the source nor changes an existing HEAD or draft.
        """
        ...

    def current_revision(self, workspace: Path) -> str:
        """Return the opaque current revision, or an empty string before initialization."""
        ...
    def retain_revision(self, workspace: Path, revision: str, *, retention_key: str) -> None:
        """Retain verified local provenance under an owner-validated literal key."""
        ...
    def import_revision(self, workspace: Path, source_workspace: Path, revision: str,
                        *, retention_key: str) -> None:
        """Import and retain an explicit local source revision without changing HEAD."""
        ...
    def fork_revision(self, workspace: Path, revision: str, destination: Path) -> None:
        """Atomically create a standalone controller workspace at an explicit revision.

        Preserve declared frozen context; do not inherit active candidates or
        introduce external object alternates, hardlinks or remote authority.
        """
        ...
    def read_working(self, workspace: Path, paths: tuple[str, ...]) -> RevisionContent:
        """Read selected working bytes without following links or interpreting content."""
        ...
    def read_source(self, workspace: Path, *, revision: str | None = None) -> RevisionContent:
        """Read complete declared source; the owner validates members and frozen constraints."""
        ...
    def install_source(self, workspace: Path, source: RevisionContent) -> None:
        """Transactionally publish complete source, including deletions, without changing Git.

        Validate fixed content and declaration before writing; preserve unrelated
        exploration and immutable inputs. Ordinary install_files remains upsert.
        """
        ...
    def source_identity(self, source: RevisionContent) -> str:
        """Return Task's pure SHA256 content identity without exposing its source codec."""
        ...
    def revision_history(self, workspace: Path, *, revision: str | None = None,
                         paths: tuple[str, ...] = (),
                         changed_path: str | None = None) -> tuple[RevisionFacts, ...]:
        """Return reachable history in stable newest-first order, with explicit path identities."""
        ...
    def revision_changes(self, workspace: Path, base_revision: str, revision: str) -> tuple[str, ...]:
        """Read exact changed paths between explicit revisions without moving the workspace."""
        ...
    def working_facts(self, workspace: Path, *, paths: tuple[str, ...] = (),
                      include_untracked: bool = True) -> WorkingFacts:
        """Read changes and explicit content identities; tracked-only checks are explicit."""
        ...
    def snapshot_files(self, workspace: Path, destination: Path, *,
                       directories: tuple[str, ...],
                       exclude_paths: tuple[str, ...] = ()) -> WorkspaceFilesSnapshot:
        """Capture selected mutable closures, including modes, deletions and empty directories."""
        ...
    def restore_files(self, snapshot: WorkspaceFilesSnapshot, workspace: Path, *,
                      exclude_paths: tuple[str, ...] = ()) -> None:
        """Validate completely, then transactionally publish selected mutable closures."""
        ...
    def clear_paths(self, workspace: Path, paths: tuple[str, ...]) -> None:
        """Clear explicit mutable paths without changing source, revision or index."""
        ...
    def filesystem_access(self, workspace: Path, *, include_git: bool = False) -> FilesystemAccess:
        """Describe immutable inputs; reject Git grants unsupported by this implementation."""
        ...
    def install_files(self, workspace: Path, files: tuple[WorkspaceFile, ...], *,
                      immutable: bool = False) -> SourceSnapshot:
        """Copy selected files; immutable preflights all targets and rejects changed bytes/modes."""
        ...
    def read_revision(self, workspace: Path, revision: str,
                      changed_paths: list[str]) -> RevisionContent:
        """Read committed bytes without creating snapshot files or modifying Git."""
        ...

