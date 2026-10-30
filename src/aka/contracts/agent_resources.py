"""Module-owned Skills for installation into an application's Agent workspace."""
from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Literal, Mapping, Protocol
from .filesystem import FilesystemAccess


@dataclass(frozen=True)
class AgentSkill:
    name: str
    source: Path
    description: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not re.fullmatch(r"[a-z][a-z0-9-]*", self.name):
            raise ValueError("Agent Skill name must be a lowercase identifier")
        if not isinstance(self.source, Path):
            raise TypeError("Agent Skill source must be a Path")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("Agent Skill description is required")
        if not self.source.is_dir() or not (self.source / "SKILL.md").is_file():
            raise ValueError(f"Agent Skill directory is missing: {self.name}")
        if self.source.is_symlink() or any(path.is_symlink() for path in self.source.rglob("*")):
            raise ValueError("Agent Skill resources must not contain symlinks")


@dataclass(frozen=True)
class AgentResourceFile:
    path: str
    mode: int
    sha256: str


@dataclass(frozen=True)
class InstalledAgentSkill:
    name: str
    source: Path
    description: str
    fingerprint: str
    files: tuple[AgentResourceFile, ...]


@dataclass(frozen=True)
class WorkspaceResourceContribution:
    """One selected module's immutable file/tree or unmounted runtime dependency."""

    name: str
    source: Path
    destination: str | None = None
    visibility: Literal["agent", "runtime"] = field(default="agent", kw_only=True)
    version: str = field(default="1", kw_only=True)

    def __post_init__(self):
        if not isinstance(self.name, str) or not re.fullmatch(r"[a-z][a-z0-9-]*", self.name):
            raise ValueError("workspace resource name must be a lowercase identifier")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("workspace resource version is required")
        if not isinstance(self.source, Path):
            raise TypeError("workspace resource source must be a Path")
        if not self.source.is_file() and not self.source.is_dir():
            raise ValueError(f"workspace resource is missing: {self.name}")
        if self.source.is_symlink() or (self.source.is_dir() and any(p.is_symlink() for p in self.source.rglob("*"))):
            raise ValueError("workspace resource must not contain symlinks")
        if self.visibility not in ("agent", "runtime"):
            raise ValueError("unknown workspace resource visibility")
        if self.visibility == "runtime":
            if self.destination is not None:
                raise ValueError("runtime-only resources must not mount in the Agent workspace")
        else:
            if not isinstance(self.destination, str):
                raise ValueError("Agent-visible resources require a destination")
            path = Path(self.destination)
            if (not path.parts or path.is_absolute() or ".." in path.parts
                    or path.as_posix() != self.destination or "\\" in self.destination
                    or any(ord(c) < 32 for c in self.destination)):
                raise ValueError("workspace resource destination must be a strict relative path")
        object.__setattr__(self, "source", self.source.resolve())


@dataclass(frozen=True)
class InstalledWorkspaceResource:
    name: str
    source: Path
    destination: str | None
    visibility: str
    version: str
    fingerprint: str
    files: tuple[AgentResourceFile, ...]


@dataclass(frozen=True)
class AgentResourceInstallation:
    """Resource discovery result; None workspace denotes a pre-install preview.

    Catalog rows preserve the existing JSON tool/Skill schema. Consumers receive
    copies so their formatting cannot change the installer-owned result.
    """

    workspace: Path | None
    _tools: tuple[dict[str, Any], ...]
    _skills: tuple[dict[str, Any], ...]
    module_instructions: str
    module_skills: tuple[InstalledAgentSkill, ...] = ()
    resources: tuple[InstalledWorkspaceResource, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "_tools", deepcopy(tuple(self._tools)))
        object.__setattr__(self, "_skills", deepcopy(tuple(self._skills)))

    def resource(self, name: str) -> InstalledWorkspaceResource:
        for resource in self.resources:
            if resource.name == name:
                return resource
        raise KeyError(f"workspace resource is not selected: {name}")

    def catalog(self) -> list[dict[str, Any]]:
        return deepcopy(list(self._tools))

    def skill_catalog(self) -> list[dict[str, Any]]:
        return deepcopy(list(self._skills))


class AgentResourceRegistry(Protocol):
    def filesystem_access(self, workspace: Path) -> FilesystemAccess:
        """Return locked, installed resource links and read-only content grants."""
        ...


@dataclass(frozen=True)
class ResourcesSnapshot:
    """Retained selected resources; storage locations do not define identity."""

    directory: Path
    identity: str
    content_directory: Path | None = None


@dataclass(frozen=True)
class ResourceDirectoryFact:
    path: str
    mode: int


@dataclass(frozen=True)
class ResourceRootFacts:
    root: Path
    kind: Literal["file", "directory"]
    mode: int
    files: tuple[AgentResourceFile, ...]
    directories: tuple[ResourceDirectoryFact, ...]


@dataclass(frozen=True)
class ResourcesSnapshotFacts:
    """Verified retained content; no registry, runtime or filesystem grant."""

    identity: str
    selection_lock: Mapping[str, Any] = field(repr=False)
    roots: tuple[ResourceRootFacts, ...]
    absent_optional_roots: tuple[Path, ...]

    def __post_init__(self):
        object.__setattr__(self, "selection_lock", MappingProxyType(deepcopy(dict(self.selection_lock))))


class ResourcesRegistry(AgentResourceRegistry, Protocol):
    """Selected resource discovery, installation, calls and retained content."""

    def export_snapshot(self, destination: Path, *,
                        content_directory: Path | None = None) -> ResourcesSnapshot: ...

    def install(self, workspace: Path) -> AgentResourceInstallation: ...
    def installed_paths(self, workspace: Path) -> tuple[str, ...]:
        """Validate this selection and return its immutable workspace-relative destinations.

        Include the selection lock; external root grants remain filesystem_access.
        """
        ...
    def refresh_custom_skills(self, workspace: Path, *, directory: str = "skills") -> None:
        """Refresh validated custom discovery links while preserving reserved resources."""
        ...

    def resource_result(self, workspace: Path | None = None) -> AgentResourceInstallation: ...

    def check_lock(self, workspace: Path) -> None: ...

    def resource_snapshot(self) -> object:
        """Return opaque selected-resource identity for a host-owned checkpoint."""
        ...

    def check_resource_snapshot(self, snapshot: object) -> None:
        """Validate the selected resources against the recorded identity before mutation."""
        ...

    def catalog(self) -> list[dict]: ...

    def skill_catalog(self) -> list[dict]: ...

    def instructions(self, phase: str, **values: str) -> str: ...

    def environment(self, workspace: Path, campaign_name: str = "") -> dict[str, str]: ...

    def call(self, name: str, request: object, workspace: Path,
             *, cwd: Path | None = None, deadline: float | None = None,
             cancel_event: object | None = None) -> object:
        """Invoke a selected tool within its declared timeout and the caller's remaining grant."""
        ...
