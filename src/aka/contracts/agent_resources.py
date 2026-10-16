"""Module-owned Skills for installation into an application's Agent workspace."""
from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from pathlib import Path
import re
from typing import Any, Literal


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
