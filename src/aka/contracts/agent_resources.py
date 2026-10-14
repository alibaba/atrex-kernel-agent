"""Module-owned Skills for installation into an application's Agent workspace."""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
import re
from typing import Any


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "_tools", deepcopy(tuple(self._tools)))
        object.__setattr__(self, "_skills", deepcopy(tuple(self._skills)))

    def catalog(self) -> list[dict[str, Any]]:
        return deepcopy(list(self._tools))

    def skill_catalog(self) -> list[dict[str, Any]]:
        return deepcopy(list(self._skills))
