"""Module-owned Skills for installation into an application's Agent workspace."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


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
