"""File contributions passed by the application to its workspace writer."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class WorkspaceFile:
    path: str
    source: Path | None = None
    text: str | None = None
    copy: Literal["cp", "copy2"] = "copy2"

    def __post_init__(self):
        if (self.source is None) == (self.text is None):
            raise ValueError("a workspace file requires either source or text")
        if self.copy not in ("cp", "copy2"):
            raise ValueError("unsupported workspace copy mode")


@dataclass(frozen=True)
class SourceContent:
    files: tuple[WorkspaceFile, ...]
    readme: str | None = None


@dataclass(frozen=True)
class PublicInputContent:
    ready: bool
    files: tuple[WorkspaceFile, ...] = ()
