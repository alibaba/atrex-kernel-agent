"""Borrowed synchronous launch policy; no Core or optimizer dependency."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Protocol, Tuple


class AgentLaunch(Protocol):
    command: list[str]
    environment: dict[str, str]

    @property
    def pass_fds(self) -> Tuple[int, ...]: ...

    def close(self) -> None: ...


class WorkspaceView(Protocol):
    def publish(self) -> None: ...
    def close(self) -> None: ...


class AgentSandbox(Protocol):
    mode: str

    def configure(self, *, mode: Optional[str] = None,
                  executable: Optional[str] = None,
                  read_only_paths: Optional[list[str]] = None) -> AgentSandbox: ...

    def prepare_environment(self, workspace: Path, environment: Mapping[str, str],
                            session_key: str) -> dict[str, str]: ...

    def start_episode(self, workspace: Path) -> None: ...

    def wrap(self, command: list[str], workspace: Path, environment: Mapping[str, str],
             *, input_files: Optional[dict[str, Path]] = None
             ) -> Tuple[AgentLaunch, Optional[WorkspaceView]]: ...
