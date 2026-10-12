"""Immutable launch configuration, owned by the application composition."""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

from .boundary import read_only_grants, sandbox_executable, wrap_agent_command
from .home import prepare_agent_environment
from .workspace import reset_episode_scratch


@dataclass(frozen=True)
class SandboxPolicy:
    repo_root: Path
    mode: str = "none"
    executable: str = "bwrap"
    read_only_paths: tuple[str, ...] = ()

    def configure(self, *, mode=None, executable=None, read_only_paths=None):
        policy = replace(
            self, mode=self.mode if mode is None else mode,
            executable=self.executable if executable is None else executable,
            read_only_paths=self.read_only_paths if read_only_paths is None else tuple(read_only_paths),
        )
        values = policy.environment({})
        sandbox_executable(values)  # fail before Campaign side effects, never silently degrade
        if policy.mode == "bwrap":
            read_only_grants(values)
        return policy

    def environment(self, environment: Mapping[str, str]) -> dict[str, str]:
        # Operator configuration wins over any caller's per-session overrides.
        return dict(environment, ATREX_AGENT_SANDBOX=self.mode,
                    ATREX_BWRAP_EXECUTABLE=self.executable,
                    ATREX_AGENT_READ_ONLY_PATHS=json.dumps(self.read_only_paths))

    def prepare_environment(self, workspace, environment, session_key):
        return prepare_agent_environment(workspace, self.environment(environment), session_key)

    def start_episode(self, workspace):
        if self.mode == "bwrap":
            reset_episode_scratch(workspace)

    def wrap(self, command, workspace, environment, *, input_files=None):
        values = self.prepare_environment(
            workspace, environment,
            environment.get("ATREX_TELEMETRY_ATTEMPT_ID") or "\0".join(command),
        )
        launch, view = wrap_agent_command(command, workspace, values, repository_root=self.repo_root,
                                          auxiliary_input_files=input_files)
        # The ownership wrapper needs recovery metadata. Only the child-facing
        # FD payload is scrubbed; these values never become bwrap argv.
        launch.environment = values
        return launch, view
