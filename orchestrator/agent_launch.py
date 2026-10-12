"""Application-scoped borrowed service; no process-global monkeypatch or Core owner."""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_SANDBOX = ContextVar("aka_agent_sandbox", default=None)


def current_sandbox():
    return _SANDBOX.get()


@contextmanager
def use_agent_sandbox(service):
    token = _SANDBOX.set(service)
    try:
        yield
    finally:
        _SANDBOX.reset(token)


def configure_agent_sandbox(*, mode=None, executable=None, read_only_paths=None):
    mode = mode if mode is not None else os.environ.get("ATREX_AGENT_SANDBOX")
    executable = executable if executable is not None else os.environ.get("ATREX_BWRAP_EXECUTABLE")
    if read_only_paths is None and "ATREX_AGENT_READ_ONLY_PATHS" in os.environ:
        read_only_paths = json.loads(os.environ["ATREX_AGENT_READ_ONLY_PATHS"])
        if not isinstance(read_only_paths, list) or not all(isinstance(p, str) for p in read_only_paths):
            raise ValueError("ATREX_AGENT_READ_ONLY_PATHS must be a JSON array of paths")
    service = current_sandbox()
    if service is None:
        if mode not in (None, "none") or executable is not None or read_only_paths:
            raise ValueError("Select the agent_sandbox service in the application composition first")
        return
    _SANDBOX.set(service.configure(mode=mode, executable=executable, read_only_paths=read_only_paths))


def prepare_agent_environment(workspace: Path, environment: dict[str, str], session_key: str):
    service = current_sandbox()
    if service is None:
        if environment.get("ATREX_AGENT_SANDBOX", "none") != "none":
            raise RuntimeError("Agent sandbox requested without an application launch service")
        return environment
    return service.prepare_environment(workspace, environment, session_key)


def start_episode(workspace: Path):
    service = current_sandbox()
    if service is not None:
        service.start_episode(workspace)
