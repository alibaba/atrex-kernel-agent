"""Private host binding for the original sandbox launcher and recovery hooks."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
import subprocess

from aka.contracts.execution import Cancellation


@dataclass(frozen=True)
class SandboxRequest:
    workspace: Path
    launcher: tuple[str, ...]
    hardware: str
    profile: str
    url: str
    timeout: int
    command: tuple[str, ...]
    environment: Mapping[str, str]
    ssh: str = ""
    ssh_init: str = ""
    health_command: str = ""
    sync: tuple[str, ...] = ()
    wall_timeout: int | None = None
    gateway_kind: str = "auto"
    preflight: bool = False
    cancel_event: Cancellation | None = None
    health_command_explicit: bool | None = None


@dataclass(frozen=True)
class ExecutionHooks:
    spawn: Callable[..., subprocess.Popen[str]]
    check_environment: Callable[[], None]
    queue_wait_grace: Callable[[Mapping[str, str]], int]


@dataclass
class SandboxInvocation:
    request: SandboxRequest
    hooks: ExecutionHooks
    result: subprocess.CompletedProcess[str] | None = None


_invocation = ContextVar("aka_legacy_sandbox_invocation", default=None)


def current_invocation() -> SandboxInvocation:
    invocation = _invocation.get()
    if invocation is None:
        raise RuntimeError("legacy execution requires a host sandbox runtime binding")
    return invocation


@contextmanager
def sandbox_invocation(request: SandboxRequest, hooks: ExecutionHooks):
    invocation = SandboxInvocation(request, hooks)
    token = _invocation.set(invocation)
    try:
        yield invocation
    finally:
        _invocation.reset(token)
