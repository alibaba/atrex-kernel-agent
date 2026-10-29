"""Private host binding for the original sandbox launcher and recovery hooks."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
import subprocess

from aka.contracts.execution import Cancellation, JobResult, ExecutionFile, OutputOptions, ProfileOptions, PreparedExecutionInputs


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
    submission: bytes | None = None
    num_gpus: int = 1
    on_observation: Callable[[JobResult], None] | None = None
    input_files: tuple[ExecutionFile, ...] = ()
    output_options: OutputOptions | None = None
    profile_options: ProfileOptions | None = None
    prepared_inputs: PreparedExecutionInputs | None = None


@dataclass(frozen=True)
class ExecutionHooks:
    spawn: Callable[..., subprocess.Popen[str]]
    check_environment: Callable[[], None]
    queue_wait_grace: Callable[[Mapping[str, str]], int]


_transport_arguments = ContextVar("aka_execution_transport_arguments", default=())


def current_transport_arguments() -> tuple[str, ...]:
    """Borrow validated CLI-only transport options within the Execution call."""
    return _transport_arguments.get()


@contextmanager
def transport_arguments(arguments: tuple[str, ...]):
    token = _transport_arguments.set(arguments)
    try:
        yield
    finally:
        _transport_arguments.reset(token)


def default_binding(context):
    """Resolve Legacy-owned configuration once; returned instances need no invocation scope."""
    import sys
    from aka.execution.configuration import legacy_settings, queue_wait_grace
    from aka.execution.processes import spawn_owned_session
    from aka.execution.recovery import raise_if_environment_blocked
    settings = legacy_settings(context.environment)
    request = SandboxRequest(
        workspace=Path("."), launcher=(sys.executable, "-m", "aka.execution.gateway.sandbox"),
        hardware=context.hardware, timeout=context.timeout, command=(),
        environment=context.environment, profile=settings.profile, url=settings.url,
        ssh=settings.ssh, ssh_init=settings.ssh_init, health_command=settings.health_command,
        wall_timeout=None,
    )
    return request, ExecutionHooks(spawn_owned_session, raise_if_environment_blocked, queue_wait_grace)
