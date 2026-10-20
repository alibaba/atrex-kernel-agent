"""Backend-neutral requests and captured results for synchronous execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


class Cancellation(Protocol):
    def is_set(self) -> bool: ...


@dataclass(frozen=True)
class CommandSpec:
    workspace: Path
    command: tuple[str, ...]
    timeout: int = 600
    environment: Mapping[str, str] = field(default_factory=dict)
    hardware: str = ""
    artifacts: tuple[str, ...] = ()
    operation: str = "run"
    cancel_event: Cancellation | None = None


@dataclass(frozen=True)
class BackendCapabilities:
    operations: frozenset[str] = frozenset(("run",))
    cancellation: bool = False

    def require(self, command: CommandSpec) -> None:
        if command.operation not in self.operations:
            raise UnsupportedExecution(f"unsupported execution operation: {command.operation}")
        if command.cancel_event is not None and not self.cancellation:
            raise UnsupportedExecution("executor does not support in-flight cancellation")


class ExecutionUnavailable(RuntimeError):
    """Explicit service outage; retry the same command without a candidate verdict."""
    pass


class UnsupportedExecution(ValueError):
    pass


@dataclass(frozen=True)
class ResourceIdentity:
    backend: str
    identifier: str | None = None


@dataclass(frozen=True)
class JobResult:
    """Captured transport output; a command exit code may remain unknown."""

    transport_returncode: int
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    resource: ResourceIdentity | None = None
    output_scope: str = "transport"


@runtime_checkable
class Executor(Protocol):
    @property
    def capabilities(self) -> BackendCapabilities: ...

    @property
    def resource_identity(self) -> ResourceIdentity | None: ...

    def run(self, command: CommandSpec) -> JobResult:
        """Capture execution output after owned local execution has stopped.

        Timeout, cancellation and cleanup semantics belong to the selected
        implementation. Legacy reaps its launcher without confirming remote
        termination; a result does not imply evaluation acceptance.
        """
        ...


def validate_executor(executor: Executor) -> None:
    if not isinstance(executor, Executor) or not callable(executor.run):
        raise TypeError("provider does not implement Executor")
    if not isinstance(executor.capabilities, BackendCapabilities):
        raise TypeError("executor capabilities must be BackendCapabilities")
    if executor.resource_identity is not None and not isinstance(executor.resource_identity, ResourceIdentity):
        raise TypeError("executor resource identity must be ResourceIdentity or None")
