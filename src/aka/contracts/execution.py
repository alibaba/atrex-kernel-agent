"""Backend-neutral workspace execution and an optional durable-job extension."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Protocol, runtime_checkable


class Cancellation(Protocol):
    def is_set(self) -> bool: ...


class ExecutionControl(Protocol):
    def checkpoint(self) -> None:
        """Check host safety; stop owned execution before propagating an exception."""
        ...


@dataclass(frozen=True)
class ExecutionBudget:
    wall_timeout: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.wall_timeout) or self.wall_timeout < 0:
            raise ValueError("execution wall timeout must be finite and non-negative")


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
    budget: ExecutionBudget | None = None
    control: ExecutionControl | None = None


@dataclass(frozen=True)
class BackendCapabilities:
    operations: frozenset[str] = frozenset(("run",))
    cancellation: bool = False
    durable_jobs: bool = False
    execution_control: bool = False

    def require(self, command: CommandSpec) -> None:
        if command.operation not in self.operations:
            raise UnsupportedExecution(f"unsupported execution operation: {command.operation}")
        if command.cancel_event is not None and not self.cancellation:
            raise UnsupportedExecution("executor does not support in-flight cancellation")
        if (command.control is not None or command.budget is not None) and not self.execution_control:
            raise UnsupportedExecution("executor does not support host execution controls")


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


@dataclass(frozen=True)
class JobHandle:
    backend: str
    identifier: str


@dataclass(frozen=True)
class JobSnapshot:
    handle: JobHandle
    status: str
    result: JobResult | None = None


@runtime_checkable
class Executor(Protocol):
    @property
    def capabilities(self) -> BackendCapabilities: ...

    @property
    def resource_identity(self) -> ResourceIdentity | None: ...

    def run(self, command: CommandSpec) -> JobResult:
        """Return or raise only after owned local execution is quiescent.

        Providers accepting controls must checkpoint at least every 0.25 seconds
        during staging, queueing and execution, including before returning a result.
        Budget expiry or a checkpoint exception requires cleanup, not success.
        These providers also quiesce owned remote work before returning or raising.
        Legacy transport compatibility does not confirm remote termination.
        """
        ...


@runtime_checkable
class DurableExecutor(Executor, Protocol):
    def submit(self, command: CommandSpec) -> JobHandle: ...

    def snapshot(self, handle: JobHandle) -> JobSnapshot: ...

    def wait(self, handle: JobHandle, *, timeout: float | None = None) -> JobSnapshot: ...

    def cancel(self, handle: JobHandle) -> bool: ...


def validate_executor(executor: Executor) -> None:
    if not isinstance(executor, Executor) or not callable(executor.run):
        raise TypeError("provider does not implement Executor")
    if not isinstance(executor.capabilities, BackendCapabilities):
        raise TypeError("executor capabilities must be BackendCapabilities")
    if executor.resource_identity is not None and not isinstance(executor.resource_identity, ResourceIdentity):
        raise TypeError("executor resource identity must be ResourceIdentity or None")
    if executor.capabilities.durable_jobs and (
        not isinstance(executor, DurableExecutor)
        or not all(callable(getattr(executor, method)) for method in ("submit", "snapshot", "wait", "cancel"))
    ):
        raise TypeError("durable jobs require the DurableExecutor extension")
