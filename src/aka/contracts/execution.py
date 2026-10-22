"""Backend-neutral requests and captured results for synchronous execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable
from types import MappingProxyType


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
    submission: bytes | None = field(default=None, repr=False)
    num_gpus: int = 1


@dataclass(frozen=True)
class BackendCapabilities:
    operations: frozenset[str] = frozenset(("run",))
    cancellation: bool = False
    prepared_requests: bool = False

    def require(self, command: CommandSpec) -> None:
        if type(command.timeout) is not int or command.timeout <= 0:
            raise UnsupportedExecution("execution timeout must be a positive integer")
        if command.submission is not None and not self.prepared_requests:
            raise UnsupportedExecution("executor does not support prepared requests")
        if type(command.num_gpus) is not int or command.num_gpus < 1:
            raise UnsupportedExecution("execution GPU count must be positive")
        if command.submission is not None and not isinstance(command.submission, bytes):
            raise UnsupportedExecution("prepared submission must be immutable bytes")
        if not command.command and command.submission is None and command.operation != "preflight":
            raise UnsupportedExecution("execution command must contain argv")
        if command.operation not in self.operations:
            raise UnsupportedExecution(f"unsupported execution operation: {command.operation}")
        if command.cancel_event is not None and not self.cancellation:
            raise UnsupportedExecution("executor does not support in-flight cancellation")


class ExecutionUnavailable(RuntimeError):
    """Explicit service outage; retry the same command without a candidate verdict."""
    def __init__(self, message, outcome=None):
        self.outcome = outcome
        super().__init__(message)


class ExecutionPending(RuntimeError):
    """Execution may still be running; this result cannot authorize resubmission."""
    def __init__(self, message, outcome=None):
        self.outcome = outcome
        super().__init__(message)


class UnsupportedExecution(ValueError):
    pass


@dataclass(frozen=True)
class ResourceIdentity:
    backend: str
    identifier: str | None = None


@dataclass(frozen=True)
class ExecutionDiagnostic:
    """Safe transport facts. Evaluation, not Execution, decides repairability."""
    category: str
    code: str
    details: Mapping[str, str | int | bool] = field(default_factory=dict)

    def __post_init__(self):
        if any(type(value) not in (str, int, bool) for value in self.details.values()):
            raise TypeError("diagnostic details must be scalar transport facts")
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))


@dataclass(frozen=True)
class ExecutionAttempt:
    """One actual backend submission/attempt, in observation order."""
    operation: str
    disposition: str = "unknown"  # not_submitted, terminal, unknown
    resource: ResourceIdentity | None = None
    termination: str | None = None
    exit_code: int | None = None
    diagnostics: tuple[ExecutionDiagnostic, ...] = ()
    attempt_id: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))


@dataclass(frozen=True)
class ExecutionFacts:
    disposition: str = "unknown"
    termination: str | None = None
    cancellation_requested: bool = False
    local_stopped: bool | None = None
    remote_terminated: bool | None = None
    cleanup: str = "unknown"  # not_required, completed, failed, unknown
    cleanup_diagnostics: tuple[ExecutionDiagnostic, ...] = ()
    basis: tuple[str, ...] = ()
    remote_cleanup: str = "unknown"

    def __post_init__(self):
        object.__setattr__(self, "cleanup_diagnostics", tuple(self.cleanup_diagnostics))
        object.__setattr__(self, "basis", tuple(self.basis))


@dataclass(frozen=True)
class ExecutionContext:
    """Host inputs copied once when borrowing an execution session."""
    environment: Mapping[str, str] = field(default_factory=dict)
    hardware: str = ""
    timeout: int = 600
    formal: bool = False

    def __post_init__(self):
        object.__setattr__(self, "environment", MappingProxyType(dict(self.environment)))


@runtime_checkable
class ExecutionBinder(Protocol):
    def bind(self, context: ExecutionContext) -> Executor: ...


@dataclass(frozen=True)
class ExecutionFile:
    """Retrieved workspace-relative content; the caller owns publication."""
    path: str
    content: bytes = field(repr=False)
    mode: int = 0o600

    def __post_init__(self):
        path = Path(self.path)
        if not self.path or path.is_absolute() or ".." in path.parts or path == Path("."):
            raise ValueError("execution file requires a workspace-relative path")
        if not isinstance(self.content, bytes):
            raise TypeError("execution file content must be immutable bytes")
        if type(self.mode) is not int or not 0 <= self.mode <= 0o777:
            raise ValueError("invalid execution file permissions")


@dataclass(frozen=True)
class ExecutionFileReceipt:
    path: str
    mode: int
    size_bytes: int
    sha256: str

    def __post_init__(self):
        import re
        path = Path(self.path)
        if path.is_absolute() or ".." in path.parts or path == Path("."):
            raise ValueError("file receipt requires a workspace-relative path")
        if type(self.size_bytes) is not int or self.size_bytes < 0 or re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None:
            raise ValueError("invalid file receipt integrity metadata")


@dataclass(frozen=True)
class JobResult:
    """Captured transport output; a command exit code may remain unknown."""

    transport_returncode: int
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    resource: ResourceIdentity | None = None
    output_scope: str = "transport"
    transport_status: str = "unknown"
    facts: ExecutionFacts = field(default_factory=ExecutionFacts)
    attempts: tuple[ExecutionAttempt, ...] = ()
    diagnostics: tuple[ExecutionDiagnostic, ...] = ()
    hardware: str = ""
    environment_identity: str | None = None
    files: tuple[ExecutionFile, ...] = field(default=(), repr=False)
    file_receipts: tuple[ExecutionFileReceipt, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "attempts", tuple(self.attempts))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(self, "files", tuple(self.files))
        import hashlib
        receipts = tuple(ExecutionFileReceipt(item.path, item.mode, len(item.content),
                         hashlib.sha256(item.content).hexdigest()) for item in self.files)
        object.__setattr__(self, "file_receipts", receipts or tuple(self.file_receipts))

    def as_dict(self) -> dict:
        """Return independent JSON-compatible facts for explicit archival."""
        def encode(value):
            if is_dataclass(value):
                return {item.name: encode(getattr(value, item.name)) for item in fields(value)}
            if isinstance(value, Mapping):
                return {key: encode(item) for key, item in value.items()}
            if isinstance(value, tuple):
                return [encode(item) for item in value]
            return value
        value = encode(self)
        value.pop("files", None)
        return value

    @property
    def primary_diagnostic(self) -> ExecutionDiagnostic | None:
        if self.transport_status in {"succeeded", "failed", "unknown"}:
            return None
        return self.diagnostics[0] if self.diagnostics else None

    @property
    def returncode(self) -> int:
        """Legacy effective code: transport failure takes precedence over command exit."""
        if type(self.transport_returncode) is not int or (
            self.exit_code is not None and type(self.exit_code) is not int
        ):
            raise TypeError("execution return codes must be integers or an unknown command exit")
        if self.output_scope not in ("transport", "command"):
            raise ValueError("unknown execution output scope")
        if self.output_scope == "command" and self.exit_code is None:
            raise ValueError("command-scoped output requires a known command exit code")
        return self.transport_returncode or self.exit_code or 0

    def require_transport(self) -> None:
        """Raise on execution failures; never turn an unknown job into a retry."""
        if self.transport_status == "infrastructure":
            if self.facts.disposition == "unknown":
                raise ExecutionPending("GPU transport unavailable; remote execution state unknown", self)
            raise ExecutionUnavailable("GPU validation transport unavailable", self)
        if self.transport_status in {"timed_out", "cancelled", "interrupted"}:
            raise ExecutionPending("execution " + self.transport_status, self)



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
