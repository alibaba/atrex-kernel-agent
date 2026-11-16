"""Backend-neutral requests and captured results for synchronous execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable, Literal
from types import MappingProxyType
from .execution_environment import EnvironmentSnapshot, EnvironmentSupport
from .execution_recovery import ExecutionCause, JobReference
from .filesystem import FilesystemAccess


class Cancellation(Protocol):
    def is_set(self) -> bool: ...


@dataclass(frozen=True)
class ProfileOptions:
    """Explicit measurement inputs; unspecified values keep producer defaults."""
    level: Literal["survey", "sol", "deep"] | None = None
    profiler: Literal["ncu", "nsys", "rocprofv3", "acu"] | None = None
    counters: tuple[str, ...] = ()
    kernel_regex: str | None = None
    top_kernels: int | None = None

    def __post_init__(self):
        if self.level not in (None, "survey", "sol", "deep"):
            raise ValueError("unsupported profile level")
        if self.profiler not in (None, "ncu", "nsys", "rocprofv3", "acu"):
            raise ValueError("unsupported profiler")
        if isinstance(self.counters, str):
            raise TypeError("profile counters require a sequence of names")
        object.__setattr__(self, "counters", tuple(self.counters))
        if any(not isinstance(c, str) or not c for c in self.counters):
            raise ValueError("profile counters must be nonempty strings")
        if self.kernel_regex is not None and not isinstance(self.kernel_regex, str):
            raise TypeError("profile kernel regex must be a string")
        if self.top_kernels is not None and (type(self.top_kernels) is not int or self.top_kernels < 1):
            raise ValueError("profile top kernels must be positive")


@dataclass(frozen=True)
class OutputOptions:
    """Explicit collection policy without introducing transport defaults."""
    inline: bool | None = None
    include_raw_profile: bool | None = None
    max_file_bytes: int | None = None

    def __post_init__(self):
        if any(value is not None and type(value) is not bool
               for value in (self.inline, self.include_raw_profile)):
            raise TypeError("output switches must be boolean or unspecified")
        if self.max_file_bytes is not None and (type(self.max_file_bytes) is not int or self.max_file_bytes < 1):
            raise ValueError("output file limit must be positive")


@dataclass(frozen=True)
class PreparedSourceArchive:
    """Immutable owner-prepared archive, independent of storage credentials."""
    role: Literal["candidate", "baseline"]
    path: str
    entry_point: str
    content: bytes = field(repr=False)

    def __post_init__(self):
        if self.role not in {"candidate", "baseline"}:
            raise ValueError("source archive role must be candidate or baseline")
        for name in (self.path, self.entry_point):
            path = Path(name)
            if (not isinstance(name, str) or not name or path.is_absolute()
                    or path.as_posix() != name or ".." in path.parts
                    or path == Path(".") or "\\" in name
                    or any(ord(char) < 32 for char in name)):
                raise ValueError("source archive paths must be safe relative names")
        if not isinstance(self.content, bytes) or not self.content:
            raise TypeError("source archive content must be nonempty immutable bytes")


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
    languages: tuple[str, ...] = ()
    requirements: tuple[str, ...] = ()
    on_observation: Callable[[JobResult], None] | None = field(default=None, repr=False, compare=False)
    filesystem_access: FilesystemAccess | None = None
    profile_options: ProfileOptions | None = None
    input_files: tuple[ExecutionFile, ...] = field(default=(), repr=False)
    output_options: OutputOptions | None = None
    prepared_inputs: PreparedExecutionInputs | None = field(default=None, repr=False)
    source_archives: tuple[PreparedSourceArchive, ...] = field(default=(), repr=False)
    idempotency_key: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "command", tuple(self.command))
        object.__setattr__(self, "environment", MappingProxyType(dict(self.environment)))
        object.__setattr__(self, "languages", tuple(self.languages))
        object.__setattr__(self, "requirements", tuple(self.requirements))
        archives = tuple(self.source_archives)
        if any(not isinstance(item, PreparedSourceArchive) for item in archives):
            raise TypeError("source archives require PreparedSourceArchive values")
        if (len({item.role for item in archives}) != len(archives)
                or len({item.path for item in archives}) != len(archives)):
            raise ValueError("source archive roles and paths must be unique")
        object.__setattr__(self, "source_archives", archives)
        if self.idempotency_key is not None and (not isinstance(self.idempotency_key, str)
                or not self.idempotency_key or "\0" in self.idempotency_key):
            raise ValueError("idempotency key must be nonempty text")
        if self.filesystem_access is not None and not isinstance(self.filesystem_access, FilesystemAccess):
            raise TypeError("execution scope requires FilesystemAccess")
        if self.profile_options is not None and not isinstance(self.profile_options, ProfileOptions):
            raise TypeError("profile inputs require ProfileOptions")
        if self.output_options is not None and not isinstance(self.output_options, OutputOptions):
            raise TypeError("output policy requires OutputOptions")
        object.__setattr__(self, "input_files", tuple(self.input_files))
        if any(not isinstance(item, ExecutionFile) for item in self.input_files):
            raise TypeError("execution inputs require immutable ExecutionFile values")
        paths = [Path(item.path).as_posix() for item in self.input_files]
        if len(paths) != len(set(paths)):
            raise ValueError("execution input paths must be unique")
        if self.prepared_inputs is not None:
            if not isinstance(self.prepared_inputs, PreparedExecutionInputs):
                raise TypeError("prepared inputs require PreparedExecutionInputs")
            prepared_paths = {Path(item.path).as_posix() for item in (
                *self.prepared_inputs.files, *self.prepared_inputs.runtime_files)}
            if prepared_paths.intersection(paths):
                raise ValueError("prepared and explicit input paths must be distinct")


@dataclass(frozen=True)
class BackendCapabilities:
    operations: frozenset[str] = frozenset(("run",))
    cancellation: bool = False
    prepared_requests: bool = False
    filesystem_isolation: bool = False
    input_files: bool = False
    output_options: bool = False
    prepared_inputs: bool = False
    source_archives: bool = False

    def require(self, command: CommandSpec) -> None:
        if command.source_archives and not self.source_archives:
            raise UnsupportedExecution("executor does not support prepared source archives")
        if command.filesystem_access is not None and not self.filesystem_isolation:
            raise UnsupportedExecution("executor cannot enforce the requested file and environment scope")
        if command.input_files and not self.input_files:
            raise UnsupportedExecution("executor does not support explicit input files")
        if command.prepared_inputs is not None and not self.prepared_inputs:
            raise UnsupportedExecution("executor does not support closed prepared inputs")
        if command.output_options is not None and not self.output_options:
            raise UnsupportedExecution("executor does not support explicit output collection")
        if command.profile_options is not None and command.operation != "profile":
            raise UnsupportedExecution("profile options require a profile operation")
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
class ExecutionBudget:
    scope: str
    requested: int | None
    effective: int
    hard_limit: int | None = None
    reason: str = "requested"

    def __post_init__(self):
        if type(self.effective) is not int or self.effective < 1:
            raise ValueError("effective execution budget must be positive")
        if self.requested is not None and (type(self.requested) is not int or self.requested < 1):
            raise ValueError("requested execution budget must be positive or unspecified")
        if self.hard_limit is not None and (type(self.hard_limit) is not int or self.hard_limit < 1):
            raise ValueError("execution hard limit must be positive")


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
    causes: tuple[ExecutionCause, ...] = ()
    idempotency_key: str | None = None
    request_identity: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(self, "causes", tuple(self.causes))


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
    filesystem_access: FilesystemAccess | None = None

    def __post_init__(self):
        object.__setattr__(self, "environment", MappingProxyType(dict(self.environment)))
        if self.filesystem_access is not None and not isinstance(self.filesystem_access, FilesystemAccess):
            raise TypeError("execution binding scope requires FilesystemAccess")


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
class PreparedExecutionInputs:
    """Host-prepared execution content; no roots, mounts or additional authority.

    Private members enter only the trusted temporary execution view and cannot
    be synchronized back as command outputs. Runtime grouping retains the
    existing separate transport bundle.
    """
    files: tuple[ExecutionFile, ...] = field(default=(), repr=False)
    runtime_files: tuple[ExecutionFile, ...] = field(default=(), repr=False)
    private_paths: tuple[str, ...] = ()

    def __post_init__(self):
        for name in ("files", "runtime_files"):
            values = tuple(getattr(self, name))
            if any(not isinstance(item, ExecutionFile) for item in values):
                raise TypeError("prepared content requires immutable ExecutionFile values")
            object.__setattr__(self, name, values)
        paths = [Path(item.path).as_posix() for item in (*self.files, *self.runtime_files)]
        if len(paths) != len(set(paths)):
            raise ValueError("prepared content paths must be unique across groups")
        private = tuple(self.private_paths)
        if any(not isinstance(path, str) or path not in paths for path in private):
            raise ValueError("private paths must name prepared content members")
        if len(private) != len(set(private)):
            raise ValueError("private prepared paths must be unique")
        object.__setattr__(self, "private_paths", private)


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
    environment_snapshot: EnvironmentSnapshot | None = None
    budgets: tuple[ExecutionBudget, ...] = ()
    environment_support: EnvironmentSupport | None = None
    causes: tuple[ExecutionCause, ...] = ()
    job_references: tuple[JobReference, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "causes", tuple(self.causes))
        object.__setattr__(self, "job_references", tuple(self.job_references))
        object.__setattr__(self, "attempts", tuple(self.attempts))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(self, "files", tuple(self.files))
        object.__setattr__(self, "budgets", tuple(self.budgets))
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
