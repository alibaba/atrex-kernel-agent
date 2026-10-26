"""Optional recovery capabilities over the existing execution result contract."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
import re
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from .execution import CommandSpec, JobResult


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise TypeError("execution cause details must contain finite JSON data")


@dataclass(frozen=True)
class ExecutionCause:
    """A backend's original error, including nested causes; no new job state."""
    category: str
    code: str
    message: str = ""
    details: Mapping = field(default_factory=dict)
    causes: tuple[ExecutionCause, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "details", _freeze(self.details))
        object.__setattr__(self, "causes", tuple(self.causes))


@dataclass(frozen=True)
class JobReference:
    """Confirmed physical job and selected route, retained for host recovery."""
    backend: str
    identifier: str
    operation: str = "run"
    endpoint: str = ""
    profile: str = ""
    request_identity: str = ""
    idempotency_key: str | None = None

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}", self.identifier):
            raise ValueError("invalid confirmed job identifier")
        if self.endpoint:
            route = urlsplit(self.endpoint)
            if route.scheme not in {"http", "https"} or not route.netloc or route.username or route.password or route.query or route.fragment:
                raise ValueError("job endpoint must be an HTTP route without credentials")


@dataclass(frozen=True)
class InfrastructureRetryPolicy:
    """Opt-in schedule and total elapsed budget; an empty schedule disables retry."""
    delays: tuple[float, ...] = ()
    max_elapsed_s: float = 0

    def __post_init__(self):
        delays = tuple(self.delays)
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in delays):
            raise ValueError("retry delays must be finite nonnegative seconds")
        if type(self.max_elapsed_s) not in (int, float) or not math.isfinite(self.max_elapsed_s) or self.max_elapsed_s < 0:
            raise ValueError("retry elapsed budget must be finite nonnegative seconds")
        if delays and self.max_elapsed_s <= 0:
            raise ValueError("configured retries require an elapsed budget")
        object.__setattr__(self, "delays", delays)


@dataclass(frozen=True)
class JobRecoveryCapabilities:
    query: bool = False
    cancellation_request: bool = False
    idempotent_replay: bool = False


@runtime_checkable
class RecoverableExecutor(Protocol):
    @property
    def recovery_capabilities(self) -> JobRecoveryCapabilities: ...

    def query_job(self, reference: JobReference, *, timeout: float,
                  retry: InfrastructureRetryPolicy = InfrastructureRetryPolicy()) -> JobResult: ...

    def cancel_job(self, reference: JobReference, *, timeout: float) -> JobResult: ...

    def replay_job(self, command: CommandSpec, *, idempotency_key: str) -> JobResult: ...
