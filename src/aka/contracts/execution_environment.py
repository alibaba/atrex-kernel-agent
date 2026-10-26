"""Environment discovery values, independent of any Executor implementation."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable


def _freeze(value):
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("environment facts require string keys")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if value is None or type(value) in (str, int, float, bool):
        if type(value) is float and not math.isfinite(value):
            raise ValueError("environment facts require finite numbers")
        return value
    raise TypeError("environment facts must be JSON data")


def environment_document(value):
    if is_dataclass(value):
        return {item.name: environment_document(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {key: environment_document(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [environment_document(item) for item in value]
    return value


def requirements_identity(requirements):
    """Empty requirements have an explicit identity as well."""
    return hashlib.sha256(json.dumps(list(requirements), separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class EnvironmentQuery:
    hardware: str = ""
    operation: str = "run"
    languages: tuple[str, ...] = ()
    requirements: tuple[str, ...] = ()
    num_gpus: int = 1
    fresh: bool = False

    def __post_init__(self):
        for name in ("languages", "requirements"):
            values = tuple(getattr(self, name))
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"{name} must contain nonempty strings")
            object.__setattr__(self, name, values)
        if type(self.num_gpus) is not int or self.num_gpus < 1:
            raise ValueError("environment GPU count must be positive")
        if type(self.fresh) is not bool:
            raise TypeError("environment freshness must be boolean")


@dataclass(frozen=True)
class EnvironmentSnapshot:
    backend: str
    endpoint: str | None
    hardware: str
    identity: str | None
    declarations: Mapping = field(default_factory=dict)
    observed: Mapping = field(default_factory=dict)
    source: str = "unknown"
    probed_at: str | None = None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "declarations", _freeze(self.declarations))
        object.__setattr__(self, "observed", _freeze(self.observed))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    def as_dict(self):
        return environment_document(self)


@dataclass(frozen=True)
class EnvironmentSupport:
    support: Literal["supported", "unsupported", "unknown"]
    availability: Literal["available", "unavailable", "unknown"]
    environment: EnvironmentSnapshot
    dimensions: Mapping[str, str]
    reasons: tuple[str, ...]
    requirements_identity: str

    def __post_init__(self):
        if self.support not in {"supported", "unsupported", "unknown"}:
            raise ValueError("invalid environment support")
        if self.availability not in {"available", "unavailable", "unknown"}:
            raise ValueError("invalid environment availability")
        object.__setattr__(self, "dimensions", _freeze(self.dimensions))
        object.__setattr__(self, "reasons", tuple(self.reasons))

    def as_dict(self):
        return environment_document(self)


@runtime_checkable
class EnvironmentExecutor(Protocol):
    """Optional extension; discovery never submits, installs or reserves a job."""
    def query_environment(self, query: EnvironmentQuery) -> EnvironmentSnapshot: ...
    def check_environment(self, query: EnvironmentQuery) -> EnvironmentSupport: ...
