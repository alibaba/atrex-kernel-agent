"""Read-only canonical measurement facts supplied by the Workflow owner."""
from collections.abc import Mapping
from dataclasses import dataclass
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol


def _freeze(value):
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("measurement keys must be strings")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if value is None or type(value) in (str, bool, int, float):
        if type(value) is float and not math.isfinite(value):
            raise ValueError("measurement numbers must be finite")
        return value
    raise ValueError("measurement result must contain JSON data")


def measurement_document(value):
    """Return detached JSON containers for an immutable measurement result."""
    if isinstance(value, Mapping):
        return {key: measurement_document(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [measurement_document(item) for item in value]
    return value


@dataclass(frozen=True)
class CanonicalMeasurement:
    version: int
    gate: str | None
    correctness: str | None
    result: Mapping[str, Any]

    def __post_init__(self):
        if type(self.version) is not int or self.version < 0:
            raise ValueError("invalid canonical measurement version")
        if any(value is not None and not isinstance(value, str)
               for value in (self.gate, self.correctness)):
            raise ValueError("invalid canonical measurement status")
        if not isinstance(self.result, Mapping):
            raise ValueError("measurement result must be a mapping")
        object.__setattr__(self, "result", _freeze(self.result))


class CanonicalMeasurementReader(Protocol):
    def baseline(self, workspace: Path) -> CanonicalMeasurement | None: ...
    def measurements(self, workspace: Path, *, before_version: int
                     ) -> tuple[CanonicalMeasurement, ...]:
        """Return available facts newest first, strictly before the given version."""
        ...
