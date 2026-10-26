"""Measurement identity and evidence quality, independent of execution state."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MappingProxyType


def _freeze(value):
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("measurement parameter keys must be strings")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise TypeError("measurement parameters must be finite JSON data")


def _document(value):
    if isinstance(value, Mapping):
        return {key: _document(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_document(item) for item in value]
    return value


@dataclass(frozen=True)
class MeasurementPoint:
    side: str
    revision: str
    repeat: int
    shape_id: str | None

    def __post_init__(self):
        if not isinstance(self.side, str) or not isinstance(self.revision, str) or not self.side or not self.revision:
            raise ValueError("measurement point requires a side and source revision")
        if type(self.repeat) is not int or self.repeat < 0:
            raise ValueError("measurement repeat must be a non-negative integer")
        if self.shape_id is not None and (not isinstance(self.shape_id, str) or not self.shape_id):
            raise TypeError("measurement shape identity must be text or unknown")

    def as_dict(self):
        return {"side": self.side, "revision": self.revision,
                "repeat": self.repeat, "shape_id": self.shape_id}


@dataclass(frozen=True)
class MeasurementIdentity:
    method: str
    points: tuple[MeasurementPoint, ...]
    input_digest: str
    parameters: Mapping = field(default_factory=dict)
    requirements: tuple[str, ...] = ()
    dependency_policy: str = "unknown"
    protocol_version: int = 1

    def __post_init__(self):
        if not isinstance(self.method, str) or not self.method or not isinstance(self.input_digest, str) or len(self.input_digest) != 64 or any(
                char not in "0123456789abcdef" for char in self.input_digest):
            raise ValueError("measurement identity requires a method and SHA256 input digest")
        points = tuple(self.points)
        if any(not isinstance(point, MeasurementPoint) for point in points) or len(set(points)) != len(points):
            raise ValueError("measurement identity requires distinct measurement points")
        requirements = tuple(self.requirements)
        if any(not isinstance(item, str) for item in requirements):
            raise TypeError("measurement requirements must be text")
        if type(self.protocol_version) is not int or self.protocol_version != 1:
            raise ValueError("unsupported measurement identity version")
        if not isinstance(self.dependency_policy, str) or not self.dependency_policy:
            raise ValueError("measurement dependency policy must be explicit or unknown")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "requirements", requirements)
        object.__setattr__(self, "parameters", _freeze(self.parameters))

    def as_dict(self):
        return {"protocol_version": self.protocol_version, "method": self.method,
                "points": [point.as_dict() for point in self.points],
                "input_digest": self.input_digest, "parameters": _document(self.parameters),
                "requirements": list(self.requirements), "dependency_policy": self.dependency_policy}

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True,
            ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class MeasurementDiagnostic:
    category: str
    code: str
    point: MeasurementPoint | None = None
    basis: tuple[str, ...] = ()

    def __post_init__(self):
        if self.point is not None and not isinstance(self.point, MeasurementPoint):
            raise TypeError("measurement diagnostic point must use the public contract")
        object.__setattr__(self, "basis", tuple(self.basis))

    def as_dict(self):
        return {"category": self.category, "code": self.code,
                "point": self.point.as_dict() if self.point else None, "basis": list(self.basis)}


@dataclass(frozen=True)
class MeasurementAssessment:
    identity: MeasurementIdentity
    completeness: str = "unknown"
    validity: str = "unknown"
    observed: tuple[MeasurementPoint, ...] = ()
    missing: tuple[MeasurementPoint, ...] = ()
    diagnostics: tuple[MeasurementDiagnostic, ...] = ()
    comparability: str = "unknown"
    environment_identity: str | None = None

    def __post_init__(self):
        if not isinstance(self.identity, MeasurementIdentity):
            raise TypeError("measurement assessment requires a public identity")
        if self.environment_identity is not None and not isinstance(self.environment_identity, str):
            raise TypeError("measurement environment identity must be text or unknown")
        if self.completeness not in {"complete", "partial", "unknown"}:
            raise ValueError("unsupported measurement completeness")
        if self.validity not in {"valid", "invalid", "unknown"}:
            raise ValueError("unsupported measurement validity")
        if self.comparability not in {"comparable", "unknown"}:
            raise ValueError("unsupported measurement comparability")
        observed, missing = tuple(self.observed), tuple(self.missing)
        expected = set(self.identity.points)
        if not set(observed).issubset(expected) or not set(missing).issubset(expected) or set(observed).intersection(missing):
            raise ValueError("measurement observation must partition the expected points")
        if len(set(observed)) != len(observed) or len(set(missing)) != len(missing):
            raise ValueError("measurement observations must be distinct")
        if self.completeness == "complete" and (missing or set(observed) != expected):
            raise ValueError("complete measurement requires every expected point")
        if self.comparability == "comparable" and (not self.environment_identity or self.identity.dependency_policy == "unknown"):
            raise ValueError("measurement comparability requires known environment and dependency policy")
        object.__setattr__(self, "observed", observed)
        object.__setattr__(self, "missing", missing)
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    def as_dict(self):
        return {"identity": self.identity.as_dict(), "identity_digest": self.identity.digest,
                "completeness": self.completeness, "validity": self.validity,
                "observed": [point.as_dict() for point in self.observed],
                "missing": [point.as_dict() for point in self.missing],
                "diagnostics": [item.as_dict() for item in self.diagnostics],
                "comparability": self.comparability, "environment_identity": self.environment_identity}
