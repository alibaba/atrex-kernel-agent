from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SUPPORTED_SCHEMA_VERSION = 1


def canonical_json(value: Mapping[str, Any]) -> str:
    """Serialize contract data deterministically and reject non-finite numbers."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("%s must be an object" % name)
    return value


def _reject_unknown(raw: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unexpected = sorted(set(raw) - allowed)
    if unexpected:
        raise ValueError("%s contains unexpected fields: %s" % (name, ", ".join(unexpected)))


def _require_schema_version(value: Any) -> int:
    if isinstance(value, bool) or value != _SUPPORTED_SCHEMA_VERSION:
        raise ValueError("schema_version must be %d" % _SUPPORTED_SCHEMA_VERSION)
    return int(value)


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("%s must be a non-empty string" % name)
    return value.strip()


def _require_identifier(value: Any, name: str) -> str:
    result = _require_string(value, name)
    if not _IDENTIFIER.fullmatch(result):
        raise ValueError("%s must be an opaque identifier" % name)
    return result


def _require_sha256(value: Any, name: str) -> str:
    result = _require_string(value, name).lower()
    if not _SHA256.fullmatch(result):
        raise ValueError("%s must be a lowercase SHA-256 digest" % name)
    return result


def _require_positive_finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be a number" % name)
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("%s must be finite and positive" % name)
    return result


def _require_ratio(value: Any, name: str) -> float:
    result = _require_positive_finite(value, name)
    if result < 1.0:
        raise ValueError("%s must be at least 1.0" % name)
    return result


def _require_strings(
    value: Any,
    name: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("%s must be a list of strings" % name)
    result = tuple(_require_string(item, "%s[]" % name) for item in value)
    if not allow_empty and not result:
        raise ValueError("%s must not be empty" % name)
    if len(set(result)) != len(result):
        raise ValueError("%s must not contain duplicates" % name)
    return result


def _enum_value(enum_type: type[Enum], value: Any, name: str) -> Enum:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("unsupported %s: %r" % (name, value)) from exc


@dataclass(frozen=True)
class OperatorIdentity:
    canonical_id: str
    aliases: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "canonical_id", _require_identifier(self.canonical_id, "operator.canonical_id"))
        object.__setattr__(self, "aliases", _require_strings(self.aliases, "operator.aliases"))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OperatorIdentity":
        raw = _require_mapping(value, "operator")
        _reject_unknown(raw, {"canonical_id", "aliases"}, "operator")
        return cls(raw.get("canonical_id"), raw.get("aliases"))

    def to_mapping(self) -> dict[str, Any]:
        return {"canonical_id": self.canonical_id, "aliases": list(self.aliases)}


@dataclass(frozen=True)
class SourceProvenance:
    project: str
    revision: str
    license: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "project", _require_identifier(self.project, "source.project"))
        object.__setattr__(self, "revision", _require_string(self.revision, "source.revision"))
        object.__setattr__(self, "license", _require_string(self.license, "source.license"))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SourceProvenance":
        raw = _require_mapping(value, "source")
        _reject_unknown(raw, {"project", "revision", "license"}, "source")
        return cls(raw.get("project"), raw.get("revision"), raw.get("license"))

    def to_mapping(self) -> dict[str, Any]:
        return {"project": self.project, "revision": self.revision, "license": self.license}


@dataclass(frozen=True)
class TargetIdentity:
    framework: str
    architecture: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "framework", _require_string(self.framework, "target.framework"))
        object.__setattr__(self, "architecture", _require_identifier(self.architecture, "target.architecture"))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TargetIdentity":
        raw = _require_mapping(value, "target")
        _reject_unknown(raw, {"framework", "architecture"}, "target")
        return cls(raw.get("framework"), raw.get("architecture"))

    def to_mapping(self) -> dict[str, Any]:
        return {"framework": self.framework, "architecture": self.architecture}


@dataclass(frozen=True)
class KnowledgeDeny:
    sources: tuple[str, ...]
    paths: tuple[str, ...]
    tags: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", _require_strings(self.sources, "knowledge_deny.sources", allow_empty=True))
        object.__setattr__(self, "paths", _require_strings(self.paths, "knowledge_deny.paths", allow_empty=True))
        object.__setattr__(self, "tags", _require_strings(self.tags, "knowledge_deny.tags", allow_empty=True))
        if not (self.sources or self.paths or self.tags):
            raise ValueError("knowledge_deny must contain at least one deny selector")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "KnowledgeDeny":
        raw = _require_mapping(value, "knowledge_deny")
        _reject_unknown(raw, {"sources", "paths", "tags"}, "knowledge_deny")
        missing = [key for key in ("sources", "paths", "tags") if key not in raw]
        if missing:
            raise ValueError("knowledge_deny is missing required fields: %s" % ", ".join(missing))
        return cls(raw["sources"], raw["paths"], raw["tags"])

    def to_mapping(self) -> dict[str, Any]:
        return {
            "sources": list(self.sources),
            "paths": list(self.paths),
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class TeacherProvenance:
    schema_version: int
    operator: OperatorIdentity
    source: SourceProvenance
    target: TargetIdentity
    knowledge_deny: KnowledgeDeny

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _require_schema_version(self.schema_version))
        if not isinstance(self.operator, OperatorIdentity):
            raise ValueError("operator must be an OperatorIdentity")
        if not isinstance(self.source, SourceProvenance):
            raise ValueError("source must be a SourceProvenance")
        if not isinstance(self.target, TargetIdentity):
            raise ValueError("target must be a TargetIdentity")
        if not isinstance(self.knowledge_deny, KnowledgeDeny):
            raise ValueError("knowledge_deny must be a KnowledgeDeny")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TeacherProvenance":
        raw = _require_mapping(value, "provenance")
        _reject_unknown(
            raw,
            {"schema_version", "operator", "source", "target", "knowledge_deny"},
            "provenance",
        )
        return cls(
            _require_schema_version(raw.get("schema_version")),
            OperatorIdentity.from_mapping(raw.get("operator")),
            SourceProvenance.from_mapping(raw.get("source")),
            TargetIdentity.from_mapping(raw.get("target")),
            KnowledgeDeny.from_mapping(raw.get("knowledge_deny")),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "operator": self.operator.to_mapping(),
            "source": self.source.to_mapping(),
            "target": self.target.to_mapping(),
            "knowledge_deny": self.knowledge_deny.to_mapping(),
        }


@dataclass(frozen=True)
class TeacherTarget:
    schema_version: int
    teacher_id: str
    geomean_latency_us: float
    latency_us_by_shape: Mapping[str, float]
    geomean_ratio: float
    shape_ratio: float
    measurement_config_hash: str
    knowledge_view_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _require_schema_version(self.schema_version))
        object.__setattr__(self, "teacher_id", _require_identifier(self.teacher_id, "teacher_id"))
        object.__setattr__(
            self,
            "geomean_latency_us",
            _require_positive_finite(self.geomean_latency_us, "geomean_latency_us"),
        )
        if not isinstance(self.latency_us_by_shape, Mapping) or not self.latency_us_by_shape:
            raise ValueError("latency_us_by_shape must be a non-empty object")
        by_shape = {
            _require_string(key, "latency_us_by_shape key"): _require_positive_finite(
                value, "latency_us_by_shape[%s]" % key
            )
            for key, value in self.latency_us_by_shape.items()
        }
        object.__setattr__(self, "latency_us_by_shape", MappingProxyType(by_shape))
        object.__setattr__(self, "geomean_ratio", _require_ratio(self.geomean_ratio, "geomean_ratio"))
        object.__setattr__(self, "shape_ratio", _require_ratio(self.shape_ratio, "shape_ratio"))
        object.__setattr__(
            self,
            "measurement_config_hash",
            _require_sha256(self.measurement_config_hash, "measurement_config_hash"),
        )
        object.__setattr__(
            self,
            "knowledge_view_hash",
            _require_sha256(self.knowledge_view_hash, "knowledge_view_hash"),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TeacherTarget":
        raw = _require_mapping(value, "teacher_target")
        _reject_unknown(
            raw,
            {
                "schema_version",
                "teacher_id",
                "geomean_latency_us",
                "latency_us_by_shape",
                "geomean_ratio",
                "shape_ratio",
                "measurement_config_hash",
                "knowledge_view_hash",
            },
            "teacher_target",
        )
        return cls(
            raw.get("schema_version"),
            raw.get("teacher_id"),
            raw.get("geomean_latency_us"),
            raw.get("latency_us_by_shape"),
            raw.get("geomean_ratio"),
            raw.get("shape_ratio"),
            raw.get("measurement_config_hash"),
            raw.get("knowledge_view_hash"),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "teacher_id": self.teacher_id,
            "geomean_latency_us": self.geomean_latency_us,
            "latency_us_by_shape": dict(self.latency_us_by_shape),
            "geomean_ratio": self.geomean_ratio,
            "shape_ratio": self.shape_ratio,
            "measurement_config_hash": self.measurement_config_hash,
            "knowledge_view_hash": self.knowledge_view_hash,
        }


@dataclass(frozen=True)
class CampaignLock:
    schema_version: int
    campaign_id: str
    teacher_id: str
    platform: str
    architecture: str
    framework: str
    workload_hash: str
    evaluator_hash: str
    measurement_config_hash: str
    knowledge_view_hash: str
    geomean_ratio: float
    shape_ratio: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _require_schema_version(self.schema_version))
        object.__setattr__(self, "campaign_id", _require_identifier(self.campaign_id, "campaign_id"))
        object.__setattr__(self, "teacher_id", _require_identifier(self.teacher_id, "teacher_id"))
        object.__setattr__(self, "platform", _require_string(self.platform, "platform"))
        object.__setattr__(self, "architecture", _require_identifier(self.architecture, "architecture"))
        object.__setattr__(self, "framework", _require_string(self.framework, "framework"))
        for field_name in (
            "workload_hash",
            "evaluator_hash",
            "measurement_config_hash",
            "knowledge_view_hash",
        ):
            object.__setattr__(self, field_name, _require_sha256(getattr(self, field_name), field_name))
        object.__setattr__(self, "geomean_ratio", _require_ratio(self.geomean_ratio, "geomean_ratio"))
        object.__setattr__(self, "shape_ratio", _require_ratio(self.shape_ratio, "shape_ratio"))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CampaignLock":
        raw = _require_mapping(value, "campaign_lock")
        _reject_unknown(
            raw,
            {
                "schema_version",
                "campaign_id",
                "teacher_id",
                "platform",
                "architecture",
                "framework",
                "workload_hash",
                "evaluator_hash",
                "measurement_config_hash",
                "knowledge_view_hash",
                "geomean_ratio",
                "shape_ratio",
            },
            "campaign_lock",
        )
        return cls(
            raw.get("schema_version"),
            raw.get("campaign_id"),
            raw.get("teacher_id"),
            raw.get("platform"),
            raw.get("architecture"),
            raw.get("framework"),
            raw.get("workload_hash"),
            raw.get("evaluator_hash"),
            raw.get("measurement_config_hash"),
            raw.get("knowledge_view_hash"),
            raw.get("geomean_ratio"),
            raw.get("shape_ratio"),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "teacher_id": self.teacher_id,
            "platform": self.platform,
            "architecture": self.architecture,
            "framework": self.framework,
            "workload_hash": self.workload_hash,
            "evaluator_hash": self.evaluator_hash,
            "measurement_config_hash": self.measurement_config_hash,
            "knowledge_view_hash": self.knowledge_view_hash,
            "geomean_ratio": self.geomean_ratio,
            "shape_ratio": self.shape_ratio,
        }

    def fingerprint(self) -> str:
        return _fingerprint(self.to_mapping())


class AbbaStatus(str, Enum):
    NOT_RUN = "NOT_RUN"
    PASS = "PASS"
    FAIL = "FAIL"
    INFRA_ERROR = "INFRA_ERROR"


class CampaignTerminalStatus(str, Enum):
    SUCCESS = "SUCCESS"
    PLATEAU = "PLATEAU"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    INFRA_ERROR = "INFRA_ERROR"
    TEACHER_LEAKAGE_VIOLATION = "TEACHER_LEAKAGE_VIOLATION"


@dataclass(frozen=True)
class TeacherProgress:
    target_id: str
    candidate_to_teacher_geomean_ratio: float
    worst_shape_ratio: float
    worst_shape_key: str
    geomean_gate_met: bool
    shape_gate_met: bool
    provisional_target_met: bool
    abba_status: AbbaStatus
    final_candidate_to_teacher_ratio: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_id", _require_identifier(self.target_id, "target_id"))
        object.__setattr__(
            self,
            "candidate_to_teacher_geomean_ratio",
            _require_positive_finite(
                self.candidate_to_teacher_geomean_ratio,
                "candidate_to_teacher_geomean_ratio",
            ),
        )
        object.__setattr__(
            self,
            "worst_shape_ratio",
            _require_positive_finite(self.worst_shape_ratio, "worst_shape_ratio"),
        )
        object.__setattr__(self, "worst_shape_key", _require_string(self.worst_shape_key, "worst_shape_key"))
        for field_name in ("geomean_gate_met", "shape_gate_met", "provisional_target_met"):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError("%s must be a boolean" % field_name)
        if not isinstance(self.abba_status, AbbaStatus):
            object.__setattr__(self, "abba_status", _enum_value(AbbaStatus, self.abba_status, "abba_status"))
        if self.final_candidate_to_teacher_ratio is not None:
            object.__setattr__(
                self,
                "final_candidate_to_teacher_ratio",
                _require_positive_finite(
                    self.final_candidate_to_teacher_ratio,
                    "final_candidate_to_teacher_ratio",
                ),
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TeacherProgress":
        raw = _require_mapping(value, "teacher_progress")
        _reject_unknown(
            raw,
            {
                "target_id",
                "candidate_to_teacher_geomean_ratio",
                "worst_shape_ratio",
                "worst_shape_key",
                "geomean_gate_met",
                "shape_gate_met",
                "provisional_target_met",
                "abba_status",
                "final_candidate_to_teacher_ratio",
            },
            "teacher_progress",
        )
        return cls(
            raw.get("target_id"),
            raw.get("candidate_to_teacher_geomean_ratio"),
            raw.get("worst_shape_ratio"),
            raw.get("worst_shape_key"),
            raw.get("geomean_gate_met"),
            raw.get("shape_gate_met"),
            raw.get("provisional_target_met"),
            _enum_value(AbbaStatus, raw.get("abba_status"), "abba_status"),
            raw.get("final_candidate_to_teacher_ratio"),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "candidate_to_teacher_geomean_ratio": self.candidate_to_teacher_geomean_ratio,
            "worst_shape_ratio": self.worst_shape_ratio,
            "worst_shape_key": self.worst_shape_key,
            "geomean_gate_met": self.geomean_gate_met,
            "shape_gate_met": self.shape_gate_met,
            "provisional_target_met": self.provisional_target_met,
            "abba_status": self.abba_status.value,
            "final_candidate_to_teacher_ratio": self.final_candidate_to_teacher_ratio,
        }


@dataclass(frozen=True)
class TeacherCampaignResult:
    schema_version: int
    campaign_id: str
    status: CampaignTerminalStatus
    reason: str
    final_version: Optional[str] = None
    final_candidate_to_teacher_ratio: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _require_schema_version(self.schema_version))
        object.__setattr__(self, "campaign_id", _require_identifier(self.campaign_id, "campaign_id"))
        if not isinstance(self.status, CampaignTerminalStatus):
            object.__setattr__(
                self,
                "status",
                _enum_value(CampaignTerminalStatus, self.status, "status"),
            )
        object.__setattr__(self, "reason", _require_string(self.reason, "reason"))
        if self.final_version is not None:
            version = _require_string(self.final_version, "final_version")
            if not re.fullmatch(r"v\d+", version):
                raise ValueError("final_version must use v<N> format")
            object.__setattr__(self, "final_version", version)
        if self.final_candidate_to_teacher_ratio is not None:
            object.__setattr__(
                self,
                "final_candidate_to_teacher_ratio",
                _require_positive_finite(
                    self.final_candidate_to_teacher_ratio,
                    "final_candidate_to_teacher_ratio",
                ),
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TeacherCampaignResult":
        raw = _require_mapping(value, "campaign_result")
        _reject_unknown(
            raw,
            {
                "schema_version",
                "campaign_id",
                "status",
                "reason",
                "final_version",
                "final_candidate_to_teacher_ratio",
            },
            "campaign_result",
        )
        return cls(
            raw.get("schema_version"),
            raw.get("campaign_id"),
            _enum_value(CampaignTerminalStatus, raw.get("status"), "status"),
            raw.get("reason"),
            raw.get("final_version"),
            raw.get("final_candidate_to_teacher_ratio"),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "status": self.status.value,
            "reason": self.reason,
            "final_version": self.final_version,
            "final_candidate_to_teacher_ratio": self.final_candidate_to_teacher_ratio,
        }
