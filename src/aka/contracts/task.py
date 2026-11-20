"""Versioned task adapter protocol. No benchmark layout or execution vendor is required.

Adapters are trusted, caller-selected code. Candidates never implement this protocol.
The host owns identities, execution receipts, persistence and source snapshots.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol
import math


PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class Metric:
    name: str
    value: float
    unit: str
    context: str = ""

    def __post_init__(self):
        if not self.name or not self.unit or type(self.value) not in (int, float) or not math.isfinite(self.value):
            raise ValueError("Metrics require a name, unit and finite numeric value")


@dataclass(frozen=True)
class Assessment:
    verdict: str
    complete: bool
    metrics: tuple[Metric, ...] = ()
    environment: str = ""
    reason: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)
    failure_kind: str = ""

    def __post_init__(self):
        if self.verdict not in {"pass", "fail", "unknown"} or type(self.complete) is not bool:
            raise ValueError("Assessment requires an explicit verdict and completeness")
        if self.failure_kind not in {"", "build", "correctness", "implementation", "execution", "infrastructure", "unknown"}:
            raise ValueError("Unknown assessment failure kind")
        object.__setattr__(self, "metrics", tuple(self.metrics))
        keys = [(m.name, m.context) for m in self.metrics]
        if len(set(keys)) != len(keys):
            raise ValueError("Metric name/context pairs must be unique")


@dataclass(frozen=True)
class EvaluationPlan:
    """An acceptance command or native request over frozen task/candidate content.

Local commands are explicit development/test backends. Remote request formats belong
to the selected executor, not to the persistent task workflow.
The command kind uses the shared Executor protocol on any selected backend.
Agate-native compatibility kinds are eval and dev; research is a command.
"""
    kind: str
    request: Mapping[str, Any]


@dataclass(frozen=True)
class ExecutionReceipt:
    status: str
    payload: Mapping[str, Any]
    job_id: str = ""


@dataclass(frozen=True)
class TaskContext:
    contract: Path
    candidate: Path
    config: Mapping[str, Any]
    target: Mapping[str, Any]
    scope: str


class TaskAdapter(Protocol):
    """All methods use caller-owned, frozen inputs; no method mutates Agent source."""
    protocol_version: int
    capabilities: frozenset[str]

    def validate(self, contract: Path, config: Mapping[str, Any]) -> None: ...
    def instructions(self, contract: Path, config: Mapping[str, Any]) -> str: ...
    def prepare(self, context: TaskContext) -> EvaluationPlan: ...
    def interpret(self, context: TaskContext, receipt: ExecutionReceipt) -> Assessment: ...
    def delivery(self, contract: Path, config: Mapping[str, Any]) -> Mapping[str, Any]: ...


class Objective(Protocol):
    """A total or partial order. Cross-environment comparison is always explicit."""
    def describe(self) -> str: ...
    def eligible(self, assessment: Assessment) -> bool: ...
    def compare(self, candidate: Assessment, incumbent: Assessment) -> str: ...
