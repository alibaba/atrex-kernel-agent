"""Synchronous original-behavior experiment coordination.

The host owns injected dependencies and supplies an executor with its transport
binding already established. Evaluation owns measurement plans and interpretation.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Protocol
import math

from .execution import CommandSpec, JobResult
from .evaluation import EvaluationFailure, EvaluationRequest, VerificationResult
from .experiment_identity import ExperimentRef

class Experiment(Protocol):
    def run(self, request: EvaluationRequest) -> VerificationResult: ...

    def run_command(self, command: CommandSpec) -> JobResult:
        """Execute once without requiring an Evaluator or changing its result."""
        ...


class ExperimentExecutionError(RuntimeError):
    """Original execution/aggregation failure, distinct from preparation errors."""


def _freeze(value):
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("experiment snapshot keys must be strings")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or type(value) in (str, int, float, bool):
        if type(value) is float and not math.isfinite(value):
            raise ValueError("experiment snapshot numbers must be finite")
        return value
    raise TypeError("experiment snapshot fields must contain JSON data")


def _json(value):
    if isinstance(value, Mapping):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json(item) for item in value]
    return value


@dataclass(frozen=True)
class ExperimentArtifact:
    """Host-private artifact identity; storage locations are not public fields."""
    experiment: ExperimentRef
    artifact_id: str
    role: str
    sha256: str
    size_bytes: int
    visibility: Literal["host"] = "host"

    def as_dict(self) -> dict:
        return {"experiment": self.experiment.as_dict(), "artifact_id": self.artifact_id,
                "role": self.role, "sha256": self.sha256, "size_bytes": self.size_bytes,
                "visibility": self.visibility}


@dataclass(frozen=True)
class ExperimentFailure:
    stage: str
    code: str
    exception_type: str = ""
    message: str = ""

    def as_dict(self) -> dict:
        return {"stage": self.stage, "code": self.code,
                "exception_type": self.exception_type, "message": self.message}


@dataclass(frozen=True)
class ExperimentStep:
    task_id: str
    index: int
    argv: tuple[str, ...]
    operation: str
    timeout: int
    hardware: str
    environment_keys: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {"task_id": self.task_id, "index": self.index, "argv": list(self.argv),
                "operation": self.operation, "timeout": self.timeout, "hardware": self.hardware,
                "environment_keys": list(self.environment_keys)}


@dataclass(frozen=True)
class ExperimentAttempt:
    attempt_id: str
    task_id: str
    disposition: Literal["intent", "not_submitted", "terminal", "unknown"]
    result: JobResult | None = None
    failure: ExperimentFailure | None = None

    def as_dict(self) -> dict:
        return {"attempt_id": self.attempt_id, "task_id": self.task_id,
                "disposition": self.disposition,
                "result": self.result.as_dict() if self.result else None,
                "failure": self.failure.as_dict() if self.failure else None}


@dataclass(frozen=True)
class ExperimentEvaluation:
    """Immutable stored VerificationResult facts; artifact is an opaque artifact ID.

    Each run follows VerificationRun.as_dict() fields and retains method-specific
    JSON result semantics. This is host evidence, not public tool feedback.
    """
    gate: str
    candidate_latency_us: float | None
    incumbent_latency_us: float | None
    improvement_pct: float | None
    candidate_performance_score: float | None = None
    incumbent_performance_score: float | None = None
    runs: tuple[Mapping[str, Any], ...] = ()
    error: str | None = None
    artifact: str | None = None
    failure: EvaluationFailure | None = None
    experiment_ref: ExperimentRef | None = None

    def __post_init__(self):
        object.__setattr__(self, "runs", tuple(_freeze(item) for item in self.runs))

    def as_dict(self) -> dict:
        value = {"gate": self.gate, "candidate_latency_us": self.candidate_latency_us,
                 "incumbent_latency_us": self.incumbent_latency_us, "improvement_pct": self.improvement_pct,
                 "candidate_performance_score": self.candidate_performance_score,
                 "incumbent_performance_score": self.incumbent_performance_score,
                 "runs": _json(self.runs), "error": self.error, "artifact": self.artifact}
        if self.failure is not None:
            from dataclasses import asdict
            value["failure"] = asdict(self.failure)
        if self.experiment_ref is not None:
            value["experiment_ref"] = self.experiment_ref.as_dict()
        return value


@dataclass(frozen=True)
class ExperimentRecord:
    """experiment-record/1 host snapshot, independent of its persistence format.

    evaluation uses VerificationResult.as_dict() semantics, with artifact locations
    replaced by artifact IDs. It is not an Agent-safe feedback projection.
    result_recorded means a return value was saved; terminal additionally requires
    every attempt to be terminal or not_submitted. Unknown facts stay unknown.
    """
    ref: ExperimentRef
    request_id: str
    kind: Literal["evaluation", "command"]
    purpose: Literal["development", "formal"]
    record_version: int
    lifecycle: Literal["registered", "prepared", "running", "result_recorded", "failed", "terminal"]
    phase: str
    request: Mapping[str, Any]
    request_digest: str
    plan_digest: str | None = None
    steps: tuple[ExperimentStep, ...] = ()
    attempts: tuple[ExperimentAttempt, ...] = ()
    evaluation: ExperimentEvaluation | None = None
    artifacts: tuple[ExperimentArtifact, ...] = ()
    failure: ExperimentFailure | None = None
    outcome: Literal["completed", "failed", "inconclusive"] | None = None
    created_at: str = ""
    updated_at: str = ""
    protocol_version: int = 1

    def __post_init__(self):
        if self.protocol_version != 1 or self.record_version < 1:
            raise ValueError("unsupported experiment record version")
        object.__setattr__(self, "request", _freeze(self.request))
        for name in ("steps", "attempts", "artifacts"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.evaluation and self.evaluation.experiment_ref != self.ref:
            raise ValueError("evaluation belongs to another experiment")
        if any(item.experiment != self.ref for item in self.artifacts):
            raise ValueError("artifact belongs to another experiment")
        if len({item.task_id for item in self.steps}) != len(self.steps):
            raise ValueError("duplicate experiment task")
        if len({item.attempt_id for item in self.attempts}) != len(self.attempts) or any(
            item.task_id not in {step.task_id for step in self.steps} for item in self.attempts
        ):
            raise ValueError("invalid experiment attempt identity")
        if self.lifecycle == "terminal" and any(
            item.disposition not in {"terminal", "not_submitted"} for item in self.attempts
        ):
            raise ValueError("unknown execution attempts cannot terminate an experiment")

    def as_dict(self) -> dict:
        return {"protocol_version": self.protocol_version, "ref": self.ref.as_dict(),
                "request_id": self.request_id, "kind": self.kind, "purpose": self.purpose,
                "record_version": self.record_version, "lifecycle": self.lifecycle,
                "phase": self.phase, "request": _json(self.request),
                "request_digest": self.request_digest, "plan_digest": self.plan_digest,
                "steps": [item.as_dict() for item in self.steps],
                "attempts": [item.as_dict() for item in self.attempts],
                "evaluation": self.evaluation.as_dict() if self.evaluation else None,
                "artifacts": [item.as_dict() for item in self.artifacts],
                "failure": self.failure.as_dict() if self.failure else None,
                "outcome": self.outcome, "created_at": self.created_at,
                "updated_at": self.updated_at}


class ExperimentRecords(Protocol):
    """Optional record-reading capability; get/find never execute or interpret."""
    def get(self, ref: ExperimentRef) -> ExperimentRecord: ...
    def find(self, request_id: str) -> ExperimentRecord: ...
    def unresolved(self) -> tuple[ExperimentRef, ...]: ...
    def read_artifact(self, artifact: ExperimentArtifact) -> bytes: ...


class RecordedExperiment(ExperimentRecords, Protocol):
    def run_recorded(
        self, request: EvaluationRequest | CommandSpec, *, request_id: str
    ) -> ExperimentRecord:
        """Persist an evaluation or command, preserving original exceptions.

        Completed calls may return result_recorded when execution termination is
        unknown. Replaying a stored result does not call the Executor again.
        """
        ...


class ExperimentRecordError(RuntimeError):
    def __init__(self, code: str, ref: ExperimentRef | None = None):
        self.code = code
        self.ref = ref
        super().__init__(code)
