"""Original evaluation facts and prepare/interpret contract for PR113."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

from .execution import CommandSpec, JobResult
from .workspace import CandidateWorkspace
from .problem import EvaluationInput
from .experiment_identity import ExperimentRef


@dataclass(frozen=True)
class NumericalProbeResult:
    """Interpreted probe facts; no raw receipt, private inputs or exception text."""

    status: str
    reason: str
    receipt_status: str
    expected_probes: int | None = None
    observed_probes: int | None = None
    failed_probes: int = 0
    metrics: Mapping[str, int | float | None] = field(default_factory=dict)
    nonfinite_outputs: tuple[str, ...] = ()
    exit_code: int | None = None
    diagnosis: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(self, "nonfinite_outputs", tuple(self.nonfinite_outputs))

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "expected_probes": self.expected_probes,
            "observed_probes": self.observed_probes,
            "failed_probes": self.failed_probes,
            "metrics": dict(self.metrics),
            "nonfinite_outputs": list(self.nonfinite_outputs),
            "exit_code": self.exit_code,
            "diagnosis": self.diagnosis,
        }


@dataclass(frozen=True)
class VerificationRun:
    revision: str
    repeat: int
    exit_code: int
    result: dict[str, Any] | None
    stdout_tail: str = ""
    stderr_tail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationResult:
    gate: str
    candidate_latency_us: float | None
    incumbent_latency_us: float | None
    improvement_pct: float | None
    runs: list[VerificationRun] = field(default_factory=list)
    error: str = ""
    artifact: str = ""
    candidate_performance_score: float | None = None
    incumbent_performance_score: float | None = None
    failure: EvaluationFailure | None = None
    experiment_ref: ExperimentRef | None = None

    @property
    def passed(self) -> bool:
        return self.gate == "PASS" and not self.error

    def as_dict(self) -> dict[str, Any]:
        value = {
            "gate": self.gate,
            "candidate_latency_us": self.candidate_latency_us,
            "incumbent_latency_us": self.incumbent_latency_us,
            "improvement_pct": self.improvement_pct,
            "candidate_performance_score": self.candidate_performance_score,
            "incumbent_performance_score": self.incumbent_performance_score,
            "runs": [run.as_dict() for run in self.runs],
            "error": self.error or None,
            "artifact": self.artifact or None,
        }
        if self.failure is not None:
            value["failure"] = asdict(self.failure)
        if self.experiment_ref is not None:
            value["experiment_ref"] = self.experiment_ref.as_dict()
        return value


@dataclass(frozen=True)
class EvaluationFailure:
    category: str
    code: str
    message: str
    candidate_repairable: bool = False


class EvaluationUnavailable(RuntimeError):
    """Validation produced no candidate verdict; baseline callers must stop before repair."""
    def __init__(self, failure: EvaluationFailure):
        self.failure = failure
        super().__init__(failure.message)


@dataclass(frozen=True)
class EvaluationRequest:
    workspace: Path
    mode: str = "abba"
    base_commit: str = ""
    candidate_commit: str = ""
    changed_paths: tuple[str, ...] = ()
    version: int = 0
    memory_version: int = 0
    canonical_workspace: Path | None = None
    expected_shape_ids: frozenset[str] | None = None
    experiment_ref: ExperimentRef | None = None


@dataclass(frozen=True)
class PreparedEvaluation:
    steps: tuple[CommandSpec, ...]
    max_parallelism: int
    context: object
    retry_keys: tuple[str, ...] = ()


class Evaluator(Protocol):
    def prepare(self, request: EvaluationRequest) -> PreparedEvaluation: ...
    def record_step(
        self, prepared: PreparedEvaluation, index: int, result: JobResult
    ) -> None: ...
    def interpret(
        self, prepared: PreparedEvaluation, results: Sequence[JobResult]
    ) -> VerificationResult: ...


@dataclass(frozen=True)
class MeasurementArtifact:
    """A completed method-owned file offered to Experiment for archival."""
    path: Path
    role: str = "measurement"


class MeasurementArtifacts(Protocol):
    """Optional artifact discovery; never runs commands or returns private directories."""
    def artifacts(
        self, prepared: PreparedEvaluation, index: int
    ) -> tuple[MeasurementArtifact, ...]: ...


@dataclass(frozen=True)
class EvaluationConfiguration:
    timeout: int = 600
    repeats: int = 2
    per_run_timeout: int = 120
    shape_batch_size: int = 4
    max_parallelism: int = 4
    private_reference_dir: Path | None = None
    native: bool = False


class EvaluatorFactory(Protocol):
    """Create isolated measurement state for each configured experiment.

    The host borrows the returned Evaluator for one operation. The provider owns
    command preparation and interpretation; the host owns execution and retries.
    """

    def create(self, configuration: EvaluationConfiguration, *,
               candidate_workspace: CandidateWorkspace,
               evaluation_input: EvaluationInput) -> Evaluator: ...
