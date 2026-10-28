"""Original evaluation facts and prepare/interpret contract for PR113."""

from __future__ import annotations
from .workflow_records import CanonicalMeasurementReader

from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

from .execution import CommandSpec, JobResult
from .workspace import CandidateWorkspace
from .problem import EvaluationInput
from .experiment_identity import ExperimentRef
from .measurement import MeasurementIdentity, MeasurementPoint, MeasurementAssessment, MeasurementDiagnostic


@dataclass(frozen=True)
class MeasurementArtifact:
    """Method-owned output content, offered to Experiment without writing a file.

    name is a logical filename within the prepared output directory, not a host
    path. Experiment owns serialization publication, integrity and archival.
    Content is host-private evidence, not an Agent feedback projection.
    """
    name: str
    content: bytes = field(repr=False)
    role: str = "measurement"
    content_type: str = "application/json"

    def __post_init__(self):
        if not self.name or Path(self.name).name != self.name or self.name in {".", ".."}:
            raise ValueError("measurement artifact requires a logical filename")
        if not isinstance(self.content, bytes):
            raise TypeError("measurement artifact content must be immutable bytes")


@dataclass(frozen=True)
class StepEvaluation:
    """Completed step outputs; returned before the final interpretation."""
    artifacts: tuple[MeasurementArtifact, ...] = ()
    assessment: MeasurementAssessment | None = None

    def __post_init__(self):
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        if self.assessment is not None and not isinstance(self.assessment, MeasurementAssessment):
            raise TypeError("step assessment must use the public measurement contract")


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
    artifacts: tuple[MeasurementArtifact, ...] = field(default=(), repr=False)
    measurements: tuple[MeasurementAssessment, ...] = ()

    def __post_init__(self):
        self.artifacts = tuple(self.artifacts)
        self.measurements = tuple(self.measurements)
        if any(not isinstance(item, MeasurementAssessment) for item in self.measurements):
            raise TypeError("verification measurements must use the public contract")

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
        if self.measurements:
            value["measurements"] = [item.as_dict() for item in self.measurements]
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
    numerical_probe: NumericalProbeRequest | None = None

    def __post_init__(self):
        if self.numerical_probe is not None and not isinstance(self.numerical_probe, NumericalProbeRequest):
            raise TypeError("numerical probe must use the public request contract")
        if (self.mode == "numerical") != (self.numerical_probe is not None):
            raise ValueError("numerical mode requires its probe request exclusively")


@dataclass(frozen=True)
class PreparationFile:
    """Immutable execution input, materialized and archived only by Experiment."""
    path: str
    content: bytes = field(repr=False)
    mode: int = 0o644
    content_type: str = "application/octet-stream"

    def __post_init__(self):
        path = Path(self.path)
        if not self.path or path.is_absolute() or ".." in path.parts or path == Path("."):
            raise ValueError("preparation file requires a workspace-relative path")
        if not isinstance(self.content, bytes):
            raise TypeError("preparation content must be immutable bytes")
        if type(self.mode) is not int or not 0 <= self.mode <= 0o777:
            raise ValueError("preparation file requires ordinary permission bits")


@dataclass(frozen=True)
class NumericalProbeRequest:
    """Frozen reviewer suite and candidate content; Evaluation owns probe preparation."""
    suite_json: str
    rotation: str
    candidate_identity: str
    candidate_files: tuple[PreparationFile, ...]

    def __post_init__(self):
        for value in (self.suite_json, self.rotation, self.candidate_identity):
            if not isinstance(value, str) or not value:
                raise ValueError("numerical probe requires nonempty text identities and suite")
        files = tuple(self.candidate_files)
        if not files or any(not isinstance(item, PreparationFile) for item in files):
            raise TypeError("numerical candidate files must use PreparationFile values")
        object.__setattr__(self, "candidate_files", files)


@dataclass(frozen=True)
class PreparedEvaluation:
    """Commands, immutable input content and an optional output mirror directory.

    Experiment owns all input/output materialization; interpretation context stays
    opaque. Evaluator.prepare describes inputs without writing workspace files.
    """
    steps: tuple[CommandSpec, ...]
    max_parallelism: int
    context: object
    retry_keys: tuple[str, ...] = ()
    output_directory: Path | None = None
    inputs: tuple[PreparationFile, ...] = ()
    fallback_steps: tuple[CommandSpec, ...] = ()
    measurement_identities: tuple[MeasurementIdentity, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "fallback_steps", tuple(self.fallback_steps))
        inputs = tuple(self.inputs)
        if any(not isinstance(item, PreparationFile) for item in inputs):
            raise TypeError("prepared inputs must be PreparationFile values")
        object.__setattr__(self, "inputs", inputs)
        identities = tuple(self.measurement_identities)
        if identities and len(identities) != len(self.steps):
            raise ValueError("prepared measurement identities must match execution steps")
        if any(not isinstance(item, MeasurementIdentity) for item in identities):
            raise TypeError("prepared measurement identities must use the public contract")
        object.__setattr__(self, "measurement_identities", identities)


class Evaluator(Protocol):
    def prepare(self, request: EvaluationRequest) -> PreparedEvaluation: ...
    # Implementations may offer fallback(prepared, results) for a prepared alternative.
    # Hosts invoke it only after all original receipts have been observed.
    # Implementations may offer pure diagnose_step(prepared, index, result) for
    # unavailable receipts, before the host raises on Execution transport facts.
    def observe_step(
        self, prepared: PreparedEvaluation, index: int, result: JobResult
    ) -> StepEvaluation: ...
    def interpret(
        self, prepared: PreparedEvaluation, results: Sequence[JobResult]
    ) -> VerificationResult: ...

class CommandEvaluator(Protocol):
    """Prepare a command measurement and interpret raw receipts for Experiment."""
    def prepare(self, command: CommandSpec) -> PreparedEvaluation: ...
    def observe_step(self, prepared: PreparedEvaluation, index: int, result: JobResult) -> StepEvaluation: ...
    def fallback(self, prepared: PreparedEvaluation, results: Sequence[JobResult]) -> tuple[CommandSpec, ...]: ...
    def interpret(self, prepared: PreparedEvaluation, results: Sequence[JobResult]) -> JobResult: ...


@dataclass(frozen=True)
class EvaluationConfiguration:
    timeout: int = 600
    repeats: int = 2
    per_run_timeout: int = 120
    shape_batch_size: int = 4
    max_parallelism: int = 4
    private_reference_dir: Path | None = None
    native: bool = False
    hardware: str = ""
    environment: Mapping[str, str] = field(default_factory=dict)
    measurement_reader: CanonicalMeasurementReader | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "environment", MappingProxyType(dict(self.environment)))


class EvaluatorFactory(Protocol):
    """Create isolated measurement state for each configured experiment.

    The host borrows the returned Evaluator for one operation. The provider owns
    command preparation and interpretation; the host owns execution and retries.
    """

    def create(self, configuration: EvaluationConfiguration, *,
               candidate_workspace: CandidateWorkspace,
               evaluation_input: EvaluationInput) -> Evaluator: ...
