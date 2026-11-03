"""Original evaluation facts and prepare/interpret contract for PR113."""

from __future__ import annotations
from .workflow_records import CanonicalMeasurementReader

from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Literal, Mapping, Protocol, Sequence, TYPE_CHECKING
from aka.core.keys import ServiceKey

from .execution import BackendCapabilities, CommandSpec, ExecutionFile, JobResult
from .workspace import CandidateWorkspace
from .problem import EvaluationInput
from .experiment_identity import ExperimentRef
from .measurement import MeasurementIdentity, MeasurementPoint, MeasurementAssessment, MeasurementDiagnostic

if TYPE_CHECKING:
    from .experiment import RetainedEpisodeReceipt


EVALUATION_EPISODE_INTERPRETER = ServiceKey(
    "evaluation_episode_interpreter", "EvaluationEpisodeInterpreter",
    module="aka.contracts.evaluation",
)


@dataclass(frozen=True)
class EpisodeMeasurementFacts:
    """Interpreted retained measurements, without a Workflow acceptance decision."""
    operation: str
    timestamp: str | None
    source_sha256: Mapping[str, str | None]
    gate: str | None
    correctness: str | None
    complete: bool
    latency_us: float | None
    case_latencies_us: Mapping[str, float]
    expected_case_ids: tuple[str, ...]
    objective: str | None
    score: float | None
    metrics: Mapping[str, float]
    assessment: MeasurementAssessment | None = None

    def __post_init__(self):
        for name in ("source_sha256", "case_latencies_us", "metrics"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        object.__setattr__(self, "expected_case_ids", tuple(self.expected_case_ids))


class EvaluationEpisodeInterpreter(Protocol):
    def interpret_episode_receipts(
        self, receipts: Iterable[RetainedEpisodeReceipt], *,
        expected_case_ids: tuple[str, ...],
    ) -> tuple[EpisodeMeasurementFacts, ...]:
        """Interpret retained method content without running or approving reuse."""
        ...


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
class EpisodeEvaluationContent:
    """Interpreted method content for host retention, without storage layout."""
    operation: str
    result_json: bytes = field(repr=False)
    job_id: str | None = None

    def __post_init__(self):
        import json
        if not isinstance(self.operation, str) or not self.operation:
            raise ValueError("retained evaluation operation must be nonempty")
        if not isinstance(self.result_json, bytes):
            raise TypeError("retained evaluation content must be immutable bytes")
        if not isinstance(json.loads(self.result_json), dict):
            raise ValueError("retained evaluation content must be a JSON object")
        if self.job_id is not None and not isinstance(self.job_id, str):
            raise TypeError("retained evaluation Job metadata must be a string or None")


@dataclass(frozen=True)
class StepEvaluation:
    """Completed step outputs; returned before the final interpretation."""
    artifacts: tuple[MeasurementArtifact, ...] = ()
    assessment: MeasurementAssessment | None = None
    workspace_files: tuple[ExecutionFile, ...] = field(default=(), repr=False)

    def __post_init__(self):
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        if self.assessment is not None and not isinstance(self.assessment, MeasurementAssessment):
            raise TypeError("step assessment must use the public measurement contract")
        object.__setattr__(self, "workspace_files", tuple(self.workspace_files))
        if any(not isinstance(item, ExecutionFile) for item in self.workspace_files):
            raise TypeError("step workspace outputs require ExecutionFile values")


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


# These are execution-policy defaults shared by the workflow configuration and
# Light's native evaluator. The enclosing allocation is derived from the work
# it contains; it is not an independent timeout users must keep in sync.
# Some production Shapes spend more than five minutes in input materialization
# plus the first JIT touch. Keep candidate construction/correctness distinct
# from the shorter steady-state performance budget.
DEFAULT_CANDIDATE_TIMEOUT = 600
DEFAULT_PERFORMANCE_TIMEOUT = 300
# One A or B run uses the same default execution budget as an ordinary
# performance measurement. Operators may still override either limit.
DEFAULT_ABBA_RUN_TIMEOUT = DEFAULT_PERFORMANCE_TIMEOUT
DEFAULT_ABBA_REPEATS = 2
ABBA_SCHEDULING_OVERHEAD = 30
MAX_TYPED_GATEWAY_JOB_TIMEOUT = 10_800


def derived_evaluation_timeout(
    *,
    repeats: int = DEFAULT_ABBA_REPEATS,
    per_run_timeout: int = DEFAULT_ABBA_RUN_TIMEOUT,
    candidate_timeout: int = DEFAULT_CANDIDATE_TIMEOUT,
    performance_timeout: int = DEFAULT_PERFORMANCE_TIMEOUT,
) -> int:
    """Return one allocation large enough for either Eval or A/B/B/A."""
    values = {
        "repeats": repeats,
        "per_run_timeout": per_run_timeout,
        "candidate_timeout": candidate_timeout,
        "performance_timeout": performance_timeout,
    }
    if any(type(value) is not int or value <= 0 for value in values.values()):
        raise ValueError("evaluation timing inputs must be positive integers")
    return max(
        candidate_timeout + performance_timeout,
        2 * repeats * per_run_timeout + ABBA_SCHEDULING_OVERHEAD,
    )


def typed_gateway_job_timeout(command_timeout: int, queue_wait_grace: int) -> int:
    """Validate the deadline sent to Agate's typed-job API.

    Agate rejects typed job deadlines above three hours.  Silently clamping a
    larger local allocation can terminate a valid Eval/ABBA schedule before its
    configured inner limits expire, so reject the impossible configuration
    before submission instead.
    """
    if type(command_timeout) is not int or command_timeout <= 0:
        raise ValueError("Gateway command timeout must be a positive integer")
    if type(queue_wait_grace) is not int or queue_wait_grace < 0:
        raise ValueError("Gateway queue wait grace must be a non-negative integer")
    requested = command_timeout + queue_wait_grace
    if requested > MAX_TYPED_GATEWAY_JOB_TIMEOUT:
        raise ValueError(
            "Configured Gateway allocation exceeds Agate's 10800-second typed-job limit: "
            f"command timeout {command_timeout}s + queue grace {queue_wait_grace}s = "
            f"{requested}s. Reduce the allocation, ABBA repeats/run timeout, or queue grace."
        )
    return requested


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


def _shape_latencies(values):
    import math
    values = dict(values)
    if any(not isinstance(shape, str) or type(value) not in (int, float)
           or not math.isfinite(value) or value <= 0 for shape, value in values.items()):
        raise ValueError("shape latency facts require string identities and positive finite values")
    return MappingProxyType(values)


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
    episode_evaluations: tuple[EpisodeEvaluationContent, ...] = field(default=(), repr=False)
    candidate_latency_us_by_shape: Mapping[str, float] = field(default_factory=dict, kw_only=True)
    incumbent_latency_us_by_shape: Mapping[str, float] = field(default_factory=dict, kw_only=True)

    def __post_init__(self):
        self.candidate_latency_us_by_shape = _shape_latencies(self.candidate_latency_us_by_shape)
        self.incumbent_latency_us_by_shape = _shape_latencies(self.incumbent_latency_us_by_shape)
        self.artifacts = tuple(self.artifacts)
        self.measurements = tuple(self.measurements)
        self.episode_evaluations = tuple(self.episode_evaluations)
        if any(not isinstance(item, MeasurementAssessment) for item in self.measurements):
            raise TypeError("verification measurements must use the public contract")
        if any(not isinstance(item, EpisodeEvaluationContent) for item in self.episode_evaluations):
            raise TypeError("retained evaluations require EpisodeEvaluationContent values")

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
        if self.candidate_latency_us_by_shape:
            value["candidate_latency_us_by_shape"] = dict(self.candidate_latency_us_by_shape)
        if self.incumbent_latency_us_by_shape:
            value["incumbent_latency_us_by_shape"] = dict(self.incumbent_latency_us_by_shape)
        return value


@dataclass(frozen=True)
class EvaluationFailure:
    category: str
    code: str
    message: str
    candidate_repairable: bool = False


@dataclass(frozen=True)
class CommandEvaluation:
    """Evaluation interpretation; Experiment publishes outputs and returns result."""
    result: JobResult
    verification: VerificationResult | None = None
    artifacts: tuple[MeasurementArtifact, ...] = ()
    episode_evaluations: tuple[EpisodeEvaluationContent, ...] = field(default=(), repr=False)

    def __post_init__(self):
        if not isinstance(self.result, JobResult):
            raise TypeError("command evaluation requires JobResult")
        if self.verification is not None and not isinstance(self.verification, VerificationResult):
            raise TypeError("command verification requires VerificationResult")
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "episode_evaluations", tuple(self.episode_evaluations))
        if any(not isinstance(item, MeasurementArtifact) for item in self.artifacts):
            raise TypeError("command artifacts require MeasurementArtifact values")
        if any(not isinstance(item, EpisodeEvaluationContent) for item in self.episode_evaluations):
            raise TypeError("retained evaluations require EpisodeEvaluationContent values")


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
    def interpret(self, prepared: PreparedEvaluation, results: Sequence[JobResult]) -> CommandEvaluation: ...


@dataclass(frozen=True)
class EvaluationConfiguration:
    timeout: int = derived_evaluation_timeout()
    repeats: int = DEFAULT_ABBA_REPEATS
    per_run_timeout: int = DEFAULT_ABBA_RUN_TIMEOUT
    shape_batch_size: int = 4
    max_parallelism: int = 4
    private_reference_dir: Path | None = None
    native: bool = False
    hardware: str = ""
    environment: Mapping[str, str] = field(default_factory=dict)
    measurement_reader: CanonicalMeasurementReader | None = field(default=None, repr=False, compare=False)
    execution_capabilities: BackendCapabilities | None = None
    comparison_method: Literal["legacy-abba", "native-abba"] = "legacy-abba"
    measurement_repetitions: int = 1

    def __post_init__(self):
        if self.comparison_method not in {"legacy-abba", "native-abba"}:
            raise ValueError("unsupported comparison method")
        if type(self.measurement_repetitions) is not int or self.measurement_repetitions not in {1, 3}:
            raise ValueError("measurement repetitions must be 1 or 3")
        object.__setattr__(self, "environment", MappingProxyType(dict(self.environment)))
        if self.execution_capabilities is not None and not isinstance(self.execution_capabilities, BackendCapabilities):
            raise TypeError("evaluation preparation requires bound BackendCapabilities")


class EvaluatorFactory(Protocol):
    """Create isolated measurement state for each configured experiment.

    The host borrows the returned Evaluator for one operation. The provider owns
    command preparation and interpretation; the host owns execution and retries.
    """

    def create(self, configuration: EvaluationConfiguration, *,
               candidate_workspace: CandidateWorkspace,
               evaluation_input: EvaluationInput) -> Evaluator: ...
    def create_command(self, configuration: EvaluationConfiguration, *,
                       evaluation_input: EvaluationInput | None) -> CommandEvaluator:
        """Optional command-preparation capability using this selected method/configuration."""
        ...
