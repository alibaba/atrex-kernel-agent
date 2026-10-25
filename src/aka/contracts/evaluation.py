"""Original evaluation facts and prepare/interpret contract for PR113."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

from .execution import CommandSpec, JobResult
from .workspace import CandidateWorkspace
from .problem import EvaluationInput


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


@dataclass(frozen=True)
class PreparedEvaluation:
    steps: tuple[CommandSpec, ...]
    max_parallelism: int
    context: object


class Evaluator(Protocol):
    def prepare(self, request: EvaluationRequest) -> PreparedEvaluation: ...
    def record_step(
        self, prepared: PreparedEvaluation, index: int, result: JobResult
    ) -> None: ...
    def interpret(
        self, prepared: PreparedEvaluation, results: Sequence[JobResult]
    ) -> VerificationResult: ...


@dataclass(frozen=True)
class EvaluationConfiguration:
    timeout: int = derived_evaluation_timeout()
    repeats: int = DEFAULT_ABBA_REPEATS
    per_run_timeout: int = DEFAULT_ABBA_RUN_TIMEOUT
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
