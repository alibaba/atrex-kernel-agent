from __future__ import annotations

from aka.contracts.evaluation import (
    EpisodeEvaluationContent,
    EvaluationFailure,
    MeasurementArtifact,
    VerificationRun,
    VerificationResult as SharedVerificationResult,
)
from aka.contracts.experiment_identity import ExperimentRef
from aka.contracts.measurement import MeasurementAssessment

from dataclasses import asdict, dataclass, field
from typing import Any

from aka.contracts.agent import (
    AgentRuntimeCapabilities,
    NormalizedAgentEvent,
    TokenUsage,
)

@dataclass(frozen=True)
class InvocationObservation:
    terminal_usage: TokenUsage
    events: tuple[NormalizedAgentEvent, ...]
    capabilities: AgentRuntimeCapabilities
    observation_errors: tuple[str, ...] = ()
    resume_usage_qualified: bool = False
    session_id: str = ""
    purpose: str = "optimization"
    failure_kind: str | None = None
    provider_error_code: str | None = None


@dataclass
class SessionResult:
    exit_status: int
    timed_out: bool
    tokens: int
    session_id: str
    resume_count: int
    stdout_tail: str = ""
    stderr_tail: str = ""
    completion_diagnosis: str = ""
    invocations: tuple[InvocationObservation, ...] = ()
    output_limit_recoveries: int = 0
    dependency_policy_recoveries: int = 0
    terminal_failure_kind: str | None = None


@dataclass
class VerificationResult(SharedVerificationResult):
    # Keep the Workflow's original positional receipt fields. Public evidence
    # additions stay keyword-only so they cannot shift those receipt fields.
    failure: EvaluationFailure | None = field(default=None, kw_only=True)
    experiment_ref: ExperimentRef | None = field(default=None, kw_only=True)
    artifacts: tuple[MeasurementArtifact, ...] = field(default=(), repr=False, kw_only=True)
    measurements: tuple[MeasurementAssessment, ...] = field(default=(), kw_only=True)
    episode_evaluations: tuple[EpisodeEvaluationContent, ...] = field(default=(), repr=False, kw_only=True)
    gateway_record_id: str = ""
    reused: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            **super().as_dict(),
            "gateway_record_id": self.gateway_record_id or None,
            "reused": self.reused,
        }


@dataclass
class SupervisorState:
    episodes: int = 0
    episode_budget_exhausted: int = 0
    interrupted: int = 0
    tokens: int = 0
    # Cumulative usage per runner invocation, independent of episode outcomes.
    usage_receipts: dict[str, int] = field(default_factory=dict)
    # Latest native provider Session for this serial trajectory. Pooled search
    # gives every trajectory its own SupervisorState, so Sessions never cross
    # island boundaries.
    session_id: str = ""
    session_backend: str = ""
    attempts: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: object) -> "SupervisorState":
        if not isinstance(value, dict):
            return cls()
        fields = cls.__dataclass_fields__
        return cls(**{key: value[key] for key in fields if key in value})

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
