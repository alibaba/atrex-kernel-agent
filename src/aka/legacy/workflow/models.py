from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from orchestrator.agent_runtime.model import (
    AgentRuntimeCapabilities,
    NormalizedAgentEvent,
    TokenUsage,
)

from long_horizon.legacy_evaluation.models import VerificationRun, VerificationResult

TERMINAL_STATUSES = frozenset({"candidate_ready", "pivot", "blocked"})


@dataclass(frozen=True)
class EpisodeHandoff:
    status: str
    candidate_commit: str = ""
    last_trial_commit: str = ""

    def as_dict(self) -> dict[str, str]:
        return {key: value for key, value in asdict(self).items() if value}


@dataclass(frozen=True)
class InvocationObservation:
    terminal_usage: TokenUsage
    events: tuple[NormalizedAgentEvent, ...]
    capabilities: AgentRuntimeCapabilities
    observation_errors: tuple[str, ...] = ()
    resume_usage_qualified: bool = False


@dataclass
class SessionResult:
    exit_status: int
    timed_out: bool
    tokens: int
    session_id: str
    resume_count: int
    handoff: EpisodeHandoff | None
    stdout_tail: str = ""
    stderr_tail: str = ""
    completion_diagnosis: str = ""
    invocations: tuple[InvocationObservation, ...] = ()


@dataclass
class SupervisorState:
    episodes: int = 0
    accepted: int = 0
    rejected: int = 0
    pivoted: int = 0
    blocked: int = 0
    protocol_failures: int = 0
    interrupted: int = 0
    tokens: int = 0
    # Cumulative usage per runner invocation, independent of episode outcomes.
    usage_receipts: dict[str, int] = field(default_factory=dict)
    consecutive_without_promotion: int = 0
    attempts: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: object) -> "SupervisorState":
        if not isinstance(value, dict):
            return cls()
        fields = cls.__dataclass_fields__
        return cls(**{key: value[key] for key in fields if key in value})

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
