from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


TERMINAL_STATUSES = frozenset({"candidate_ready", "pivot", "blocked"})


@dataclass(frozen=True)
class EpisodeHandoff:
    status: str
    candidate_commit: str = ""
    last_trial_commit: str = ""

    def as_dict(self) -> dict[str, str]:
        return {key: value for key, value in asdict(self).items() if value}


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

    @property
    def passed(self) -> bool:
        return self.gate == "PASS" and not self.error

    def as_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "candidate_latency_us": self.candidate_latency_us,
            "incumbent_latency_us": self.incumbent_latency_us,
            "improvement_pct": self.improvement_pct,
            "runs": [run.as_dict() for run in self.runs],
            "error": self.error or None,
            "artifact": self.artifact or None,
        }


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
