"""Acceptance of original evaluation facts, without workflow state changes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .evaluation import VerificationResult


@dataclass(frozen=True)
class AcceptanceRequirements:
    min_improvement_pct: float = 0.0
    mode: str = "abba"
    conversion_pending: bool = False


@dataclass(frozen=True)
class AcceptanceDecision:
    accepted: bool
    verification: VerificationResult


@dataclass(frozen=True)
class FrameworkCandidateFacts:
    """Observed candidate facts collected by the Workflow before pure acceptance."""
    candidate_blob: str
    baseline_changes: tuple[str, ...] = ()
    structure_violations: tuple[str, ...] = ()
    is_gluon: bool = False

    def __post_init__(self):
        object.__setattr__(self, "baseline_changes", tuple(self.baseline_changes))
        object.__setattr__(self, "structure_violations", tuple(self.structure_violations))


@dataclass(frozen=True)
class FrameworkCandidateRequest:
    workspace: Path
    framework: str
    v0_blob: str
    baseline_commit: str
    include_policy_review: bool = True
    source_checks_enabled: bool = True
    facts: FrameworkCandidateFacts | None = None


class AcceptancePolicy(Protocol):
    def decide(
        self, verification: VerificationResult, requirements: AcceptanceRequirements
    ) -> AcceptanceDecision: ...

    def check_framework_candidate(self, request: FrameworkCandidateRequest) -> str: ...
