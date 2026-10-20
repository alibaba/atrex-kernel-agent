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
class FrameworkCandidateRequest:
    workspace: Path
    framework: str
    v0_blob: str
    baseline_commit: str
    include_policy_review: bool = True
    source_checks_enabled: bool = True


class AcceptancePolicy(Protocol):
    def decide(
        self, verification: VerificationResult, requirements: AcceptanceRequirements
    ) -> AcceptanceDecision: ...

    def check_framework_candidate(self, request: FrameworkCandidateRequest) -> str: ...
