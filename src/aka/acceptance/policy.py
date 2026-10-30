"""Preserve the original score threshold and conversion parity decisions."""

from __future__ import annotations

from dataclasses import replace
import math
from aka.contracts.acceptance import (
    AcceptanceDecision,
    AcceptanceRequirements,
    FrameworkCandidateRequest,
)
from aka.contracts.evaluation import VerificationResult


CONVERT_PERF_TOL = 0.05


class PerformanceScorePolicy:
    def __init__(
        self,
        *,
        conversion_tolerance: float = CONVERT_PERF_TOL,
    ):
        self.conversion_tolerance = conversion_tolerance

    def conversion_parity_passes(self, verification: VerificationResult) -> bool:
        candidate = verification.candidate_latency_us
        incumbent = verification.incumbent_latency_us
        if not isinstance(candidate, (int, float)) or not isinstance(
            incumbent, (int, float)
        ):
            return False
        if candidate > incumbent * (1.0 + self.conversion_tolerance):
            return False
        return bool(verification.runs) and all(
            run.exit_code == 0
            and isinstance(run.result, dict)
            and bool(run.result.get("all_pass"))
            for run in verification.runs
        )

    def decide(
        self, verification: VerificationResult, requirements: AcceptanceRequirements
    ) -> AcceptanceDecision:
        result = verification
        if (
            requirements.mode in {"abba", "fast"}
            and result.passed
            and (
                result.improvement_pct is None
                or result.improvement_pct <= requirements.min_improvement_pct
            )
        ):
            if requirements.mode == "fast":
                error = (
                    "fast evaluator performance-score improvement "
                    f"{result.improvement_pct:.6f}% did not exceed "
                    f"{requirements.min_improvement_pct:.3f}%"
                )
            else:
                improvement = result.improvement_pct
                error = (
                    f"candidate improvement {improvement if improvement is not None else 'unknown'} "
                    f"in performance score did not exceed {requirements.min_improvement_pct:.3f}%"
                )
            result = replace(result, gate="FAIL", error=error)
        if (
            requirements.conversion_pending
            and not result.passed
            and self.conversion_parity_passes(result)
        ):
            result = replace(result, gate="PASS", error="")
        return AcceptanceDecision(result.passed, result)

    def check_framework_candidate(self, request: FrameworkCandidateRequest) -> str:
        """Compute acceptance from already observed facts, without I/O or callbacks."""
        facts = request.facts
        if facts is None:
            raise ValueError("framework acceptance requires observed candidate facts")
        candidate_blob = facts.candidate_blob
        if not candidate_blob or candidate_blob == request.v0_blob:
            return "the session left the V0 kernel unchanged; no framework implementation was produced"
        violations = facts.structure_violations if request.source_checks_enabled else ()
        if violations:
            return (
                f"the candidate is not a self-contained {request.framework} implementation: "
                + "; ".join(violations)
            )
        if (
            request.source_checks_enabled
            and request.framework.lower() in {"triton", "gluon"}
            and facts.is_gluon
        ):
            # A Gluon v1 would permanently disarm the orchestrator's mandatory Triton->Gluon latch.
            return "the framework baseline must be plain Triton; Gluon is a later orchestrator escalation"
        mutated = facts.baseline_changes
        if mutated:
            return "the session modified immutable ground truth: " + ", ".join(mutated)
        return ""


class GeoMeanLatencyPolicy(PerformanceScorePolicy):
    """Promote from the evaluator's public GeoMean latency facts."""

    @staticmethod
    def _usable_latencies(verification: VerificationResult) -> bool:
        try:
            return all(
                type(value) in (int, float) and math.isfinite(value) and value > 0
                for value in (
                    verification.candidate_latency_us, verification.incumbent_latency_us
                )
            )
        except OverflowError:
            return False

    def conversion_parity_passes(self, verification: VerificationResult) -> bool:
        return self._usable_latencies(verification) and super().conversion_parity_passes(verification)

    def decide(
        self, verification: VerificationResult, requirements: AcceptanceRequirements
    ) -> AcceptanceDecision:
        if requirements.mode not in {"abba", "fast"} or not verification.passed:
            return super().decide(verification, requirements)
        candidate = verification.candidate_latency_us
        incumbent = verification.incumbent_latency_us
        try:
            usable = self._usable_latencies(verification)
            improvement = (
                (incumbent - candidate) / incumbent * 100.0 if usable else None
            )
        except (OverflowError, ZeroDivisionError):
            improvement = None
        if improvement is None or not math.isfinite(improvement):
            return AcceptanceDecision(
                False,
                replace(
                    verification,
                    gate="ERROR",
                    improvement_pct=None,
                    error="GeoMean latency comparison requires complete positive finite latency facts",
                ),
            )
        result = replace(verification, improvement_pct=improvement)
        if improvement <= requirements.min_improvement_pct:
            result = replace(
                result,
                gate="FAIL",
                error=(
                    f"GeoMean latency reduction {improvement:.6f}% did not exceed "
                    f"{requirements.min_improvement_pct:.3f}%"
                ),
            )
        return super().decide(result, requirements)
