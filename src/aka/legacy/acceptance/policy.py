"""Preserve the original score threshold and conversion parity decisions."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Sequence
from aka.contracts.acceptance import (
    AcceptanceDecision,
    AcceptanceRequirements,
    FrameworkCandidateRequest,
)
from aka.contracts.evaluation import VerificationResult
from aka.contracts.workspace import CandidateWorkspace


class LegacyAcceptancePolicy:
    def __init__(
        self,
        *,
        conversion_tolerance: float = 0.05,
        candidate_workspace: CandidateWorkspace | None = None,
        production_policy: Callable[[], Sequence[str]] | None = None,
    ):
        self.conversion_tolerance = conversion_tolerance
        self.candidate_workspace = candidate_workspace
        self.production_policy = production_policy

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
        """Static acceptance checks on the candidate about to be validated and committed.

        Everything is judged from the worktree: that is what the gateway uploads, and it lets a
        session that wrote the kernel but never committed it still be accepted.
        """
        from orchestrator.hardware import kernel_is_gluon
        from orchestrator.optimization_policy import production_structure_violations
        from aka.legacy.task.git import git_worktree_blob

        candidate_blob = git_worktree_blob(request.workspace, "kernel.py")
        if not candidate_blob or candidate_blob == request.v0_blob:
            return "the session left the V0 kernel unchanged; no framework implementation was produced"
        violations = (
            self.production_policy()
            if request.include_policy_review
            else production_structure_violations(request.workspace, request.framework)
        )
        if violations:
            return (
                f"the candidate is not a self-contained {request.framework} implementation: "
                + "; ".join(violations)
            )
        if request.framework.lower() in {"triton", "gluon"} and kernel_is_gluon(
            request.workspace
        ):
            # A Gluon v1 would permanently disarm the orchestrator's mandatory Triton->Gluon latch.
            return "the framework baseline must be plain Triton; Gluon is a later orchestrator escalation"
        mutated = self.candidate_workspace.baseline_changes(
            request.workspace, request.baseline_commit
        )
        if mutated:
            return "the session modified immutable ground truth: " + ", ".join(mutated)
        return ""
