"""Synchronous original-behavior experiment coordination.

The host owns injected dependencies and supplies an executor with its transport
binding already established. Evaluation owns measurement plans and interpretation.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from .execution import CommandSpec, JobResult

if TYPE_CHECKING:
    from .evaluation import EvaluationRequest, VerificationResult


class Experiment(Protocol):
    def run(self, request: EvaluationRequest) -> VerificationResult: ...

    def run_command(self, command: CommandSpec) -> JobResult:
        """Execute once without requiring an Evaluator or changing its result."""
        ...


class ExperimentExecutionError(RuntimeError):
    """Original execution/aggregation failure, distinct from preparation errors."""
