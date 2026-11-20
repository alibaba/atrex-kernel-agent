"""Supervisor-owned evaluator harness selection.

The selected harness is execution input, never part of the Agent workspace or
Agent Git identity.  Runtime stages it only inside an ephemeral request tree.
"""

from __future__ import annotations

from pathlib import Path

from aka.evaluation.resources import native_harness_script
from aka.resources.bundles import resource_path as resource


NATIVE_EVALUATOR_HARNESS = native_harness_script().resolve(strict=True)
SOL_EVALUATOR_HARNESS = (Path(__file__).resolve().parents[1] / "task/problem/sol_harness.py").resolve(strict=True)


def evaluator_harness(workspace: Path) -> Path:
    """Return the trusted harness for one staged operator contract."""
    if (workspace / "definition.json").is_file() and (
        workspace / "workload.jsonl"
    ).is_file():
        return SOL_EVALUATOR_HARNESS
    return NATIVE_EVALUATOR_HARNESS
