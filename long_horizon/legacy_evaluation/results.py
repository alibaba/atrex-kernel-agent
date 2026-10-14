"""Preserved interpretation of production evaluation records for the PR5 handoff."""

from __future__ import annotations
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any
from orchestrator.constants import CONVERT_PERF_TOL
from orchestrator.workspace_state import read_memory
from .models import VerificationResult

EPISODE_EVALUATIONS_PATH = Path(".atrex_long_horizon/evaluations.jsonl")


def _iso_timestamp(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _conversion_parity_passes(verification: VerificationResult) -> bool:
    candidate = verification.candidate_latency_us
    incumbent = verification.incumbent_latency_us
    if not isinstance(candidate, (int, float)) or not isinstance(
        incumbent, (int, float)
    ):
        return False
    if candidate > incumbent * (1.0 + CONVERT_PERF_TOL):
        return False
    return bool(verification.runs) and all(
        run.exit_code == 0
        and isinstance(run.result, dict)
        and bool(run.result.get("all_pass"))
        for run in verification.runs
    )


def _representative_candidate_result(
    verification: VerificationResult | None,
) -> dict[str, Any]:
    """Return the latest real candidate measurement from authoritative verification."""
    if verification is None:
        return {}
    for run in reversed(verification.runs):
        if run.revision == "candidate" and isinstance(run.result, dict):
            return run.result
    return {}


def _candidate_shape_latencies(
    verification: VerificationResult | None,
) -> tuple[dict[str, float], int]:
    """Aggregate real per-shape candidate latency across authoritative ABBA repeats."""
    values: dict[str, list[float]] = {}
    measured_runs = 0
    if verification is None:
        return {}, measured_runs
    for run in verification.runs:
        if run.revision != "candidate" or not isinstance(run.result, dict):
            continue
        by_shape = run.result.get("latency_us_by_shape")
        if not isinstance(by_shape, dict):
            continue
        measured_runs += 1
        for shape_id, raw_value in by_shape.items():
            if (
                isinstance(raw_value, (int, float))
                and not isinstance(raw_value, bool)
                and raw_value > 0
                and math.isfinite(float(raw_value))
            ):
                values.setdefault(str(shape_id), []).append(float(raw_value))
    return (
        {
            shape_id: (
                samples[0]
                if len(samples) == 1
                else math.exp(sum(math.log(value) for value in samples) / len(samples))
            )
            for shape_id, samples in values.items()
            if samples
        },
        measured_runs,
    )


def _positive_finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number > 0.0 and math.isfinite(number) else None


def _latest_complete_canonical_performance(
    workspace: Path,
    *,
    before_version: int,
    expected_shape_ids: set[str] | None,
    required_performance_objective: str | None = None,
) -> tuple[dict[str, Any], int] | None:
    """Find the newest complete real-shape incumbent measurement before a round.

    A pivot, blocked session, protocol failure, or supervisor interruption may have no
    candidate verification at all.  Its memory must still carry the real per-shape
    performance of the unchanged canonical incumbent rather than replacing known facts
    with nulls.  Only a complete, finite canonical record is eligible for carry-forward.
    """
    for version in range(before_version - 1, -1, -1):
        memory = read_memory(workspace, version)
        if not isinstance(memory, dict):
            continue
        if (memory.get("quality_gate") or {}).get("result") != "PASS":
            continue
        performance = memory.get("performance")
        if not isinstance(performance, dict):
            continue
        if (
            required_performance_objective is not None
            and performance.get("performance_objective")
            != required_performance_objective
        ):
            continue
        by_shape = performance.get("latency_us_by_shape")
        if not isinstance(by_shape, dict) or not by_shape:
            continue
        normalized: dict[str, float] = {}
        for shape_id, raw_value in by_shape.items():
            value = _positive_finite(raw_value)
            if value is None:
                normalized = {}
                break
            normalized[str(shape_id)] = value
        if not normalized:
            continue
        if expected_shape_ids is not None and set(normalized) != expected_shape_ids:
            continue
        latency = _positive_finite(
            performance.get("latency_us_geomean", performance.get("latency_us"))
        )
        performance_score = _positive_finite(
            performance.get(
                "performance_score", performance.get("speedup_vs_ref_geomean")
            )
        )
        if latency is None or performance_score is None:
            continue
        carried = dict(performance)
        carried["latency_us"] = latency
        carried["latency_us_geomean"] = latency
        carried["latency_us_arith_mean"] = _positive_finite(
            performance.get("latency_us_arith_mean")
        ) or (sum(normalized.values()) / len(normalized))
        carried["latency_us_by_shape"] = normalized
        carried["performance_score"] = performance_score
        return carried, version
    return None


def _latest_complete_episode_performance(
    episode_workspace: Path | None,
    *,
    expected_shape_ids: set[str] | None,
    required_performance_objective: str | None = None,
) -> dict[str, Any] | None:
    """Read the latest complete supervisor-independent measurement from an episode."""
    if episode_workspace is None:
        return None
    path = episode_workspace / EPISODE_EVALUATIONS_PATH
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        kernel_path = episode_workspace / "kernel.py"
        kernel_bytes = kernel_path.read_bytes()
        kernel_sha256 = hashlib.sha256(kernel_bytes).hexdigest()
        kernel_mtime = kernel_path.stat().st_mtime
        manifest = episode_workspace / "solution.json"
        solution_sha256 = (
            hashlib.sha256(manifest.read_bytes()).hexdigest()
            if manifest.exists()
            else None
        )
    except (OSError, UnicodeError):
        return None
    for line in reversed(lines):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        result = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(result, dict) or not result.get("all_pass"):
            continue
        if (
            required_performance_objective is not None
            and result.get("performance_objective") != required_performance_objective
        ):
            continue
        schema_version = payload.get("schema_version")
        if isinstance(schema_version, int) and schema_version >= 3:
            if (
                "solution_sha256" not in payload
                or payload["solution_sha256"] != solution_sha256
            ):
                continue
        elif solution_sha256 is not None:
            # Legacy receipts do not certify a candidate with a manifest.
            continue
        recorded_kernel_sha256 = payload.get("kernel_sha256")
        if isinstance(schema_version, int) and schema_version >= 2:
            if recorded_kernel_sha256 != kernel_sha256:
                continue
        else:
            # Schema v1 predates code fingerprints.  It is safe only when the
            # current kernel has not been modified since this measurement.
            try:
                measured_at = _iso_timestamp(str(payload["timestamp"]))
            except (KeyError, TypeError, ValueError):
                continue
            if measured_at < kernel_mtime:
                continue
        by_shape = result.get("latency_us_by_shape")
        if not isinstance(by_shape, dict) or not by_shape:
            continue
        normalized: dict[str, float] = {}
        for shape_id, raw_value in by_shape.items():
            value = _positive_finite(raw_value)
            if value is None:
                normalized = {}
                break
            normalized[str(shape_id)] = value
        if not normalized:
            continue
        if expected_shape_ids is not None and set(normalized) != expected_shape_ids:
            continue
        latency = _positive_finite(
            result.get("latency_us_geomean", result.get("latency_us"))
        )
        performance_score = _positive_finite(
            result.get("performance_score", result.get("speedup_vs_ref_geomean"))
        )
        if latency is None or performance_score is None:
            continue
        return {
            "all_pass": True,
            "latency_us_geomean": latency,
            "latency_us_arith_mean": _positive_finite(
                result.get("latency_us_arith_mean")
            )
            or (sum(normalized.values()) / len(normalized)),
            "latency_us_by_shape": normalized,
            "speedup_vs_ref_mean": result.get("speedup_vs_ref_mean"),
            "speedup_vs_ref_geomean": result.get("speedup_vs_ref_geomean"),
            "performance_score": performance_score,
            "performance_objective": result.get("performance_objective"),
            "max_abs_err": result.get("max_abs_err"),
            "max_rel_err": result.get("max_rel_err"),
            "eval_id": result.get("eval_id"),
            "timestamp": payload.get("timestamp"),
        }
    return None


def _episode_evaluation_count(episode_workspace: Path) -> int:
    """Count durable evaluator results emitted by one episode."""
    path = episode_workspace / EPISODE_EVALUATIONS_PATH
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return 0
    count = 0
    for line in lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("result"), dict):
            count += 1
    return count
