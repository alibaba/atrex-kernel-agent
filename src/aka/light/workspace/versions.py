"""Controller v1+ facts, independent of Agent-visible summary files.

These Git-tracked facts retain atomic Kernel/version promotion and recovery when
V0 evaluator facts are Supervisor-private; they must not create a metadata-only
source commit.
"""
from pathlib import Path
import math

from aka.legacy.task.persistence import durable_write_json
from aka.light.workspace.baseline import read_baseline_record

STATE_DIRECTORY = ".atrex_versions"
FACT_FIELDS = frozenset({
    "version", "timestamp", "performance", "correctness", "quality_gate",
    "git_commit_hash", "long_horizon",
})


def version_record_path(workspace: Path, version: int) -> Path:
    if isinstance(version, bool) or not isinstance(version, int) or version < 0:
        raise ValueError("Version must be a non-negative integer")
    return workspace / STATE_DIRECTORY / f"v{version}.json"


def save_version_record(workspace: Path, version: int, record: dict) -> Path:
    if version == 0:
        raise ValueError("V0 facts belong to the Campaign Measurement Store")
    if record.get("version") != f"v{version}":
        raise ValueError("Version record identity does not match")
    version_record_path(workspace, version)  # Validate before constructing paths.
    record = objective_score_report(workspace, version, record)
    path = workspace / STATE_DIRECTORY / f"v{version}.json"
    durable_write_json(path, {key: value for key, value in record.items() if key in FACT_FIELDS})
    return path


def objective_score_report(workspace: Path, version: int, record: dict) -> dict:
    """Keep objective scores distinct from unweighted latency-derived speedups.

    Ratios are only comparable when both complete versions name the same
    objective. This is reporting, not a promotion gate or a new measurement.
    """
    performance = record.get("performance", {})
    objective = performance.get("performance_objective")
    if not objective:
        return record
    performance = dict(performance)
    performance["latency_speedup_vs_v0_geomean"] = None
    performance["speedup_vs_ref_geomean"] = None
    performance["score_ratio_vs_v0"] = None
    if version == 0:
        baseline = record
    else:
        baseline = read_baseline_record(workspace) or {}
    base = baseline.get("performance", {})
    values = (performance.get("performance_score"), base.get("performance_score"))
    complete = all(row.get("performance", {}).get("measurement_status") == "complete"
                   and row.get("correctness", {}).get("status") == "PASS" for row in (record, baseline))
    if (complete and base.get("performance_objective") == objective
            and all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in values)):
        performance["score_ratio_vs_v0"] = values[0] / values[1]
        latencies = (base.get("latency_us_geomean"), performance.get("latency_us_geomean"))
        if all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in latencies):
            performance["latency_speedup_vs_v0_geomean"] = latencies[0] / latencies[1]
    return dict(record, performance=performance)


def version_commit_paths(workspace: Path, version: int) -> list[str]:
    if version == 0:
        raise ValueError("V0 facts are not committed to the candidate repository")
    return [str(version_record_path(workspace, version).relative_to(workspace))]
