"""Controller version facts, independent of optional Agent memory summaries.

These Git-tracked facts retain atomic Kernel/version promotion and recovery when
memory generation is disabled. They are never projected into Agent drafts or
GPU bundles. Existing memory-only campaigns remain readable without migration.
"""
from pathlib import Path
import json
import math

from aka.legacy.task.persistence import durable_write_json

STATE_DIRECTORY = ".atrex_versions"
FACT_FIELDS = frozenset({
    "version", "timestamp", "performance", "correctness", "quality_gate",
    "git_commit_hash", "long_horizon", "framework_baseline",
})


def version_record_path(workspace: Path, version: int, *, prefer_summary: bool = False) -> Path:
    if isinstance(version, bool) or not isinstance(version, int) or version < 0:
        raise ValueError("Version must be a non-negative integer")
    private = workspace / STATE_DIRECTORY / f"v{version}.json"
    memory = workspace / "memory" / f"v{version}.json"
    if prefer_summary and memory.exists():
        return memory
    return private if private.exists() or not memory.exists() else memory


def save_version_record(workspace: Path, version: int, record: dict, *, generate_memory: bool = False) -> Path:
    if record.get("version") != f"v{version}":
        raise ValueError("Version record identity does not match")
    version_record_path(workspace, version)  # Validate before constructing paths.
    record = objective_score_report(workspace, version, record)
    path = workspace / STATE_DIRECTORY / f"v{version}.json"
    durable_write_json(path, {key: value for key, value in record.items() if key in FACT_FIELDS})
    if generate_memory:
        durable_write_json(workspace / "memory" / f"v{version}.json", record)
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
        path = version_record_path(workspace, 0)
        baseline = json.loads(path.read_text()) if path.is_file() else {}
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


def version_commit_paths(workspace: Path, version: int, *, generate_memory: bool) -> list[str]:
    paths = [str(version_record_path(workspace, version).relative_to(workspace))]
    summary = f"memory/v{version}.json"
    if generate_memory and summary not in paths and (workspace / summary).exists():
        paths.append(summary)
    return paths
