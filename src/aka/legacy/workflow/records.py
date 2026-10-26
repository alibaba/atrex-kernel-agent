"""Workflow-owned canonical record codec, storage and measurement projection.

Old records remain readable; absent or corrupt records do not fabricate facts.
Evaluation receives detached measurements instead of private record dictionaries.
"""
from __future__ import annotations
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone
from aka.contracts.content import WorkspaceFile
from aka.contracts.workflow_records import CanonicalMeasurement
from aka.legacy.task.io import atomic_write_json
from aka.legacy.task.candidate_workspace.state import resolve_framework_baseline_commit


def latest_version(workspace: Path) -> int:
    """Largest STRICT integer version present as memory/v<N>.json.

    Note: int('71_512') == 71512 in Python (PEP 515 underscores-in-numeric-literals).
    Experimental files named like v71_512.json (an experiment branch for the v71 round) must
    NOT count as v71512 — they're scratch variants of v71, not real iterations. We therefore
    accept only stems matching ^v\\d+\\.json$ exactly.
    """
    _STRICT = re.compile(r"^v(\d+)\.json$")
    mem = workspace / "memory"
    if not mem.exists():
        return -1
    vs = []
    for p in mem.glob("v*.json"):
        m = _STRICT.match(p.name)
        if m is not None:
            vs.append(int(m.group(1)))
    return max(vs) if vs else -1


def decode_record(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("canonical Workflow record must be a JSON object")
    return value


def write_record(path: Path, record: dict) -> None:
    if not isinstance(record, dict):
        raise ValueError("canonical Workflow record must be a JSON object")
    atomic_write_json(path, record)


def read_memory(workspace: Path, n: int) -> Optional[dict]:
    try:
        return decode_record(workspace / "memory" / f"v{n}.json")
    except (ValueError, OSError):
        return None


def speedup_vs_reference(
    workspace: Path,
    candidate_latency_us: object,
    reported_speedup: object = None,
) -> Optional[float]:
    """Return a real reported speedup or derive it from canonical V0 latency."""

    def positive(value: object) -> Optional[float]:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        number = float(value)
        return number if number > 0.0 and math.isfinite(number) else None

    reported = positive(reported_speedup)
    if reported is not None:
        return reported
    candidate = positive(candidate_latency_us)
    baseline = read_memory(workspace, 0)
    performance = (
        baseline.get("performance")
        if isinstance(baseline, dict)
        and isinstance(baseline.get("performance"), dict)
        else {}
    )
    reference = positive(
        performance.get("latency_us_geomean", performance.get("latency_us"))
    )
    if candidate is None or reference is None:
        return None
    return reference / candidate


def peak_util(mem: Optional[dict]) -> float:
    """Max of tflops / bandwidth peak utilization (%), 0 if unknown."""
    if not mem:
        return 0.0
    perf = mem.get("performance") or {}
    vals = [perf.get("tflops_peak_utilization_pct"), perf.get("bandwidth_peak_utilization_pct")]
    return max([float(v) for v in vals if isinstance(v, (int, float))] or [0.0])


def best_validated_latency_us(workspace: Path, *, from_version: Optional[int] = None) -> Optional[float]:
    """Best correctness-passing measured latency in a workspace.

    Versions before a pinned framework baseline are excluded by default: production cannot ship
    the PyTorch V0 wrapper, so treating its (typically much faster, library-backed) latency as the
    incumbent would reject every candidate a from-scratch DSL kernel can produce.
    """
    if from_version is None:
        _pinned_commit, pinned_version = resolve_framework_baseline_commit(workspace)
        from_version = pinned_version
    best: Optional[float] = None
    for version in range(from_version, latest_version(workspace) + 1):
        memory = read_memory(workspace, version)
        if not memory:
            continue
        gate = (memory.get("quality_gate") or {}).get("result")
        correctness = (memory.get("correctness") or {}).get("status")
        if gate != "PASS" and correctness != "PASS":
            continue
        latency = (memory.get("performance") or {}).get("latency_us")
        if isinstance(latency, (int, float)):
            best = float(latency) if best is None else min(best, float(latency))
    return best


def write_memory(workspace: Path, version: int | str, record: dict) -> Path:
    name = f"v{version}" if isinstance(version, int) else version
    if re.fullmatch(r"v[0-9]+", name) is None:
        raise ValueError("invalid canonical record version")
    path = workspace / "memory" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_record(path, record)
    return path


def promotion_files(*, episode: int, memory_version: int, evidence: dict,
                    memory_record: dict) -> tuple[WorkspaceFile, ...]:
    if type(episode) is not int or episode < 0 or type(memory_version) is not int or memory_version < 0:
        raise ValueError("invalid Workflow record identity")
    return tuple(WorkspaceFile(path=name, text=json.dumps(value, indent=2, ensure_ascii=False,
                             allow_nan=False) + "\n")
                 for name, value in ((f"memory/long_horizon_e{episode:04d}.json", evidence),
                                      (f"memory/v{memory_version}.json", memory_record)))


def _measurement(workspace: Path, version: int) -> CanonicalMeasurement | None:
    record = read_memory(workspace, version)
    if not isinstance(record, dict):
        return None
    performance = record.get("performance")
    if not isinstance(performance, dict):
        return None
    result = dict(performance)
    result.setdefault("latency_us_geomean", performance.get("latency_us"))
    result.setdefault("performance_score", performance.get("speedup_vs_ref_geomean"))
    gate = record.get("quality_gate")
    correctness = record.get("correctness")
    gate = gate.get("result") if isinstance(gate, dict) else None
    correctness = correctness.get("status") if isinstance(correctness, dict) else None
    result["all_pass"] = correctness == "PASS"
    if isinstance(record.get("correctness"), dict):
        for key in ("max_abs_err", "max_rel_err"):
            result[key] = record["correctness"].get(key)
    try:
        return CanonicalMeasurement(version, gate, correctness, result)
    except ValueError:
        return None


class WorkflowMeasurementReader:
    def baseline(self, workspace):
        return _measurement(workspace, 0)

    def measurements(self, workspace, *, before_version):
        return tuple(fact for version in range(min(before_version - 1, latest_version(workspace)), -1, -1)
                     if (fact := _measurement(workspace, version)) is not None)


def _record_local_test_result(workspace: Path, version: str, result: dict) -> Path:
    """Merge a remote --no-memory test result into local optimizer memory."""
    mem_dir = workspace / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    path = mem_dir / f"{version}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        data = {}
    data.setdefault("version", version)
    data.setdefault("masked", False)
    data["timestamp"] = datetime.now(timezone.utc).isoformat()
    perf = data.setdefault("performance", {})
    perf["measurement_conditions"] = result.get("measurement_conditions", {})
    perf["latency_us"] = result.get("latency_us_geomean")
    perf["latency_us_geomean"] = result.get("latency_us_geomean")
    perf["latency_us_arith_mean"] = result.get("latency_us_arith_mean")
    by_shape = result.get("latency_us_by_shape", {})
    by_shape = by_shape if isinstance(by_shape, dict) else {}
    perf["latency_us_by_shape"] = by_shape
    perf["measurement_scope"] = "real_evaluator_shapes"
    perf["shape_ids_are_opaque"] = (workspace / "agent_problem.json").is_file()
    perf["measurement_status"] = (
        "complete" if result.get("all_pass") and by_shape else "incomplete"
    )
    perf["measured_shape_count"] = len(by_shape)
    performance_objective = result.get("performance_objective")
    perf["performance_objective"] = performance_objective
    perf["performance_score"] = result.get(
        "performance_score", result.get("speedup_vs_ref_geomean")
    )
    perf["speedup_vs_ref_mean"] = result.get("speedup_vs_ref_mean")
    perf["speedup_vs_ref_geomean"] = (
        None
        if performance_objective == "shape_speedup_arithmetic_mean"
        else speedup_vs_reference(
            workspace,
            result.get("latency_us_geomean"),
            result.get("speedup_vs_ref_geomean"),
        )
    )
    all_pass = bool(result.get("all_pass"))
    corr = data.setdefault("correctness", {})
    corr["status"] = "PASS" if all_pass else "FAIL"
    corr["max_abs_err"] = result.get("max_abs_err")
    corr["max_rel_err"] = result.get("max_rel_err")
    gate = data.setdefault("quality_gate", {})
    gate["result"] = "PASS" if all_pass else "FAIL"
    failures = result.get("failures") or []
    gate["failure_reason"] = None if all_pass else "; ".join(map(str, failures))[:2000]
    return write_memory(workspace, version, data)


def record_initial_measurement(workspace: Path, source_commit: str) -> str:
    """Keep V0 measurement metadata in its original second, source-stable commit."""
    import subprocess
    record = read_memory(workspace, 0)
    if record is None:
        return source_commit
    record["git_commit_hash"] = source_commit
    record.setdefault("optimization", {})["action_category"] = "baseline"
    write_memory(workspace, 0, record)
    subprocess.run(["git", "add", "memory/v0.json"], cwd=workspace, check=True)
    subprocess.run(["git", "commit", "-m", "V0: record baseline measurement"],
                   cwd=workspace, check=True, stdout=subprocess.DEVNULL)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=workspace, text=True).strip()


def reject_production_commit(
    workspace: Path,
    version: int,
    pre_head: str,
    violations: list[str],
) -> Path:
    """Revert a violating kernel commit and preserve an actionable local record."""
    memory = read_memory(workspace, version) or {}
    if pre_head:
        subprocess.run(
            ["git", "reset", "--hard", pre_head],
            cwd=str(workspace),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    memory["version"] = f"v{version}"
    memory["masked"] = False
    memory["git_commit_hash"] = None
    memory["quality_gate"] = {
        "result": "FAIL",
        "failure_reason": "production policy violation: " + "; ".join(violations),
    }
    memory["optimization"] = {
        "action_category": "production_policy_rejection",
        "action_description": "reverted candidate that used a forbidden dependency or wrong framework",
    }
    pitfalls = memory.setdefault("pitfalls_and_fixes", [])
    pitfalls.append({
        "error_type": "production_policy",
        "error_message": "; ".join(violations),
        "lesson": "implement the candidate directly and exclusively in the selected framework",
    })
    return write_memory(workspace, version, memory)
