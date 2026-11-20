"""V0 compatibility readers backed by the Campaign Measurement Store.

V0 is an ordinary full Evaluate, stored beside every other immutable Gateway
Record under ``measurements/``.  It is private input/reference evidence rather
than an accepted submission, so it does not enter the Pareto frontier or winner
selection.  Pre-MeasurementStore baseline files remain readable and are
migrated on first access; new Campaigns never create a ``baseline/`` directory.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Mapping

from aka.workflow.light.support.paths import (
    supervisor_campaign_base_root,
    supervisor_measurement_root,
)


def baseline_root(workspace: Path) -> Path:
    """Return the removed legacy V0 directory for migration only."""
    return supervisor_campaign_base_root(workspace) / "baseline"


def baseline_record_path(workspace: Path) -> Path:
    return baseline_root(workspace) / "v0.json"


def _legacy_paths(workspace: Path) -> tuple[Path, ...]:
    return (
        baseline_record_path(workspace),
        workspace / ".atrex_versions" / "v0.json",
        workspace / "memory" / "v0.json",
    )


def _read_object(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _measurement_store(workspace: Path, measurement_root: Path | None = None):
    from aka.experiment.measurement_store import MeasurementStore

    return MeasurementStore(measurement_root or supervisor_measurement_root(workspace))


def baseline_measurement_spec(
    workspace: Path,
    commit: str,
    *,
    measurement_root: Path | None = None,
):
    """Return the canonical V0 MeasurementStore request and Kernel identity."""
    from aka.task.source.formats import Artifact, from_git

    store = _measurement_store(workspace, measurement_root)
    source = from_git(workspace, commit)
    artifact = Artifact(source)
    identity = store.kernel(source, commit)
    options: dict[str, object] = {
        "command": ["runtime-private-evaluator", "--version", "v0"],
        "evaluation_mode": "full",
        "evaluation_policy": {"mode": "full"},
    }
    if artifact.layout.archived:
        options["source_snapshots"] = {"candidate": artifact.digest}
        options["source_contracts"] = {"candidate": artifact.contract}
        inputs = {
            name: hashlib.sha256(contents).hexdigest()
            for name, contents in artifact.files.items()
        }
    else:
        inputs = {artifact.layout.python_entrypoint: artifact.digest}
    request = {
        "operation": "evaluate",
        "options": options,
        "inputs": inputs,
        "kernel_ids": {"candidate": commit},
        "visibility": "supervisor_private_v0",
    }
    return store, request, {"candidate": identity}


def _job_ids(record_directory: Path, stderr: str = "") -> list[str]:
    values = re.findall(r"\bev_[A-Za-z0-9_.-]+\b", stderr)
    for path in record_directory.glob("**/jobs/*/state.json"):
        value = _read_object(path)
        process = value.get("process") if isinstance(value, dict) else None
        stdout = process.get("stdout") if isinstance(process, dict) else None
        if not isinstance(stdout, str):
            continue
        try:
            payload = json.loads(stdout)
        except (TypeError, ValueError, json.JSONDecodeError):
            values.extend(re.findall(r"\bev_[A-Za-z0-9_.-]+\b", stdout))
            continue
        if isinstance(payload, dict):
            job_id = payload.get("job_id") or payload.get("id")
            if isinstance(job_id, str) and job_id:
                values.append(job_id)
    return list(dict.fromkeys(values))


def _baseline_summary(record: Mapping[str, object], record_directory: Path) -> dict | None:
    response = record.get("response")
    response = response if isinstance(response, Mapping) else {}
    receipt = response.get("agent_result")
    receipt = receipt if isinstance(receipt, Mapping) else {}
    result = receipt.get("result")
    result = result if isinstance(result, Mapping) else {}
    if receipt.get("status") != "completed" or result.get("correct") is not True:
        return None
    kernels = record.get("kernels")
    kernels = kernels if isinstance(kernels, Mapping) else {}
    candidate = kernels.get("candidate")
    candidate = candidate if isinstance(candidate, Mapping) else {}
    commit = candidate.get("kernel_id")
    if not isinstance(commit, str) or not commit:
        return None
    latencies = result.get("latency_us_by_shape")
    latencies = dict(latencies) if isinstance(latencies, Mapping) else {}
    correctness = result.get("correctness")
    correctness = dict(correctness) if isinstance(correctness, Mapping) else {}
    performance = {
        "latency_us": result.get("latency_us_geomean"),
        "latency_us_geomean": result.get("latency_us_geomean"),
        "latency_us_arith_mean": result.get("latency_us_arith_mean"),
        "latency_us_by_shape": latencies,
        "measurement_scope": "real_evaluator_shapes",
        "measurement_status": "complete" if latencies else "incomplete",
        "measured_shape_count": len(latencies),
    }
    return {
        "version": "v0",
        "timestamp": record.get("created_at"),
        "git_commit_hash": commit,
        "gateway_record_id": record.get("gateway_record_id"),
        "backend_job_ids": _job_ids(
            record_directory,
            str(response.get("stderr") or ""),
        ),
        "performance": performance,
        "correctness": {
            "status": "PASS",
            "max_abs_err": correctness.get(
                "max_abs_err", result.get("max_abs_err")
            ),
            "max_rel_err": correctness.get(
                "max_rel_err", result.get("max_rel_err")
            ),
        },
        "quality_gate": {"result": "PASS", "failure_reason": None},
        "optimization": {
            "action_category": "v0_reference",
            "action_description": (
                "selected initial implementation measured by the official "
                "full-workload evaluator"
            ),
            "expected_impact": (
                "correctness and latency reference for later candidates"
            ),
            "risks_and_rollback": "none; immutable V0 reference measurement",
        },
    }


def read_baseline_gateway_record(
    workspace: Path, *, measurement_root: Path | None = None
) -> dict | None:
    """Return the latest passing V0 Gateway Record from MeasurementStore."""
    store = _measurement_store(workspace, measurement_root)
    candidates = []
    for record_id in store.record_ids():
        try:
            record = store.read(record_id)
        except (FileNotFoundError, OSError, ValueError, KeyError, TypeError):
            continue
        request = record.get("request")
        if (
            isinstance(request, Mapping)
            and request.get("visibility") == "supervisor_private_v0"
            and _baseline_summary(
                record, store.root / "records" / record_id
            ) is not None
        ):
            candidates.append(record)
    if not candidates:
        return None
    return max(candidates, key=lambda item: str(item.get("created_at") or ""))


def read_baseline_record(
    workspace: Path,
    *,
    migrate_legacy: bool = True,
    measurement_root: Path | None = None,
) -> dict | None:
    """Read V0 facts from MeasurementStore, with legacy files as fallback."""
    measurement = read_baseline_gateway_record(
        workspace, measurement_root=measurement_root
    )
    if measurement is not None:
        store = _measurement_store(workspace, measurement_root)
        return _baseline_summary(
            measurement,
            store.root / "records" / str(measurement["gateway_record_id"]),
        )
    for legacy in _legacy_paths(workspace):
        record = _read_object(legacy)
        if record is None:
            continue
        if migrate_legacy and isinstance(record.get("git_commit_hash"), str):
            try:
                ensure_baseline_gateway_record(
                    workspace, measurement_root=measurement_root
                )
            except (OSError, ValueError, RuntimeError):
                pass
            measurement = read_baseline_gateway_record(
                workspace, measurement_root=measurement_root
            )
            if measurement is not None:
                store = _measurement_store(workspace, measurement_root)
                return _baseline_summary(
                    measurement,
                    store.root
                    / "records"
                    / str(measurement["gateway_record_id"]),
                )
        return record
    return None


def _baseline_result(record: Mapping[str, object]) -> dict[str, object]:
    """Reconstruct the bounded evaluator receipt sealed into the private V0 Record."""
    performance = record.get("performance")
    performance = performance if isinstance(performance, Mapping) else {}
    correctness = record.get("correctness")
    correctness = correctness if isinstance(correctness, Mapping) else {}
    passed = str(correctness.get("status", "")).upper() == "PASS"
    result: dict[str, object] = {
        "all_pass": passed,
        "correct": passed,
        "mode": "full",
        "latency_us_geomean": performance.get("latency_us_geomean"),
        "latency_us_arith_mean": performance.get("latency_us_arith_mean"),
        "latency_us_by_shape": performance.get("latency_us_by_shape"),
        "max_abs_err": correctness.get("max_abs_err"),
        "max_rel_err": correctness.get("max_rel_err"),
    }
    return {key: value for key, value in result.items() if value is not None}


def ensure_baseline_gateway_record(
    workspace: Path, *, measurement_root: Path | None = None
) -> str | None:
    """Return a real private ``gateway-`` identity for the measured V0.

    Current Campaigns record V0 directly and return above.  The conversion below
    exists only for old Campaigns whose completed V0 receipt predates the
    MeasurementStore.  The migrated Record remains Supervisor-private.
    """
    measurement = read_baseline_gateway_record(
        workspace, measurement_root=measurement_root
    )
    if measurement is not None:
        value = measurement.get("gateway_record_id")
        return value if isinstance(value, str) else None
    record = read_baseline_record(workspace, migrate_legacy=False)
    if not isinstance(record, dict):
        return None
    commit = record.get("git_commit_hash")
    if not isinstance(commit, str) or not commit:
        return None

    from aka.experiment.measurement_store import DuplicateTask

    store, request, kernels = baseline_measurement_spec(
        workspace, commit, measurement_root=measurement_root
    )
    result = _baseline_result(record)
    response = {
        "exit_code": 0 if result.get("all_pass") is True else 1,
        "stdout": json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n",
        "stderr": (
            "[supervisor] backend_job_ids="
            + ",".join(
                item
                for item in record.get("backend_job_ids", [])
                if isinstance(item, str)
            )
            + "\n"
            if isinstance(record.get("backend_job_ids"), list)
            and record.get("backend_job_ids")
            else ""
        ),
    }
    try:
        with store.reserve(request, kernels) as task:
            gateway_record_id = task.finish(response, cacheable=True)[
                "gateway_record_id"
            ]
    except DuplicateTask as duplicate:
        gateway_record_id = duplicate.record_id
    if not isinstance(gateway_record_id, str):
        raise RuntimeError("V0 Gateway Record did not produce a stable identity")
    return gateway_record_id
