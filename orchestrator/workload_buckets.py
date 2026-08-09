"""Operator layout detection and workload bucket partitioning.

Reads the operator directory (SOL workload.jsonl or atrex-bench shapes.json), validates
inspector-proposed buckets, and materializes one bucket operator directory per bucket.
"""
from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .constants import (
    DISPATCH_VISIBILITY_POLICY,
    FRAMEWORK_BASELINE_CATEGORY,
    WORKLOAD_BUCKET_CONTRACT_FILE,
    WORKLOAD_BUCKETS_FILE,
)
from .hardware import _workspace_slug


def is_sol_op(op_dir: Path) -> bool:
    """A SOL-ExecBench op dir carries definition.json + workload.jsonl next to reference.py."""
    return (op_dir / "definition.json").is_file() and (op_dir / "workload.jsonl").is_file()


def find_atrex_bench_root(op_dir: Path) -> Optional[Path]:
    """Return the canonical Atrex-Bench checkout owning a native shapes op."""
    for candidate in (op_dir, *op_dir.parents):
        if (
            (candidate / "scripts" / "run_eval.py").is_file()
            and (candidate / "src" / "atrex_bench").is_dir()
        ):
            return candidate
    return None


def is_bucketable_op(op_dir: Path) -> bool:
    """Whether the op exposes an enumerable SOL workload or atrex shape set."""
    return is_sol_op(op_dir) or (op_dir / "shapes.json").is_file()


@dataclass(frozen=True)
class WorkloadBucket:
    """One inspector-produced, disjoint subset of workload.jsonl."""

    name: str
    workload_indices: tuple[int, ...]
    rationale: str = ""


@dataclass(frozen=True)
class WorkloadSource:
    """Ordered workload view over SOL JSONL or atrex-bench shapes.json."""

    kind: str
    filename: str
    ids: tuple[str, ...]
    entries: tuple[dict, ...]
    raw_lines: tuple[str, ...] = ()


def _read_workloads(path: Path) -> tuple[list[str], list[dict]]:
    raw_lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    workloads: list[dict] = []
    for index, line in enumerate(raw_lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid workload.jsonl line {index + 1}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"workload.jsonl line {index + 1} is not a JSON object")
        workloads.append(value)
    if not workloads:
        raise ValueError("workload.jsonl contains no workloads")
    return raw_lines, workloads


def _shape_id_sort_key(shape_id: str) -> tuple[int, object]:
    return (0, int(shape_id)) if re.fullmatch(r"\d+", shape_id) else (1, shape_id)


def _read_workload_source(op_dir: Path) -> WorkloadSource:
    workload_path = op_dir / "workload.jsonl"
    if workload_path.is_file():
        lines, workloads = _read_workloads(workload_path)
        return WorkloadSource(
            kind="sol",
            filename="workload.jsonl",
            ids=tuple(str(index) for index in range(len(workloads))),
            entries=tuple(workloads),
            raw_lines=tuple(lines),
        )

    shapes_path = op_dir / "shapes.json"
    if not shapes_path.is_file():
        raise ValueError(f"operator has neither workload.jsonl nor shapes.json: {op_dir}")
    try:
        shapes = json.loads(shapes_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid shapes.json: {exc}") from exc
    if not isinstance(shapes, dict) or not shapes:
        raise ValueError("shapes.json must contain a non-empty object keyed by shape id")
    ids = tuple(sorted((str(key) for key in shapes), key=_shape_id_sort_key))
    entries: list[dict] = []
    for shape_id in ids:
        value = shapes.get(shape_id)
        if not isinstance(value, dict):
            raise ValueError(f"shapes.json entry {shape_id!r} is not an object")
        entries.append(value)
    return WorkloadSource(
        kind="shapes",
        filename="shapes.json",
        ids=ids,
        entries=tuple(entries),
    )


def validate_workload_buckets(
    manifest: object,
    workload_count: int,
    max_buckets: int,
    *,
    require_visibility_policy: bool = False,
) -> list[WorkloadBucket]:
    """Validate the inspector contract and return normalized buckets.

    Exact, disjoint coverage is deliberately enforced here rather than trusted
    to the coding agent.  A missing workload would make a bucket kernel look
    faster without ever proving it correct; an overlap would bias optimization
    resources and make aggregation provenance ambiguous.
    """
    if not isinstance(manifest, dict) or not isinstance(manifest.get("buckets"), list):
        raise ValueError(f"{WORKLOAD_BUCKETS_FILE} must contain a top-level buckets list")
    if isinstance(manifest.get("schema_version"), bool) or manifest.get("schema_version") != 1:
        raise ValueError(f"{WORKLOAD_BUCKETS_FILE} schema_version must be 1")
    visibility_policy = manifest.get("dispatch_visibility_policy")
    if visibility_policy not in (None, DISPATCH_VISIBILITY_POLICY):
        raise ValueError(
            f"{WORKLOAD_BUCKETS_FILE} dispatch visibility policy is unsupported: "
            f"{visibility_policy!r}"
        )
    if require_visibility_policy and visibility_policy != DISPATCH_VISIBILITY_POLICY:
        raise ValueError(
            f"{WORKLOAD_BUCKETS_FILE} must declare dispatch_visibility_policy="
            f"{DISPATCH_VISIBILITY_POLICY!r}"
        )
    if (
        isinstance(manifest.get("workload_count"), bool)
        or manifest.get("workload_count") != workload_count
    ):
        raise ValueError(
            f"{WORKLOAD_BUCKETS_FILE} workload_count does not match workload.jsonl"
        )
    raw_buckets = manifest["buckets"]
    if not raw_buckets:
        raise ValueError("workload inspector returned no buckets")
    if len(raw_buckets) > max_buckets:
        raise ValueError(
            f"workload inspector returned {len(raw_buckets)} buckets; maximum is {max_buckets}"
        )

    buckets: list[WorkloadBucket] = []
    seen_names: set[str] = set()
    owners: dict[int, str] = {}
    for position, raw in enumerate(raw_buckets):
        if not isinstance(raw, dict):
            raise ValueError(f"bucket {position} is not an object")
        raw_name = raw.get("name")
        if not isinstance(raw_name, str):
            raise ValueError(f"bucket {position} has no string name")
        name = _workspace_slug(raw_name)
        if name in seen_names:
            raise ValueError(f"duplicate bucket name: {name}")
        seen_names.add(name)
        indices = raw.get("workload_indices")
        if not isinstance(indices, list) or not indices:
            raise ValueError(f"bucket {name} has no workload_indices")
        normalized: list[int] = []
        for index in indices:
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError(f"bucket {name} contains a non-integer workload index")
            if index < 0 or index >= workload_count:
                raise ValueError(f"bucket {name} workload index {index} is out of range")
            if index in owners:
                raise ValueError(
                    f"workload index {index} appears in both {owners[index]} and {name}"
                )
            owners[index] = name
            normalized.append(index)
        rationale = raw.get("rationale")
        buckets.append(
            WorkloadBucket(
                name=name,
                workload_indices=tuple(sorted(normalized)),
                rationale=rationale if isinstance(rationale, str) else "",
            )
        )

    missing = sorted(set(range(workload_count)) - set(owners))
    if missing:
        raise ValueError(f"workload inspector omitted workload indices: {missing}")
    return buckets


def _normalized_bucket_manifest(buckets: list[WorkloadBucket], workload_count: int) -> dict:
    return {
        "schema_version": 1,
        "dispatch_visibility_policy": DISPATCH_VISIBILITY_POLICY,
        "workload_count": workload_count,
        "buckets": [
            {
                "name": bucket.name,
                "workload_indices": list(bucket.workload_indices),
                "rationale": bucket.rationale,
            }
            for bucket in buckets
        ],
    }


def _materialize_bucket_op(
    source_op: Path,
    destination: Path,
    bucket: WorkloadBucket,
    workload_source: WorkloadSource,
    generalization_evidence: dict,
) -> None:
    """Create a derived op dir whose evaluator sees exactly one bucket."""
    destination.mkdir(parents=True, exist_ok=True)
    for source in source_op.iterdir():
        if source.name in (".git", "__pycache__", workload_source.filename):
            continue
        target = destination / source.name
        # sol_seed consumes top-level SOL ground truth.  Do not recursively
        # copy arbitrary op subdirectories: --workspace is allowed to live
        # under the op dir, and copying it here would recurse into the newly
        # created aggregate/bucket repositories.
        if source.is_file():
            shutil.copy2(source, target)
    (destination / WORKLOAD_BUCKET_CONTRACT_FILE).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bucket": bucket.name,
                "dispatch_strategy": "structural_range_tree_v1",
                "requirement": (
                    "The kernel must support the whole routed shape regime, including the "
                    "recorded unseen one-step neighbor. Exact benchmark-shape lookup tables "
                    "or equality-only dispatch are forbidden."
                ),
                "generalization_evidence": generalization_evidence,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    if workload_source.kind == "sol":
        selected = [workload_source.raw_lines[index] for index in bucket.workload_indices]
        (destination / "workload.jsonl").write_text(
            "\n".join(selected) + "\n", encoding="utf-8"
        )
        return

    selected_ids = [workload_source.ids[index] for index in bucket.workload_indices]
    selected_shapes = {
        shape_id: workload_source.entries[index]
        for index, shape_id in zip(bucket.workload_indices, selected_ids)
    }
    (destination / "shapes.json").write_text(
        json.dumps(selected_shapes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    roofline_path = source_op / "roofline.json"
    if roofline_path.is_file():
        try:
            roofline = json.loads(roofline_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid roofline.json: {exc}") from exc
        if isinstance(roofline, dict) and isinstance(roofline.get("shapes"), dict):
            all_roofline_shapes = roofline["shapes"]
            roofline["shapes"] = {
                shape_id: all_roofline_shapes[shape_id]
                for shape_id in selected_ids
                if shape_id in all_roofline_shapes
            }
            (destination / "roofline.json").write_text(
                json.dumps(roofline, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )


def _bucket_baseline_memory(
    aggregate_memory: dict,
    bucket: WorkloadBucket,
    workload_source: WorkloadSource,
    *,
    aggregate_workspace: Path,
    aggregate_baseline_commit: str,
    aggregate_baseline_version: int = 0,
) -> dict:
    """Derive one bucket's V0 record from the aggregate baseline measurement.

    The aggregate evaluator already measured every workload independently and
    persisted those measurements in ``latency_us_by_shape``.  Re-running the
    same baseline once per bucket wastes both coding-agent and GPU time.  A
    bucket baseline is therefore the exact subset of those measurements, with
    scalar latency aggregates recomputed over only that subset.

    The source is the aggregate framework baseline when the campaign pinned one, otherwise
    the original V0 record.
    """
    performance = aggregate_memory.get("performance") or {}
    per_workload = performance.get("latency_us_by_shape")
    if not isinstance(per_workload, dict):
        raise RuntimeError(
            f"aggregate memory/v{aggregate_baseline_version}.json has no "
            "performance.latency_us_by_shape; cannot derive bucket baselines without re-running it"
        )

    selected: dict[str, float] = {}
    selected_ids: list[str] = []
    ordered_keys = list(per_workload)
    for index in bucket.workload_indices:
        entry = workload_source.entries[index]
        candidates = [workload_source.ids[index]]
        for field_name in ("uuid", "id"):
            value = entry.get(field_name)
            if isinstance(value, (str, int)) and not isinstance(value, bool):
                candidates.append(str(value))
        key = next((candidate for candidate in candidates if candidate in per_workload), None)
        # Older SOL records can be keyed by evaluator UUID while the source view
        # exposes only ordinal ids.  Equal cardinality makes the evaluator's
        # stable insertion order an unambiguous final fallback.
        if key is None and len(ordered_keys) == len(workload_source.entries):
            key = ordered_keys[index]
        if key is None:
            raise RuntimeError(
                f"aggregate V0 has no per-workload latency for bucket {bucket.name!r} "
                f"workload index {index}"
            )
        latency = per_workload.get(key)
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(float(latency))
            or float(latency) <= 0
        ):
            raise RuntimeError(
                f"aggregate V0 latency for workload {key!r} is not a positive finite number"
            )
        selected[str(key)] = float(latency)
        selected_ids.append(str(key))

    latencies = list(selected.values())
    geomean = (
        latencies[0]
        if len(latencies) == 1
        else math.exp(sum(math.log(value) for value in latencies) / len(latencies))
    )
    arithmetic_mean = sum(latencies) / len(latencies)
    # JSON round-tripping gives a deep copy while also guaranteeing the derived
    # record remains JSON serializable like every other memory entry.
    derived = json.loads(json.dumps(aggregate_memory))
    derived["version"] = "v0"
    derived["masked"] = False
    derived["timestamp"] = datetime.now(timezone.utc).isoformat()
    # A bucket baseline is neither the campaign's framework baseline nor an aggregate candidate.
    derived.pop(FRAMEWORK_BASELINE_CATEGORY, None)
    derived.pop("aggregation", None)
    derived_performance = derived.setdefault("performance", {})
    derived_performance["latency_us"] = geomean
    derived_performance["latency_us_geomean"] = geomean
    derived_performance["latency_us_arith_mean"] = arithmetic_mean
    derived_performance["latency_us_by_shape"] = selected
    derived_performance["derived_from_aggregate_v0"] = True
    derived_performance["derived_from_aggregate_version"] = f"v{aggregate_baseline_version}"
    derived_performance["aggregate_v0_latency_us"] = performance.get("latency_us")
    derived_performance["aggregate_baseline_latency_us"] = performance.get("latency_us")
    # Throughput/utilization scalars in aggregate V0 were reduced over the full
    # workload set.  They are not mathematically valid for a subset and could
    # incorrectly trip the bucket's peak-utilization stop condition.
    for aggregate_only_metric in (
        "tflops",
        "bandwidth_gbps",
        "tflops_peak_utilization_pct",
        "bandwidth_peak_utilization_pct",
        "speedup_vs_ref_geomean",
    ):
        derived_performance.pop(aggregate_only_metric, None)
    source_label = (
        f"framework baseline v{aggregate_baseline_version}"
        if aggregate_baseline_version > 0
        else "V0"
    )
    derived_performance["measurement_note"] = (
        f"Mechanically selected aggregate {source_label} per-workload timings for bucket "
        f"{bucket.name}; no separate baseline evaluation was run."
    )
    optimization = derived.setdefault("optimization", {})
    optimization["action_category"] = "baseline"
    optimization["action_description"] = (
        f"V0 mechanically derived from aggregate {source_label} for bucket {bucket.name}; "
        f"selected workloads: {', '.join(selected_ids)}. No separate baseline session "
        "or GPU evaluation was run."
    )
    derived["baseline_derivation"] = {
        "source": (
            "aggregate_framework_baseline" if aggregate_baseline_version > 0 else "aggregate_v0"
        ),
        "aggregate_workspace": str(aggregate_workspace),
        "aggregate_v0_commit": aggregate_baseline_commit,
        "aggregate_baseline_commit": aggregate_baseline_commit,
        "aggregate_baseline_version": f"v{aggregate_baseline_version}",
        "bucket": bucket.name,
        "workload_indices": list(bucket.workload_indices),
        "workload_ids": selected_ids,
    }
    derived["git_commit_hash"] = None
    return derived
