#!/usr/bin/env python3
"""Shared validation for hash-bound PPU profiling evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable


ACU_SCHEMA = "ppu-acu-extraction/v4"
TIMELINE_SCHEMA = "ppu-fixed-slot-receipt/v5"
COMPARISON_SCHEMA = "ppu-acu-comparison/v1"
CALIBRATION_SCHEMA = "ppu-calibration-measurement/v1"
ENVELOPE_SCHEMA = "ppu-envelope-measurement/v1"
ACU_CORRECTNESS_SCHEMA = "ppu-profile-correctness/v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
ACU_METRIC_GROUPS = {
    "ce__total_cta_num.sum": "launch",
    "cu__cycles_active.avg": "compute",
    "cu__inst_executed.avg.per_cycle_active": "compute",
    "cu__inst_executed.avg.pct_of_peak_sustained_elapsed": "compute",
    "cu__inst_executed_pipe_tensor_fp8.avg.pct_of_peak_sustained_active": "tensor",
    "dram__bytes_read.sum.pct_of_peak_sustained_elapsed": "memory",
    "dram__bytes_write.sum.pct_of_peak_sustained_elapsed": "memory",
    "ksd__requests_hit_rate.pct": "ksd",
    "ksd__requests_load_pipe_ws.sum": "ksd",
    "ksd__requests_store_pipe_ws.sum": "ksd",
    "kvd__requests_hit_rate.pct": "kvd",
    "kvd__requests_load_pipe_lsu.sum": "kvd",
    "kvd__requests_store_pipe_lsu.sum": "kvd",
    "l2__requests_hit_rate.pct": "l2",
}
ACU_ACTIVITY_METRICS = {
    "ksd__requests_hit_rate.pct": (
        "ksd__requests_load_pipe_ws.sum",
        "ksd__requests_store_pipe_ws.sum",
    ),
    "kvd__requests_hit_rate.pct": (
        "kvd__requests_load_pipe_lsu.sum",
        "kvd__requests_store_pipe_lsu.sum",
    ),
}


class EvidenceError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def acu_metric_unit(name: str) -> str:
    if name.endswith(".pct") or ".pct_of_peak_" in name:
        return "percent"
    if "per_cycle" in name:
        return "per_cycle"
    if "cycles" in name:
        return "cycles"
    if "requests_" in name and name.endswith(".sum"):
        return "requests"
    return "unitless"


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def digest_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise EvidenceError(f"cannot hash artifact {path}: {error}") from error
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"cannot read {label} {path}: {error}") from error
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def attempt_file(owner_path: Path, value: str, label: str) -> Path:
    """Resolve capture inputs inside the directory containing its manifest/collection."""
    root = owner_path.resolve().parent
    candidate = Path(value)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    require(resolved.is_relative_to(root), f"{label} is outside the attempt directory")
    require(resolved.is_file(), f"{label} is not a regular file: {resolved}")
    return resolved


def _descriptor_path(owner_path: Path, descriptor: dict[str, Any], label: str) -> Path:
    logical_path = descriptor.get("path")
    require(isinstance(logical_path, str) and logical_path.strip(), f"{label}.path is required")
    path = Path(logical_path)
    if not path.is_absolute():
        path = owner_path.parent / path
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise EvidenceError(f"cannot resolve {label}: {error}") from error
    require(path.is_file(), f"{label} is not a regular file: {path}")
    require(not path.is_symlink(), f"{label} must not be a symbolic link: {path}")
    return resolved


def verify_descriptor(
    owner_path: Path,
    descriptor: object,
    label: str,
    *,
    expected_path: Path | None = None,
) -> dict[str, Any]:
    require(isinstance(descriptor, dict), f"{label} descriptor is missing")
    path = _descriptor_path(owner_path, descriptor, label)
    expected_hash = descriptor.get("sha256")
    require(
        isinstance(expected_hash, str) and _SHA256_RE.fullmatch(expected_hash) is not None,
        f"{label}.sha256 must be a lowercase SHA-256",
    )
    require(sha256_file(path) == expected_hash, f"{label} content hash mismatch")
    expected_size = descriptor.get("size_bytes")
    require(
        isinstance(expected_size, int)
        and not isinstance(expected_size, bool)
        and expected_size >= 0,
        f"{label}.size_bytes must be a non-negative integer",
    )
    require(path.stat().st_size == expected_size, f"{label} size mismatch")
    if expected_path is not None:
        require(
            path == expected_path.resolve(),
            f"{label} does not identify the supplied artifact",
        )
    return descriptor


def verify_descriptor_tree(owner_path: Path, value: object, label: str) -> None:
    if isinstance(value, dict):
        descriptor_fields = {"path", "sha256", "size_bytes"}
        present_fields = descriptor_fields.intersection(value)
        if present_fields:
            require(
                present_fields == descriptor_fields,
                f"{label} has an incomplete artifact descriptor",
            )
            verify_descriptor(owner_path, value, label)
            return
        nested = [item for item in value.values() if isinstance(item, (dict, list))]
        require(bool(nested), f"{label} contains no artifact descriptor")
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                verify_descriptor_tree(owner_path, item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            verify_descriptor_tree(owner_path, item, f"{label}[{index}]")
    else:
        raise EvidenceError(f"{label} must contain artifact descriptors")


def paths_alias(left: Path, right: Path) -> bool:
    if left.resolve() == right.resolve():
        return True
    try:
        return os.path.samefile(left, right)
    except OSError:
        return False


def ensure_no_output_aliases(
    outputs: Iterable[Path], inputs: Iterable[Path], label: str
) -> None:
    output_paths = list(outputs)
    input_paths = list(inputs)
    for index, output in enumerate(output_paths):
        require(
            all(not paths_alias(output, other) for other in output_paths[index + 1 :]),
            f"{label} outputs must be distinct",
        )
        require(
            all(not paths_alias(output, source) for source in input_paths),
            f"{label} output must not overwrite an input artifact",
        )


def descriptor_paths(owner_path: Path, value: object) -> set[Path]:
    paths: set[Path] = set()
    if isinstance(value, dict):
        descriptor_fields = {"path", "sha256", "size_bytes"}
        if descriptor_fields.issubset(value):
            paths.add(_descriptor_path(owner_path, value, "artifact"))
        else:
            for item in value.values():
                paths.update(descriptor_paths(owner_path, item))
    elif isinstance(value, list):
        for item in value:
            paths.update(descriptor_paths(owner_path, item))
    return paths


def portable_descriptors(value: object, owner_path: Path) -> object:
    if isinstance(value, dict):
        result = {
            key: portable_descriptors(item, owner_path)
            for key, item in value.items()
        }
        if {"path", "sha256", "size_bytes"}.issubset(value):
            result["path"] = os.path.relpath(
                Path(str(value["path"])).resolve(), owner_path.parent.resolve()
            )
        return result
    if isinstance(value, list):
        return [portable_descriptors(item, owner_path) for item in value]
    return value


def binding_view(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: binding_view(item)
            for key, item in sorted(value.items())
            if key not in {"path", "declared_path"}
        }
    if isinstance(value, list):
        return [binding_view(item) for item in value]
    return value


def require_sha256(value: object, label: str) -> str:
    require(
        isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
        f"{label} must be a lowercase SHA-256",
    )
    return value


def acu_binding_payload(metadata: dict[str, Any]) -> dict[str, Any]:
    outputs = metadata.get("outputs")
    require(isinstance(outputs, dict), "ACU outputs are missing")
    return {
        "producer": metadata.get("producer"),
        "identity": metadata.get("identity"),
        "kernel_sha256": metadata.get("kernel_sha256"),
        "inputs": binding_view(metadata.get("inputs")),
        "bound_artifacts": binding_view(metadata.get("bound_artifacts")),
        "pm_csv": binding_view(outputs.get("pm_csv")),
        "launch": metadata.get("launch"),
        "metric_summaries": metadata.get("metric_summaries"),
    }


def _finite_number(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise EvidenceError(f"{label} must be numeric") from error
    require(math.isfinite(result), f"{label} must be finite")
    return result


def _validate_acu_derived_data(
    metadata: dict[str, Any],
    report_path: Path,
    raw_path: Path,
    collection_path: Path,
    pm_path: Path,
) -> None:
    raw_path = attempt_file(collection_path, str(raw_path.resolve()), "ACU raw CSV")
    report_path = attempt_file(collection_path, str(report_path.resolve()), "ACU report")
    try:
        with raw_path.open(newline="", encoding="utf-8") as source:
            raw_rows = list(csv.DictReader(source))
        with pm_path.open(newline="", encoding="utf-8") as source:
            pm_rows = list(csv.DictReader(source))
    except (OSError, UnicodeError, csv.Error) as error:
        raise EvidenceError(f"cannot read ACU CSV evidence: {error}") from error
    require(len(raw_rows) == 1, "ACU raw CSV must contain exactly one kernel row")
    require(bool(pm_rows), "ACU PM CSV must contain at least one sample")
    collection = load_json(collection_path, "ACU collection descriptor")
    require(
        collection.get("schema") == "ppu-acu-collection/v2",
        "ACU collection descriptor must use ppu-acu-collection/v2",
    )
    try:
        from .acu_report import _load_collection, extract_pm_packets
    except ImportError:
        from acu_report import _load_collection, extract_pm_packets

    collected = _load_collection(collection_path, report_path)
    expected_identity = {
        field: collected[field]
        for field in (
            "kernel_name",
            "kernel_specialization",
            "workload_identity",
            "device_identity",
            "runtime_identity",
            "cache_policy",
            "clock_configuration",
        )
    }
    require(metadata.get("identity") == expected_identity, "ACU identity differs from collection")
    require(
        metadata.get("producer") == collected["producer"],
        "ACU producer differs from collection",
    )
    require(
        metadata.get("evidence_grade") == collected["evidence_grade"],
        "ACU evidence grade differs from collection",
    )
    require(
        metadata.get("kernel_sha256") == collected["kernel_sha256"],
        "ACU kernel hash differs from collection",
    )
    require(
        binding_view(metadata.get("bound_artifacts"))
        == binding_view(collected["bound_artifacts"]),
        "ACU bound artifacts differ from collection",
    )

    packets = extract_pm_packets(report_path.read_bytes())
    expected_samples: dict[tuple[int, str, int], tuple[int, int, float]] = {}
    for packet_index, (_, metrics) in enumerate(packets):
        for metric_name, samples in metrics:
            for sample_index, sample in enumerate(samples):
                expected_samples[(packet_index, metric_name, sample_index)] = sample
    require(bool(expected_samples), "ACU report contains no PM payload")
    require(len(expected_samples) == len(pm_rows), "ACU PM CSV row count differs from report")
    raw = raw_rows[0]
    launch = metadata.get("launch")
    require(isinstance(launch, dict), "ACU launch summary is missing")

    def parse_dims(value: object, label: str) -> list[int]:
        require(isinstance(value, str), f"{label} must be text")
        try:
            result = [int(part.strip()) for part in value.strip().strip("()").split(",")]
        except ValueError as error:
            raise EvidenceError(f"{label} is invalid") from error
        require(
            len(result) == 3 and all(dimension > 0 for dimension in result),
            f"{label} must contain three positive integers",
        )
        return result

    launch_fields = {
        "duration_ns": ("ppu__time_duration.sum", float),
        "occupancy_blocks_per_cu": ("launch__occupancy_blocks_per_cu", float),
        "registers_per_thread": ("launch__registers_per_thread", int),
        "shared_mem_per_block": ("launch__shared_mem_per_block", int),
        "cu_count": ("device__attribute_cu_count", int),
    }
    for field, (column, cast) in launch_fields.items():
        actual_number = _finite_number(raw.get(column), f"ACU raw {column}")
        expected_number = _finite_number(launch.get(field), f"ACU launch {field}")
        if cast is int:
            require(actual_number.is_integer(), f"ACU raw {column} must be integral")
            require(expected_number.is_integer(), f"ACU launch {field} must be integral")
            actual = int(actual_number)
            expected = int(expected_number)
        else:
            actual = actual_number
            expected = expected_number
        require(actual == expected, f"ACU launch {field} does not match raw CSV")
    for field, column in {
        "pm_interval_ns": "pmsampler__interval_time.max",
        "dropped_samples": "pmsampler__dropped_samples.max",
        "buffer_size_bytes": "pmsampler__buffer_size_bytes.max",
    }.items():
        raw_value = raw.get(column)
        actual = (
            None
            if raw_value in (None, "")
            else _finite_number(raw_value, f"ACU raw {column}")
        )
        expected = launch.get(field)
        require(
            (actual is None and expected is None)
            or (actual is not None and expected is not None and actual == float(expected)),
            f"ACU launch {field} does not match raw CSV",
        )
    require(
        launch.get("grid") == parse_dims(raw.get("Grid Size"), "ACU raw Grid Size"),
        "ACU launch grid does not match raw CSV",
    )
    require(
        launch.get("block")
        == parse_dims(raw.get("Block Size"), "ACU raw Block Size"),
        "ACU launch block does not match raw CSV",
    )
    device = int(_finite_number(raw.get("Device"), "ACU raw Device"))
    require(launch.get("device") == device, "ACU launch device does not match raw CSV")
    identity = metadata.get("identity")
    require(isinstance(identity, dict), "ACU identity is missing")
    require(
        raw.get("Kernel Name") == identity.get("kernel_name"),
        "ACU kernel name does not match raw CSV",
    )
    device_identity = identity.get("device_identity")
    require(isinstance(device_identity, dict), "ACU device identity is invalid")
    require(
        device_identity.get("physical_device") == device,
        "ACU identity device does not match raw CSV",
    )

    sample_values: dict[tuple[int, float, float, str], float] = {}
    for row in pm_rows:
        packet_number = _finite_number(row.get("packet_index"), "PM packet_index")
        sample_number = _finite_number(row.get("sample_index"), "PM sample_index")
        require(packet_number.is_integer(), "PM packet_index must be integral")
        require(sample_number.is_integer(), "PM sample_index must be integral")
        packet_index = int(packet_number)
        sample_index = int(sample_number)
        start_ns = _finite_number(row.get("window_start_ns"), "PM window_start_ns")
        end_ns = _finite_number(row.get("window_end_ns"), "PM window_end_ns")
        interval_ns = _finite_number(row.get("interval_ns"), "PM interval_ns")
        require(start_ns >= 0 and end_ns > start_ns, "PM window bounds are invalid")
        require(interval_ns == end_ns - start_ns, "PM interval does not match its window")
        metric_name = row.get("metric_name")
        require(isinstance(metric_name, str) and metric_name, "PM metric name is missing")
        known_metric = metric_name in ACU_METRIC_GROUPS
        require(
            row.get("logical_metric_group")
            == ACU_METRIC_GROUPS.get(metric_name, "unclassified"),
            f"PM metric group mismatch for {metric_name}",
        )
        require(
            row.get("metric_unit")
            == (acu_metric_unit(metric_name) if known_metric else "unknown"),
            f"PM metric unit mismatch for {metric_name}",
        )
        require(
            row.get("scope")
            == ("device_global_aggregate" if known_metric else "unknown"),
            f"PM metric scope mismatch for {metric_name}",
        )
        value = _finite_number(row.get("metric_value"), "PM metric value")
        row["_packet_index"] = str(packet_index)
        row["_sample_index"] = str(sample_index)
        row["_window_start_ns"] = str(start_ns)
        row["_window_end_ns"] = str(end_ns)
        identity = (packet_index, start_ns, end_ns, metric_name)
        require(identity not in sample_values, "duplicate ACU PM sample identity")
        expected_sample = expected_samples.get(
            (packet_index, metric_name, sample_index)
        )
        require(expected_sample is not None, "ACU PM CSV sample is absent from report")
        expected_start, expected_end, expected_value = expected_sample
        require(
            start_ns == expected_start
            and end_ns == expected_end
            and value == expected_value,
            "ACU PM CSV sample does not match report payload",
        )
        sample_values[identity] = value

    for row in pm_rows:
        packet_index = int(row["_packet_index"])
        start_ns = float(row["_window_start_ns"])
        end_ns = float(row["_window_end_ns"])
        metric_name = row["metric_name"]
        if metric_name.endswith("requests_hit_rate.pct"):
            activity_names = ACU_ACTIVITY_METRICS.get(metric_name)
            if activity_names is None:
                expected_validity = "unknown_no_activity_counter"
            else:
                activity = [
                    sample_values.get((packet_index, start_ns, end_ns, name))
                    for name in activity_names
                ]
                available = [value for value in activity if value is not None]
                if any(value > 0 for value in available):
                    expected_validity = "valid_activity_positive"
                elif len(available) == len(activity):
                    expected_validity = "no_activity_hit_rate_not_interpretable"
                else:
                    expected_validity = "unknown_no_activity_counter"
        elif metric_name in ACU_METRIC_GROUPS:
            expected_validity = "valid"
        else:
            expected_validity = "unknown_semantics"
        require(
            row.get("validity") == expected_validity,
            f"PM validity mismatch for {metric_name}",
        )

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in pm_rows:
        key = f"packet_{row.get('packet_index')}:{row.get('metric_name')}"
        grouped.setdefault(key, []).append(row)
    summaries = metadata.get("metric_summaries")
    require(isinstance(summaries, dict), "ACU metric summaries are missing")
    require(set(summaries) == set(grouped), "ACU metric summary streams do not match PM CSV")
    for key, rows in grouped.items():
        rows = sorted(rows, key=lambda row: int(row["_sample_index"]))
        summary = summaries.get(key)
        require(isinstance(summary, dict), f"ACU metric summary {key} is invalid")
        metric_name = rows[0].get("metric_name")
        require(
            isinstance(metric_name, str) and metric_name,
            f"{key} metric name is missing",
        )
        expected_group = ACU_METRIC_GROUPS.get(metric_name, "unclassified")
        expected_unit = (
            acu_metric_unit(metric_name)
            if metric_name in ACU_METRIC_GROUPS
            else "unknown"
        )
        require(
            summary.get("metric_name") == metric_name,
            f"{key} metric name mismatch",
        )
        require(
            summary.get("logical_metric_group") == expected_group,
            f"{key} metric group mismatch",
        )
        require(summary.get("unit") == expected_unit, f"{key} metric unit mismatch")
        require(
            summary.get("scope")
            == ("device_global_aggregate" if metric_name in ACU_METRIC_GROUPS else "unknown"),
            f"{key} metric scope mismatch",
        )
        require(
            summary.get("packet_index") == int(rows[0]["packet_index"]),
            f"{key} packet index mismatch",
        )
        require(summary.get("sample_count") == len(rows), f"{key} sample count mismatch")
        valid_rows = [
            row
            for row in rows
            if row.get("validity") in {"valid", "valid_activity_positive"}
        ]
        require(
            summary.get("valid_sample_count") == len(valid_rows),
            f"{key} valid sample count mismatch",
        )
        require(
            summary.get("excluded_sample_count") == len(rows) - len(valid_rows),
            f"{key} excluded sample count mismatch",
        )
        intervals = [
            _finite_number(row.get("interval_ns"), f"{key}.interval_ns")
            for row in valid_rows
        ]
        coverage_end = max(
            (
                _finite_number(row.get("window_end_ns"), f"{key}.window_end_ns")
                for row in valid_rows
            ),
            default=None,
        )
        require(
            coverage_end is None or coverage_end <= float(launch["duration_ns"]),
            f"{key} extends beyond the kernel duration",
        )
        require(summary.get("coverage_end_ns") == coverage_end, f"{key} coverage end mismatch")
        require(
            summary.get("coverage_ratio")
            == (
                min(sum(intervals) / float(launch["duration_ns"]), 1.0)
                if intervals
                else None
            ),
            f"{key} coverage ratio mismatch",
        )
        require(
            summary.get("interval_min_ns") == min(intervals, default=None),
            f"{key} minimum interval mismatch",
        )
        require(
            summary.get("interval_max_ns") == max(intervals, default=None),
            f"{key} maximum interval mismatch",
        )
        validity_counts: dict[str, int] = {}
        for row in rows:
            state = str(row.get("validity"))
            validity_counts[state] = validity_counts.get(state, 0) + 1
        require(
            summary.get("validity_counts") == validity_counts,
            f"{key} validity counts mismatch",
        )
        expected_mean = None
        values: list[float] = []
        if valid_rows:
            require(
                float(valid_rows[0]["_window_start_ns"]) == 0,
                f"{key} valid coverage does not start at zero",
            )
            require(
                all(
                    float(left["_window_end_ns"]) == float(right["_window_start_ns"])
                    for left, right in zip(valid_rows, valid_rows[1:])
                ),
                f"{key} has gaps or overlaps in interpretable coverage",
            )
            total_interval = sum(
                _finite_number(row.get("interval_ns"), f"{key}.interval_ns")
                for row in valid_rows
            )
            require(total_interval > 0, f"{key} has no positive valid interval")
            values = [
                _finite_number(row.get("metric_value"), f"{key}.metric_value")
                for row in valid_rows
            ]
            expected_mean = sum(
                value
                * _finite_number(row.get("interval_ns"), f"{key}.interval_ns")
                for value, row in zip(values, valid_rows, strict=True)
            ) / total_interval
        observed_mean = summary.get("time_weighted_mean")
        if expected_mean is None:
            require(observed_mean is None, f"{key} mean must be null without valid samples")
            require(summary.get("min") is None, f"{key} minimum must be null")
            require(summary.get("max") is None, f"{key} maximum must be null")
        else:
            require(
                isinstance(observed_mean, (int, float))
                and not isinstance(observed_mean, bool)
                and math.isclose(
                    float(observed_mean),
                    expected_mean,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ),
                f"{key} mean does not match PM CSV",
            )
            require(summary.get("min") == min(values), f"{key} minimum mismatch")
            require(summary.get("max") == max(values), f"{key} maximum mismatch")

    dropped_samples = launch.get("dropped_samples")
    require(
        dropped_samples in (None, 0, 0.0),
        "accepted ACU evidence contains dropped PM samples",
    )
    observed_intervals = sorted(
        {
            _finite_number(row.get("interval_ns"), "PM interval_ns")
            for row in pm_rows
            if row["validity"] in {"valid", "valid_activity_positive"}
        }
    )
    require(
        bool(observed_intervals),
        "accepted ACU evidence has no interpretable PM intervals",
    )
    if launch.get("pm_interval_ns") is not None:
        require(
            float(launch["pm_interval_ns"]) == max(observed_intervals),
            "ACU reported PM interval does not match sample windows",
        )
    if len(observed_intervals) > 1:
        median = observed_intervals[len(observed_intervals) // 2]
        require(
            median > 0
            and (max(observed_intervals) - min(observed_intervals)) / median
            <= 0.01,
            "accepted ACU evidence has excessive PM interval jitter",
        )
    require(
        any(
            isinstance(summary, dict)
            and int(summary.get("valid_sample_count", 0)) > 0
            for summary in summaries.values()
        ),
        "accepted ACU evidence has no interpretable metric",
    )
    for key, summary in summaries.items():
        if int(summary.get("valid_sample_count", 0)) > 0:
            require(
                int(summary["valid_sample_count"]) >= 10,
                f"accepted ACU evidence has too few samples for {key}",
            )
            require(
                float(launch["duration_ns"]) - float(summary["coverage_end_ns"])
                <= float(summary["interval_max_ns"]),
                f"accepted ACU evidence has incomplete tail coverage for {key}",
            )
    requested = collection.get("requested_metrics")
    if requested is not None:
        require(
            isinstance(requested, list)
            and all(isinstance(name, str) and name for name in requested),
            "collection.requested_metrics is invalid",
        )
        observed = {row["metric_name"] for row in pm_rows}
        require(
            set(requested).issubset(observed),
            "accepted ACU evidence is missing requested metrics",
        )
        for name in requested:
            require(
                any(
                    row["metric_name"] == name
                    and row["validity"] in {"valid", "valid_activity_positive"}
                    for row in pm_rows
                ),
                f"accepted ACU requested metric {name} is not interpretable",
            )
    validation = metadata.get("validation")
    require(isinstance(validation, dict), "ACU validation summary is missing")
    require(not validation.get("errors"), "accepted ACU evidence reports errors")
    require(not validation.get("warnings"), "accepted ACU evidence reports warnings")


def validate_acu_metadata(
    path: Path,
    *,
    expected_pm: Path | None = None,
    expected_raw: Path | None = None,
    require_decision: bool = True,
) -> dict[str, Any]:
    metadata = load_json(path, "ACU extraction metadata")
    require(
        metadata.get("schema") == ACU_SCHEMA,
        f"ACU metadata must use {ACU_SCHEMA}",
    )
    validation = metadata.get("validation")
    require(isinstance(validation, dict), "ACU validation is missing")
    require(
        validation.get("status") == "accepted",
        "ACU extraction metadata is not accepted",
    )
    grade = metadata.get("evidence_grade")
    require(grade in {"diagnostic", "decision"}, "ACU evidence grade is invalid")
    if require_decision:
        require(grade == "decision", "decision-grade ACU evidence is required")
    require(
        metadata.get("producer") == {"name": "acu", "version": "2.2"},
        "unsupported ACU producer",
    )

    inputs = metadata.get("inputs")
    outputs = metadata.get("outputs")
    bound = metadata.get("bound_artifacts")
    require(isinstance(inputs, dict), "ACU inputs are missing")
    require(isinstance(outputs, dict), "ACU outputs are missing")
    require(isinstance(bound, dict), "ACU bound artifacts are missing")
    for name in ("report", "raw_csv", "collection"):
        verify_descriptor(path, inputs.get(name), f"inputs.{name}")
    verify_descriptor(
        path, outputs.get("pm_csv"), "outputs.pm_csv", expected_path=expected_pm
    )
    if expected_raw is not None:
        verify_descriptor(
            path,
            inputs.get("raw_csv"),
            "inputs.raw_csv",
            expected_path=expected_raw,
        )
    verify_descriptor_tree(path, bound, "bound_artifacts")
    for field in ("source_artifacts", "binary_artifacts", "workload_inputs"):
        rows = bound.get(field)
        require(isinstance(rows, list), f"bound_artifacts.{field} must be a list")
        if field == "source_artifacts" or grade == "decision":
            require(bool(rows), f"{grade} evidence requires bound_artifacts.{field}")
    verify_descriptor(path, bound.get("producer"), "bound_artifacts.producer")

    kernel = bound.get("authoritative_kernel")
    verify_descriptor(path, kernel, "bound_artifacts.authoritative_kernel")
    kernel_sha256 = require_sha256(metadata.get("kernel_sha256"), "kernel_sha256")
    require(kernel.get("sha256") == kernel_sha256, "authoritative kernel hash mismatch")
    if grade == "decision":
        correctness = bound.get("correctness")
        verify_descriptor(path, correctness, "bound_artifacts.correctness")
        correctness_path = _descriptor_path(
            path, correctness, "bound_artifacts.correctness"
        )
        correctness_record = load_json(correctness_path, "correctness evidence")
        require(
            correctness_record.get("schema") == ACU_CORRECTNESS_SCHEMA
            and correctness_record.get("validation") == "accepted",
            f"correctness evidence must be accepted {ACU_CORRECTNESS_SCHEMA}",
        )
        require(
            correctness_record.get("kernel_sha256") == kernel_sha256,
            "correctness evidence does not bind the authoritative kernel",
        )
        identity = metadata.get("identity")
        require(isinstance(identity, dict), "ACU identity is missing")
        require(
            correctness_record.get("workload_identity")
            == identity.get("workload_identity"),
            "correctness evidence workload identity mismatch",
        )
        require(
            correctness_record.get("device_identity")
            == identity.get("device_identity"),
            "correctness evidence device identity mismatch",
        )
        checks = correctness_record.get("checks")
        require(
            isinstance(checks, list)
            and bool(checks)
            and all(
                isinstance(check, dict)
                and isinstance(check.get("name"), str)
                and bool(check["name"].strip())
                and check.get("status") == "passed"
                for check in checks
            ),
            "correctness evidence requires named passed checks",
        )

    report_path = _descriptor_path(path, inputs["report"], "inputs.report")
    raw_path = _descriptor_path(path, inputs["raw_csv"], "inputs.raw_csv")
    collection_path = _descriptor_path(
        path, inputs["collection"], "inputs.collection"
    )
    pm_path = _descriptor_path(path, outputs["pm_csv"], "outputs.pm_csv")
    _validate_acu_derived_data(
        metadata, report_path, raw_path, collection_path, pm_path
    )
    expected_payload = acu_binding_payload(metadata)
    require(
        metadata.get("binding_payload") == expected_payload,
        "ACU binding payload does not match the receipt",
    )
    require(
        digest_json(expected_payload) == metadata.get("evidence_id"),
        "ACU extraction evidence id is invalid",
    )
    return metadata


def validate_timeline_receipt(
    path: Path,
    *,
    expected_outputs: dict[str, Path] | None = None,
    require_decision: bool = False,
    recompute_outputs: bool = True,
) -> dict[str, Any]:
    receipt = load_json(path, "timeline receipt")
    require(
        receipt.get("schema") == TIMELINE_SCHEMA,
        f"timeline receipt must use {TIMELINE_SCHEMA}",
    )
    require(receipt.get("validation") == "accepted", "timeline receipt is not accepted")
    grade = receipt.get("evidence_grade")
    require(grade in {"diagnostic", "decision"}, "timeline evidence grade is invalid")
    if require_decision:
        require(grade == "decision", "decision-grade timeline evidence is required")
    payload = receipt.get("binding_payload")
    require(isinstance(payload, dict), "timeline binding payload is missing")
    require(
        digest_json(payload) == receipt.get("evidence_id"),
        "timeline receipt evidence id is invalid",
    )
    inputs = receipt.get("inputs")
    outputs = receipt.get("outputs")
    provenance = receipt.get("provenance")
    coverage = receipt.get("coverage")
    require(isinstance(inputs, dict), "timeline inputs are missing")
    require(isinstance(outputs, dict), "timeline outputs are missing")
    require(isinstance(provenance, dict), "timeline provenance is missing")
    require(isinstance(coverage, dict), "timeline coverage evidence is missing")
    for name in ("raw", "manifest", "event_dictionary", "correctness"):
        verify_descriptor(path, inputs.get(name), f"inputs.{name}")
    for name in ("canonical", "perfetto", "summary"):
        expected = expected_outputs.get(name) if expected_outputs else None
        verify_descriptor(path, outputs.get(name), f"outputs.{name}", expected_path=expected)
    verify_descriptor_tree(path, provenance, "provenance")
    instrumented_launch = coverage.get("instrumented_launch")
    if instrumented_launch is not None:
        verify_descriptor_tree(path, instrumented_launch, "coverage.instrumented_launch")
    require(
        payload.get("instrumented_launch") == binding_view(instrumented_launch),
        "timeline coverage binding drifted",
    )
    for field in ("instrumented_sources", "compiled_binaries", "workload_inputs"):
        rows = provenance.get(field)
        require(isinstance(rows, list), f"timeline provenance.{field} must be a list")
        if field == "instrumented_sources" or grade == "decision":
            require(bool(rows), f"{grade} timeline evidence requires provenance.{field}")
    require(payload.get("inputs") == binding_view(inputs), "timeline input bindings drifted")
    require(
        payload.get("provenance") == binding_view(provenance),
        "timeline provenance bindings drifted",
    )
    kernel = provenance.get("authoritative_kernel")
    if grade == "decision" or kernel is not None:
        verify_descriptor(path, kernel, "provenance.authoritative_kernel")
        kernel_sha256 = require_sha256(receipt.get("kernel_sha256"), "kernel_sha256")
        require(kernel.get("sha256") == kernel_sha256, "authoritative kernel hash mismatch")
        require(
            payload.get("kernel_sha256") == kernel_sha256,
            "timeline binding payload kernel hash mismatch",
        )
        correctness_path = _descriptor_path(
            path, inputs["correctness"], "inputs.correctness"
        )
        correctness = load_json(correctness_path, "timeline correctness evidence")
        require(
            correctness.get("schema") == "ppu-timeline-correctness/v2"
            and correctness.get("validation") == "accepted",
            "timeline correctness evidence must be accepted ppu-timeline-correctness/v2",
        )
        require(
            correctness.get("kernel_sha256") == kernel_sha256,
            "timeline correctness evidence does not bind the authoritative kernel",
        )
        identity = receipt.get("identity")
        require(isinstance(identity, dict), "timeline receipt identity is missing")
        require(
            correctness.get("kernel_name") == identity.get("kernel_name"),
            "timeline correctness kernel identity mismatch",
        )
        require(
            correctness.get("workload_identity") == identity.get("workload"),
            "timeline correctness workload identity mismatch",
        )
        require(
            correctness.get("device_identity") == identity.get("device"),
            "timeline correctness device identity mismatch",
        )
        checks = correctness.get("checks")
        require(
            isinstance(checks, list)
            and bool(checks)
            and all(
                isinstance(check, dict)
                and isinstance(check.get("name"), str)
                and bool(check["name"].strip())
                and check.get("status") == "passed"
                for check in checks
            ),
            "timeline correctness requires named passed checks",
        )
    if recompute_outputs:
        try:
            from .timeline import decode
        except ImportError:
            from timeline import decode

        raw_path = _descriptor_path(path, inputs["raw"], "inputs.raw")
        manifest_path = _descriptor_path(
            path, inputs["manifest"], "inputs.manifest"
        )
        dictionary_path = _descriptor_path(
            path, inputs["event_dictionary"], "inputs.event_dictionary"
        )
        with tempfile.TemporaryDirectory(prefix="ppu-timeline-validate-") as directory:
            prefix = Path(directory) / "recomputed"
            decode(
                raw_path,
                manifest_path,
                dictionary_path,
                prefix,
                self_validate=False,
            )
            regenerated_receipt = load_json(
                Path(f"{prefix}.receipt.json"), "regenerated timeline receipt"
            )
            for field in (
                "evidence_id",
                "evidence_grade",
                "kernel_sha256",
                "binding_payload",
                "identity",
            ):
                require(
                    receipt.get(field) == regenerated_receipt.get(field),
                    f"timeline receipt {field} does not match manifest inputs",
                )
            require(
                binding_view(receipt.get("inputs"))
                == binding_view(regenerated_receipt.get("inputs")),
                "timeline receipt inputs do not match manifest inputs",
            )
            require(
                binding_view(receipt.get("provenance"))
                == binding_view(regenerated_receipt.get("provenance")),
                "timeline receipt provenance does not match manifest inputs",
            )
            require(
                binding_view(receipt.get("coverage"))
                == binding_view(regenerated_receipt.get("coverage")),
                "timeline receipt coverage does not match manifest inputs",
            )
            for name in ("canonical", "perfetto", "summary"):
                regenerated = Path(f"{prefix}.{name}.json")
                require(
                    sha256_file(regenerated) == outputs[name]["sha256"],
                    f"timeline {name} does not match raw inputs",
                )
    return receipt
