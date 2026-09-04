#!/usr/bin/env python3
"""Validate and export raw launch facts and PM samples from an ACU 2.2 report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Sequence


METRIC_GROUPS = {
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

ACTIVITY_METRICS = {
    "ksd__requests_hit_rate.pct": (
        "ksd__requests_load_pipe_ws.sum",
        "ksd__requests_store_pipe_ws.sum",
    ),
    "kvd__requests_hit_rate.pct": (
        "kvd__requests_load_pipe_lsu.sum",
        "kvd__requests_store_pipe_lsu.sum",
    ),
}
COLLECTION_SCHEMA = "ppu-acu-collection/v1"
EXTRACTION_SCHEMA = "ppu-acu-extraction/v3"
SUPPORTED_PRODUCER = {"name": "acu", "version": "2.2"}
EVIDENCE_GRADES = {"diagnostic", "decision"}
VALID_PM_STATES = {"valid", "valid_activity_positive"}
MIN_SAMPLES = 10


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_descriptor(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"artifact is not a regular file: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _load_collection(path: Path, report_path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read ACU collection descriptor {path}: {error}") from error
    if not isinstance(value, dict) or value.get("schema") != COLLECTION_SCHEMA:
        raise RuntimeError(f"ACU collection descriptor must use {COLLECTION_SCHEMA}")
    if value.get("producer") != SUPPORTED_PRODUCER:
        raise RuntimeError("ACU producer must be exactly acu 2.2")
    producer_artifact = value.get("producer_artifact")
    if not isinstance(producer_artifact, dict):
        raise RuntimeError("collection.producer_artifact must be an object")
    producer_path = producer_artifact.get("path")
    producer_identity = producer_artifact.get("identity")
    if not isinstance(producer_path, str) or not producer_path.strip():
        raise RuntimeError("collection.producer_artifact.path is required")
    if not isinstance(producer_identity, str) or not producer_identity.strip():
        raise RuntimeError("collection.producer_artifact.identity is required")
    resolved_producer = Path(producer_path)
    if not resolved_producer.is_absolute():
        resolved_producer = path.parent / resolved_producer
    try:
        producer_text = resolved_producer.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise RuntimeError(f"cannot read ACU producer artifact: {error}") from error
    if "acu" not in producer_text.lower() or not re.search(
        r"(?<![0-9])2\.2(?:\.[0-9]+)?(?![0-9])", producer_text
    ):
        raise RuntimeError("producer artifact does not identify ACU 2.2")
    grade = value.get("evidence_grade")
    if grade not in EVIDENCE_GRADES:
        raise RuntimeError("evidence_grade must be diagnostic or decision")
    declared_report = value.get("report")
    if not isinstance(declared_report, str) or not declared_report.strip():
        raise RuntimeError("collection.report is required")
    declared_report_path = Path(declared_report)
    if not declared_report_path.is_absolute():
        declared_report_path = path.parent / declared_report_path
    if declared_report_path.resolve() != report_path.resolve():
        raise RuntimeError("collection.report does not identify the extracted report")
    required_text = (
        "kernel_name",
        "kernel_specialization",
        "workload_identity",
        "cache_policy",
        "clock_configuration",
    )
    for field in required_text:
        if not isinstance(value.get(field), str) or not value[field].strip():
            raise RuntimeError(f"collection.{field} is required")
    for field in ("device_identity", "runtime_identity"):
        if not isinstance(value.get(field), dict) or not value[field]:
            raise RuntimeError(f"collection.{field} is required")
    physical_device = value["device_identity"].get("physical_device")
    if (
        not isinstance(physical_device, int)
        or isinstance(physical_device, bool)
        or physical_device < 0
    ):
        raise RuntimeError(
            "collection.device_identity.physical_device must be a non-negative integer"
        )
    requested_metrics = value.get("requested_metrics")
    if requested_metrics is not None:
        if (
            not isinstance(requested_metrics, list)
            or any(
                not isinstance(metric, str) or not metric.strip()
                for metric in requested_metrics
            )
        ):
            raise RuntimeError(
                "collection.requested_metrics must be a list of non-empty strings"
            )
        normalized_metrics = [metric.strip() for metric in requested_metrics]
        if len(set(normalized_metrics)) != len(normalized_metrics):
            raise RuntimeError(
                "collection.requested_metrics must not contain duplicates"
            )
        value["requested_metrics"] = normalized_metrics

    artifacts: dict[str, list[dict]] = {}
    for field in ("source_artifacts", "binary_artifacts", "workload_inputs"):
        rows = value.get(field, [])
        if not isinstance(rows, list):
            raise RuntimeError(f"collection.{field} must be a list")
        if field == "source_artifacts" or grade == "decision":
            if not rows:
                raise RuntimeError(f"{grade} evidence requires collection.{field}")
        descriptors = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise RuntimeError(f"collection.{field}[{index}] must be an object")
            artifact_path = row.get("path")
            identity = row.get("identity")
            if not isinstance(artifact_path, str) or not artifact_path.strip():
                raise RuntimeError(f"collection.{field}[{index}].path is required")
            if not isinstance(identity, str) or not identity.strip():
                raise RuntimeError(f"collection.{field}[{index}].identity is required")
            resolved = Path(artifact_path)
            if not resolved.is_absolute():
                resolved = path.parent / resolved
            descriptor = _file_descriptor(resolved)
            descriptor["declared_path"] = artifact_path
            descriptor["identity"] = identity.strip()
            descriptors.append(descriptor)
        artifacts[field] = descriptors
    producer_descriptor = _file_descriptor(resolved_producer)
    producer_descriptor["declared_path"] = producer_path
    producer_descriptor["identity"] = producer_identity.strip()
    return {
        **value,
        "bound_artifacts": {**artifacts, "producer": producer_descriptor},
    }


def _varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(data) and shift < 70:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
    raise ValueError("truncated or overlong protobuf varint")


def _fields(data: bytes) -> Iterator[tuple[int, int, int | bytes]]:
    offset = 0
    while offset < len(data):
        tag, offset = _varint(data, offset)
        number = tag >> 3
        wire_type = tag & 7
        if number == 0:
            raise ValueError("invalid protobuf field number 0")
        if wire_type == 0:
            value, offset = _varint(data, offset)
        elif wire_type in (1, 5):
            size = 8 if wire_type == 1 else 4
            end = offset + size
            if end > len(data):
                raise ValueError("truncated protobuf fixed-width field")
            value = data[offset:end]
            offset = end
        elif wire_type == 2:
            length, offset = _varint(data, offset)
            end = offset + length
            if end > len(data):
                raise ValueError("truncated protobuf length-delimited field")
            value = data[offset:end]
            offset = end
        else:
            raise ValueError(f"unsupported protobuf wire type {wire_type}")
        yield number, wire_type, value


def _parse_sample(message: bytes) -> tuple[int, int, float]:
    start_ns = end_ns = None
    value = None
    for number, wire_type, raw in _fields(message):
        if number == 1 and wire_type == 0:
            start_ns = int(raw)
        elif number == 2 and wire_type == 1:
            value = struct.unpack("<d", raw)[0]
        elif number == 3 and wire_type == 0:
            end_ns = int(raw)
    if (
        start_ns is None
        or end_ns is None
        or value is None
        or not math.isfinite(value)
        or end_ns <= start_ns
    ):
        raise ValueError("invalid ACU PM sample")
    return start_ns, end_ns, value


def _parse_metric(message: bytes) -> tuple[str, list[tuple[int, int, float]]]:
    name = None
    samples = []
    for number, wire_type, raw in _fields(message):
        if number == 1 and wire_type == 2:
            name = bytes(raw).decode("utf-8")
        elif number == 3 and wire_type == 2:
            samples.append(_parse_sample(bytes(raw)))
    if name is None or not samples:
        raise ValueError("invalid ACU PM metric")
    return name, samples


def _parse_payload(
    message: bytes,
) -> tuple[dict[int, int], list[tuple[str, list[tuple[int, int, float]]]]] | None:
    headers = {}
    metrics = []
    for number, wire_type, raw in _fields(message):
        if number in (1, 2, 4, 5) and wire_type == 0:
            headers[number] = int(raw)
        elif number == 3 and wire_type == 2:
            metrics.append(_parse_metric(bytes(raw)))
    return (headers, metrics) if metrics else None


def extract_pm_packets(
    report: bytes,
) -> list[tuple[dict[int, int], list[tuple[str, list[tuple[int, int, float]]]]]]:
    """Decode ACU TracePacket field 88 / nested message field 15."""
    packets = []
    for trace_number, trace_wire, trace_packet in _fields(report):
        if trace_number != 1 or trace_wire != 2:
            continue
        for packet_number, packet_wire, packet_value in _fields(bytes(trace_packet)):
            if packet_number != 88 or packet_wire != 2:
                continue
            for nested_number, nested_wire, nested_value in _fields(
                bytes(packet_value)
            ):
                if nested_number == 15 and nested_wire == 2:
                    parsed = _parse_payload(bytes(nested_value))
                    if parsed is not None:
                        packets.append(parsed)
    return packets


def _metric_unit(name: str) -> str:
    if name.endswith(".pct") or ".pct_of_peak_" in name:
        return "percent"
    if "per_cycle" in name:
        return "per_cycle"
    if "cycles" in name:
        return "cycles"
    if "requests_" in name and name.endswith(".sum"):
        return "requests"
    return "unitless"


def _parse_dims(value: str, field: str) -> list[int]:
    try:
        result = [int(part.strip()) for part in value.strip().strip("()").split(",")]
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError(f"ACU raw {field} is invalid") from error
    if len(result) != 3 or any(item <= 0 for item in result):
        raise RuntimeError(
            f"ACU raw {field} must contain three positive integers"
        )
    return result


def _finite_number(row: dict[str, str], field: str, *, positive: bool) -> float:
    raw = row.get(field)
    try:
        value = float(raw) if raw not in (None, "") else math.nan
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"ACU raw {field} is not numeric") from error
    if (
        not math.isfinite(value)
        or (positive and value <= 0)
        or (not positive and value < 0)
    ):
        qualifier = "positive and finite" if positive else "non-negative and finite"
        raise RuntimeError(f"ACU raw {field} must be {qualifier}")
    return value


def _integer_number(row: dict[str, str], field: str, *, positive: bool) -> int:
    value = _finite_number(row, field, positive=positive)
    if not value.is_integer():
        raise RuntimeError(f"ACU raw {field} must be an integer")
    return int(value)


def _optional_number(
    row: dict[str, str], field: str, *, integer: bool = False
) -> int | float | None:
    raw = row.get(field)
    if raw in (None, ""):
        return None
    value = _finite_number(row, field, positive=False)
    if integer:
        if not value.is_integer():
            raise RuntimeError(f"ACU raw {field} must be an integer")
        return int(value)
    return value


def _read_raw_csv(
    path: Path, collection: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
    except (OSError, UnicodeError, csv.Error) as error:
        raise RuntimeError(f"cannot read ACU raw CSV {path}: {error}") from error
    if len(rows) != 1:
        raise RuntimeError("ACU raw CSV must contain exactly one filtered kernel row")
    row = rows[0]
    required_fields = (
        "Kernel Name",
        "Grid Size",
        "Block Size",
        "Device",
        "ppu__time_duration.sum",
        "device__attribute_cu_count",
        "launch__occupancy_blocks_per_cu",
        "launch__registers_per_thread",
        "launch__shared_mem_per_block",
    )
    missing = [field for field in required_fields if row.get(field) in (None, "")]
    if missing:
        raise RuntimeError(
            "ACU raw CSV is missing required fields: " + ", ".join(missing)
        )

    launch = {
        "kernel_name": row["Kernel Name"],
        "grid": _parse_dims(row["Grid Size"], "Grid Size"),
        "block": _parse_dims(row["Block Size"], "Block Size"),
        "device": _integer_number(row, "Device", positive=False),
        "duration_ns": _finite_number(
            row, "ppu__time_duration.sum", positive=True
        ),
        "pm_interval_ns": _optional_number(row, "pmsampler__interval_time.max"),
        "dropped_samples": _optional_number(
            row, "pmsampler__dropped_samples.max", integer=True
        ),
        "buffer_size_bytes": _optional_number(
            row, "pmsampler__buffer_size_bytes.max", integer=True
        ),
        "cu_count": _integer_number(
            row, "device__attribute_cu_count", positive=True
        ),
        "occupancy_blocks_per_cu": _finite_number(
            row, "launch__occupancy_blocks_per_cu", positive=True
        ),
        "registers_per_thread": _integer_number(
            row, "launch__registers_per_thread", positive=False
        ),
        "shared_mem_per_block": _integer_number(
            row, "launch__shared_mem_per_block", positive=False
        ),
    }
    errors = []
    if launch["kernel_name"] != collection["kernel_name"]:
        errors.append("ACU raw kernel name does not match collection.kernel_name")
    if launch["device"] != collection["device_identity"]["physical_device"]:
        errors.append(
            "ACU raw physical device does not match collection.device_identity"
        )
    return launch, errors


def _validity_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        validity = row["validity"]
        result[validity] = result.get(validity, 0) + 1
    return result


def _summarize_streams(
    rows: list[dict[str, Any]], duration_ns: float
) -> tuple[dict[str, dict[str, Any]], list[str], list[str], list[str]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["packet_index"], row["metric_name"])].append(row)

    summaries: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []
    for (packet_index, metric_name), stream in sorted(grouped.items()):
        label = f"packet {packet_index} metric {metric_name}"
        ordered = sorted(stream, key=lambda row: row["sample_index"])
        if [row["sample_index"] for row in ordered] != list(range(len(ordered))):
            errors.append(f"{label} has non-contiguous sample indices")
        if ordered[0]["window_start_ns"] != 0:
            errors.append(f"{label} does not start at kernel-relative zero")
        if any(
            left["window_end_ns"] != right["window_start_ns"]
            for left, right in zip(ordered, ordered[1:])
        ):
            errors.append(f"{label} has a gap or overlap between PM windows")

        coverage_end_ns = max(row["window_end_ns"] for row in ordered)
        if coverage_end_ns > duration_ns:
            errors.append(f"{label} extends beyond the ACU kernel duration")
        interval_max_ns = max(row["interval_ns"] for row in ordered)
        final_gap_ns = duration_ns - coverage_end_ns
        if final_gap_ns > interval_max_ns:
            warnings.append(
                f"{label} leaves {final_gap_ns:g} ns of the kernel tail unsampled"
            )
        elif final_gap_ns > 0:
            notes.append(
                f"{label} omits a final partial interval of {final_gap_ns:g} ns"
            )

        valid_rows = [row for row in ordered if row["validity"] in VALID_PM_STATES]
        if valid_rows and len(valid_rows) < MIN_SAMPLES:
            warnings.append(
                f"{label} has only {len(valid_rows)} interpretable samples"
            )
        valid_interval_ns = sum(row["interval_ns"] for row in valid_rows)
        values = [row["metric_value_number"] for row in valid_rows]
        summaries[f"packet_{packet_index}:{metric_name}"] = {
            "packet_index": packet_index,
            "metric_name": metric_name,
            "logical_metric_group": ordered[0]["logical_metric_group"],
            "scope": ordered[0]["scope"],
            "unit": ordered[0]["metric_unit"],
            "sample_count": len(ordered),
            "valid_sample_count": len(valid_rows),
            "excluded_sample_count": len(ordered) - len(valid_rows),
            "validity_counts": _validity_counts(ordered),
            "coverage_end_ns": coverage_end_ns,
            "coverage_ratio": min(coverage_end_ns / duration_ns, 1.0),
            "interval_min_ns": min(row["interval_ns"] for row in ordered),
            "interval_max_ns": interval_max_ns,
            "time_weighted_mean": (
                sum(
                    row["metric_value_number"] * row["interval_ns"]
                    for row in valid_rows
                )
                / valid_interval_ns
                if valid_interval_ns
                else None
            ),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    return summaries, errors, warnings, notes


def export(
    report_path: Path,
    raw_csv_path: Path,
    collection_path: Path,
    csv_path: Path,
    metadata_path: Path,
) -> dict:
    collection = _load_collection(collection_path, report_path)
    launch, raw_errors = _read_raw_csv(raw_csv_path, collection)
    packets = extract_pm_packets(report_path.read_bytes())
    if not packets:
        raise RuntimeError(
            "no ACU PM payload; expected ACU 2.2 field 88/message field 15"
        )

    rows = []
    packet_summaries = []
    for packet_index, (headers, metrics) in enumerate(packets):
        for metric_name, samples in metrics:
            known_metric = metric_name in METRIC_GROUPS
            for sample_index, (start_ns, end_ns, value) in enumerate(samples):
                rows.append(
                    {
                        "packet_index": packet_index,
                        "logical_metric_group": METRIC_GROUPS.get(
                            metric_name, "unclassified"
                        ),
                        "header_field_1": headers.get(1, ""),
                        "header_field_2": headers.get(2, ""),
                        "header_field_4": headers.get(4, ""),
                        "header_field_5": headers.get(5, ""),
                        "metric_name": metric_name,
                        "metric_unit": (
                            _metric_unit(metric_name) if known_metric else "unknown"
                        ),
                        "sample_index": sample_index,
                        "window_start_ns": start_ns,
                        "window_end_ns": end_ns,
                        "interval_ns": end_ns - start_ns,
                        "metric_value": f"{value:.17g}",
                        "metric_value_number": value,
                        "scope": (
                            "device_global_aggregate" if known_metric else "unknown"
                        ),
                        "validity": (
                            "pending_activity_check"
                            if known_metric
                            and metric_name.endswith("requests_hit_rate.pct")
                            else ("valid" if known_metric else "unknown_semantics")
                        ),
                    }
                )
        packet_summaries.append(
            {
                "packet_index": packet_index,
                "raw_header_fields": {
                    str(key): value for key, value in sorted(headers.items())
                },
                "metric_names": [name for name, _ in metrics],
                "sample_count_min": min(len(samples) for _, samples in metrics),
                "sample_count_max": max(len(samples) for _, samples in metrics),
            }
        )

    by_window_metric = {}
    for row in rows:
        key = (
            row["packet_index"],
            row["window_start_ns"],
            row["window_end_ns"],
            row["metric_name"],
        )
        if key in by_window_metric:
            raise RuntimeError(f"duplicate ACU PM sample identity: {key}")
        by_window_metric[key] = float(row["metric_value"])
    for row in rows:
        if row["validity"] != "pending_activity_check":
            continue
        activity_names = ACTIVITY_METRICS.get(row["metric_name"])
        if activity_names is None:
            row["validity"] = "unknown_no_activity_counter"
            continue
        values = [
            by_window_metric.get(
                (
                    row["packet_index"],
                    row["window_start_ns"],
                    row["window_end_ns"],
                    name,
                )
            )
            for name in activity_names
        ]
        available = [value for value in values if value is not None]
        if any(value > 0 for value in available):
            row["validity"] = "valid_activity_positive"
        elif len(available) == len(activity_names):
            row["validity"] = "no_activity_hit_rate_not_interpretable"
        else:
            row["validity"] = "unknown_no_activity_counter"

    stream_summaries, stream_errors, warnings, notes = _summarize_streams(
        rows, launch["duration_ns"]
    )
    errors = [*raw_errors, *stream_errors]
    dropped_samples = launch["dropped_samples"]
    if dropped_samples is None:
        notes.append(
            "ACU raw page omitted dropped-sample aggregate; "
            "PM window continuity was checked"
        )
    elif dropped_samples > 0:
        warnings.append(f"ACU reported {dropped_samples} dropped PM samples")

    observed_intervals = sorted({row["interval_ns"] for row in rows})
    reported_interval = launch["pm_interval_ns"]
    if reported_interval is not None and reported_interval != max(observed_intervals):
        errors.append(
            "ACU raw PM interval disagrees with the maximum exact sample window"
        )
    if len(observed_intervals) > 1:
        median_interval = sorted(observed_intervals)[len(observed_intervals) // 2]
        jitter = (max(observed_intervals) - min(observed_intervals)) / median_interval
        message = f"PM intervals are {observed_intervals} (jitter {jitter:.3%})"
        (warnings if jitter > 0.01 else notes).append(message)

    observed_metrics = sorted({row["metric_name"] for row in rows})
    requested_metrics = collection.get("requested_metrics")
    missing_requested_metrics = (
        sorted(set(requested_metrics) - set(observed_metrics))
        if requested_metrics is not None
        else []
    )
    if missing_requested_metrics:
        warnings.append(
            "ACU report omitted requested metrics: "
            + ", ".join(missing_requested_metrics)
        )
    uninterpretable_requested_metrics = []
    if requested_metrics is not None:
        for metric_name in sorted(set(requested_metrics) & set(observed_metrics)):
            metric_rows = [row for row in rows if row["metric_name"] == metric_name]
            if not any(
                row["validity"] in VALID_PM_STATES for row in metric_rows
            ):
                uninterpretable_requested_metrics.append(metric_name)
    if uninterpretable_requested_metrics:
        warnings.append(
            "requested metrics lack interpretable samples: "
            + ", ".join(uninterpretable_requested_metrics)
        )
    unknown_metrics = sorted(set(observed_metrics) - set(METRIC_GROUPS))
    if unknown_metrics:
        notes.append(
            "unknown metrics were preserved without inferred scope or validity: "
            + ", ".join(unknown_metrics)
        )
    if not any(row["validity"] in VALID_PM_STATES for row in rows):
        warnings.append(
            "ACU report contains no metric with validated interpretation semantics"
        )

    status = "rejected" if errors else ("warning" if warnings else "accepted")
    csv_rows = [
        {key: value for key, value in row.items() if key != "metric_value_number"}
        for row in rows
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    input_artifacts = {
        "report": _file_descriptor(report_path),
        "raw_csv": _file_descriptor(raw_csv_path),
        "collection": _file_descriptor(collection_path),
    }
    output_artifact = _file_descriptor(csv_path)
    identity = {
        field: collection[field]
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
    evidence_payload = {
        "producer": collection["producer"],
        "identity": identity,
        "inputs": input_artifacts,
        "bound_artifacts": collection["bound_artifacts"],
        "pm_csv": output_artifact,
    }
    evidence_id = hashlib.sha256(_canonical_json(evidence_payload)).hexdigest()
    metadata = {
        "schema": EXTRACTION_SCHEMA,
        "validation": {
            "status": status,
            "errors": errors,
            "warnings": warnings,
            "notes": notes,
        },
        "evidence_id": evidence_id,
        "evidence_grade": collection["evidence_grade"],
        "producer": collection["producer"],
        "identity": identity,
        "inputs": input_artifacts,
        "bound_artifacts": collection["bound_artifacts"],
        "outputs": {"pm_csv": output_artifact},
        "source": "ACU .acurep Perfetto protobuf field 88/message field 15",
        "time_semantics": "each row covers [window_start_ns, window_end_ns)",
        "alignment": "kernel-relative, no duration rescaling",
        "scope": (
            "verified metrics are device/global aggregates; unknown metrics retain "
            "unknown scope; neither is attributed to one CTA or source range"
        ),
        "replay_semantics": (
            "packet headers are preserved; logical groups do not prove simultaneity"
        ),
        "packets": packet_summaries,
        "row_count": len(rows),
        "validity_counts": _validity_counts(rows),
        "launch": launch,
        "sampling": {
            "requested_metrics": requested_metrics,
            "observed_metrics": observed_metrics,
            "missing_requested_metrics": missing_requested_metrics,
            "uninterpretable_requested_metrics": (
                uninterpretable_requested_metrics
            ),
            "unknown_metrics": unknown_metrics,
            "observed_intervals_ns": observed_intervals,
            "minimum_interpretable_samples_per_metric": MIN_SAMPLES,
        },
        "metric_summaries": stream_summaries,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    metadata = export(
        args.report,
        args.raw_csv,
        args.collection,
        args.csv,
        args.metadata,
    )
    print(
        f"ACU extraction {metadata['validation']['status']}: exported "
        f"{metadata['row_count']} PM rows from {len(metadata['packets'])} packet(s)"
    )
    return 2 if metadata["validation"]["status"] == "rejected" else 0


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()
