#!/usr/bin/env python3
"""Export time-windowed PM samples from an immutable ACU 2.2 report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import struct
from pathlib import Path
from typing import Iterator, Sequence


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
EXTRACTION_SCHEMA = "ppu-acu-extraction/v2"
SUPPORTED_PRODUCER = {"name": "acu", "version": "2.2"}
EVIDENCE_GRADES = {"diagnostic", "decision"}


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


def export(
    report_path: Path,
    raw_csv_path: Path,
    collection_path: Path,
    csv_path: Path,
    metadata_path: Path,
) -> dict:
    collection = _load_collection(collection_path, report_path)
    packets = extract_pm_packets(report_path.read_bytes())
    if not packets:
        raise RuntimeError(
            "no ACU PM payload; expected ACU 2.2 field 88/message field 15"
        )

    rows = []
    packet_summaries = []
    for packet_index, (headers, metrics) in enumerate(packets):
        for metric_name, samples in metrics:
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
                        "metric_unit": _metric_unit(metric_name),
                        "sample_index": sample_index,
                        "window_start_ns": start_ns,
                        "window_end_ns": end_ns,
                        "interval_ns": end_ns - start_ns,
                        "metric_value": f"{value:.17g}",
                        "scope": "device_global_aggregate",
                        "validity": (
                            "pending_activity_check"
                            if metric_name.endswith("requests_hit_rate.pct")
                            else "valid"
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

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

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
    validity_counts: dict[str, int] = {}
    for row in rows:
        validity = row["validity"]
        validity_counts[validity] = validity_counts.get(validity, 0) + 1
    metadata = {
        "schema": EXTRACTION_SCHEMA,
        "validation": "accepted",
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
        "scope": "device/global aggregate; not attributable to one CTA or range",
        "replay_semantics": (
            "packet headers are preserved; logical groups do not prove simultaneity"
        ),
        "packets": packet_summaries,
        "row_count": len(rows),
        "validity_counts": validity_counts,
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
        f"exported {metadata['row_count']} PM rows from "
        f"{len(metadata['packets'])} packet(s)"
    )
    return 0


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()
