#!/usr/bin/env python3
"""Export time-windowed PM samples from an immutable ACU 2.2 report."""

from __future__ import annotations

import argparse
import csv
import json
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
    if start_ns is None or end_ns is None or value is None or end_ns <= start_ns:
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


def export(report_path: Path, csv_path: Path, metadata_path: Path) -> dict:
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

    by_window_metric = {
        (row["window_start_ns"], row["window_end_ns"], row["metric_name"]): float(
            row["metric_value"]
        )
        for row in rows
    }
    for row in rows:
        if row["validity"] != "pending_activity_check":
            continue
        activity_names = ACTIVITY_METRICS.get(row["metric_name"])
        if activity_names is None:
            row["validity"] = "unknown_no_activity_counter"
            continue
        values = [
            by_window_metric.get((row["window_start_ns"], row["window_end_ns"], name))
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

    metadata = {
        "schema_version": 1,
        "source": "ACU .acurep Perfetto protobuf field 88/message field 15",
        "time_semantics": "each row covers [window_start_ns, window_end_ns)",
        "alignment": "kernel-relative, no duration rescaling",
        "scope": "device/global aggregate; not attributable to one CTA or range",
        "replay_semantics": (
            "packet headers are preserved; logical groups do not prove simultaneity"
        ),
        "packets": packet_summaries,
        "row_count": len(rows),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    metadata = export(args.report, args.csv, args.metadata)
    print(
        f"exported {metadata['row_count']} PM rows from "
        f"{len(metadata['packets'])} packet(s)"
    )
    return 0


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()
