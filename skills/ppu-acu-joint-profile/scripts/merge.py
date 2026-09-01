#!/usr/bin/env python3
"""Merge a PPU sparse timeline with ACU PM windows without time rescaling."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


WARNING_DELTA = 0.03
REJECT_DELTA = 0.05
MIN_SAMPLES = 10
MEASUREMENT_SCHEMA = "ppu-timeline-measurement/v1"


def _parse_dims(value: str) -> list[int]:
    return [int(part.strip()) for part in value.strip().strip("()").split(",")]


def _optional_number(row: dict[str, str], name: str, cast):
    value = row.get(name)
    return None if value in (None, "") else cast(float(value))


def _read_acu_raw(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 1:
        raise RuntimeError("ACU raw CSV must contain exactly one filtered kernel row")
    row = rows[0]
    return {
        "kernel_name": row["Kernel Name"],
        "grid": _parse_dims(row["Grid Size"]),
        "block": _parse_dims(row["Block Size"]),
        "device": int(row["Device"]),
        "duration_ns": float(row["ppu__time_duration.sum"]),
        "pm_interval_ns": _optional_number(row, "pmsampler__interval_time.max", float),
        "dropped_samples": _optional_number(row, "pmsampler__dropped_samples.max", int),
        "buffer_size_bytes": _optional_number(
            row, "pmsampler__buffer_size_bytes.max", int
        ),
        "cu_count": int(float(row["device__attribute_cu_count"])),
        "occupancy_blocks_per_cu": float(row["launch__occupancy_blocks_per_cu"]),
        "registers_per_thread": int(float(row["launch__registers_per_thread"])),
        "shared_mem_per_block": int(float(row["launch__shared_mem_per_block"])),
    }


def _read_pm(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise RuntimeError("PM CSV contains no samples")
    for row in rows:
        row["packet_index"] = int(row["packet_index"])
        row["sample_index"] = int(row["sample_index"])
        row["window_start_ns"] = int(row["window_start_ns"])
        row["window_end_ns"] = int(row["window_end_ns"])
        row["interval_ns"] = int(row["interval_ns"])
        row["metric_value"] = float(row["metric_value"])
    return rows


def _read_measurement(path: Path, label: str) -> dict[str, Any]:
    measurement = json.loads(path.read_text(encoding="utf-8"))
    if measurement.get("schema") != MEASUREMENT_SCHEMA:
        raise RuntimeError(f"{label} has an unsupported measurement schema")
    runs = measurement.get("runs")
    if not isinstance(runs, list) or not runs:
        raise RuntimeError(f"{label} has no raw ordered samples")
    schedule = measurement.get("schedule")
    if (
        not isinstance(schedule, list)
        or not schedule
        or any(
            not isinstance(group, str) or not group or set(group) - {"A", "B"}
            for group in schedule
        )
    ):
        raise RuntimeError(f"{label} has an invalid schedule")
    expected_order = [arm for group in schedule for arm in group]
    if len(runs) != len(expected_order):
        raise RuntimeError(f"{label} schedule does not match its raw samples")
    latencies: dict[str, list[float]] = {"A": [], "B": []}
    for index, (run, expected_arm) in enumerate(zip(runs, expected_order)):
        if not isinstance(run, dict) or run.get("order") != index:
            raise RuntimeError(f"{label} has invalid sample order")
        arm = run.get("arm")
        sample = run.get("sample")
        if arm != expected_arm or arm not in latencies or not isinstance(sample, dict):
            raise RuntimeError(f"{label} has invalid sample arm")
        latency = sample.get("latency_ms")
        if (
            not isinstance(latency, (int, float))
            or isinstance(latency, bool)
            or not math.isfinite(latency)
            or latency <= 0
        ):
            raise RuntimeError(f"{label} has invalid sample latency")
        if (
            sample.get("correctness") != "passed"
            or sample.get("synchronized") is not True
        ):
            raise RuntimeError(f"{label} has an invalid timing sample")
        if sample.get("workload_identity") != measurement.get("workload_identity"):
            raise RuntimeError(f"{label} sample workload identity drifted")
        if sample.get("device_identity") != measurement.get("device_identity"):
            raise RuntimeError(f"{label} sample device identity drifted")
        if sample.get("warmup") != measurement.get("warmup") or sample.get(
            "iterations"
        ) != measurement.get("iterations"):
            raise RuntimeError(f"{label} sample timing configuration drifted")
        latencies[arm].append(float(latency))
    if not latencies["A"] or not latencies["B"]:
        raise RuntimeError(f"{label} must contain both A and B samples")
    recomputed = {
        "baseline_median_ms": statistics.median(latencies["A"]),
        "instrumented_median_ms": statistics.median(latencies["B"]),
    }
    recomputed["relative_overhead"] = (
        recomputed["instrumented_median_ms"] / recomputed["baseline_median_ms"] - 1.0
    )
    summary = measurement.get("summary")
    summary_matches = isinstance(summary, dict)
    if summary_matches:
        for name, value in recomputed.items():
            reported = summary.get(name)
            if (
                not isinstance(reported, (int, float))
                or isinstance(reported, bool)
                or not math.isclose(float(reported), value, rel_tol=1e-12)
            ):
                summary_matches = False
                break
    if not summary_matches:
        raise RuntimeError(f"{label} summary does not match its raw samples")
    return measurement


def _event_interval_ns(event: dict[str, Any]) -> tuple[float, float] | None:
    if event.get("ph") != "X":
        return None
    start = float(event.get("ts", 0)) * 1000.0
    return start, start + float(event.get("dur", 0)) * 1000.0


def _overlaps(left_start: float, left_end: float, right_start: float, right_end: float):
    return left_start < right_end and right_start < left_end


def _metric_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["metric_name"]].append(row["metric_value"])
    return {
        name: {
            "samples": len(values),
            "mean": statistics.fmean(values),
            "min": min(values),
            "max": max(values),
        }
        for name, values in sorted(grouped.items())
    }


def _validate_pm_windows(
    by_metric: dict[str, list[dict[str, Any]]], errors: list[str]
) -> None:
    for name, rows in by_metric.items():
        if len(rows) < MIN_SAMPLES:
            errors.append(f"metric {name} has only {len(rows)} samples")
        ordered = sorted(rows, key=lambda row: row["sample_index"])
        if [row["sample_index"] for row in ordered] != list(range(len(ordered))):
            errors.append(f"metric {name} has non-contiguous sample indices")
        if ordered[0]["window_start_ns"] != 0:
            errors.append(f"metric {name} does not start at kernel-relative zero")
        if any(
            left["window_end_ns"] != right["window_start_ns"]
            for left, right in zip(ordered, ordered[1:])
        ):
            errors.append(f"metric {name} has a gap or overlap between PM windows")


def merge(
    timeline_path: Path,
    pm_path: Path,
    acu_raw_path: Path,
    output_prefix: Path,
    perturbation_path: Path | None = None,
    density_sensitivity_path: Path | None = None,
    allow_duration_mismatch: bool = False,
) -> dict[str, Any]:
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    timeline_meta = timeline.get("ppuTimeline", {})
    if timeline_meta.get("schemaVersion") != 4:
        raise RuntimeError("PPU timeline schemaVersion 4 is required")
    if timeline_meta.get("captureValidation") != "accepted":
        raise RuntimeError(
            "PPU timeline must come from an accepted timeline.py decode capture"
        )
    if timeline_meta.get("clockScope") != "owner_local":
        raise RuntimeError("PPU timeline clockScope must be owner_local")
    if (
        timeline_meta.get("timerSource") != "globaltimer"
        or timeline_meta.get("timerUnit") != "ns"
        or timeline_meta.get("timerContractValidation") != "accepted"
    ):
        raise RuntimeError("PPU timeline requires the accepted globaltimer ns contract")
    pm_rows = _read_pm(pm_path)
    acu = _read_acu_raw(acu_raw_path)
    perturbation = (
        _read_measurement(perturbation_path, "A/B perturbation")
        if perturbation_path is not None
        else None
    )
    density_sensitivity = (
        _read_measurement(density_sensitivity_path, "B/C density sensitivity")
        if density_sensitivity_path is not None
        else None
    )

    observed_intervals = sorted({row["interval_ns"] for row in pm_rows})
    reported_interval = acu["pm_interval_ns"]
    acu["pm_interval_ns"] = max(observed_intervals)
    acu["pm_interval_source"] = (
        "acu_raw_pmsampler_column"
        if reported_interval is not None
        else "exact_pm_sample_windows"
    )
    if reported_interval is not None and reported_interval != acu["pm_interval_ns"]:
        raise RuntimeError(
            "ACU raw PM interval disagrees with the sample-window maximum"
        )

    errors: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []
    for label, measurement in (
        ("A/B perturbation", perturbation),
        ("B/C density sensitivity", density_sensitivity),
    ):
        if measurement is None:
            notes.append(f"{label} measurement was not supplied")
            continue
        if measurement.get("workload_identity") != timeline_meta.get(
            "workloadIdentity"
        ):
            errors.append(f"{label} workload identity mismatch")
        if measurement.get("device_identity") != timeline_meta.get("deviceIdentity"):
            errors.append(f"{label} device identity mismatch")
    if timeline_meta.get("kernelName") != acu["kernel_name"]:
        errors.append("kernel identity mismatch")
    if timeline_meta.get("grid") != acu["grid"]:
        errors.append("grid mismatch")
    if timeline_meta.get("blockDims") != acu["block"]:
        errors.append("block dimensions mismatch")
    if acu["dropped_samples"] is None:
        notes.append(
            "ACU omitted dropped-sample aggregate; window continuity was validated"
        )
    elif acu["dropped_samples"] != 0:
        errors.append(f"ACU dropped {acu['dropped_samples']} PM samples")

    timeline_duration = float(timeline_meta["kernelDurationNs"])
    duration_delta = abs(timeline_duration - acu["duration_ns"]) / acu["duration_ns"]
    if duration_delta > REJECT_DELTA:
        message = f"timeline/ACU duration delta {duration_delta:.2%} exceeds 5%"
        if allow_duration_mismatch:
            warnings.append(message + " (explicitly allowed)")
        else:
            errors.append(message)
    elif duration_delta > WARNING_DELTA:
        warnings.append(f"timeline/ACU duration delta {duration_delta:.2%} exceeds 3%")

    by_metric: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pm_rows:
        by_metric[row["metric_name"]].append(row)
    _validate_pm_windows(by_metric, errors)

    if len(observed_intervals) != 1:
        jitter = (
            max(observed_intervals) - min(observed_intervals)
        ) / statistics.median(observed_intervals)
        message = f"PM intervals are {observed_intervals} (jitter {jitter:.3%})"
        (warnings if jitter > 0.01 else notes).append(message)
    coverage_end_ns = max(row["window_end_ns"] for row in pm_rows)
    final_gap_ns = acu["duration_ns"] - coverage_end_ns
    if final_gap_ns > 0:
        message = f"final partial interval is not sampled ({final_gap_ns:g} ns)"
        (warnings if final_gap_ns > acu["pm_interval_ns"] else notes).append(message)

    analysis_owner = timeline_meta.get("analysisOwner")
    analysis_window_site_id = timeline_meta.get("analysisWindowSiteId")
    analysis_site_ids = timeline_meta.get("analysisSiteIds")
    if not isinstance(analysis_owner, int) or isinstance(analysis_owner, bool):
        raise RuntimeError("joint analysis requires an explicit analysisOwner")
    if not isinstance(analysis_window_site_id, int) or isinstance(
        analysis_window_site_id, bool
    ):
        raise RuntimeError("joint analysis requires an explicit analysisWindowSiteId")
    if (
        not isinstance(analysis_site_ids, list)
        or not analysis_site_ids
        or not all(
            isinstance(site_id, int) and not isinstance(site_id, bool)
            for site_id in analysis_site_ids
        )
    ):
        raise RuntimeError("joint analysis requires non-empty analysisSiteIds")

    analysis_windows = []
    detail_events = []
    coverage_meta = timeline_meta.get("coverage", {})
    coverage_all_blocks = coverage_meta.get("allBlocks") is True
    coverage_range_site_id = coverage_meta.get("rangeSiteId")
    coverage_events = []
    for event in timeline.get("traceEvents", []):
        interval = _event_interval_ns(event)
        if interval is None:
            continue
        args = event.get("args", {})
        owner = args.get("owner")
        site_id = args.get("site_id")
        if owner == analysis_owner and site_id == analysis_window_site_id:
            analysis_windows.append((event, interval))
        elif owner == analysis_owner and site_id in analysis_site_ids:
            detail_events.append((event, interval))
        if coverage_range_site_id is not None and site_id == coverage_range_site_id:
            coverage_events.append((event, interval))

    grid_blocks = math.prod(acu["grid"])
    if len(analysis_windows) != 1:
        errors.append(
            "expected exactly one range for the declared analysis owner/window, "
            f"found {len(analysis_windows)}"
        )
    if not detail_events:
        errors.append("declared analysis sites produced no ranges")
    if len(analysis_windows) != 1:
        dispatch_upper_ns = timeline_duration
    else:
        window_start_ns, window_end_ns = analysis_windows[0][1]
        dispatch_upper_ns = timeline_duration - window_end_ns
        if dispatch_upper_ns < 0:
            errors.append("analysis window ends after the HGGC kernel duration")
            dispatch_upper_ns = 0
        else:
            notes.append(
                f"analysis-owner origin offset is [0, {dispatch_upper_ns:g}] ns"
            )
        for event, interval in detail_events:
            if interval[0] < window_start_ns or interval[1] > window_end_ns:
                errors.append(
                    f"analysis range {event['name']!r} is outside the analysis window"
                )

    one_wave_capacity = int(acu["cu_count"] * acu["occupancy_blocks_per_cu"])
    coverage_distribution_valid = False
    if coverage_all_blocks:
        coverage_counts: dict[int, int] = defaultdict(int)
        for event, _ in coverage_events:
            block_id = event.get("args", {}).get("block")
            if isinstance(block_id, int) and not isinstance(block_id, bool):
                coverage_counts[block_id] += 1
        expected_counts = {block_id: 1 for block_id in range(grid_blocks)}
        if coverage_counts != expected_counts:
            errors.append(
                "declared all-block coverage does not contain exactly one range per block"
            )
        elif grid_blocks > one_wave_capacity:
            warnings.append(
                "grid exceeds one-wave capacity; normalized coverage-duration "
                "survival is omitted"
            )
        else:
            coverage_distribution_valid = True
    else:
        notes.append(
            "capture declares partial block coverage; normalized all-block duration "
            "survival is omitted"
        )
    if coverage_all_blocks and coverage_range_site_id is None:
        errors.append("declared all-block coverage is missing rangeSiteId")
    if coverage_all_blocks and not coverage_events:
        warnings.append(
            "declared all-block coverage produced no comparable coverage ranges"
        )

    joined_rows = []
    for row in pm_rows:
        start = row["window_start_ns"]
        end = row["window_end_ns"]
        normalized_survival = ""
        if coverage_distribution_valid:
            normalized_survival = sum(
                _overlaps(start, end, 0, interval[1] - interval[0])
                for _, interval in coverage_events
            )
        possible = sorted(
            {
                event["name"]
                for event, interval in detail_events
                if _overlaps(start, end, interval[0], interval[1] + dispatch_upper_ns)
            }
        )
        guaranteed = sorted(
            {
                event["name"]
                for event, interval in detail_events
                if interval[0] + dispatch_upper_ns < end and start < interval[1]
            }
        )
        joined = dict(row)
        joined["normalized_coverage_duration_survival_count"] = normalized_survival
        joined["possible_overlapping_sampled_ranges"] = ";".join(possible)
        joined["guaranteed_overlapping_sampled_ranges"] = ";".join(guaranteed)
        joined["analysis_owner_origin_offset_upper_ns"] = dispatch_upper_ns
        joined["join_semantics"] = (
            "bounded_dispatch_offset;possible_and_guaranteed_overlap;"
            "device_global_metrics_not_ownership"
        )
        joined_rows.append(joined)

    joint_events = list(timeline.get("traceEvents", []))
    joint_events.extend(
        [
            {
                "name": "process_name",
                "ph": "M",
                "pid": 8,
                "args": {"name": "ACU PM Sampling (device/global)"},
            },
            {
                "name": "process_name",
                "ph": "M",
                "pid": 9,
                "args": {"name": "Selected-range placement envelopes"},
            },
        ]
    )
    for event, interval in detail_events:
        joint_events.append(
            {
                "name": event["name"] + " [possible placement]",
                "cat": "dispatch-offset uncertainty",
                "ph": "X",
                "ts": interval[0] / 1000.0,
                "dur": (interval[1] - interval[0] + dispatch_upper_ns) / 1000.0,
                "pid": 9,
                "tid": 1,
                "args": {
                    "dispatch_offset_lower_ns": 0,
                    "dispatch_offset_upper_ns": dispatch_upper_ns,
                    "semantics": "placement envelope, not activity duration",
                },
            }
        )

    metric_tids = {name: index for index, name in enumerate(sorted(by_metric), 1)}
    for row in joined_rows:
        tid = metric_tids[row["metric_name"]]
        start_us = row["window_start_ns"] / 1000.0
        duration_us = row["interval_ns"] / 1000.0
        joint_events.append(
            {
                "name": row["metric_name"],
                "cat": "ACU PM window",
                "ph": "X",
                "ts": start_us,
                "dur": duration_us,
                "pid": 8,
                "tid": tid,
                "args": {
                    "value": row["metric_value"],
                    "unit": row["metric_unit"],
                    "logical_metric_group": row["logical_metric_group"],
                    "acu_packet_index": row["packet_index"],
                    "scope": row["scope"],
                    "validity": row["validity"],
                    "possible_ranges": row["possible_overlapping_sampled_ranges"],
                    "guaranteed_ranges": row["guaranteed_overlapping_sampled_ranges"],
                    "join_semantics": row["join_semantics"],
                },
            }
        )
        joint_events.append(
            {
                "name": row["metric_name"],
                "cat": "ACU PM counter",
                "ph": "C",
                "ts": start_us + duration_us,
                "pid": 8,
                "tid": tid,
                "args": {"value": row["metric_value"]},
            }
        )

    status = "rejected" if errors else ("warning" if warnings else "accepted")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    joined_csv = Path(f"{output_prefix}.joint_samples.csv")
    with joined_csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(joined_rows[0]))
        writer.writeheader()
        writer.writerows(joined_rows)

    trace = {
        "displayTimeUnit": "ns",
        "traceEvents": joint_events,
        "ppuJointProfile": {
            "schemaVersion": 2,
            "timerSource": timeline_meta["timerSource"],
            "timerUnit": timeline_meta["timerUnit"],
            "alignment": (
                "ACU is kernel-start-relative; analysis ranges are owner-origin-"
                "relative with a bounded origin offset; no scaling"
            ),
            "pmWindowSemantics": "[window_start_ns, window_end_ns)",
            "attributionRule": "device/global overlap is not range ownership",
            "validationStatus": status,
        },
    }
    trace_path = Path(f"{output_prefix}.perfetto.json")
    trace_path.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")

    summary = {
        "schema_version": 2,
        "validation": {
            "status": status,
            "errors": errors,
            "warnings": warnings,
            "notes": notes,
            "duration": {
                "timeline_ns": timeline_duration,
                "acu_ns": acu["duration_ns"],
                "relative_delta": duration_delta,
            },
            "identity": {
                "kernel_name": acu["kernel_name"],
                "grid": acu["grid"],
                "block": acu["block"],
            },
        },
        "acu_launch": acu,
        "pm": {
            "metrics": _metric_summary(pm_rows),
            "packet_metric_membership": {
                str(packet): sorted(
                    {
                        row["metric_name"]
                        for row in pm_rows
                        if row["packet_index"] == packet
                    }
                )
                for packet in sorted({row["packet_index"] for row in pm_rows})
            },
            "coverage_end_ns": coverage_end_ns,
            "window_continuity_validated": not any(
                "sample indices" in error or "PM windows" in error for error in errors
            ),
        },
        "timeline": {
            "timer": {
                "source": timeline_meta["timerSource"],
                "unit": timeline_meta["timerUnit"],
                "contract_validation": timeline_meta["timerContractValidation"],
            },
            "capture_mode": timeline_meta.get("captureMode"),
            "sampling_rationale": timeline_meta.get("samplingRationale"),
            "owner_count": len(timeline_meta.get("owners", [])),
            "coverage": coverage_meta,
            "coverage_range_count": len(coverage_events),
            "one_wave_capacity": one_wave_capacity,
            "normalized_coverage_duration_distribution_valid": (
                coverage_distribution_valid
            ),
            "detail_ranges": [event["name"] for event, _ in detail_events],
            "analysis_owner": analysis_owner,
            "analysis_block": timeline_meta.get("analysisBlock"),
            "analysis_thread": timeline_meta.get("analysisThread"),
            "analysis_tile": timeline_meta.get("tile"),
            "analysis_k_stage": timeline_meta.get("kStage"),
            "analysis_window_site_id": analysis_window_site_id,
            "analysis_owner_origin_offset_bounds_ns": [0, dispatch_upper_ns],
        },
        "probe_effect": {
            "a_b_end_to_end_perturbation": (
                perturbation["summary"] if perturbation is not None else None
            ),
            "b_c_density_sensitivity": (
                density_sensitivity["summary"]
                if density_sensitivity is not None
                else None
            ),
        },
        "interpretation_limits": [
            "PM values are device/global aggregates, not per CTA or range.",
            "Different replay packets are not proof of simultaneity.",
            "A PM window cannot resolve a much shorter K-stage.",
            "Cross-domain PPU globaltimer starts are not compared.",
            "Analysis-range launch placement is a bounded uncertainty envelope.",
            "All-block duration survival exists only for explicit complete, comparable coverage.",
        ],
    }
    summary_path = Path(f"{output_prefix}.summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "inputs": {
            "timeline": str(timeline_path.resolve()),
            "pm_samples": str(pm_path.resolve()),
            "acu_raw": str(acu_raw_path.resolve()),
            "a_b_perturbation": (
                str(perturbation_path.resolve())
                if perturbation_path is not None
                else None
            ),
            "b_c_density_sensitivity": (
                str(density_sensitivity_path.resolve())
                if density_sensitivity_path is not None
                else None
            ),
        },
        "outputs": {
            "joint_samples": str(joined_csv),
            "perfetto": str(trace_path),
            "summary": str(summary_path),
        },
        "equivalence_contract": {
            "duration_warning_threshold": WARNING_DELTA,
            "duration_reject_threshold": REJECT_DELTA,
            "minimum_samples_per_metric": MIN_SAMPLES,
            "no_duration_rescaling": True,
        },
    }
    Path(f"{output_prefix}.run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if errors:
        raise RuntimeError("joint profile rejected: " + "; ".join(errors))
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--pm-csv", type=Path, required=True)
    parser.add_argument("--acu-raw-csv", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--perturbation", type=Path)
    parser.add_argument("--density-sensitivity", type=Path)
    parser.add_argument("--allow-duration-mismatch", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = merge(
        args.timeline,
        args.pm_csv,
        args.acu_raw_csv,
        args.output_prefix,
        perturbation_path=args.perturbation,
        density_sensitivity_path=args.density_sensitivity,
        allow_duration_mismatch=args.allow_duration_mismatch,
    )
    print(
        f"joint profile {summary['validation']['status']}; duration delta "
        f"{summary['validation']['duration']['relative_delta']:.2%}"
    )
    return 0


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()
