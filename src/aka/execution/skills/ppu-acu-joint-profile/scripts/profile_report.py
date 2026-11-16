#!/usr/bin/env python3
"""Build hash-bound PPU profile comparisons, calibrations, and envelopes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

try:
    from .evidence import (
        CALIBRATION_SCHEMA,
        COMPARISON_SCHEMA,
        ENVELOPE_SCHEMA,
        EvidenceError,
        binding_view,
        descriptor_paths,
        digest_json,
        ensure_no_output_aliases,
        load_json,
        portable_descriptors,
        require,
        require_sha256,
        sha256_file,
        validate_acu_metadata,
        validate_timeline_receipt,
        verify_descriptor,
        verify_descriptor_tree,
    )
except ImportError:
    from evidence import (
        CALIBRATION_SCHEMA,
        COMPARISON_SCHEMA,
        ENVELOPE_SCHEMA,
        EvidenceError,
        binding_view,
        descriptor_paths,
        digest_json,
        ensure_no_output_aliases,
        load_json,
        portable_descriptors,
        require,
        require_sha256,
        sha256_file,
        validate_acu_metadata,
        validate_timeline_receipt,
        verify_descriptor,
        verify_descriptor_tree,
    )


CALIBRATION_SPEC_SCHEMA = "ppu-calibration-spec/v1"
IDENTITY_FIELDS = (
    "device_identity",
    "runtime_identity",
    "cache_policy",
    "clock_configuration",
)
DIRECTIONS = {"higher_is_better", "lower_is_better"}
ENVELOPE_KINDS = {"bandwidth", "compute", "launch_merge", "random_gather"}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _require_distinct_output(output: Path, *inputs: Path) -> None:
    ensure_no_output_aliases([output], inputs, "profile report")


def _require_not_transitive_output(
    output: Path, *documents: tuple[Path, dict[str, Any]]
) -> None:
    transitive = {
        path
        for owner_path, document in documents
        for path in descriptor_paths(owner_path, document)
    }
    ensure_no_output_aliases([output], transitive, "profile report transitive")


def _descriptor(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"artifact is not a regular file: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _number(
    value: object,
    label: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value),
        f"{label} must be finite",
    )
    result = float(value)
    if positive:
        require(result > 0, f"{label} must be positive")
    elif nonnegative:
        require(result >= 0, f"{label} must be non-negative")
    return result


def _same_identity(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    shared: dict[str, Any] = {}
    for field in ("kernel_name", "workload_identity", *IDENTITY_FIELDS):
        require(left.get(field) == right.get(field), f"ACU comparison identity drift: {field}")
        shared[field] = left[field]
    return shared


def _delta(candidate: float, incumbent: float) -> dict[str, float]:
    return {
        "incumbent": incumbent,
        "candidate": candidate,
        "absolute_delta": candidate - incumbent,
        "relative_delta": candidate / incumbent - 1.0,
    }


def _metric_deltas(
    incumbent: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    incumbent_metrics = incumbent.get("metric_summaries")
    candidate_metrics = candidate.get("metric_summaries")
    require(isinstance(incumbent_metrics, dict), "incumbent metric summaries are missing")
    require(isinstance(candidate_metrics, dict), "candidate metric summaries are missing")
    require(
        set(incumbent_metrics) == set(candidate_metrics),
        "ACU comparison requires identical packet/metric streams",
    )
    result: dict[str, dict[str, Any]] = {}
    for name in sorted(incumbent_metrics):
        before = incumbent_metrics[name]
        after = candidate_metrics[name]
        require(isinstance(before, dict) and isinstance(after, dict), f"metric {name} is invalid")
        for field in ("metric_name", "logical_metric_group", "scope", "unit"):
            require(before.get(field) == after.get(field), f"metric {name} drift: {field}")
        incumbent_value = before.get("time_weighted_mean")
        candidate_value = after.get("time_weighted_mean")
        if incumbent_value is None or candidate_value is None:
            result[name] = {
                "metric_name": before["metric_name"],
                "unit": before["unit"],
                "incumbent": incumbent_value,
                "candidate": candidate_value,
                "comparable": False,
            }
            continue
        incumbent_number = _number(
            incumbent_value, f"{name}.incumbent", nonnegative=True
        )
        candidate_number = _number(
            candidate_value, f"{name}.candidate", nonnegative=True
        )
        result[name] = {
            "metric_name": before["metric_name"],
            "logical_metric_group": before["logical_metric_group"],
            "scope": before["scope"],
            "unit": before["unit"],
            "comparable": True,
            "incumbent": incumbent_number,
            "candidate": candidate_number,
            "absolute_delta": candidate_number - incumbent_number,
            "relative_delta": (
                candidate_number / incumbent_number - 1.0
                if incumbent_number
                else None
            ),
        }
    return result


def _descriptor_target(owner_path: Path, descriptor: dict[str, Any]) -> Path:
    target = Path(descriptor["path"])
    return target if target.is_absolute() else owner_path.parent / target


def _comparison_sections(
    incumbent: dict[str, Any], candidate: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    identity = _same_identity(incumbent["identity"], candidate["identity"])
    require(incumbent["producer"] == candidate["producer"], "ACU producer drift")
    incumbent_duration = _number(
        incumbent["launch"]["duration_ns"], "incumbent duration", positive=True
    )
    candidate_duration = _number(
        candidate["launch"]["duration_ns"], "candidate duration", positive=True
    )
    launch = {
        "duration_ns": {
            **_delta(candidate_duration, incumbent_duration),
            "speedup": incumbent_duration / candidate_duration,
        }
    }
    for field in (
        "occupancy_blocks_per_cu",
        "registers_per_thread",
        "shared_mem_per_block",
    ):
        before = _number(incumbent["launch"][field], f"incumbent {field}")
        after = _number(candidate["launch"][field], f"candidate {field}")
        launch[field] = (
            {
                "incumbent": before,
                "candidate": after,
                "absolute_delta": after - before,
                "relative_delta": None,
            }
            if before == 0
            else _delta(after, before)
        )
    return identity, launch, _metric_deltas(incumbent, candidate)


def comparison_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "kernel_sha256": report.get("kernel_sha256"),
        "identity": report.get("identity"),
        "incumbent": binding_view(report.get("incumbent")),
        "candidate": binding_view(report.get("candidate")),
        "launch": report.get("launch"),
        "metrics": report.get("metrics"),
    }


def validate_comparison(path: Path) -> dict[str, Any]:
    report = load_json(path, "ACU comparison")
    require(report.get("schema") == COMPARISON_SCHEMA, f"comparison must use {COMPARISON_SCHEMA}")
    require(report.get("validation") == "accepted", "ACU comparison is not accepted")
    require(report.get("evidence_grade") == "decision", "ACU comparison is not decision-grade")
    source_metadata: dict[str, dict[str, Any]] = {}
    for name in ("incumbent", "candidate"):
        row = report.get(name)
        require(isinstance(row, dict), f"comparison.{name} is missing")
        artifact = row.get("artifact")
        verify_descriptor(path, artifact, f"comparison.{name}.artifact")
        require(isinstance(artifact, dict), f"comparison.{name}.artifact is missing")
        metadata = validate_acu_metadata(
            _descriptor_target(path, artifact), require_decision=True
        )
        require(
            row.get("evidence_id") == metadata.get("evidence_id"),
            f"comparison.{name} evidence id mismatch",
        )
        require(
            row.get("kernel_sha256") == metadata.get("kernel_sha256"),
            f"comparison.{name} kernel hash mismatch",
        )
        require_sha256(row.get("kernel_sha256"), f"comparison.{name}.kernel_sha256")
        source_metadata[name] = metadata
    identity, launch, metrics = _comparison_sections(
        source_metadata["incumbent"], source_metadata["candidate"]
    )
    require(
        report.get("kernel_sha256")
        == source_metadata["candidate"].get("kernel_sha256"),
        "ACU comparison root kernel hash mismatch",
    )
    require(report.get("identity") == identity, "ACU comparison identity was not recomputed")
    require(report.get("launch") == launch, "ACU comparison launch deltas were not recomputed")
    require(report.get("metrics") == metrics, "ACU comparison metric deltas were not recomputed")
    expected_payload = comparison_payload(report)
    require(
        report.get("binding_payload") == expected_payload,
        "ACU comparison binding payload mismatch",
    )
    require(
        digest_json(expected_payload) == report.get("evidence_id"),
        "ACU comparison evidence id is invalid",
    )
    return report


def compare_acu(incumbent_path: Path, candidate_path: Path, output: Path) -> dict[str, Any]:
    _require_distinct_output(output, incumbent_path, candidate_path)
    incumbent = validate_acu_metadata(incumbent_path, require_decision=True)
    candidate = validate_acu_metadata(candidate_path, require_decision=True)
    _require_not_transitive_output(
        output,
        (incumbent_path, incumbent),
        (candidate_path, candidate),
    )
    identity, launch, metrics = _comparison_sections(incumbent, candidate)

    report: dict[str, Any] = {
        "schema": COMPARISON_SCHEMA,
        "validation": "accepted",
        "evidence_grade": "decision",
        "kernel_sha256": candidate["kernel_sha256"],
        "identity": identity,
        "incumbent": {
            "artifact": portable_descriptors(_descriptor(incumbent_path), output),
            "evidence_id": incumbent["evidence_id"],
            "kernel_sha256": incumbent["kernel_sha256"],
            "kernel_specialization": incumbent["identity"]["kernel_specialization"],
        },
        "candidate": {
            "artifact": portable_descriptors(_descriptor(candidate_path), output),
            "evidence_id": candidate["evidence_id"],
            "kernel_sha256": candidate["kernel_sha256"],
            "kernel_specialization": candidate["identity"]["kernel_specialization"],
        },
        "launch": launch,
        "metrics": metrics,
    }
    report["binding_payload"] = comparison_payload(report)
    report["evidence_id"] = digest_json(report["binding_payload"])
    _write_json(output, report)
    validate_comparison(output)
    return report


def calibration_payload(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        "identity": receipt.get("identity"),
        "measurements": receipt.get("measurements"),
        "source_artifacts": binding_view(receipt.get("source_artifacts")),
    }


def validate_calibration(path: Path) -> dict[str, Any]:
    receipt = load_json(path, "PPU calibration")
    require(
        receipt.get("schema") == CALIBRATION_SCHEMA,
        f"calibration must use {CALIBRATION_SCHEMA}",
    )
    require(receipt.get("validation") == "accepted", "PPU calibration is not accepted")
    require(
        receipt.get("evidence_grade") == "decision",
        "PPU calibration is not decision-grade",
    )
    identity = receipt.get("identity")
    require(isinstance(identity, dict), "calibration identity is missing")
    for field in IDENTITY_FIELDS:
        require(field in identity, f"calibration identity lacks {field}")
    measurements = receipt.get("measurements")
    require(
        isinstance(measurements, dict) and measurements,
        "calibration measurements are missing",
    )
    sources = receipt.get("source_artifacts")
    require(
        isinstance(sources, list) and sources,
        "calibration source artifacts are required",
    )
    verify_descriptor_tree(path, sources, "source_artifacts")
    for name, measurement in measurements.items():
        require(isinstance(name, str) and name, "calibration measurement name is invalid")
        require(isinstance(measurement, dict), f"calibration measurement {name} is invalid")
        value = _number(
            measurement.get("value"), f"calibration {name}.value", positive=True
        )
        require(
            measurement.get("kind") in ENVELOPE_KINDS,
            f"calibration {name}.kind is invalid",
        )
        require(
            isinstance(measurement.get("unit"), str) and measurement["unit"],
            f"calibration {name}.unit is required",
        )
        require(
            measurement.get("direction") in DIRECTIONS,
            f"calibration {name}.direction is invalid",
        )
        source_index = measurement.get("source_artifact")
        pointer = measurement.get("json_pointer")
        require(
            isinstance(source_index, int)
            and not isinstance(source_index, bool)
            and 0 <= source_index < len(sources),
            f"calibration {name}.source_artifact is invalid",
        )
        require(
            isinstance(pointer, str) and pointer.startswith("/"),
            f"calibration {name}.json_pointer is invalid",
        )
        source = sources[source_index]
        require(isinstance(source, dict), f"calibration source {source_index} is invalid")
        source_document = load_json(
            _descriptor_target(path, source), f"calibration source {source_index}"
        )
        require(
            value == _number(
                _json_pointer(source_document, pointer),
                f"calibration source {name}",
                positive=True,
            ),
            f"calibration {name}.value does not match its source artifact",
        )
    expected_payload = calibration_payload(receipt)
    require(
        receipt.get("binding_payload") == expected_payload,
        "calibration binding payload mismatch",
    )
    require(
        digest_json(expected_payload) == receipt.get("evidence_id"),
        "calibration evidence id is invalid",
    )
    return receipt


def seal_calibration(spec_path: Path, output: Path) -> dict[str, Any]:
    _require_distinct_output(output, spec_path)
    spec = load_json(spec_path, "PPU calibration spec")
    require(
        spec.get("schema") == CALIBRATION_SPEC_SCHEMA,
        f"calibration spec must use {CALIBRATION_SPEC_SCHEMA}",
    )
    identity = spec.get("identity")
    require(isinstance(identity, dict), "calibration identity is missing")
    for field in IDENTITY_FIELDS:
        require(field in identity, f"calibration identity lacks {field}")
    measurements = spec.get("measurements")
    require(
        isinstance(measurements, dict) and measurements,
        "calibration measurements are required",
    )
    sources = spec.get("source_artifacts")
    require(
        isinstance(sources, list) and sources,
        "calibration source artifacts are required",
    )
    source_descriptors = []
    source_paths: list[Path] = []
    for index, source in enumerate(sources):
        require(isinstance(source, dict), f"source_artifacts[{index}] must be an object")
        source_path = source.get("path")
        identity_text = source.get("identity")
        require(
            isinstance(source_path, str) and source_path,
            f"source_artifacts[{index}].path is required",
        )
        require(
            isinstance(identity_text, str) and identity_text,
            f"source_artifacts[{index}].identity is required",
        )
        path = Path(source_path)
        if not path.is_absolute():
            path = spec_path.parent / path
        _require_distinct_output(output, path)
        source_paths.append(path)
        source_descriptors.append(
            portable_descriptors(
                {**_descriptor(path), "identity": identity_text}, output
            )
        )
    sealed_measurements: dict[str, dict[str, Any]] = {}
    for name, measurement in measurements.items():
        require(isinstance(name, str) and name, "calibration measurement name is invalid")
        require(isinstance(measurement, dict), f"calibration measurement {name} is invalid")
        source_index = measurement.get("source_artifact")
        pointer = measurement.get("json_pointer")
        require(
            isinstance(source_index, int)
            and not isinstance(source_index, bool)
            and 0 <= source_index < len(source_paths),
            f"calibration {name}.source_artifact is invalid",
        )
        require(
            isinstance(pointer, str) and pointer.startswith("/"),
            f"calibration {name}.json_pointer is invalid",
        )
        require(
            measurement.get("kind") in ENVELOPE_KINDS,
            f"calibration {name}.kind is invalid",
        )
        require(
            isinstance(measurement.get("unit"), str) and measurement["unit"],
            f"calibration {name}.unit is required",
        )
        require(
            measurement.get("direction") in DIRECTIONS,
            f"calibration {name}.direction is invalid",
        )
        source_document = load_json(
            source_paths[source_index], f"calibration source {source_index}"
        )
        sealed_measurements[name] = {
            "kind": measurement["kind"],
            "value": _number(
                _json_pointer(source_document, pointer),
                f"calibration source {name}",
                positive=True,
            ),
            "unit": measurement["unit"],
            "direction": measurement["direction"],
            "source_artifact": source_index,
            "json_pointer": pointer,
        }
    receipt: dict[str, Any] = {
        "schema": CALIBRATION_SCHEMA,
        "validation": "accepted",
        "evidence_grade": "decision",
        "identity": identity,
        "measurements": sealed_measurements,
        "source_artifacts": source_descriptors,
    }
    receipt["binding_payload"] = calibration_payload(receipt)
    receipt["evidence_id"] = digest_json(receipt["binding_payload"])
    _write_json(output, receipt)
    validate_calibration(output)
    return receipt


def _pointer_parts(pointer: str) -> list[str]:
    require(pointer.startswith("/"), "current pointer must be an RFC 6901 JSON pointer")
    return [part.replace("~1", "/").replace("~0", "~") for part in pointer.split("/")[1:]]


def _json_pointer(document: object, pointer: str) -> object:
    value = document
    for key in _pointer_parts(pointer):
        require(
            isinstance(value, dict) and key in value,
            f"current pointer does not resolve: {pointer}",
        )
        value = value[key]
    return value


def _current_measurement(
    metadata: dict[str, Any], pointer: str, kind: str
) -> tuple[float, str]:
    parts = _pointer_parts(pointer)
    if parts == ["launch", "duration_ns"]:
        require(kind == "launch_merge", "kernel duration is only valid for launch_merge envelopes")
        return _number(_json_pointer(metadata, pointer), "current value", positive=True), "ns"
    require(
        len(parts) == 3
        and parts[0] == "metric_summaries"
        and parts[2] == "time_weighted_mean",
        "current pointer must select launch.duration_ns or a metric time_weighted_mean",
    )
    metric = metadata["metric_summaries"].get(parts[1])
    require(isinstance(metric, dict), "current metric summary is missing")
    groups = {
        "compute": {"compute", "tensor"},
        "bandwidth": {"memory"},
        "random_gather": {"ksd", "kvd", "l2"},
    }
    require(kind in groups, f"{kind} envelopes cannot use an ACU metric stream")
    require(
        metric.get("logical_metric_group") in groups[kind],
        f"current metric group is not valid for {kind}",
    )
    unit = metric.get("unit")
    require(isinstance(unit, str) and unit, "current metric unit is missing")
    return (
        _number(_json_pointer(metadata, pointer), "current value", nonnegative=True),
        unit,
    )


def envelope_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "kernel_sha256": report.get("kernel_sha256"),
        "kind": report.get("kind"),
        "direction": report.get("direction"),
        "current_value": report.get("current_value"),
        "measured_bound": report.get("measured_bound"),
        "headroom_pct": report.get("headroom_pct"),
        "unit": report.get("unit"),
        "identity": report.get("identity"),
        "inputs": binding_view(report.get("inputs")),
        "selectors": report.get("selectors"),
    }


def validate_envelope(path: Path) -> dict[str, Any]:
    report = load_json(path, "PPU envelope")
    require(report.get("schema") == ENVELOPE_SCHEMA, f"envelope must use {ENVELOPE_SCHEMA}")
    require(report.get("validation") == "accepted", "PPU envelope is not accepted")
    require(report.get("evidence_grade") == "decision", "PPU envelope is not decision-grade")
    require(report.get("kind") in ENVELOPE_KINDS, "PPU envelope kind is invalid")
    inputs = report.get("inputs")
    selectors = report.get("selectors")
    require(isinstance(inputs, dict), "PPU envelope inputs are missing")
    require(isinstance(selectors, dict), "PPU envelope selectors are missing")
    verify_descriptor_tree(path, inputs, "inputs")
    current_descriptor = inputs.get("current")
    calibration_descriptor = inputs.get("calibration")
    require(isinstance(current_descriptor, dict), "PPU envelope current input is missing")
    require(isinstance(calibration_descriptor, dict), "PPU envelope calibration input is missing")
    current = validate_acu_metadata(
        _descriptor_target(path, current_descriptor), require_decision=True
    )
    calibration = validate_calibration(_descriptor_target(path, calibration_descriptor))
    expected_identity = {
        field: current["identity"][field] for field in IDENTITY_FIELDS
    }
    for field in IDENTITY_FIELDS:
        require(
            expected_identity[field] == calibration["identity"].get(field),
            f"envelope identity drift: {field}",
        )
    require(
        report.get("identity") == expected_identity,
        "envelope report identity does not match its sources",
    )
    pointer = selectors.get("current_pointer")
    bound_name = selectors.get("calibration_measurement")
    require(isinstance(pointer, str), "envelope current pointer is missing")
    require(isinstance(bound_name, str), "envelope calibration selector is missing")
    measurement = calibration["measurements"].get(bound_name)
    require(
        isinstance(measurement, dict),
        f"calibration lacks measurement {bound_name}",
    )
    require(
        measurement.get("kind") == report.get("kind"),
        "envelope kind does not match calibration",
    )
    current_value, current_unit = _current_measurement(
        current, pointer, report["kind"]
    )
    measured_bound = _number(measurement.get("value"), "measured bound", positive=True)
    require(measurement.get("unit") == current_unit, "envelope units do not match")
    direction = measurement.get("direction")
    if direction == "higher_is_better":
        require(current_value <= measured_bound, "current value exceeds measured upper bound")
        headroom_pct = (measured_bound - current_value) / measured_bound * 100.0
    else:
        require(direction == "lower_is_better", "calibration direction is invalid")
        require(current_value >= measured_bound, "current value is below measured lower bound")
        headroom_pct = (current_value - measured_bound) / current_value * 100.0
    require(
        report.get("kernel_sha256") == current.get("kernel_sha256"),
        "envelope kernel hash mismatch",
    )
    require(
        report.get("current_value") == current_value,
        "envelope current value was not recomputed",
    )
    require(
        report.get("measured_bound") == measured_bound,
        "envelope bound was not recomputed",
    )
    require(report.get("direction") == direction, "envelope direction mismatch")
    require(report.get("unit") == measurement.get("unit"), "envelope unit mismatch")
    require(report.get("headroom_pct") == headroom_pct, "envelope headroom was not recomputed")
    expected_payload = envelope_payload(report)
    require(
        report.get("binding_payload") == expected_payload,
        "envelope binding payload mismatch",
    )
    require(
        digest_json(expected_payload) == report.get("evidence_id"),
        "envelope evidence id is invalid",
    )
    return report


def build_envelope(
    current_path: Path,
    calibration_path: Path,
    kind: str,
    current_pointer: str,
    bound_name: str,
    output: Path,
) -> dict[str, Any]:
    _require_distinct_output(output, current_path, calibration_path)
    require(kind in ENVELOPE_KINDS, f"unsupported envelope kind: {kind}")
    current = validate_acu_metadata(current_path, require_decision=True)
    calibration = validate_calibration(calibration_path)
    _require_not_transitive_output(
        output,
        (current_path, current),
        (calibration_path, calibration),
    )
    for field in IDENTITY_FIELDS:
        require(
            current["identity"].get(field) == calibration["identity"].get(field),
            f"envelope identity drift: {field}",
        )
    measurement = calibration["measurements"].get(bound_name)
    require(
        isinstance(measurement, dict),
        f"calibration lacks measurement {bound_name}",
    )
    require(measurement.get("kind") == kind, "envelope kind does not match calibration")
    current_value, current_unit = _current_measurement(
        current, current_pointer, kind
    )
    bound = _number(measurement["value"], "measured bound", positive=True)
    require(measurement.get("unit") == current_unit, "envelope units do not match")
    direction = measurement["direction"]
    if direction == "higher_is_better":
        require(current_value <= bound, "current value exceeds measured upper bound")
        headroom_pct = (bound - current_value) / bound * 100.0
    else:
        require(current_value >= bound, "current value is below measured lower bound")
        headroom_pct = (current_value - bound) / current_value * 100.0
    report: dict[str, Any] = {
        "schema": ENVELOPE_SCHEMA,
        "validation": "accepted",
        "evidence_grade": "decision",
        "kernel_sha256": current["kernel_sha256"],
        "kind": kind,
        "direction": direction,
        "current_value": current_value,
        "measured_bound": bound,
        "headroom_pct": headroom_pct,
        "unit": measurement["unit"],
        "identity": {field: current["identity"][field] for field in IDENTITY_FIELDS},
        "inputs": portable_descriptors(
            {
                "current": _descriptor(current_path),
                "calibration": _descriptor(calibration_path),
            },
            output,
        ),
        "selectors": {
            "current_pointer": current_pointer,
            "calibration_measurement": bound_name,
        },
    }
    report["binding_payload"] = envelope_payload(report)
    report["evidence_id"] = digest_json(report["binding_payload"])
    _write_json(output, report)
    validate_envelope(output)
    return report


def validate_profile_artifact(path: Path) -> dict[str, Any]:
    document = load_json(path, "PPU profile artifact")
    schema = document.get("schema")
    if schema == "ppu-acu-extraction/v4":
        return validate_acu_metadata(path, require_decision=True)
    if schema == COMPARISON_SCHEMA:
        return validate_comparison(path)
    if schema == CALIBRATION_SCHEMA:
        return validate_calibration(path)
    if schema == ENVELOPE_SCHEMA:
        return validate_envelope(path)
    if schema == "ppu-fixed-slot-receipt/v5":
        return validate_timeline_receipt(path, require_decision=True)
    if schema == "ppu-critical-path-report/v3":
        try:
            from .critical_path import validate_report
        except ImportError:
            from critical_path import validate_report

        return validate_report(path, require_decision=True)
    if schema == "ppu-joint-profile/v4":
        try:
            from .merge import validate_joint_summary
        except ImportError:
            from merge import validate_joint_summary

        return validate_joint_summary(path, require_accepted=True)
    raise EvidenceError(f"unsupported PPU profile artifact schema: {schema}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    compare = commands.add_parser("compare-acu", help="Compare accepted ACU receipts")
    compare.add_argument("--incumbent", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)

    calibration = commands.add_parser(
        "seal-calibration", help="Hash-bind a measured PPU calibration"
    )
    calibration.add_argument("--spec", type=Path, required=True)
    calibration.add_argument("--output", type=Path, required=True)

    envelope = commands.add_parser(
        "envelope", help="Compare current evidence with a measured bound"
    )
    envelope.add_argument("--current", type=Path, required=True)
    envelope.add_argument("--calibration", type=Path, required=True)
    envelope.add_argument("--kind", choices=sorted(ENVELOPE_KINDS), required=True)
    envelope.add_argument("--current-pointer", required=True)
    envelope.add_argument("--bound-name", required=True)
    envelope.add_argument("--output", type=Path, required=True)

    validate = commands.add_parser(
        "validate", help="Validate one terminal-reusable PPU profile artifact"
    )
    validate.add_argument("--artifact", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "compare-acu":
            report = compare_acu(args.incumbent, args.candidate, args.output)
            speedup = report["launch"]["duration_ns"]["speedup"]
            print(f"PPU ACU comparison accepted: speedup={speedup:.6f}")
        elif args.command == "seal-calibration":
            report = seal_calibration(args.spec, args.output)
            print(f"PPU calibration accepted: {len(report['measurements'])} bounds")
        elif args.command == "validate":
            report = validate_profile_artifact(args.artifact)
            print(f"PPU profile artifact accepted: {report['schema']}")
        else:
            report = build_envelope(
                args.current,
                args.calibration,
                args.kind,
                args.current_pointer,
                args.bound_name,
                args.output,
            )
            print(
                f"PPU envelope accepted: {report['kind']} "
                f"headroom={report['headroom_pct']:.6f}%"
            )
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
