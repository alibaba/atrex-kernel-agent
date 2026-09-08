#!/usr/bin/env python3
"""Close an agent-declared PPU owner-local critical path across captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import tempfile
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

try:
    from .evidence import (
        binding_view,
        descriptor_paths,
        ensure_no_output_aliases,
        portable_descriptors,
        validate_timeline_receipt,
        verify_descriptor_tree,
    )
except ImportError:
    from evidence import (
        binding_view,
        descriptor_paths,
        ensure_no_output_aliases,
        portable_descriptors,
        validate_timeline_receipt,
        verify_descriptor_tree,
    )


PLAN_SCHEMA = "ppu-critical-path-plan/v2"
CANONICAL_SCHEMA = "ppu-fixed-slot-canonical/v5"
REPORT_SCHEMA = "ppu-critical-path-report/v3"


class CriticalPathError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CriticalPathError(message)


def _load(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CriticalPathError(f"cannot read {label} {path}: {error}") from error
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


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


def _descriptor(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"artifact is not a regular file: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _capture_receipt(path: Path, canonical_path: Path) -> dict[str, Any]:
    try:
        return validate_timeline_receipt(
            path,
            expected_outputs={"canonical": canonical_path},
        )
    except RuntimeError as error:
        raise CriticalPathError(str(error)) from error


def _positive_number(value: object, label: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0,
        f"{label} must be positive and finite",
    )
    return float(value)


def _site(value: object, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    site_id = value.get("site_id")
    name = value.get("name")
    _require(
        isinstance(site_id, int)
        and not isinstance(site_id, bool)
        and 0 <= site_id <= 0xFFFF,
        f"{label}.site_id must fit uint16",
    )
    _require(isinstance(name, str) and name.strip(), f"{label}.name is required")
    return {"site_id": site_id, "name": name}


def _summary(values: list[float]) -> dict[str, float | int]:
    _require(bool(values), "cannot summarize an empty value list")
    return {
        "count": len(values),
        "min_ns": min(values),
        "median_ns": statistics.median(values),
        "mean_ns": statistics.fmean(values),
        "max_ns": max(values),
    }


def _interval_union_ns(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0
    ordered = sorted(intervals)
    start, end = ordered[0]
    total = 0
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def _plan(path: Path) -> dict[str, Any]:
    plan = _load(path, "critical-path plan")
    _require(plan.get("schema") == PLAN_SCHEMA, "unknown critical-path plan schema")
    parent = _site(plan.get("parent"), "parent")
    raw_components = plan.get("components", [])
    _require(
        isinstance(raw_components, list),
        "components must be a list when present",
    )
    components = [
        _site(value, f"components[{index}]")
        for index, value in enumerate(raw_components)
    ]
    component_ids = [component["site_id"] for component in components]
    component_names = [component["name"] for component in components]
    _require(
        len(set(component_ids)) == len(component_ids),
        "component site ids must be unique",
    )
    _require(
        len(set(component_names)) == len(component_names),
        "component names must be unique",
    )
    _require(
        parent["site_id"] not in component_ids,
        "parent site cannot also be a component",
    )

    clean_reference = plan.get("clean_reference")
    clean_samples: list[float] | None = None
    if clean_reference is not None:
        _require(isinstance(clean_reference, dict), "clean_reference must be an object")
        source = clean_reference.get("source")
        _require(
            isinstance(source, str) and source.strip(),
            "clean_reference.source is required",
        )
        identity = clean_reference.get("identity")
        _require(
            isinstance(identity, dict) and identity,
            "clean_reference.identity is required",
        )
        artifact = clean_reference.get("artifact")
        _require(
            isinstance(artifact, dict), "clean_reference.artifact must be an object"
        )
        artifact_path = artifact.get("path")
        artifact_sha256 = artifact.get("sha256")
        _require(
            isinstance(artifact_path, str) and artifact_path.strip(),
            "clean_reference.artifact.path is required",
        )
        resolved_artifact = Path(artifact_path)
        if not resolved_artifact.is_absolute():
            resolved_artifact = path.parent / resolved_artifact
        _require(
            resolved_artifact.is_file(),
            "clean_reference artifact is not a regular file",
        )
        _require(
            isinstance(artifact_sha256, str)
            and artifact_sha256 == _sha256(resolved_artifact),
            "clean_reference artifact hash mismatch",
        )
        measurement = _load(resolved_artifact, "clean-reference measurement")
        _require(
            measurement.get("schema") == "ppu-clean-measurement/v1"
            and measurement.get("validation") == "accepted",
            "clean-reference measurement must be accepted ppu-clean-measurement/v1",
        )
        _require(
            measurement.get("identity") == identity,
            "clean-reference measurement identity mismatch",
        )
        raw_samples = measurement.get("duration_ns_samples")
        _require(
            isinstance(raw_samples, list) and raw_samples,
            "clean-reference measurement samples must be non-empty",
        )
        clean_samples = [
            _positive_number(value, f"clean reference sample {index}")
            for index, value in enumerate(raw_samples)
        ]

    stability = plan.get("stability")
    if stability is not None:
        _require(isinstance(stability, dict), "stability must be an object")
        threshold = stability.get("material_relative_spread")
        _require(
            isinstance(threshold, (int, float))
            and not isinstance(threshold, bool)
            and math.isfinite(threshold)
            and threshold >= 0,
            "stability.material_relative_spread must be non-negative",
        )

    owner_topology = plan.get("owner_topology", "same")
    _require(
        owner_topology in {"same", "declared_variation"},
        "owner_topology must be same or declared_variation",
    )

    return {
        **plan,
        "parent": parent,
        "components": components,
        "clean_samples": clean_samples,
        "owner_topology": owner_topology,
    }


def _canonical(path: Path, receipt_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    capture = _load(path, "canonical capture")
    receipt = _capture_receipt(receipt_path, path)
    _require(capture.get("schema") == CANONICAL_SCHEMA, "unsupported canonical schema")
    _require(capture.get("clock_scope") == "owner_local", "owner-local clock required")
    validation = capture.get("validation")
    _require(
        isinstance(validation, dict)
        and validation.get("capture") == "accepted"
        and validation.get("timer_contract") == "accepted"
        and validation.get("correctness") == "accepted",
        "capture, timer contract, and correctness must be accepted",
    )
    identity = capture.get("identity")
    _require(isinstance(identity, dict) and identity, "capture identity is required")
    for field in (
        "kernel_name",
        "workload_identity",
        "device_identity",
        "runtime_identity",
        "launch_id",
        "grid",
        "block",
        "kernel_specialization",
        "cache_policy",
        "clock_configuration",
    ):
        _require(field in identity, f"capture identity needs {field}")
    timer = capture.get("timer")
    _require(
        isinstance(timer, dict)
        and timer.get("source") == "globaltimer"
        and timer.get("unit") == "ns",
        "canonical timer must be globaltimer in ns",
    )
    owners = capture.get("owners")
    sites = capture.get("sites")
    events = capture.get("events")
    _require(isinstance(owners, list) and owners, "capture owners are required")
    _require(isinstance(sites, list) and sites, "capture sites are required")
    _require(isinstance(events, list), "capture events must be a list")
    evidence = capture.get("evidence")
    _require(
        isinstance(evidence, dict)
        and evidence.get("id") == receipt.get("evidence_id")
        and evidence.get("grade") == receipt.get("evidence_grade"),
        "canonical evidence binding does not match its receipt",
    )
    return capture, receipt


def analyze(
    plan_path: Path,
    captures_with_receipts: list[tuple[Path, Path]],
    output: Path,
    *,
    self_validate: bool = True,
) -> dict[str, Any]:
    input_paths = [plan_path, *(path for pair in captures_with_receipts for path in pair)]
    try:
        ensure_no_output_aliases([output], input_paths, "critical-path")
    except RuntimeError as error:
        raise CriticalPathError(str(error)) from error
    plan = _plan(plan_path)
    clean_reference = plan.get("clean_reference")
    if isinstance(clean_reference, dict):
        artifact = clean_reference.get("artifact")
        if isinstance(artifact, dict) and isinstance(artifact.get("path"), str):
            clean_path = Path(artifact["path"])
            if not clean_path.is_absolute():
                clean_path = plan_path.parent / clean_path
            try:
                ensure_no_output_aliases(
                    [output], [clean_path], "critical-path clean reference"
                )
            except RuntimeError as error:
                raise CriticalPathError(str(error)) from error
    _require(bool(captures_with_receipts), "at least one canonical capture is required")
    captures_with_receipts = sorted(
        captures_with_receipts,
        key=lambda pair: (_sha256(pair[1]), _sha256(pair[0])),
    )
    captures = [
        (
            capture_path,
            receipt_path,
            *_canonical(capture_path, receipt_path),
        )
        for capture_path, receipt_path in captures_with_receipts
    ]

    reference_identity: dict[str, Any] | None = None
    launch_ids: set[int] = set()
    parent_id = plan["parent"]["site_id"]
    components_by_id = {
        component["site_id"]: component for component in plan["components"]
    }
    has_components = bool(components_by_id)
    all_instances: list[dict[str, Any]] = []
    capture_reports: list[dict[str, Any]] = []
    reference_sites: dict[int, dict[str, Any]] | None = None
    reference_owners: list[dict[str, Any]] | None = None
    receipt_ids: list[str] = []
    evidence_grades: list[str] = []
    kernel_sha256: str | None = None
    kernel_identity_initialized = False

    for capture_path, receipt_path, capture, receipt in captures:
        try:
            ensure_no_output_aliases(
                [output],
                descriptor_paths(receipt_path, receipt),
                "critical-path transitive",
            )
        except RuntimeError as error:
            raise CriticalPathError(str(error)) from error
        receipt_ids.append(receipt["evidence_id"])
        evidence_grades.append(receipt["evidence_grade"])
        capture_kernel_sha256 = receipt.get("kernel_sha256")
        _require(
            capture.get("kernel_sha256") == capture_kernel_sha256,
            f"canonical authoritative kernel mismatch in {capture_path}",
        )
        if not kernel_identity_initialized:
            kernel_sha256 = capture_kernel_sha256
            kernel_identity_initialized = True
        _require(
            capture_kernel_sha256 == kernel_sha256,
            f"authoritative kernel drifted in {capture_path}",
        )
        identity = capture["identity"]
        comparable_identity = {
            key: identity[key]
            for key in (
                "kernel_name",
                "workload_identity",
                "device_identity",
                "runtime_identity",
                "grid",
                "block",
                "kernel_specialization",
                "cache_policy",
                "clock_configuration",
            )
        }
        comparable_identity["capture_mode"] = capture["capture_mode"]
        if reference_identity is None:
            reference_identity = comparable_identity
        _require(
            comparable_identity == reference_identity,
            f"capture identity drifted in {capture_path}",
        )
        selected_ids = {parent_id, *components_by_id}
        sites_by_id = {
            site.get("site_id"): site
            for site in capture["sites"]
            if isinstance(site, dict)
        }
        selected_sites: dict[int, dict[str, Any]] = {}
        for site_id in selected_ids:
            site = sites_by_id.get(site_id)
            _require(site is not None, f"capture {capture_path} lacks site {site_id}")
            plan_site = (
                plan["parent"] if site_id == parent_id else components_by_id[site_id]
            )
            _require(
                site.get("name") == plan_site["name"],
                f"plan/capture site name mismatch for site {site_id}",
            )
            selected_sites[site_id] = {
                field: site.get(field)
                for field in (
                    "site_id",
                    "name",
                    "kind",
                    "role",
                    "boundary_semantics",
                    "async_domain",
                    "source_anchor",
                )
            }
        if reference_sites is None:
            reference_sites = selected_sites
        _require(
            selected_sites == reference_sites,
            f"selected site semantics drifted in {capture_path}",
        )
        if plan["owner_topology"] == "same":
            if reference_owners is None:
                reference_owners = capture["owners"]
            _require(
                capture["owners"] == reference_owners,
                f"owner topology drifted in {capture_path}; declare variation "
                "explicitly if intentional",
            )
        launch_id = identity["launch_id"]
        _require(
            isinstance(launch_id, int) and not isinstance(launch_id, bool),
            f"capture {capture_path} launch_id must be an integer",
        )
        _require(launch_id not in launch_ids, f"duplicate launch_id {launch_id}")
        launch_ids.add(launch_id)

        range_events = [
            event
            for event in capture["events"]
            if isinstance(event, dict) and event.get("type") == "range"
        ]
        owner_reports: list[dict[str, Any]] = []
        capture_instances: list[dict[str, Any]] = []
        events_by_owner_site: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(
            list
        )
        for event in range_events:
            for field in (
                "raw_start",
                "raw_end",
                "owner_relative_start_ns",
                "duration_ns",
            ):
                value = event.get(field)
                _require(
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value >= 0,
                    f"canonical {field} must be a non-negative integer",
                )
            _require(
                event["duration_ns"] == event["raw_end"] - event["raw_start"],
                "canonical duration must equal raw_end - raw_start",
            )
            events_by_owner_site[(event["owner"], event["site_id"])].append(event)

        for owner in capture["owners"]:
            owner_id = owner.get("owner")
            _require(
                isinstance(owner_id, int) and not isinstance(owner_id, bool),
                "owner id must be an integer",
            )
            parents = sorted(
                events_by_owner_site[(owner_id, parent_id)],
                key=lambda event: event["raw_start"],
            )
            _require(parents, f"owner {owner_id} emitted no parent range")
            for left, right in zip(parents, parents[1:]):
                _require(
                    left["raw_end"] <= right["raw_start"],
                    f"owner {owner_id} has overlapping parent ranges",
                )
            parent_starts = [parent["raw_start"] for parent in parents]
            components_by_occurrence: dict[int, list[dict[str, Any]]] = defaultdict(
                list
            )
            for site_id in components_by_id:
                for event in events_by_owner_site[(owner_id, site_id)]:
                    occurrence = bisect_right(parent_starts, event["raw_start"]) - 1
                    _require(
                        occurrence >= 0
                        and event["raw_end"] <= parents[occurrence]["raw_end"],
                        f"owner {owner_id} component site {site_id} is outside every parent",
                    )
                    parent = parents[occurrence]
                    _require(
                        event["owner_relative_start_ns"]
                        - parent["owner_relative_start_ns"]
                        == event["raw_start"] - parent["raw_start"],
                        "canonical relative and raw timestamps disagree",
                    )
                    components_by_occurrence[occurrence].append(event)

            owner_instances: list[dict[str, Any]] = []
            for occurrence, parent in enumerate(parents):
                parent_duration = _positive_number(
                    parent.get("duration_ns"), "parent duration"
                )
                component_durations: dict[str, float] = defaultdict(float)
                component_occurrences: dict[str, int] = defaultdict(int)
                intervals: list[tuple[float, float]] = []
                for event in components_by_occurrence[occurrence]:
                    site_id = event["site_id"]
                    component = components_by_id[site_id]
                    duration = _positive_number(
                        event.get("duration_ns"),
                        f"component {component['name']} duration",
                    )
                    name = component["name"]
                    component_durations[name] += duration
                    component_occurrences[name] += 1
                    relative_start = (
                        event["owner_relative_start_ns"]
                        - parent["owner_relative_start_ns"]
                    )
                    intervals.append((relative_start, relative_start + duration))

                component_sum = sum(component_durations.values())
                component_union = _interval_union_ns(intervals)
                _require(
                    component_union <= parent_duration,
                    "component union exceeds parent duration",
                )
                uncovered_gap = parent_duration - component_union
                instance = {
                    "capture_evidence_id": receipt["evidence_id"],
                    "launch_id": launch_id,
                    "owner": owner_id,
                    "owner_label": owner.get("label"),
                    "block": owner.get("block"),
                    "thread": owner.get("thread"),
                    "occurrence": occurrence,
                    "parent_duration_ns": parent_duration,
                    "component_duration_ns": dict(component_durations),
                    "component_occurrences": dict(component_occurrences),
                    "component_sum_ns": component_sum,
                    "component_union_ns": component_union,
                    "component_overlap_ns": max(0.0, component_sum - component_union),
                    "uncovered_gap_ns": uncovered_gap if has_components else None,
                    "component_union_fraction": (
                        component_union / parent_duration if has_components else None
                    ),
                }
                owner_instances.append(instance)
                capture_instances.append(instance)
                all_instances.append(instance)

            owner_parent_values = [
                instance["parent_duration_ns"] for instance in owner_instances
            ]
            owner_report = {
                "owner": owner_id,
                "owner_label": owner.get("label"),
                "block": owner.get("block"),
                "thread": owner.get("thread"),
                "parent_duration": _summary(owner_parent_values),
            }
            if has_components:
                owner_report.update(
                    {
                        "uncovered_gap": _summary(
                            [
                                instance["uncovered_gap_ns"]
                                for instance in owner_instances
                            ]
                        ),
                        "component_union_fraction_median": statistics.median(
                            instance["component_union_fraction"]
                            for instance in owner_instances
                        ),
                    }
                )
            owner_reports.append(owner_report)

        slowest_owner = max(
            owner_reports, key=lambda owner: owner["parent_duration"]["median_ns"]
        )
        owner_medians = [
            owner["parent_duration"]["median_ns"] for owner in owner_reports
        ]
        capture_reports.append(
            {
                "capture_evidence_id": receipt["evidence_id"],
                "launch_id": launch_id,
                "owner_count": len(owner_reports),
                "unique_blocks": len({owner["block"] for owner in owner_reports}),
                "owners": owner_reports,
                "slowest_owner_by_parent_median": {
                    "owner": slowest_owner["owner"],
                    "owner_label": slowest_owner["owner_label"],
                    "block": slowest_owner["block"],
                    "thread": slowest_owner["thread"],
                    "median_ns": slowest_owner["parent_duration"]["median_ns"],
                },
                "owner_median_spread_ns": max(owner_medians) - min(owner_medians),
                "owner_median_relative_spread": (
                    (max(owner_medians) - min(owner_medians))
                    / statistics.median(owner_medians)
                ),
                "observed_parent_duration": _summary(
                    [instance["parent_duration_ns"] for instance in capture_instances]
                ),
            }
        )

    parent_values = [instance["parent_duration_ns"] for instance in all_instances]
    component_report: dict[str, Any] = {}
    for component in plan["components"]:
        name = component["name"]
        values = [
            instance["component_duration_ns"][name]
            for instance in all_instances
            if name in instance["component_duration_ns"]
        ]
        component_report[name] = {
            "site_id": component["site_id"],
            "parents_with_component": len(values),
            "parent_count": len(all_instances),
            "duration": _summary(values) if values else None,
        }
        _require(values, f"declared component {name!r} was not observed")

    capture_inputs = sorted(
        (
            {
                "canonical": _descriptor(capture_path),
                "receipt": _descriptor(receipt_path),
            }
            for capture_path, receipt_path in captures_with_receipts
        ),
        key=lambda row: row["receipt"]["sha256"],
    )
    inputs = portable_descriptors(
        {"plan": _descriptor(plan_path), "captures": capture_inputs}, output
    )
    binding_payload = {
        "inputs": binding_view(inputs),
        "capture_evidence_ids": sorted(receipt_ids),
        "kernel_sha256": kernel_sha256,
    }
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "validation": "accepted",
        "evidence_id": hashlib.sha256(_canonical_json(binding_payload)).hexdigest(),
        "evidence_grade": (
            "decision" if all(grade == "decision" for grade in evidence_grades) else "diagnostic"
        ),
        "kernel_sha256": kernel_sha256,
        "binding_payload": binding_payload,
        "inputs": inputs,
        "plan": {
            "source": "inputs.plan",
            "parent": plan["parent"],
            "components": plan["components"],
            "owner_topology": plan["owner_topology"],
        },
        "identity": reference_identity,
        "capture_count": len(capture_reports),
        "launch_ids": sorted(launch_ids),
        "capture_evidence_ids": sorted(receipt_ids),
        "parent_duration": _summary(parent_values),
        "components": component_report,
        "captures": capture_reports,
        "instances": all_instances,
        "interpretation_limits": [
            "Durations are comparable across owners; owner-local starts are not aligned.",
            "The component union avoids double-counting overlap; uncovered time is "
            "reported rather than assigned.",
            "The plan declares semantic parent and component sites; the analyzer does "
            "not infer source phases.",
        ],
    }

    if plan["components"]:
        report["closure"] = {
            "component_union": _summary(
                [instance["component_union_ns"] for instance in all_instances]
            ),
            "component_overlap": _summary(
                [instance["component_overlap_ns"] for instance in all_instances]
            ),
            "uncovered_gap": _summary(
                [instance["uncovered_gap_ns"] for instance in all_instances]
            ),
            "component_union_fraction_median": statistics.median(
                instance["component_union_fraction"] for instance in all_instances
            ),
        }

    if plan["clean_samples"] is not None:
        _require(
            plan["clean_reference"]["identity"]
            == {
                key: value
                for key, value in reference_identity.items()
                if key != "capture_mode"
            },
            "clean_reference identity does not match the captures",
        )
        clean = _summary(plan["clean_samples"])
        instrumented_median = report["parent_duration"]["median_ns"]
        report["clean_reference"] = {
            **clean,
            "source": plan["clean_reference"]["source"],
            "instrumented_parent_median_ns": instrumented_median,
            "relative_delta": instrumented_median / clean["median_ns"] - 1.0,
        }

    stability = plan.get("stability")
    if stability is not None:
        threshold = float(stability["material_relative_spread"])
        observed_spreads = [
            capture["owner_median_relative_spread"] for capture in capture_reports
        ]
        report["topology_stability"] = {
            "agent_declared_material_relative_spread": threshold,
            "capture_relative_spread": {
                "count": len(observed_spreads),
                "min": min(observed_spreads),
                "median": statistics.median(observed_spreads),
                "max": max(observed_spreads),
            },
            "material_spread_observed": any(
                spread > threshold for spread in observed_spreads
            ),
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if self_validate:
        validate_report(
            output, require_decision=report["evidence_grade"] == "decision"
        )
    return report


def _descriptor_target(owner_path: Path, descriptor: dict[str, Any]) -> Path:
    target = Path(descriptor["path"])
    return target if target.is_absolute() else owner_path.parent / target


def validate_report(path: Path, *, require_decision: bool = True) -> dict[str, Any]:
    report = _load(path, "critical-path report")
    _require(report.get("schema") == REPORT_SCHEMA, "unsupported critical-path report")
    _require(report.get("validation") == "accepted", "critical-path report is not accepted")
    if require_decision:
        _require(
            report.get("evidence_grade") == "decision",
            "decision-grade critical-path report is required",
        )
    kernel_sha256 = report.get("kernel_sha256")
    _require(
        isinstance(kernel_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", kernel_sha256),
        "critical-path report has no authoritative kernel hash",
    )
    inputs = report.get("inputs")
    _require(isinstance(inputs, dict), "critical-path report inputs are missing")
    try:
        verify_descriptor_tree(path, inputs, "inputs")
    except RuntimeError as error:
        raise CriticalPathError(str(error)) from error
    plan_descriptor = inputs.get("plan")
    captures = inputs.get("captures")
    _require(isinstance(plan_descriptor, dict), "critical-path plan descriptor is missing")
    _require(isinstance(captures, list) and captures, "critical-path capture inputs are missing")
    plan_path = _descriptor_target(path, plan_descriptor)
    plan = _plan(plan_path)
    _require(
        report.get("plan", {}).get("parent") == plan["parent"]
        and report.get("plan", {}).get("components") == plan["components"]
        and report.get("plan", {}).get("owner_topology") == plan["owner_topology"],
        "critical-path report plan does not match its bound input",
    )
    receipt_ids = []
    capture_paths: list[tuple[Path, Path]] = []
    for index, capture in enumerate(captures):
        _require(isinstance(capture, dict), f"critical-path capture {index} is invalid")
        canonical = capture.get("canonical")
        receipt_descriptor = capture.get("receipt")
        _require(isinstance(canonical, dict), f"critical-path capture {index} lacks canonical")
        _require(
            isinstance(receipt_descriptor, dict),
            f"critical-path capture {index} lacks receipt",
        )
        canonical_path = _descriptor_target(path, canonical)
        receipt_path = _descriptor_target(path, receipt_descriptor)
        receipt = validate_timeline_receipt(
            receipt_path,
            expected_outputs={"canonical": canonical_path},
            require_decision=require_decision,
        )
        capture_paths.append((canonical_path, receipt_path))
        _require(
            receipt.get("kernel_sha256") == kernel_sha256,
            "critical-path capture kernel hash mismatch",
        )
        receipt_ids.append(receipt["evidence_id"])
    expected_payload = {
        "inputs": binding_view(inputs),
        "capture_evidence_ids": sorted(receipt_ids),
        "kernel_sha256": kernel_sha256,
    }
    _require(
        report.get("binding_payload") == expected_payload,
        "critical-path binding payload mismatch",
    )
    _require(
        hashlib.sha256(_canonical_json(expected_payload)).hexdigest()
        == report.get("evidence_id"),
        "critical-path evidence id mismatch",
    )
    with tempfile.NamedTemporaryFile(
        prefix="ppu-critical-path-validate-",
        suffix=".json",
        dir=path.parent,
        delete=False,
    ) as temporary:
        regenerated_path = Path(temporary.name)
    try:
        regenerated = analyze(
            plan_path,
            capture_paths,
            regenerated_path,
            self_validate=False,
        )
    finally:
        regenerated_path.unlink(missing_ok=True)
    _require(report == regenerated, "critical-path report does not match bound inputs")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze an agent-declared PPU owner-local critical path"
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--capture",
        type=Path,
        nargs=2,
        action="append",
        metavar=("CANONICAL", "RECEIPT"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = analyze(args.plan, [tuple(value) for value in args.capture], args.output)
    message = (
        "PPU critical path accepted: "
        f"{report['capture_count']} captures, "
        f"{report['parent_duration']['count']} parent instances"
    )
    if "closure" in report:
        message += (
            ", "
            f"{report['closure']['component_union_fraction_median']:.2%} median closure"
        )
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
