#!/usr/bin/env python3
"""Close an agent-declared PPU owner-local critical path across captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


PLAN_SCHEMA = "ppu-critical-path-plan/v1"
CANONICAL_SCHEMA = "ppu-fixed-slot-canonical/v4"
REPORT_SCHEMA = "ppu-critical-path-report/v2"
RECEIPT_SCHEMA = "ppu-fixed-slot-receipt/v4"


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


def _capture_receipt(path: Path, canonical_path: Path) -> dict[str, Any]:
    receipt = _load(path, "timeline receipt")
    _require(
        receipt.get("schema") == RECEIPT_SCHEMA
        and receipt.get("validation") == "accepted",
        "canonical capture requires an accepted timeline receipt",
    )
    _require(
        receipt.get("evidence_grade") in {"diagnostic", "decision"},
        "timeline receipt has an invalid evidence grade",
    )
    binding_payload = receipt.get("binding_payload")
    _require(
        isinstance(binding_payload, dict)
        and hashlib.sha256(_canonical_json(binding_payload)).hexdigest()
        == receipt.get("evidence_id"),
        "timeline receipt evidence id is invalid",
    )
    outputs = receipt.get("outputs")
    _require(isinstance(outputs, dict), "timeline receipt outputs are missing")
    descriptor = outputs.get("canonical")
    _require(isinstance(descriptor, dict), "timeline canonical binding is missing")
    _require(
        descriptor.get("sha256") == _sha256(canonical_path),
        "canonical capture content hash does not match its receipt",
    )
    _require(
        descriptor.get("size_bytes") == canonical_path.stat().st_size,
        "canonical capture size does not match its receipt",
    )
    return receipt


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
        return 0.0
    ordered = sorted(intervals)
    start, end = ordered[0]
    total = 0.0
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
        raw_samples = clean_reference.get("duration_ns_samples")
        _require(
            isinstance(raw_samples, list) and raw_samples,
            "clean_reference.duration_ns_samples must be non-empty",
        )
        clean_samples = [
            _positive_number(value, f"clean reference sample {index}")
            for index, value in enumerate(raw_samples)
        ]
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
) -> dict[str, Any]:
    plan = _plan(plan_path)
    _require(bool(captures_with_receipts), "at least one canonical capture is required")
    captures = [
        (capture_path, *_canonical(capture_path, receipt_path))
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

    for capture_path, capture, receipt in captures:
        receipt_ids.append(receipt["evidence_id"])
        evidence_grades.append(receipt["evidence_grade"])
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
        for owner in capture["owners"]:
            owner_id = owner.get("owner")
            _require(
                isinstance(owner_id, int) and not isinstance(owner_id, bool),
                "owner id must be an integer",
            )
            parents = sorted(
                (
                    event
                    for event in range_events
                    if event.get("owner") == owner_id
                    and event.get("site_id") == parent_id
                ),
                key=lambda event: event["raw_start"],
            )
            _require(parents, f"owner {owner_id} emitted no parent range")
            for left, right in zip(parents, parents[1:]):
                _require(
                    left["raw_end"] <= right["raw_start"],
                    f"owner {owner_id} has overlapping parent ranges",
                )

            owner_component_events = [
                event
                for event in range_events
                if event.get("owner") == owner_id
                and event.get("site_id") in components_by_id
            ]
            for event in owner_component_events:
                _require(
                    any(
                        event["raw_start"] >= parent["raw_start"]
                        and event["raw_end"] <= parent["raw_end"]
                        for parent in parents
                    ),
                    f"owner {owner_id} component site {event['site_id']} is outside every parent",
                )

            owner_instances: list[dict[str, Any]] = []
            for occurrence, parent in enumerate(parents):
                parent_duration = _positive_number(
                    parent.get("duration_ns"), "parent duration"
                )
                component_durations: dict[str, float] = defaultdict(float)
                component_occurrences: dict[str, int] = defaultdict(int)
                intervals: list[tuple[float, float]] = []
                for event in range_events:
                    site_id = event.get("site_id")
                    if (
                        event.get("owner") != owner_id
                        or site_id not in components_by_id
                    ):
                        continue
                    if not (
                        event["raw_start"] >= parent["raw_start"]
                        and event["raw_end"] <= parent["raw_end"]
                    ):
                        continue
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
                uncovered_gap = max(0.0, parent_duration - component_union)
                instance = {
                    "capture": str(capture_path.resolve()),
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
                "capture": str(capture_path.resolve()),
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

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "validation": "accepted",
        "evidence_id": hashlib.sha256(
            _canonical_json(
                {
                    "plan_sha256": _sha256(plan_path),
                    "capture_evidence_ids": receipt_ids,
                }
            )
        ).hexdigest(),
        "evidence_grade": (
            "decision" if all(grade == "decision" for grade in evidence_grades) else "diagnostic"
        ),
        "plan": {
            "source": str(plan_path.resolve()),
            "parent": plan["parent"],
            "components": plan["components"],
            "owner_topology": plan["owner_topology"],
        },
        "identity": reference_identity,
        "capture_count": len(capture_reports),
        "launch_ids": sorted(launch_ids),
        "capture_evidence_ids": receipt_ids,
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
