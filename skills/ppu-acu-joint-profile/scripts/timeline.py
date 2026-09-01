#!/usr/bin/env python3
"""Decode validated PPU fixed-slot timelines and measure probe perturbation."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import struct
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


MAGIC = 0x0000314C54555050
ABI_MAJOR = 1
ABI_MINOR = 0
HEADER = struct.Struct("<Q4H10IQ")
RECORD = struct.Struct("<QII")
COMMITTED = 1 << 31
STATUS_NAMES = {
    1 << 0: "overflow",
    1 << 1: "bad_header",
    1 << 2: "bad_owner",
    1 << 3: "duplicate_owner",
}
KIND_NAMES = {0: "begin", 1: "end", 2: "instant", 3: "counter"}
MANIFEST_SCHEMA = "ppu-fixed-slot-timeline-manifest/v4"
DICTIONARY_SCHEMA = "ppu-fixed-slot-events/v2"
CANONICAL_SCHEMA = "ppu-fixed-slot-canonical/v4"
RECEIPT_SCHEMA = "ppu-fixed-slot-receipt/v4"
CORRECTNESS_SCHEMA = "ppu-timeline-correctness/v1"
MEASUREMENT_SCHEMA = "ppu-timeline-measurement/v1"
SAMPLE_PREFIX = "__PPU_TIMELINE_SAMPLE__="
TIMER_SOURCE = "globaltimer"
TIMER_UNIT = "ns"


class TimelineError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TimelineError(message)


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TimelineError(f"cannot read {label} {path}: {error}") from error
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _identity(value: object, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict) and value, f"{label} is required")
    return value


def _manifest_artifact(
    manifest_path: Path, manifest: dict[str, Any], field: str, label: str
) -> tuple[Path, dict[str, Any]]:
    value = manifest.get(field)
    _require(isinstance(value, str) and value.strip(), f"{field} is required")
    path = Path(value)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path, _load_object(path, label)


def _triple(value: object, label: str) -> tuple[int, int, int]:
    _require(isinstance(value, list) and len(value) == 3, f"{label} must be [x,y,z]")
    _require(
        all(isinstance(item, int) and not isinstance(item, bool) for item in value),
        f"{label} dimensions must be integers",
    )
    result = tuple(value)
    _require(all(item > 0 for item in result), f"{label} dimensions must be positive")
    return result  # type: ignore[return-value]


def _parse_header(raw: bytes) -> dict[str, int]:
    _require(len(raw) >= HEADER.size, "raw capture is smaller than the 64-byte header")
    names = (
        "magic",
        "abi_major",
        "abi_minor",
        "header_bytes",
        "record_bytes",
        "capacity",
        "owner_count",
        "records_per_owner",
        "status",
        "grid_x",
        "grid_y",
        "grid_z",
        "block_x",
        "block_y",
        "block_z",
        "launch_id",
    )
    header = dict(zip(names, HEADER.unpack_from(raw), strict=True))
    _require(header["magic"] == MAGIC, "raw capture has the wrong PPU timeline magic")
    _require(header["abi_major"] == ABI_MAJOR, "unsupported PPU timeline ABI major")
    _require(header["abi_minor"] == ABI_MINOR, "unsupported PPU timeline ABI minor")
    _require(header["header_bytes"] == HEADER.size, "header size does not match ABI v1")
    _require(header["record_bytes"] == RECORD.size, "record size does not match ABI v1")
    _require(header["owner_count"] > 0, "owner_count must be positive")
    _require(header["records_per_owner"] > 0, "records_per_owner must be positive")
    _require(
        header["capacity"] == header["owner_count"] * header["records_per_owner"],
        "capacity must equal owner_count * records_per_owner",
    )
    expected_bytes = (
        header["header_bytes"]
        + header["capacity"] * header["record_bytes"]
        + header["owner_count"] * 8
    )
    _require(
        len(raw) == expected_bytes,
        f"raw capture has {len(raw)} bytes; expected {expected_bytes}",
    )
    if header["status"]:
        known = [name for bit, name in STATUS_NAMES.items() if header["status"] & bit]
        unknown = header["status"] & ~sum(STATUS_NAMES)
        if unknown:
            known.append(f"unknown_status_0x{unknown:x}")
        raise TimelineError("device rejected capture: " + ", ".join(known))
    return header


def _validate_manifest(
    manifest: dict[str, Any], header: dict[str, int], manifest_path: Path
) -> dict[str, Any]:
    _require(manifest.get("schema") == MANIFEST_SCHEMA, "unknown manifest schema")
    _require(
        manifest.get("backend") == "ppu_fixed_slot",
        "manifest backend must be ppu_fixed_slot",
    )
    grid = _triple(manifest.get("grid"), "manifest.grid")
    block = _triple(manifest.get("block"), "manifest.block")
    _require(grid == tuple(header[f"grid_{axis}"] for axis in "xyz"), "grid mismatch")
    _require(
        block == tuple(header[f"block_{axis}"] for axis in "xyz"), "block mismatch"
    )
    launch_id = manifest.get("launch_id")
    _require(
        isinstance(launch_id, int) and not isinstance(launch_id, bool),
        "launch_id must be an integer",
    )
    _require(launch_id == header["launch_id"], "launch_id mismatch")
    records_per_owner = manifest.get("records_per_owner")
    _require(
        isinstance(records_per_owner, int)
        and not isinstance(records_per_owner, bool)
        and records_per_owner == header["records_per_owner"],
        "records_per_owner mismatch",
    )
    _require(
        manifest.get("clock_scope") == "owner_local", "clock_scope must be owner_local"
    )
    capture_mode = manifest.get("capture_mode")
    _require(capture_mode in {"coarse", "fine"}, "capture_mode must be coarse or fine")
    _require(
        isinstance(manifest.get("sampling_rationale"), str)
        and manifest["sampling_rationale"].strip(),
        "sampling_rationale is required",
    )
    _require(
        isinstance(manifest.get("kernel_name"), str)
        and manifest["kernel_name"].strip(),
        "kernel_name is required",
    )
    duration = manifest.get("kernel_duration_ns")
    _require(
        isinstance(duration, (int, float))
        and not isinstance(duration, bool)
        and math.isfinite(duration)
        and duration > 0,
        "kernel_duration_ns must be positive",
    )
    _require(
        isinstance(manifest.get("workload_identity"), str)
        and manifest["workload_identity"].strip(),
        "workload_identity is required",
    )
    device_identity = _identity(manifest.get("device_identity"), "device_identity")
    runtime_identity = _identity(manifest.get("runtime_identity"), "runtime_identity")
    _require(
        "timer_tick_ns" not in manifest and "timer_calibration" not in manifest,
        "manifest v4 uses the documented globaltimer nanosecond contract; remove timer conversion fields",
    )
    timer = manifest.get("timer")
    _require(isinstance(timer, dict), "manifest.timer must be an object")
    _require(
        timer.get("source") == TIMER_SOURCE,
        "manifest.timer.source must be globaltimer",
    )
    _require(
        timer.get("unit") == TIMER_UNIT,
        "manifest.timer.unit must be ns",
    )
    correctness_path, correctness = _manifest_artifact(
        manifest_path, manifest, "correctness_artifact", "correctness evidence"
    )
    _require(
        correctness.get("schema") == CORRECTNESS_SCHEMA
        and correctness.get("validation") == "accepted",
        "correctness evidence must be an accepted artifact",
    )
    _require(
        correctness.get("kernel_name") == manifest["kernel_name"],
        "correctness kernel identity mismatch",
    )
    _require(
        correctness.get("workload_identity") == manifest["workload_identity"],
        "correctness workload identity mismatch",
    )
    _require(
        correctness.get("device_identity") == device_identity,
        "correctness device identity mismatch",
    )
    correctness_checks = correctness.get("checks")
    _require(
        isinstance(correctness_checks, list) and correctness_checks,
        "correctness evidence needs at least one check",
    )
    _require(
        all(
            isinstance(check, dict)
            and isinstance(check.get("name"), str)
            and check["name"].strip()
            and check.get("status") == "passed"
            for check in correctness_checks
        ),
        "every correctness check must be named and passed",
    )
    layout = manifest.get("owner_layout")
    _require(isinstance(layout, dict), "owner_layout must be an object")
    _require(
        layout.get("kind") == "explicit_writers",
        "owner_layout.kind must be explicit_writers",
    )
    raw_owners = layout.get("owners")
    _require(
        isinstance(raw_owners, list) and raw_owners,
        "owner_layout.owners must be a non-empty list",
    )
    _require(
        len(raw_owners) == header["owner_count"],
        "owner_layout.owners count must match the raw owner_count",
    )
    owners: list[dict[str, Any]] = []
    labels: set[str] = set()
    for expected_owner, raw_owner in enumerate(raw_owners):
        _require(
            isinstance(raw_owner, dict),
            f"owner_layout.owners[{expected_owner}] must be an object",
        )
        _require(
            raw_owner.get("owner") == expected_owner,
            "owner ids must be dense and ordered from zero",
        )
        owner_block = raw_owner.get("block")
        owner_thread = raw_owner.get("thread")
        _require(
            isinstance(owner_block, int)
            and not isinstance(owner_block, bool)
            and 0 <= owner_block < math.prod(grid),
            f"owner {expected_owner} block is outside the grid",
        )
        _require(
            isinstance(owner_thread, int)
            and not isinstance(owner_thread, bool)
            and 0 <= owner_thread < math.prod(block),
            f"owner {expected_owner} thread is outside the block",
        )
        label = raw_owner.get("label")
        purpose = raw_owner.get("purpose")
        _require(
            isinstance(label, str) and label.strip(),
            f"owner {expected_owner} needs a label",
        )
        _require(label not in labels, f"duplicate owner label {label!r}")
        _require(
            isinstance(purpose, str) and purpose.strip(),
            f"owner {expected_owner} needs a purpose",
        )
        owners.append(
            {
                "owner": expected_owner,
                "block": owner_block,
                "thread": owner_thread,
                "label": label,
                "purpose": purpose,
            }
        )
        labels.add(label)

    analysis = manifest.get("analysis")
    if analysis is not None:
        _require(isinstance(analysis, dict), "analysis must be an object")
        analysis_owner = analysis.get("owner")
        _require(
            isinstance(analysis_owner, int)
            and not isinstance(analysis_owner, bool)
            and 0 <= analysis_owner < len(owners),
            "analysis.owner is outside the declared owner list",
        )
        window_site_id = analysis.get("window_site_id")
        _require(
            window_site_id is None
            or (
                isinstance(window_site_id, int)
                and not isinstance(window_site_id, bool)
                and 0 <= window_site_id <= 0xFFFF
            ),
            "analysis.window_site_id must fit uint16 when present",
        )
        analysis_site_ids = analysis.get("site_ids")
        if analysis_site_ids is not None:
            _require(
                isinstance(analysis_site_ids, list) and analysis_site_ids,
                "analysis.site_ids must be a non-empty list when present",
            )
            _require(
                all(
                    isinstance(site_id, int)
                    and not isinstance(site_id, bool)
                    and 0 <= site_id <= 0xFFFF
                    for site_id in analysis_site_ids
                ),
                "analysis.site_ids entries must fit uint16",
            )
            _require(
                len(set(analysis_site_ids)) == len(analysis_site_ids),
                "analysis.site_ids contains duplicates",
            )
            _require(
                window_site_id not in analysis_site_ids,
                "analysis.window_site_id must not also appear in analysis.site_ids",
            )
        for field in ("tile", "k_stage"):
            value = analysis.get(field)
            _require(
                value is None
                or (
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value >= 0
                ),
                f"analysis.{field} must be a non-negative integer when present",
            )

    coverage = manifest.get("coverage", {"all_blocks": False})
    _require(isinstance(coverage, dict), "coverage must be an object")
    all_blocks = coverage.get("all_blocks", False)
    _require(isinstance(all_blocks, bool), "coverage.all_blocks must be boolean")
    range_site_id = coverage.get("range_site_id")
    _require(
        range_site_id is None
        or (
            isinstance(range_site_id, int)
            and not isinstance(range_site_id, bool)
            and 0 <= range_site_id <= 0xFFFF
        ),
        "coverage.range_site_id must fit uint16 when present",
    )
    if all_blocks:
        _require(
            range_site_id is not None,
            "coverage.range_site_id is required when all_blocks is true",
        )
        _require(
            {owner["block"] for owner in owners} == set(range(math.prod(grid))),
            "all-block coverage requires at least one declared owner for every block",
        )
    return {
        "grid": grid,
        "block": block,
        "timer": {"source": TIMER_SOURCE, "unit": TIMER_UNIT},
        "correctness_path": correctness_path,
        "correctness": correctness,
        "runtime_identity": runtime_identity,
        "owners": owners,
        "analysis": analysis,
        "coverage": {"all_blocks": all_blocks, "range_site_id": range_site_id},
    }


def _load_sites(
    dictionary: dict[str, Any], manifest_info: dict[str, Any]
) -> dict[int, dict[str, Any]]:
    _require(
        dictionary.get("schema") == DICTIONARY_SCHEMA, "unknown event dictionary schema"
    )
    raw_sites = dictionary.get("sites")
    _require(
        isinstance(raw_sites, list) and raw_sites,
        "event dictionary sites must be non-empty",
    )
    sites: dict[int, dict[str, Any]] = {}
    names: set[str] = set()
    owner_count = len(manifest_info["owners"])
    for position, site in enumerate(raw_sites):
        _require(isinstance(site, dict), f"sites[{position}] must be an object")
        site_id = site.get("site_id")
        name = site.get("name")
        kind = site.get("kind")
        role = site.get("role", "observation")
        _require(
            isinstance(site_id, int)
            and not isinstance(site_id, bool)
            and 0 <= site_id <= 0xFFFF,
            "site_id must fit uint16",
        )
        _require(isinstance(name, str) and name.strip(), f"site {site_id} needs a name")
        _require(
            kind in {"range", "instant", "counter", "any"},
            f"site {site_id} has invalid kind",
        )
        _require(
            isinstance(role, str) and role.strip(), f"site {site_id} has invalid role"
        )
        allowed_owners = site.get("owners")
        if allowed_owners is not None:
            _require(
                isinstance(allowed_owners, list) and allowed_owners,
                f"site {site_id} owners must be a non-empty list when present",
            )
            _require(
                all(
                    isinstance(owner, int)
                    and not isinstance(owner, bool)
                    and 0 <= owner < owner_count
                    for owner in allowed_owners
                ),
                f"site {site_id} owners reference an undeclared owner",
            )
            _require(
                len(set(allowed_owners)) == len(allowed_owners),
                f"site {site_id} owners contain duplicates",
            )
        for field in ("boundary_semantics", "async_domain", "source_anchor"):
            _require(
                isinstance(site.get(field), str) and site[field].strip(),
                f"site {site_id} needs {field}",
            )
        _require(site_id not in sites, f"duplicate site_id {site_id}")
        _require(name not in names, f"duplicate site name {name!r}")
        sites[site_id] = {**site, "role": role}
        names.add(name)

    analysis = manifest_info["analysis"]
    if analysis is not None and analysis.get("window_site_id") is not None:
        site_id = analysis["window_site_id"]
        site = sites.get(site_id)
        _require(
            site is not None, "analysis.window_site_id is absent from the dictionary"
        )
        _require(site["kind"] == "range", "analysis window site must be a range")
        allowed = site.get("owners")
        _require(
            allowed is None or analysis["owner"] in allowed,
            "analysis window site excludes analysis.owner",
        )
    if analysis is not None and analysis.get("site_ids") is not None:
        for site_id in analysis["site_ids"]:
            site = sites.get(site_id)
            _require(
                site is not None,
                f"analysis site {site_id} is absent from the dictionary",
            )
            _require(
                site["kind"] == "range", f"analysis site {site_id} must be a range"
            )
            allowed = site.get("owners")
            _require(
                allowed is None or analysis["owner"] in allowed,
                f"analysis site {site_id} excludes analysis.owner",
            )

    coverage = manifest_info["coverage"]
    if coverage["range_site_id"] is not None:
        site = sites.get(coverage["range_site_id"])
        _require(
            site is not None, "coverage.range_site_id is absent from the dictionary"
        )
        _require(site["kind"] == "range", "coverage range site must be a range")
    return sites


def _block_xyz(owner: int, grid: tuple[int, int, int]) -> list[int]:
    x = owner % grid[0]
    y = (owner // grid[0]) % grid[1]
    z = owner // (grid[0] * grid[1])
    return [x, y, z]


def _decode_records(
    raw: bytes,
    header: dict[str, int],
    manifest_info: dict[str, Any],
    sites: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    claims_offset = header["header_bytes"] + header["capacity"] * header["record_bytes"]
    for owner in range(header["owner_count"]):
        owner_info = manifest_info["owners"][owner]
        (claim,) = struct.unpack_from("<Q", raw, claims_offset + owner * 8)
        if not claim:
            errors.append(f"owner {owner} has no writer claim")
        else:
            claimed_block = ((claim >> 32) & 0xFFFFFFFF) - 1
            claimed_thread = (claim & 0xFFFFFFFF) - 1
            if claimed_block != owner_info["block"]:
                errors.append(
                    f"owner {owner} claims block {claimed_block}; "
                    f"expected {owner_info['block']}"
                )
            if claimed_thread != owner_info["thread"]:
                errors.append(
                    f"owner {owner} claims thread {claimed_thread}; "
                    f"expected {owner_info['thread']}"
                )
        saw_hole = False
        previous_timestamp = -1
        for sequence in range(header["records_per_owner"]):
            slot = owner * header["records_per_owner"] + sequence
            offset = header["header_bytes"] + slot * header["record_bytes"]
            timestamp, payload, tag = RECORD.unpack_from(raw, offset)
            committed = bool(tag & COMMITTED)
            if not committed:
                if timestamp or payload or tag:
                    errors.append(f"slot {slot} is nonzero without commit")
                saw_hole = True
                continue
            if saw_hole:
                errors.append(f"owner {owner} has a committed record after a hole")
            site_id = tag & 0xFFFF
            kind = KIND_NAMES[(tag >> 16) & 0x3]
            flags = (tag >> 18) & 0x1FFF
            site = sites.get(site_id)
            if site is None:
                errors.append(f"slot {slot} references unknown site {site_id}")
                continue
            declared_kind = site["kind"]
            if declared_kind == "range" and kind not in {"begin", "end"}:
                errors.append(f"slot {slot} kind {kind} does not match range site")
            elif declared_kind not in {"any", "range", kind}:
                errors.append(f"slot {slot} kind {kind} does not match {declared_kind}")
            allowed_owners = site.get("owners")
            if allowed_owners is not None and owner not in allowed_owners:
                errors.append(f"site {site_id} excludes emitting owner {owner}")
            if timestamp == 0:
                errors.append(f"slot {slot} has a zero timestamp")
            if timestamp < previous_timestamp:
                errors.append(
                    f"owner {owner} timestamps decrease at sequence {sequence}"
                )
            previous_timestamp = timestamp
            records.append(
                {
                    "owner": owner,
                    "owner_label": owner_info["label"],
                    "owner_purpose": owner_info["purpose"],
                    "block": owner_info["block"],
                    "block_xyz": _block_xyz(owner_info["block"], manifest_info["grid"]),
                    "thread": owner_info["thread"],
                    "sequence": sequence,
                    "raw_timestamp": timestamp,
                    "payload": payload,
                    "flags": flags,
                    "site_id": site_id,
                    "site": site,
                    "kind": kind,
                }
            )
    if errors:
        raise TimelineError("; ".join(errors))
    return records


def _canonicalize(
    records: list[dict[str, Any]], owner_count: int
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    stacks: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    events: list[dict[str, Any]] = []
    errors: list[str] = []
    for record in records:
        key = (record["owner"], record["site_id"])
        if record["kind"] == "begin":
            stacks[key].append(record)
        elif record["kind"] == "end":
            if not stacks[key]:
                errors.append(
                    f"owner {record['owner']} site {record['site_id']} has unmatched end"
                )
                continue
            begin = stacks[key].pop()
            events.append(
                {
                    "type": "range",
                    "name": record["site"]["name"],
                    "role": record["site"].get("role", "observation"),
                    "owner": record["owner"],
                    "owner_label": record["owner_label"],
                    "owner_purpose": record["owner_purpose"],
                    "block": record["block"],
                    "block_xyz": record["block_xyz"],
                    "thread": record["thread"],
                    "site_id": record["site_id"],
                    "begin_sequence": begin["sequence"],
                    "end_sequence": record["sequence"],
                    "raw_start": begin["raw_timestamp"],
                    "raw_end": record["raw_timestamp"],
                    "duration_ns": record["raw_timestamp"] - begin["raw_timestamp"],
                    "begin_payload": begin["payload"],
                    "end_payload": record["payload"],
                    "boundary_semantics": record["site"].get("boundary_semantics"),
                    "async_domain": record["site"].get("async_domain"),
                    "source_anchor": record["site"].get("source_anchor"),
                }
            )
        elif record["kind"] in {"instant", "counter"}:
            events.append(
                {
                    "type": record["kind"],
                    "name": record["site"]["name"],
                    "role": record["site"].get("role", "observation"),
                    "owner": record["owner"],
                    "owner_label": record["owner_label"],
                    "owner_purpose": record["owner_purpose"],
                    "block": record["block"],
                    "block_xyz": record["block_xyz"],
                    "thread": record["thread"],
                    "site_id": record["site_id"],
                    "sequence": record["sequence"],
                    "raw_timestamp": record["raw_timestamp"],
                    "payload": record["payload"],
                    "boundary_semantics": record["site"].get("boundary_semantics"),
                    "async_domain": record["site"].get("async_domain"),
                    "source_anchor": record["site"].get("source_anchor"),
                }
            )
    for (owner, site_id), pending in stacks.items():
        if pending:
            errors.append(
                f"owner {owner} site {site_id} has {len(pending)} unmatched begin record(s)"
            )

    origins: dict[int, int] = {}
    for owner in range(owner_count):
        owner_records = [record for record in records if record["owner"] == owner]
        if not owner_records:
            errors.append(f"owner {owner} emitted no records")
            continue
        origins[owner] = min(owner_records, key=lambda record: record["sequence"])[
            "raw_timestamp"
        ]
    if errors:
        raise TimelineError("; ".join(errors))

    for event in events:
        raw_start = event.get("raw_start", event.get("raw_timestamp"))
        event["owner_relative_start_ns"] = raw_start - origins[event["owner"]]
    events.sort(
        key=lambda event: (
            event["owner"],
            event["owner_relative_start_ns"],
            event["site_id"],
        )
    )
    return events, origins


def _perfetto(
    manifest: dict[str, Any],
    manifest_info: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    owner_pid = 6
    analysis = manifest.get("analysis")
    coverage = manifest.get("coverage", {"all_blocks": False})
    sites_by_id = {event["site_id"]: event["name"] for event in events}
    trace_events: list[dict[str, Any]] = [
        {
            "name": "process_name",
            "ph": "M",
            "pid": owner_pid,
            "args": {"name": "PPU owner-local captures (lanes are unaligned)"},
        }
    ]
    for owner in manifest["owner_layout"]["owners"]:
        trace_events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": owner_pid,
                "tid": owner["owner"],
                "args": {
                    "name": (
                        f"{owner['label']} (block {owner['block']}, "
                        f"thread {owner['thread']})"
                    )
                },
            }
        )
    for event in events:
        common = {
            "name": event["name"],
            "cat": "PPU owner-local timeline",
            "ts": event["owner_relative_start_ns"] / 1000.0,
            "pid": owner_pid,
            "tid": event["owner"],
            "args": {
                "block": event["block"],
                "block_xyz": event["block_xyz"],
                "thread": event["thread"],
                "owner": event["owner"],
                "owner_label": event["owner_label"],
                "owner_purpose": event["owner_purpose"],
                "site_id": event["site_id"],
                "role": event["role"],
                "clock_scope": "owner_local",
                "boundary_semantics": event.get("boundary_semantics"),
                "async_domain": event.get("async_domain"),
                "source_anchor": event.get("source_anchor"),
            },
        }
        if event["type"] == "range":
            trace_events.append(
                {**common, "ph": "X", "dur": event["duration_ns"] / 1000.0}
            )
        elif event["type"] == "instant":
            trace_events.append({**common, "ph": "i", "s": "t"})
        else:
            trace_events.append(
                {
                    **common,
                    "ph": "C",
                    "args": {**common["args"], "value": event["payload"]},
                }
            )
    return {
        "displayTimeUnit": "ns",
        "ppuTimeline": {
            "schemaVersion": 4,
            "kernelName": manifest["kernel_name"],
            "kernelDurationNs": manifest["kernel_duration_ns"],
            "grid": manifest["grid"],
            "blockDims": manifest["block"],
            "captureMode": manifest["capture_mode"],
            "samplingRationale": manifest["sampling_rationale"],
            "owners": manifest["owner_layout"]["owners"],
            "ownerEventPid": owner_pid,
            "analysisOwner": analysis.get("owner") if analysis else None,
            "analysisBlock": (
                manifest["owner_layout"]["owners"][analysis["owner"]]["block"]
                if analysis
                else None
            ),
            "analysisThread": (
                manifest["owner_layout"]["owners"][analysis["owner"]]["thread"]
                if analysis
                else None
            ),
            "analysisWindowSiteId": (
                analysis.get("window_site_id") if analysis else None
            ),
            "analysisSiteIds": analysis.get("site_ids") if analysis else None,
            "analysisWindowEventName": (
                sites_by_id.get(analysis.get("window_site_id")) if analysis else None
            ),
            "tile": analysis.get("tile") if analysis else None,
            "kStage": analysis.get("k_stage") if analysis else None,
            "coverage": {
                "allBlocks": coverage.get("all_blocks", False),
                "rangeSiteId": coverage.get("range_site_id"),
                "rangeEventName": sites_by_id.get(coverage.get("range_site_id")),
            },
            "clockScope": "owner_local",
            "launchId": manifest["launch_id"],
            "workloadIdentity": manifest.get("workload_identity"),
            "deviceIdentity": manifest.get("device_identity"),
            "runtimeIdentity": manifest.get("runtime_identity"),
            "timerSource": manifest_info["timer"]["source"],
            "timerUnit": manifest_info["timer"]["unit"],
            "timerContractValidation": "accepted",
            "correctnessValidation": "accepted",
            "captureValidation": "accepted",
        },
        "traceEvents": trace_events,
    }


def decode(
    raw_path: Path,
    manifest_path: Path,
    dictionary_path: Path,
    output_prefix: Path,
) -> dict[str, Any]:
    raw = raw_path.read_bytes()
    manifest = _load_object(manifest_path, "manifest")
    dictionary = _load_object(dictionary_path, "event dictionary")
    header = _parse_header(raw)
    manifest_info = _validate_manifest(manifest, header, manifest_path)
    sites = _load_sites(dictionary, manifest_info)
    records = _decode_records(raw, header, manifest_info, sites)
    events, _ = _canonicalize(records, header["owner_count"])
    analysis = manifest_info["analysis"]
    if analysis is not None and analysis.get("window_site_id") is not None:
        windows = [
            event
            for event in events
            if event["type"] == "range"
            and event["owner"] == analysis["owner"]
            and event["site_id"] == analysis["window_site_id"]
        ]
        _require(
            len(windows) == 1, "analysis owner must emit exactly one analysis window"
        )
        window = windows[0]
        for event in events:
            if (
                event["type"] == "range"
                and event["owner"] == analysis["owner"]
                and event["site_id"] in (analysis.get("site_ids") or [])
            ):
                _require(
                    event["raw_start"] >= window["raw_start"]
                    and event["raw_end"] <= window["raw_end"],
                    f"analysis site {event['site_id']} is outside the analysis window",
                )

    coverage = manifest_info["coverage"]
    if coverage["all_blocks"]:
        coverage_ranges = [
            event
            for event in events
            if event["type"] == "range"
            and event["site_id"] == coverage["range_site_id"]
        ]
        counts_by_block: dict[int, int] = defaultdict(int)
        for event in coverage_ranges:
            counts_by_block[event["block"]] += 1
        _require(
            counts_by_block
            == {block: 1 for block in range(math.prod(manifest_info["grid"]))},
            "all-block coverage requires exactly one coverage range per block",
        )
    perfetto = _perfetto(manifest, manifest_info, events)

    range_durations: dict[str, list[float]] = defaultdict(list)
    for event in events:
        if event["type"] == "range":
            range_durations[event["name"]].append(event["duration_ns"])
    summary = {
        "schema": "ppu-fixed-slot-summary/v4",
        "validation": "accepted",
        "clock_scope": "owner_local",
        "capture_mode": manifest["capture_mode"],
        "sampling_rationale": manifest["sampling_rationale"],
        "owner_count": header["owner_count"],
        "record_count": len(records),
        "owners": manifest["owner_layout"]["owners"],
        "analysis": manifest.get("analysis"),
        "coverage": manifest_info["coverage"],
        "identity": {
            "kernel_name": manifest["kernel_name"],
            "workload_identity": manifest["workload_identity"],
            "device_identity": manifest["device_identity"],
            "runtime_identity": manifest["runtime_identity"],
            "launch_id": manifest["launch_id"],
        },
        "timer": {
            "validation": "accepted",
            "source": manifest_info["timer"]["source"],
            "unit": manifest_info["timer"]["unit"],
            "conversion": "identity",
        },
        "correctness": {
            "validation": "accepted",
            "checks": manifest_info["correctness"]["checks"],
            "artifact": str(manifest_info["correctness_path"].resolve()),
        },
        "ranges": {
            name: {
                "count": len(values),
                "min_ns": min(values),
                "median_ns": statistics.median(values),
                "max_ns": max(values),
            }
            for name, values in sorted(range_durations.items())
        },
        "interpretation_limits": [
            "Only differences within one owner are ordered.",
            "Issue markers do not imply asynchronous completion; use an explicit wait boundary.",
            "Probe cost is included in local ranges and must be bounded separately.",
        ],
    }
    canonical = {
        "schema": CANONICAL_SCHEMA,
        "clock_scope": "owner_local",
        "timer": {
            "source": manifest_info["timer"]["source"],
            "unit": manifest_info["timer"]["unit"],
        },
        "capture_mode": manifest["capture_mode"],
        "identity": {
            "kernel_name": manifest["kernel_name"],
            "workload_identity": manifest["workload_identity"],
            "device_identity": manifest["device_identity"],
            "runtime_identity": manifest["runtime_identity"],
            "launch_id": manifest["launch_id"],
        },
        "validation": {
            "capture": "accepted",
            "timer_contract": "accepted",
            "correctness": "accepted",
        },
        "grid": manifest["grid"],
        "block": manifest["block"],
        "coverage": manifest_info["coverage"],
        "owners": manifest["owner_layout"]["owners"],
        "events": events,
    }
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = {
        "canonical": Path(f"{output_prefix}.canonical.json"),
        "perfetto": Path(f"{output_prefix}.perfetto.json"),
        "summary": Path(f"{output_prefix}.summary.json"),
        "receipt": Path(f"{output_prefix}.receipt.json"),
    }
    outputs["canonical"].write_text(
        json.dumps(canonical, indent=2) + "\n", encoding="utf-8"
    )
    outputs["perfetto"].write_text(
        json.dumps(perfetto, indent=2) + "\n", encoding="utf-8"
    )
    outputs["summary"].write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "validation": "accepted",
        "inputs": {
            "raw": str(raw_path.resolve()),
            "manifest": str(manifest_path.resolve()),
            "event_dictionary": str(dictionary_path.resolve()),
            "correctness": str(manifest_info["correctness_path"].resolve()),
        },
        "identity": {
            "kernel_name": manifest["kernel_name"],
            "workload": manifest.get("workload_identity"),
            "device": manifest.get("device_identity"),
            "runtime": manifest.get("runtime_identity"),
            "launch_id": manifest["launch_id"],
            "capture_mode": manifest["capture_mode"],
        },
        "validated_invariants": [
            "ABI, dimensions, launch, capacity, and status",
            "one expected writer claim per explicitly declared owner",
            "no committed record after a hole",
            "known site, allowed owner, and matching event kind",
            "monotonic owner-local timestamps",
            "balanced ranges and first-record owner-local origins",
            "declared analysis-window and all-block coverage contracts when enabled",
            "globaltimer source and nanosecond unit match the PPU timer contract",
            "numerical correctness evidence matches kernel, workload, and device",
        ],
        "outputs": {
            name: str(path.resolve())
            for name, path in outputs.items()
            if name != "receipt"
        },
    }
    outputs["receipt"].write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _parse_command(value: str) -> list[str]:
    try:
        command = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(
            f"command must be a JSON argv array: {error}"
        ) from error
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(item, str) and item for item in command)
    ):
        raise argparse.ArgumentTypeError(
            "command must be a non-empty JSON array of strings"
        )
    return command


def _run_sample(
    command: list[str],
    label: str,
    workload: str,
    warmup: int,
    iterations: int,
    timeout: float,
) -> dict[str, Any]:
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=timeout, check=False
    )
    if result.returncode:
        raise TimelineError(
            f"{label} command exited with {result.returncode}: {result.stderr.strip()}"
        )
    lines = (result.stdout + "\n" + result.stderr).splitlines()
    sample_lines = [
        line[len(SAMPLE_PREFIX) :] for line in lines if line.startswith(SAMPLE_PREFIX)
    ]
    _require(
        len(sample_lines) == 1,
        f"{label} command must emit exactly one {SAMPLE_PREFIX} line",
    )
    try:
        sample = json.loads(sample_lines[0])
    except json.JSONDecodeError as error:
        raise TimelineError(f"{label} emitted invalid sample JSON: {error}") from error
    _require(isinstance(sample, dict), f"{label} sample must be an object")
    latency = sample.get("latency_ms")
    _require(
        isinstance(latency, (int, float))
        and not isinstance(latency, bool)
        and math.isfinite(latency)
        and latency > 0,
        f"{label} latency_ms must be positive and finite",
    )
    _require(sample.get("correctness") == "passed", f"{label} correctness did not pass")
    _require(sample.get("synchronized") is True, f"{label} timing was not synchronized")
    _require(
        sample.get("workload_identity") == workload,
        f"{label} workload identity drifted",
    )
    _require(sample.get("warmup") == warmup, f"{label} warmup count drifted")
    _require(sample.get("iterations") == iterations, f"{label} iteration count drifted")
    _require(
        isinstance(sample.get("device_identity"), dict),
        f"{label} needs device_identity",
    )
    return sample


def measure(
    baseline_command: list[str],
    instrumented_command: list[str],
    workload: str,
    warmup: int,
    iterations: int,
    schedule: list[str],
    timeout: float,
    output: Path,
) -> dict[str, Any]:
    commands = {
        "A": ("baseline", baseline_command),
        "B": ("instrumented", instrumented_command),
    }
    runs: list[dict[str, Any]] = []
    device_identity: dict[str, Any] | None = None
    for sequence in schedule:
        _require(
            sequence and set(sequence) <= {"A", "B"},
            f"invalid schedule sequence {sequence!r}",
        )
        for arm in sequence:
            label, command = commands[arm]
            sample = _run_sample(command, label, workload, warmup, iterations, timeout)
            if device_identity is None:
                device_identity = sample["device_identity"]
            _require(
                sample["device_identity"] == device_identity,
                "device identity drifted across samples",
            )
            runs.append(
                {"order": len(runs), "arm": arm, "variant": label, "sample": sample}
            )
    by_arm = {
        arm: [float(run["sample"]["latency_ms"]) for run in runs if run["arm"] == arm]
        for arm in ("A", "B")
    }
    _require(
        all(by_arm.values()), "schedule must include baseline and instrumented samples"
    )
    baseline_median = statistics.median(by_arm["A"])
    instrumented_median = statistics.median(by_arm["B"])
    result = {
        "schema": MEASUREMENT_SCHEMA,
        "schedule": schedule,
        "fresh_process_per_sample": True,
        "workload_identity": workload,
        "device_identity": device_identity,
        "warmup": warmup,
        "iterations": iterations,
        "runs": runs,
        "summary": {
            "baseline_median_ms": baseline_median,
            "instrumented_median_ms": instrumented_median,
            "relative_overhead": instrumented_median / baseline_median - 1.0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _decode_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strictly decode a PPU fixed-slot raw timeline"
    )
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--event-dictionary", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser


def decode_main(argv: Sequence[str] | None = None) -> int:
    args = _decode_parser().parse_args(argv)
    summary = decode(args.raw, args.manifest, args.event_dictionary, args.output_prefix)
    print(
        f"PPU timeline accepted: {summary['record_count']} records, {summary['owner_count']} owners"
    )
    return 0


def _measure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure PPU timeline perturbation in fresh processes"
    )
    parser.add_argument("--baseline-command", type=_parse_command, required=True)
    parser.add_argument("--instrumented-command", type=_parse_command, required=True)
    parser.add_argument("--workload-identity", required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--schedule", default="ABBA,BAAB")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def measure_main(argv: Sequence[str] | None = None) -> int:
    args = _measure_parser().parse_args(argv)
    _require(args.warmup >= 0 and args.iterations > 0, "warmup/iterations are invalid")
    schedule = [
        item.strip().upper() for item in args.schedule.split(",") if item.strip()
    ]
    result = measure(
        args.baseline_command,
        args.instrumented_command,
        args.workload_identity,
        args.warmup,
        args.iterations,
        schedule,
        args.timeout,
        args.output,
    )
    print(
        f"PPU timeline relative overhead: {result['summary']['relative_overhead']:.2%}"
    )
    return 0


def decode_entrypoint() -> None:
    raise SystemExit(decode_main())


def measure_entrypoint() -> None:
    raise SystemExit(measure_main())


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in {"decode", "measure"}:
        print("usage: timeline.py {decode|measure} ...", file=sys.stderr)
        return 2
    command, command_args = args[0], args[1:]
    if command == "decode":
        return decode_main(command_args)
    return measure_main(command_args)


if __name__ == "__main__":
    raise SystemExit(main())
