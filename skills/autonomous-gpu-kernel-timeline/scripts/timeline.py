#!/usr/bin/env python3
"""Validate and export autonomous GPU timeline evidence.

The tool is intentionally capture-agnostic: AKA keeps the candidate's real launch path and gives
this program the resulting byte buffer plus the manifest and event dictionary used for that launch.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
import shutil
import statistics
import struct
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


MAGIC = 0x31544C5845525441
ABI_MAJOR = 1
ABI_MINOR = 0
HEADER = struct.Struct("<Q4H10IQ")
RECORD = struct.Struct("<QII")
COMMITTED_FLAG = 0x8
STATUS_NAMES = {
    1 << 0: "overflow",
    1 << 1: "bad_header",
    1 << 2: "bad_owner",
    1 << 3: "bad_sm",
    1 << 4: "duplicate_owner",
}
KIND_NAMES = {0: "begin", 1: "end", 2: "instant", 3: "counter"}
MANIFEST_SCHEMA = "atrex-autonomous-timeline-manifest/v1"
DICTIONARY_SCHEMA = "atrex-autonomous-timeline-events/v1"
RECEIPT_SCHEMA = "atrex-autonomous-timeline-receipt/v1"
SUMMARY_SCHEMA = "atrex-autonomous-timeline-summary/v1"
IKET_MANIFEST_SCHEMA = "atrex-autonomous-iket-manifest/v1"
IKET_NATIVE_INDEX_SCHEMA = "atrex-autonomous-iket-native-index/v1"
BINARY_IDENTITY_SCHEMA = "atrex-autonomous-binary-identity/v1"
MEASUREMENT_SCHEMA = "atrex-autonomous-timeline-measurement/v1"
MEASUREMENT_SAMPLE_PREFIX = "__ATREX_TIMELINE_SAMPLE__="


class TimelineError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TimelineError(message)


def digest(path: Path) -> str:
    if path.is_dir():
        value = hashlib.sha256()
        files = sorted(item for item in path.rglob("*") if item.is_file() or item.is_symlink())
        require(files, f"source snapshot directory is empty: {path}")
        for item in files:
            require(not item.is_symlink(), f"source snapshot contains a symlink: {item}")
            relative = item.relative_to(path).as_posix().encode("utf-8")
            value.update(len(relative).to_bytes(8, "little"))
            value.update(relative)
            value.update(bytes.fromhex(digest(item)))
        return value.hexdigest()
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TimelineError(f"cannot read {label} {path}: {error}") from error
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def load_gzip_object(path: Path, label: str) -> dict[str, Any]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TimelineError(f"cannot read {label} {path}: {error}") from error
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def evaluator_source(instrumented_source: Path) -> Path:
    return instrumented_source / "kernel.py" if instrumented_source.is_dir() else instrumented_source


def load_evaluator_correctness(path: Path, instrumented_source: Path) -> dict[str, Any]:
    """Select the latest sandbox evaluator record for the instrumented kernel."""

    source = evaluator_source(instrumented_source)
    require(source.is_file(), f"instrumented evaluator source does not exist: {source}")
    source_digest = digest(source)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise TimelineError(f"cannot read evaluator correctness evidence {path}: {error}") from error
    require(lines, f"evaluator correctness evidence is empty: {path}")
    selected: dict[str, Any] | None = None
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise TimelineError(
                f"invalid evaluator correctness record at {path}:{line_number}: {error}"
            ) from error
        require(isinstance(record, dict),
                f"evaluator correctness record {line_number} must be an object")
        if record.get("kernel_sha256") == source_digest:
            selected = record
    require(selected is not None,
            "evaluator correctness evidence has no result for the instrumented source")
    require(
        isinstance(selected.get("schema_version"), int) and selected["schema_version"] >= 2,
        "evaluator correctness record predates source-bound sandbox results",
    )
    result = selected.get("result")
    require(isinstance(result, dict), "evaluator correctness record has no result object")
    require(isinstance(result.get("all_pass"), bool),
            "evaluator correctness result has no boolean all_pass")
    failures = result.get("failures", [])
    require(isinstance(failures, list), "evaluator correctness failures must be a list")
    require(result["all_pass"] == (len(failures) == 0),
            "evaluator correctness all_pass contradicts failures")
    return selected


def correctness_from_evidence(
    evidence: object, instrumented_source_sha256: str
) -> tuple[str, bool]:
    if evidence is None:
        return "unknown", False
    require(isinstance(evidence, dict), "correctness_evidence must be an object")
    require(
        isinstance(evidence.get("schema_version"), int) and evidence["schema_version"] >= 2,
        "correctness evidence predates source-bound sandbox results",
    )
    require(evidence.get("kernel_sha256") == instrumented_source_sha256,
            "correctness evidence does not match instrumented source")
    result = evidence.get("result")
    require(isinstance(result, dict), "correctness evidence has no result object")
    all_pass = result.get("all_pass")
    require(isinstance(all_pass, bool), "correctness evidence has no boolean all_pass")
    failures = result.get("failures", [])
    require(isinstance(failures, list), "correctness evidence failures must be a list")
    require(all_pass == (len(failures) == 0),
            "correctness evidence all_pass contradicts failures")
    return ("passed" if all_pass else "failed"), True


def triple(value: object, label: str) -> tuple[int, int, int]:
    require(isinstance(value, list) and len(value) == 3, f"{label} must be [x, y, z]")
    result = tuple(int(item) for item in value)
    require(all(item > 0 for item in result), f"{label} dimensions must be positive")
    return result  # type: ignore[return-value]


def parse_header(raw: bytes) -> dict[str, int]:
    require(len(raw) >= HEADER.size, "raw buffer is smaller than the 64-byte header")
    fields = HEADER.unpack_from(raw)
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
    header = dict(zip(names, fields, strict=True))
    require(header["magic"] == MAGIC, "raw buffer magic does not identify the ATREX timeline ABI")
    require(header["abi_major"] == ABI_MAJOR, f"unsupported ABI major {header['abi_major']}")
    require(header["abi_minor"] == ABI_MINOR, f"unsupported ABI minor {header['abi_minor']}")
    require(header["header_bytes"] == HEADER.size, "header size does not match ABI v1")
    require(header["record_bytes"] == RECORD.size, "record size does not match ABI v1")
    require(header["owner_count"] > 0, "owner_count must be positive")
    require(header["records_per_owner"] > 0, "records_per_owner must be positive")
    require(
        header["capacity"] == header["owner_count"] * header["records_per_owner"],
        "capacity must equal owner_count * records_per_owner",
    )
    expected_bytes = (
        header["header_bytes"]
        + header["capacity"] * header["record_bytes"]
        + header["owner_count"] * 8
    )
    require(len(raw) == expected_bytes, f"raw buffer has {len(raw)} bytes; expected {expected_bytes}")
    if header["status"]:
        known = [name for bit, name in STATUS_NAMES.items() if header["status"] & bit]
        unknown = header["status"] & ~sum(STATUS_NAMES)
        if unknown:
            known.append(f"unknown_status_0x{unknown:x}")
        raise TimelineError("device reported invalid capture status: " + ", ".join(known))
    return header


def load_sites(dictionary: dict[str, Any]) -> dict[int, dict[str, Any]]:
    require(dictionary.get("schema") == DICTIONARY_SCHEMA, "unknown event dictionary schema")
    raw_sites = dictionary.get("sites")
    require(isinstance(raw_sites, list) and raw_sites, "event dictionary sites must be non-empty")
    sites: dict[int, dict[str, Any]] = {}
    names: set[str] = set()
    for position, raw_site in enumerate(raw_sites):
        require(isinstance(raw_site, dict), f"sites[{position}] must be an object")
        site_id = raw_site.get("site_id")
        name = raw_site.get("name")
        kind = raw_site.get("kind", "any")
        require(isinstance(site_id, int) and 0 <= site_id <= 0xFFFF, "site_id must fit uint16")
        require(isinstance(name, str) and name.strip(), f"site {site_id} has no name")
        require(
            kind in {"any", "range", "instant", "counter"},
            f"site {site_id} has an invalid declared kind",
        )
        require(site_id not in sites, f"duplicate site_id {site_id}")
        require(name not in names, f"duplicate site name {name!r}")
        sites[site_id] = raw_site
        names.add(name)
    return sites


def validate_manifest(manifest: dict[str, Any], header: dict[str, int]) -> list[dict[str, Any]]:
    require(manifest.get("schema") == MANIFEST_SCHEMA, "unknown manifest schema")
    require(manifest.get("backend") == "cuda", "manifest backend must be cuda")
    grid = triple(manifest.get("grid"), "manifest.grid")
    block = triple(manifest.get("block"), "manifest.block")
    require(grid == (header["grid_x"], header["grid_y"], header["grid_z"]), "grid mismatch")
    require(
        block == (header["block_x"], header["block_y"], header["block_z"]), "block mismatch"
    )
    require(manifest.get("launch_id") == header["launch_id"], "launch_id mismatch")
    require(
        manifest.get("records_per_owner") == header["records_per_owner"],
        "records_per_owner mismatch",
    )
    layout = manifest.get("owner_layout")
    require(isinstance(layout, dict), "manifest.owner_layout must be an object")
    require(layout.get("kind") == "cta_writers", "owner layout must be cta_writers")
    writers = layout.get("writers")
    require(isinstance(writers, list) and writers, "owner layout writers must be non-empty")
    seen_labels: set[str] = set()
    for ordinal, writer in enumerate(writers):
        require(isinstance(writer, dict), f"writer {ordinal} must be an object")
        require(writer.get("ordinal") == ordinal, "writer ordinals must be dense and ordered")
        label = writer.get("label")
        require(isinstance(label, str) and label.strip(), f"writer {ordinal} needs a label")
        require(label not in seen_labels, f"duplicate writer label {label!r}")
        seen_labels.add(label)
        warp = writer.get("warp")
        thread = writer.get("thread")
        lane = writer.get("lane", 0)
        require(
            (isinstance(warp, int) and not isinstance(warp, bool) and warp >= 0)
            or (isinstance(thread, int) and not isinstance(thread, bool) and thread >= 0),
            f"writer {ordinal} must declare warp or thread",
        )
        require(
            isinstance(lane, int) and not isinstance(lane, bool) and 0 <= lane < 32,
            f"writer {ordinal} lane is invalid",
        )
        block_threads = math.prod(block)
        expected_thread = thread if thread is not None else int(warp) * 32 + lane
        require(expected_thread < block_threads, f"writer {ordinal} thread is outside the block")
        if warp is not None and thread is not None:
            require(
                thread == warp * 32 + lane,
                f"writer {ordinal} thread does not match warp/lane",
            )
    expected_owners = math.prod(grid) * len(writers)
    require(header["owner_count"] == expected_owners, "owner_count does not match grid and writers")
    return writers


def owner_identity(
    owner: int, header: dict[str, int], writers: list[dict[str, Any]]
) -> dict[str, Any]:
    writers_per_cta = len(writers)
    cta_linear, ordinal = divmod(owner, writers_per_cta)
    gx, gy = header["grid_x"], header["grid_y"]
    x = cta_linear % gx
    y = (cta_linear // gx) % gy
    z = cta_linear // (gx * gy)
    writer = writers[ordinal]
    warp = writer.get("warp")
    if warp is None:
        warp = int(writer["thread"]) // 32
    return {
        "owner": owner,
        "cta_linear": cta_linear,
        "cta": [x, y, z],
        "writer_ordinal": ordinal,
        "writer_label": writer["label"],
        "warp": warp,
        "group": writer.get("group"),
    }


def decode_records(
    raw: bytes,
    header: dict[str, int],
    writers: list[dict[str, Any]],
    sites: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    sm_by_owner: dict[int, int] = {}
    claims_offset = header["header_bytes"] + header["capacity"] * header["record_bytes"]
    for owner in range(header["owner_count"]):
        identity = owner_identity(owner, header, writers)
        (claim,) = struct.unpack_from("<Q", raw, claims_offset + owner * 8)
        if claim:
            claimed_cta = ((claim >> 32) & 0xFFFFFFFF) - 1
            claimed_thread = (claim & 0xFFFFFFFF) - 1
            if claimed_cta != identity["cta_linear"]:
                errors.append(
                    f"owner {owner} claims CTA {claimed_cta}, expected {identity['cta_linear']}"
                )
            writer = writers[identity["writer_ordinal"]]
            expected_thread = writer.get("thread")
            if expected_thread is None:
                expected_thread = int(writer["warp"]) * 32 + int(writer.get("lane", 0))
            if claimed_thread != expected_thread:
                errors.append(
                    f"owner {owner} claims thread {claimed_thread}, expected {expected_thread}"
                )
        saw_empty = False
        previous_timestamp = -1
        for sequence in range(header["records_per_owner"]):
            slot = owner * header["records_per_owner"] + sequence
            offset = header["header_bytes"] + slot * header["record_bytes"]
            timestamp, payload, tag = RECORD.unpack_from(raw, offset)
            flags = (tag >> 28) & 0xF
            committed = bool(flags & COMMITTED_FLAG)
            if not committed:
                if timestamp or payload or tag:
                    errors.append(f"slot {slot} is nonzero without the committed flag")
                saw_empty = True
                continue
            if claim == 0:
                errors.append(f"owner {owner} has records but no writer claim")
            if saw_empty:
                errors.append(f"owner {owner} has a committed record after an empty sequence slot")
            site_id = tag & 0xFFFF
            kind_value = (tag >> 16) & 0x3
            sm = (tag >> 18) & 0x3FF
            if site_id not in sites:
                errors.append(f"slot {slot} references unknown site_id {site_id}")
                continue
            declared_kind = sites[site_id].get("kind", "any")
            observed_kind = KIND_NAMES[kind_value]
            if declared_kind == "range" and observed_kind not in {"begin", "end"}:
                errors.append(
                    f"slot {slot} kind {observed_kind} does not match range site {site_id}"
                )
            elif declared_kind not in {"any", "range", observed_kind}:
                errors.append(
                    f"slot {slot} kind {observed_kind} does not match site {site_id}"
                )
            if timestamp == 0:
                errors.append(f"slot {slot} has a zero timestamp")
            if timestamp < previous_timestamp:
                errors.append(f"owner {owner} timestamps decrease at sequence {sequence}")
            previous_timestamp = timestamp
            if owner in sm_by_owner and sm_by_owner[owner] != sm:
                errors.append(f"owner {owner} changes physical SM within one launch")
            sm_by_owner.setdefault(owner, sm)
            records.append(
                {
                    **identity,
                    "sequence": sequence,
                    "timestamp_ns": timestamp,
                    "payload": payload,
                    "site_id": site_id,
                    "site": sites[site_id],
                    "kind": observed_kind,
                    "physical_sm": sm,
                    "flags": flags & ~COMMITTED_FLAG,
                }
            )
    if errors:
        raise TimelineError("; ".join(errors))
    return records, pair_ranges(records)


def pair_ranges(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stacks: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    canonical: list[dict[str, Any]] = []
    errors: list[str] = []
    for record in records:
        key = (record["owner"], record["site_id"])
        if record["kind"] == "begin":
            stacks[key].append(record)
        elif record["kind"] == "end":
            if not stacks[key]:
                errors.append(
                    f"owner {record['owner']} site {record['site_id']} ends without a begin"
                )
                continue
            begin = stacks[key].pop()
            canonical.append(
                {
                    **{key: begin[key] for key in (
                        "owner",
                        "cta_linear",
                        "cta",
                        "writer_ordinal",
                        "writer_label",
                        "warp",
                        "group",
                        "site_id",
                        "site",
                        "physical_sm",
                    )},
                    "kind": "range",
                    "begin_sequence": begin["sequence"],
                    "end_sequence": record["sequence"],
                    "timestamp_ns": begin["timestamp_ns"],
                    "end_timestamp_ns": record["timestamp_ns"],
                    "duration_ns": record["timestamp_ns"] - begin["timestamp_ns"],
                    "begin_payload": begin["payload"],
                    "end_payload": record["payload"],
                }
            )
        else:
            canonical.append(record)
    for (owner, site_id), pending in stacks.items():
        if pending:
            errors.append(f"owner {owner} site {site_id} has {len(pending)} unmatched begin event(s)")
    if errors:
        raise TimelineError("; ".join(errors))
    canonical.sort(key=lambda item: (item["timestamp_ns"], item["owner"], item.get("sequence", -1)))
    return canonical


def event_args(event: dict[str, Any]) -> dict[str, Any]:
    site = event["site"]
    args = {
        "site_id": event["site_id"],
        "cta": event["cta"],
        "cta_linear": event["cta_linear"],
        "writer": event["writer_label"],
        "writer_ordinal": event["writer_ordinal"],
        "warp": event["warp"],
        "group": event["group"],
        "physical_sm": event["physical_sm"],
        "timestamp_ns": event["timestamp_ns"],
        "boundary_semantics": site.get("boundary_semantics"),
        "async_domain": site.get("async_domain"),
        "source_anchor": site.get("source_anchor"),
    }
    return {key: value for key, value in args.items() if value is not None}


def make_perfetto(canonical: list[dict[str, Any]]) -> dict[str, Any]:
    require(canonical, "capture contains no committed events")
    origin = min(event["timestamp_ns"] for event in canonical)
    trace_events: list[dict[str, Any]] = []
    owner_metadata: set[tuple[int, int]] = set()
    sm_ids = sorted({event["physical_sm"] for event in canonical})
    for sm in sm_ids:
        trace_events.append(
            {"name": "process_name", "ph": "M", "pid": sm + 1, "tid": 0, "args": {"name": f"SM {sm}"}}
        )
    for event in canonical:
        pid = event["physical_sm"] + 1
        tid = event["owner"] + 1
        metadata_key = (pid, tid)
        if metadata_key not in owner_metadata:
            cta = ",".join(str(value) for value in event["cta"])
            trace_events.append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": pid,
                    "tid": tid,
                    "args": {"name": f"cta({cta})/{event['writer_label']}"},
                }
            )
            owner_metadata.add(metadata_key)
        base = {
            "name": event["site"]["name"],
            "pid": pid,
            "tid": tid,
            "ts": (event["timestamp_ns"] - origin) / 1000.0,
            "args": event_args(event),
        }
        if event["kind"] == "range":
            base.update(
                {
                    "ph": "X",
                    "dur": event["duration_ns"] / 1000.0,
                    "args": {
                        **base["args"],
                        "end_timestamp_ns": event["end_timestamp_ns"],
                        "duration_ns": event["duration_ns"],
                        "begin_payload": event["begin_payload"],
                        "end_payload": event["end_payload"],
                    },
                }
            )
        elif event["kind"] == "instant":
            base.update({"ph": "i", "s": "t"})
            base["args"]["payload"] = event["payload"]
            base["args"]["sequence"] = event["sequence"]
        elif event["kind"] == "counter":
            base.update({"ph": "C"})
            base["args"]["value"] = event["payload"]
            base["args"]["sequence"] = event["sequence"]
        else:
            raise TimelineError(f"unpaired range event {event['kind']}")
        trace_events.append(base)
    return {
        "displayTimeUnit": "ns",
        "otherData": {"globaltimer_origin_ns": origin, "clock": "%globaltimer"},
        "traceEvents": trace_events,
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_gzip_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    with path.open("wb") as destination:
        with gzip.GzipFile(filename="", mode="wb", fileobj=destination, mtime=0) as compressed:
            compressed.write(payload)


def summarize(canonical: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    ranges = [event for event in canonical if event["kind"] == "range"]
    site_duration: dict[str, int] = defaultdict(int)
    site_count: dict[str, int] = defaultdict(int)
    for event in canonical:
        name = event["site"]["name"]
        site_count[name] += 1
        if event["kind"] == "range":
            site_duration[name] += event["duration_ns"]
    return {
        "schema": SUMMARY_SCHEMA,
        "committed_records": len(records),
        "canonical_events": len(canonical),
        "ranges": len(ranges),
        "instants": sum(event["kind"] == "instant" for event in canonical),
        "counters": sum(event["kind"] == "counter" for event in canonical),
        "owners_observed": len({event["owner"] for event in canonical}),
        "physical_sms": sorted({event["physical_sm"] for event in canonical}),
        "site_event_count": dict(sorted(site_count.items())),
        "site_total_duration_ns": dict(sorted(site_duration.items())),
    }


def finite_samples(value: object, field: str) -> list[float]:
    require(isinstance(value, list) and value, f"measurement {field} must be non-empty")
    result = [float(item) for item in value]
    require(all(math.isfinite(item) and item > 0 for item in result), f"measurement {field} is invalid")
    return result


def measurement_summary(path: Path) -> dict[str, Any]:
    measurement = load_object(path, "measurement")
    require(measurement.get("schema") == MEASUREMENT_SCHEMA, "unknown measurement schema")
    schedule = measurement.get("schedule")
    require(
        isinstance(schedule, list)
        and schedule
        and all(item in {"baseline", "instrumented"} for item in schedule),
        "measurement schedule is invalid",
    )
    require(len(schedule) % 4 == 0, "measurement schedule must contain complete ABBA/BAAB blocks")
    for offset in range(0, len(schedule), 4):
        require(
            schedule[offset : offset + 4]
            in (
                ["baseline", "instrumented", "instrumented", "baseline"],
                ["instrumented", "baseline", "baseline", "instrumented"],
            ),
            "measurement schedule must use ABBA or BAAB blocks",
        )
    blocks = [schedule[offset : offset + 4] for offset in range(0, len(schedule), 4)]
    require(
        blocks.count(["baseline", "instrumented", "instrumented", "baseline"])
        == blocks.count(["instrumented", "baseline", "baseline", "instrumented"])
        and len(blocks) >= 2,
        "measurement must contain equally many ABBA and BAAB blocks",
    )
    runs = measurement.get("runs")
    require(isinstance(runs, list) and len(runs) == len(schedule), "measurement runs do not match schedule")
    baseline: list[float] = []
    instrumented: list[float] = []
    for index, (variant, run) in enumerate(zip(schedule, runs, strict=True)):
        require(isinstance(run, dict), f"measurement run {index} must be an object")
        require(run.get("ordinal") == index, f"measurement run {index} has an invalid ordinal")
        require(run.get("variant") == variant, f"measurement run {index} does not match schedule")
        samples = finite_samples([run.get("latency_ms")], f"runs[{index}].latency_ms")
        (baseline if variant == "baseline" else instrumented).extend(samples)
    require(len(baseline) == len(instrumented), "paired measurement sample counts differ")
    warmup = measurement.get("warmup")
    iterations = measurement.get("iterations")
    require(isinstance(warmup, int) and not isinstance(warmup, bool) and warmup >= 0,
            "measurement warmup is invalid")
    require(isinstance(iterations, int) and not isinstance(iterations, bool) and iterations > 0,
            "measurement iterations is invalid")
    require(measurement.get("synchronized") is True, "measurement must be device synchronized")
    workload_identity = measurement.get("workload_identity")
    require(isinstance(workload_identity, str) and workload_identity.strip(),
            "measurement workload_identity is missing")
    device_identity = measurement.get("device_identity")
    require(isinstance(device_identity, dict) and device_identity,
            "measurement device_identity is missing")
    variants = measurement.get("variants")
    require(isinstance(variants, dict), "measurement variants are missing")
    variant_hashes: dict[str, dict[str, str]] = {}
    for variant in ("baseline", "instrumented"):
        identity = variants.get(variant)
        require(isinstance(identity, dict), f"measurement {variant} identity is missing")
        hashes: dict[str, str] = {}
        for artifact in ("source", "binary"):
            relative = identity.get(artifact)
            expected = identity.get(f"{artifact}_sha256")
            if artifact == "binary" and relative is None and expected is None:
                continue
            require(isinstance(relative, str) and relative, f"measurement {variant} {artifact} is missing")
            require(isinstance(expected, str) and len(expected) == 64,
                    f"measurement {variant} {artifact} hash is missing")
            artifact_path = Path(relative)
            if not artifact_path.is_absolute():
                artifact_path = (path.parent / artifact_path).resolve()
            if artifact == "source":
                require(artifact_path.exists(),
                        f"measurement {variant} {artifact} is missing: {artifact_path}")
            else:
                require(artifact_path.is_file(),
                        f"measurement {variant} {artifact} is missing: {artifact_path}")
            require(digest(artifact_path) == expected,
                    f"measurement {variant} {artifact} hash mismatch")
            hashes[f"{artifact}_sha256"] = expected
        variant_hashes[variant] = hashes
    aggregation = measurement.get("aggregation")
    require(
        aggregation == {
            "center": "median",
            "overhead_percent": "(instrumented_ms / baseline_ms - 1) * 100",
        },
        "measurement aggregation formula is invalid",
    )
    baseline_ms = statistics.median(baseline)
    instrumented_ms = statistics.median(instrumented)
    overhead = (instrumented_ms / baseline_ms - 1.0) * 100.0
    return {
        "path": str(path.resolve()),
        "sha256": digest(path),
        "baseline_ms": baseline_ms,
        "instrumented_ms": instrumented_ms,
        "overhead_percent": overhead,
        "workload_identity": workload_identity,
        "device_identity": device_identity,
        "variants": variant_hashes,
    }


def command_argv(value: str, label: str) -> list[str]:
    try:
        command = json.loads(value)
    except json.JSONDecodeError as error:
        raise TimelineError(f"{label} must be a JSON argv array: {error}") from error
    require(
        isinstance(command, list)
        and command
        and all(isinstance(item, str) and item for item in command),
        f"{label} must be a non-empty JSON array of strings",
    )
    return command


def measurement_sample(stdout: str, label: str) -> dict[str, Any]:
    matches = [
        line[len(MEASUREMENT_SAMPLE_PREFIX) :]
        for line in stdout.splitlines()
        if line.startswith(MEASUREMENT_SAMPLE_PREFIX)
    ]
    require(len(matches) == 1, f"{label} must emit exactly one timeline sample record")
    try:
        sample = json.loads(matches[0])
    except json.JSONDecodeError as error:
        raise TimelineError(f"{label} emitted an invalid timeline sample: {error}") from error
    require(isinstance(sample, dict), f"{label} timeline sample must be an object")
    return sample


def measure_command(arguments: argparse.Namespace) -> int:
    output = arguments.output.resolve()
    require(not output.exists(), f"measurement output already exists: {output}")
    require(arguments.warmup >= 0, "measurement warmup must be non-negative")
    require(arguments.iterations > 0, "measurement iterations must be positive")
    require(arguments.timeout > 0, "measurement timeout must be positive")
    require(arguments.workload_identity.strip(), "measurement workload identity is empty")
    baseline_command = command_argv(arguments.baseline_command, "baseline command")
    instrumented_command = command_argv(arguments.instrumented_command, "instrumented command")
    commands = {"baseline": baseline_command, "instrumented": instrumented_command}
    sources = {
        "baseline": arguments.baseline_source.resolve(),
        "instrumented": arguments.instrumented_source.resolve(),
    }
    binaries = {
        "baseline": arguments.baseline_binary.resolve() if arguments.baseline_binary else None,
        "instrumented": (
            arguments.instrumented_binary.resolve() if arguments.instrumented_binary else None
        ),
    }
    source_hashes: dict[str, str] = {}
    for variant in ("baseline", "instrumented"):
        require(sources[variant].exists(), f"{variant} source is missing: {sources[variant]}")
        source_hashes[variant] = digest(sources[variant])
    orders = arguments.order or ["ABBA", "BAAB"]
    require(
        orders.count("ABBA") == orders.count("BAAB") and len(orders) >= 2,
        "measurement requires equally many ABBA and BAAB blocks",
    )
    schedule = [
        "baseline" if symbol == "A" else "instrumented"
        for order in orders
        for symbol in order
    ]
    runs: list[dict[str, Any]] = []
    device_identity: dict[str, Any] | None = None
    run_cwd = arguments.cwd.resolve() if arguments.cwd else Path.cwd().resolve()
    require(run_cwd.is_dir(), f"measurement cwd is missing: {run_cwd}")
    for ordinal, variant in enumerate(schedule):
        environment = os.environ.copy()
        environment.update(
            {
                "ATREX_TIMELINE_VARIANT": variant,
                "ATREX_TIMELINE_WARMUP": str(arguments.warmup),
                "ATREX_TIMELINE_ITERATIONS": str(arguments.iterations),
            }
        )
        try:
            completed = subprocess.run(
                commands[variant],
                cwd=run_cwd,
                env=environment,
                text=True,
                capture_output=True,
                timeout=arguments.timeout,
            )
        except subprocess.TimeoutExpired as error:
            raise TimelineError(f"measurement run {ordinal} ({variant}) timed out") from error
        require(
            completed.returncode == 0,
            f"measurement run {ordinal} ({variant}) failed with exit code "
            f"{completed.returncode}: {completed.stderr[-1000:]}",
        )
        sample = measurement_sample(completed.stdout, f"measurement run {ordinal} ({variant})")
        latency = finite_samples([sample.get("latency_ms")], f"runs[{ordinal}].latency_ms")[0]
        require(sample.get("correctness") == "passed",
                f"measurement run {ordinal} ({variant}) did not pass correctness")
        require(sample.get("synchronized") is True,
                f"measurement run {ordinal} ({variant}) was not device synchronized")
        require(sample.get("workload_identity") == arguments.workload_identity,
                f"measurement run {ordinal} ({variant}) changed workload identity")
        require(sample.get("warmup") == arguments.warmup,
                f"measurement run {ordinal} ({variant}) changed warmup")
        require(sample.get("iterations") == arguments.iterations,
                f"measurement run {ordinal} ({variant}) changed iterations")
        current_device = sample.get("device_identity")
        require(isinstance(current_device, dict) and current_device,
                f"measurement run {ordinal} ({variant}) has no device identity")
        if device_identity is None:
            device_identity = current_device
        require(current_device == device_identity,
                f"measurement run {ordinal} ({variant}) changed device identity")
        runs.append({"ordinal": ordinal, "variant": variant, "latency_ms": latency})
    for variant in ("baseline", "instrumented"):
        require(digest(sources[variant]) == source_hashes[variant],
                f"{variant} source changed during measurement")
        if binaries[variant] is not None:
            require(binaries[variant].is_file(), f"{variant} binary is missing: {binaries[variant]}")
    output.parent.mkdir(parents=True, exist_ok=True)
    variants = {
        variant: {
            "command": commands[variant],
            "source": relative_artifact(sources[variant], output.parent),
            "source_sha256": source_hashes[variant],
            "binary": (
                relative_artifact(binaries[variant], output.parent) if binaries[variant] else None
            ),
            "binary_sha256": digest(binaries[variant]) if binaries[variant] else None,
        }
        for variant in ("baseline", "instrumented")
    }
    write_json(
        output,
        {
            "schema": MEASUREMENT_SCHEMA,
            "schedule": schedule,
            "runs": runs,
            "warmup": arguments.warmup,
            "iterations": arguments.iterations,
            "synchronized": True,
            "workload_identity": arguments.workload_identity,
            "device_identity": device_identity,
            "variants": variants,
            "aggregation": {
                "center": "median",
                "overhead_percent": "(instrumented_ms / baseline_ms - 1) * 100",
            },
        },
    )
    summary = measurement_summary(output)
    print(json.dumps({"measurement": str(output), **summary}, sort_keys=True))
    return 0


def classify_evidence(
    *,
    stage: str,
    correctness: str,
    correctness_verified: bool,
    has_binary: bool,
    measurement: dict[str, Any] | None,
    low_overhead_target: float,
) -> tuple[str, list[str]]:
    if stage == "exploration":
        reasons = ["stage_is_exploration"]
        if correctness != "passed":
            reasons.append("correctness_not_passed")
        if measurement is None:
            reasons.append("overhead_not_measured")
        elif measurement["overhead_percent"] > low_overhead_target:
            reasons.append("overhead_above_target")
        return "exploration", reasons

    reasons = []
    if correctness != "passed":
        reasons.append("correctness_not_passed")
    elif not correctness_verified:
        reasons.append("correctness_not_verified")
    if not has_binary:
        reasons.append("binary_identity_missing")
    if measurement is None:
        reasons.append("overhead_not_measured")
    elif measurement["overhead_percent"] > low_overhead_target:
        reasons.append("overhead_above_target")
    if reasons:
        return "diagnostic", reasons
    return "decision_grade", [
        "correctness_passed",
        "provenance_bound",
        "trace_integrity_passed",
        "measurement_recomputed",
        "overhead_within_target",
    ]


def relative_artifact(path: Path, receipt_dir: Path) -> str:
    return os.path.relpath(path.resolve(), receipt_dir.resolve())


def write_receipt(
    *,
    output_dir: Path,
    backend: str,
    raw_path: Path,
    manifest_path: Path,
    dictionary_path: Path,
    clean_source: Path,
    instrumented_source: Path,
    binary_path: Path | None,
    workload_identity: str,
    reported_correctness: str | None,
    correctness_evidence_path: Path | None,
    stage: str,
    measurement_path: Path | None,
    low_overhead_target: float,
    trace_path: Path,
    summary_path: Path,
) -> Path:
    require(math.isfinite(low_overhead_target) and low_overhead_target >= 0,
            "low-overhead target must be finite and non-negative")
    require(clean_source.exists(), f"clean source does not exist: {clean_source}")
    require(instrumented_source.exists(),
            f"instrumented source does not exist: {instrumented_source}")
    for path, label in (
        (raw_path, "native/canonical input"),
        (manifest_path, "manifest"),
        (dictionary_path, "event dictionary"),
        (trace_path, "trace"),
        (summary_path, "summary"),
    ):
        require(path.is_file(), f"{label} does not exist: {path}")
    if binary_path is not None:
        require(binary_path.is_file(), f"binary identity does not exist: {binary_path}")
    measurement = measurement_summary(measurement_path) if measurement_path else None
    if measurement is not None:
        require(measurement["workload_identity"] == workload_identity,
                "measurement workload identity does not match receipt")
        require(measurement["variants"]["baseline"]["source_sha256"] == digest(clean_source),
                "measurement baseline source does not match clean source")
        require(
            measurement["variants"]["instrumented"]["source_sha256"]
            == digest(instrumented_source),
            "measurement instrumented source does not match receipt",
        )
    correctness_evidence = (
        load_evaluator_correctness(correctness_evidence_path, instrumented_source)
        if correctness_evidence_path is not None
        else None
    )
    verified_correctness, correctness_verified = correctness_from_evidence(
        correctness_evidence, digest(evaluator_source(instrumented_source))
    )
    if reported_correctness is not None and correctness_verified:
        require(reported_correctness == verified_correctness,
                "reported correctness contradicts evaluator evidence")
    correctness = verified_correctness if correctness_verified else (reported_correctness or "unknown")
    evidence_class, classification_reasons = classify_evidence(
        stage=stage,
        correctness=correctness,
        correctness_verified=correctness_verified,
        has_binary=binary_path is not None,
        measurement=measurement,
        low_overhead_target=low_overhead_target,
    )
    receipt_path = output_dir / "receipt.json"
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "backend": backend,
        "stage": stage,
        "clean_source": relative_artifact(clean_source, output_dir),
        "clean_source_sha256": digest(clean_source),
        "instrumented_source": relative_artifact(instrumented_source, output_dir),
        "instrumented_source_sha256": digest(instrumented_source),
        "binary": relative_artifact(binary_path, output_dir) if binary_path else None,
        "binary_sha256": digest(binary_path) if binary_path else None,
        "raw": relative_artifact(raw_path, output_dir),
        "raw_sha256": digest(raw_path),
        "manifest": relative_artifact(manifest_path, output_dir),
        "manifest_sha256": digest(manifest_path),
        "event_dictionary": relative_artifact(dictionary_path, output_dir),
        "event_dictionary_sha256": digest(dictionary_path),
        "workload_identity": workload_identity,
        "correctness": correctness,
        "correctness_evidence": correctness_evidence,
        "trace_integrity": "passed",
        "trace": relative_artifact(trace_path, output_dir),
        "trace_sha256": digest(trace_path),
        "summary": relative_artifact(summary_path, output_dir),
        "summary_sha256": digest(summary_path),
        "evidence_class": evidence_class,
        "classification_reasons": classification_reasons,
        "low_overhead_target_percent": low_overhead_target,
        "measurement": measurement,
    }
    if measurement is not None:
        measurement["path"] = relative_artifact(measurement_path, output_dir)
    write_json(receipt_path, receipt)
    return receipt_path


def decode_command(arguments: argparse.Namespace) -> int:
    raw_path = arguments.raw.resolve()
    manifest_path = arguments.manifest.resolve()
    dictionary_path = arguments.dictionary.resolve()
    raw = raw_path.read_bytes()
    header = parse_header(raw)
    manifest = load_object(manifest_path, "manifest")
    dictionary = load_object(dictionary_path, "event dictionary")
    sites = load_sites(dictionary)
    writers = validate_manifest(manifest, header)
    records, canonical = decode_records(raw, header, writers, sites)
    perfetto = make_perfetto(canonical)
    output_dir = arguments.output_dir.resolve()
    trace_path = output_dir / "trace.perfetto.json.gz"
    summary_path = output_dir / "trace.summary.json"
    write_gzip_json(trace_path, perfetto)
    summary = summarize(canonical, records)
    write_json(summary_path, summary)

    clean_source = arguments.clean_source.resolve()
    instrumented_source = arguments.instrumented_source.resolve()
    receipt_path = write_receipt(
        output_dir=output_dir,
        backend="cuda",
        raw_path=raw_path,
        manifest_path=manifest_path,
        dictionary_path=dictionary_path,
        clean_source=clean_source,
        instrumented_source=instrumented_source,
        binary_path=arguments.binary.resolve() if arguments.binary else None,
        workload_identity=arguments.workload_identity,
        reported_correctness=arguments.correctness,
        correctness_evidence_path=(
            arguments.correctness_evidence.resolve() if arguments.correctness_evidence else None
        ),
        stage=arguments.stage,
        measurement_path=arguments.measurement.resolve() if arguments.measurement else None,
        low_overhead_target=arguments.low_overhead_target,
        trace_path=trace_path,
        summary_path=summary_path,
    )
    print(json.dumps({"receipt": str(receipt_path), "summary": summary}, sort_keys=True))
    return 0


def iket_adapter() -> Any:
    path = Path(__file__).resolve().parents[1] / "backends" / "cutedsl_backend" / "adapter.py"
    spec = importlib.util.spec_from_file_location("atrex_iket_adapter", path)
    require(spec is not None and spec.loader is not None, f"cannot load IKeT adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capture_iket_command(arguments: argparse.Namespace) -> int:
    executable = shutil.which("run-iket")
    require(executable is not None, "run-iket is unavailable; install an IKeT-enabled CuTe DSL")
    command = list(arguments.target_command)
    if command and command[0] == "--":
        command.pop(0)
    require(command, "capture-iket requires a target command after --")
    output_dir = arguments.output_dir.resolve()
    require(not output_dir.exists(), f"IKeT attempt directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    profiler_command = [
        executable,
        "--output-dir",
        str(output_dir),
        "profile",
        "--keep",
        "--postprocess",
        "all",
        "--",
        *command,
    ]
    completed = subprocess.run(profiler_command, text=True, capture_output=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run-iket.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "run-iket.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    write_json(
        output_dir / "capture-command.json",
        {
            "schema": "atrex-autonomous-iket-capture/v1",
            "profiler": str(Path(executable).resolve()),
            "profiler_sha256": digest(Path(executable).resolve()),
            "target_command": command,
            "exit_code": completed.returncode,
        },
    )
    require(completed.returncode == 0, f"run-iket failed with exit code {completed.returncode}")
    trace_json = [path for path in output_dir.glob("*.trace.json") if path.stat().st_size > 0]
    perfetto = [path for path in output_dir.glob("*.pftrace") if path.stat().st_size > 0]
    require(trace_json, "run-iket produced no non-empty *.trace.json")
    require(perfetto, "run-iket produced no non-empty *.pftrace")
    print(json.dumps({"iket_run": str(output_dir), "trace_json": len(trace_json), "perfetto": len(perfetto)}))
    return 0


def export_iket_command(arguments: argparse.Namespace) -> int:
    adapter = iket_adapter()
    run_dir = arguments.run_dir.resolve()
    require(run_dir.is_dir(), f"IKeT run directory does not exist: {run_dir}")
    native_perfetto = sorted(path for path in run_dir.glob("*.pftrace") if path.stat().st_size > 0)
    require(native_perfetto, f"no non-empty IKeT *.pftrace output found in {run_dir}")
    try:
        canonical, metadata = adapter.normalize_run(
            run_dir,
            kernel_pattern=arguments.kernel_regex,
            dictionary_path=arguments.dictionary.resolve(),
        )
        perfetto = adapter.make_perfetto(canonical)
        summary = adapter.summarize(canonical, metadata)
        binaries = adapter.target_binaries(run_dir, arguments.kernel_regex)
    except adapter.IketError as error:
        raise TimelineError(str(error)) from error

    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "trace.perfetto.json.gz"
    summary_path = output_dir / "trace.summary.json"
    native_index_path = output_dir / "iket.native-index.json"
    binary_identity_path = output_dir / "binary.identity.json"
    manifest_path = output_dir / "iket.manifest.json"
    write_gzip_json(trace_path, perfetto)
    write_json(summary_path, {"schema": SUMMARY_SCHEMA, **summary})
    native_trace_paths = [Path(value) for value in metadata["trace_files"]]
    write_json(
        native_index_path,
        {
            "schema": IKET_NATIVE_INDEX_SCHEMA,
            "trace_json": [
                {"path": str(path), "sha256": digest(path), "size_bytes": path.stat().st_size}
                for path in native_trace_paths
            ],
            "pftrace": [
                {"path": str(path), "sha256": digest(path), "size_bytes": path.stat().st_size}
                for path in native_perfetto
            ],
        },
    )
    binary_path: Path | None = None
    if binaries:
        write_json(
            binary_identity_path,
            {
                "schema": BINARY_IDENTITY_SCHEMA,
                "binaries": [
                    {"path": str(path), "sha256": digest(path), "size_bytes": path.stat().st_size}
                    for path in binaries
                ],
            },
        )
        binary_path = binary_identity_path
    write_json(
        manifest_path,
        {
            "schema": IKET_MANIFEST_SCHEMA,
            "backend": "iket",
            "kernel_regex": arguments.kernel_regex,
            "launches": metadata["launches"],
            "native_index": relative_artifact(native_index_path, output_dir),
            "binary_identity": (
                relative_artifact(binary_identity_path, output_dir) if binary_path else None
            ),
        },
    )
    receipt_path = write_receipt(
        output_dir=output_dir,
        backend="iket",
        raw_path=native_index_path,
        manifest_path=manifest_path,
        dictionary_path=arguments.dictionary.resolve(),
        clean_source=arguments.clean_source.resolve(),
        instrumented_source=arguments.instrumented_source.resolve(),
        binary_path=binary_path,
        workload_identity=arguments.workload_identity,
        reported_correctness=arguments.correctness,
        correctness_evidence_path=(
            arguments.correctness_evidence.resolve() if arguments.correctness_evidence else None
        ),
        stage=arguments.stage,
        measurement_path=arguments.measurement.resolve() if arguments.measurement else None,
        low_overhead_target=arguments.low_overhead_target,
        trace_path=trace_path,
        summary_path=summary_path,
    )
    print(json.dumps({"receipt": str(receipt_path), "summary": summary}, sort_keys=True))
    return 0


def profile_iket_command(arguments: argparse.Namespace) -> int:
    capture_iket_command(
        argparse.Namespace(output_dir=arguments.run_dir, target_command=arguments.target_command)
    )
    return export_iket_command(
        argparse.Namespace(
            run_dir=arguments.run_dir,
            kernel_regex=arguments.kernel_regex,
            dictionary=arguments.dictionary,
            clean_source=arguments.clean_source,
            instrumented_source=arguments.instrumented_source,
            workload_identity=arguments.workload_identity,
            correctness=arguments.correctness,
            correctness_evidence=arguments.correctness_evidence,
            measurement=arguments.measurement,
            stage=arguments.stage,
            low_overhead_target=arguments.low_overhead_target,
            output_dir=arguments.evidence_dir,
        )
    )


def receipt_artifact_path(receipt_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (receipt_path.parent / path).resolve()


def verify_digest(
    receipt: dict[str, Any], receipt_path: Path, path_field: str, hash_field: str
) -> None:
    path_value = receipt.get(path_field)
    hash_value = receipt.get(hash_field)
    require(isinstance(path_value, str) and path_value, f"receipt is missing {path_field}")
    require(isinstance(hash_value, str) and len(hash_value) == 64, f"receipt is missing {hash_field}")
    path = receipt_artifact_path(receipt_path, path_value)
    require(path.exists(), f"receipt artifact is missing: {path}")
    require(digest(path) == hash_value, f"receipt hash mismatch for {path_field}")


def verify_canonical_products(receipt: dict[str, Any], receipt_path: Path) -> None:
    def artifact(field: str) -> Path:
        value = receipt.get(field)
        require(isinstance(value, str) and value, f"receipt is missing {field}")
        return receipt_artifact_path(receipt_path, value)

    backend = receipt["backend"]
    raw_path = artifact("raw")
    manifest_path = artifact("manifest")
    dictionary_path = artifact("event_dictionary")
    trace_path = artifact("trace")
    summary_path = artifact("summary")
    trace = load_gzip_object(trace_path, "canonical trace")
    summary = load_object(summary_path, "trace summary")
    manifest = load_object(manifest_path, "manifest")
    require(manifest.get("backend") == backend, "receipt and manifest backend differ")

    if backend == "cuda":
        raw = raw_path.read_bytes()
        header = parse_header(raw)
        sites = load_sites(load_object(dictionary_path, "event dictionary"))
        writers = validate_manifest(manifest, header)
        records, canonical = decode_records(raw, header, writers, sites)
        require(trace == make_perfetto(canonical), "CUDA canonical trace is not reproducible")
        require(summary == summarize(canonical, records), "CUDA trace summary is not reproducible")
        return

    adapter = iket_adapter()
    try:
        declared = adapter.load_event_dictionary(dictionary_path)
    except adapter.IketError as error:
        raise TimelineError(str(error)) from error
    events = trace.get("traceEvents")
    require(isinstance(events, list), "IKeT canonical trace has no traceEvents list")
    origin = trace.get("otherData", {}).get("origin_ns")
    require(isinstance(origin, int) and origin > 0, "IKeT canonical trace has no valid origin")
    counts: dict[str, int] = defaultdict(int)
    durations: dict[str, int] = defaultdict(int)
    physical_sms: set[int] = set()
    lanes: set[tuple[int, int]] = set()
    metadata_lanes: set[tuple[int, int]] = set()
    for index, event in enumerate(events):
        require(isinstance(event, dict), f"IKeT trace event {index} is invalid")
        phase = event.get("ph")
        pid = event.get("pid")
        tid = event.get("tid")
        require(isinstance(pid, int) and pid > 0 and isinstance(tid, int) and tid >= 0,
                f"IKeT trace event {index} has invalid pid/tid")
        lane = (pid, tid)
        if phase == "M":
            require(event.get("name") == "thread_name", "unexpected IKeT metadata event")
            metadata_lanes.add(lane)
            continue
        require(phase in {"X", "i"}, f"IKeT trace event {index} has invalid phase")
        name = event.get("name")
        require(isinstance(name, str) and name in declared,
                f"IKeT trace event {index} has an undeclared name")
        args = event.get("args")
        require(isinstance(args, dict), f"IKeT trace event {index} has invalid args")
        timestamp = args.get("timestamp_ns")
        sm = args.get("physical_sm")
        require(isinstance(timestamp, int) and timestamp >= origin,
                f"IKeT trace event {index} has invalid timestamp")
        require(isinstance(sm, int) and sm >= 0 and pid == sm + 1,
                f"IKeT trace event {index} has invalid physical SM identity")
        ts = event.get("ts")
        require(isinstance(ts, (int, float)) and math.isfinite(float(ts)),
                f"IKeT trace event {index} has invalid display timestamp")
        require(math.isclose(float(ts) * 1000.0, timestamp - origin, abs_tol=1e-9),
                f"IKeT trace event {index} timestamp is not reproducible")
        declared_kind = declared[name].get("kind", "any")
        observed_kind = "range" if phase == "X" else "instant"
        require(declared_kind in {"any", observed_kind},
                f"IKeT trace event {name!r} kind differs from dictionary")
        counts[name] += 1
        physical_sms.add(sm)
        lanes.add(lane)
        if phase == "X":
            duration = args.get("duration_ns")
            end = args.get("end_timestamp_ns")
            require(isinstance(duration, int) and duration >= 0,
                    f"IKeT range {name!r} has invalid duration")
            require(isinstance(end, int) and end == timestamp + duration,
                    f"IKeT range {name!r} has invalid end timestamp")
            display_duration = event.get("dur")
            require(
                isinstance(display_duration, (int, float))
                and math.isclose(float(display_duration) * 1000.0, duration, abs_tol=1e-9),
                f"IKeT range {name!r} display duration is not reproducible",
            )
            durations[name] += duration
    require(lanes <= metadata_lanes, "IKeT trace is missing lane metadata")
    require(set(counts) == set(declared), "IKeT trace does not contain exactly the declared events")
    launches = manifest.get("launches")
    require(isinstance(launches, list) and launches, "IKeT manifest has no launches")
    expected = {
        "schema": SUMMARY_SCHEMA,
        "backend": "iket",
        "matched_launches": len(launches),
        "canonical_events": sum(counts.values()),
        "ranges": sum(event.get("ph") == "X" for event in events),
        "instants": sum(event.get("ph") == "i" for event in events),
        "physical_sms": sorted(physical_sms),
        "event_count": dict(sorted(counts.items())),
        "event_total_duration_ns": dict(sorted(durations.items())),
        "launches": launches,
    }
    require(summary == expected, "IKeT trace summary is not reproducible")


def validate_receipt_command(arguments: argparse.Namespace) -> int:
    receipt_path = arguments.receipt.resolve()
    receipt = load_object(receipt_path, "receipt")
    require(receipt.get("schema") == RECEIPT_SCHEMA, "unknown receipt schema")
    require(receipt.get("backend") in {"cuda", "iket"}, "receipt backend is invalid")
    require(receipt.get("stage") in {"exploration", "final"}, "receipt stage is invalid")
    require(receipt.get("correctness") in {"passed", "failed", "unknown"},
            "receipt correctness is invalid")
    require(receipt.get("trace_integrity") == "passed", "receipt trace integrity is not passed")
    workload_identity = receipt.get("workload_identity")
    require(isinstance(workload_identity, str) and workload_identity.strip(),
            "receipt workload identity is missing")
    for path_field, hash_field in (
        ("clean_source", "clean_source_sha256"),
        ("instrumented_source", "instrumented_source_sha256"),
        ("raw", "raw_sha256"),
        ("manifest", "manifest_sha256"),
        ("event_dictionary", "event_dictionary_sha256"),
        ("trace", "trace_sha256"),
        ("summary", "summary_sha256"),
    ):
        verify_digest(receipt, receipt_path, path_field, hash_field)
    if receipt.get("binary") is not None:
        verify_digest(receipt, receipt_path, "binary", "binary_sha256")
    instrumented_source = receipt_artifact_path(
        receipt_path, str(receipt.get("instrumented_source", ""))
    )
    verified_correctness, correctness_verified = correctness_from_evidence(
        receipt.get("correctness_evidence"), digest(evaluator_source(instrumented_source))
    )
    if correctness_verified:
        require(verified_correctness == receipt.get("correctness"),
                "receipt correctness contradicts evaluator evidence")
    verify_canonical_products(receipt, receipt_path)
    measurement = receipt.get("measurement")
    if measurement is not None:
        require(isinstance(measurement, dict), "receipt measurement must be an object")
        path = receipt_artifact_path(receipt_path, str(measurement.get("path", "")))
        require(path.is_file(), "measurement artifact is missing")
        require(digest(path) == measurement.get("sha256"), "measurement hash mismatch")
        recomputed = measurement_summary(path)
        for field in ("baseline_ms", "instrumented_ms", "overhead_percent"):
            require(
                math.isclose(float(measurement[field]), float(recomputed[field]), rel_tol=1e-12),
                f"measurement {field} is not reproducible",
            )
        require(recomputed["workload_identity"] == workload_identity,
                "measurement workload identity does not match receipt")
        require(
            recomputed["variants"]["baseline"]["source_sha256"]
            == receipt.get("clean_source_sha256"),
            "measurement baseline source does not match receipt",
        )
        require(
            recomputed["variants"]["instrumented"]["source_sha256"]
            == receipt.get("instrumented_source_sha256"),
            "measurement instrumented source does not match receipt",
        )
    target = receipt.get("low_overhead_target_percent")
    require(
        isinstance(target, (int, float)) and math.isfinite(float(target)) and float(target) >= 0,
        "receipt low-overhead target is invalid",
    )
    expected_class, expected_reasons = classify_evidence(
        stage=str(receipt.get("stage")),
        correctness=str(receipt.get("correctness")),
        correctness_verified=correctness_verified,
        has_binary=receipt.get("binary") is not None,
        measurement=measurement if isinstance(measurement, dict) else None,
        low_overhead_target=float(target),
    )
    require(receipt.get("evidence_class") == expected_class, "receipt evidence class is invalid")
    require(
        receipt.get("classification_reasons") == expected_reasons,
        "receipt classification reasons are invalid",
    )
    print(json.dumps({"receipt": str(arguments.receipt.resolve()), "valid": True}))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    measure = commands.add_parser(
        "measure", help="capture fresh-process counterbalanced instrumentation overhead"
    )
    measure.add_argument("--baseline-command", required=True, help="JSON argv array")
    measure.add_argument("--instrumented-command", required=True, help="JSON argv array")
    measure.add_argument("--baseline-source", type=Path, required=True)
    measure.add_argument("--instrumented-source", type=Path, required=True)
    measure.add_argument("--baseline-binary", type=Path)
    measure.add_argument("--instrumented-binary", type=Path)
    measure.add_argument("--workload-identity", required=True)
    measure.add_argument("--warmup", type=int, required=True)
    measure.add_argument("--iterations", type=int, required=True)
    measure.add_argument(
        "--order", action="append", choices=("ABBA", "BAAB"),
        help="four-run counterbalanced block; defaults to one ABBA and one BAAB block",
    )
    measure.add_argument("--timeout", type=int, default=120)
    measure.add_argument("--cwd", type=Path)
    measure.add_argument("--output", type=Path, required=True)
    measure.set_defaults(handler=measure_command)

    decode = commands.add_parser("decode", help="strictly decode raw CUDA timeline evidence")
    decode.add_argument("--raw", type=Path, required=True)
    decode.add_argument("--manifest", type=Path, required=True)
    decode.add_argument("--dictionary", type=Path, required=True)
    decode.add_argument("--clean-source", type=Path, required=True)
    decode.add_argument("--instrumented-source", type=Path, required=True)
    decode.add_argument("--binary", type=Path)
    decode.add_argument("--workload-identity", required=True)
    decode.add_argument("--correctness", choices=("passed", "failed", "unknown"))
    decode.add_argument("--correctness-evidence", type=Path)
    decode.add_argument("--measurement", type=Path)
    decode.add_argument("--stage", choices=("exploration", "final"), default="exploration")
    decode.add_argument("--low-overhead-target", type=float, default=10.0)
    decode.add_argument("--output-dir", type=Path, required=True)
    decode.set_defaults(handler=decode_command)

    validate = commands.add_parser("validate", help="recompute every hash and measurement in a receipt")
    validate.add_argument("receipt", type=Path)
    validate.set_defaults(handler=validate_receipt_command)

    capture_iket = commands.add_parser(
        "capture-iket", help="run a CuTe DSL command under IKeT in a fresh attempt directory"
    )
    capture_iket.add_argument("--output-dir", type=Path, required=True)
    capture_iket.add_argument("target_command", nargs=argparse.REMAINDER)
    capture_iket.set_defaults(handler=capture_iket_command)

    export_iket = commands.add_parser(
        "export-iket", help="strictly normalize one completed IKeT run"
    )
    export_iket.add_argument("--run-dir", type=Path, required=True)
    export_iket.add_argument("--kernel-regex", required=True)
    export_iket.add_argument("--dictionary", type=Path, required=True)
    export_iket.add_argument("--clean-source", type=Path, required=True)
    export_iket.add_argument("--instrumented-source", type=Path, required=True)
    export_iket.add_argument("--workload-identity", required=True)
    export_iket.add_argument("--correctness", choices=("passed", "failed", "unknown"))
    export_iket.add_argument("--correctness-evidence", type=Path)
    export_iket.add_argument("--measurement", type=Path)
    export_iket.add_argument("--stage", choices=("exploration", "final"), default="exploration")
    export_iket.add_argument("--low-overhead-target", type=float, default=10.0)
    export_iket.add_argument("--output-dir", type=Path, required=True)
    export_iket.set_defaults(handler=export_iket_command)

    profile_iket = commands.add_parser(
        "profile-iket", help="capture and normalize IKeT evidence before a remote job exits"
    )
    profile_iket.add_argument("--run-dir", type=Path, required=True)
    profile_iket.add_argument("--evidence-dir", type=Path, required=True)
    profile_iket.add_argument("--kernel-regex", required=True)
    profile_iket.add_argument("--dictionary", type=Path, required=True)
    profile_iket.add_argument("--clean-source", type=Path, required=True)
    profile_iket.add_argument("--instrumented-source", type=Path, required=True)
    profile_iket.add_argument("--workload-identity", required=True)
    profile_iket.add_argument("--correctness", choices=("passed", "failed", "unknown"))
    profile_iket.add_argument("--correctness-evidence", type=Path)
    profile_iket.add_argument("--measurement", type=Path)
    profile_iket.add_argument("--stage", choices=("exploration", "final"), default="exploration")
    profile_iket.add_argument("--low-overhead-target", type=float, default=10.0)
    profile_iket.add_argument("target_command", nargs=argparse.REMAINDER)
    profile_iket.set_defaults(handler=profile_iket_command)
    return root


def main(argv: Iterable[str] | None = None) -> int:
    arguments = parser().parse_args(list(argv) if argv is not None else None)
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (TimelineError, OSError) as error:
        print(f"timeline: {error}", file=sys.stderr)
        raise SystemExit(2) from error
