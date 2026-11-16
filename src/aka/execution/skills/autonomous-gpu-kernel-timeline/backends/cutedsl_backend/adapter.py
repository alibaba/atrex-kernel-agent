"""Strict normalization for official IKeT JSON traces."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DICTIONARY_SCHEMA = "atrex-autonomous-iket-events/v1"


class IketError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise IketError(message)


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise IketError(f"cannot read {label} {path}: {error}") from error
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def load_event_dictionary(path: Path) -> dict[str, dict[str, Any]]:
    dictionary = load_json(path, "IKeT event dictionary")
    require(dictionary.get("schema") == DICTIONARY_SCHEMA, "unknown IKeT dictionary schema")
    raw_events = dictionary.get("events")
    require(isinstance(raw_events, list) and raw_events, "IKeT dictionary events must be non-empty")
    events: dict[str, dict[str, Any]] = {}
    for index, event in enumerate(raw_events):
        require(isinstance(event, dict), f"events[{index}] must be an object")
        name = event.get("name")
        kind = event.get("kind", "any")
        require(isinstance(name, str) and name, f"events[{index}] has no name")
        require(len(name) <= 32, f"IKeT event name exceeds 32 characters: {name!r}")
        require(name not in events, f"duplicate IKeT event name {name!r}")
        require(kind in {"any", "instant", "range"}, f"invalid IKeT event kind for {name!r}")
        events[name] = event
    return events


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    require(isinstance(value, int) and not isinstance(value, bool), f"{label} must be an integer")
    require(value >= minimum, f"{label} must be at least {minimum}")
    return value


def _location(
    locations: list[object],
    index: object,
    label: str,
    *,
    grid: list[int],
    block: list[int],
) -> dict[str, Any]:
    location_index = _integer(index, f"{label}.locIdx")
    require(location_index < len(locations), f"{label}.locIdx is outside locationTable")
    raw = locations[location_index]
    require(isinstance(raw, dict), f"locationTable[{location_index}] must be an object")
    cta = raw.get("ctaId")
    require(
        isinstance(cta, list)
        and len(cta) == 3
        and all(isinstance(item, int) and item >= 0 for item in cta),
        f"locationTable[{location_index}].ctaId must be three non-negative integers",
    )
    require(
        all(coordinate < extent for coordinate, extent in zip(cta, grid, strict=True)),
        f"locationTable[{location_index}].ctaId is outside the launch grid",
    )
    warp = _integer(raw.get("warpId"), f"locationTable[{location_index}].warpId")
    warp_count = (block[0] * block[1] * block[2] + 31) // 32
    require(warp < warp_count, f"locationTable[{location_index}].warpId is outside the block")
    physical_sm = _integer(raw.get("smId"), f"locationTable[{location_index}].smId")
    require(physical_sm < 1024, f"locationTable[{location_index}].smId exceeds 10-bit identity")
    return {
        "location_index": location_index,
        "cta": cta,
        "warp": warp,
        "physical_sm": physical_sm,
        "gpc": raw.get("gpcId"),
        "tpc": raw.get("tpcId"),
        "cluster": raw.get("clusterId"),
    }


def _name(strings: list[object], index: object, label: str) -> str:
    string_index = _integer(index, label)
    require(string_index < len(strings), f"{label} is outside stringTable")
    value = strings[string_index]
    require(isinstance(value, str) and value, f"stringTable[{string_index}] is not a name")
    return value


def normalize_run(
    run_dir: Path,
    *,
    kernel_pattern: str,
    dictionary_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        matcher = re.compile(kernel_pattern)
    except re.error as error:
        raise IketError(f"invalid target kernel regex: {error}") from error
    declared = load_event_dictionary(dictionary_path)
    trace_paths = sorted(run_dir.glob("*.trace.json"))
    require(trace_paths, f"no IKeT *.trace.json output found in {run_dir}")

    canonical: list[dict[str, Any]] = []
    matched_launches: list[dict[str, Any]] = []
    observed_kinds: dict[str, set[str]] = defaultdict(set)
    for trace_path in trace_paths:
        trace = load_json(trace_path, "IKeT trace")
        strings = trace.get("stringTable")
        locations = trace.get("locationTable")
        launches = trace.get("launches")
        graph_launches = trace.get("graphLaunches")
        require(isinstance(strings, list), f"{trace_path.name} stringTable must be a list")
        require(isinstance(locations, list), f"{trace_path.name} locationTable must be a list")
        require(isinstance(launches, list), f"{trace_path.name} launches must be a list")
        require(isinstance(graph_launches, dict), f"{trace_path.name} graphLaunches must be an object")
        indexed_launches = [
            (f"{trace_path.name}:{launch_index}", launch)
            for launch_index, launch in enumerate(launches)
        ]
        for graph_key, entries in graph_launches.items():
            require(isinstance(graph_key, str) and graph_key,
                    f"{trace_path.name} graph launch key is invalid")
            require(isinstance(entries, list),
                    f"{trace_path.name} graphLaunches[{graph_key!r}] must be a list")
            indexed_launches.extend(
                (f"{trace_path.name}:graph:{graph_key}:{launch_index}", launch)
                for launch_index, launch in enumerate(entries)
            )
        for launch_key, launch in indexed_launches:
            require(isinstance(launch, dict), f"{launch_key} launch is invalid")
            kernel = launch.get("kernelName")
            require(isinstance(kernel, str) and kernel, "IKeT launch has no kernelName")
            if matcher.fullmatch(kernel) is None:
                continue
            grid = [
                _integer(launch.get(f"gridDim{axis}"), f"{launch_key}.gridDim{axis}", minimum=1)
                for axis in "XYZ"
            ]
            block = [
                _integer(launch.get(f"blockDim{axis}"), f"{launch_key}.blockDim{axis}", minimum=1)
                for axis in "XYZ"
            ]
            matched_launches.append(
                {
                    "launch_key": launch_key,
                    "kernel": kernel,
                    "grid_id": launch.get("gridId"),
                    "context_id": launch.get("contextId"),
                    "grid": grid,
                    "block": block,
                }
            )
            for marker_index, marker in enumerate(launch.get("markers", [])):
                require(isinstance(marker, dict), f"{launch_key} marker {marker_index} is invalid")
                name = _name(strings, marker.get("markerNameIdx"), f"{launch_key}.markerNameIdx")
                location = _location(
                    locations, marker.get("locIdx"), f"{launch_key}.marker", grid=grid, block=block
                )
                timestamp = _integer(marker.get("timestamp"), f"{launch_key}.marker.timestamp", minimum=1)
                observed_kinds[name].add("instant")
                canonical.append(
                    {
                        **location,
                        "launch_key": launch_key,
                        "kernel": kernel,
                        "grid_id": launch.get("gridId"),
                        "kind": "instant",
                        "name": name,
                        "timestamp_ns": timestamp,
                        "payload_type": marker.get("payloadType"),
                        "payload": marker.get("payloadVal"),
                    }
                )
            for range_index, item in enumerate(launch.get("ranges", [])):
                require(isinstance(item, dict), f"{launch_key} range {range_index} is invalid")
                name = _name(strings, item.get("rangeNameIdx"), f"{launch_key}.rangeNameIdx")
                location_indices = item.get("warpLocIdxs")
                require(
                    isinstance(location_indices, list) and len(location_indices) == 2,
                    f"{launch_key} range {range_index} has invalid warpLocIdxs",
                )
                start_location = _location(
                    locations,
                    location_indices[0],
                    f"{launch_key}.range.start",
                    grid=grid,
                    block=block,
                )
                end_location = _location(
                    locations,
                    location_indices[1],
                    f"{launch_key}.range.end",
                    grid=grid,
                    block=block,
                )
                require(
                    all(
                        start_location[field] == end_location[field]
                        for field in ("cta", "warp", "physical_sm")
                    ),
                    f"{launch_key} range {name!r} changes CTA, warp, or physical SM",
                )
                start = _integer(item.get("startTs"), f"{launch_key}.range.startTs", minimum=1)
                end = _integer(item.get("endTs"), f"{launch_key}.range.endTs", minimum=1)
                require(end >= start, f"{launch_key} range {name!r} ends before it starts")
                observed_kinds[name].add("range")
                canonical.append(
                    {
                        **start_location,
                        "launch_key": launch_key,
                        "kernel": kernel,
                        "grid_id": launch.get("gridId"),
                        "kind": "range",
                        "name": name,
                        "timestamp_ns": start,
                        "end_timestamp_ns": end,
                        "duration_ns": end - start,
                        "end_location": end_location,
                        "range_id": item.get("rangeId"),
                        "range_scope": item.get("rangeScope"),
                        "range_type": item.get("rangeType"),
                        "internal_events": item.get("internalEvents", []),
                    }
                )

    require(matched_launches, f"no IKeT launch fully matches {kernel_pattern!r}")
    require(canonical, "matched IKeT launch contains no markers or ranges")
    for name, declaration in declared.items():
        require(name in observed_kinds, f"declared IKeT event {name!r} was not observed")
        declared_kind = declaration.get("kind", "any")
        if declared_kind != "any":
            require(
                observed_kinds[name] == {declared_kind},
                f"IKeT event {name!r} kind does not match its declaration",
            )
    unexpected = sorted(set(observed_kinds) - set(declared))
    require(not unexpected, "IKeT trace contains undeclared events: " + ", ".join(unexpected))
    canonical.sort(key=lambda item: (item["timestamp_ns"], item["launch_key"], item["location_index"]))
    return canonical, {"trace_files": [str(path) for path in trace_paths], "launches": matched_launches}


def make_perfetto(canonical: list[dict[str, Any]]) -> dict[str, Any]:
    origin = min(event["timestamp_ns"] for event in canonical)
    lanes = sorted({(event["launch_key"], event["location_index"]) for event in canonical})
    lane_ids = {key: index + 1 for index, key in enumerate(lanes)}
    trace_events: list[dict[str, Any]] = []
    for launch_key, location_index in lanes:
        event = next(
            item
            for item in canonical
            if item["launch_key"] == launch_key and item["location_index"] == location_index
        )
        cta = ",".join(str(value) for value in event["cta"])
        trace_events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": event["physical_sm"] + 1,
                "tid": lane_ids[(launch_key, location_index)],
                "args": {"name": f"{event['kernel']}/cta({cta})/warp{event['warp']}"},
            }
        )
    for event in canonical:
        args = {
            "backend": "iket",
            "launch_key": event["launch_key"],
            "kernel": event["kernel"],
            "grid_id": event["grid_id"],
            "cta": event["cta"],
            "warp": event["warp"],
            "physical_sm": event["physical_sm"],
            "timestamp_ns": event["timestamp_ns"],
        }
        item = {
            "name": event["name"],
            "pid": event["physical_sm"] + 1,
            "tid": lane_ids[(event["launch_key"], event["location_index"])],
            "ts": (event["timestamp_ns"] - origin) / 1000.0,
            "args": args,
        }
        if event["kind"] == "range":
            item.update({"ph": "X", "dur": event["duration_ns"] / 1000.0})
            args.update(
                {
                    "end_timestamp_ns": event["end_timestamp_ns"],
                    "duration_ns": event["duration_ns"],
                    "range_id": event["range_id"],
                    "range_scope": event["range_scope"],
                }
            )
        else:
            item.update({"ph": "i", "s": "t"})
            args.update({"payload_type": event["payload_type"], "payload": event["payload"]})
        trace_events.append(item)
    return {
        "displayTimeUnit": "ns",
        "otherData": {"backend": "iket", "clock": "IKeT device timestamp", "origin_ns": origin},
        "traceEvents": trace_events,
    }


def summarize(canonical: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    counts = Counter(event["name"] for event in canonical)
    durations: dict[str, int] = defaultdict(int)
    for event in canonical:
        if event["kind"] == "range":
            durations[event["name"]] += event["duration_ns"]
    return {
        "backend": "iket",
        "matched_launches": len(metadata["launches"]),
        "canonical_events": len(canonical),
        "ranges": sum(event["kind"] == "range" for event in canonical),
        "instants": sum(event["kind"] == "instant" for event in canonical),
        "physical_sms": sorted({event["physical_sm"] for event in canonical}),
        "event_count": dict(sorted(counts.items())),
        "event_total_duration_ns": dict(sorted(durations.items())),
        "launches": metadata["launches"],
    }


def target_binaries(run_dir: Path, kernel_pattern: str) -> list[Path]:
    try:
        matcher = re.compile(kernel_pattern)
    except re.error as error:
        raise IketError(f"invalid target kernel regex: {error}") from error
    binaries: set[Path] = set()
    for tracker_path in sorted(run_dir.glob("tracker/*/tracker.json")):
        tracker = load_json(tracker_path, "IKeT tracker")
        launches = tracker.get("launches")
        graph_launches = tracker.get("graphLaunches")
        modules = tracker.get("modules")
        require(isinstance(launches, list), f"{tracker_path} launches must be a list")
        require(isinstance(graph_launches, dict), f"{tracker_path} graphLaunches must be an object")
        require(isinstance(modules, list), f"{tracker_path} modules must be a list")
        all_launches = list(launches)
        for graph_key, entries in graph_launches.items():
            require(isinstance(graph_key, str) and graph_key,
                    f"{tracker_path} graph launch key is invalid")
            require(isinstance(entries, list),
                    f"{tracker_path} graphLaunches[{graph_key!r}] must be a list")
            all_launches.extend(entries)
        target_module_ids = {
            launch.get("moduleId")
            for launch in all_launches
            if isinstance(launch, dict)
            and isinstance(launch.get("kernelName"), str)
            and matcher.fullmatch(launch["kernelName"])
        }
        for module in modules:
            if not isinstance(module, dict) or module.get("moduleId") not in target_module_ids:
                continue
            image = module.get("image")
            require(isinstance(image, str) and image, "matched IKeT module has no image")
            path = Path(image)
            if not path.is_file():
                path = tracker_path.parent / Path(image).name
            require(path.is_file(), f"matched IKeT binary is missing: {image}")
            binaries.add(path.resolve())
    return sorted(binaries)
