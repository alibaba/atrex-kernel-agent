#!/usr/bin/env python3
"""Read authoritative Runtime Tool audit calls, outcomes and durations."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from observability._internal.common import (
    add_output_arguments,
    discover_scopes,
    emit,
    KIND_PATTERN,
    main_error,
    observed_at,
    elapsed_seconds,
    age_seconds,
    read_json,
    runtime_audit_paths,
    table,
)


def _tool(operation: object, argv: object) -> str:
    values = [str(item) for item in argv] if isinstance(argv, list) else []
    text = " ".join(values)
    if operation == "gateway":
        match = KIND_PATTERN.search(text)
        return match.group(1) if match else "gateway"
    # Wiki audit argv starts with the natural-language question, not a command
    # name.  Treating that value as the tool explodes the summary cardinality
    # and can expose an entire prompt as a table label.
    if operation == "wiki":
        return "wiki-query"
    if operation in {"journal", "plugin"} and values:
        return values[0]
    return str(operation or "unknown")


def _duration(value: object) -> float | None:
    if isinstance(value, dict):
        for key in ("duration_seconds", "elapsed_seconds", "total_latency_seconds"):
            item = value.get(key)
            if type(item) in {int, float} and item >= 0:
                return float(item)
        for item in value.values():
            if (found := _duration(item)) is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            if (found := _duration(item)) is not None:
                return found
    return None


def _stream_duration(*streams: object) -> float | None:
    """Return the first structured duration without letting later streams erase it."""
    for stream in streams:
        if not isinstance(stream, str):
            continue
        for line in stream.splitlines():
            payload = line.split("=", 1)[-1] if "_JSON=" in line else line
            try:
                parsed = json.loads(payload)
            except ValueError:
                continue
            if (duration := _duration(parsed)) is not None:
                return duration
    return None


def _status_success(status: object) -> bool | None:
    value = str(status or "").casefold()
    if value in {"completed", "succeeded", "success", "passed", "reused"}:
        return True
    if value in {
        "failed", "failure", "error", "cancelled", "canceled", "rejected",
        "timed_out", "timeout",
    }:
        return False
    return None


def _audit_calls(scope) -> list[dict]:
    rows = []
    for path in runtime_audit_paths(scope):
        value = read_json(path, default={})
        if not isinstance(value, dict):
            continue
        state = value.get("state")
        duration = value.get("duration_seconds")
        if type(duration) not in {int, float} or duration < 0:
            duration = elapsed_seconds(value.get("started_at"), value.get("finished_at"))
        if duration is None:
            duration = _stream_duration(value.get("stdout"), value.get("stderr"))
        exit_code = value.get("exit_code")
        if state == "succeeded":
            success = True
        elif state == "failed":
            success = False
        else:
            success = exit_code == 0 if type(exit_code) is int else None
        started_at = value.get("started_at") or value.get("created_at")
        rows.append(
            {
                "scope": scope.label,
                "trajectory": scope.trajectory,
                "request_id": path.stem.removeprefix("request-"),
                "created_at": started_at,
                "started_at": started_at,
                "updated_at": value.get("updated_at") or value.get("finished_at") or started_at,
                "finished_at": value.get("finished_at"),
                "tool": value.get("tool") or _tool(value.get("operation"), value.get("argv")),
                "operation_family": value.get("operation"),
                "status": state,
                "success": success,
                "success_basis": (
                    "request_lifecycle"
                    if state in {"succeeded", "failed", "pending", "running"}
                    else "audit_exit_code"
                    if type(exit_code) is int
                    else "unavailable"
                ),
                "exit_code": exit_code if type(exit_code) is int else None,
                "duration_seconds": duration,
                "pending_age_seconds": age_seconds(started_at)
                if state in {"running", "pending"}
                else None,
                "http_status": value.get("http_status"),
                "error": value.get("error") if isinstance(value.get("error"), dict) else None,
                "gateway_record_id": value.get("gateway_record_id"),
            }
        )
    by_request_id = {row["request_id"]: row for row in rows}
    # Wiki receipts carry authoritative queue/total durations even when their
    # audit response does not contain a duration marker.
    for path in sorted((scope.audit_root / "wiki" / "requests").glob("*.json")):
        value = read_json(path, default={})
        if not isinstance(value, dict):
            continue
        receipt = {
                "scope": scope.label,
                "trajectory": scope.trajectory,
                "request_id": value.get("request_id") or path.stem,
                "created_at": value.get("finished_at"),
                "tool": "wiki-query",
                "operation_family": "wiki_receipt",
                "success": _status_success(value.get("status")),
                "success_basis": "wiki_receipt_status",
                "duration_seconds": value.get("total_latency_ms") / 1000.0
                if type(value.get("total_latency_ms")) in {int, float}
                else None,
                "status": value.get("status"),
                "cache_hit": value.get("cache_hit") is True,
            }
        if receipt["request_id"] in by_request_id:
            existing = by_request_id[receipt["request_id"]]
            existing.update(
                duration_seconds=receipt["duration_seconds"],
                status=receipt["status"],
                cache_hit=receipt["cache_hit"],
                success=receipt["success"],
                success_basis="wiki_receipt_status",
            )
        else:
            rows.append(receipt)
    return rows


def snapshot(workspace: Path) -> dict:
    calls = []
    for scope in discover_scopes(workspace):
        calls.extend(_audit_calls(scope))
    grouped = defaultdict(list)
    for row in calls:
        grouped[row["tool"]].append(row)
    summary = []
    for tool, rows in sorted(grouped.items()):
        durations = [row["duration_seconds"] for row in rows if isinstance(row["duration_seconds"], (int, float))]
        successes = sum(row["success"] is True for row in rows)
        failures = sum(row["success"] is False for row in rows)
        summary.append(
            {
                "tool": tool,
                "calls": len(rows),
                "successes": successes,
                "failures": failures,
                "success_rate": successes / (successes + failures) if successes + failures else None,
                "average_duration_seconds": sum(durations) / len(durations) if durations else None,
                "duration_samples": len(durations),
            }
        )
    return {
        "schema_version": 2,
        "observed_at": observed_at(),
        "campaign_workspace": str(Path(workspace).expanduser().resolve()),
        "summary": summary,
        "calls": calls,
    }


def render(value: dict) -> str:
    return "\n".join(
        [
            "AKA Light Runtime Tool 使用",
            "",
            table(
                ["Tool", "Calls", "Success", "Failure", "Rate", "Avg(s)", "Timed"],
                [[row["tool"], row["calls"], row["successes"], row["failures"],
                  round(row["success_rate"], 4) if row["success_rate"] is not None else None,
                  round(row["average_duration_seconds"], 3) if row["average_duration_seconds"] is not None else None,
                  row["duration_samples"]]
                 for row in value["summary"]],
            ),
        ]
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    add_output_arguments(parser)
    return parser.parse_args(argv)


def _main(argv=None) -> int:
    options = parse_args(argv)
    value = snapshot(options.workspace)
    emit(value, as_json=options.json, compact=options.compact, text=render(value))
    return 0


def main(argv=None) -> int:
    return main_error("light-tool-usage", _main, argv)


if __name__ == "__main__":
    raise SystemExit(main())
