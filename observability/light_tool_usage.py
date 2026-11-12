#!/usr/bin/env python3
"""Report Runtime Tool calls, outcomes, durations and Session attribution."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from observability.common import (
    add_output_arguments,
    discover_scopes,
    emit,
    KIND_PATTERN,
    load_snapshot,
    main_error,
    observed_at,
    read_json,
    table,
)


def _tool(operation: object, argv: object) -> str:
    values = [str(item) for item in argv] if isinstance(argv, list) else []
    text = " ".join(values)
    if operation == "gateway":
        match = KIND_PATTERN.search(text)
        return match.group(1) if match else "gateway"
    if operation in {"journal", "wiki", "plugin"} and values:
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


def _structured_success(content: object) -> tuple[bool | None, str]:
    """Read an explicit result status; never infer failure from prose substrings."""
    if not isinstance(content, str):
        return None, "unavailable"
    candidates = []
    payloads = [content, *content.splitlines()]
    for line in payloads:
        payload = line.split("=", 1)[-1] if "_JSON=" in line else line
        try:
            value = json.loads(payload)
        except ValueError:
            continue
        if isinstance(value, dict):
            candidates.append(value)
    for value in candidates:
        exit_code = value.get("exit_code")
        if type(exit_code) is int:
            return exit_code == 0, "structured_exit_code"
        success = value.get("success")
        if type(success) is bool:
            return success, "structured_success"
        ok = value.get("ok")
        if type(ok) is bool:
            return ok, "structured_ok"
        status = str(value.get("status") or "").casefold()
        if status in {"completed", "succeeded", "success", "passed", "pass", "reused"}:
            return True, "structured_status"
        if status in {"failed", "failure", "error", "cancelled", "canceled", "rejected"}:
            return False, "structured_status"
        if isinstance(value.get("error"), dict) or value.get("error_class"):
            return False, "structured_error"
    return None, "unavailable"


def _audit_calls(scope) -> list[dict]:
    rows = []
    for path in sorted(scope.audit_root.glob("request-*.json")):
        value = read_json(path, default={})
        if not isinstance(value, dict):
            continue
        duration = _stream_duration(value.get("stdout"), value.get("stderr"))
        rows.append(
            {
                "scope": scope.label,
                "trajectory": scope.trajectory,
                "request_id": path.stem.removeprefix("request-"),
                "created_at": value.get("created_at"),
                "tool": _tool(value.get("operation"), value.get("argv")),
                "operation_family": value.get("operation"),
                "success": value.get("exit_code") == 0,
                "success_basis": "audit_exit_code",
                "exit_code": value.get("exit_code"),
                "duration_seconds": duration,
                "gateway_record_id": value.get("gateway_record_id"),
                "session_id": None,
                "episode": None,
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
                "tool": value.get("tool") or "wiki",
                "operation_family": "wiki_receipt",
                "success": value.get("status") in {"completed", "reused"},
                "success_basis": "wiki_receipt_status",
                "exit_code": None,
                "duration_seconds": value.get("total_latency_ms") / 1000.0
                if type(value.get("total_latency_ms")) in {int, float}
                else None,
                "gateway_record_id": None,
                "session_id": None,
                "episode": None,
                "status": value.get("status"),
                "cache_hit": value.get("cache_hit") is True,
            }
        if receipt["request_id"] in by_request_id:
            existing = by_request_id[receipt["request_id"]]
            existing.update(
                duration_seconds=receipt["duration_seconds"],
                status=receipt["status"],
                cache_hit=receipt["cache_hit"],
            )
        else:
            rows.append(receipt)
    return rows


def _session_calls(scope) -> list[dict]:
    snapshot = load_snapshot(scope)
    rows = []
    for session in snapshot.sessions:
        pending = {}
        for event in session.events:
            if not isinstance(event, dict):
                continue
            if event.get("kind") == "tool_call":
                text = str(event.get("content") or "")
                match = KIND_PATTERN.search(text)
                tool = match.group(1) if match else str(event.get("tool_name") or "unknown")
                row = {
                    "scope": scope.label,
                    "trajectory": scope.trajectory,
                    "session_id": session.session_id,
                    "episode": session.episode,
                    "tool_id": event.get("tool_id"),
                    "tool": tool,
                    "sequence": event.get("sequence"),
                    "success": None,
                    "success_basis": "unavailable",
                }
                rows.append(row)
                if row["tool_id"]:
                    pending[row["tool_id"]] = row
            elif event.get("kind") == "tool_result" and event.get("tool_id") in pending:
                success, basis = _structured_success(event.get("content"))
                pending[event["tool_id"]]["success"] = success
                pending[event["tool_id"]]["success_basis"] = basis
    return rows


def snapshot(workspace: Path) -> dict:
    calls = []
    attributed = []
    for scope in discover_scopes(workspace):
        calls.extend(_audit_calls(scope))
        attributed.extend(_session_calls(scope))
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
        "schema_version": 1,
        "observed_at": observed_at(),
        "campaign_workspace": str(Path(workspace).expanduser().resolve()),
        "summary": summary,
        "calls": calls,
        "session_attribution": attributed,
        "attribution_note": (
            "Audit receipts contain authoritative outcomes but not Session/Episode IDs. "
            "Session transcripts contain attribution but cannot be safely joined to every audit receipt on current main. "
            "Session success is reported only when the tool result carries a structured status; otherwise it is null."
        ),
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
            "",
            value["attribution_note"],
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
