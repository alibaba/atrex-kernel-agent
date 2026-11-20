#!/usr/bin/env python3
"""Summarize physical Agate calls and select recent calls by kind/outcome."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from observability._internal.common import add_output_arguments, emit, main_error, observed_at, parse_timestamp, table
from observability._internal.agate_state import snapshot as agate_snapshot
from observability._internal.view_helpers import FILTER_STATUSES, grouped_summary, normalized_outcome


def _timestamp(row: dict) -> float:
    value = parse_timestamp(row.get("checkpoint_updated_at"))
    return value.timestamp() if value is not None else float("-inf")


def _status(row: dict) -> str:
    if row.get("pending_seconds") is not None:
        return "pending"
    return normalized_outcome(status=row.get("status"), error=row.get("error"))


def snapshot(
    workspace: Path,
    *,
    kinds: tuple[str, ...] = (),
    statuses: tuple[str, ...] = (),
    lineages: tuple[str, ...] = (),
    since: str | None = None,
    until: str | None = None,
    last: int | None = None,
) -> dict:
    if last is not None and (type(last) is not int or last <= 0):
        raise ValueError("--last must be a positive integer")
    invalid = set(statuses) - set(FILTER_STATUSES)
    if invalid:
        raise ValueError("--status must be one of: " + ", ".join(FILTER_STATUSES))
    since_at = parse_timestamp(since) if since else None
    until_at = parse_timestamp(until) if until else None
    if since and since_at is None:
        raise ValueError("--since must be an ISO-8601 timestamp")
    if until and until_at is None:
        raise ValueError("--until must be an ISO-8601 timestamp")
    if since_at is not None and until_at is not None and since_at > until_at:
        raise ValueError("--since must not be later than --until")
    raw = agate_snapshot(workspace)
    calls = []
    for row in raw.get("submission_attempts", []):
        timing = row.get("timing") if isinstance(row.get("timing"), dict) else {}
        public_row = dict(row)
        public_row["agate_job_id"] = public_row.pop("job_id", None)
        calls.append(
            {
                **public_row,
                "kind": row.get("operation") or "unknown",
                "status": _status(row),
                "queue_seconds": timing.get("queue_seconds"),
                "queue_seconds_source": timing.get("queue_seconds_source"),
                "waiting_seconds": timing.get("queue_seconds"),
                "execution_seconds": timing.get("execution_seconds"),
                "execution_seconds_source": timing.get("execution_seconds_source"),
                "total_seconds": timing.get("total_seconds"),
                "total_seconds_source": timing.get("total_seconds_source"),
                "pending_age_seconds": row.get("pending_seconds"),
            }
        )
    summary = grouped_summary(
        calls,
        key="kind",
        duration_fields=("queue_seconds", "execution_seconds", "total_seconds"),
    )
    for row in summary:
        # Compatibility alias. New consumers should use queue_seconds because it
        # is Agate's submitted→started interval, not Runtime end-to-end waiting.
        for prefix in ("average", "p50", "p95", "max"):
            row[f"{prefix}_waiting_seconds"] = row.get(f"{prefix}_queue_seconds")
        row["waiting_seconds_samples"] = row.get("queue_seconds_samples")
        row["waiting_seconds_coverage"] = row.get("queue_seconds_coverage")
        kind_calls = [call for call in calls if call.get("kind") == row.get("kind")]
        for field in ("queue_seconds", "execution_seconds", "total_seconds"):
            row[f"{field}_source_counts"] = dict(Counter(
                call.get(f"{field}_source")
                for call in kind_calls
                if call.get(field) is not None and call.get(f"{field}_source")
            ))
    selected = calls
    if kinds:
        wanted = set(kinds)
        selected = [row for row in selected if row["kind"] in wanted]
    if statuses:
        wanted_statuses = set(statuses)
        selected = [row for row in selected if row["status"] in wanted_statuses]
    if lineages:
        wanted_lineages = set(lineages)
        selected = [row for row in selected if row.get("scope") in wanted_lineages]
    if since_at is not None or until_at is not None:
        bounded = []
        for row in selected:
            timestamp = parse_timestamp(row.get("checkpoint_updated_at"))
            if timestamp is None:
                continue
            if since_at is not None and timestamp < since_at:
                continue
            if until_at is not None and timestamp > until_at:
                continue
            bounded.append(row)
        selected = bounded
    selected = sorted(selected, key=_timestamp, reverse=True)
    if last is not None:
        selected = selected[:last]
    if not kinds and not statuses and not lineages and since is None and until is None and last is None:
        selected = []
    return {
        "schema_version": 1,
        "observed_at": observed_at(),
        "campaign_workspace": raw.get("campaign_workspace"),
        "summary": summary,
        "calls": selected,
        "detail_selection": {
            "kinds": list(dict.fromkeys(kinds)),
            "statuses": list(dict.fromkeys(statuses)),
            "lineages": list(dict.fromkeys(lineages)),
            "since": since,
            "until": until,
            "last": last,
            "returned": len(selected),
            "campaign_total": len(calls),
            "summary_scope": "campaign",
        },
    }


def render(value: dict) -> str:
    lines = [
        "AKA Light Agate calls",
        "",
        table(
            [
                "Kind", "Calls", "Succeeded", "Failed", "Pending", "Unknown",
                "Avg queue(s)", "P95 queue(s)", "Avg exec(s)", "Avg total(s)",
            ],
            [
                [
                    row["kind"],
                    row["calls"],
                    row["succeeded"],
                    row["failed"],
                    row["pending"],
                    row["unknown"],
                    row["average_queue_seconds"],
                    row["p95_queue_seconds"],
                    row["average_execution_seconds"],
                    row["average_total_seconds"],
                ]
                for row in value["summary"]
            ],
        ),
    ]
    if value["calls"]:
        lines += [
            "",
            "Selected calls",
            table(
                ["Time", "Kind", "Status", "Wait(s)", "Exec(s)", "Job", "Gateway Record", "Error"],
                [
                    [
                        row.get("checkpoint_updated_at"),
                        row.get("kind"),
                        row.get("status"),
                        row.get("waiting_seconds"),
                        row.get("execution_seconds"),
                        row.get("agate_job_id"),
                        row.get("gateway_record_id"),
                        (row.get("error") or {}).get("reason"),
                    ]
                    for row in value["calls"]
                ],
            ),
        ]
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--kind", action="append", default=[], help="Agate operation kind; repeatable")
    parser.add_argument("--status", action="append", choices=FILTER_STATUSES, default=[], help="outcome; repeatable")
    parser.add_argument("--lineage", action="append", default=[], help="lineage scope; repeatable")
    parser.add_argument("--since", help="inclusive ISO-8601 checkpoint-time lower bound")
    parser.add_argument("--until", help="inclusive ISO-8601 checkpoint-time upper bound")
    parser.add_argument("--last", type=int, help="newest N physical submissions after filtering")
    add_output_arguments(parser)
    return parser.parse_args(argv)


def _main(argv=None) -> int:
    options = parse_args(argv)
    value = snapshot(
        options.workspace,
        kinds=tuple(options.kind),
        statuses=tuple(options.status),
        lineages=tuple(options.lineage),
        since=options.since,
        until=options.until,
        last=options.last,
    )
    emit(value, as_json=options.json, compact=options.compact, text=render(value))
    return 0


def main(argv=None) -> int:
    return main_error("light-agate-calls", _main, argv)


if __name__ == "__main__":
    raise SystemExit(main())
