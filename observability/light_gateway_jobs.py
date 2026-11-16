#!/usr/bin/env python3
"""List every Agate submission belonging to one logical Gateway Record."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for entry in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from aka.experiment.measurement_store import RECORD_ID  # noqa: E402
from observability._internal.common import (  # noqa: E402
    add_output_arguments,
    discover_scopes,
    emit,
    job_attempt_rows,
    main_error,
    observed_at,
    table,
)
from observability._internal.gateway_record import gateway_record_locations  # noqa: E402


def snapshot(workspace: Path, record_id: str) -> dict[str, Any]:
    if RECORD_ID.fullmatch(record_id) is None:
        raise ValueError("--gateway-record-id must be a complete gateway- ID")
    raw_locations = gateway_record_locations(
        workspace,
        record_id,
        discover=discover_scopes,
        attempt_reader=job_attempt_rows,
    )
    locations = []
    attempts = []
    for location in raw_locations:
        rows = location["attempts"]
        locations.append(
            {
                "scope": location["scope_name"],
                "trajectory": location["trajectory"],
                "workspace": location["workspace"],
                "record_directory": location["record_directory"],
                "operation": location["operation"],
                "complete": location["complete"],
                "runtime_request_id": location["runtime_request_id"],
                "submission_attempt_count": len(rows),
            }
        )
        for row in rows:
            attempts.append(
                {
                    "scope": location["scope_name"],
                    "trajectory": location["trajectory"],
                    "operation": location["operation"],
                    **{
                        key: row[key]
                        for key in (
                            "checkpoint_path",
                            "checkpoint_id",
                            "submission",
                            "checkpoint_phase",
                            "runtime_request_ids",
                            "agate_job_id",
                            "raw_status",
                            "status",
                            "reused",
                            "shape_ids",
                            "backend_agate_job_ids",
                            "retry_counters",
                            "error",
                            "snapshot_path",
                        )
                    },
                }
            )
    if not locations:
        raise ValueError("Gateway Record is not available in this Campaign")
    status_counts = Counter(row["status"] for row in attempts)
    agate_job_ids = {row["agate_job_id"] for row in attempts if row["agate_job_id"]}
    backend_agate_job_ids = {
        item for row in attempts for item in row["backend_agate_job_ids"]
    }
    return {
        "schema_version": 1,
        "observed_at": observed_at(),
        "campaign_workspace": str(workspace.expanduser().resolve()),
        "gateway_record_id": record_id,
        "summary": {
            "record_location_count": len(locations),
            "submission_attempt_count": len(attempts),
            "agate_job_id_count": len(agate_job_ids),
            "backend_agate_job_id_count": len(backend_agate_job_ids),
            "reused_attempt_count": sum(row["reused"] for row in attempts),
            "status_counts": dict(sorted(status_counts.items())),
        },
        "locations": locations,
        "attempts": attempts,
    }


def render(value: dict[str, Any]) -> str:
    summary = value["summary"]
    lines = [
        f"Gateway Record: {value['gateway_record_id']}",
        (
            f"Attempts: {summary['submission_attempt_count']}  "
            f"Agate Job IDs: {summary['agate_job_id_count']}  "
            f"Backend IDs: {summary['backend_agate_job_id_count']}  "
            f"Status: {summary['status_counts']}"
        ),
        "",
        table(
            [
                "Scope",
                "Checkpoint",
                "Try",
                "Job ID",
                "Status",
                "Reused",
                "Error",
                "Shapes",
            ],
            [
                [
                    row["scope"],
                    row["checkpoint_path"],
                    row["submission"],
                    row["agate_job_id"],
                    row["status"],
                    row["reused"],
                    (row["error"] or {}).get("reason"),
                    ",".join(row["shape_ids"] or []),
                ]
                for row in value["attempts"]
            ],
        ),
    ]
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--gateway-record-id", required=True)
    add_output_arguments(parser)
    return parser.parse_args(argv)


def _main(argv=None) -> int:
    options = parse_args(argv)
    value = snapshot(options.workspace, options.gateway_record_id)
    emit(value, as_json=options.json, compact=options.compact, text=render(value))
    return 0


def main(argv=None) -> int:
    return main_error("light-gateway-jobs", _main, argv)


if __name__ == "__main__":
    raise SystemExit(main())
