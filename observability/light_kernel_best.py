#!/usr/bin/env python3
"""Report Campaign-wide accepted Kernel winners by GeoMean and by Shape."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from observability._internal.common import add_output_arguments, emit, main_error, observed_at, table
from observability._internal.submissions import campaign_submission_snapshot
from observability._internal.view_helpers import evidence_health


def snapshot(workspace: Path) -> dict:
    submission = campaign_submission_snapshot(workspace)
    accepted = submission["accepted_submissions"]
    geomean_best = min(
        accepted,
        key=lambda row: (row["latency_us_geomean"], row["kernel_commit_id"]),
        default=None,
    )
    shape_best: dict[str, dict] = {}
    for row in accepted:
        for shape_id, raw_latency in (row.get("latency_us_by_shape") or {}).items():
            if type(raw_latency) not in {int, float}:
                continue
            latency = float(raw_latency)
            previous = shape_best.get(str(shape_id))
            if previous is None or latency < previous["latency_us"]:
                shape_best[str(shape_id)] = {
                    "shape_id": str(shape_id),
                    "latency_us": latency,
                    "kernel_commit_id": row["kernel_commit_id"],
                    "gateway_record_id": row["gateway_record_id"],
                    "lineage": row["scope"],
                }
    return {
        "schema_version": 1,
        "observed_at": observed_at(),
        "campaign_workspace": str(Path(workspace).expanduser().resolve()),
        "accepted_submission_count": len(accepted),
        "geomean_best": geomean_best,
        "shape_best": {key: shape_best[key] for key in sorted(shape_best)},
        "evidence_health": evidence_health(workspace, submission),
    }


def render(value: dict) -> str:
    best = value.get("geomean_best") or {}
    shape_rows = [
        [shape, row["latency_us"], row["kernel_commit_id"], row["lineage"], row["gateway_record_id"]]
        for shape, row in value["shape_best"].items()
    ]
    return "\n".join(
        [
            "AKA Light accepted Kernel winners",
            "",
            table(
                ["Metric", "Latency(us)", "Commit", "Lineage", "Gateway Record"],
                [[
                    "GeoMean",
                    best.get("latency_us_geomean"),
                    best.get("kernel_commit_id"),
                    best.get("scope"),
                    best.get("gateway_record_id"),
                ]],
            ),
            "",
            "Per-Shape accepted winners",
            table(["Shape", "Latency(us)", "Commit", "Lineage", "Gateway Record"], shape_rows),
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
    return main_error("light-kernel-best", _main, argv)


if __name__ == "__main__":
    raise SystemExit(main())
