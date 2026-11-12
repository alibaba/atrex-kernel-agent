#!/usr/bin/env python3
"""Report submit-kernel decisions and exact full-Evaluate evidence."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
from observability.common import add_output_arguments, emit, main_error, observed_at, table
from observability.submissions import campaign_submission_snapshot


def snapshot(workspace: Path) -> dict:
    value = campaign_submission_snapshot(workspace)
    return {"schema_version": 1, "observed_at": observed_at(),
            "campaign_workspace": str(Path(workspace).expanduser().resolve()),
            **{key: value[key] for key in ("submissions", "accepted_submissions", "lineage_best", "global_best")}}


def render(value: dict) -> str:
    return "\n".join(["AKA Light Kernel submissions", "", table(
        ["Scope", "Episode", "Commit", "Accepted", "Evidence", "GeoMean(us)", "Record", "Diagnostics"],
        [[row["scope"], row["episode"], row["kernel_commit_id"], row["accepted"], row["evidence_status"],
          round(row["latency_us_geomean"], 6) if row["latency_us_geomean"] else None,
          row["gateway_record_id"], len(row["diagnostics"])] for row in value["submissions"]])])


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
    return main_error("light-submission-status", _main, argv)


if __name__ == "__main__":
    raise SystemExit(main())
