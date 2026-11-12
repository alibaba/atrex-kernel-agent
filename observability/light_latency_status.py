#!/usr/bin/env python3
"""Report accepted-submission and observed full-Evaluate latency."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
from observability.submissions import (
    accepted_best as _accepted_best,
    best_passing_full_eval as _best_passing_full_eval,
    eligible_full_eval,
    full_eval_rows,
)
from observability.common import discover_scopes


def measurement_workspaces(workspace: Path, campaign_root: Path | None = None) -> list[Path]:
    """Compatibility helper returning every Campaign measurement scope."""
    del campaign_root
    return [scope.workspace for scope in discover_scopes(workspace)]


def accepted_best(workspace: Path, campaign_root: Path | None = None) -> dict | None:
    """Return the fastest verified accepted submit-kernel decision."""
    del campaign_root
    return _accepted_best(workspace)


def best_passing_full_eval(workspace: Path, campaign_root: Path | None = None) -> dict | None:
    """Return the fastest passing standard full Evaluate, submitted or not."""
    del campaign_root
    return _best_passing_full_eval(workspace)


def campaign_snapshot(workspace: Path) -> dict:
    workspace = workspace.expanduser().resolve()
    if not workspace.is_dir():
        raise ValueError(f"Campaign workspace does not exist: {workspace}")
    return {"schema_version": 3, "observed_at": datetime.now(timezone.utc).isoformat(),
            "campaign_workspace": str(workspace), "accepted_best": accepted_best(workspace),
            "best_passing_full_eval": best_passing_full_eval(workspace)}


def render(snapshot: dict, *, compact: bool) -> str:
    return json.dumps(snapshot, sort_keys=True, separators=(",", ":") if compact else None,
                      indent=None if compact else 2)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--watch", type=float, metavar="SECONDS")
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--json", action="store_true")
    options = parser.parse_args(argv)
    if options.watch is not None and (not math.isfinite(options.watch) or options.watch <= 0):
        parser.error("--watch must be a finite positive number of seconds")
    return options


def main(argv=None) -> int:
    options = parse_args(argv)
    while True:
        try:
            print(render(campaign_snapshot(options.workspace), compact=options.compact), flush=True)
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"light-latency-status: {error}", file=sys.stderr, flush=True)
            if options.watch is None:
                return 1
        if options.watch is None:
            return 0
        time.sleep(options.watch)


if __name__ == "__main__":
    raise SystemExit(main())
