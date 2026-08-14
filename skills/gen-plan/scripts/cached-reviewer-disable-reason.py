#!/usr/bin/env python3
"""Print a campaign-cached disable reason, or nothing when a reviewer may run."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


CACHE_RELATIVE_PATH = Path(
    ".atrex_long_horizon/plan_reviewer_availability.json"
)


def _campaign_workspace() -> Path | None:
    completed = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        return None
    common = Path(completed.stdout.strip())
    return common.parent if common.name == ".git" else None


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in {"codex", "qoder"}:
        return 2
    workspace = _campaign_workspace()
    if workspace is None:
        return 0
    try:
        value = json.loads(
            (workspace / CACHE_RELATIVE_PATH).read_text(encoding="utf-8")
        )
        record = value["reviewers"][argv[1]]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError):
        return 0
    if record.get("available") is False:
        reason = " ".join(str(record.get("reason") or "startup probe failed").split())
        print(reason[:500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
