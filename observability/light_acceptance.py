#!/usr/bin/env python3
"""Compatibility alias for light_submission_status.py."""
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from observability.light_submission_status import main, render, snapshot

if __name__ == "__main__":
    raise SystemExit(main())
