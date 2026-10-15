#!/usr/bin/env python3
"""Historical optimization command; adapt inputs before Bootstrap."""
from pathlib import Path
import sys


def main(argv=None):
    if __name__ == "__main__":
        root = str(Path(__file__).resolve().parent.parent)
        if root not in sys.path:
            sys.path.insert(0, root)
    from orchestrator._bootstrap import run
    return run(argv)


def __getattr__(name):
    # Older Python callers opt into the historical surface. Normal launch never
    # imports it, and runtime modules do not call back through this module.
    from aka.legacy.compat import optimize
    return getattr(optimize, name)


if __name__ == "__main__":
    raise SystemExit(main())
