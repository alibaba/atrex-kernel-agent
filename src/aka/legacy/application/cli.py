"""Application command; framework options precede an explicit argument boundary."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from aka.contracts.application import ApplicationRequest

from .host import PROFILES_DIR, run_application


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    boundary = argv.index("--") if "--" in argv else len(argv)
    parser = argparse.ArgumentParser(
        prog="aka optimize",
        usage="%(prog)s --repo-root PATH [FRAMEWORK OPTIONS] -- [OPTIMIZATION ARGS]",
        description="Run the selected application from an explicit AKA checkout.",
        epilog="Example: aka optimize --repo-root /path/to/aka -- --op-dir /path/to/problem --platform L20N --framework Triton. Problem directories use native Atrex-Bench or SOL layouts.",
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--profile", default="application")
    parser.add_argument("--profiles-dir", type=Path, default=PROFILES_DIR)
    parser.add_argument("--patch", action="append", type=Path, default=[])
    options = parser.parse_args(argv[:boundary])
    if boundary == len(argv):
        parser.error("separate application arguments with -- (use -- --help for application help)")
    root = options.repo_root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"checkout directory does not exist: {root}")
    # An explicit checkout is a declared runtime dependency, not the current cwd.
    # Keep this import path alive during application execution, including lazy imports.
    added = str(root) not in sys.path
    if added:
        sys.path.insert(0, str(root))
    try:
        return run_application(
            ApplicationRequest(root, argv[boundary + 1:]),
            profile=options.profile,
            profiles_dir=options.profiles_dir.expanduser().resolve(),
            patch_files=tuple(path.expanduser().resolve() for path in options.patch),
        )
    finally:
        if added:
            sys.path.remove(str(root))
