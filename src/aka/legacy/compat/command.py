"""Application command; framework options precede an explicit argument boundary."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

from aka.contracts.application import ApplicationRequest

from .host import run_application
from aka.legacy.workflow.process_environment import invocation_environment


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    boundary = argv.index("--") if "--" in argv else len(argv)
    parser = argparse.ArgumentParser(
        prog="aka optimize",
        usage="%(prog)s --repo-root PATH [EXECUTION OPTIONS] -- [OPTIMIZATION ARGS]",
        description="Run the retained optimization CLI from an explicit AKA checkout.",
        epilog="Example: aka optimize --repo-root /path/to/aka -- --op-dir /path/to/problem --platform L20N --framework Triton. Problem directories use native Atrex-Bench or SOL layouts.",
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--execution-profile", default=None)
    parser.add_argument("--execution-patch", action="append", type=Path, default=[])
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
        execution_selection = None
        if options.execution_profile is not None or options.execution_patch:
            from aka.legacy.execution.selection import selected_execution, read_execution_patches
            execution_selection = dict(selected_execution())
            if options.execution_profile is not None:
                execution_selection["profile"] = options.execution_profile
            if options.execution_patch:
                execution_selection["patches"] = [*execution_selection.get("patches", ()),
                    *read_execution_patches(options.execution_patch)]
        environment = {}
        if execution_selection is not None:
            from aka.bootstrap.continuation import read
            if read(os.environ) is not None:
                parser.error("cannot override Execution selection during continuation")
            import json
            environment["AKA_EXECUTION_SELECTION"] = json.dumps(execution_selection, sort_keys=True)
        with invocation_environment(environment):
            return run_application(ApplicationRequest(root, argv[boundary + 1:]))

    finally:
        if added:
            sys.path.remove(str(root))
