"""Application command; framework options precede an explicit argument boundary."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

from aka.contracts.application import ApplicationRequest

from .host import run_application
from .profiles import DEFAULT_LAUNCH_PROFILE, launch_profile_path
from aka.legacy.workflow.process_environment import invocation_environment


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    boundary = argv.index("--") if "--" in argv else len(argv)
    parser = argparse.ArgumentParser(
        prog="aka optimize",
        usage="%(prog)s --repo-root PATH [EXECUTION OPTIONS] -- [OPTIMIZATION ARGS]",
        description="Translate the retained optimization CLI for installed AKA modules and explicit tool resources.",
        epilog="Example: aka optimize --repo-root /path/to/public-resources -- --op-dir /path/to/problem --platform L20N --framework Triton. Problem directories use native Atrex-Bench or SOL layouts.",
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument(
        "--profile",
        default=None,
        help=(f"launch profile (default: {DEFAULT_LAUNCH_PROFILE}): light, light-sandbox, "
              "observed, sandbox, workflow (Legacy), or an explicit profile JSON path"),
    )
    parser.add_argument("--patch", action="append", type=Path, default=[],
                        help="apply an additional launch-composition patch")
    parser.add_argument("--execution-profile", default=None)
    parser.add_argument("--execution-patch", action="append", type=Path, default=[])
    options = parser.parse_args(argv[:boundary])
    if boundary == len(argv):
        parser.error("separate application arguments with -- (use -- --help for application help)")
    root = options.repo_root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"tool resource directory does not exist: {root}")
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
    from aka.bootstrap.continuation import read
    continuing = read(os.environ) is not None
    if continuing and (options.profile is not None or options.patch):
        parser.error("cannot override the launch profile during continuation")
    profile = options.profile
    if not continuing:
        try:
            profile = launch_profile_path(profile)
        except ValueError as exc:
            parser.error(str(exc))
    if execution_selection is not None:
        if continuing:
            parser.error("cannot override Execution selection during continuation")
        import json
        environment["AKA_EXECUTION_SELECTION"] = json.dumps(execution_selection, sort_keys=True)
    with invocation_environment(environment):
        return run_application(
            ApplicationRequest(root, argv[boundary + 1:]),
            profile=profile,
            patch_files=options.patch,
        )
