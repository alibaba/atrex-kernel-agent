"""Run a canonical Workflow request through Bootstrap."""
from __future__ import annotations

import argparse
from pathlib import Path
import os

from aka.bootstrap.host import run_profile, resume
from aka.bootstrap.continuation import read, restore
from aka.contracts.startup import Invocation
from aka.contracts.workflow_launch import decode_launch

from importlib.resources import files

PROFILES_DIR = Path(str(files("aka.legacy.workflow"))) / "profiles"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="aka workflow")
    parser.add_argument("--repo-root", required=True, type=Path,
                        help="explicit public tool resource root for the selected profile")
    parser.add_argument("--profile", type=Path, default=PROFILES_DIR / "workflow.json")
    parser.add_argument("--patch", type=Path, action="append", default=[])
    parser.add_argument("--workflow-request", required=True)
    options = parser.parse_args(argv)
    arguments = ("--workflow-request", options.workflow_request)
    decode_launch(arguments)
    root = options.repo_root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"tool resource directory does not exist: {root}")
    payload = read(os.environ)
    if payload is not None:
        if options.patch or options.profile != PROFILES_DIR / "workflow.json":
            parser.error("cannot override a reconstructed launch selection")
        selection = restore(payload)
        if selection.composition.variables.get("repo_root") != str(root):
            parser.error("a different tool resource root is requested by continuation")
        return resume(Invocation(arguments))
    return run_profile(options.profile, Invocation(arguments),
        patch_files=options.patch, variables={"repo_root": str(root)})


if __name__ == "__main__":
    raise SystemExit(main())
