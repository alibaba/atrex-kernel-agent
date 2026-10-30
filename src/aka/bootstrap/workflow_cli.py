"""Run a canonical Workflow request through Bootstrap."""
from __future__ import annotations

import argparse
from pathlib import Path
import os

from aka.bootstrap.host import run_profile, resume
from aka.bootstrap.continuation import read, restore
from aka.contracts.startup import Invocation
from .workflow_request import register_request_options, request_arguments

from importlib.resources import files

def profile_path(value):
    if value in (None, "", "workflow"):
        return Path(str(files("aka.legacy.workflow"))) / "profiles/workflow.json"
    if value in {"observed", "sandbox", "light", "light-sandbox"}:
        return Path(str(files("aka.light.workflow"))) / "profiles" / (value + ".json")
    return Path(value).expanduser().resolve()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="aka workflow / aka optimize",
        description="Run a public Workflow request with an explicitly selected composition.")
    parser.add_argument("--repo-root", required=True, type=Path,
                        help="explicit public tool resource root for the selected profile")
    parser.add_argument("--profile", help="workflow, light, light-sandbox, or a launch profile path")
    parser.add_argument("--patch", type=Path, action="append", default=[])
    register_request_options(parser)
    options = parser.parse_args(argv)
    try:
        arguments = request_arguments(options, parser)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    root = options.repo_root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"tool resource directory does not exist: {root}")
    payload = read(os.environ)
    if payload is not None:
        if options.patch or options.profile is not None:
            parser.error("cannot override a reconstructed launch selection")
        if options.workflow_request is None:
            parser.error("continuation requires its recorded --workflow-request")
        selection = restore(payload)
        if selection.composition.variables.get("repo_root") != str(root):
            parser.error("a different tool resource root is requested by continuation")
        return resume(Invocation(arguments))
    return run_profile(profile_path(options.profile), Invocation(arguments),
        patch_files=options.patch, variables={"repo_root": str(root)})


if __name__ == "__main__":
    raise SystemExit(main())
