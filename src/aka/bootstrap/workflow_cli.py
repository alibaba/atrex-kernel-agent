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
    if value in (None, ""):
        value = "light"
    if value == "workflow":
        return Path(str(files("aka.workflow.legacy"))) / "profiles/workflow.json"
    if value in {"observed", "sandbox", "light", "light-sandbox"}:
        try:
            path = Path(str(files("aka.workflow.light"))) / "profiles" / (value + ".json")
        except ImportError as exc:
            raise ValueError(
                f"Launch profile {value!r} requires matching atrex-aka-light; "
                "install it or explicitly select --profile workflow for Legacy") from exc
        if not path.is_file():
            raise ValueError(f"Packaged launch profile is missing: {path}")
        return path
    return Path(value).expanduser().resolve()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="aka workflow / aka optimize",
        description="Run a public Workflow request with an explicitly selected composition.")
    parser.add_argument("--repo-root", required=True, type=Path,
                        help="explicit public tool resource root for the selected profile")
    parser.add_argument("--profile", help="launch profile (default: light; workflow selects Legacy)")
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
    try:
        profile = profile_path(options.profile)
    except ValueError as error:
        parser.error(str(error))
    return run_profile(profile, Invocation(arguments),
        patch_files=options.patch, variables={"repo_root": str(root)})


if __name__ == "__main__":
    raise SystemExit(main())
