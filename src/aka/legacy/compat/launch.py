"""Launch the retained optimization CLI through Bootstrap."""
from __future__ import annotations

import os
import json
from pathlib import Path

from aka.bootstrap.continuation import read, restore
from aka.bootstrap.host import run_selection
from aka.bootstrap.profile import resolve_profile
from aka.contracts.startup import Invocation
from aka.contracts.workflow_launch import decode_launch
from aka.legacy.workflow.process_environment import (
    pending_recovery_directory, requires_default_recovery,
)
from .profiles import launch_profile_path


def run_workflow(repo_root, argv, *, profile=None, patch_files=()):
    root, arguments = Path(repo_root).resolve(), tuple(argv)
    decode_launch(arguments)
    if pending_recovery_directory() is not None or requires_default_recovery():
        raise RuntimeError("retired script recovery metadata; use the installed profile continuation")
    payload = read(os.environ)
    if payload is not None:
        if profile is not None or patch_files:
            raise RuntimeError("cannot override the launch profile during continuation")
        selection = restore(payload)
        if selection.composition.variables.get("repo_root") != str(root):
            raise RuntimeError("a different tool resource root is requested by continuation")
    else:
        # Retained CLI Execution options become an explicit composition patch
        # before Bootstrap freezes the selected module and configuration.
        from aka.legacy.execution.selection import SELECTION_ENV
        configured = os.environ.get(SELECTION_ENV)
        patches = [{"id": "execution", "config": json.loads(configured)}] if configured else []
        selection = resolve_profile(
            launch_profile_path(profile),
            variables={"repo_root": str(root)},
            patches=patches,
            patch_files=patch_files,
        )
    return run_selection(selection, Invocation(arguments))
