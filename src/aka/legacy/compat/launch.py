"""Launch the retained optimization CLI through Bootstrap."""
from __future__ import annotations

import os
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

from aka.bootstrap.continuation import read, restore
from aka.bootstrap.host import run_selection
from aka.bootstrap.profile import resolve_profile
from aka.contracts.startup import Invocation
from aka.contracts.workflow_launch import decode_launch
from aka.core.errors import CompositionError
from aka.legacy.workflow.process_environment import (
    default_recovery_profile, invocation_environment,
    pending_recovery_directory, requires_default_recovery,
)
from aka.bootstrap.recovery_metadata import complete_migration

PROFILES_DIR = Path(__file__).parents[1] / "workflow" / "profiles"


def run_workflow(repo_root, argv):
    root, arguments = Path(repo_root).resolve(), tuple(argv)
    launch = decode_launch(arguments)
    script = root / "orchestrator" / "optimize.py"
    if not script.is_file():
        script = Path(__file__).with_name("entry.py")
    pending = pending_recovery_directory()
    if pending is not None:
        environment = complete_migration(pending,
            owner=os.environ.get("ATREX_ENVIRONMENT_RECOVERY_OWNER", "1") != "0",
            repo_root=root, environment=os.environ,
            configuration={"environment_state_file": str(Path(os.environ["ATREX_ENVIRONMENT_STATE_FILE"]).expanduser().resolve()),
                "cwd": str(Path.cwd().resolve()), "command": [str(Path(sys.executable).resolve()),
                    str(script.resolve()), *(launch.restart_arguments or arguments)]})
        if environment is None:
            raise RuntimeError("recovery migration completed concurrently; retry with the recorded launch selection")
        with invocation_environment(environment):
            return run_workflow(root, arguments)
    resources = sys.modules.get("aka.legacy.resources.paths")
    if resources is not None and resources.REPO_ROOT != root:
        raise RuntimeError("a different AKA checkout is already loaded")
    payload = read(os.environ)
    legacy_recovery = requires_default_recovery()
    if legacy_recovery and payload is not None:
        raise CompositionError("recovery", "schema 3 recovery requires the built-in default legacy profile")
    if payload is not None:
        selection = restore(payload)
        if selection.composition.variables.get("repo_root") != str(root):
            raise RuntimeError("a different AKA checkout is requested by continuation")
    else:
        selection = resolve_profile(PROFILES_DIR / "workflow.json", variables={"repo_root": str(root)})
        # The retained CLI is a real external recovery entry, not a pluggable launcher.
        selection = replace(selection, resources=(*selection.resources,
                            (script.parent, (script.name,)), (Path(__file__).parent, ("launch.py", "entry.py", "host.py"))))
    with (default_recovery_profile() if legacy_recovery else nullcontext()), invocation_environment(selection.environment):
        return run_selection(selection, Invocation(arguments))
