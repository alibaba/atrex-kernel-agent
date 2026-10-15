"""Historical application entry adapts checkout inputs to a generic launch profile."""
from __future__ import annotations

import os
import sys
import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

from aka.bootstrap.continuation import read, restore
from aka.bootstrap.host import run_selection
from aka.bootstrap.profile import resolve_profile
from aka.contracts.startup import Invocation
from aka.core.errors import CompositionError
from .processes import (default_recovery_profile, invocation_environment,
                        pending_recovery_directory, requires_default_recovery)
from .recovery import complete_migration
from .execution import SELECTION_ENV as EXECUTION_SELECTION_ENV

PROFILES_DIR = Path(__file__).with_name("profiles")
COMPOSITION_VARS = ("repo_root", "workspace", "campaign_name", "operator", "platform", "arch",
                    "framework", "optimization_mode")


def application_variables(repo_root, values=None):
    values = dict(values or {})
    if set(values) - set(COMPOSITION_VARS):
        raise CompositionError("variables", "unsupported application variables")
    root = Path(repo_root).resolve()
    if "repo_root" in values and (not isinstance(values["repo_root"], str)
            or not values["repo_root"].strip() or Path(values["repo_root"]).resolve() != root):
        raise CompositionError("variables", "repo_root conflicts with explicit repo_root")
    return {**{name: str(values.get(name, "")) for name in COMPOSITION_VARS}, "repo_root": str(root)}


def run_application(request, *, profile="application", profiles_dir=PROFILES_DIR,
                    patch_files=(), patches=(), variables=None, tokens=(), execution_selection=None, canonical=False):
    pending = pending_recovery_directory()
    if pending is not None:
        if (profile != "application" or Path(profiles_dir).resolve() != PROFILES_DIR.resolve()
                or patch_files or patches or variables or tokens or execution_selection is not None):
            raise CompositionError("recovery", "cannot override a pending default legacy launch migration")
        from . import launch as process_launch
        script = (process_launch.optimizer_entrypoint() if process_launch is not None
                  else request.repo_root / "orchestrator/optimize.py")
        environment = complete_migration(pending,
            owner=os.environ.get("ATREX_ENVIRONMENT_RECOVERY_OWNER", "1") != "0",
            repo_root=request.repo_root, environment=os.environ,
            configuration={"environment_state_file": str(Path(os.environ["ATREX_ENVIRONMENT_STATE_FILE"]).expanduser().resolve()),
                "cwd": str(Path.cwd().resolve()), "command": [str(Path(sys.executable).resolve()),
                str(Path(script).resolve()), *(sys.argv[1:] if request.argv is None else request.argv)]})
        if environment is None:
            raise RuntimeError("recovery migration completed concurrently; retry with the recorded launch selection")
        with invocation_environment(environment):
            return run_application(request, canonical=canonical)
    resources = sys.modules.get("aka.legacy.resources.paths")
    if resources is not None and resources.REPO_ROOT != request.repo_root.resolve():
        metadata = resources.REPO_ROOT / ".internal-runtime.json"
        try:
            source_root = Path(json.loads(metadata.read_text())["open_root"]).resolve()
        except (OSError, ValueError, KeyError, TypeError):
            source_root = None
        if source_root != request.repo_root.resolve():
            raise RuntimeError("a different AKA checkout is already loaded")
    payload = read(os.environ)
    legacy_recovery = requires_default_recovery()
    if legacy_recovery and (payload is not None or profile != "application"
            or Path(profiles_dir).resolve() != PROFILES_DIR.resolve()
            or patch_files or patches or variables or tokens or execution_selection is not None):
        raise CompositionError("recovery", "schema 3 recovery requires the built-in default legacy profile without overrides")
    if payload is not None:
        if (profile != "application" or Path(profiles_dir) != PROFILES_DIR or patch_files
                or patches or variables or execution_selection is not None):
            raise CompositionError("continuation", "cannot override a reconstructed launch selection")
        selection = restore(payload, tokens=tokens)
        from . import launch as process_launch
        script = (process_launch.optimizer_entrypoint() if process_launch is not None
                  else request.repo_root / "orchestrator/optimize.py")
        expected_script = selection.bindings.get("optimizer_script")
        if expected_script is None:
            raise CompositionError("continuation", "legacy startup requires an optimizer_script binding")
        if Path(expected_script).resolve() != Path(script).resolve():
            raise RuntimeError("reconstructed launch selected a different optimizer entrypoint")
        if selection.composition.variables.get("repo_root") != str(request.repo_root):
            raise RuntimeError("a different AKA checkout is requested by continuation")
    else:
        selection = resolve_profile(Path(profiles_dir) / f"{profile}.json",
            patch_files=patch_files, patches=patches, tokens=tokens,
            variables=application_variables(request.repo_root, variables))
        # A selected internal launcher is an explicit process dependency. Avoid
        # importing the optimizer during assembly or modifying its module identity.
        from . import launch as process_launch
        script = (process_launch.optimizer_entrypoint() if process_launch is not None
                  else request.repo_root / "orchestrator/optimize.py")
        script = Path(script).resolve()
        paths = [script.name]
        if script != request.repo_root / "orchestrator/optimize.py":
            # Hash static internal adapters and selected wiki; generated runtime
            # views and trace records are outputs, not implementation identity.
            paths.extend(name for name in ("scripts", "gpu-wiki", "skills", "runtime_contract")
                         if (script.parent / name).exists())
        bindings = process_launch.optimizer_resources() if process_launch is not None else ()
        bound_resources = tuple((path.parent, (path.name,)) if path.is_file() else (path, (".",))
                                for path in bindings)
        selection = replace(selection, resources=(*selection.resources, *bound_resources, (script.parent, tuple(paths))),
                            bindings={**selection.bindings, "optimizer_script": str(script)})
    if execution_selection is not None:
        if not any(row.name in {"aka.legacy.application.plugin", "aka.legacy.workflow.plugin"} for row in selection.composition.enabled):
            raise CompositionError("application", "execution selection requires a Workflow launch")
        selection = replace(selection, environment={**selection.environment,
            EXECUTION_SELECTION_ENV: json.dumps(dict(execution_selection), sort_keys=True)})
    with (default_recovery_profile() if legacy_recovery else nullcontext()), invocation_environment(selection.environment):
        if canonical or (request.argv and request.argv[0] == "--workflow-request"):
            from aka.legacy.workflow.invocation import decode_launch
            decode_launch(request.argv)
            argv = request.argv
        else:
            from aka.legacy.compat import cli as legacy_cli
            argv = legacy_cli.adapt_argv(request.argv)
            if payload is None:
                # Raw restart arguments are re-adapted by the external caller.
                # Freeze that caller's conversion too, without making normal
                # Workflow composition depend on the compatibility package.
                adapter = Path(legacy_cli.__file__).parent
                selection = replace(selection, resources=(*selection.resources,
                    (adapter, ("cli.py", "workflow.py", "problem.py", "configuration.py"))))
        return run_selection(selection, Invocation(argv))
