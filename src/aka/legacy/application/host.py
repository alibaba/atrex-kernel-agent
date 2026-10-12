"""Historical application entry adapts checkout inputs to a generic launch profile."""
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
from aka.core.errors import CompositionError
from .processes import (default_recovery_profile, invocation_environment,
                        pending_recovery_directory, requires_default_recovery)
from .recovery import complete_migration
from aka.legacy.agent_sandbox.tokens import AGENT_SANDBOX

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
                    patch_files=(), patches=(), variables=None, tokens=()):
    tokens = tuple(dict.fromkeys((*tokens, AGENT_SANDBOX)))
    pending = pending_recovery_directory()
    if pending is not None:
        if (profile != "application" or Path(profiles_dir).resolve() != PROFILES_DIR.resolve()
                or patch_files or patches or variables or tokens != (AGENT_SANDBOX,)):
            raise CompositionError("recovery", "cannot override a pending default legacy launch migration")
        process_launch = sys.modules.get("orchestrator.process_launch")
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
            return run_application(request)
    optimizer = sys.modules.get("orchestrator.optimize")
    if optimizer is not None and Path(optimizer.__file__).resolve() != request.repo_root / "orchestrator/optimize.py":
        raise RuntimeError("a different AKA checkout is already loaded")
    payload = read(os.environ)
    legacy_recovery = requires_default_recovery()
    if legacy_recovery and (payload is not None or profile != "application"
            or Path(profiles_dir).resolve() != PROFILES_DIR.resolve()
            or patch_files or patches or variables or tokens != (AGENT_SANDBOX,)):
        raise CompositionError("recovery", "schema 3 recovery requires the built-in default legacy profile without overrides")
    if payload is not None:
        if profile != "application" or Path(profiles_dir) != PROFILES_DIR or patch_files or patches or variables:
            raise CompositionError("continuation", "cannot override a reconstructed launch selection")
        selection = restore(payload, tokens=tokens)
        process_launch = sys.modules.get("orchestrator.process_launch")
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
        process_launch = sys.modules.get("orchestrator.process_launch")
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
    with (default_recovery_profile() if legacy_recovery else nullcontext()), invocation_environment(selection.environment):
        return run_selection(selection, Invocation(request.argv))
