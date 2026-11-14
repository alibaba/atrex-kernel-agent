"""Build native Workflow inputs while retaining Workflow-owned path conventions."""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
import re

from aka.contracts.workflow import CampaignRequest, CampaignRuntimeSettings, ProblemSource, WorkflowSettings
from aka.contracts.workflow_launch import WorkflowLaunch


def problem_source(directory, optimization_mode, *, workspace, suffix, expose_evaluator_shapes=True):
    """Select the existing retained-input namespace without preparing or reading Task."""
    origin = Path(directory).resolve()
    identity = str(origin) + "\0" + optimization_mode + "\0" + suffix
    if optimization_mode == "production" and expose_evaluator_shapes:
        identity += "\0expose-evaluator-shapes"
    key = hashlib.sha256(identity.encode()).hexdigest()[:24]
    return ProblemSource(origin, Path(workspace) / ".aka-problems" / key)


def create_launch(problem_directory: Path, *, workspace_base: Path,
                  settings: WorkflowSettings, runtime: CampaignRuntimeSettings | None = None,
                  workspace_suffix: str = "", provided_settings: tuple[str, ...] | None = None,
                  provided_runtime: tuple[str, ...] | None = None) -> WorkflowLaunch:
    """Construct a public request; None provided-fields means values are already frozen.

    Native frontends pass their explicit field names to allow the selected
    composition to supply unspecified defaults. Restores replay the recorded
    WorkflowLaunch instead of calling this constructor with current defaults.
    """
    if not isinstance(settings, WorkflowSettings):
        raise TypeError("settings require WorkflowSettings")
    if runtime is not None and not isinstance(runtime, CampaignRuntimeSettings):
        raise TypeError("runtime requires CampaignRuntimeSettings")
    if workspace_suffix and re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", workspace_suffix) is None:
        raise ValueError("workspace suffix must be normalized lowercase alphanumeric/underscore")
    base = Path(workspace_base).expanduser().resolve()
    source = problem_source(problem_directory, settings.optimization_mode, workspace=base,
                            suffix=workspace_suffix or f"{settings.framework}_{settings.platform}",
                            expose_evaluator_shapes=settings.expose_evaluator_shapes)
    name = source.directory.name + ("_" + workspace_suffix if workspace_suffix else "")
    effective_runtime = runtime if runtime is not None else CampaignRuntimeSettings()
    effective_runtime = replace(effective_runtime, hardware=effective_runtime.hardware or settings.platform)
    return WorkflowLaunch(CampaignRequest(source, base / ("kernel_opt_" + name), name, settings,
                          effective_runtime),
                          base, workspace_suffix, provided_settings=provided_settings,
                          provided_runtime=provided_runtime)
