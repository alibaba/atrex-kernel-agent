"""Versioned process input owned by Workflow, carried by generic Invocation.argv."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from aka.contracts.workflow import (
    CampaignRequest, CampaignRuntimeSettings, ProblemSource, WorkflowSettings,
)


@dataclass(frozen=True)
class WorkflowLaunch:
    request: CampaignRequest
    workspace_base: Path
    workspace_suffix: str = ""
    restart_arguments: tuple[str, ...] = ()
    child_arguments: tuple[str, ...] = ()
    child_bindings: tuple[tuple[int, str], ...] = ()
    provided_settings: tuple[str, ...] | None = None
    provided_runtime: tuple[str, ...] | None = None

    def __post_init__(self):
        settings, runtime = self.request.settings, self.request.runtime
        if not isinstance(settings, WorkflowSettings) or not isinstance(runtime, CampaignRuntimeSettings):
            raise TypeError("launch requires typed Campaign settings")
        if not isinstance(self.request.problem, ProblemSource):
            raise TypeError("launch requires a ProblemSource")
        for name in ("platform", "framework", "arch", "agent_cli", "optimization_mode"):
            if not isinstance(getattr(settings, name), str):
                raise TypeError(f"{name} must be a string")
        for name in ("hardware",):
            if not isinstance(getattr(runtime, name), str):
                raise TypeError(f"{name} must be a string")
        for name, value in (("timeout", runtime.timeout), ("verify_repeats", runtime.verify_repeats),
                            ("shape_batch_size", runtime.shape_batch_size),
                            ("verify_run_timeout", runtime.verify_run_timeout),
                            ("fast_trials", settings.fast_trials),
                            ("framework_baseline_timeout", settings.framework_baseline_timeout)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if settings.conversion_budget_action not in {"fail", "retain_verified"}:
            raise ValueError("conversion_budget_action must be fail or retain_verified")
        if type(settings.resume_framework_candidate) is not bool:
            raise TypeError("resume_framework_candidate must be a boolean")
        if runtime.timeout > 600 or 2 * runtime.verify_repeats * runtime.verify_run_timeout + 30 > runtime.timeout:
            raise ValueError("ABBA verification must fit one allocation of at most 600 seconds")
        if not 0 <= runtime.min_improvement_pct < 100:
            raise ValueError("min_improvement_pct must be in [0, 100)")
        for name in ("convert_after", "fast_episodes", "handoff_resumes"):
            value = getattr(settings, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("restart_arguments", "child_arguments"):
            values = getattr(self, name)
            if isinstance(values, (str, bytes)) or any(not isinstance(v, str) for v in values):
                raise TypeError(f"{name} must contain strings")
            object.__setattr__(self, name, tuple(values))
        from dataclasses import fields
        for name, kind in (("provided_settings", WorkflowSettings), ("provided_runtime", CampaignRuntimeSettings)):
            values = getattr(self, name)
            if values is not None:
                allowed = {field.name for field in fields(kind)}
                if not isinstance(values, (tuple, list)) or any(not isinstance(v, str) or v not in allowed for v in values) or len(set(values)) != len(values):
                    raise ValueError(f"{name} must contain unique known fields")
                object.__setattr__(self, name, tuple(values))
        seen = set()
        for index, key in self.child_bindings:
            if type(index) is not int or not 0 <= index < len(self.child_arguments) or index in seen:
                raise ValueError("invalid child argument binding index")
            if key not in {"framework", "workspace", "suffix", "arch"}:
                raise ValueError("unknown child argument binding")
            seen.add(index)
        object.__setattr__(self, "child_bindings", tuple(tuple(v) for v in self.child_bindings))


def encode_launch(launch: WorkflowLaunch) -> tuple[str, str]:
    return ("--workflow-request", json.dumps(
        {"version": 1, **asdict(launch)}, default=lambda p: str(p) if isinstance(p, Path) else p,
        sort_keys=True, separators=(",", ":")))


def decode_launch(argv) -> WorkflowLaunch:
    if argv is None or len(argv) != 2 or argv[0] != "--workflow-request":
        raise ValueError("Workflow startup requires --workflow-request JSON; adapt legacy CLI before Bootstrap")
    values = json.loads(argv[1])
    if not isinstance(values, dict) or values.pop("version", None) != 1:
        raise ValueError("unsupported Workflow launch version")
    request = values.pop("request")
    source = request.pop("problem")
    source["directory"] = Path(source["directory"])
    if source.get("snapshot_directory") is not None:
        source["snapshot_directory"] = Path(source["snapshot_directory"])
    settings = request.pop("settings")
    values.setdefault("provided_settings", tuple(settings))
    values.setdefault("provided_runtime", tuple(request.get("runtime", {})))
    for name in ("framework_baseline_reviewers", "fast_episode_reviewers", "full_episode_reviewers"):
        if name in settings:
            settings[name] = tuple(settings[name])
    request["workspace"] = Path(request["workspace"])
    campaign = CampaignRequest(problem=ProblemSource(**source), settings=WorkflowSettings(**settings),
        runtime=CampaignRuntimeSettings(**request.pop("runtime", {})), **request)
    values["workspace_base"] = Path(values["workspace_base"])
    for name in ("restart_arguments", "child_arguments"):
        sequence = values.get(name, ())
        if not isinstance(sequence, (tuple, list)) or any(not isinstance(v, str) for v in sequence):
            raise TypeError(f"{name} must contain strings")
        values[name] = tuple(sequence)
    return WorkflowLaunch(request=campaign, **values)
