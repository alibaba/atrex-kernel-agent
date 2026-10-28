"""Select Workflow-owned rules and validate profile defaults before assembly."""
from dataclasses import dataclass, fields
import math
from typing import get_origin, get_type_hints

from aka.contracts.workflow import WorkflowSettings, CampaignRuntimeSettings
from .constants import CONVERT_PERF_TOL, AGENT_CLI_CHOICES

GOAL_AFTER_EPISODES = 50
GOAL_STALL_THRESHOLD = 3
GOAL_HANDOFF_RESUMES = 20

_CHOICES = {
    "agent_cli": AGENT_CLI_CHOICES,
    "optimization_mode": ("leaderboard", "production"),
    "framework_baseline": ("auto", "always", "never"),
    "gateway_candidate_recovery": ("disabled", "typed-once"),
    "conversion_budget_action": ("fail", "retain_verified"),
    "review_trace_mode": ("off", "errors", "all"),
    "long_reviewer_session": ("", "codex", "qoder", "claude"),
}
_POSITIVE = {"fast_trials", "framework_baseline_timeout", "timeout", "verify_repeats",
             "verify_run_timeout", "shape_batch_size"}
_NONNEGATIVE = {"convert_after", "fast_episodes", "handoff_resumes"}


def defaults_schema(kind):
    properties = {}
    for name, annotation in get_type_hints(kind).items():
        if get_origin(annotation) is tuple:
            schema = {"type": "array", "items": {"type": "string", "enum": ["codex", "qoder"]}}
        else:
            schema = {"type": {str: "string", int: "integer", float: "number", bool: "boolean"}[annotation]}
            if name in _POSITIVE or name in _NONNEGATIVE:
                schema["minimum"] = 1 if name in _POSITIVE else 0
            if name == "min_improvement_pct":
                schema["minimum"] = 0
                schema["maximum"] = 100
            if name in _CHOICES:
                schema["enum"] = list(_CHOICES[name])
        properties[name] = schema
    return {"type": "object", "properties": properties, "additionalProperties": False}


def validate_defaults(kind, values):
    if not isinstance(values, dict):
        raise TypeError(f"{kind.__name__} profile defaults must be an object")
    annotations = get_type_hints(kind)
    for name, value in values.items():
        if name not in annotations:
            raise ValueError(f"unknown {kind.__name__} profile field: {name}")
        annotation = annotations[name]
        if get_origin(annotation) is tuple:
            if not isinstance(value, (tuple, list)) or any(v not in ("codex", "qoder") for v in value):
                raise ValueError(f"{name} requires supported reviewer names")
        elif annotation is float:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} requires a finite number")
        elif type(value) is not annotation:
            raise TypeError(f"{name} requires {annotation.__name__}")
        if (name in _POSITIVE or name in _NONNEGATIVE) and value < (1 if name in _POSITIVE else 0):
            raise ValueError(f"{name} is outside its supported range")
        if name == "min_improvement_pct" and not 0 <= value < 100:
            raise ValueError("min_improvement_pct must be in [0, 100)")
        if name in _CHOICES and value not in _CHOICES[name]:
            raise ValueError(f"unsupported {name}: {value!r}")


@dataclass(frozen=True)
class LegacyRuleSet:
    conversion_tolerance: float = CONVERT_PERF_TOL
    goal_after_episodes: int = GOAL_AFTER_EPISODES
    goal_stall_threshold: int = GOAL_STALL_THRESHOLD
    goal_handoff_resumes: int = GOAL_HANDOFF_RESUMES

    def __post_init__(self):
        if type(self.conversion_tolerance) not in (int, float) or not math.isfinite(self.conversion_tolerance) or self.conversion_tolerance < 0:
            raise ValueError("conversion_tolerance must be finite and non-negative")
        for name in ("goal_after_episodes", "goal_stall_threshold", "goal_handoff_resumes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    def assemble(self, **dependencies):
        from .composition import compose_workflow
        return compose_workflow(rule_set=self, **dependencies)


def select_rule_set(config):
    identity = config.get("rule_set", "legacy-v1")
    if identity != "legacy-v1":
        raise ValueError(f"unsupported Workflow rule_set: {identity!r}")
    return LegacyRuleSet(**{field.name: config[field.name] for field in fields(LegacyRuleSet) if field.name in config})


def validate_config(config):
    select_rule_set(config)
    for key, kind in (("settings_defaults", WorkflowSettings), ("runtime_defaults", CampaignRuntimeSettings)):
        validate_defaults(kind, config.get(key, {}))
