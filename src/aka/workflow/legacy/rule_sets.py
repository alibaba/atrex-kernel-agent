"""Select Workflow-owned rules and validate profile defaults before assembly."""
from dataclasses import dataclass, fields
import math
from typing import get_origin, get_type_hints

from aka.contracts.workflow import WorkflowSettings, CampaignRuntimeSettings
from .constants import CONVERT_PERF_TOL, AGENT_CLI_CHOICES

GOAL_AFTER_EPISODES = 50
GOAL_STALL_THRESHOLD = 3
GOAL_HANDOFF_RESUMES = 20
DEFAULT_RULE_SET = "geomean-latency-v1"

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

    def create_acceptance(self):
        from aka.acceptance.policy import PerformanceScorePolicy
        return PerformanceScorePolicy(
            conversion_tolerance=self.conversion_tolerance
        )

    def promotion_directive(self, min_improvement_pct):
        return (
            "## Promotion rule: performance score (`legacy-v1`)\n\n"
            "Maximize `performance_score`, the arithmetic mean of per-workload speedups. "
            "Higher is better; full-workload GeoMean latency remains diagnostic evidence. "
            "Ordinary and Fast promotion requires complete passing correctness and a score "
            f"improvement strictly greater than {min_improvement_pct:g}% over the incumbent. "
            "Framework bring-up keeps its separate correctness gate; mandatory conversion "
            "keeps the supervisor's existing latency parity rule."
        )

    def fast_selection_values(self):
        return {
            "FAST_BEST_INITIALIZATION": (
                "At episode start, record the incumbent `HEAD`, its canonical `performance_score`, latency, and\n"
                "matching kernel/manifest pair as the initial `best_commit`, `best_score`, and `best_latency`. The score is the\n"
                "optimization objective and higher is better; latency remains diagnostic evidence. Each trial starts\n"
                "from the best passing kernel found so far, not automatically from the immediately preceding trial. A\n"
                "failed or lower-scoring trial must not contaminate the next trial."
            ),
            "FAST_BEST_COMPARISON": "`best_score`",
            "FAST_KEEP_CONDITION": (
                "If the result passes and its `performance_score` exceeds `best_score`, update `best_commit`,\n"
                "`best_score`, and `best_latency`."
            ),
            "FAST_FINAL_SELECTION": "the highest-scoring passing strict improvement",
        }


@dataclass(frozen=True)
class GeoMeanLatencyRuleSet(LegacyRuleSet):
    def create_acceptance(self):
        from aka.acceptance.policy import GeoMeanLatencyPolicy
        return GeoMeanLatencyPolicy(
            conversion_tolerance=self.conversion_tolerance
        )

    def promotion_directive(self, min_improvement_pct):
        return (
            "## Promotion rule: GeoMean latency (`geomean-latency-v1`)\n\n"
            "Minimize the complete workload's GeoMean latency; lower is better. "
            "Ordinary and Fast promotion requires complete passing correctness and positive finite "
            "candidate/incumbent GeoMean latency facts with the same workload and measurement method. "
            "Latency reduction is `(incumbent_latency - candidate_latency) / incumbent_latency * 100`; "
            f"it must be strictly greater than {min_improvement_pct:g}% over the incumbent. "
            "Keep the raw `performance_score` and latency facts in memory/journal evidence, but select "
            "candidates by GeoMean latency. Missing, incomplete, nonfinite or nonpositive latency is "
            "insufficient evidence; do not infer a candidate defect or rerun outside the authorized flow. "
            "Framework bring-up keeps its separate correctness gate; mandatory conversion "
            "keeps the supervisor's existing latency parity rule."
        )

    def fast_selection_values(self):
        return {
            "FAST_BEST_INITIALIZATION": (
                "At episode start, record the incumbent `HEAD`, its canonical `performance_score`, full-workload\n"
                "GeoMean latency, and matching kernel/manifest pair as the initial `best_commit`, `best_score`, and\n"
                "`best_latency`. GeoMean latency is the optimization objective and lower is better; `best_score`\n"
                "retains the raw score as evidence. Only complete, comparable, positive finite latency facts can\n"
                "establish a best candidate. If the incumbent latency is unavailable, report missing evidence;\n"
                "do not invent it or run an extra evaluator. Each trial starts from the best passing kernel found\n"
                "so far, not automatically from the immediately preceding trial. A failed, incomplete or slower\n"
                "trial must not contaminate the next trial."
            ),
            "FAST_BEST_COMPARISON": "`best_latency` using full-workload GeoMean latency",
            "FAST_KEEP_CONDITION": (
                "If the result passes with complete comparable positive finite full-workload GeoMean latency\n"
                "strictly below `best_latency`, update `best_commit`, `best_score`, and `best_latency`."
            ),
            "FAST_FINAL_SELECTION": "the lowest-GeoMean-latency passing candidate whose latency reduction strictly exceeds the configured promotion threshold",
        }


def select_rule_set(config):
    # Frozen configs predate the explicit rule identity. Core restores resolved
    # configs without current Defaults, so omission retains the old score rule.
    identity = config.get("rule_set", "legacy-v1")
    kinds = {"legacy-v1": LegacyRuleSet, DEFAULT_RULE_SET: GeoMeanLatencyRuleSet}
    if identity not in kinds:
        raise ValueError(f"unsupported Workflow rule_set: {identity!r}")
    return kinds[identity](**{field.name: config[field.name] for field in fields(LegacyRuleSet) if field.name in config})


def validate_config(config):
    select_rule_set(config)
    for key, kind in (("settings_defaults", WorkflowSettings), ("runtime_defaults", CampaignRuntimeSettings)):
        validate_defaults(kind, config.get(key, {}))
