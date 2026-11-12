"""Explicit optimization requests; legacy entry conversion lives outside this contract."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
from collections.abc import Callable

from .evaluation import (
    DEFAULT_ABBA_REPEATS,
    DEFAULT_EVALUATION_STEP_TIMEOUT,
    DEFAULT_SHAPE_BATCH_SIZE,
    derived_evaluation_timeout,
)


@dataclass(frozen=True)
class WorkflowSettings:
    platform: str
    framework: str
    specify_dsl: bool = True
    expose_evaluator_shapes: bool = True
    # Operator-supplied read-only engineering references.  The CLI derives each
    # stable mount name from the directory basename and freezes absolute paths.
    agent_reference_paths: tuple[tuple[str, str], ...] = ()
    notes: str = "none"
    arch: str = ""
    max_iters: int = 20
    token_budget: int = 0
    target_util: float = 90.0
    setup_timeout: int = 7200
    max_stall: int = 0
    fast_episodes: int = 2
    fast_trials: int = 5
    convert_after: int = 3
    agent_cli: str = "claude"
    optimization_mode: str = "leaderboard"
    # Translate a reference-wrapper V0 into the target framework before normal
    # optimization Episodes. ``never`` remains an explicit opt-out.
    framework_baseline: str = "auto"
    gateway_candidate_recovery: str = "disabled"
    framework_baseline_timeout: int = 28800
    handoff_resumes: int = 2
    review_trace_mode: str = "errors"
    long_reviewer_session: str = ""
    framework_baseline_reviewers: tuple[str, ...] = ()
    fast_episode_reviewers: tuple[str, ...] = ()
    full_episode_reviewers: tuple[str, ...] = ()
    conversion_budget_action: str = "fail"
    resume_framework_candidate: bool = False

    def __post_init__(self):
        if type(self.specify_dsl) is not bool:
            raise TypeError("specify_dsl must be a boolean")
        references = tuple(tuple(item) for item in self.agent_reference_paths)
        names = []
        for item in references:
            if len(item) != 2 or not all(isinstance(value, str) for value in item):
                raise TypeError("agent_reference_paths must contain (name, path) pairs")
            name, raw_path = item
            path = Path(raw_path).expanduser()
            if (
                not name
                or name in {".", ".."}
                or "/" in name
                or "\\" in name
                or not path.is_absolute()
                or not path.is_dir()
            ):
                raise ValueError(
                    "agent_reference_paths require unique safe names and existing absolute directories"
                )
            names.append(name)
        if len(names) != len(set(names)):
            raise ValueError("agent_reference_paths contain duplicate mount names")
        object.__setattr__(self, "agent_reference_paths", references)
        for name in ("setup_timeout", "framework_baseline_timeout"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.review_trace_mode not in {"off", "errors", "all"}:
            raise ValueError(
                "review_trace_mode must be one of: off, errors, all"
            )
        if type(self.expose_evaluator_shapes) is not bool:
            raise TypeError("expose_evaluator_shapes must be a boolean")


@dataclass(frozen=True)
class ProblemSource:
    """Problem location and optional retained-input location, resolved by Workflow."""

    directory: Path
    snapshot_directory: Path | None = None


@dataclass(frozen=True)
class CampaignRuntimeSettings:
    """Selected execution and measurement settings for one Campaign."""
    hardware: str = ""
    # None derives the enclosing allocation from the inner Eval/ABBA limits.
    # An explicit value remains an advanced operator override.
    timeout: int | None = None
    verify_repeats: int = DEFAULT_ABBA_REPEATS
    evaluation_step_timeout: int = DEFAULT_EVALUATION_STEP_TIMEOUT
    min_improvement_pct: float = 0.0
    shape_batch_size: int = DEFAULT_SHAPE_BATCH_SIZE
    # None selects a verified product default when one is known (L20N: 2370).
    # Eval always requests clock locking; this field selects the graphics bin.
    gpu_clock_mhz: int | None = None

    def __post_init__(self):
        if self.timeout is None:
            object.__setattr__(self, "timeout", derived_evaluation_timeout(
                repeats=self.verify_repeats,
                step_timeout=self.evaluation_step_timeout,
            ))
        if (
            type(self.evaluation_step_timeout) is not int
            or self.evaluation_step_timeout <= 0
        ):
            raise ValueError("evaluation_step_timeout must be a positive integer")
        if type(self.shape_batch_size) is not int or self.shape_batch_size <= 0:
            raise ValueError("shape_batch_size must be a positive integer")
        if self.gpu_clock_mhz is not None and (
            type(self.gpu_clock_mhz) is not int or self.gpu_clock_mhz <= 0
        ):
            raise ValueError("gpu_clock_mhz must be a positive integer")



@dataclass(frozen=True)
class CampaignRequest:
    problem: ProblemSource
    workspace: Path
    name: str
    settings: WorkflowSettings
    runtime: CampaignRuntimeSettings = field(default_factory=CampaignRuntimeSettings)


class Campaign(Protocol):
    workspace: Path

    def prepare(self, *, recovery_ready: Callable[[], None] | None = None) -> None: ...
    def run(self) -> str: ...


class Workflow(Protocol):
    def campaign(self, request: CampaignRequest) -> Campaign: ...


class WorkflowObserver(Protocol):
    def run_started(self, launch: object) -> None: ...
    def run_finished(self, exit_code: int, error: str = "") -> None: ...

    def campaign_started(self, request: CampaignRequest) -> None:
        """Observe the resolved workspace before preparation; auxiliary failures fail open."""
        ...
