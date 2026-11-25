"""Explicit AKA Light optimization requests."""

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
    # Operator-supplied engineering references. Bubblewrap mounts them read-only;
    # native execution links them without a filesystem protection boundary.
    agent_reference_paths: tuple[tuple[str, str], ...] = ()
    notes: str = "none"
    arch: str = ""
    max_iters: int = 20
    token_budget: int = 0
    target_util: float = 90.0
    setup_timeout: int = 7200
    agent_cli: str = "claude"
    agent_model: str | None = None
    optimization_mode: str = "fast"

    def __post_init__(self):
        if self.agent_model is not None and (not isinstance(self.agent_model, str) or not self.agent_model.strip()):
            raise ValueError("agent_model must be a non-empty model name")
        if self.agent_model is not None:
            object.__setattr__(self, "agent_model", self.agent_model.strip())
        if type(self.specify_dsl) is not bool:
            raise TypeError("specify_dsl must be a boolean")
        if self.optimization_mode not in {
            "fast", "performance", "ultra"
        }:
            raise ValueError(
                f"unsupported optimization_mode: {self.optimization_mode!r}"
            )
        if self.optimization_mode in {"fast", "performance"}:
            object.__setattr__(self, "specify_dsl", False)
        elif self.optimization_mode == "ultra":
            if not self.specify_dsl or not self.framework.strip():
                raise ValueError(
                    "ultra mode requires specify_dsl=true and one explicit framework"
                )
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
        for name in ("setup_timeout",):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
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
    warmup_iters: int = 10
    bench_iters: int = 20
    min_improvement_pct: float = 0.0
    shape_batch_size: int = DEFAULT_SHAPE_BATCH_SIZE
    # Retained for decoding existing requests. Light only requests clock locking;
    # Agate selects the graphics frequency.
    gpu_clock_mhz: int | None = None
    agate_tools: tuple[str, ...] = ("env", "evaluate", "command")

    def __post_init__(self):
        from .agate_tools import normalize_agate_tools
        object.__setattr__(self, "agate_tools", normalize_agate_tools(self.agate_tools))
        if type(self.warmup_iters) is not int or self.warmup_iters < 0:
            raise ValueError("warmup_iters must be a non-negative integer")
        if type(self.bench_iters) is not int or self.bench_iters <= 0:
            raise ValueError("bench_iters must be a positive integer")
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
