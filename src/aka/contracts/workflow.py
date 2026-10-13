"""Explicit optimization requests; legacy entry conversion lives outside this contract."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from collections.abc import Callable
from .problem import LoadedProblem


@dataclass(frozen=True)
class WorkflowSettings:
    platform: str
    framework: str
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
    framework_baseline: str = "auto"
    framework_baseline_timeout: int = 10800
    handoff_resumes: int = 2
    production_review_timeout: int = 600
    long_reviewer_session: str = ""
    framework_baseline_reviewers: tuple[str, ...] = ()
    fast_episode_reviewers: tuple[str, ...] = ()
    full_episode_reviewers: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProblemRequest:
    """Explicit source and retained-input location; None requests an uncaptured load."""

    directory: Path
    snapshot_directory: Path | None
    optimization_mode: str = "leaderboard"


@dataclass(frozen=True)
class CampaignRequest:
    problem: LoadedProblem
    workspace: Path
    name: str
    settings: WorkflowSettings


class Campaign(Protocol):
    workspace: Path

    def prepare(self, *, recovery_ready: Callable[[], None] | None = None) -> None: ...
    def run(self) -> str: ...


class Workflow(Protocol):
    def load_problem(self, request: ProblemRequest) -> LoadedProblem:
        """Load or restore run inputs through the selected Problem module, without GPU work."""
        ...
    def campaign(self, request: CampaignRequest) -> Campaign: ...
