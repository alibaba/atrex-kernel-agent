"""Explicit optimization requests; legacy entry conversion lives outside this contract."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
from collections.abc import Callable


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
class ProblemSource:
    """Problem location and optional retained-input location, resolved by Workflow."""

    directory: Path
    snapshot_directory: Path | None = None


@dataclass(frozen=True)
class CampaignRuntimeSettings:
    """Selected execution and measurement settings for one Campaign."""
    hardware: str = ""
    profile: str = ""
    url: str = ""
    timeout: int = 600
    verify_repeats: int = 2
    verify_run_timeout: int = 120
    min_improvement_pct: float = 0.0
    ssh: str = ""
    ssh_init: str = ""
    ssh_gpu: int | None = None
    health_command: str = ""



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
