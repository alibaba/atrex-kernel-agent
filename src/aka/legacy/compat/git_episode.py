"""Compatibility adapters to task-owned Git and candidate workspace modules."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aka.legacy.task.git import (
    _git,
    git_text,
    git_head,
    working_changes,
    changed_paths,
    ignored_evidence_files,
    protected_violation,
    _manifest_deleted,
)
from aka.legacy.task.candidate_workspace.policy import (
    PROTECTED_PATHS,
    PROTECTED_PREFIXES,
    EPISODE_EVIDENCE_PREFIXES,
    CANDIDATE_PATHS,
    TIMELINE_PROBE_MARKERS,
)
from aka.contracts.workspace import CandidateHandle
from aka.legacy.application.workspace import candidate_workspace
from aka.legacy.workflow.protocol import atomic_write_json
import subprocess


@dataclass(frozen=True)
class EpisodeWorktree(CandidateHandle):
    provider: Any = field(default=None, repr=False, compare=False, kw_only=True)

    @classmethod
    def plan(
        cls, incumbent_workspace, episode, base_commit, root=None, *, provider=None
    ):
        with candidate_workspace(provider=provider) as provider:
            handle = provider.plan(incumbent_workspace, episode, base_commit, root)
        return cls(
            handle.episode,
            handle.base_commit,
            handle.branch,
            handle.path,
            provider=provider,
        )

    def materialize(self, incumbent_workspace):
        with candidate_workspace(provider=self.provider) as provider:
            return provider.materialize(self, incumbent_workspace)

    @classmethod
    def create(
        cls, incumbent_workspace, episode, base_commit, root=None, *, provider=None
    ):
        planned = cls.plan(
            incumbent_workspace, episode, base_commit, root, provider=provider
        )
        planned.materialize(incumbent_workspace)
        return planned

    def validate_candidate(self, candidate_commit):
        with candidate_workspace(provider=self.provider) as provider:
            return provider.validate_candidate(self, candidate_commit)

    def archive(self, destination, candidate_commit="HEAD"):
        with candidate_workspace(provider=self.provider) as provider:
            return provider.archive(self, destination, candidate_commit)

    def remove(self, incumbent_workspace):
        with candidate_workspace(provider=self.provider) as provider:
            return provider.remove(self, incumbent_workspace)


def promote_candidate(incumbent_workspace, *, provider=None, **options):
    with candidate_workspace(provider=provider) as provider:
        return provider.promote(incumbent_workspace, **options)


def record_episode_outcome(
    incumbent_workspace: Path,
    *,
    base_commit: str,
    version: int,
    episode: int,
    status: str,
    memory_record: dict[str, Any],
) -> str:
    """Advance main-compatible version history without changing the incumbent kernel."""
    if git_head(incumbent_workspace) != base_commit:
        raise RuntimeError("incumbent advanced during episode; refusing outcome record")
    memory_path = incumbent_workspace / "memory" / f"v{version}.json"
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(memory_path, memory_record)
    subprocess.run(
        ["git", "add", str(memory_path.relative_to(incumbent_workspace))],
        cwd=str(incumbent_workspace),
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=atrex-long-horizon",
            "-c",
            "user.email=atrex-long-horizon@local",
            "commit",
            "--only",
            "-m",
            f"v{version}: long-horizon episode {episode} {status}",
            "--",
            str(memory_path.relative_to(incumbent_workspace)),
        ],
        cwd=str(incumbent_workspace),
        check=True,
        capture_output=True,
        text=True,
    )
    return git_head(incumbent_workspace)
