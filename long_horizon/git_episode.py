from __future__ import annotations

import subprocess
import uuid
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from . import main_adapter
from .protocol import atomic_write_json
from .store import CampaignStore


PROTECTED_PATHS = frozenset(
    {
        *main_adapter.IMMUTABLE_BASELINE_PATHS,
        "definition.json",
        "reference.py",
        "workload.jsonl",
        "test_kernel.py",
        "config.json",
        "input.py",
        "shapes.json",
        "metadata.json",
        "roofline.json",
        "valid.py",
        "CLAUDE.md",
        "README.md",
        ".gitignore",
    }
)
PROTECTED_PREFIXES = (
    "memory/",
    "eval/",
    ".claude/",
    ".qoder/",
    ".agents/",
    ".atrex_",
)
EPISODE_EVIDENCE_PREFIXES = ("plans/", "profiles/", ".humanize/")


def _git(workspace: Path, *args: str, check: bool = True, binary: bool = False):
    result = subprocess.run(
        ["git", *args], cwd=str(workspace), capture_output=True, text=not binary
    )
    if check and result.returncode:
        stderr = result.stderr.decode(errors="replace") if binary else result.stderr
        raise RuntimeError(f"git {' '.join(args)} failed: {str(stderr)[-1200:]}")
    return result


def git_text(workspace: Path, *args: str, check: bool = True) -> str:
    return _git(workspace, *args, check=check).stdout.strip()


def git_head(workspace: Path) -> str:
    return git_text(workspace, "rev-parse", "HEAD")


def working_changes(workspace: Path) -> list[str]:
    # Porcelain status uses its first two columns for XY state.  Preserve the
    # leading space on an unstaged first entry; git_text().strip() would remove
    # it and shift the pathname by one character.
    output = _git(workspace, "status", "--porcelain", "--untracked-files=all").stdout
    return [line[3:].strip('"') for line in output.splitlines() if len(line) >= 4]


def changed_paths(workspace: Path, base_commit: str, candidate_commit: str = "HEAD") -> list[str]:
    output = git_text(
        workspace, "diff", "--name-only", "--no-renames", base_commit, candidate_commit, "--"
    )
    return sorted(path for path in output.splitlines() if path)


def ignored_evidence_files(workspace: Path) -> list[str]:
    output = git_text(
        workspace,
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
        "--",
        *[prefix.rstrip("/") for prefix in EPISODE_EVIDENCE_PREFIXES],
    )
    return sorted(path for path in output.splitlines() if path)


def protected_violation(paths: list[str]) -> str:
    for value in paths:
        normalized = PurePosixPath(value).as_posix()
        if normalized in PROTECTED_PATHS or normalized.startswith(PROTECTED_PREFIXES):
            return f"candidate modified protected path: {normalized}"
    return ""


@dataclass(frozen=True)
class EpisodeWorktree:
    episode: int
    base_commit: str
    branch: str
    path: Path

    @classmethod
    def plan(
        cls,
        incumbent_workspace: Path,
        episode: int,
        base_commit: str,
        root: Path | None = None,
    ) -> "EpisodeWorktree":
        branch = f"atrex/long-e{episode:04d}-{uuid.uuid4().hex[:8]}"
        worktree_root = root or (
            incumbent_workspace.parent
            / ".atrex_long_horizon_worktrees"
            / incumbent_workspace.name
        )
        worktree_root.mkdir(parents=True, exist_ok=True)
        path = worktree_root / f"e{episode:04d}-{uuid.uuid4().hex[:8]}"
        return cls(episode=episode, base_commit=base_commit, branch=branch, path=path)

    def materialize(self, incumbent_workspace: Path) -> None:
        subprocess.run(
            ["git", "worktree", "add", "-b", self.branch, str(self.path), self.base_commit],
            cwd=str(incumbent_workspace), check=True, capture_output=True, text=True,
        )
        CampaignStore.ensure_excluded(self.path)

    @classmethod
    def create(
        cls,
        incumbent_workspace: Path,
        episode: int,
        base_commit: str,
        root: Path | None = None,
    ) -> "EpisodeWorktree":
        planned = cls.plan(incumbent_workspace, episode, base_commit, root)
        planned.materialize(incumbent_workspace)
        return planned

    def validate_candidate(self, candidate_commit: str) -> tuple[str, list[str]]:
        resolved = git_text(
            self.path, "rev-parse", "--verify", f"{candidate_commit}^{{commit}}", check=False
        )
        if not resolved:
            return "candidate_commit does not resolve", []
        if resolved != git_head(self.path):
            return "candidate_commit must equal episode HEAD", []
        branch = git_text(self.path, "symbolic-ref", "--quiet", "--short", "HEAD", check=False)
        if branch != self.branch:
            return "episode worktree left its isolated branch", []
        ancestor = _git(
            self.path, "merge-base", "--is-ancestor", self.base_commit, resolved, check=False
        )
        if ancestor.returncode:
            return "candidate_commit is not descended from incumbent", []
        dirty = working_changes(self.path)
        if dirty:
            return "candidate_ready requires a clean worktree: " + ", ".join(dirty[:8]), []
        paths = changed_paths(self.path, self.base_commit, resolved)
        if not paths:
            return "candidate has no changes relative to incumbent", []
        if paths != ["kernel.py"]:
            return "candidate commit may change only kernel.py", paths
        return "", paths

    def archive(self, destination: Path, candidate_commit: str = "HEAD") -> Path:
        destination.mkdir(parents=True, exist_ok=True)
        committed_patch = _git(
            self.path, "diff", "--binary", self.base_commit, candidate_commit, "--", binary=True
        ).stdout
        (destination / "candidate.patch").write_bytes(committed_patch)
        worktree_patch = _git(
            self.path, "diff", "--binary", self.base_commit, "--", binary=True
        ).stdout
        (destination / "worktree.patch").write_bytes(worktree_patch)
        archived_files = destination / "worktree_files"
        paths = set(changed_paths(self.path, self.base_commit, candidate_commit))
        paths.update(working_changes(self.path))
        paths.update(ignored_evidence_files(self.path))
        for relative in sorted(paths):
            source = self.path / relative
            if not source.is_file() or source.is_symlink():
                continue
            target = archived_files / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        atomic_write_json(
            destination / "git.json",
            {
                "episode": self.episode,
                "base_commit": self.base_commit,
                "branch": self.branch,
                "head": git_head(self.path),
                "dirty_paths": working_changes(self.path),
            },
        )
        return destination

    def remove(self, incumbent_workspace: Path) -> None:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(self.path)],
            cwd=str(incumbent_workspace), check=True, capture_output=True, text=True,
        )


def promote_candidate(
    incumbent_workspace: Path,
    *,
    base_commit: str,
    candidate_commit: str,
    episode: int,
    evidence: dict[str, Any],
    memory_version: int,
    memory_record: dict[str, Any],
) -> str:
    if git_head(incumbent_workspace) != base_commit:
        raise RuntimeError("incumbent advanced during episode; refusing promotion")
    try:
        subprocess.run(
            ["git", "merge", "--squash", "--no-commit", candidate_commit],
            cwd=str(incumbent_workspace), check=True, capture_output=True, text=True,
        )
        staged = git_text(incumbent_workspace, "diff", "--cached", "--name-only").splitlines()
        violation = protected_violation([path for path in staged if path])
        if violation:
            raise RuntimeError(violation)
        memory_dir = incumbent_workspace / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(memory_dir / f"long_horizon_e{episode:04d}.json", evidence)
        atomic_write_json(memory_dir / f"v{memory_version}.json", memory_record)
        subprocess.run(
            ["git", "add", f"memory/long_horizon_e{episode:04d}.json", f"memory/v{memory_version}.json"],
            cwd=str(incumbent_workspace), check=True,
        )
        subprocess.run(
            [
                "git", "-c", "user.name=atrex-long-horizon",
                "-c", "user.email=atrex-long-horizon@local",
                "commit", "-m", f"episode {episode}: promote verified long-horizon candidate",
            ],
            cwd=str(incumbent_workspace), check=True, capture_output=True, text=True,
        )
        return git_head(incumbent_workspace)
    except Exception:
        subprocess.run(
            ["git", "reset", "--hard", base_commit], cwd=str(incumbent_workspace),
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        raise


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
            "-m",
            f"v{version}: long-horizon episode {episode} {status}",
        ],
        cwd=str(incumbent_workspace),
        check=True,
        capture_output=True,
        text=True,
    )
    return git_head(incumbent_workspace)
