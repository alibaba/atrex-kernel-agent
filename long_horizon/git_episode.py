from __future__ import annotations

import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from . import main_adapter
from .protocol import atomic_write_json
from .promotion_audit import AUDIT_TRAILER, write_promotion_audit
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
        "agent_problem.json",
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
EPISODE_EVIDENCE_PREFIXES = ("scratch/",)
TIMELINE_PROBE_MARKERS = (
    "cute.experimental.iket.",
    "atrex_timeline.cuh",
    "atrex::timeline::Recorder",
    "ATREX_TIMELINE_ENABLED",
    "ppu_timeline.cuh",
    "ppu_acu_profile::timeline",
    "PPU_TIMELINE_ENABLED",
)


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


def changed_paths(
    workspace: Path, base_commit: str, candidate_commit: str = "HEAD"
) -> list[str]:
    output = git_text(
        workspace,
        "diff",
        "--name-only",
        "--no-renames",
        base_commit,
        candidate_commit,
        "--",
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
            [
                "git",
                "worktree",
                "add",
                "-b",
                self.branch,
                str(self.path),
                self.base_commit,
            ],
            cwd=str(incumbent_workspace),
            check=True,
            capture_output=True,
            text=True,
        )
        CampaignStore.ensure_excluded(self.path)
        self.reset_scratch()

    def reset_scratch(self) -> None:
        """Start with empty temporary storage before the Episode's first Agent launch.

        Do not call this when resuming an already-started Episode. Unlink a
        checkout-provided symlink instead of traversing its target.
        """
        scratch = self.path / "scratch"
        if scratch.is_symlink() or scratch.is_file():
            scratch.unlink()
        elif scratch.exists():
            shutil.rmtree(scratch)
        scratch.mkdir(mode=0o700)

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
            self.path,
            "rev-parse",
            "--verify",
            f"{candidate_commit}^{{commit}}",
            check=False,
        )
        if not resolved:
            return "candidate_commit does not resolve", []
        if resolved != git_head(self.path):
            return "candidate_commit must equal episode HEAD", []
        branch = git_text(
            self.path, "symbolic-ref", "--quiet", "--short", "HEAD", check=False
        )
        if branch != self.branch:
            return "episode worktree left its isolated branch", []
        ancestor = _git(
            self.path,
            "merge-base",
            "--is-ancestor",
            self.base_commit,
            resolved,
            check=False,
        )
        if ancestor.returncode:
            return "candidate_commit is not descended from incumbent", []
        dirty = working_changes(self.path)
        violation = protected_violation(dirty)
        if violation:
            return violation, []
        kernel_matches = _git(
            self.path,
            "diff",
            "--quiet",
            resolved,
            "--",
            "kernel.py",
            check=False,
        )
        if kernel_matches.returncode:
            return "worktree kernel.py must match candidate_commit", []
        paths = changed_paths(self.path, self.base_commit, resolved)
        if not paths:
            return "candidate has no changes relative to incumbent", []
        if paths != ["kernel.py"]:
            return "candidate commit may change only kernel.py", paths
        kernel_text = (self.path / "kernel.py").read_text(encoding="utf-8", errors="replace")
        if any(marker in kernel_text for marker in TIMELINE_PROBE_MARKERS):
            return "candidate kernel.py still contains timeline profiling probes", paths
        return "", paths

    def commit_candidate(self, expected_source: bytes) -> str:
        """Commit the measured source on behalf of the Agent, never other files."""
        kernel = self.path / "kernel.py"
        if kernel.is_symlink() or not kernel.is_file():
            raise ValueError("candidate kernel.py must be a regular file")
        if kernel.read_bytes() != expected_source:
            raise ValueError("kernel.py changed during report validation; submit again")
        if git_text(self.path, "symbolic-ref", "--quiet", "--short", "HEAD") != self.branch:
            raise ValueError("Supervisor candidate branch does not match the Episode")
        if _git(
            self.path, "merge-base", "--is-ancestor", self.base_commit, "HEAD", check=False
        ).returncode:
            raise ValueError("candidate is not descended from the Episode baseline")
        violation = protected_violation(working_changes(self.path))
        if violation:
            raise ValueError(violation)
        paths = git_text(self.path, "diff", "--name-only", self.base_commit, "--").splitlines()
        # A legacy baseline can still track scratch files. Clearing the new
        # workspace must not turn their deletion into a candidate code change.
        paths = [path for path in paths if not path.startswith("scratch/")]
        if "kernel.py" not in paths:
            raise ValueError("candidate has no Kernel change; report pivot or select another")
        if paths != ["kernel.py"]:
            raise ValueError("candidate may change only kernel.py relative to its baseline")
        if any(marker.encode() in expected_source for marker in TIMELINE_PROBE_MARKERS):
            raise ValueError("candidate kernel.py still contains timeline profiling probes")
        # Avoid an unrelated index entry entering the commit. --only also leaves
        # unrelated untracked diagnostics untouched. Replayed reports reuse HEAD.
        staged = git_text(self.path, "diff", "--cached", "--name-only", "--").splitlines()
        if any(path != "kernel.py" for path in staged):
            raise ValueError("Supervisor index contains unexpected staged files")
        if _git(self.path, "show", "HEAD:kernel.py", binary=True).stdout != expected_source:
            result = _git(
                self.path,
                "-c", "core.hooksPath=/dev/null",
                "-c", "commit.gpgSign=false",
                "-c", "user.name=AKA Supervisor",
                "-c", "user.email=supervisor@atrex.local",
                "commit", "--only", "-m", f"Episode {self.episode}: selected Kernel",
                "--", "kernel.py",
                check=False,
            )
            if result.returncode:
                raise ValueError(f"Supervisor could not commit kernel.py: {result.stderr[-1200:]}")
        commit = git_head(self.path)
        if _git(self.path, "show", f"{commit}:kernel.py", binary=True).stdout != expected_source:
            raise ValueError("Kernel changed during commit; restore measured source and resubmit")
        violation, _ = self.validate_candidate(commit)
        if violation:
            raise ValueError(violation)
        return commit

    def archive(self, destination: Path, candidate_commit: str = "HEAD") -> Path:
        destination.mkdir(parents=True, exist_ok=True)
        committed_patch = _git(
            self.path,
            "diff",
            "--binary",
            self.base_commit,
            candidate_commit,
            "--",
            binary=True,
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
            cwd=str(incumbent_workspace),
            check=True,
            capture_output=True,
            text=True,
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
            cwd=str(incumbent_workspace),
            check=True,
            capture_output=True,
            text=True,
        )
        staged = git_text(
            incumbent_workspace, "diff", "--cached", "--name-only"
        ).splitlines()
        violation = protected_violation([path for path in staged if path])
        if violation:
            raise RuntimeError(violation)
        memory_dir = incumbent_workspace / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(memory_dir / f"v{memory_version}.json", memory_record)
        audit_digest = write_promotion_audit(incumbent_workspace, episode, {
            **evidence, "episode": episode, "version": memory_version,
            "base_commit": base_commit, "candidate_commit": candidate_commit,
            "accepted": True,
        })
        subprocess.run(
            [
                "git",
                "add",
                f"memory/v{memory_version}.json",
            ],
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
                f"episode {episode}: promote verified long-horizon candidate\n\n"
                f"{AUDIT_TRAILER}{audit_digest}",
                "--",
                "kernel.py",
                f"memory/v{memory_version}.json",
            ],
            cwd=str(incumbent_workspace),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        subprocess.run(
            ["git", "reset", "--hard", base_commit],
            cwd=str(incumbent_workspace),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        raise
    return git_head(incumbent_workspace)


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
