from __future__ import annotations


import subprocess
import uuid
import shutil
import os
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

# Re-export shared facts/fields for existing callers; policy stays local.
from aka.legacy.task.git import (
    _git, git_text, git_head, working_changes, changed_paths,
)

import aka.light.workflow.context as main_adapter
from aka.light.workflow.protocol import atomic_write_json
from aka.light.workflow.store import CampaignStore
from aka.light.workspace.versions import save_version_record, version_commit_paths


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
EPISODE_EVIDENCE_PREFIXES = ("plans/", "profiles/", ".humanize/")
TIMELINE_PROBE_MARKERS = (
    "cute.experimental.iket.",
    "atrex_timeline.cuh",
    "atrex::timeline::Recorder",
    "ATREX_TIMELINE_ENABLED",
    "ppu_timeline.cuh",
    "ppu_acu_profile::timeline",
    "PPU_TIMELINE_ENABLED",
)


def git_blob(workspace: Path, revision: str, path: str) -> bytes:
    """Read an exact controller-selected committed file."""
    return _git(workspace, "show", f"{revision}:{path}", binary=True).stdout


def warn_kernel_mismatch(workspace: Path, sealed_source: bytes) -> bool:
    """Diagnose a mutable copy without changing the sealed acceptance subject."""
    from aka.light.workspace.files import read_input

    try:
        from aka.light.workspace.sources import snapshot as source_snapshot
        mismatch = source_snapshot(workspace) != sealed_source
    except (OSError, ValueError):
        mismatch = True
    if mismatch:
        logging.getLogger(__name__).warning(
            "Candidate Kernel integrity warning: %s/kernel.py is unreadable or differs from the sealed commit; "
            "acceptance uses the sealed commit", workspace,
        )
    return mismatch


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

    def validate_candidate(self, candidate_commit: str, *, require_worktree_match: bool = True) -> tuple[str, list[str]]:
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
        from aka.light.workspace.sources import from_git, unpack, snapshot as source_snapshot, allowed_changes
        policy, _ = unpack(from_git(self.path, self.base_commit))
        if policy is not None:
            try:
                source = from_git(self.path, resolved)
                if unpack(source)[0] != policy:
                    return "candidate changed its source contract", []
                paths = changed_paths(self.path, self.base_commit, resolved)
                if not allowed_changes(paths, policy):
                    return "candidate changed a fixed or out-of-scope file", paths
                if require_worktree_match and source_snapshot(self.path, policy=policy) != source:
                    return "worktree source tree must match the sealed candidate", paths
                if any(marker.encode() in source for marker in TIMELINE_PROBE_MARKERS):
                    return "candidate contains timeline profiling probes", paths
                return "", paths
            except (OSError, ValueError, RuntimeError) as error:
                return str(error), []
        dirty = working_changes(self.path)
        violation = protected_violation(dirty)
        if violation:
            return violation, []
        if require_worktree_match and _git(
            self.path, "diff", "--quiet", resolved, "--", "kernel.py", check=False,
        ).returncode:
            return "worktree kernel.py must match candidate_commit", []
        paths = changed_paths(self.path, self.base_commit, resolved)
        if not paths:
            return "candidate has no changes relative to incumbent", []
        if paths != ["kernel.py"]:
            return "candidate commit may change only kernel.py", paths
        kernel_text = git_blob(self.path, resolved, "kernel.py").decode("utf-8", errors="replace")
        if any(marker in kernel_text for marker in TIMELINE_PROBE_MARKERS):
            return "candidate kernel.py still contains timeline profiling probes", paths
        return "", paths

    def restore_sealed_candidate(self, candidate_commit: str, source: bytes) -> tuple[str, list[str]]:
        """Refresh the controller copy only from a validated private report/commit."""
        violation, paths = self.validate_candidate(candidate_commit, require_worktree_match=False)
        if violation:
            return violation, paths
        from aka.light.workspace.sources import from_git, install
        if from_git(self.path, candidate_commit) != source:
            return "private report source does not match candidate_commit", paths
        if warn_kernel_mismatch(self.path, source):
            from aka.light.workspace.files import publish

            install(self.path, source)
        return "", paths

    def reset_scratch(self) -> None:
        from aka.light.workspace.auxiliary import reset_episode_scratch

        reset_episode_scratch(self.path)

    def commit_candidate(self, expected_source: bytes) -> str:
        """Seal exactly the measured bytes, using a private index and a HEAD CAS.

        Never ask Git to reread an Agent-writable candidate or run commit hooks.
        The real Episode worktree is not mounted in the Agent namespace.
        """
        from aka.light.workspace.files import publish
        from aka.light.workspace.sources import unpack, commit_tree_candidate
        if unpack(expected_source)[0] is not None:
            return commit_tree_candidate(self, expected_source)

        head = git_head(self.path)
        if git_text(self.path, "symbolic-ref", "--quiet", "--short", "HEAD") != self.branch:
            raise RuntimeError("Supervisor candidate branch does not match the Episode")
        if _git(self.path, "merge-base", "--is-ancestor", self.base_commit, head, check=False).returncode:
            raise RuntimeError("candidate is not descended from the Episode baseline")
        if any(marker.encode() in expected_source for marker in TIMELINE_PROBE_MARKERS):
            raise ValueError("candidate kernel.py still contains timeline profiling probes")
        if _git(self.path, "show", f"{self.base_commit}:kernel.py", binary=True).stdout == expected_source:
            raise ValueError("candidate has no Kernel change; report pivot or select another")
        violation = protected_violation(working_changes(self.path))
        if violation:
            raise RuntimeError(violation)
        if any(path != "kernel.py" for path in changed_paths(self.path, self.base_commit, head)):
            raise RuntimeError("candidate may change only kernel.py relative to its baseline")
        if any(path != "kernel.py" for path in git_text(self.path, "diff", "--cached", "--name-only").splitlines()):
            raise RuntimeError("Supervisor index contains unexpected staged files")
        if _git(self.path, "show", f"{head}:kernel.py", binary=True).stdout != expected_source:
            blob = subprocess.run(
                ["git", "hash-object", "-w", "--stdin"], cwd=self.path,
                input=expected_source, capture_output=True, check=True,
            ).stdout.decode().strip()
            with tempfile.TemporaryDirectory(prefix="aka-candidate-index-") as temporary:
                environment = os.environ | {
                    "GIT_INDEX_FILE": str(Path(temporary) / "index"),
                    "GIT_AUTHOR_NAME": "AKA Supervisor", "GIT_COMMITTER_NAME": "AKA Supervisor",
                    "GIT_AUTHOR_EMAIL": "supervisor@atrex.local", "GIT_COMMITTER_EMAIL": "supervisor@atrex.local",
                }

                def run(*args: str) -> str:
                    return subprocess.run(["git", *args], cwd=self.path, env=environment,
                                          capture_output=True, text=True, check=True).stdout.strip()

                run("read-tree", head)
                run("update-index", "--add", "--cacheinfo", f"100644,{blob},kernel.py")
                tree = run("write-tree")
                commit = run("-c", "commit.gpgSign=false", "commit-tree", tree,
                             "-p", head, "-m", f"Episode {self.episode}: selected Kernel")
                run("update-ref", "HEAD", commit, head)
                head = commit
        publish(self.path, "kernel.py", expected_source)
        _git(self.path, "read-tree", head)
        violation, _ = self.validate_candidate(head)
        if violation:
            raise RuntimeError(violation)
        return head

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
    generate_memory: bool = False,
) -> str:
    if git_head(incumbent_workspace) != base_commit:
        raise RuntimeError("incumbent advanced during episode; refusing promotion")
    from aka.light.acceptance.promotion_audit import AUDIT_TRAILER, write_promotion_audit

    audit_digest = write_promotion_audit(incumbent_workspace, episode, evidence)
    try:
        subprocess.run(
            ["git", "merge", "--squash", "--no-commit", candidate_commit],
            cwd=str(incumbent_workspace),
            check=True,
            capture_output=True,
            text=True,
        )
        staged = git_text(
            incumbent_workspace, "diff", "--cached", "--name-only", "--no-renames", "-z"
        ).split("\0")
        staged = [name for name in staged if name]
        from aka.light.workspace.sources import from_git, unpack, allowed_changes
        source_policy, _ = unpack(from_git(incumbent_workspace, base_commit))
        if (not allowed_changes(staged, source_policy) if source_policy else staged != ["kernel.py"]):
            raise RuntimeError("Promotion may change only the sealed candidate source")
        violation = protected_violation([path for path in staged if path])
        if violation:
            raise RuntimeError(violation)
        save_version_record(
            incumbent_workspace, memory_version, memory_record, generate_memory=generate_memory
        )
        paths = version_commit_paths(incumbent_workspace, memory_version, generate_memory=generate_memory)
        subprocess.run(
            [
                "git",
                "add",
                *paths,
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
                f"episode {episode}: promote verified long-horizon candidate\n\n{AUDIT_TRAILER}{audit_digest}",
                "--",
                *staged,
                *paths,
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
    # A read failure after commit must not reset a successful promotion.
    return git_head(incumbent_workspace)


def record_episode_outcome(
    incumbent_workspace: Path,
    *,
    base_commit: str,
    version: int,
    episode: int,
    status: str,
    memory_record: dict[str, Any],
    generate_memory: bool = False,
) -> str:
    """Advance main-compatible version history without changing the incumbent kernel."""
    if git_head(incumbent_workspace) != base_commit:
        raise RuntimeError("incumbent advanced during episode; refusing outcome record")
    save_version_record(
        incumbent_workspace, version, memory_record, generate_memory=generate_memory
    )
    paths = version_commit_paths(incumbent_workspace, version, generate_memory=generate_memory)
    subprocess.run(
        ["git", "add", *paths],
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
            *paths,
        ],
        cwd=str(incumbent_workspace),
        check=True,
        capture_output=True,
        text=True,
    )
    return git_head(incumbent_workspace)
