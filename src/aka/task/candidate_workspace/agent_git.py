"""Controlled local Git for one Agent trajectory.

The repository is Agent-owned engineering history and its full clean HEAD commit is
the Agent-visible Kernel identity. Runtime may initialize V0 and import peer refs,
but it never mirrors, promotes, rolls back, or rewrites Agent work.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any

IGNORED = (
    "scratch/", "tools/", "skills/", "input/", "eval/", "profiles/",
    "plans/", "directions/", "experiments/", "records/", "research-library/", "reference/",
    "reference-projects/", "references/", "gpu-wiki/", "atrex-bench/",
    ".atrex_*/", ".orchestrator_mode.json", ".orchestrator_state.json",
    ".claude/", ".qoder/", ".agents/",
    "README.md", "CLAUDE.md", "WORKSPACE.md", "PROGRESS.md", "frontier.json", "agent_problem.json",
    "reference.py", "input.py", "shapes.json", "metadata.json", "roofline.json", "valid.py",
    "definition.json", "workload.jsonl", "solution.json", "config.json", "test_kernel.py",
    "profile_driver.py", "*.jsonl",
)

TRACKED_GITIGNORE = (
    "# Agent Git tracks only the Kernel source tree.",
    "/*",
    "!/.gitignore",
    "!/kernel.py",
    "!/impl/",
    "!/impl/**",
)


def _gitignore_for_candidate_paths(paths: list[str]) -> str:
    """Ignore the workspace except the exact selected candidate source paths."""
    normalized = sorted({path.strip("/") for path in paths if path.strip("/")})
    if all(path == "kernel.py" or path.startswith("impl/") for path in normalized):
        return "\n".join(TRACKED_GITIGNORE) + "\n"

    tree: dict[str, dict] = {}
    for path in normalized:
        node = tree
        for part in Path(path).parts:
            node = node.setdefault(part, {})

    lines = [TRACKED_GITIGNORE[0], "/*", "!/.gitignore"]

    def emit(node: dict[str, dict], parent: tuple[str, ...] = ()) -> None:
        for name, children in sorted(node.items()):
            parts = (*parent, name)
            absolute = "/" + "/".join(parts)
            if children:
                lines.extend((f"!{absolute}/", f"{absolute}/*"))
                emit(children, parts)
            else:
                lines.append(f"!{absolute}")

    emit(tree)
    return "\n".join(lines) + "\n"


def gitignore_contents(candidate_paths: list[str] | None = None) -> str:
    """Return the tracked policy; source-tree workspaces use the fixed contract."""
    return _gitignore_for_candidate_paths(candidate_paths or ["kernel.py"])


def initialize_gitignore(workspace: Path) -> None:
    """Install the tracked ignore policy before the initial Kernel commit."""
    try:
        from aka.task.source.formats import commit_paths

        candidate_paths = commit_paths(workspace, normalize=False)
    except (OSError, UnicodeError, ValueError):
        candidate_paths = ["kernel.py"]
    (workspace / ".gitignore").write_text(
        gitignore_contents(candidate_paths), encoding="utf-8"
    )


def kernel_git_paths(workspace: Path) -> list[str]:
    """Return the only paths Runtime may put in the initial Agent commit."""
    paths = [".gitignore", "kernel.py"]
    implementation = workspace / "impl"
    if implementation.is_dir() and any(
        path.is_file() for path in implementation.rglob("*")
    ):
        paths.append("impl")
    return paths

_RESUME_REMOTE_PREFIX = "refs/remotes/atrex-resume/heads/"
_RESUME_BRANCH_PREFIX = "refs/atrex/resume-branch/"
GIT_COMMIT_ID = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")


class DirtyKernelWorktree(ValueError):
    """The Agent repository has no exact committed identity for its current Kernel."""

    def __init__(self, paths: list[str], message: str = "Kernel worktree is not clean"):
        super().__init__(message)
        self.paths = tuple(paths)


def _run(
    workspace: Path, *args: str, check: bool = True, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment.update(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_TERMINAL_PROMPT="0",
        GIT_ASKPASS="/usr/bin/false",
    )
    environment.update(extra_env or {})
    return subprocess.run(
        ["git", *args], cwd=workspace, env=environment,
        text=True, capture_output=True, check=check,
    )


class AgentGit:
    def __init__(self, workspace: Path, bundle: Path | None = None):
        self.workspace = workspace
        self.bundle = bundle
        self.checkout_state = (
            bundle.with_name(bundle.name + ".checkout.json")
            if bundle is not None
            else None
        )

    def _configure(self) -> None:
        settings = {
            "user.name": "Atrex Agent",
            "user.email": "agent@localhost",
            "core.hooksPath": ".git/disabled-hooks",
            "credential.helper": "",
            "commit.gpgSign": "false",
            "tag.gpgSign": "false",
            "protocol.file.allow": "never",
            "fetch.recurseSubmodules": "false",
            "submodule.recurse": "false",
            "advice.detachedHead": "false",
        }
        for key, value in settings.items():
            _run(self.workspace, "config", "--local", key, value)
        hooks = self.workspace / ".git" / "disabled-hooks"
        hooks.mkdir(parents=True, exist_ok=True)

    def _ensure_private_excludes(self) -> None:
        """Upgrade Runtime exclusions without dirtying the tracked repository."""
        from aka.task.source.formats import read

        result = _run(
            self.workspace,
            "rev-parse",
            "--git-path",
            "info/exclude",
        )
        path = Path(result.stdout.strip())
        if not path.is_absolute():
            path = self.workspace / path
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        try:
            candidate_paths = sorted(read(self.workspace).files)
        except (OSError, ValueError):
            candidate_paths = []

        def ignored(candidate: str) -> bool:
            for pattern in IGNORED:
                normalized = pattern.lstrip("/")
                if normalized.endswith("/"):
                    root = normalized.rstrip("/")
                    if candidate == root or candidate.startswith(root + "/"):
                        return True
                elif fnmatch.fnmatchcase(candidate, normalized):
                    return True
            return False

        # A layout may promote evaluator metadata (for example solution.json)
        # into the exact Kernel identity.  Keep the broad Runtime exclusions,
        # then explicitly re-include every candidate-owned path they match.
        reincluded = [
            "!/" + candidate.replace(" ", "\\ ")
            for candidate in sorted(candidate_paths)
            if ignored(candidate)
        ]
        marker = "# Atrex Runtime private workspace exclusions"
        prefix = existing.split(marker, 1)[0].rstrip()
        desired = (
            (prefix + "\n\n" if prefix else "")
            + marker
            + "\n"
            + "\n".join((*IGNORED, *reincluded))
            + "\n"
        )
        if existing == desired:
            return
        path.write_text(desired, encoding="utf-8")

    def _restore_branches(self) -> None:
        """Restore every Agent branch and the previously checked-out branch."""
        if self.checkout_state is None:
            return
        rows = _run(
            self.workspace,
            "for-each-ref",
            "--format=%(refname) %(objectname)",
            _RESUME_REMOTE_PREFIX,
        ).stdout.splitlines()
        branches: list[str] = []
        for row in rows:
            remote_ref, object_name = row.split(" ", 1)
            branch = remote_ref.removeprefix(_RESUME_REMOTE_PREFIX)
            if not branch:
                continue
            local_ref = "refs/heads/" + branch
            _run(self.workspace, "check-ref-format", local_ref)
            _run(self.workspace, "update-ref", local_ref, object_name)
            branches.append(branch)

        active = ""
        try:
            checkout = json.loads(self.checkout_state.read_text(encoding="utf-8"))
            candidate = checkout.get("branch") if isinstance(checkout, dict) else None
            if isinstance(candidate, str):
                active = candidate
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
        # Read the retired in-Git marker only for migration. New persistence
        # never writes controller state into the Agent repository.
        markers = _run(
            self.workspace,
            "for-each-ref",
            "--format=%(refname)",
            _RESUME_BRANCH_PREFIX,
        ).stdout.splitlines()
        if not active and len(markers) == 1:
            active = markers[0].removeprefix(_RESUME_BRANCH_PREFIX)
        if active not in branches:
            active = "work" if "work" in branches else (sorted(branches)[0] if branches else "")
        if active:
            _run(self.workspace, "checkout", "-f", active)

    def _fetch_bundle(self) -> None:
        """Import recovery and peer refs without replacing live local branches."""
        if self.bundle is None or not self.bundle.is_file():
            return
        _run(
            self.workspace,
            "-c",
            "protocol.file.allow=always",
            "fetch",
            str(self.bundle),
            "+refs/heads/*:refs/remotes/atrex-resume/heads/*",
            "+refs/tags/*:refs/tags/*",
            "+refs/atrex/*:refs/atrex/*",
            "+refs/remotes/atrex/*:refs/remotes/atrex/*",
            "+refs/notes/*:refs/notes/*",
        )

    def _record_active_branch(self) -> None:
        """Store checkout recovery outside Agent Git."""
        if self.checkout_state is None:
            return
        branch = _run(
            self.workspace, "symbolic-ref", "--quiet", "--short", "HEAD", check=False
        ).stdout.strip()
        self.checkout_state.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.checkout_state.with_name(
            self.checkout_state.name + "." + uuid.uuid4().hex + ".tmp"
        )
        temporary.write_text(
            json.dumps({"branch": branch}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.checkout_state)

    def prepare(self) -> None:
        initialized = (self.workspace / ".git").is_dir()
        if not initialized:
            _run(self.workspace, "init", "-b", "work")
            self._configure()
            initialize_gitignore(self.workspace)
            self._fetch_bundle()
            if self.bundle is not None and self.bundle.is_file():
                self._restore_branches()
        else:
            self._configure()
            # Epoch sharing updates the durable bundle between Sessions. A
            # stable repository must import those refs in place instead of
            # waiting for a future workspace reconstruction.
            self._fetch_bundle()
        # The tracked .gitignore is created before the initial Kernel commit.
        # Runtime upgrades belong in Git's private exclude file so resuming an
        # older trajectory never mutates Agent-visible source history.
        self._ensure_private_excludes()
        # Bootstrap is the only Runtime-authored Agent commit. Once a repository
        # has a HEAD, the Agent exclusively owns its branches and commits.
        if self.head() is None:
            self.checkpoint("Runtime: initialize trajectory baseline")

    def install_runtime_ref(
        self,
        ref: str,
        source: bytes,
        *,
        message: str,
        commit_id: str,
    ) -> str:
        """Expose an existing Agent/peer commit without changing its identity."""

        del message  # Retained in the call contract as an operator-readable label.

        if not ref.startswith("refs/") or ".." in ref or any(char.isspace() for char in ref):
            raise ValueError("Runtime Git ref is invalid")
        if GIT_COMMIT_ID.fullmatch(commit_id) is None:
            raise ValueError("Runtime Kernel identity is not a Git commit ID")
        present = _run(
            self.workspace, "cat-file", "-e", f"{commit_id}^{{commit}}", check=False
        )
        if present.returncode != 0:
            raise ValueError(
                "Runtime can expose only an existing local or imported peer Agent commit"
            )
        from aka.task.source.formats import from_git

        if from_git(self.workspace, commit_id) != source:
            raise ValueError("Runtime Kernel commit does not match its saved source")
        _run(self.workspace, "update-ref", ref, commit_id)
        return commit_id

    def commit_reachable(self, commit_id: str) -> bool:
        """Return whether a complete commit is reachable from Agent-visible Git state."""
        if GIT_COMMIT_ID.fullmatch(commit_id) is None:
            return False
        ancestor = _run(
            self.workspace,
            "merge-base",
            "--is-ancestor",
            commit_id,
            "HEAD",
            check=False,
        )
        if ancestor.returncode == 0:
            return True
        refs = _run(
            self.workspace,
            "for-each-ref",
            "--format=%(refname)",
            "--contains",
            commit_id,
            "refs/heads",
            "refs/tags",
            "refs/atrex",
            "refs/remotes/atrex",
            check=False,
        )
        return refs.returncode == 0 and bool(refs.stdout.strip())

    def find_kernel_commit(self, source: bytes) -> str | None:
        """Find an existing Agent commit for exact Kernel bytes without changing Git.

        Episode boundaries must not manufacture a second commit for an already
        promoted Kernel. Prefer the current HEAD, then search Agent-visible
        history and imported shared refs. Agent Git is an unrestricted engineering
        history: intermediate commits may intentionally omit or break the Kernel
        source tree. Such revisions are not discovery failures; they simply cannot
        identify the requested incumbent and must be skipped. Explicit Runtime
        requests and terminal acceptance retain their stricter source validation.
        """
        from aka.task.source.formats import from_git

        head = self.head()
        candidates: list[str] = [head] if head else []
        history = _run(
            self.workspace,
            "rev-list",
            "--date-order",
            "--all",
            check=False,
        )
        if history.returncode == 0:
            candidates.extend(history.stdout.splitlines())
        seen: set[str] = set()
        for commit_id in candidates:
            if commit_id in seen or GIT_COMMIT_ID.fullmatch(commit_id) is None:
                continue
            seen.add(commit_id)
            try:
                if from_git(self.workspace, commit_id) == source:
                    return commit_id
            except (
                OSError,
                UnicodeError,
                ValueError,
                RuntimeError,
                subprocess.SubprocessError,
            ):
                continue
        return None

    def head(self) -> str | None:
        result = _run(self.workspace, "rev-parse", "HEAD", check=False)
        value = result.stdout.strip()
        return value if result.returncode == 0 and value else None

    def kernel_commit(self, expected_source: bytes | None = None) -> str:
        """Return HEAD only when it exactly identifies the visible Kernel source.

        Runtime-managed/public files are ignored by the repository.  Any
        remaining status entry is therefore an Agent-owned change that must be
        committed before a Kernel-bound request can acquire a stable identity.
        """
        status = _run(
            self.workspace,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            check=False,
        )
        if status.returncode != 0:
            raise DirtyKernelWorktree([], status.stderr.strip() or "Git status failed")
        changes = [line[3:] if len(line) > 3 else line for line in status.stdout.splitlines()]
        if changes:
            raise DirtyKernelWorktree(changes)
        head = self.head()
        if head is None or GIT_COMMIT_ID.fullmatch(head) is None:
            raise DirtyKernelWorktree([], "Kernel worktree has no valid HEAD commit")
        from aka.task.source.formats import from_git, snapshot

        try:
            committed = from_git(self.workspace, head)
            visible = snapshot(self.workspace)
        except (OSError, UnicodeError, ValueError) as error:
            raise DirtyKernelWorktree(
                [], "HEAD or the visible worktree is not a complete Kernel source tree"
            ) from error
        if committed != visible:
            raise DirtyKernelWorktree(
                [], "HEAD does not contain the exact visible Kernel source tree"
            )
        if expected_source is not None and committed != expected_source:
            raise DirtyKernelWorktree(
                [], "HEAD does not contain the exact Kernel source submitted for evaluation"
            )
        return head

    def checkpoint(self, message: str) -> str | None:
        from aka.task.source.formats import commit_paths

        paths = commit_paths(self.workspace, normalize=self.head() is not None)
        paths.append(".gitignore")
        _run(self.workspace, "add", "--", *paths)
        staged = _run(self.workspace, "diff", "--cached", "--quiet", check=False)
        if staged.returncode == 1 or self.head() is None:
            _run(self.workspace, "commit", "--no-gpg-sign", "-m", message)
        elif staged.returncode != 0:
            raise RuntimeError(f"Agent Git could not inspect staged source: {staged.stderr.strip()}")
        return self.head()

    def capture(self) -> dict[str, Any]:
        from aka.task.source.formats import commit_paths

        head = self.head()
        paths = commit_paths(self.workspace, normalize=False)
        status = _run(
            self.workspace,
            "status", "--porcelain=v1", "--untracked-files=no", "--",
            *paths,
            check=False,
        )
        clean = status.returncode == 0 and not status.stdout.strip()
        tree = None
        if clean and head:
            result = _run(self.workspace, "rev-parse", f"{head}^{{tree}}", check=False)
            if result.returncode == 0:
                tree = result.stdout.strip() or None
        return {"head": head, "clean": clean, "tree": tree}

    def persist(self) -> str | None:
        # Persist refs only. Uncommitted work remains in the stable workspace,
        # but the Supervisor never manufactures Agent history on Session exit.
        head = self.head()
        if head is None:
            return None
        if self.bundle is None:
            return head
        self._record_active_branch()
        self.bundle.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.bundle.with_suffix(".tmp")
        temporary.unlink(missing_ok=True)
        result = _run(
            self.workspace, "bundle", "create", str(temporary), "--all", check=False
        )
        if result.returncode != 0:
            raise RuntimeError(f"Agent Git bundle failed: {result.stderr.strip()}")
        os.replace(temporary, self.bundle)
        return head

    def audit(self) -> str:
        return json.dumps(self.capture(), sort_keys=True)
