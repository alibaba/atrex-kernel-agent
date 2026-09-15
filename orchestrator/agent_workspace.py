"""Persistent, Git-free Agent files, separate from the Supervisor's worktree.

Gateway requests read this directory directly. Publication copies only candidate
files and scratch diagnostics back; it never imports Agent-authored Git/control state.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from pathlib import Path

from .durable_state import durable_write_json, fsync_directory

CANDIDATE_FILES = ("kernel.py", "solution.json")
PUBLIC_INPUT_FILES = (
    "README.md", "CLAUDE.md", "agent_problem.json", "reference.py", "input.py",
    "definition.json", "workload.jsonl", "shapes.json", "roofline.json",
    "metadata.json", "valid.py",
)
WORKSPACE_ROLE_ENV = "ATREX_AGENT_WORKSPACE_ROLE"
# Selected by the Supervisor, not by files supplied by a coding Agent.
WORKSPACE_LAYOUTS = {
    "optimizer": (PUBLIC_INPUT_FILES, (), CANDIDATE_FILES),
    "production-review": (("review_request.json",), ("candidate",), ("dependency_review.json",)),
    "problem-generation": (("reference.py", "input.py", "shapes.json", "metadata.json"), (), ("agent_problem.json",)),
    "baseline-exit-review": (("crash_record.json",), ("candidate",), ("resume.json",)),
    "baseline-correctness-review": ((), ("context",), ("correctness_review.md",)),
}


def _open_regular(path: Path):
    # Anchor every path component, not just the final file: an Agent may rename
    # scratch directories while the Supervisor is archiving them.
    directory = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    finally:
        os.close(directory)
    stream = os.fdopen(fd, "rb")
    try:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError(f"Agent output must be a regular file: {path.name}")
    except BaseException:
        stream.close()
        raise
    return stream


def _regular_bytes(path: Path, *, limit: int = -1) -> bytes:
    with _open_regular(path) as stream:
        return stream.read(limit)


def _digest(path: Path) -> str | None:
    try:
        return hashlib.sha256(_regular_bytes(path)).hexdigest()
    except FileNotFoundError:
        return None


def _copy_file(source: Path, destination: Path) -> None:
    data = _regular_bytes(source)
    if destination.is_symlink():
        raise ValueError(f"publication destination cannot be a symlink: {destination.name}")
    parent = destination.parent
    while not parent.exists():
        parent = parent.parent
    if parent.is_symlink():
        raise ValueError("publication directory cannot be a symlink")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".copy-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            os.fchmod(stream.fileno(), source.stat().st_mode & 0o777)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        fsync_directory(destination.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _scratch_files(root: Path) -> list[Path]:
    if root.is_symlink():
        raise ValueError("scratch cannot be a symlink")
    files: list[Path] = []
    if not root.exists():
        return files
    for parent, directories, names in os.walk(root, followlinks=False):
        for name in directories:
            if (Path(parent) / name).is_symlink():
                raise ValueError("scratch directories cannot be symlinks")
        for name in names:
            path = Path(parent) / name
            if path.is_symlink() or not path.is_file():
                raise ValueError("scratch outputs must be regular files")
            files.append(path.relative_to(root))
    return files


class AgentWorkspace:
    def __init__(self, worktree: Path, scope: Path, *, role: str = "optimizer") -> None:
        if role not in WORKSPACE_LAYOUTS:
            raise ValueError(f"Unknown Agent workspace role: {role}")
        self.role = role
        self.input_files, self.input_trees, self.output_files = WORKSPACE_LAYOUTS[role]
        self.worktree = worktree.resolve()
        scope = scope.resolve()
        self.root = scope / "agent-workspace"
        self.state_path = scope / "agent-workspace-state.json"

    @property
    def read_only_paths(self) -> tuple[str, ...]:
        return (*self.input_files, *self.input_trees)

    def _state(self) -> dict:
        if not self.state_path.exists():
            return {"candidate": {}}
        return json.loads(self.state_path.read_text())

    def prepare(self) -> Path:
        if self.root.is_symlink():
            raise ValueError("Agent workspace cannot be a symlink")
        fresh = not self.root.exists()
        self.root.mkdir(parents=True, mode=0o700, exist_ok=True)
        state = self._state()
        if not fresh and state.get("role", "optimizer") != self.role:
            raise ValueError("Cannot change the role of an existing Agent workspace")
        state["role"] = self.role
        for name in self.output_files:
            source, target = self.worktree / name, self.root / name
            digest = _digest(source)
            # Preserve edits after an interrupted process. A subsequent trusted
            # reset/conversion of the worktree takes precedence over that draft.
            if fresh or name not in state["candidate"] or state["candidate"][name] != digest:
                if digest is not None:
                    _copy_file(source, target)
                else:
                    target.unlink(missing_ok=True)
            state["candidate"][name] = digest
        for name in self.input_files:
            source, target = self.worktree / name, self.root / name
            if source.is_file():
                _copy_file(source, target)
            else:
                target.unlink(missing_ok=True)
        for name in self.input_trees:
            source, target = self.worktree / name, self.root / name
            if target.is_symlink():
                raise ValueError("Agent input directory cannot be a symlink")
            target.mkdir(exist_ok=True)
            for relative in _scratch_files(source):
                destination = target / relative
                for parent in destination.parents:
                    if parent == target:
                        break
                    if parent.is_symlink():
                        raise ValueError("Agent input directories cannot be symlinks")
                _copy_file(source / relative, destination)
        memory = self.root / "memory"
        if memory.is_symlink():
            raise ValueError("Agent memory cannot be a symlink")
        memory.mkdir(exist_ok=True)
        for source in (self.worktree / "memory").glob("v*.json") if self.role == "optimizer" else ():
            if source.stem[1:].isascii() and source.stem[1:].isdigit():
                _copy_file(source, memory / source.name)
        scratch = self.root / "scratch"
        if scratch.is_symlink():
            raise ValueError("Agent scratch cannot be a symlink")
        scratch.mkdir(exist_ok=True)
        if fresh:
            for relative in _scratch_files(self.worktree / "scratch"):
                _copy_file(self.worktree / "scratch" / relative, scratch / relative)
        durable_write_json(self.state_path, state)
        return self.root

    def publish(self) -> None:
        """Publish candidate files/diagnostics without accepting private metadata."""
        state = self._state()
        scratch_files = _scratch_files(self.root / "scratch")
        for name in self.output_files:
            source, target = self.root / name, self.worktree / name
            digest = _digest(source)
            if digest != _digest(target):
                if digest is None:
                    target.unlink(missing_ok=True)
                else:
                    _copy_file(source, target)
            state["candidate"][name] = digest
        for relative in scratch_files:
            source = self.root / "scratch" / relative
            target = self.worktree / "scratch" / relative
            # Validate all existing destination directories, not just the leaf.
            for parent in (target, *target.parents):
                if parent == self.worktree:
                    break
                if parent.is_symlink():
                    raise ValueError("scratch publication cannot traverse symlinks")
            if _digest(source) != _digest(target):
                _copy_file(source, target)
        # Keep previous diagnostics on the Supervisor side for audit; the live
        # Agent's scratch remains the source for requests, including deletions.
        durable_write_json(self.state_path, state)


def project_reference_tree(source: Path, destination: Path) -> Path:
    """Optional read-only references omit Git metadata rather than masking it."""
    if not destination.exists():
        shutil.copytree(source, destination, symlinks=True, ignore=shutil.ignore_patterns(
            ".git", "__pycache__", ".venv", "node_modules",
        ))
    return destination
