"""Persistent Git-free Episode drafts; the controller retains its real worktree."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .agent_home import open_private_directory
from .durable_state import durable_write_json
from .session_tail import read_regular_bytes

EPISODE_WORKSPACE_ENV = "ATREX_EPISODE_WORKSPACE"
PUBLIC_FILES = (
    "README.md", "CLAUDE.md", "agent_problem.json", "reference.py", "input.py",
    "definition.json", "workload.jsonl", "solution.json", "shapes.json", "metadata.json", "roofline.json",
    "valid.py", "test_kernel.py", "profile_driver.py", "config.json", ".orchestrator_mode.json",
)
DIAGNOSTIC_TREES = ("scratch", "plans", "profiles", ".humanize")
MAX_FILE = 16 * 1024 * 1024
MAX_TOTAL = 64 * 1024 * 1024


def _read(path: Path) -> bytes:
    value = read_regular_bytes(path, limit=MAX_FILE + 1)
    if len(value) > MAX_FILE:
        raise ValueError(f"Episode file exceeds 16 MiB: {path.name}")
    return value


def _tree(root: Path):
    """Never follow mutable symlinks; enforce aggregate limits before publication."""
    count = total = 0
    if root.is_symlink():
        raise ValueError("Episode diagnostics cannot be symlinks")
    for parent, dirs, names in os.walk(root, followlinks=False):
        for name in dirs:
            if (Path(parent) / name).is_symlink():
                raise ValueError("Episode diagnostic directories cannot be symlinks")
        for name in names:
            path = Path(parent) / name
            data = _read(path)
            count += 1
            total += len(data)
            if count > 4096 or total > MAX_TOTAL:
                raise ValueError("Episode diagnostics exceed 4096 files / 64 MiB")
            yield path.relative_to(root).as_posix(), data


class EpisodeWorkspace:
    def __init__(self, worktree: Path, directory: Path, *, is_ppu: bool = False):
        self.worktree = worktree
        self.root = directory / "workspace"
        self.state_path = directory / "state.json"
        self.is_ppu = is_ppu

    def prepare(self) -> Path:
        from supervisor.workspace import publish
        from .workspace_runtime import link_runtime

        fresh = not self.state_path.exists()
        state = {} if fresh else json.loads(_read(self.state_path))
        with open_private_directory(self.root):
            pass
        source = _read(self.worktree / "kernel.py")
        digest = hashlib.sha256(source).hexdigest()
        # Preserve a crashed invocation's draft unless the controller changed
        # its own source since the last publication (e.g. conversion/reset).
        if fresh or digest != state.get("kernel_digest"):
            publish(self.root, "kernel.py", source)
        for name in PUBLIC_FILES:
            try:
                content = _read(self.worktree / name)
            except FileNotFoundError:
                continue
            publish(self.root, name, content)
        with open_private_directory(self.root / "memory"):
            pass
        for path in (self.worktree / "memory").glob("v*.json"):
            if path.stem[1:].isascii() and path.stem[1:].isdigit():
                publish(self.root, f"memory/{path.name}", _read(path))
        for name in (*DIAGNOSTIC_TREES, ".atrex_long_horizon"):
            with open_private_directory(self.root / name):
                pass
            if fresh and name in DIAGNOSTIC_TREES:
                for relative, content in _tree(self.worktree / name):
                    publish(self.root, f"{name}/{relative}", content)
        # These are controller-created asset links, never copied from Agent files.
        # Do not install the evaluator or copy .git / control checkpoints.
        if fresh:
            link_runtime(self.root, is_ppu=self.is_ppu)
        durable_write_json(self.state_path, {"kernel_digest": digest})
        return self.root

    def publish(self) -> None:
        from supervisor.workspace import publish

        content = _read(self.root / "kernel.py")
        publish(self.worktree, "kernel.py", content)
        for name in DIAGNOSTIC_TREES:
            for relative, data in _tree(self.root / name):
                publish(self.worktree, f"{name}/{relative}", data)
        # Phase markers remain part of the old workflow, not a handoff authority.
        try:
            telemetry = _read(self.root / ".atrex_long_horizon/telemetry.jsonl")
        except FileNotFoundError:
            pass
        else:
            publish(self.worktree, ".atrex_long_horizon/telemetry.jsonl", telemetry)
        durable_write_json(self.state_path, {"kernel_digest": hashlib.sha256(content).hexdigest()})
