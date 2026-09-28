"""Persistent Git-free Episode drafts; the controller retains its real worktree."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

from .agent_home import open_private_directory
from .durable_state import durable_write_json
from .session_tail import read_regular_bytes

EPISODE_WORKSPACE_ENV = "ATREX_EPISODE_WORKSPACE"
PUBLIC_FILES = (
    "README.md", "CLAUDE.md", "agent_problem.json", "reference.py", "input.py",
    "definition.json", "workload.jsonl", "solution.json", "shapes.json", "metadata.json", "roofline.json",
    "valid.py",
)
DIAGNOSTIC_TREES = ("scratch",)
MAX_FILE = 16 * 1024 * 1024
MAX_TOTAL = 64 * 1024 * 1024


def _read(path: Path) -> bytes:
    value = read_regular_bytes(path, limit=MAX_FILE + 1)
    if len(value) > MAX_FILE:
        raise ValueError(f"Episode file exceeds 16 MiB: {path.name}")
    return value


def bounded_tree_entries(root: Path):
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
            if name == "CLAUDE.md":
                # Instructions follow the running controller, not an old V0's
                # retired Setup/Fast workflow. Do not rewrite the committed V0.
                content = _read(Path(__file__).resolve().parents[1] / "reference/CLAUDE.md")
                publish(self.root, name, content)
                continue
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
        for name in DIAGNOSTIC_TREES:
            with open_private_directory(self.root / name):
                pass
            if fresh:
                for relative, content in bounded_tree_entries(self.worktree / name):
                    publish(self.root, f"{name}/{relative}", content)
        # These are controller-created asset links, never copied from Agent files.
        # Do not install the evaluator or copy .git / control checkpoints.
        if fresh:
            link_runtime(self.root, is_ppu=self.is_ppu)
        durable_write_json(self.state_path, dict(state, kernel_digest=digest))
        return self.root

    def publish(self, *, sealed_source: bytes | None = None) -> None:
        from supervisor.workspace import publish

        state = json.loads(_read(self.state_path))
        if sealed_source is None:
            content = _read(self.root / "kernel.py")
        else:
            warning = ""
            try:
                if _read(self.root / "kernel.py") != sealed_source:
                    warning = "post_report_draft_changed"
            except (OSError, ValueError):
                warning = "post_report_draft_unreadable"
            if warning:
                logging.getLogger(__name__).warning(
                    "Episode Kernel integrity warning: %s; preserving sealed candidate; "
                    "Agent draft remains at %s", warning, self.root,
                )
                state["integrity_warnings"] = sorted(set(state.get("integrity_warnings", [])) | {warning})
                # Record the diagnostic even if unrelated publication fails later.
                durable_write_json(self.state_path, state)
            content = sealed_source
        publish(self.worktree, "kernel.py", content)
        for name in DIAGNOSTIC_TREES:
            for relative, data in bounded_tree_entries(self.root / name):
                publish(self.worktree, f"{name}/{relative}", data)
        durable_write_json(self.state_path, dict(state, kernel_digest=hashlib.sha256(content).hexdigest()))
