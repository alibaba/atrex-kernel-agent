"""Workflow chooses immutable reusable-file checkpoints across Episodes."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import re
import tempfile

from aka.contracts.workspace import WorkspaceFilesSnapshot
from aka.filesystem import open_private_directory, read_regular_bytes
from aka.light.orchestrator.durable_state import durable_write_json, fsync_directory
from aka.contracts.content import WorkspaceFile

TREES = ("skills", "tools")
READMES = {
    "skills": """# Reusable Skills

Store reusable methods, workflows and lessons in `<name>/SKILL.md`, with supporting
files alongside it. Maintain an index below: name, purpose, when to use it and its
entry path. Update this README whenever you add, change or remove a custom Skill.
Read new Skills directly; discovery links are refreshed before the next Session.

Supervisor-installed Skills are reserved and read-only in sandboxed Sessions.
Do not replace them. Your own Skill directories are writable. Put one-off probes,
requests and outputs in `scratch/`, not here. The Session contract states whether
custom Skills carry forward to the next Episode; current files are your starting state.

## Custom Skill index

No custom Skills yet.
""",
    "tools": """# Reusable tools

Store reusable scripts here. Maintain an index below: path, purpose, invocation,
dependencies and important limitations. Update this README whenever you add,
change or remove a custom tool. Use `scratch/` for temporary inputs and outputs.

`sandbox.py` and `plugin.py` are Supervisor-installed HTTP clients: do not modify,
replace or duplicate them. Your other scripts are writable. To execute a custom
GPU probe, use the GPU measurement Skill, for example:
`python3 tools/sandbox.py --kind dev --no-sync -- python3 tools/probe.py`.
Declare extra script dependencies with `--input`; local helpers may run directly.
The Session contract states whether custom tools carry forward to the next Episode.

## Custom tool index

No custom tools yet.
""",
}



def ensure_readmes(workspace, workspace_provider):
    files = tuple(WorkspaceFile(root + "/README.md", text=text) for root, text in READMES.items()
                  if not (workspace / root / "README.md").exists())
    if files:
        workspace_provider.install_files(workspace, files)


def _pointer(path):
    value = json.loads(read_regular_bytes(path, limit=2 * 1024 * 1024))
    if (not isinstance(value, dict) or value.get("schema_version") != 2
            or not isinstance(value.get("snapshot"), dict)
            or set(value["snapshot"]) != {"directory", "identity"}):
        raise ValueError("Reusable state requires a shared Workspace files snapshot")
    return value


class AdaptiveState:
    def __init__(self, root, episode, reserved, workspace_provider):
        if type(episode) is not int or episode < 1:
            raise ValueError("Reusable state needs a controller Episode number")
        self.root, self.episode, self.reserved = root, episode, reserved
        self.workspace_provider = workspace_provider

    @property
    def pointer(self):
        return self.root / "episodes" / f"e{self.episode:08d}.json"

    def previous(self):
        directory = self.root / "episodes"
        with open_private_directory(directory):
            pass
        for path in sorted(directory.glob("e*.json"), reverse=True):
            if re.fullmatch(r"e[0-9]{8,}\.json", path.name) and int(path.stem[1:]) < self.episode:
                return _pointer(path)
        return None

    def restore(self, workspace, pointer):
        record = pointer["snapshot"]
        snapshot = WorkspaceFilesSnapshot(Path(record["directory"]), record["identity"])
        self.workspace_provider.restore_files(snapshot, workspace, exclude_paths=self.reserved)

    def save(self, workspace):
        with open_private_directory(self.root) as parent:
            lock = os.open(".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600, dir_fd=parent)
            try:
                fcntl.flock(lock, fcntl.LOCK_EX)
                directory = self.root / "snapshots"
                with open_private_directory(directory):
                    pass
                with tempfile.TemporaryDirectory(prefix=".saving-", dir=directory) as temporary:
                    snapshot = self.workspace_provider.snapshot_files(workspace, Path(temporary) / "files",
                        directories=TREES, exclude_paths=self.reserved)
                    target = directory / snapshot.identity
                    if target.exists():
                        # The owner verifies the content already stored at this identity.
                        confirmed = self.workspace_provider.snapshot_files(workspace, target,
                            directories=TREES, exclude_paths=self.reserved)
                        if confirmed.identity != snapshot.identity:
                            raise ValueError("Reusable checkpoint content identity changed")
                    else:
                        os.rename(snapshot.directory, target)
                        fsync_directory(directory)
                value = {"schema_version": 2, "episode": self.episode,
                         "snapshot": {"directory": str(target), "identity": snapshot.identity}}
                durable_write_json(self.pointer, value)
                return value
            finally:
                os.close(lock)
