"""Recoverable, controller-only publication of a selected search checkpoint."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

from aka.legacy.task.persistence import durable_write_json
from aka.light.workspace.files import publish
from aka.light.agent.session_tail import read_regular_bytes
from aka.light.workspace.git import git_blob, git_head
from .search import read_state


def git(workspace: Path, *args, data=None, environment=None) -> bytes:
    result = subprocess.run(["git", *args], cwd=workspace, input=data,
                            capture_output=True, env=environment)
    if result.returncode:
        raise RuntimeError(f"Search Git operation failed: {result.stderr.decode(errors='replace')[-1200:]}")
    return result.stdout


def dirty_paths(workspace):
    # NUL delimiting avoids treating spaces/quotes as pathname syntax.
    values = git(workspace, "diff", "--name-only", "-z") + git(workspace, "diff", "--cached", "--name-only", "-z")
    return {value.decode() for value in values.split(b"\0") if value}


def commit_files(workspace: Path, transaction: Path, files: dict[str, bytes | None], message: str) -> str:
    """No reset/cherry-pick and no hooks. Never overwrite unrelated dirty files.

    Prepare the complete immutable Git commit before changing HEAD or working files.
    The durable receipt makes a crash between publication, ref update and index
    refresh replayable. This runs only while no Episode uses this controller copy.
    """
    checksums = {name: hashlib.sha256(data).hexdigest() if data is not None else None for name, data in files.items()}
    try:
        receipt = read_state(transaction)
    except FileNotFoundError:
        if dirty_paths(workspace):
            raise RuntimeError("Search publication requires a clean controller index and tracked workspace")
        base = git_head(workspace)
        with tempfile.TemporaryDirectory(prefix="aka-search-index-") as temporary:
            environment = dict(os.environ, GIT_INDEX_FILE=str(Path(temporary) / "index"))
            git(workspace, "read-tree", base, environment=environment)
            for name, data in files.items():
                if data is None:
                    git(workspace, "update-index", "--force-remove", "--", name, environment=environment)
                    continue
                blob = git(workspace, "hash-object", "-w", "--stdin", data=data).decode().strip()
                git(workspace, "update-index", "--add", "--cacheinfo", "100644", blob, name,
                    environment=environment)
            tree = git(workspace, "write-tree", environment=environment).decode().strip()
            target = git(workspace, "-c", "commit.gpgSign=false", "commit-tree", tree,
                         "-p", base, "-m", message).decode().strip()
        receipt = {"base": base, "target": target, "files": checksums, "message": message}
        durable_write_json(transaction, receipt)
    if receipt["files"] != checksums or receipt["message"] != message:
        raise RuntimeError("Search publication replay differs from its sealed transaction")
    if git_head(workspace) not in {receipt["base"], receipt["target"]}:
        raise RuntimeError("Controller HEAD changed outside its pending search publication")
    if not dirty_paths(workspace) <= files.keys():
        raise RuntimeError("Unrelated tracked changes block search publication recovery")
    for name, data in files.items():
        target = subprocess.run(["git", "show", f"{receipt['target']}:{name}"], cwd=workspace, capture_output=True)
        if (target.stdout if target.returncode == 0 else None) != data:
            raise RuntimeError("Search publication commit failed integrity verification")
        try:
            current = read_regular_bytes(workspace / name, limit=16 * 1024 * 1024 + 1)
        except FileNotFoundError:
            current = None
        old = subprocess.run(["git", "show", f"{receipt['base']}:{name}"],
                             cwd=workspace, capture_output=True)
        previous = old.stdout if old.returncode == 0 else None
        if current not in (previous, data):
            raise RuntimeError(f"Refusing to replace unexpected content during search publication: {name}")
    for name, data in files.items():
        if data is None:
            from aka.light.agent.home import open_private_directory
            with open_private_directory((workspace / name).parent) as parent:
                try:
                    os.unlink(Path(name).name, dir_fd=parent)
                except FileNotFoundError:
                    pass
        else:
            publish(workspace, name, data)
    if git_head(workspace) == receipt["base"]:
        git(workspace, "update-ref", "HEAD", receipt["target"], receipt["base"])
    git(workspace, "read-tree", receipt["target"])  # Index only; no broad worktree reset.
    return receipt["target"]
