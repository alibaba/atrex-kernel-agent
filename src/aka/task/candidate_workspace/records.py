"""Mechanical working-file reads and explicit, nonpromotion Git transactions."""
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tempfile

from aka.contracts.content import WorkspaceFile
from aka.contracts.workspace import RevisionContent
from .content import selected_file_bytes


def _git(workspace, *args, index=None, data=None, check=True):
    environment = os.environ.copy()
    environment.pop("GIT_INDEX_FILE", None)
    if index is not None:
        environment["GIT_INDEX_FILE"] = str(index)
    result = subprocess.run(["git", *args], cwd=workspace, env=environment,
                            input=data, capture_output=True)
    if check and result.returncode:
        raise RuntimeError(f"workspace git {args[0]} failed: "
                           f"{result.stderr.decode(errors='replace')[-1200:]}")
    return result


def current_revision(workspace: Path) -> str:
    workspace = Path(workspace).absolute()
    if workspace.is_symlink():
        raise ValueError("workspace must not be a symlink")
    if not workspace.exists():
        return ""
    if not workspace.is_dir():
        raise ValueError("workspace must be a directory")
    metadata = workspace / ".git"
    if metadata.is_symlink():
        raise ValueError("workspace Git metadata must not be a symlink")
    if not metadata.exists():
        return ""
    # Validate metadata separately so corrupt repositories are not reported as unborn.
    _git(workspace, "rev-parse", "--git-dir")
    result = _git(workspace, "rev-parse", "--verify", "HEAD^{commit}", check=False)
    if not result.returncode:
        return result.stdout.decode().strip()
    symbolic = _git(workspace, "symbolic-ref", "--quiet", "HEAD", check=False)
    if not symbolic.returncode:
        reference = symbolic.stdout.decode().strip()
        missing = _git(workspace, "show-ref", "--verify", "--quiet", reference, check=False)
        if missing.returncode == 1:
            return ""
    raise RuntimeError("cannot read workspace revision: "
                       + result.stderr.decode(errors="replace")[-1200:])


def read_working(workspace: Path, paths: tuple[str, ...]) -> RevisionContent:
    workspace = Path(workspace).absolute()
    if not workspace.is_dir():
        raise ValueError("workspace must be a directory")
    _selected(workspace, tuple(WorkspaceFile(name, text="") for name in paths))
    return RevisionContent({name: (workspace / name).read_bytes()
                            if (workspace / name).exists() else None for name in paths})


def _selected(workspace, files):
    for item in files:
        if any(part.casefold() == ".git" for part in PurePosixPath(item.path).parts):
            raise ValueError(f"workspace path contains Git metadata: {item.path}")
    return selected_file_bytes(workspace, files)
