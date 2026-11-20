"""Bounded content snapshots and exact Git revisions, independent of source layout."""
from __future__ import annotations
import hashlib
import json
import os
import shutil
from pathlib import Path
import stat
import subprocess
import uuid

MAX_BYTES = 64 * 1024 * 1024
MAX_FILES = 4096


def document(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(document(value).encode()).hexdigest()


def relative(name):
    path = Path(name)
    if not isinstance(name, str) or not name or path.is_absolute() or ".." in path.parts or path == Path(".") or ".git" in path.parts:
        raise ValueError(f"Invalid content path: {name!r}")
    return path


def tree(root, *, include_scratch=False, include_ignored=False, allow_empty=False):
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"Content root must be a regular directory: {root}")
    selected = None
    if not include_ignored and (root / ".git").exists():
        selected = {name.decode() for name in git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard", binary=True).split(b"\0") if name}
    result, total = {}, 0
    for current, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(name for name in dirs if include_ignored or name not in {
            ".git", "__pycache__", ".venv", ".aka-cache", ".aka-output", ".aka-build"})
        for name in dirs:
            if (Path(current) / name).is_symlink():
                raise ValueError("Content snapshots cannot contain symlink directories")
        for name in sorted(names):
            path = Path(current) / name
            relative_name = path.relative_to(root).as_posix()
            if selected is not None and relative_name not in selected and not (include_scratch and relative_name.startswith("scratch/")):
                continue
            if not include_ignored and name.endswith(".pyc"):
                continue
            if not stat.S_ISREG(path.lstat().st_mode):
                raise ValueError("Content snapshots require regular files")
            data = path.read_bytes()
            total += len(data)
            if total > MAX_BYTES or len(result) >= MAX_FILES:
                raise ValueError("Content snapshot exceeds 64 MiB / 4096 files")
            result[relative_name] = data
    if not result and not allow_empty:
        raise ValueError("Content snapshot is empty")
    return result


def identity(files):
    return digest({name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())})


def materialize(root, files):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        path = root / relative(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def atomic_snapshot(root, files, *, include_ignored=False):
    """Publish a complete source directory, leaving no partial snapshot on ENOSPC."""
    root = Path(root)
    expected = identity(files)
    if root.exists():
        if identity(tree(root,include_ignored=include_ignored,allow_empty=not files)) != expected:
            raise ValueError('Frozen candidate source changed')
        return
    staging = root.parent.parent / '.source-staging' / uuid.uuid4().hex
    try:
        materialize(staging, files)
        if identity(tree(staging,include_ignored=include_ignored,allow_empty=not files)) != expected:
            raise ValueError('Staged source snapshot does not match its content hash')
        root.parent.mkdir(parents=True, exist_ok=True)
        os.rename(staging, root)
    finally:
        if staging.exists():shutil.rmtree(staging)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    with temporary.open("x", encoding="utf-8") as output:
        output.write(document(value) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def git(root, *args, binary=False):
    result = subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgSign=false", *args],
                            cwd=root, capture_output=True, check=True)
    return result.stdout if binary else result.stdout.decode().strip()


def initialize_source(root, files):
    materialize(root, files)
    git(root, "init", "-q")
    git(root, "config", "user.name", "AKA Agent")
    git(root, "config", "user.email", "agent@localhost")
    (Path(root) / ".git" / "info" / "exclude").write_text(
        "__pycache__/\n*.pyc\n.aka-*/\n.atrex*/\n.claude/\n.codex/\nscratch/\n", encoding="utf-8")
    git(root, "add", "--all")
    git(root, "commit", "-qm", "Initial task source")
    return git(root, "rev-parse", "HEAD")


def committed_source(root):
    if git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Commit the complete candidate and leave the source workspace clean")
    revision = git(root, "rev-parse", "HEAD")
    result, total = {}, 0
    for row in git(root, "ls-tree", "-rz", "HEAD", binary=True).split(b"\0"):
        if not row:
            continue
        header, name = row.split(b"\t", 1)
        mode, kind, blob = header.decode().split()
        if mode not in {"100644", "100755"} or kind != "blob":
            raise ValueError("Candidates cannot contain symlinks or submodules")
        name = name.decode()
        relative(name)
        data = git(root, "cat-file", "blob", blob, binary=True)
        result[name] = data
        total += len(data)
        if total > MAX_BYTES or len(result) > MAX_FILES:
            raise ValueError("Candidate exceeds the content budget")
    if not result:
        raise ValueError("Candidate is empty")
    return revision, result
