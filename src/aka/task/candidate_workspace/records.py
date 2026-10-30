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


def _write_file(target, content, mode, *, source=None, copy="copy2"):
    target.parent.mkdir(parents=True, exist_ok=True)
    if source is not None:
        if copy == "cp":
            subprocess.run(["cp", str(source), str(target)], check=True, capture_output=True)
        else:
            shutil.copy2(source, target)
    else:
        target.write_bytes(content)
        if mode is not None:
            target.chmod(mode)


def _restore_files(workspace, original, directories):
    for name, prior in original.items():
        target = workspace / name
        if prior is None:
            target.unlink(missing_ok=True)
        else:
            # A partial copy2 may already have applied a read-only source mode.
            if target.exists():
                target.chmod(prior[1] | stat.S_IWUSR)
            target.write_bytes(prior[0])
            target.chmod(prior[1])
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass


def _stage_files(workspace, index, selected, modes, deleted_paths=()):
    if deleted_paths:
        _git(workspace, "update-index", "--force-remove", "--", *deleted_paths, index=index)
    for name, content in selected.items():
        blob = _git(workspace, "hash-object", "-w", "--stdin", data=content).stdout.decode().strip()
        mode = "100755" if modes[name] is not None and modes[name] & 0o111 else "100644"
        _git(workspace, "update-index", "--add", "--cacheinfo", f"{mode},{blob},{name}", index=index)


def record(workspace: Path, *, expected_revision: str,
           files: tuple[WorkspaceFile, ...], message: str,
           deleted_paths: tuple[str, ...] = ()) -> str:
    workspace = Path(workspace).absolute()
    if current_revision(workspace) != expected_revision:
        raise RuntimeError("workspace revision advanced; refusing record")
    files = tuple(files)
    deleted_paths = tuple(deleted_paths)
    # Combined validation rejects overlap, traversal and non-file replacements.
    _selected(workspace, (*files, *(WorkspaceFile(name, text="") for name in deleted_paths)))
    if deleted_paths:
        from .source import validate_source_deletions
        validate_source_deletions(workspace, deleted_paths)
    from .context import context_record_file, uses_context
    if uses_context(workspace):
        anchor = context_record_file(workspace, (*tuple(item.path for item in files), *deleted_paths))
        if anchor is not None:
            files = (*files, anchor)
    # Validate sources before loading bytes (directories/FIFOs must not be read).
    for item in files:
        if item.source is not None:
            mode = item.source.lstat().st_mode
            if not stat.S_ISREG(mode) or mode & (stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX):
                raise ValueError(f"workspace source must be a regular file with ordinary modes: {item.path}")
    selected = _selected(workspace, files)
    from .source import validate_source_record
    validate_source_record(workspace, selected)
    modes, original, directories = {}, {}, set()
    for item in files:
        target = workspace / item.path
        if target.exists() and target.stat().st_nlink != 1:
            raise ValueError(f"workspace record target must not have hard links: {item.path}")
        prior = (target.read_bytes(), stat.S_IMODE(target.stat().st_mode)) if target.exists() else None
        original[item.path] = prior
        modes[item.path] = (stat.S_IMODE(item.source.stat().st_mode)
                            if item.source is not None and (item.copy == "copy2" or prior is None)
                            else prior[1] if prior is not None else None)
        if modes[item.path] is not None and modes[item.path] & (stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX):
            raise ValueError(f"unsupported workspace file mode: {item.path}")
        parent = target.parent
        while parent != workspace and not parent.exists():
            directories.add(parent)
            parent = parent.parent
    for name in deleted_paths:
        target = workspace / name
        if target.exists() and target.stat().st_nlink != 1:
            raise ValueError(f"workspace record target must not have hard links: {name}")
        original[name] = (target.read_bytes(), stat.S_IMODE(target.stat().st_mode)) if target.exists() else None
    index_path = Path(_git(workspace, "rev-parse", "--git-path", "index").stdout.decode().strip())
    if not index_path.is_absolute():
        index_path = workspace / index_path
    if index_path.is_symlink():
        raise ValueError("workspace Git index must not be a symlink")
    lock = Path(str(index_path) + ".lock")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
    os.close(descriptor)
    revision = None
    committed = False
    prior_index = prior_index_mode = None
    index_snapshot_ready = files_started = False
    try:
        prior_index = index_path.read_bytes() if index_path.exists() else None
        prior_index_mode = stat.S_IMODE(index_path.stat().st_mode) if prior_index is not None else None
        index_snapshot_ready = True
        if current_revision(workspace) != expected_revision:
            raise RuntimeError("workspace revision advanced; refusing record")
        with tempfile.TemporaryDirectory(prefix="aka-record-", dir=index_path.parent) as temporary:
            commit_index = Path(temporary) / "commit-index"
            prepared_index = Path(temporary) / "prepared-index"
            _git(workspace, "read-tree", expected_revision or "--empty", index=commit_index)
            if index_path.exists():
                shutil.copy2(index_path, prepared_index)
            else:
                _git(workspace, "read-tree", expected_revision or "--empty", index=prepared_index)
            _stage_files(workspace, commit_index, selected, modes, deleted_paths)
            _stage_files(workspace, prepared_index, selected, modes, deleted_paths)
            sources = {}
            for number, item in enumerate(files):
                if item.source is not None:
                    snapshot = Path(temporary) / f"source-{number}"
                    snapshot.write_bytes(selected[item.path])
                    shutil.copystat(item.source, snapshot)
                    sources[item.path] = snapshot
            files_started = True
            for name in deleted_paths:
                (workspace / name).unlink(missing_ok=True)
            for item in files:
                name = item.path
                _write_file(workspace / name, selected[name], modes[name],
                            source=sources.get(name), copy=item.copy)
                modes[name] = stat.S_IMODE((workspace / name).stat().st_mode)
            # cp/new-file umask behavior determines the actual recorded executable bit.
            _stage_files(workspace, commit_index, selected, modes, deleted_paths)
            _stage_files(workspace, prepared_index, selected, modes, deleted_paths)
            tree = _git(workspace, "write-tree", index=commit_index).stdout.decode().strip()
            unchanged = expected_revision and tree == _git(
                workspace, "rev-parse", f"{expected_revision}^{{tree}}").stdout.decode().strip()
            if unchanged or not (selected or deleted_paths):
                os.replace(prepared_index, index_path)
                return expected_revision
            parents = ("-p", expected_revision) if expected_revision else ()
            revision = _git(workspace, "-c", "user.name=atrex-long-horizon", "-c",
                            "user.email=atrex-long-horizon@local", "commit-tree", tree,
                            *parents, "-m", message).stdout.decode().strip()
            try:
                _git(workspace, "update-ref", "HEAD", revision, expected_revision)
                committed = True
            except BaseException:
                # An interrupted client may have missed a successful ref update.
                try:
                    confirmed = _git(workspace, "rev-parse", "--verify", "HEAD", check=False)
                except BaseException:
                    # Publication is uncertain: do not erase a possibly committed result.
                    committed = True
                    raise
                if confirmed.returncode:
                    committed = True
                    raise
                committed = not confirmed.returncode and confirmed.stdout.decode().strip() == revision
                if committed:
                    os.replace(prepared_index, index_path)
                raise
            os.replace(prepared_index, index_path)
            return revision
    except BaseException:
        if not committed:
            if files_started:
                _restore_files(workspace, original, directories)
            if index_snapshot_ready and prior_index is None:
                index_path.unlink(missing_ok=True)
            elif index_snapshot_ready and (not index_path.exists() or index_path.read_bytes() != prior_index):
                if index_path.exists():
                    index_path.chmod(prior_index_mode | stat.S_IWUSR)
                index_path.write_bytes(prior_index)
                index_path.chmod(prior_index_mode)
        raise
    finally:
        lock.unlink(missing_ok=True)
