"""Restore explicit Git revisions with index locking, CAS and file rollback."""
from pathlib import Path, PurePosixPath
import os
import shutil
import stat
import tempfile

from .history import resolve_revision, validate_paths
from .records import _git, current_revision


def _tracked_paths(workspace, revision=None):
    if revision is None:
        result = _git(workspace, "ls-files", "--stage", "-z")
    else:
        result = _git(workspace, "ls-tree", "-r", "-z", "--full-tree", revision)
    paths = []
    for entry in result.stdout.split(b"\0"):
        if entry:
            paths.append(entry.split(b"\t", 1)[1].decode("utf-8", errors="surrogateescape"))
    return validate_paths(tuple(dict.fromkeys(paths)))


def _exists(path):
    return path.exists() or path.is_symlink()


def _backup_roots(workspace, paths):
    roots = set()
    directories = set()
    for name in paths:
        relative = PurePosixPath(name)
        root = name
        for number in range(1, len(relative.parts)):
            parent_name = PurePosixPath(*relative.parts[:number]).as_posix()
            parent = workspace / parent_name
            if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
                root = parent_name
                break
        roots.add(root)
        parent = (workspace / root).parent
        while parent != workspace and not _exists(parent):
            directories.add(parent)
            parent = parent.parent
    minimal = []
    for name in sorted(roots, key=lambda item: (len(PurePosixPath(item).parts), item)):
        path = PurePosixPath(name)
        if not any(PurePosixPath(other) in path.parents for other in minimal):
            minimal.append(name)
    return tuple(minimal), directories


def _copy(source, destination):
    if source.is_symlink():
        destination.symlink_to(os.readlink(source))
    elif source.is_dir():
        shutil.copytree(source, destination, symlinks=True)
    elif source.is_file():
        shutil.copy2(source, destination)
    else:
        raise ValueError(f"unsupported workspace restore file type: {source}")


def _remove(path):
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        path.unlink()
    elif path.exists():
        def writable(function, target, error):
            os.chmod(target, os.lstat(target).st_mode | stat.S_IWUSR | stat.S_IXUSR)
            function(target)
        shutil.rmtree(path, onerror=writable)


def _restore_files(workspace, roots, backup, present, directories):
    for number, name in enumerate(roots):
        target = workspace / name
        _remove(target)
        if present[number]:
            target.parent.mkdir(parents=True, exist_ok=True)
            _copy(backup / str(number), target)
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass


def restore_revision(workspace: Path, *, expected_revision: str,
                     revision: str, retain_working_files: bool) -> str:
    workspace = Path(workspace).absolute()
    if type(retain_working_files) is not bool:
        raise TypeError("retain_working_files must be a boolean")
    if current_revision(workspace) != expected_revision:
        raise RuntimeError("workspace revision advanced; refusing restore")
    target_revision = resolve_revision(workspace, revision)
    index = Path(_git(workspace, "rev-parse", "--git-path", "index").stdout.decode().strip())
    if not index.is_absolute():
        index = workspace / index
    if index.is_symlink():
        raise ValueError("workspace Git index must not be a symlink")
    lock = Path(str(index) + ".lock")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
    os.close(descriptor)
    try:
        if current_revision(workspace) != expected_revision:
            raise RuntimeError("workspace revision advanced; refusing restore")
        original_index = index.read_bytes() if index.exists() else None
        original_mode = stat.S_IMODE(index.stat().st_mode) if original_index is not None else None
        # Validate every tracked destination before any checkout or index change.
        paths = tuple(dict.fromkeys((*_tracked_paths(workspace),
                                   *_tracked_paths(workspace, target_revision))))
        roots, directories = _backup_roots(workspace, paths)
        with tempfile.TemporaryDirectory(prefix="aka-restore-", dir=index.parent) as temporary:
            temporary = Path(temporary)
            prepared = temporary / "index"
            if original_index is not None:
                shutil.copy2(index, prepared)
            else:
                _git(workspace, "read-tree", "--empty", index=prepared)
            backup = temporary / "files"
            backup.mkdir()
            present = []
            if not retain_working_files:
                for number, name in enumerate(roots):
                    target = workspace / name
                    present.append(_exists(target))
                    if present[-1]:
                        _copy(target, backup / str(number))
            modified = published = False
            try:
                modified = not retain_working_files
                arguments = ("--reset", "-u") if modified else ("--reset",)
                _git(workspace, "read-tree", *arguments, target_revision, index=prepared)
                try:
                    _git(workspace, "update-ref", "-m", f"reset: moving to {target_revision}",
                         "HEAD", target_revision, expected_revision)
                    published = True
                except BaseException:
                    try:
                        actual = current_revision(workspace)
                    except BaseException:
                        # Publication is uncertain; never erase a possibly confirmed restore.
                        published = True
                        raise
                    published = actual == target_revision
                    if published:
                        os.replace(prepared, index)
                    raise
                os.replace(prepared, index)
                return target_revision
            except BaseException:
                if not published:
                    if modified:
                        _restore_files(workspace, roots, backup, present, directories)
                    if original_index is None:
                        index.unlink(missing_ok=True)
                    elif not index.exists() or index.read_bytes() != original_index:
                        if index.exists():
                            index.chmod(original_mode | stat.S_IWUSR)
                        index.write_bytes(original_index)
                        index.chmod(original_mode)
                raise
    finally:
        lock.unlink(missing_ok=True)
