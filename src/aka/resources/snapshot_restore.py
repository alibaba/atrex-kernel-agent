"""Prevalidated, no-replace transactions for exact original resource roots."""
from contextlib import ExitStack
import ctypes
from dataclasses import dataclass
import errno
import os
from pathlib import Path
import secrets
import stat
import sys


def _exists(path):
    return path.exists() or path.is_symlink()


def _parents_are_directories(path):
    for parent in path.parents:
        if parent.is_symlink() or (_exists(parent) and not parent.is_dir()):
            raise ValueError(f"original resource parent is replaced: {path}")


def validate_original_destinations(manifest, *, absent=()):
    """Read every original root before any staging or destination write."""
    from . import snapshot_store
    missing = []
    for target in absent:
        _parents_are_directories(target)
        if _exists(target):
            raise ValueError(f"original optional resource appeared after retention: {target}")
    for row in manifest["roots"]:
        target = Path(row["source"])
        _parents_are_directories(target)
        if _exists(target):
            if snapshot_store.inspect_root(target) != row:
                raise ValueError(f"original resource differs from retained identity: {target}")
        else:
            missing.append(row)
    return tuple(missing)


def _identity(value):
    return value.st_dev, value.st_ino


def _stat(parent, name):
    try:
        return os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _open_directory(parent, name, stack):
    descriptor = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
    stack.callback(os.close, descriptor)
    return descriptor


@dataclass(frozen=True)
class _Directory:
    path: Path
    descriptor: int
    chain: tuple

    def validate(self):
        for parent, name, identity in self.chain:
            value = _stat(parent, name)
            if value is None or not stat.S_ISDIR(value.st_mode) or _identity(value) != identity:
                raise ValueError(f"original resource parent is replaced: {self.path}")


def _pin_directory(path, stack, created):
    descriptor = _open_directory(None, path.anchor, stack)
    chain = []
    current = Path(path.anchor)
    for name in path.parts[1:]:
        parent = descriptor
        current = current / name
        if _stat(parent, name) is None:
            _Directory(current.parent, parent, tuple(chain)).validate()
            try:
                os.mkdir(name, dir_fd=parent)
            except FileExistsError:
                pass
            else:
                value = _stat(parent, name)
                created.append((parent, name, _identity(value)))
                os.fsync(parent)
        descriptor = _open_directory(parent, name, stack)
        chain.append((parent, name, _identity(os.fstat(descriptor))))
    result = _Directory(path, descriptor, tuple(chain))
    result.validate()
    return result


def _stage_directory(parent, stack, staging):
    parent.validate()
    name = ".resource-restore-" + secrets.token_hex(16)
    os.mkdir(name, mode=0o700, dir_fd=parent.descriptor)
    descriptor = _open_directory(parent.descriptor, name, stack)
    identity = _identity(os.fstat(descriptor))
    staging.append((parent.descriptor, name, identity))
    return _Directory(parent.path / name, descriptor,
                      (*parent.chain, (parent.descriptor, name, identity)))


def _write(parent, name, payload, mode):
    descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         mode=0o600, dir_fd=parent)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fchmod(stream.fileno(), mode)
        os.fsync(stream.fileno())


def _stage(row, directory, reader, stack):
    directory.validate()
    if row["kind"] == "file":
        _write(directory.descriptor, "root", reader(row["entries"][0]["sha256"]), row["mode"])
    else:
        # Stage directory roots as siblings of their targets. Darwin requires a
        # writable directory when changing its parent, even if both parents are
        # writable; a same-parent rename preserves read-only root modes exactly.
        root = directory.descriptor
        directories = {".": root}
        for entry in sorted(row["entries"], key=lambda item: len(Path(item["path"]).parts)):
            path = Path(entry["path"])
            parent = directories[path.parent.as_posix()]
            directory.validate()
            if entry["kind"] == "directory":
                os.mkdir(path.name, dir_fd=parent)
                directories[path.as_posix()] = _open_directory(parent, path.name, stack)
            else:
                _write(parent, path.name, reader(entry["sha256"]), entry["mode"])
        for entry in sorted(row["entries"], key=lambda item: len(Path(item["path"]).parts), reverse=True):
            if entry["kind"] == "directory":
                descriptor = directories[entry["path"]]
                os.fchmod(descriptor, entry["mode"])
                os.fsync(descriptor)
        os.fchmod(root, row["mode"])
        os.fsync(root)
    os.fsync(directory.descriptor)
    return directory.path / "root" if row["kind"] == "file" else directory.path


def _publish_noreplace(staging, target, *, source_parent, target_parent):
    """Publish through pinned directories, without following replaced parent paths."""
    source_parent.validate()
    target_parent.validate()
    library = ctypes.CDLL(None, use_errno=True)
    old, new = os.fsencode(staging.name), os.fsencode(target.name)
    if sys.platform == "darwin":
        function = getattr(library, "renameatx_np", None)
        if function is None:
            raise RuntimeError("resource restoration requires renameatx_np")
        flag = 0x00000004  # Darwin RENAME_EXCL.
    elif sys.platform.startswith("linux"):
        function = getattr(library, "renameat2", None)
        if function is None:
            raise RuntimeError("resource restoration requires renameat2")
        flag = 1  # Linux RENAME_NOREPLACE.
    else:
        raise RuntimeError("no-replace resource restoration is unavailable on this host")
    function.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                         ctypes.c_char_p, ctypes.c_uint)
    function.restype = ctypes.c_int
    result = function(source_parent.descriptor, old, target_parent.descriptor, new, flag)
    if result:
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise FileExistsError(code, os.strerror(code), str(target))
        raise OSError(code, os.strerror(code), str(target))


def _remove_at(parent, name, *, expected_identity=None):
    value = os.stat(name, dir_fd=parent, follow_symlinks=False)
    identity = _identity(value)
    if expected_identity is not None and identity != expected_identity:
        return
    if stat.S_ISDIR(value.st_mode):
        descriptor = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        try:
            if _identity(os.fstat(descriptor)) != identity:
                return
            os.fchmod(descriptor, stat.S_IMODE(value.st_mode) | stat.S_IRWXU)
            for member in os.listdir(descriptor):
                _remove_at(descriptor, member)
        finally:
            os.close(descriptor)
        current = _stat(parent, name)
        if current is not None and _identity(current) == identity:
            os.rmdir(name, dir_fd=parent)
    else:
        current = _stat(parent, name)
        if current is not None and _identity(current) == identity:
            os.unlink(name, dir_fd=parent)


def _cleanup_fact(error, cleanup_error):
    message = f"Resources restoration cleanup failed: {cleanup_error}"
    error.resources_cleanup_errors = (*getattr(error, "resources_cleanup_errors", ()), message)
    add_note = getattr(error, "add_note", None)
    if add_note is not None:
        add_note(message)


def _remove_confirmed(rows, error):
    for parent, name, identity in reversed(rows):
        try:
            value = _stat(parent, name)
            if value is not None and _identity(value) == identity:
                _remove_at(parent, name, expected_identity=identity)
                os.fsync(parent)
        except OSError as cleanup_error:
            _cleanup_fact(error, cleanup_error)


def restore_original_roots(snapshot, manifest, *, finish, absent=()):
    """Publish only missing roots; roll back confirmed publications on BaseException."""
    from . import snapshot_store
    missing = validate_original_destinations(manifest, absent=absent)
    reader = snapshot_store.object_reader(snapshot, manifest)
    installed, parents, staging = [], [], []
    with ExitStack() as stack:
        try:
            staged = []
            for row in missing:
                target = Path(row["source"])
                parent = _pin_directory(target.parent, stack, parents)
                directory = _stage_directory(parent, stack, staging)
                content = _stage(row, directory, reader, stack)
                actual = snapshot_store.inspect_root(content)
                if {**actual, "source": row["source"]} != row:
                    raise ValueError("staged resource identity differs from retained content")
                source_parent = directory if row["kind"] == "file" else parent
                staged.append((content, target, source_parent, parent,
                               _identity(os.stat(content.name, dir_fd=source_parent.descriptor,
                                                 follow_symlinks=False))))
            # All retained objects and staged roots pass validation before publication.
            for content, target, directory, parent, identity in staged:
                try:
                    _publish_noreplace(content, target, source_parent=directory, target_parent=parent)
                finally:
                    # Recognize completed publication even if the syscall's caller was interrupted.
                    value = _stat(parent.descriptor, target.name)
                    if value is not None and _identity(value) == identity:
                        installed.append((parent.descriptor, target.name, identity))
                os.fsync(parent.descriptor)
                parent.validate()
            validate_original_destinations(manifest, absent=absent)
            result = finish()
            cleanup_error = RuntimeError("Resources staging cleanup failed")
            _remove_confirmed(staging, cleanup_error)
            if getattr(cleanup_error, "resources_cleanup_errors", None):
                raise cleanup_error
        except BaseException as error:
            _remove_confirmed(installed, error)
            _remove_confirmed(staging, error)
            for parent, name, identity in reversed(parents):
                try:
                    value = _stat(parent, name)
                    if value is not None and _identity(value) == identity:
                        os.rmdir(name, dir_fd=parent)
                        os.fsync(parent)
                except OSError:
                    pass  # Preserve newly populated parents belonging to another writer.
            raise
        else:
            return result
