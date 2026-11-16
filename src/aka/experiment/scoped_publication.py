"""Publish granted bytes through directory descriptors without following links."""
import os
from pathlib import Path
import stat
import uuid


def validate_target(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise PermissionError("scoped publication requires an absolute granted path")
    for parent in path.parents:
        if parent.is_symlink() or parent.exists() and not parent.is_dir():
            raise PermissionError("scoped publication cannot traverse a link or file")
    if path.is_symlink() or path.exists() and (not path.is_file() or path.stat().st_nlink != 1):
        raise PermissionError("scoped publication cannot replace a link or non-regular file")


def _check_leaf(directory, name):
    try:
        existing = os.stat(name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISREG(existing.st_mode) or existing.st_nlink != 1:
        raise PermissionError("scoped publication cannot replace a link or non-regular file")


def publish(path, content, *, mode):
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise PermissionError("safe scoped publication is unavailable")
    validate_target(path)
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    temporary = None
    try:
        for name in path.parent.parts[1:]:
            try:
                os.mkdir(name, mode=0o700, dir_fd=directory)
            except FileExistsError:
                pass
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        _check_leaf(directory, path.name)
        temporary = ".pending-" + uuid.uuid4().hex
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=directory)
        with os.fdopen(fd, "wb") as output:
            output.write(content)
            os.fchmod(output.fileno(), mode)
            output.flush()
            os.fsync(output.fileno())
        _check_leaf(directory, path.name)
        os.replace(temporary, path.name, src_dir_fd=directory, dst_dir_fd=directory)
        temporary = None
        os.fsync(directory)
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary, dir_fd=directory)
            except FileNotFoundError:
                pass
        os.close(directory)
