"""Crash-recoverable publication of complete files on local and NFS storage.

The reserved staging name is an intent for exactly one destination and one
expected payload. Cooperating writers serialize with a persistent, never-unlinked
lock. Recovery only removes that staging name after checking its bytes, modes,
link count and inode relationship; unrelated aliases are never repaired.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import stat

from .filesystem import file_signature, open_private_directory

_LOCK = ".aka-publication.lock"


def _same_file(left, right):
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _assert_directory(path, descriptor):
    value = Path(path).lstat()
    if not stat.S_ISDIR(value.st_mode) or not _same_file(value, os.fstat(descriptor)):
        raise ValueError("publication directory changed during IO")


def _read(parent, name, limit):
    before = os.stat(name, dir_fd=parent, follow_symlinks=False)
    if (not stat.S_ISREG(before.st_mode)
            or before.st_mode & (stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX)):
        raise ValueError("publication requires an ordinary regular file")
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    try:
        if file_signature(os.fstat(descriptor)) != file_signature(before):
            raise ValueError("publication file replaced during read")
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            payload = stream.read(limit + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    final = os.stat(name, dir_fd=parent, follow_symlinks=False)
    if file_signature(before) != file_signature(after) or file_signature(before) != file_signature(final):
        raise ValueError("publication file changed during read")
    return payload, final


def _sync_verified(parent, name, expected):
    """Persist the already-verified inode, rejecting changes across fsync.

    A complete readonly staging file may have survived death before its data
    fsync. Closing this descriptor before publication or alias cleanup is also
    required for NFS; a directory fsync alone never establishes file durability.
    """
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    try:
        if file_signature(os.fstat(descriptor)) != file_signature(expected):
            raise ValueError("publication file changed before data fsync")
        os.fsync(descriptor)
        if file_signature(os.fstat(descriptor)) != file_signature(expected):
            raise ValueError("publication file changed during data fsync")
    finally:
        os.close(descriptor)
    current = os.stat(name, dir_fd=parent, follow_symlinks=False)
    if file_signature(current) != file_signature(expected):
        raise ValueError("publication file changed after data fsync")


def _optional(parent, name, limit):
    try:
        return _read(parent, name, limit)
    except FileNotFoundError:
        return None


def _ordinary(value, mode, links=1):
    if stat.S_IMODE(value.st_mode) != mode or value.st_nlink != links:
        raise ValueError("publication file has unexpected modes or aliases")


@contextmanager
def _locked(path, lock_directory=None):
    path = Path(path).absolute()
    if path.name == _LOCK:
        raise ValueError("publication destination overlaps the reserved writer lock")
    with open_private_directory(path.parent) as parent:
        directory = path.parent if lock_directory is None else Path(lock_directory).absolute()
        with open_private_directory(directory) as locks:
            flags = os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK
            try:
                descriptor = os.open(_LOCK, flags | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=locks)
            except FileExistsError:
                descriptor = os.open(_LOCK, flags, dir_fd=locks)
            else:
                # Only a newly created lock is ours to normalize. Never chmod
                # an existing object that could be an unknown alias.
                try:
                    os.fchmod(descriptor, 0o600)
                    os.fsync(descriptor)
                except BaseException:
                    os.close(descriptor)
                    raise
            try:
                before = os.fstat(descriptor)
                if not stat.S_ISREG(before.st_mode) or before.st_size != 0:
                    raise ValueError("publication lock must be an empty regular file")
                _ordinary(before, 0o600)
                # NFS exclusive locks require a write-capable descriptor.
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                current = os.stat(_LOCK, dir_fd=locks, follow_symlinks=False)
                if not _same_file(current, os.fstat(descriptor)) or current.st_size != 0:
                    raise ValueError("publication lock changed")
                _ordinary(current, 0o600)
                os.fsync(locks)
                _assert_directory(directory, locks)
                _assert_directory(path.parent, parent)
                yield path, parent
                _assert_directory(path.parent, parent)
            finally:
                os.close(descriptor)


def _unlink(parent, name, expected):
    current = os.stat(name, dir_fd=parent, follow_symlinks=False)
    if file_signature(current) != file_signature(expected):
        raise ValueError("publication staging file changed before cleanup")
    os.unlink(name, dir_fd=parent)
    os.fsync(parent)


def _stage(parent, name, payload, mode):
    descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=parent)
    with os.fdopen(descriptor, "wb") as stream:
        # Establish a readable staging intent even under restrictive umasks,
        # before writing bytes that may need recovery after process death.
        os.fchmod(stream.fileno(), 0o600)
        stream.write(payload)
        stream.flush()
        os.fchmod(stream.fileno(), mode)
        os.fsync(stream.fileno())
    # Closing before link/unlink avoids NFS silly-rename aliases.
    os.fsync(parent)


def publish_immutable(path, payload):
    """Publish 0444 content without replacing a destination, recovering its intent.

    A crash after link leaves exactly the destination and its reserved staging
    alias. Only that verified pair can be reconciled. A crash while staging can
    be retried only with the same expected payload. Unknown aliases or altered
    content remain errors and are preserved for diagnosis. Death between
    exclusive creation and initial permission normalization can leave an
    unrecognized mode; that boundary fails closed instead of chmodding an
    unknown file during recovery.
    """
    with _locked(path) as (path, parent):
        temporary = "." + path.name + ".aka-publish"
        staged = _optional(parent, temporary, len(payload))
        existing = _optional(parent, path.name, len(payload))
        if staged is not None:
            data, value = staged
            if value.st_nlink == 2:
                if (existing is None or not _same_file(value, existing[1])
                        or data != payload or existing[0] != payload):
                    raise ValueError("unknown immutable publication alias")
                _ordinary(value, 0o444, links=2)
                _sync_verified(parent, temporary, value)
                _unlink(parent, temporary, value)
                staged = None
                existing = _optional(parent, path.name, len(payload))
            else:
                if (value.st_nlink != 1 or stat.S_IMODE(value.st_mode) not in (0o600, 0o444)
                        or not payload.startswith(data)):
                    raise ValueError("immutable publication intent differs from requested content")
                if data != payload or stat.S_IMODE(value.st_mode) != 0o444:
                    _unlink(parent, temporary, value)
                    staged = None
        if existing is not None:
            _ordinary(existing[1], 0o444)
            if existing[0] != payload:
                raise ValueError("immutable publication identity collision")
            _sync_verified(parent, path.name, existing[1])
            if staged is not None:
                _unlink(parent, temporary, staged[1])
            return
        if staged is None:
            _stage(parent, temporary, payload, 0o444)
        else:
            # A previous writer may have set readonly mode but died before
            # fsync returned. Verify and persist the retained intent again.
            _sync_verified(parent, temporary, staged[1])
        # The lock serializes participating publishers; link still protects
        # against an unrelated creator, without any replacing rename fallback.
        try:
            os.link(temporary, path.name, src_dir_fd=parent, dst_dir_fd=parent,
                    follow_symlinks=False)
        except FileExistsError:
            other, value = _read(parent, path.name, len(payload))
            _ordinary(value, 0o444)
            if other != payload:
                raise ValueError("immutable publication identity collision")
            _sync_verified(parent, path.name, value)
        os.fsync(parent)
        data, value = _read(parent, temporary, len(payload))
        if data != payload or value.st_nlink not in (1, 2):
            raise ValueError("immutable publication staging changed")
        _ordinary(value, 0o444, links=value.st_nlink)
        if value.st_nlink == 2:
            target = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            if not _same_file(value, target):
                raise ValueError("immutable publication has an unknown alias")
        _unlink(parent, temporary, value)
        data, value = _read(parent, path.name, len(payload))
        _ordinary(value, 0o444)
        if data != payload:
            raise ValueError("immutable publication content changed")


def replace_bytes(path, payload, *, mode=0o644, temporary_name=None, lock_directory=None):
    """Durably replace a mutable file and reconcile only its matching intent.

    This is for a caller-owned mutable management file, never immutable content.
    Recovery accepts a complete or interrupted prefix of the requested bytes,
    with a single link and only staging/final ordinary permissions. A different
    intent is preserved and reported rather than silently discarded.
    """
    if type(mode) is not int or not 0 <= mode <= 0o777:
        raise ValueError("invalid publication mode")
    with _locked(path, lock_directory) as (path, parent):
        temporary = temporary_name or ("." + path.name + ".aka-replace")
        if Path(temporary).name != temporary or temporary in (path.name, _LOCK, ".", ".."):
            raise ValueError("invalid publication staging name")
        existing = _optional(parent, path.name, len(payload))
        if existing is not None:
            _ordinary(existing[1], mode)
        staged = _optional(parent, temporary, len(payload))
        if staged is not None:
            data, value = staged
            if (value.st_nlink != 1 or stat.S_IMODE(value.st_mode) not in (0o600, mode)
                    or not payload.startswith(data)):
                raise ValueError("publication intent differs from requested content")
            _unlink(parent, temporary, value)
        _stage(parent, temporary, payload, mode)
        os.replace(temporary, path.name, src_dir_fd=parent, dst_dir_fd=parent)
        os.fsync(parent)
        data, value = _read(parent, path.name, len(payload))
        _ordinary(value, mode)
        if data != payload:
            raise ValueError("replaced publication content changed")
