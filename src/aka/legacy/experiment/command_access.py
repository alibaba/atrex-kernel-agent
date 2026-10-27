"""Authorize and snapshot endpoint inputs through pinned, no-follow descriptors."""
from contextlib import contextmanager
import os
from pathlib import Path
import stat

from aka.contracts.execution import ExecutionFile
from aka.contracts.filesystem import FilesystemAccess, FilesystemMount
from aka.contracts.filesystem_access import require_path_access


@contextmanager
def pinned(path):
    path = Path(path)
    if (not path.is_absolute() or ".." in path.parts or not hasattr(os, "O_NOFOLLOW")
            or not hasattr(os, "O_DIRECTORY")):
        raise PermissionError("safe scoped input access is unavailable")
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for index, name in enumerate(path.parts[1:]):
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            if index < len(path.parts) - 2:
                flags |= os.O_DIRECTORY
            child = os.open(name, flags, dir_fd=directory)
            os.close(directory)
            directory = child
        yield directory
    finally:
        os.close(directory)


def _identity(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
            value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def read_input(access, path, *, max_file_bytes=None):
    source = require_path_access(access, path)
    with pinned(source) as descriptor:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise PermissionError("command input must be an unlinked regular file")
        if max_file_bytes is not None and before.st_size > max_file_bytes:
            return None
        chunks = bytearray()
        while True:
            data = os.read(descriptor, 64 * 1024)
            if not data:
                break
            chunks.extend(data)
            if max_file_bytes is not None and len(chunks) > max_file_bytes:
                return None
        if _identity(before) != _identity(os.fstat(descriptor)):
            raise PermissionError("command input changed during its snapshot")
        frozen = bytes(chunks)
        # The current path must still identify the original inode and expose
        # the same bytes on a second complete read. Timestamps alone can miss
        # same-length writes within a filesystem's timestamp resolution. This
        # proves two observable passes agree, not an atomic concurrent snapshot.
        with pinned(source) as current:
            if _identity(before) != _identity(os.fstat(current)):
                raise PermissionError("command input was replaced during its snapshot")
            offset = 0
            while True:
                data = os.read(current, 64 * 1024)
                if not data:
                    break
                if data != frozen[offset:offset + len(data)]:
                    raise PermissionError("command input bytes changed during its snapshot")
                offset += len(data)
            if (offset != len(frozen) or _identity(before) != _identity(os.fstat(current))
                    or _identity(before) != _identity(os.fstat(descriptor))):
                raise PermissionError("command input changed during its snapshot")
        return frozen, stat.S_IMODE(before.st_mode)


def snapshot_inputs(workspace, access, selections, *, max_file_bytes=None):
    from aka.legacy.execution.gateway.sandbox import INPUT_SKIP_DIRS, INPUT_SKIP_PATHS, INPUT_SKIP_SUFFIXES
    files, skipped = {}, []

    def visit(relative):
        path = workspace / relative
        source = require_path_access(access, path)
        with pinned(source) as descriptor:
            before = os.fstat(descriptor)
            if stat.S_ISDIR(before.st_mode):
                children = sorted(os.listdir(descriptor))
                for name in children:
                    if name in INPUT_SKIP_DIRS and stat.S_ISDIR(
                            os.stat(name, dir_fd=descriptor, follow_symlinks=False).st_mode):
                        continue
                    visit(relative / name)
                if _identity(before) != _identity(os.fstat(descriptor)):
                    raise PermissionError("command input directory changed during its snapshot")
                with pinned(source) as current:
                    if _identity(before) != _identity(os.fstat(current)):
                        raise PermissionError("command input directory was replaced")
                return
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise PermissionError("command input must be an unlinked regular file")
        name = relative.as_posix()
        if name in INPUT_SKIP_PATHS or relative.suffix in INPUT_SKIP_SUFFIXES:
            skipped.append(name)
            return
        frozen = read_input(access, path, max_file_bytes=max_file_bytes)
        if frozen is None:
            skipped.append(name)
        else:
            content, mode = frozen
            files[name] = ExecutionFile(name, content, mode)

    for selection in selections:
        relative = Path(selection)
        if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
            raise ValueError("explicit command input must be workspace-relative")
        visit(relative)
    return tuple(files[name] for name in sorted(files)), tuple(skipped)


def client_access(access):
    """Project existing grants into the Agent view, including source aliases."""
    def restrictions(paths):
        result = set(paths)
        for path in paths:
            physical = []
            for mount in access.mounts:
                if path.is_relative_to(mount.target):
                    physical.append(mount.source / path.relative_to(mount.target))
                elif mount.target.is_relative_to(path):
                    physical.append(mount.source)
            for source in physical:
                for mount in access.mounts:
                    if source.is_relative_to(mount.source):
                        result.add(mount.target / source.relative_to(mount.source))
                    elif mount.source.is_relative_to(source):
                        result.add(mount.target)
        return tuple(sorted(result, key=str))
    readonly = (*access.read_only_paths, *(mount.target for mount in access.mounts if not mount.writable))
    return FilesystemAccess(tuple(FilesystemMount(mount.target, mount.target, mount.writable)
                                  for mount in access.mounts),
                            restrictions(access.hidden_paths), restrictions(readonly))


def access_data(access):
    return {"mounts": [{"source": str(m.source), "target": str(m.target), "writable": m.writable}
                       for m in access.mounts], "hidden_paths": list(map(str, access.hidden_paths)),
            "read_only_paths": list(map(str, access.read_only_paths))}


def decode_access(value):
    return FilesystemAccess(tuple(FilesystemMount(Path(m["source"]), Path(m["target"]), m["writable"])
                                  for m in value["mounts"]),
                            tuple(map(Path, value["hidden_paths"])),
                            tuple(map(Path, value["read_only_paths"])))
