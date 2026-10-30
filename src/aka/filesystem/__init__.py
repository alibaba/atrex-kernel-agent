"""Filesystem support with caller-owned size limits and publication policy."""
from contextlib import contextmanager, suppress
import os
from pathlib import Path
import stat


@contextmanager
def open_private_directory(path):
    """Open or create a directory chain without following links."""
    path = Path(os.path.abspath(path))
    descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for name in path.parts[1:]:
            with suppress(FileExistsError):
                os.mkdir(name, mode=0o700, dir_fd=descriptor)
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


def _open_regular(path):
    path = Path(path).absolute()
    directory = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    finally:
        os.close(directory)
    stream = os.fdopen(descriptor, "rb")
    try:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("Filesystem input must be a regular file")
    except BaseException:
        stream.close()
        raise
    return stream


def read_regular_bytes(path, limit=-1):
    """Read a bounded prefix; callers detect oversize by requesting limit + 1.

    A limit of -1 reads the whole file. This performs no size-policy decision
    and never follows links in the file or its parent components.
    """
    with _open_regular(path) as stream:
        return stream.read(limit)
