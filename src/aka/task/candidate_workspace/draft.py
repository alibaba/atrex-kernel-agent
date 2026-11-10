"""Explicit Git-free roots with a controller-owned, hidden metadata placeholder."""
import os
from pathlib import Path

from .files import _directory, _read_descriptor

MARKER = "aka-git-free-draft"
FORMAT = b"aka-git-free-draft-v1\n"


def _validate(directory):
    if set(os.listdir(directory)) != {MARKER}:
        raise ValueError("Git-free draft contains unexpected Git metadata")
    descriptor = os.open(MARKER, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    try:
        if _read_descriptor(descriptor, len(FORMAT))[0] != FORMAT:
            raise ValueError("Invalid Git-free draft metadata")
    finally:
        os.close(descriptor)


def is_draft(workspace):
    metadata = Path(workspace) / ".git"
    if not metadata.is_dir() or not (metadata / MARKER).exists():
        return False
    with _directory(metadata) as directory:
        _validate(directory)
    return True


def prepare_draft(workspace):
    """Declare a draft, accepting only an absent/empty previous mountpoint.

    Never convert a real or damaged repository into a draft. The marker stays
    below the hidden .git mount, so sandboxed Agent writes cannot change it.
    """
    with _directory(Path(workspace) / ".git", create=True) as directory:
        if not os.listdir(directory):
            descriptor = os.open(MARKER, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                 0o600, dir_fd=directory)
            try:
                with os.fdopen(descriptor, "wb", closefd=False) as stream:
                    stream.write(FORMAT)
                    stream.flush()
                    os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.fsync(directory)
        _validate(directory)
