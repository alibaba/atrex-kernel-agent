"""Publish selected resource outputs using pinned directory descriptors."""
from pathlib import Path
import os
import uuid

from aka.filesystem import open_private_directory


def publish(workspace: Path, relative: str, data: bytes, *, append: bool = False) -> None:
    """Write using pinned directory descriptors, not Agent-controlled symlinks."""
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Unsafe publication path")
    with open_private_directory(workspace / path.parent) as parent:
        if append:
            descriptor = os.open(path.name, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW,
                                 0o600, dir_fd=parent)
            with os.fdopen(descriptor, "ab") as output:
                output.write(data)
            return
        temporary = ".runtime-" + uuid.uuid4().hex
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=parent)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(data)
            os.replace(temporary, path.name, src_dir_fd=parent, dst_dir_fd=parent)
        finally:
            try:
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass
