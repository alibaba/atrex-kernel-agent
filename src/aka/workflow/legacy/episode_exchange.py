"""Agent-writable episode messages, separate from private Supervisor state."""
from pathlib import Path

from aka.core.filesystem import durable_write_text
from .runtime_paths import EXCHANGE_DIR, RUNTIME_DIR

FILES = ("journal.json", "handoff.json", "telemetry.jsonl")


def prepare(workspace):
    workspace = Path(workspace).resolve(strict=True)
    directory = workspace / EXCHANGE_DIR
    if directory.is_symlink():
        raise ValueError("Episode exchange must not be a symlink")
    try:
        directory.mkdir(mode=0o700)
        created = True
    except FileExistsError:
        if not directory.is_dir():
            raise ValueError("Episode exchange must be a directory")
        created = False
    # Existing episodes retain their messages; private state is never copied out.
    for name in FILES:
        target = directory / name
        if target.is_symlink():
            raise ValueError("Episode exchange message must not be a symlink")
        source = workspace / RUNTIME_DIR / name
        if created and source.is_file():
            if source.is_symlink() or source.parent.is_symlink():
                raise ValueError("Legacy episode message must not be a symlink")
            durable_write_text(target, source.read_text(encoding="utf-8"))
    return directory
