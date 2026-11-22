"""Describe the persistent workspace; its Git history belongs to the Agent."""
from pathlib import Path
from aka.contracts.filesystem import FilesystemAccess


def filesystem_access(workspace, *, include_git=False):
    workspace = Path(workspace).resolve(strict=True)
    metadata = workspace / ".git"
    if metadata.is_symlink() or metadata.is_file():
        raise ValueError("Light requires a standalone Agent Git repository")
    return FilesystemAccess(hidden_paths=() if include_git else (metadata,))
