"""Compatibility exports for the shared implementation."""
from orchestrator._bootstrap import task_modules
task_modules()
from aka.legacy.task.persistence import (
    fsync_directory,
    ensure_private_directory,
    durable_write_text,
    durable_write_json,
    durable_replace,
    durable_unlink,
    durable_rmdir,
)
