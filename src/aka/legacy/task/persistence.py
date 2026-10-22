"""Retained persistence import path for package-local filesystem mechanics."""
from aka.legacy._filesystem import (
    fsync_directory, ensure_private_directory, durable_write_text, durable_write_json,
    durable_replace, durable_unlink, durable_rmdir,
)
