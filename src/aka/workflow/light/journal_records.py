"""Private Workflow JSON documents, separate from shared Experiment records."""
import json
from pathlib import Path

from aka.core.filesystem import durable_write_json
from aka.core.filesystem import open_private_directory, read_regular_bytes

MAX_RECORD_BYTES = 32 * 1024 * 1024


def read_json(path, *, limit=MAX_RECORD_BYTES):
    data = read_regular_bytes(path, limit=limit + 1)
    if len(data) > limit:
        raise ValueError("Workflow record exceeds its size limit")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("Workflow record must be an object")
    return value


def private_write(path, value):
    path = Path(path)
    with open_private_directory(path.parent):
        durable_write_json(path, value)
