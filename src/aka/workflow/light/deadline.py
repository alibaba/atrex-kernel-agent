"""Validate the Workflow-owned Episode deadline configuration."""
import math


def episode_timeout(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError("episode_timeout_seconds must be finite and greater than zero")
    return float(value)


def episode_deadline(seconds, directory=None, *, resumed=False):
    """Translate a durable wall-clock deadline to this process's monotonic clock."""
    import json
    import time
    from pathlib import Path
    from aka.core.filesystem import durable_write_json, read_regular_bytes

    seconds = episode_timeout(seconds)
    now = time.time()
    if directory is None:
        if resumed:
            raise ValueError("resumed Episode requires a durable deadline directory")
        return time.monotonic() + seconds
    path = Path(directory) / "episode-deadline.json"
    try:
        value = json.loads(read_regular_bytes(path, limit=4096))
    except FileNotFoundError:
        if resumed:
            raise ValueError("resumed Episode has no frozen deadline; restore its budget before continuing")
        value = {"schema": 1, "policy": "wall-clock", "seconds": seconds,
                 "started_at": now, "expires_at": now + seconds}
        durable_write_json(path, value)
    if (not isinstance(value, dict) or type(value.get("schema")) is not int or value["schema"] != 1
            or value.get("policy") != "wall-clock" or value.get("seconds") != seconds
            or any(type(value.get(key)) not in (int, float) or not math.isfinite(value[key])
                   for key in ("started_at", "expires_at"))
            or value["expires_at"] != value["started_at"] + seconds):
        raise ValueError("Episode deadline is invalid or its frozen budget changed")
    if now < value["started_at"]:
        raise ValueError("wall clock moved before Episode start; reconcile its budget")
    return time.monotonic() + max(0.0, value["expires_at"] - now)
