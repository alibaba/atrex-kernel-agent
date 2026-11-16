"""Validate the Workflow-owned Episode active-time allowance."""
import math


def episode_timeout(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError("episode_timeout_seconds must be finite and greater than zero")
    return float(value)


def episode_budget(seconds, directory=None, *, resumed=False):
    from pathlib import Path
    from aka.workflow.budgets import durable_budget
    seconds = None if seconds is None else episode_timeout(seconds)
    path = None if directory is None else Path(directory) / "episode-deadline.json"
    return durable_budget(path, seconds, resumed=resumed)
