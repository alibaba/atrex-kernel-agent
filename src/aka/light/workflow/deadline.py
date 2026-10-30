"""Validate the Workflow-owned Episode deadline configuration."""
import math


def episode_timeout(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError("episode_timeout_seconds must be finite and greater than zero")
    return float(value)
