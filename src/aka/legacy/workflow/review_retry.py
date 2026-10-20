"""Workflow review_retry implementation and callable interface."""
from __future__ import annotations

from aka.legacy.experiment.retry import retry_infrastructure
import hashlib
import json
import time
from pathlib import Path
from aka.legacy.task.durable import durable_write_json


def retry_review(workspace: Path, stage: str, operation):
    """Allow one fresh review after a timeout, durably bound to its evidence.

    The operation must create a fresh isolated session and restore its inputs.
    Explicit service outages retain their separate infrastructure retry policy.
    """
    key = hashlib.sha256(stage.encode()).hexdigest()[:24]
    path = workspace / ".atrex_long_horizon" / "review_timeouts" / f"{key}.json"
    record = json.loads(path.read_text()) if path.is_file() else {}
    timeout_count = record.get("timeout_count", 0)
    while timeout_count < 2:
        try:
            result = retry_infrastructure(workspace, stage, operation)
        except TimeoutError:
            timeout_count += 1
            durable_write_json(path, {
                "schema_version": 1, "stage": stage,
                "timeout_count": timeout_count,
                "last_timeout_at": time.time(),
            })
            if timeout_count < 2:
                print(f"[review] {stage}: execution timed out; retrying in a fresh session", flush=True)
        else:
            if timeout_count:
                durable_write_json(path, {
                    "schema_version": 1, "stage": stage, "timeout_count": 0,
                })
            return result
    raise ValueError(
        "reviewer infrastructure timed out twice for this evidence; "
        "validation blocked (increase the reviewer timeout to retry)"
    )
