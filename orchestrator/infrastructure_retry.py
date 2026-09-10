"""Wait for unavailable validation services without completing an episode."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from concurrent.futures import CancelledError
from pathlib import Path
from threading import Event

from .durable_state import durable_write_json

POLL_SECONDS = 30 * 60
INFRASTRUCTURE_MARKER = "__ATREX_INFRASTRUCTURE_UNAVAILABLE__"


class InfrastructureUnavailable(RuntimeError):
    pass


def check_transport(process: subprocess.CompletedProcess) -> None:
    """Only classify transport failures; never infer an outage from a kernel failure."""
    output = (process.stdout or "") + "\n" + (process.stderr or "")
    if any(marker in output for marker in (
        INFRASTRUCTURE_MARKER,
        "generalized gateway response unavailable",
        "did not contain an artifact frame",
        "Connection refused", "Connection reset by peer",
        "Temporary failure in name resolution",
    )) or re.search(r"HTTP(?: Error)?\s*[:=]?\s*(429|502|503|504)\b", output):
        raise InfrastructureUnavailable("GPU validation transport unavailable")


def _wait_until(deadline: float, cancel: Event) -> None:
    while (remaining := deadline - time.time()) > 0:
        if cancel.wait(min(60, remaining)):
            raise CancelledError("validation batch cancelled")


def retry_infrastructure(workspace: Path, stage: str, operation, *, cancel=None):
    """Retry the same step every 30 minutes, retaining its deadline across restarts.

    Stage identity must include the candidate/contract digest. This loop does not
    update campaign counters, finalize a journal, or start another coding session.
    """
    cancel = cancel if cancel is not None else Event()
    key = hashlib.sha256(stage.encode()).hexdigest()[:24]
    path = workspace / ".atrex_long_horizon" / "infrastructure" / f"{key}.json"
    record = json.loads(path.read_text()) if path.is_file() else {}
    while True:
        if record.get("status") == "waiting_for_infrastructure":
            _wait_until(record["next_retry_at"], cancel)
        if cancel.is_set():
            raise CancelledError("validation batch cancelled")
        try:
            result = operation()
        except (InfrastructureUnavailable, subprocess.TimeoutExpired, ConnectionError) as exc:
            now = time.time()
            reason = str(exc) if isinstance(exc, InfrastructureUnavailable) else type(exc).__name__
            record = {
                "schema_version": 1, "stage": stage,
                "status": "waiting_for_infrastructure",
                "retry_count": record.get("retry_count", 0) + 1,
                "last_failure_at": now, "next_retry_at": now + POLL_SECONDS,
                "poll_seconds": POLL_SECONDS, "reason": reason,
            }
            durable_write_json(path, record, indent=2, ensure_ascii=False)
            print(f"[infrastructure] {stage}: {reason}; retry in 30 minutes; "
                  f"candidate and episode preserved; state={path}", flush=True)
        except Exception:
            if record:
                record.update(status="step_failed", finished_at=time.time())
                durable_write_json(path, record, indent=2, ensure_ascii=False)
            raise
        else:
            if record:
                record.update(status="recovered", recovered_at=time.time())
                durable_write_json(path, record, indent=2, ensure_ascii=False)
                print(f"[infrastructure] {stage}: recovered; continuing validation", flush=True)
            return result
