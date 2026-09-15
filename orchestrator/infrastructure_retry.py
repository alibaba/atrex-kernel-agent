"""Wait for unavailable validation services without completing an episode."""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from concurrent.futures import CancelledError
from pathlib import Path
from threading import Event

from .durable_state import durable_write_json

POLL_SECONDS = 30 * 60
INFRASTRUCTURE_MARKER = "__ATREX_INFRASTRUCTURE_UNAVAILABLE__"
INFRASTRUCTURE_EXIT_CODE = 75


class InfrastructureUnavailable(RuntimeError):
    pass


def check_transport(process: subprocess.CompletedProcess) -> None:
    """Only classify transport failures; never infer an outage from a kernel failure."""
    if (process.returncode == INFRASTRUCTURE_EXIT_CODE
            and INFRASTRUCTURE_MARKER in (process.stderr or "").splitlines()):
        raise InfrastructureUnavailable("GPU validation transport unavailable")


def check_review_service(result) -> None:
    """Classify execution timeouts separately from explicit service outages."""
    if result.timed_out:
        raise TimeoutError("reviewer exceeded its configured execution timeout")
    if result.exit_status == 0:
        return
    for line in getattr(result, "stdout_tail", "").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("type") != "error":
            continue
        error = event.get("error")
        if isinstance(error, dict) and error.get("type") in {
            "overloaded_error", "rate_limit_error", "service_unavailable_error",
        }:
            raise InfrastructureUnavailable(f"review service unavailable: {error['type']}")
    raise ValueError(f"reviewer failed (exit={result.exit_status}); no service-outage evidence")


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
        except InfrastructureUnavailable as exc:
            now = time.time()
            reason = str(exc)
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
