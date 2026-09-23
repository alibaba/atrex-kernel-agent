"""Wait for unavailable validation services without completing an episode."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from concurrent.futures import CancelledError
from pathlib import Path
from threading import Event

from .durable_state import durable_write_json
from .session_tail import read_regular_bytes

POLL_SECONDS = 30 * 60
MAX_STATE_BYTES = 64 * 1024


def _state_path(workspace: Path, category: str, key: str) -> Path:
    # Retry budgets are controller authority, not Agent-writable checkpoints.
    from .supervisor_runtime import supervisor_campaign_root
    return supervisor_campaign_root(workspace) / category / f"{key}.json"


class InfrastructureUnavailable(RuntimeError):
    pass


class ReviewerExecutionTimeout(RuntimeError):
    """A completed review session explicitly exceeded its execution deadline."""


class RetryStateError(ValueError):
    """Unverifiable private retry state; never silently reset a durable budget."""


def _load_retry_state(path: Path, stage: str, category: str) -> dict:
    try:
        raw = read_regular_bytes(path, limit=MAX_STATE_BYTES + 1)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        error = exc
    else:
        try:
            if len(raw) > MAX_STATE_BYTES:
                raise ValueError("state exceeds size limit")
            record = json.loads(raw)
            if (not isinstance(record, dict) or type(record.get("schema_version")) is not int
                    or record["schema_version"] != 1 or record.get("stage") != stage):
                raise ValueError("invalid state schema or stage identity")
            count_key = "timeout_count" if category == "review_timeouts" else "retry_count"
            count = record.get(count_key)
            if type(count) is not int or count < 0 or (count_key == "timeout_count" and count > 2):
                raise ValueError(f"invalid {count_key}")
            if category == "infrastructure":
                if record.get("status") not in ("waiting_for_infrastructure", "recovered", "step_failed"):
                    raise ValueError("invalid infrastructure status")
                if record["status"] == "waiting_for_infrastructure" and "next_retry_at" not in record:
                    raise ValueError("missing infrastructure retry deadline")
            for field in ("next_retry_at", "last_failure_at", "last_timeout_at", "recovered_at", "finished_at"):
                if field in record and (type(record[field]) not in (int, float)
                        or not math.isfinite(record[field]) or record[field] < 0):
                    raise ValueError(f"invalid {field}")
            return record
        except (ValueError, UnicodeError, OverflowError, RecursionError) as exc:
            error = exc
    logging.getLogger(__name__).error("Cannot read trusted %s retry state %s: %s", category, path, error)
    raise RetryStateError(
        f"Cannot verify {category} retry state; validation is blocked before another review. "
        "Ask the operator to inspect the state file identified in the Supervisor log and "
        "restore a valid same-stage backup or repair access, then resume. "
        "The state was not overwritten; do not delete it or reset the retry count."
    ) from error


def check_review_service(result) -> None:
    """Classify execution timeouts separately from explicit service outages."""
    if result.timed_out:
        raise ReviewerExecutionTimeout("reviewer exceeded its configured execution timeout")
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
    path = _state_path(workspace, "infrastructure", key)
    record = _load_retry_state(path, stage, "infrastructure")
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
        except ReviewerExecutionTimeout:
            # retry_review owns this expected outcome and its durable budget.
            # Preserve any prior outage record; this is not a failed service step.
            raise
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
    """Allow one fresh review after a classified execution timeout, durably bound to its evidence.

    The operation must create a fresh isolated session and restore its inputs.
    Only ReviewerExecutionTimeout consumes this budget; OS/socket timeouts propagate.
    Explicit service outages retain their separate infrastructure retry policy.
    """
    key = hashlib.sha256(stage.encode()).hexdigest()[:24]
    path = _state_path(workspace, "review_timeouts", key)
    record = _load_retry_state(path, stage, "review_timeouts")
    timeout_count = record.get("timeout_count", 0)
    while timeout_count < 2:
        try:
            result = retry_infrastructure(workspace, stage, operation)
        except ReviewerExecutionTimeout:
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
        "reviewer execution timed out twice for this evidence; "
        "validation blocked (increase the reviewer timeout to retry)"
    )
