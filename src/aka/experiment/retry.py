"""Bounded Experiment retries for proven unsubmitted infrastructure failures."""
from __future__ import annotations

import fcntl
import time
from concurrent.futures import CancelledError
from threading import Event
from aka.execution.transport import InfrastructureUnavailable
import hashlib
import json
import math
from pathlib import Path
from aka.contracts.execution import JobResult
from aka.contracts.execution_recovery import InfrastructureRetryPolicy
from aka.core.filesystem import durable_write_json, ensure_private_directory


def _wait_until(deadline: float, cancel: Event) -> None:
    while (remaining := deadline - time.time()) > 0:
        if cancel.wait(min(60, remaining)):
            raise CancelledError("validation batch cancelled")


def _safe_to_retry(error):
    result = error.outcome
    return (isinstance(result, JobResult) and not result.files
        and not getattr(result, "job_references", ())
        and all(not getattr(attempt, "job_references", ()) for attempt in result.attempts)
        and _safe_facts(result.as_dict()))


def _safe_facts(result):
    return (isinstance(result, dict) and result.get("transport_status") == "infrastructure"
        and isinstance(result.get("facts"), dict) and result["facts"].get("disposition") == "not_submitted"
        and not result["facts"].get("cancellation_requested")
        and result.get("output_scope") == "transport" and result.get("exit_code") is None
        and not result.get("stdout") and not result.get("file_receipts")
        and not result.get("job_references")
        and isinstance(result.get("attempts"), (list, tuple))
        and all(isinstance(attempt, dict) and attempt.get("disposition") == "not_submitted"
                and attempt.get("exit_code") is None and not attempt.get("job_references")
                for attempt in result["attempts"]))


def _stop(message, record, outcome=None):
    error = InfrastructureUnavailable(message, outcome)
    error.retry_record = dict(record)
    return error


def _load(path, stage, policy):
    if not path.is_file():
        return {"schema_version": 2, "stage": stage, "status": "ready",
            "attempt_count": 0, "retry_count": 0,
            "policy": {"delays": list(policy.delays), "max_elapsed_s": policy.max_elapsed_s}}
    record = None
    try:
        record = json.loads(path.read_text())
        if not isinstance(record, dict) or record.get("stage") != stage:
            raise ValueError("retry record stage does not match")
        if record.get("schema_version") != 2:
            raise ValueError("legacy retry record has no safe replay proof")
        if record.get("policy") != {"delays": list(policy.delays), "max_elapsed_s": policy.max_elapsed_s}:
            raise ValueError("retry policy changed; existing attempt budget must be reconciled")
        for field in ("attempt_count", "retry_count"):
            if type(record.get(field)) is not int or record[field] < 0:
                raise ValueError("retry count is invalid")
        if record["retry_count"] != max(0, record["attempt_count"] - 1):
            raise ValueError("retry count does not match reserved attempts")
        if record.get("status") == "waiting_for_infrastructure":
            for field in ("first_failure_at", "deadline", "next_retry_at"):
                if type(record.get(field)) not in (int, float) or not math.isfinite(record[field]):
                    raise ValueError("retry timestamps are invalid")
            if (record.get("safe_to_retry") is not True or not _safe_facts(record.get("last_outcome"))
                    or record["attempt_count"] < 1
                    or record["deadline"] != record["first_failure_at"] + policy.max_elapsed_s
                    or record["next_retry_at"] < record["first_failure_at"]):
                raise ValueError("retry record has no safe bounded continuation")
        elif record.get("status") != "ready" or record["attempt_count"]:
            raise ValueError("prior step requires host reconciliation before another attempt")
        return record
    except (ValueError, TypeError, KeyError) as error:
        retained = dict(record) if isinstance(record, dict) else {"stage": stage}
        retained["record_path"] = str(path)
        raise _stop("infrastructure retry record requires reconciliation: " + str(error),
            retained) from error


def retry_infrastructure(workspace: Path, stage: str, operation, *, cancel=None, policy=None):
    """Use the bounded default schedule only when Execution proves safe admission.

    An empty schedule dispatches once without a retry ledger; Experiment owns
    the attempt record and any decision to call again. For configured retries,
    stage identifies one experiment step and its immutable inputs. A crash during
    an attempt, an old record, or a changed policy needs host reconciliation;
    this helper never queries or replaces an existing Job. Elapsed time starts
    at the first infrastructure failure and includes waiting across restarts.
    """
    policy = InfrastructureRetryPolicy() if policy is None else policy
    if not isinstance(policy, InfrastructureRetryPolicy):
        raise TypeError("policy must be an InfrastructureRetryPolicy")
    cancel = cancel if cancel is not None else Event()
    if not policy.delays:
        if cancel.is_set():
            raise CancelledError("validation batch cancelled")
        return operation()
    key = hashlib.sha256(stage.encode()).hexdigest()[:24]
    path = Path(workspace) / ".atrex_long_horizon" / "infrastructure" / f"{key}.json"
    ensure_private_directory(path.parent)
    with path.with_suffix(".lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise _stop("infrastructure retry step is already active", {"stage": stage}) from error
        record = _load(path, stage, policy)
        try:
            return _run(path, record, policy, operation, cancel)
        except CancelledError:
            record.update(status="cancelled", cancelled_at=time.time())
            durable_write_json(path, record, indent=2, ensure_ascii=False)
            raise


def _run(path, record, policy, operation, cancel):
    while True:
        if cancel.is_set():
            raise CancelledError("validation batch cancelled")
        if record["status"] == "waiting_for_infrastructure":
            if (record["attempt_count"] > len(policy.delays)
                    or time.time() >= record["deadline"]
                    or record["next_retry_at"] >= record["deadline"]):
                record.update(status="retry_exhausted", finished_at=time.time())
                durable_write_json(path, record, indent=2, ensure_ascii=False)
                raise _stop("infrastructure retry budget exhausted", record)
            _wait_until(record["next_retry_at"], cancel)
            if time.time() >= record["deadline"]:
                record.update(status="retry_exhausted", finished_at=time.time())
                durable_write_json(path, record, indent=2, ensure_ascii=False)
                raise _stop("infrastructure retry elapsed budget exhausted", record)
        if cancel.is_set():
            raise CancelledError("validation batch cancelled")
        record.update(status="attempting", attempt_count=record["attempt_count"] + 1,
            attempt_started_at=time.time())
        record["retry_count"] = record["attempt_count"] - 1
        durable_write_json(path, record, indent=2, ensure_ascii=False)
        try:
            result = operation()
        except InfrastructureUnavailable as error:
            now = time.time()
            record.setdefault("first_failure_at", now)
            record.setdefault("deadline", record["first_failure_at"] + policy.max_elapsed_s)
            safe = _safe_to_retry(error)
            record.update(last_failure_at=now, reason=str(error), safe_to_retry=safe)
            if isinstance(error.outcome, JobResult):
                outcome = error.outcome.as_dict()
                outcome.pop("stdout", None)
                outcome.pop("stderr", None)
                record["last_outcome"] = outcome
            exhausted = record["attempt_count"] > len(policy.delays) or now >= record["deadline"]
            if not safe or exhausted:
                record.update(status="unsafe_failure" if not safe else "retry_exhausted", finished_at=now)
                durable_write_json(path, record, indent=2, ensure_ascii=False)
                error.retry_record = dict(record)
                raise
            delay = policy.delays[record["attempt_count"] - 1]
            record.update(status="waiting_for_infrastructure", next_retry_at=now + delay)
            durable_write_json(path, record, indent=2, ensure_ascii=False)
            print(f"[infrastructure] {record['stage']}: retry {record['attempt_count']}/{len(policy.delays)} "
                f"in {delay:g} seconds; deadline={record['deadline']}; state={path}", flush=True)
        except CancelledError:
            raise
        except BaseException:
            record.update(status="step_failed", finished_at=time.time())
            durable_write_json(path, record, indent=2, ensure_ascii=False)
            raise
        else:
            record.update(status="recovered", recovered_at=time.time())
            durable_write_json(path, record, indent=2, ensure_ascii=False)
            return result
