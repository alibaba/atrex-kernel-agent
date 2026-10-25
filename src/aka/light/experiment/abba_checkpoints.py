"""Private, crash-durable checkpoints for completed Shape batches.

A completed batch is a measurement fact, including an explicit correctness
failure. Incomplete execution is not a measurement and must never enter this
store. Both ordinary Evaluate and ABBA use the same envelope and Record-evidence
rules; their payload validators remain operation-specific.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aka.legacy.task.persistence import durable_write_json, ensure_private_directory
from aka.light.experiment.store import private_write, read_json
from aka.light.evaluation.evaluation_failures import embedded_device_failure

MAX_CHECKPOINT_BYTES = 32 * 1024 * 1024


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _quarantine_invalid_checkpoint(
    root: Path, identity: dict[str, Any], label: str, error: BaseException
) -> None:
    """Preserve corrupt private evidence while allowing the batch to rerun."""
    key = _digest(identity)
    path = root / f"{key}.json"
    if root.is_symlink() or path.is_symlink():
        raise RuntimeError(f"{label} batch checkpoint path is unsafe") from error
    if not path.exists():
        return
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    quarantine = root / f"{key}.invalid-{suffix}.json"
    try:
        os.replace(path, quarantine)
    except FileNotFoundError:
        return  # Another worker already quarantined the same checkpoint.
    except OSError as exc:
        raise RuntimeError(
            f"{label} batch checkpoint is invalid and could not be quarantined: {key}"
        ) from exc
    logging.getLogger(__name__).warning(
        "%s batch checkpoint %s was invalid and moved to %s; the batch will be rerun",
        label,
        key,
        quarantine.name,
    )


def _load_checkpoint(root: Path, identity: dict[str, Any], label: str) -> dict[str, Any] | None:
    key = _digest(identity)
    path = root / f"{key}.json"
    if root.is_symlink() or path.is_symlink():
        raise RuntimeError(f"{label} batch checkpoint path is unsafe")
    if not path.exists():
        return None
    try:
        envelope = read_json(path, limit=MAX_CHECKPOINT_BYTES)
        value = envelope["record"]
        if envelope["content_digest"] != _digest(value) or value["identity"] != identity:
            raise ValueError("checkpoint digest or request identity mismatch")
        if (
            not isinstance(value["completed_at"], str)
            or not isinstance(value["stdout"], str)
            or not isinstance(value["stderr"], str)
            or isinstance(value.get("returncode", 0), bool)
            or not isinstance(value.get("returncode", 0), int)
        ):
            raise ValueError("incomplete checkpoint")
        value.setdefault("returncode", 0)  # Read checkpoints written before this field existed.
        return value
    except (OSError, ValueError, KeyError, TypeError) as exc:
        _quarantine_invalid_checkpoint(root, identity, label, exc)
        return None


def _save_checkpoint(
    root: Path,
    identity: dict[str, Any],
    payload: dict[str, Any],
    *,
    stdout: str,
    stderr: str,
    returncode: int,
    label: str,
    load,
) -> dict[str, Any]:
    if (
        not isinstance(payload, dict)
        or not isinstance(stdout, str)
        or not isinstance(stderr, str)
        or isinstance(returncode, bool)
        or not isinstance(returncode, int)
    ):
        raise ValueError(f"{label} batch checkpoint fields are invalid")
    record = {
        "identity": identity,
        "payload": payload,
        "stdout": stdout,
        "stderr": stderr,
        "returncode": returncode,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    envelope = {"record": record, "content_digest": _digest(record)}
    if len(json.dumps(envelope).encode()) > MAX_CHECKPOINT_BYTES:
        raise RuntimeError(f"{label} batch checkpoint exceeds its byte limit")
    if root.is_symlink():
        raise RuntimeError(f"{label} batch checkpoint directory is unsafe")
    ensure_private_directory(root)
    lock_path = root / ".lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        existing = load(identity)
        if existing is not None:
            return existing
        durable_write_json(root / f"{_digest(identity)}.json", envelope)
        return record
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _retain_terminal_job(
    evidence_root: Path,
    namespace: str,
    identity: dict[str, Any],
    record: dict[str, Any],
) -> None:
    """Copy reusable evidence into a replacement Gateway Record.

    Record validation intentionally reads terminal physical-job states rather
    than trusting the batch index. Materialize a bounded, content-bound state
    from the captured raw job response so a Record made entirely from reused
    batches remains independently auditable and cacheable.
    """
    request = identity.get("request")
    if not isinstance(request, dict):
        raise ValueError("Shape batch identity has no Eval request")
    job_identity = {
        "kind": "eval",
        "payload": request,
        "batch_checkpoint": namespace,
        "batch_checkpoint_digest": _digest(identity),
    }
    process = subprocess.CompletedProcess(
        [], record["returncode"], record["stdout"], record["stderr"]
    )
    from aka.light.execution.gateway_jobs import cacheable_job, job_from_process

    if not cacheable_job(job_from_process(process), job_identity):
        raise ValueError("Shape batch is not complete terminal Eval evidence")
    state = {
        "phase": "terminal",
        "submission": 0,
        "retries": {},
        "identity": job_identity,
        "process": {
            "returncode": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
        },
    }
    path = evidence_root / "jobs" / _digest(job_identity) / "state.json"
    try:
        existing = read_json(path)
    except FileNotFoundError:
        private_write(path, state)
    else:
        if existing != state:
            raise RuntimeError("Retained Shape batch evidence is inconsistent")


def validate_batch(
    payload: object, schedule: list[dict[str, int | str]], shape_ids: list[str]
) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema_version") != 1 or payload.get("error"):
        raise ValueError("ABBA batch has no complete measurement payload")
    if embedded_device_failure({"status": "succeeded", "result": payload}):
        raise ValueError("ABBA device unavailability is not a Kernel verdict")
    runs = payload.get("runs")
    if not isinstance(runs, list) or any(not isinstance(row, dict) for row in runs):
        raise ValueError("ABBA batch has invalid measurement rows")
    if [{"revision": row.get("revision"), "repeat": row.get("repeat")} for row in runs] != schedule:
        raise ValueError("ABBA batch did not execute the exact requested schedule")
    for row in runs:
        result = row.get("result")
        if not isinstance(result, dict) or not isinstance(result.get("all_pass"), bool):
            raise ValueError("ABBA batch has an unfinished evaluator result")
        code = row.get("exit_code")
        if isinstance(code, bool) or not isinstance(code, int) or code < 0:
            raise ValueError("ABBA batch evaluator was interrupted")
        if result["all_pass"] is False:
            # No latency is required to preserve a completed negative correctness result.
            continue
        values = result.get("latency_us_by_shape")
        if code != 0 or not isinstance(values, dict) or set(values) != set(shape_ids):
            raise ValueError("ABBA batch has incomplete or inconsistent Shape coverage")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in [*values.values(), result.get("latency_us_geomean")]
        ):
            raise ValueError("ABBA batch has invalid Shape latency")
    return payload


class AbbaBatchStore:
    """Store completed Shape batches under a caller-selected recovery scope.

    Recorded ABBA uses a repetition-scoped directory in the Campaign
    Measurement Store as ``evidence_root``.  Checkpoints therefore survive a
    replacement Gateway Record without collapsing independent measurement
    repetitions.  Standalone CLI execution may still use its physical-job
    root.  In either case the content-derived identity below is the authority
    for reuse.
    """
    def __init__(self, evidence_root: Path) -> None:
        self.evidence_root = evidence_root
        self.root = evidence_root / "abba-batches"

    def load(self, identity: dict[str, Any]) -> dict[str, Any] | None:
        value = _load_checkpoint(self.root, identity, "ABBA")
        if value is None:
            return None
        try:
            if embedded_device_failure({"status": "succeeded", "result": value["payload"]}):
                return None  # Older releases sealed device outages as negative evidence.
            validate_batch(value["payload"], identity["schedule"], identity["shape_ids"])
            from aka.light.evaluation.native_abba import parse_native_abba

            if parse_native_abba(json.loads(value["stdout"]), len(identity["schedule"]) // 2,
                                 identity["shape_ids"]) != value["payload"]:
                raise ValueError("checkpoint payload disagrees with captured execution output")
            return value
        except (OSError, ValueError, KeyError, TypeError) as exc:
            _quarantine_invalid_checkpoint(self.root, identity, "ABBA", exc)
            return None

    def save(
        self,
        identity: dict[str, Any],
        payload: dict[str, Any],
        *,
        stdout: str,
        stderr: str,
        returncode: int = 0,
    ) -> dict[str, Any]:
        validate_batch(payload, identity["schedule"], identity["shape_ids"])
        return _save_checkpoint(
            self.root,
            identity,
            payload,
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            label="ABBA",
            load=self.load,
        )

    def retain_terminal_job(self, identity: dict[str, Any], record: dict[str, Any]) -> None:
        _retain_terminal_job(self.evidence_root, "abba", identity, record)


class EvalBatchStore:
    """Store complete ordinary-Eval Shape batches under a recovery scope."""

    def __init__(self, evidence_root: Path) -> None:
        self.evidence_root = evidence_root
        self.root = evidence_root / "eval-batches"

    @staticmethod
    def _validate(identity: dict[str, Any], record: dict[str, Any]) -> None:
        request = identity.get("request")
        shape_ids = identity.get("shape_ids")
        reference = request.get("reference") if isinstance(request, dict) else None
        if (
            identity.get("transport") != "agate_native_eval"
            or not isinstance(request, dict)
            or not isinstance(shape_ids, list)
            or not shape_ids
            or any(not isinstance(shape_id, str) or not shape_id for shape_id in shape_ids)
            or len(shape_ids) != len(set(shape_ids))
            or not isinstance(reference, dict)
        ):
            raise ValueError("Eval batch has an invalid request identity")
        shapes = reference.get("shapes")
        if not isinstance(shapes, dict) or list(shapes) != shape_ids:
            raise ValueError("Eval batch request has inconsistent Shape coverage")
        try:
            job = json.loads(record["stdout"])
        except json.JSONDecodeError as exc:
            raise ValueError("Eval batch raw response is not JSON") from exc
        if job != record["payload"]:
            raise ValueError("Eval batch payload disagrees with captured execution output")
        from aka.light.execution.gateway_jobs import cacheable_job

        if not cacheable_job(job, {"kind": "eval", "payload": request}):
            raise ValueError("Eval batch has no complete terminal measurement")

    def load(self, identity: dict[str, Any]) -> dict[str, Any] | None:
        record = _load_checkpoint(self.root, identity, "Eval")
        if record is None:
            return None
        try:
            self._validate(identity, record)
            return record
        except (OSError, ValueError, KeyError, TypeError) as exc:
            _quarantine_invalid_checkpoint(self.root, identity, "Eval", exc)
            return None

    def save(
        self,
        identity: dict[str, Any],
        job: dict[str, Any],
        *,
        stdout: str,
        stderr: str,
        returncode: int,
    ) -> dict[str, Any]:
        candidate = {
            "identity": identity,
            "payload": job,
            "stdout": stdout,
            "stderr": stderr,
            "returncode": returncode,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        self._validate(identity, candidate)
        return _save_checkpoint(
            self.root,
            identity,
            job,
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            label="Eval",
            load=self.load,
        )

    def retain_terminal_job(self, identity: dict[str, Any], record: dict[str, Any]) -> None:
        self._validate(identity, record)
        _retain_terminal_job(self.evidence_root, "eval", identity, record)
