"""Private, crash-durable checkpoints for physical ABBA batches.

A completed batch is a measurement fact, including an explicit correctness failure.
Incomplete execution is not a measurement and must never enter this store.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orchestrator.durable_state import durable_write_json, ensure_private_directory
from supervisor.measurement_records import read_json

MAX_CHECKPOINT_BYTES = 32 * 1024 * 1024


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def validate_batch(
    payload: object, schedule: list[dict[str, int | str]], shape_ids: list[str]
) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema_version") != 1 or payload.get("error"):
        raise ValueError("ABBA batch has no complete measurement payload")
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
    def __init__(self, evidence_root: Path, history_roots: list[Path]) -> None:
        self.root = evidence_root / "abba-batches"
        self.roots = [self.root, *(root / "abba-batches" for root in reversed(history_roots))]

    def load(self, identity: dict[str, Any]) -> dict[str, Any] | None:
        key = _digest(identity)
        for root in self.roots:
            path = root / f"{key}.json"
            if root.is_symlink() or path.is_symlink():
                raise RuntimeError("ABBA batch checkpoint path is unsafe")
            if not path.exists():
                continue
            try:
                envelope = read_json(path, limit=MAX_CHECKPOINT_BYTES)
                value = envelope["record"]
                if envelope["content_digest"] != _digest(value) or value["identity"] != identity:
                    raise ValueError("checkpoint digest or request identity mismatch")
                if (
                    not isinstance(value["completed_at"], str)
                    or not isinstance(value["stdout"], str)
                    or not isinstance(value["stderr"], str)
                ):
                    raise ValueError("incomplete checkpoint")
                validate_batch(value["payload"], identity["schedule"], identity["shape_ids"])
                from long_horizon.verifier import parse_abba_payload

                if parse_abba_payload(value["stdout"]) != value["payload"]:
                    raise ValueError("checkpoint payload disagrees with captured execution output")
                return value
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise RuntimeError(f"ABBA batch checkpoint is invalid: {key}") from exc
        return None

    def save(
        self, identity: dict[str, Any], payload: dict[str, Any], *, stdout: str, stderr: str
    ) -> dict[str, Any]:
        validate_batch(payload, identity["schedule"], identity["shape_ids"])
        record = {
            "identity": identity,
            "payload": payload,
            "stdout": stdout,
            "stderr": stderr,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        envelope = {"record": record, "content_digest": _digest(record)}
        if len(json.dumps(envelope).encode()) > MAX_CHECKPOINT_BYTES:
            raise RuntimeError("ABBA batch checkpoint exceeds its byte limit")
        if self.root.is_symlink():
            raise RuntimeError("ABBA batch checkpoint directory is unsafe")
        ensure_private_directory(self.root)
        lock_path = self.root / ".lock"
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            existing = self.load(identity)
            if existing is not None:
                return existing
            durable_write_json(self.root / f"{_digest(identity)}.json", envelope)
            return record
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
