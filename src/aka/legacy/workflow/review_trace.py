"""Durable, private reviewer-session evidence with bounded retention modes."""

from __future__ import annotations

import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from aka.legacy.task.candidate_workspace.excludes import VERIFY_DIR
from aka.legacy.task.durable import durable_write_json


REVIEW_TRACE_MODES = frozenset({"off", "errors", "all"})


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class ReviewTrace:
    """Track one reviewer attempt without retaining its trusted input tree."""

    def __init__(self, workspace: Path, role: str, mode: str):
        if mode not in REVIEW_TRACE_MODES:
            raise ValueError("review trace mode must be one of: off, errors, all")
        if not role or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
            for character in role
        ):
            raise ValueError("review trace role must be lowercase alphanumeric/hyphen")
        self.workspace = Path(workspace)
        self.role = role
        self.mode = mode
        self.attempt_id = uuid.uuid4().hex
        root = self.workspace / VERIFY_DIR
        self.receipt_path = root / "review-attempts" / f"{role}-{self.attempt_id}.json"
        self.observation_directory = (
            None
            if mode == "off"
            else root / "private-review-observations" / role / self.attempt_id
        )
        if self.observation_directory is not None:
            self.observation_directory.mkdir(parents=True, exist_ok=False)
        self._receipt = {
            "schema_version": 1,
            "attempt_id": self.attempt_id,
            "role": role,
            "trace_mode": mode,
            "state": "running",
            "started_at": _timestamp(),
            "trace_retained": False,
        }
        durable_write_json(self.receipt_path, self._receipt, indent=2)

    def record_session(self, result) -> None:
        if self.observation_directory is not None:
            durable_write_json(
                self.observation_directory / "session-result.json",
                {
                    "schema_version": 1,
                    "session_id": str(result.session_id or ""),
                    "exit_status": int(result.exit_status),
                    "timed_out": bool(result.timed_out),
                    "tokens": int(getattr(result, "tokens", 0) or 0),
                    "stdout_tail": str(result.stdout_tail or ""),
                    "stderr_tail": str(result.stderr_tail or ""),
                    "observation_errors": list(result.observation_errors),
                },
                indent=2,
                ensure_ascii=False,
            )
        self._receipt.update(
            state="session_finished",
            session_id=str(result.session_id or ""),
            exit_status=int(result.exit_status),
            timed_out=bool(result.timed_out),
            observation_errors=list(result.observation_errors),
        )
        durable_write_json(self.receipt_path, self._receipt, indent=2)

    def write_private_json(self, name: str, value) -> None:
        if self.observation_directory is None:
            return
        if not name.endswith(".json") or "/" in name or "\\" in name:
            raise ValueError("private review trace name must be one JSON filename")
        durable_write_json(
            self.observation_directory / name,
            value,
            indent=2,
            ensure_ascii=False,
        )

    def finish(self, outcome: str, *, failed: bool, **facts) -> bool:
        observation_exists = (
            self.observation_directory is not None
            and self.observation_directory.is_dir()
        )
        retain = observation_exists and (
            self.mode == "all" or (self.mode == "errors" and failed)
        )
        if observation_exists and not retain:
            shutil.rmtree(self.observation_directory)
        self._receipt.update(
            state="finished",
            finished_at=_timestamp(),
            outcome=outcome,
            trace_retained=retain,
            **facts,
        )
        durable_write_json(self.receipt_path, self._receipt, indent=2)
        return retain
