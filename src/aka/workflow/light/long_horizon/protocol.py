from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .models import EpisodeHandoff, TERMINAL_STATUSES


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def normalize_handoff(value: object) -> EpisodeHandoff | None:
    if not isinstance(value, dict):
        return None
    status = value.get("status")
    if status not in TERMINAL_STATUSES:
        return None
    candidate = value.get("candidate_commit", "")
    if status == "candidate_ready" and (not isinstance(candidate, str) or not candidate.strip()):
        return None
    trial = value.get("last_trial_commit", "")
    return EpisodeHandoff(
        status=str(status),
        candidate_commit=candidate.strip() if isinstance(candidate, str) else "",
        last_trial_commit=trial.strip() if isinstance(trial, str) else "",
    )


def read_handoff(path: Path) -> EpisodeHandoff | None:
    try:
        return normalize_handoff(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def handoff_diagnosis(path: Path) -> str:
    if not path.exists():
        return "handoff file is missing"
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return f"handoff cannot be read: {type(exc).__name__}"
    if not text.strip():
        return "handoff file is empty"
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        return f"handoff is invalid JSON at line {exc.lineno}, column {exc.colno}"
    if not isinstance(value, dict):
        return "handoff must be a JSON object"
    if value.get("status") not in TERMINAL_STATUSES:
        return "handoff status must be candidate_ready, pivot, or blocked"
    if value.get("status") == "candidate_ready" and not str(value.get("candidate_commit", "")).strip():
        return "candidate_ready requires candidate_commit"
    return "handoff schema is valid but its episode completion contract is not satisfied"
