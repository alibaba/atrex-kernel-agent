from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import TERMINAL_STATUSES
from .protocol import atomic_write_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def initialize(path: Path, *, episode: int, base_commit: str, branch: str) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "episode": episode,
        "base_commit": base_commit,
        "episode_branch": branch,
        "state": "in_progress",
        "experiments": [],
        "outcome": None,
        "created_at": utc_now(),
        "finalized_at": None,
    }
    atomic_write_json(path, value)
    return value


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("unsupported episode journal")
    return value


def append_experiment(path: Path, experiment: dict[str, Any]) -> dict[str, Any]:
    value = load(path)
    if value.get("state") != "in_progress":
        raise ValueError("cannot append to a finalized episode journal")
    if not isinstance(experiment, dict) or not experiment:
        raise ValueError("experiment must be a non-empty JSON object")
    entry = dict(experiment)
    entry.setdefault("timestamp", utc_now())
    experiments = value.setdefault("experiments", [])
    if not isinstance(experiments, list):
        raise ValueError("journal experiments must be a list")
    experiments.append(entry)
    atomic_write_json(path, value)
    return value


def finalize(
    path: Path,
    *,
    state: str,
    outcome: dict[str, Any],
    candidate_commit: str = "",
) -> dict[str, Any]:
    value = load(path)
    if state not in TERMINAL_STATUSES:
        raise ValueError("state must be candidate_ready, pivot, or blocked")
    if value.get("state") not in {"in_progress", state}:
        raise ValueError(f"cannot change finalized state to {state}")
    if not isinstance(outcome, dict) or not str(outcome.get("summary", "")).strip():
        raise ValueError("outcome.summary must be non-empty")
    directions = outcome.get("next_directions", [])
    if not isinstance(directions, list) or any(not isinstance(item, str) for item in directions):
        raise ValueError("outcome.next_directions must be a list of strings")
    experiments = value.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("a terminal journal requires at least one experiment")
    if state == "candidate_ready" and not candidate_commit.strip():
        raise ValueError("candidate_ready requires candidate_commit")
    value["state"] = state
    value["outcome"] = outcome
    value["candidate_commit"] = candidate_commit.strip() or None
    value["finalized_at"] = utc_now()
    atomic_write_json(path, value)
    return value


def validate_terminal(
    path: Path,
    *,
    expected_episode: int,
    base_commit: str,
    branch: str,
    state: str,
    candidate_commit: str = "",
) -> str:
    try:
        value = load(path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return f"episode journal is missing or invalid: {exc}"
    if value.get("episode") != expected_episode:
        return "episode journal identity does not match the active episode"
    if value.get("base_commit") != base_commit or value.get("episode_branch") != branch:
        return "episode journal base commit or branch does not match"
    if value.get("state") != state:
        return "episode journal state does not match handoff"
    experiments = value.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        return "episode journal has no structured experiments"
    outcome = value.get("outcome")
    if not isinstance(outcome, dict) or not str(outcome.get("summary", "")).strip():
        return "episode journal has no terminal outcome summary"
    directions = outcome.get("next_directions", [])
    if not isinstance(directions, list) or any(not isinstance(item, str) for item in directions):
        return "episode journal next_directions is invalid"
    if not value.get("finalized_at"):
        return "episode journal is not finalized"
    if state == "candidate_ready" and value.get("candidate_commit") != candidate_commit:
        return "episode journal candidate_commit does not match handoff"
    return ""


def _json_object(raw: str, label: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage one long-horizon episode journal.")
    sub = parser.add_subparsers(dest="command", required=True)
    append = sub.add_parser("append")
    append.add_argument("--path", required=True)
    append.add_argument("--experiment-json", required=True)
    finish = sub.add_parser("finalize")
    finish.add_argument("--path", required=True)
    finish.add_argument("--state", choices=sorted(TERMINAL_STATUSES), required=True)
    finish.add_argument("--outcome-json", required=True)
    finish.add_argument("--candidate-commit", default="")
    args = parser.parse_args(argv)
    try:
        if args.command == "append":
            value = append_experiment(
                Path(args.path), _json_object(args.experiment_json, "--experiment-json")
            )
        else:
            value = finalize(
                Path(args.path),
                state=args.state,
                outcome=_json_object(args.outcome_json, "--outcome-json"),
                candidate_commit=args.candidate_commit,
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(value, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
