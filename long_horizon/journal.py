from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import TERMINAL_STATUSES
from .protocol import atomic_write_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def infer_live_memory_path(path: Path) -> Path | None:
    """Resolve the incumbent workspace from a regular or linked Git worktree."""
    worktree = path.resolve().parent.parent
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=str(worktree), capture_output=True, text=True,
        )
    except OSError:
        return None
    if result.returncode:
        return None
    common_dir = Path(result.stdout.strip())
    if not common_dir.is_absolute():
        common_dir = (worktree / common_dir).resolve()
    if common_dir.name != ".git":
        return None
    return common_dir.parent / "memory" / "live.json"


def sync_live_memory(
    path: Path,
    value: dict[str, Any],
    *,
    phase: str = "",
    canonical_memory: str = "",
    accepted: bool | None = None,
    memory_version: int | None = None,
    episode: int | None = None,
) -> dict[str, Any]:
    """Write a non-canonical progress view without changing version history."""
    experiments = value.get("experiments")
    if not isinstance(experiments, list):
        experiments = []
    journal_state = str(value.get("state", "in_progress"))
    effective_phase = phase or (
        "exploring" if journal_state == "in_progress" else "awaiting_verification"
    )
    raw_version = value.get("memory_version") if memory_version is None else memory_version
    version = (
        int(raw_version)
        if isinstance(raw_version, int) and not isinstance(raw_version, bool)
        else None
    )
    raw_episode = value.get("episode") if episode is None else episode
    live = {
        "schema_version": "atrex_long_horizon_live_v1",
        "canonical": False,
        "canonical_memory_recorded": effective_phase == "recorded",
        "note": (
            "Live optimization progress only; memory/vN.json is authoritative after supervisor verification."
        ),
        "version": f"v{version}" if version is not None else None,
        "episode": raw_episode,
        "phase": effective_phase,
        "journal_state": journal_state,
        "experiment_count": len(experiments),
        "latest_experiment": experiments[-1] if experiments else None,
        "outcome": value.get("outcome"),
        "candidate_commit": value.get("candidate_commit"),
        "base_commit": value.get("base_commit"),
        "episode_branch": value.get("episode_branch"),
        "created_at": value.get("created_at"),
        "updated_at": utc_now(),
        "canonical_memory": canonical_memory or None,
        "accepted": accepted,
    }
    atomic_write_json(path, live)
    return live


def _sync_live_best_effort(
    journal_path: Path,
    value: dict[str, Any],
    live_path: Path | None,
) -> None:
    destination = live_path or infer_live_memory_path(journal_path)
    if destination is None:
        return
    overrides: dict[str, Any] = {}
    if not isinstance(value.get("memory_version"), int):
        active_path = destination.parent.parent / ".atrex_long_horizon" / "active_episode.json"
        try:
            active = json.loads(active_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            active = None
        if (
            isinstance(active, dict)
            and active.get("episode") == value.get("episode")
            and isinstance(active.get("memory_version"), int)
        ):
            overrides["memory_version"] = active["memory_version"]
            overrides["episode"] = active["episode"]
            overrides["phase"] = str(active.get("phase", "exploring"))
    try:
        sync_live_memory(destination, value, **overrides)
    except OSError:
        # A diagnostic mirror must never invalidate the authoritative journal.
        pass


def initialize(
    path: Path,
    *,
    episode: int,
    base_commit: str,
    branch: str,
    memory_version: int | None = None,
    live_path: Path | None = None,
) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "episode": episode,
        "memory_version": memory_version,
        "base_commit": base_commit,
        "episode_branch": branch,
        "state": "in_progress",
        "experiments": [],
        "outcome": None,
        "created_at": utc_now(),
        "finalized_at": None,
    }
    atomic_write_json(path, value)
    _sync_live_best_effort(path, value, live_path)
    return value


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("unsupported episode journal")
    return value


def append_experiment(
    path: Path,
    experiment: dict[str, Any],
    *,
    live_path: Path | None = None,
) -> dict[str, Any]:
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
    _sync_live_best_effort(path, value, live_path)
    return value


def finalize(
    path: Path,
    *,
    state: str,
    outcome: dict[str, Any],
    candidate_commit: str = "",
    live_path: Path | None = None,
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
    _sync_live_best_effort(path, value, live_path)
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
    parser.add_argument(
        "--live-path",
        default="",
        help="Optional non-canonical memory/live.json progress mirror.",
    )
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
                Path(args.path),
                _json_object(args.experiment_json, "--experiment-json"),
                live_path=Path(args.live_path) if args.live_path else None,
            )
        else:
            value = finalize(
                Path(args.path),
                state=args.state,
                outcome=_json_object(args.outcome_json, "--outcome-json"),
                candidate_commit=args.candidate_commit,
                live_path=Path(args.live_path) if args.live_path else None,
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(value, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
