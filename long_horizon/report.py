from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .journal import finalize, validate_terminal
from .models import TERMINAL_STATUSES
from .protocol import atomic_write_json


def _json_object(raw: str, label: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def submit_report(
    *,
    journal_path: Path,
    handoff_path: Path,
    expected_episode: int,
    base_commit: str,
    branch: str,
    state: str,
    outcome: dict[str, Any],
    candidate_commit: str = "",
    last_trial_commit: str = "",
    live_path: Path | None = None,
) -> dict[str, Any]:
    """Validate and atomically publish a repairable terminal report.

    Validation always precedes handoff publication.  ``journal.finalize`` permits
    replacing a report in the same terminal state, so an Agent may correct bad
    fields and invoke this function again without losing the Episode.
    """
    candidate_commit = candidate_commit.strip()
    last_trial_commit = last_trial_commit.strip()
    finalize(
        journal_path,
        state=state,
        outcome=outcome,
        candidate_commit=candidate_commit,
        live_path=live_path,
    )
    diagnosis = validate_terminal(
        journal_path,
        expected_episode=expected_episode,
        base_commit=base_commit,
        branch=branch,
        state=state,
        candidate_commit=candidate_commit,
    )
    if diagnosis:
        raise ValueError(diagnosis)
    handoff: dict[str, str] = {"status": state}
    if candidate_commit:
        handoff["candidate_commit"] = candidate_commit
    if last_trial_commit:
        handoff["last_trial_commit"] = last_trial_commit
    atomic_write_json(handoff_path, handoff)
    return {
        "ok": True,
        "status": state,
        "message": "terminal report validated and published",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and publish one long-horizon terminal report."
    )
    parser.add_argument("--journal-path", required=True)
    parser.add_argument("--handoff-path", required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--live-path", default="")
    parser.add_argument("--state", choices=sorted(TERMINAL_STATUSES), required=True)
    parser.add_argument("--outcome-json", required=True)
    parser.add_argument("--candidate-commit", default="")
    parser.add_argument("--last-trial-commit", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = submit_report(
            journal_path=Path(args.journal_path),
            handoff_path=Path(args.handoff_path),
            expected_episode=args.episode,
            base_commit=args.base_commit,
            branch=args.branch,
            state=args.state,
            outcome=_json_object(args.outcome_json, "--outcome-json"),
            candidate_commit=args.candidate_commit,
            last_trial_commit=args.last_trial_commit,
            live_path=Path(args.live_path) if args.live_path else None,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "repairable": True,
                    "error": str(exc),
                    "message": "fix the reported field and run the same command again",
                },
                ensure_ascii=False,
            )
        )
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
