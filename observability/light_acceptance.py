#!/usr/bin/env python3
"""Explain Episode handoff, acceptance and promotion from Supervisor Attempts."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from observability.common import (
    add_output_arguments,
    attempt_documents,
    discover_scopes,
    emit,
    finite_positive,
    main_error,
    observed_at,
    table,
)


def _verification(attempt: dict) -> dict:
    value = attempt.get("verification")
    return value if isinstance(value, dict) else {}


def _reason(attempt: dict) -> str | None:
    verification = _verification(attempt)
    for value in (
        attempt.get("violation"),
        verification.get("error"),
        attempt.get("summary"),
        attempt.get("status"),
    ):
        if value not in (None, "", "PASS", "pass", "candidate_ready"):
            return str(value)
    return None


def _record_ids(attempt: dict) -> list[str]:
    verification = _verification(attempt)
    values = (
        attempt.get("gateway_record_id"),
        verification.get("gateway_record_id"),
        verification.get("artifact"),
    )
    return list(
        dict.fromkeys(value for value in values if isinstance(value, str) and value)
    )


def snapshot(workspace: Path) -> dict:
    rows = []
    promotions = []
    for scope in discover_scopes(workspace):
        incumbent_latency = None
        for attempt in attempt_documents(scope):
            verification = _verification(attempt)
            episode = attempt.get("episode")
            accepted = attempt.get("accepted") is True
            is_promotion = accepted and type(episode) is int and episode > 0
            candidate_latency = (
                finite_positive(verification.get("candidate_latency_us"))
                or finite_positive(attempt.get("latency_us_geomean"))
            )
            previous = (
                finite_positive(verification.get("incumbent_latency_us"))
                or incumbent_latency
            )
            improvement = verification.get("improvement_pct")
            if improvement is None and candidate_latency is not None and previous is not None:
                improvement = (previous - candidate_latency) * 100.0 / previous
            gate = verification.get("gate")
            if gate is None and episode == 0 and accepted:
                gate = "PASS"
            row = {
                "scope": scope.label,
                "trajectory": scope.trajectory,
                "episode": episode,
                "version": attempt.get("version", attempt.get("memory_version")),
                "kernel_commit_id": (
                    attempt.get("candidate_kernel_commit_id")
                    or attempt.get("candidate_commit")
                ),
                "gate": gate,
                "correctness": "PASS" if gate == "PASS" else None,
                "candidate_latency_us": candidate_latency,
                "incumbent_latency_us": previous,
                "improvement_pct": improvement,
                "promoted": is_promotion,
                "accepted": accepted,
                "decision_reason": (
                    "promotion gate passed" if is_promotion
                    else "framework baseline Episode 0 accepted" if accepted and episode == 0
                    else _reason(attempt)
                ),
                "status": attempt.get("status"),
                "evidence": _record_ids(attempt),
            }
            rows.append(row)
            if is_promotion:
                promotions.append(row)
            if accepted and candidate_latency is not None:
                incumbent_latency = candidate_latency
    rows.sort(
        key=lambda row: (
            str(row["scope"]),
            row["episode"] if isinstance(row["episode"], int) else -1,
        )
    )
    return {
        "schema_version": 2,
        "observed_at": observed_at(),
        "campaign_workspace": str(Path(workspace).expanduser().resolve()),
        "candidates": rows,
        "promotions": promotions,
    }


def render(value: dict) -> str:
    return "\n".join(
        [
            "AKA Light 接受与晋升",
            "",
            table(
                ["Scope", "Episode", "Version", "Gate", "Promoted", "Latency(us)", "Improve(%)", "Reason"],
                [
                    [
                        row["scope"], row["episode"], row["version"], row["gate"], row["promoted"],
                        round(row["candidate_latency_us"], 6) if row["candidate_latency_us"] else None,
                        round(float(row["improvement_pct"]), 4) if isinstance(row["improvement_pct"], (int, float)) else None,
                        row["decision_reason"],
                    ]
                    for row in value["candidates"]
                ],
            ),
        ]
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    add_output_arguments(parser)
    return parser.parse_args(argv)


def _main(argv=None) -> int:
    options = parse_args(argv)
    value = snapshot(options.workspace)
    emit(value, as_json=options.json, compact=options.compact, text=render(value))
    return 0


def main(argv=None) -> int:
    return main_error("light-acceptance", _main, argv)


if __name__ == "__main__":
    raise SystemExit(main())
