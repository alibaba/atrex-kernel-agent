#!/usr/bin/env python3
"""Explain candidate gates, promotion decisions and their evidence."""
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
    load_snapshot,
    main_error,
    observed_at,
    table,
)


def _reason(attempt: dict, candidate) -> str | None:
    verification = attempt.get("verification") if isinstance(attempt.get("verification"), dict) else {}
    return next(
        (
            str(value)
            for value in (
                attempt.get("violation"),
                verification.get("error"),
                attempt.get("summary"),
                candidate.result.gate if candidate else None,
                candidate.result.correctness if candidate else None,
                candidate.result.state if candidate else None,
            )
            if value not in (None, "", "PASS", "pass")
        ),
        None,
    )


def snapshot(workspace: Path) -> dict:
    rows = []
    promotions = []
    for scope in discover_scopes(workspace):
        public = load_snapshot(scope)
        candidates = {candidate.version: candidate for candidate in public.candidates}
        attempts = {row["episode"]: row for row in attempt_documents(scope)}
        incumbent_latency = None
        for candidate in sorted(public.candidates, key=lambda item: item.version):
            attempt = attempts.get(candidate.episode, {})
            verification = attempt.get("verification") if isinstance(attempt.get("verification"), dict) else {}
            candidate_latency = finite_positive(
                verification.get("candidate_latency_us")
            ) or finite_positive(candidate.result.latency_us)
            previous = incumbent_latency
            if candidate_latency is not None and previous is not None:
                improvement = (previous - candidate_latency) * 100.0 / previous
            else:
                improvement = verification.get("improvement_pct")
            evidence = []
            record_id = verification.get("gateway_record_id")
            if isinstance(record_id, str) and record_id:
                evidence.append(record_id)
            episode = next((item for item in public.episodes if item.episode == candidate.episode), None)
            if episode and episode.final_candidate:
                for evaluation in episode.final_candidate.evaluations:
                    evidence.append(
                        {
                            "kind": evaluation.kind,
                            "timestamp": evaluation.timestamp,
                            "state": evaluation.result.state,
                            "latency_us": evaluation.result.latency_us,
                            "source_sha256": dict(evaluation.source_sha256),
                        }
                    )
            accepted = candidate.promoted and candidate.result.state == "pass"
            is_promotion = accepted and candidate.episode is not None
            row = {
                "scope": scope.label,
                "trajectory": scope.trajectory,
                "episode": candidate.episode,
                "version": candidate.version,
                "git_commit": candidate.revision,
                "gate": verification.get("gate", candidate.result.gate),
                "correctness": candidate.result.correctness,
                "candidate_latency_us": candidate_latency,
                "incumbent_latency_us": finite_positive(verification.get("incumbent_latency_us")) or previous,
                "improvement_pct": improvement,
                "promoted": is_promotion,
                "accepted": accepted,
                "decision_reason": (
                    "promotion gate passed" if is_promotion
                    else "baseline accepted" if accepted
                    else _reason(attempt, candidate)
                ),
                "status": attempt.get("status"),
                "evidence": evidence,
            }
            rows.append(row)
            if is_promotion:
                promotions.append(row)
            # The accepted Bootstrap/Framework Baseline is the incumbent for
            # the first Episode even though it is not itself a promotion.
            if accepted and candidate_latency is not None:
                incumbent_latency = candidate_latency
        # Terminal attempts may have no CandidateFact when source capture failed.
        known = {row["episode"] for row in rows if row["scope"] == scope.label}
        for episode, attempt in sorted(attempts.items()):
            if episode in known:
                continue
            verification = attempt.get("verification") if isinstance(attempt.get("verification"), dict) else {}
            rows.append(
                {
                    "scope": scope.label,
                    "trajectory": scope.trajectory,
                    "episode": episode,
                    "version": attempt.get("version"),
                    "git_commit": attempt.get("candidate_commit"),
                    "gate": verification.get("gate"),
                    "correctness": None,
                    "candidate_latency_us": finite_positive(verification.get("candidate_latency_us")),
                    "incumbent_latency_us": finite_positive(verification.get("incumbent_latency_us")),
                    "improvement_pct": verification.get("improvement_pct"),
                    "promoted": attempt.get("accepted") is True,
                    "accepted": attempt.get("accepted") is True,
                    "decision_reason": _reason(attempt, candidates.get(attempt.get("version"))),
                    "status": attempt.get("status"),
                    "evidence": [verification["gateway_record_id"]]
                    if isinstance(verification.get("gateway_record_id"), str)
                    else [],
                }
            )
    rows.sort(key=lambda row: (str(row["scope"]), row["version"] if isinstance(row["version"], int) else -1))
    return {
        "schema_version": 1,
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
