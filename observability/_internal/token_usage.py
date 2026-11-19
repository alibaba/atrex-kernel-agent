#!/usr/bin/env python3
"""Build Session/Episode token usage and token-checkpoint projections."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from observability._internal.common import (
    add_output_arguments,
    attempt_documents,
    discover_scopes,
    elapsed_seconds,
    emit,
    main_error,
    observed_at,
    read_json,
    startup_stage,
    table,
    telemetry_documents,
)
from aka.workflow.light.long_horizon.review_usage import REVIEW_KINDS


from aka.agent.usage_summary import TOKEN_KEYS, usage as _usage, summarize_usage, session_usage


def _summed_session_usage(rows: list[dict]) -> dict:
    return summarize_usage(rows)["usage"]


def _review_rows(scope) -> list[dict]:
    rows = []
    from aka.workflow.light.support.paths import supervisor_campaign_base_root
    from aka.workflow.light.long_horizon.review_archive import bound_review_usage_rows

    campaign_root = supervisor_campaign_base_root(scope.workspace)
    lineage = f"T{scope.trajectory}" if scope.trajectory is not None else None
    for value in bound_review_usage_rows(campaign_root, lineage=lineage):
        rows.append({
            "scope": scope.label,
            "trajectory": scope.trajectory,
            "review_kind": value.get("review_kind"),
            "label": value.get("label"),
            "backend": None,
            "model": None,
            "recorded_at": value.get("recorded_at"),
            "session_id": value.get("session_id"),
            "exit_status": value.get("exit_status"),
            "timed_out": value.get("timed_out") is True,
            "usage": _usage(value.get("usage")),
            "observation_errors": [],
            "record_path": value.get("source"),
        })
    # Read-only compatibility for pre-object-bound Review receipts.
    root = scope.audit_root / "review-usage"
    for path in sorted(root.glob("*/*.json")):
        value = read_json(path, default={})
        if (
            not isinstance(value, dict)
            or value.get("schema") != "atrex.light-review-usage.v1"
        ):
            continue
        kind = value.get("review_kind")
        if not isinstance(kind, str) or not kind:
            continue
        rows.append({
            "scope": scope.label,
            "trajectory": scope.trajectory,
            "review_kind": kind,
            "label": value.get("label"),
            "backend": value.get("backend"),
            "model": value.get("model"),
            "recorded_at": value.get("recorded_at"),
            "session_id": value.get("session_id"),
            "exit_status": value.get("exit_status"),
            "timed_out": value.get("timed_out") is True,
            "usage": _usage(value.get("usage")),
            "observation_errors": (
                value.get("observation_errors")
                if isinstance(value.get("observation_errors"), list)
                else []
            ),
            "record_path": str(path),
        })
    return rows


def _review_summary(rows: list[dict]) -> dict:
    return summarize_usage(rows) | {
        "successful_invocation_count": sum(row.get("exit_status") == 0 and not row.get("timed_out") for row in rows),
        "failed_invocation_count": sum(row.get("exit_status") not in {0, None} for row in rows),
        "timed_out_invocation_count": sum(bool(row.get("timed_out")) for row in rows),
    }


def _token_checkpoint_rows(scope) -> list[dict]:
    rows = []
    for path in sorted((scope.audit_root / "token-checkpoints").glob("*/*.json")):
        value = read_json(path, default={})
        if (
            not isinstance(value, dict)
            or value.get("schema") != "atrex.light-token-checkpoint.v1"
        ):
            continue
        all_paths = value.get("all_paths")
        all_paths = all_paths if isinstance(all_paths, dict) else {}
        main = value.get("main_agent")
        main = main if isinstance(main, dict) else {}
        reviews = value.get("reviews")
        reviews = reviews if isinstance(reviews, dict) else {}
        rows.append({
            "scope": scope.label,
            "trajectory": scope.trajectory,
            "checkpoint_id": value.get("checkpoint_id"),
            "event": value.get("event"),
            "occurred_at": value.get("occurred_at"),
            "identity": value.get("identity") if isinstance(value.get("identity"), dict) else {},
            "usage": _usage(all_paths.get("usage")),
            "main_agent_usage": _usage(main.get("usage")),
            "review_usage": _usage(reviews.get("usage")),
            "reviews_by_kind": (
                value.get("reviews_by_kind")
                if isinstance(value.get("reviews_by_kind"), dict)
                else {}
            ),
            "warnings": value.get("warnings") if isinstance(value.get("warnings"), list) else [],
            "record_path": str(path),
        })
    return rows


def _all_paths_totals(main_usage, review_summary, *, main_rows, review_rows) -> dict:
    return summarize_usage([*main_rows, *review_rows])["usage"] | {
        "main_agent_total_tokens": main_usage.get("total_tokens"),
        "review_total_tokens": review_summary["usage"].get("total_tokens"),
    }


def _session_rows(scope) -> list[dict]:
    rows = []
    root = scope.segment_root / "episodes"
    for path in sorted(root.glob("e*/sessions/run-*/token-usage.json")):
        value = read_json(path, default={})
        if not isinstance(value, dict):
            continue
        try:
            episode = int(path.parents[2].name[1:])
        except (ValueError, IndexError):
            episode = None
        rows.append(
            {
                "scope": scope.label,
                "trajectory": scope.trajectory,
                "episode": episode,
                "stage": "episode",
                "run": path.parent.name,
                "session_id": value.get("session_id"),
                "backend": value.get("backend"),
                "state": value.get("state"),
                "started_at": value.get("started_at"),
                "finished_at": value.get("finished_at"),
                "wall_seconds": elapsed_seconds(value.get("started_at"), value.get("finished_at")),
                "usage": session_usage(value),
                "accounting_basis": value.get("accounting_basis"),
                "warnings": value.get("warnings") if isinstance(value.get("warnings"), list) else [],
                "capture_errors": value.get("capture_errors") if isinstance(value.get("capture_errors"), list) else [],
            }
        )
    startup_root = scope.workspace.parent / ".atrex-session-traces" / scope.workspace.name
    for path in sorted(startup_root.glob("run-*/token-usage.json")):
        value = read_json(path, default={})
        if not isinstance(value, dict):
            continue
        context = value.get("context") if isinstance(value.get("context"), dict) else {}
        captured_workspace = context.get("workspace")
        if isinstance(captured_workspace, str):
            try:
                if Path(captured_workspace).resolve() != scope.workspace.resolve():
                    continue
            except OSError:
                continue
        rows.append(
            {
                "scope": scope.label,
                "trajectory": scope.trajectory,
                "episode": 0,
                "stage": startup_stage(scope.workspace),
                "run": path.parent.name,
                "session_id": value.get("session_id"),
                "backend": value.get("backend"),
                "state": value.get("state"),
                "started_at": value.get("started_at"),
                "finished_at": value.get("finished_at"),
                "wall_seconds": elapsed_seconds(value.get("started_at"), value.get("finished_at")),
                "usage": session_usage(value),
                "accounting_basis": value.get("accounting_basis"),
                "warnings": value.get("warnings") if isinstance(value.get("warnings"), list) else [],
                "capture_errors": value.get("capture_errors") if isinstance(value.get("capture_errors"), list) else [],
            }
        )
    return rows


def snapshot(workspace: Path) -> dict:
    sessions = []
    episodes = []
    reviews = []
    review_index = {}
    token_checkpoints = []
    for scope in discover_scopes(workspace):
        attempts = {
            attempt["episode"]: attempt
            for attempt in attempt_documents(scope)
            if type(attempt.get("episode")) is int
        }
        scope_sessions = _session_rows(scope)
        # A central Review is visible from both Campaign and lineage scopes.
        # Keep its owner scope when available, but count its immutable path once.
        for row in _review_rows(scope):
            key = row["record_path"]
            previous = review_index.get(key)
            if previous is None:
                review_index[key] = len(reviews)
                reviews.append(row)
            elif row["trajectory"] is not None:
                reviews[previous] = row
        token_checkpoints.extend(_token_checkpoint_rows(scope))
        sessions.extend(scope_sessions)
        session_by_episode: dict[int, list[dict]] = {}
        for session in scope_sessions:
            if session["stage"] == "episode" and isinstance(session["episode"], int):
                session_by_episode.setdefault(session["episode"], []).append(session)
        telemetry_episodes = set()
        for telemetry in telemetry_documents(scope):
            episode = telemetry.get("episode")
            if isinstance(episode, int):
                telemetry_episodes.add(episode)
            phase_tokens = telemetry.get("phase_tokens") if isinstance(telemetry.get("phase_tokens"), dict) else {}
            terminal = _usage(phase_tokens.get("terminal_usage"))
            episodes.append(
                {
                    "scope": scope.label,
                    "trajectory": scope.trajectory,
                    "episode": episode,
                    "status": telemetry.get("status"),
                    "resume_count": telemetry.get("resume_count"),
                    "invocation_count": telemetry.get("invocation_count"),
                    "usage": terminal,
                    "control_tokens": telemetry.get("control_tokens"),
                    "measurement": telemetry.get("measurement"),
                    "reason_codes": telemetry.get("reason_codes") or [],
                    "session_count": len(session_by_episode.get(episode, [])),
                    "wall_seconds": sum(row["wall_seconds"] or 0.0 for row in session_by_episode.get(episode, [])) or None,
                }
            )
        for episode, episode_sessions in sorted(session_by_episode.items()):
            if episode in telemetry_episodes:
                continue
            terminal = _summed_session_usage(episode_sessions)
            candidate = attempts.get(episode)
            episodes.append(
                {
                    "scope": scope.label,
                    "trajectory": scope.trajectory,
                    "episode": episode,
                    "status": candidate.get("status") if candidate else "running_or_interrupted",
                    "resume_count": None,
                    "invocation_count": None,
                    "usage": terminal,
                    "control_tokens": None,
                    "measurement": "session_capture_fallback",
                    "reason_codes": ["telemetry_summary_unavailable"],
                    "session_count": len(episode_sessions),
                    "wall_seconds": sum(row["wall_seconds"] or 0.0 for row in episode_sessions) or None,
                }
            )

    episodes.sort(key=lambda row: (row["scope"], row["episode"] if isinstance(row["episode"], int) else -1))
    startup_sessions = [row for row in sessions if row["stage"] != "episode"]
    token_totals = {
        key: (
            sum(values)
            if (values := [
                row["usage"][key]
                for row in [*episodes, *startup_sessions]
                if type(row["usage"].get(key)) is int
            ])
            else None
        )
        for key in TOKEN_KEYS
    }
    startup_token_totals = {
        key: (
            sum(values)
            if (values := [
                row["usage"][key]
                for row in startup_sessions
                if type(row["usage"].get(key)) is int
            ])
            else None
        )
        for key in TOKEN_KEYS
    }
    reviews.sort(key=lambda row: (
        row["scope"], row["review_kind"], str(row.get("recorded_at") or "")
    ))
    token_checkpoints.sort(key=lambda row: (
        str(row.get("occurred_at") or ""),
        row["scope"],
        str(row.get("checkpoint_id") or ""),
    ))
    known_review_kinds = REVIEW_KINDS | {
        row["review_kind"] for row in reviews
    }
    review_kinds = {
        kind: _review_summary([
            row for row in reviews if row["review_kind"] == kind
        ])
        for kind in sorted(known_review_kinds)
    }
    review_summary = _review_summary(reviews)
    all_paths_totals = _all_paths_totals(
        token_totals,
        review_summary,
        main_rows=[*episodes, *startup_sessions],
        review_rows=reviews,
    )
    result = {
        "schema_version": 5,
        "observed_at": observed_at(),
        "campaign_workspace": str(Path(workspace).expanduser().resolve()),
        "main_agent_totals": token_totals,
        "all_paths_totals": all_paths_totals,
        "review_totals": review_summary,
        "review_totals_by_kind": review_kinds,
        "sessions": sessions,
        "episodes": episodes,
        "reviews": reviews,
        "token_checkpoints": token_checkpoints,
        "passing_full_evaluate_token_checkpoints": [
            row for row in token_checkpoints
            if row.get("event") == "passing-full-evaluate"
        ],
        "accepted_submission_token_checkpoints": [
            row for row in token_checkpoints
            if row.get("event") == "accepted-submit-kernel"
        ],
    }
    if startup_sessions:
        result["startup_totals"] = startup_token_totals
    return result


def render(value: dict) -> str:
    overall = value.get("all_paths_totals") or {}
    lines = [
        "AKA Light Token 与时间",
        "",
        "All paths",
        table(
            ["Total", "Main Agent", "Reviews", "Input", "Output", "Reasoning", "Measurement"],
            [[
                overall.get("total_tokens"),
                overall.get("main_agent_total_tokens"),
                overall.get("review_total_tokens"),
                overall.get("input_tokens"),
                overall.get("output_tokens"),
                overall.get("reasoning_tokens"),
                overall.get("measurement"),
            ]],
        ),
        "",
        "Main Agent",
    ]
    lines.append(table(
        ["Scope", "Episode", "Tokens", "Reasoning", "Wall(s)", "Sessions"],
        [[row["scope"], row["episode"], row["usage"].get("total_tokens"),
          row["usage"].get("reasoning_tokens"),
          round(row["wall_seconds"], 1) if row["wall_seconds"] else None,
          row["session_count"]]
         for row in value["episodes"]],
    ))
    startup_sessions = [row for row in value["sessions"] if row["stage"] != "episode"]
    if startup_sessions:
        lines += ["", "Pre-Episode / Legacy Baseline Sessions"]
        lines.append(table(
            ["Scope", "Stage", "Session", "Backend", "State", "Tokens", "Reasoning", "Wall(s)"],
            [[
                row["scope"], row["stage"], row["session_id"], row["backend"], row["state"],
                row["usage"].get("total_tokens"),
                row["usage"].get("reasoning_tokens"),
                round(row["wall_seconds"], 1) if row["wall_seconds"] else None,
            ] for row in startup_sessions],
        ))
    review_rows = value.get("review_totals_by_kind") or {}
    if review_rows:
        lines += ["", "Independent Reviews"]
        lines.append(table(
            ["Review kind", "Calls", "Total", "Input", "Output", "Reasoning", "Timeout", "Failed"],
            [[
                kind,
                summary["invocation_count"],
                summary["usage"].get("total_tokens"),
                summary["usage"].get("input_tokens"),
                summary["usage"].get("output_tokens"),
                summary["usage"].get("reasoning_tokens"),
                summary["timed_out_invocation_count"],
                summary["failed_invocation_count"],
            ] for kind, summary in review_rows.items()],
        ))
    checkpoints = value.get("token_checkpoints") or []
    if checkpoints:
        lines += ["", "Kernel milestones"]
        lines.append(table(
            ["Scope", "Event", "Commit", "Gateway Record", "Total", "Main", "Reviews", "Reasoning", "Measurement"],
            [[
                row["scope"],
                row["event"],
                row.get("identity", {}).get("kernel_commit_id"),
                row.get("identity", {}).get("gateway_record_id"),
                row["usage"].get("total_tokens"),
                row["main_agent_usage"].get("total_tokens"),
                row["review_usage"].get("total_tokens"),
                row["usage"].get("reasoning_tokens"),
                row["usage"].get("measurement"),
            ] for row in checkpoints],
        ))
    return "\n".join(lines)


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
    return main_error("light-token-time", _main, argv)


if __name__ == "__main__":
    raise SystemExit(main())
