#!/usr/bin/env python3
"""Report Session/Episode token usage, wall time and token efficiency."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from observability.common import (
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
from aka.light.workflow.review_usage import REVIEW_KINDS


TOKEN_KEYS = (
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "total_tokens",
)


def _usage(value: object) -> dict:
    value = value if isinstance(value, dict) else {}
    return {
        key: value.get(key) if type(value.get(key)) is int else None
        for key in TOKEN_KEYS
    } | {"measurement": value.get("measurement", "unavailable")}


def _summed_session_usage(rows: list[dict]) -> dict:
    usage = {}
    for key in TOKEN_KEYS:
        values = [row["usage"].get(key) for row in rows if type(row["usage"].get(key)) is int]
        usage[key] = sum(values) if values else None
    return usage | {"measurement": "session_capture_fallback"}


def _review_rows(scope) -> list[dict]:
    rows = []
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
    available = {
        key: sum(type(row["usage"].get(key)) is int for row in rows)
        for key in TOKEN_KEYS
    }


    usage = {
        key: (
            sum(row["usage"][key] for row in rows if type(row["usage"].get(key)) is int)
            if available[key]
            else None
        )
        for key in TOKEN_KEYS
    }
    total_available = available["total_tokens"]
    measurements = [row["usage"].get("measurement") for row in rows]
    usage["measurement"] = (
        "unavailable"
        if not total_available
        else "exact"
        if total_available == len(rows) and all(value == "exact" for value in measurements)
        else "estimated"
        if total_available == len(rows)
        and all(value in {"exact", "estimated"} for value in measurements)
        else "partial"
    )
    return {
        "invocation_count": len(rows),
        "successful_invocation_count": sum(
            row.get("exit_status") == 0 and not row.get("timed_out") for row in rows
        ),
        "failed_invocation_count": sum(
            row.get("exit_status") not in {0, None} for row in rows
        ),
        "timed_out_invocation_count": sum(bool(row.get("timed_out")) for row in rows),
        "usage_available_invocations": available,
        "usage": usage,
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


def _all_paths_totals(
    main_usage: dict,
    review_summary: dict,
    *,
    main_rows: list[dict],
    review_rows: list[dict],
) -> dict:
    """Combine disjoint main-Agent and reviewer bills without double-counting reasoning."""
    review_usage = review_summary["usage"]
    combined = {}
    for key in TOKEN_KEYS:
        values = [
            value
            for value in (main_usage.get(key), review_usage.get(key))
            if type(value) is int
        ]
        combined[key] = sum(values) if values else None

    rows = [*main_rows, *review_rows]
    observed_total_rows = sum(
        type(row.get("usage", {}).get("total_tokens")) is int for row in rows
    )
    measurements = [row.get("usage", {}).get("measurement") for row in rows]
    combined["measurement"] = (
        "unavailable"
        if not rows or not observed_total_rows
        else "exact"
        if observed_total_rows == len(rows) and all(value == "exact" for value in measurements)
        else "estimated"
        if observed_total_rows == len(rows)
        and all(value in {"exact", "estimated"} for value in measurements)
        else "partial"
    )
    combined["main_agent_total_tokens"] = main_usage.get("total_tokens")
    combined["review_total_tokens"] = review_usage.get("total_tokens")
    return combined


def _session_rows(scope) -> list[dict]:
    rows = []
    root = scope.workspace / ".atrex_long_horizon" / "episodes"
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
                "usage": _usage(value.get("total")),
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
                "usage": _usage(value.get("total")),
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
    token_checkpoints = []
    incidents = Counter()
    for scope in discover_scopes(workspace):
        attempts = {
            attempt["episode"]: attempt
            for attempt in attempt_documents(scope)
            if type(attempt.get("episode")) is int
        }
        scope_sessions = _session_rows(scope)
        reviews.extend(_review_rows(scope))
        token_checkpoints.extend(_token_checkpoint_rows(scope))
        sessions.extend(scope_sessions)
        session_by_episode: dict[int, list[dict]] = {}
        for session in scope_sessions:
            if session["stage"] == "episode" and isinstance(session["episode"], int):
                session_by_episode.setdefault(session["episode"], []).append(session)
            text = " ".join([*session["warnings"], *session["capture_errors"]]).casefold()
            if "output" in text and "trunc" in text:
                incidents["output_truncation"] += 1
            if "rate_limit" in text or "throttl" in text:
                incidents["rate_limit"] += 1
        telemetry_episodes = set()
        for telemetry in telemetry_documents(scope):
            episode = telemetry.get("episode")
            if isinstance(episode, int):
                telemetry_episodes.add(episode)
            phase_tokens = telemetry.get("phase_tokens") if isinstance(telemetry.get("phase_tokens"), dict) else {}
            terminal = _usage(phase_tokens.get("terminal_usage"))
            invocations = telemetry.get("invocations") if isinstance(telemetry.get("invocations"), list) else []
            for invocation in invocations:
                if not isinstance(invocation, dict):
                    continue
                kind = str(invocation.get("failure_kind") or "").casefold()
                code = str(invocation.get("provider_error_code") or "").casefold()
                if "output" in kind and "limit" in kind:
                    incidents["output_truncation"] += 1
                if "rate" in kind or "rate" in code or "throttl" in code:
                    incidents["rate_limit"] += 1
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
                    "active_seconds": None,
                    "agate_wait_seconds": None,
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
                    "active_seconds": None,
                    "agate_wait_seconds": None,
                }
            )
        counts = Counter(
            row["episode"] for row in scope_sessions
            if row["stage"] == "episode" and isinstance(row["episode"], int)
        )
        incidents["session_restart"] += sum(max(0, count - 1) for count in counts.values())

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
    episode_totals = [row["usage"].get("total_tokens") for row in episodes if isinstance(row["usage"].get("total_tokens"), int)]
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
    return {
        "schema_version": 4,
        "observed_at": observed_at(),
        "campaign_workspace": str(Path(workspace).expanduser().resolve()),
        # Compatibility alias retained for existing consumers. Both fields are
        # main-Agent-only and intentionally exclude every independent reviewer.
        "totals": token_totals,
        "main_agent_totals": token_totals,
        "all_paths_totals": all_paths_totals,
        "startup_totals": startup_token_totals,
        "review_totals": review_summary,
        "review_totals_by_kind": review_kinds,
        "token_availability": {
            "input_tokens": "available when the Agent backend reports usage",
            "output_tokens": "available when the Agent backend reports usage",
            "reasoning_tokens": (
                "reported separately when the backend exposes a native reasoning counter; "
                "otherwise null and never inferred from output_tokens"
            ),
            "cache_read_tokens": "available when the Agent backend reports usage",
            "cache_write_tokens": "available when the Agent backend reports usage",
            "episode_fallback": (
                "when telemetry.summary.json is absent, totals use the sum of that Episode's "
                "captured Session usage and mark measurement=session_capture_fallback"
            ),
        },
        "episode_unit": "one Agent-defined Light Episode segment",
        "average_tokens_per_episode": sum(episode_totals) / len(episode_totals) if episode_totals else None,
        "incidents": dict(sorted(incidents.items())),
        "time_availability": {
            "wall_seconds": "available for finalized captured Sessions",
            "active_seconds": "not durably persisted by current main",
            "agate_wait_seconds": "not durably persisted by current main",
        },
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
    lines += ["", f"Average tokens per Episode: {value['average_tokens_per_episode'] if value['average_tokens_per_episode'] is not None else '—'}",
              f"Incidents: {value['incidents']}",
              "Active time / Agate wait time: unavailable in durable main records"]
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
