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
    discover_scopes,
    elapsed_seconds,
    emit,
    finite_positive,
    load_snapshot,
    main_error,
    observed_at,
    read_json,
    startup_stage,
    table,
    telemetry_documents,
)


TOKEN_KEYS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "total_tokens")


def _usage(value: object) -> dict:
    value = value if isinstance(value, dict) else {}
    return {key: value.get(key) if type(value.get(key)) is int else None for key in TOKEN_KEYS} | {
        "reasoning_tokens": None,
        "measurement": value.get("measurement", "unavailable")
    }


def _summed_session_usage(rows: list[dict]) -> dict:
    usage = {}
    for key in TOKEN_KEYS:
        values = [row["usage"].get(key) for row in rows if type(row["usage"].get(key)) is int]
        usage[key] = sum(values) if values else None
    return usage | {"reasoning_tokens": None, "measurement": "session_capture_fallback"}


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
    incidents = Counter()
    for scope in discover_scopes(workspace):
        public = load_snapshot(scope)
        candidates = {candidate.episode: candidate for candidate in public.candidates if candidate.episode is not None}
        episode_facts = {episode.episode: episode for episode in public.episodes}
        scope_sessions = _session_rows(scope)
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
            candidate = candidates.get(episode)
            latency = finite_positive(candidate.result.latency_us) if candidate else None
            prior = None
            earlier = [
                item for item in public.candidates
                if item.promoted and candidate is not None and item.version < candidate.version
            ]
            if earlier:
                prior = finite_positive(max(earlier, key=lambda item: item.version).result.latency_us)
            total = terminal.get("total_tokens")
            improvement = ((prior - latency) if prior and latency else None)
            episodes.append(
                {
                    "scope": scope.label,
                    "trajectory": scope.trajectory,
                    "episode": episode,
                    "version": telemetry.get("version"),
                    "status": telemetry.get("status"),
                    "accepted": telemetry.get("accepted"),
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
                    "latency_improvement_us": improvement,
                    "latency_improvement_us_per_million_tokens": (
                        improvement * 1_000_000 / total
                        if improvement is not None and isinstance(total, int) and total > 0
                        else None
                    ),
                }
            )
        for episode, episode_sessions in sorted(session_by_episode.items()):
            if episode in telemetry_episodes:
                continue
            terminal = _summed_session_usage(episode_sessions)
            candidate = candidates.get(episode)
            latency = finite_positive(candidate.result.latency_us) if candidate else None
            earlier = [
                item for item in public.candidates
                if item.promoted and candidate is not None and item.version < candidate.version
            ]
            prior = (
                finite_positive(max(earlier, key=lambda item: item.version).result.latency_us)
                if earlier else None
            )
            improvement = (prior - latency) if prior and latency else None
            total = terminal.get("total_tokens")
            fact = episode_facts.get(episode)
            episodes.append(
                {
                    "scope": scope.label,
                    "trajectory": scope.trajectory,
                    "episode": episode,
                    "version": candidate.version if candidate else None,
                    "status": fact.status if fact else "running_or_interrupted",
                    "accepted": fact.accepted if fact else False,
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
                    "latency_improvement_us": improvement,
                    "latency_improvement_us_per_million_tokens": (
                        improvement * 1_000_000 / total
                        if improvement is not None and isinstance(total, int) and total > 0
                        else None
                    ),
                }
            )
        for episode in public.episodes:
            if episode.status == "invalid_handoff":
                incidents["invalid_handoff"] += 1
        counts = Counter(
            row["episode"] for row in scope_sessions
            if row["stage"] == "episode" and isinstance(row["episode"], int)
        )
        incidents["session_restart"] += sum(max(0, count - 1) for count in counts.values())

    episodes.sort(key=lambda row: (row["scope"], row["episode"] if isinstance(row["episode"], int) else -1))
    startup_sessions = [row for row in sessions if row["stage"] != "episode"]
    token_totals = {
        key: (
            sum(row["usage"].get(key) or 0 for row in episodes)
            + sum(row["usage"].get(key) or 0 for row in startup_sessions)
        )
        for key in TOKEN_KEYS
    }
    startup_token_totals = {
        key: sum(row["usage"].get(key) or 0 for row in startup_sessions)
        for key in TOKEN_KEYS
    }
    episode_totals = [row["usage"].get("total_tokens") for row in episodes if isinstance(row["usage"].get("total_tokens"), int)]
    return {
        "schema_version": 1,
        "observed_at": observed_at(),
        "campaign_workspace": str(Path(workspace).expanduser().resolve()),
        "totals": token_totals,
        "startup_totals": startup_token_totals,
        "token_availability": {
            "input_tokens": "available when the Agent backend reports usage",
            "output_tokens": "available when the Agent backend reports usage",
            "reasoning_tokens": (
                "not reported as a separate counter by current main; a backend may account "
                "reasoning inside output_tokens"
            ),
            "cache_read_tokens": "available when the Agent backend reports usage",
            "cache_write_tokens": "available when the Agent backend reports usage",
            "episode_fallback": (
                "when telemetry.summary.json is absent, totals use the sum of that Episode's "
                "captured Session usage and mark measurement=session_capture_fallback"
            ),
        },
        "attempt_unit": "one Light Episode",
        "average_tokens_per_attempt": sum(episode_totals) / len(episode_totals) if episode_totals else None,
        "incidents": dict(sorted(incidents.items())),
        "time_availability": {
            "wall_seconds": "available for finalized captured Sessions",
            "active_seconds": "not durably persisted by current main",
            "agate_wait_seconds": "not durably persisted by current main",
        },
        "sessions": sessions,
        "episodes": episodes,
    }


def render(value: dict) -> str:
    lines = ["AKA Light Token 与时间", ""]
    lines.append(table(
        ["Scope", "Episode", "Tokens", "Wall(s)", "Accepted", "Improve us/Mtok", "Sessions"],
        [[row["scope"], row["episode"], row["usage"].get("total_tokens"),
          round(row["wall_seconds"], 1) if row["wall_seconds"] else None, row["accepted"],
          round(row["latency_improvement_us_per_million_tokens"], 4)
          if row["latency_improvement_us_per_million_tokens"] is not None else None,
          row["session_count"]]
         for row in value["episodes"]],
    ))
    startup_sessions = [row for row in value["sessions"] if row["stage"] != "episode"]
    if startup_sessions:
        lines += ["", "Bootstrap / Framework Baseline Sessions"]
        lines.append(table(
            ["Scope", "Stage", "Session", "Backend", "State", "Tokens", "Wall(s)"],
            [[
                row["scope"], row["stage"], row["session_id"], row["backend"], row["state"],
                row["usage"].get("total_tokens"),
                round(row["wall_seconds"], 1) if row["wall_seconds"] else None,
            ] for row in startup_sessions],
        ))
    lines += ["", f"Average tokens per Episode: {value['average_tokens_per_attempt'] if value['average_tokens_per_attempt'] is not None else '—'}",
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
