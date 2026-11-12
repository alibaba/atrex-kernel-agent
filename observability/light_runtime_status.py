#!/usr/bin/env python3
"""Report Campaign, trajectory, Episode and Session progress for AKA Light."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from observability.common import (
    TERMINAL_CAMPAIGN_PHASES,
    TERMINAL_EPISODE_STATUSES,
    KIND_PATTERN,
    add_output_arguments,
    age_seconds,
    attempt_documents,
    discover_scopes,
    emit,
    main_error,
    observed_at,
    read_json,
    search_state,
    startup_stage,
    table,
)
from aka.legacy.workflow.public_state import writer_active


AGATE_KINDS = frozenset(
    {"run", "evaluate", "same_allocation_abba", "abba", "profile", "check", "disassemble", "dev"}
)
def _episode_rows(attempts: list[dict], active: int | None) -> list[dict]:
    """Render Episode boundaries as execution segments, not acceptance gates."""
    rows = []
    for attempt in attempts:
        episode = attempt.get("episode")
        if type(episode) is not int:
            continue
        status = str(attempt.get("status") or "unmeasured")
        if episode == active:
            category = "running"
        elif status in TERMINAL_EPISODE_STATUSES:
            category = "failed" if status in {"failed", "interrupted"} else "completed"
        else:
            category = "stalled"
        rows.append(
            {
                "episode": episode,
                "status": status,
                "category": category,
                "session_id": attempt.get("session_id"),
                "agent_head_commit_id": (
                    attempt.get("candidate_kernel_commit_id")
                    or attempt.get("candidate_commit")
                ),
                "summary": attempt.get("summary"),
                "violation": attempt.get("violation"),
            }
        )
    if active is not None and all(row["episode"] != active for row in rows):
        rows.append(
            {
                "episode": active,
                "status": "running",
                "category": "running",
                "session_id": None,
                "agent_head_commit_id": None,
                "summary": None,
                "violation": None,
            }
        )
    return sorted(rows, key=lambda row: row["episode"])


def _session_rows(scope) -> list[dict]:
    rows = []
    root = scope.workspace / ".atrex_long_horizon" / "episodes"
    for usage_path in sorted(root.glob("e*/sessions/run-*/token-usage.json")):
        value = read_json(usage_path, default={})
        if not isinstance(value, dict):
            continue
        try:
            episode = int(usage_path.parents[2].name[1:])
        except (ValueError, IndexError):
            episode = None
        rows.append(
            {
                "session_id": value.get("session_id") or usage_path.parent.name,
                "runtime": value.get("backend"),
                "episode": episode,
                "status": value.get("state"),
                "updated_at": _mtime(usage_path),
                "resume_supported": value.get("session_id") is not None,
                "active": value.get("state") == "running",
            }
        )
    return rows


def _progress(scope) -> dict:
    root = scope.workspace / ".atrex_long_horizon"
    state = read_json(root / "state.json", default={})
    state = state if isinstance(state, dict) else {}
    active = read_json(root / "active_episode.json", default={})
    active = active if isinstance(active, dict) else {}
    identity = read_json(root / "public-identity.json", default={})
    identity = identity if isinstance(identity, dict) else {}
    running = writer_active(scope.workspace)
    terminal = read_json(scope.workspace / "trace-retention-manifest.json", default={})
    terminal = terminal if isinstance(terminal, dict) else {}
    if running:
        phase = str(active.get("phase") or "running")
    elif (root / "pause-requested").exists():
        phase = "paused"
    else:
        phase = str(terminal.get("status") or "stopped")
    paths = [root / "state.json", root / "active_episode.json", scope.workspace / "memory/live.json"]
    updated = max((value for value in (_mtime(path) for path in paths) if value), default=None)
    current_episode = active.get("episode") if type(active.get("episode")) is int else None
    return {
        "campaign_id": identity.get("campaign_id"),
        "phase": phase,
        "writer_active": running,
        "episode": current_episode,
        "completed_episodes": state.get("episodes"),
        "updated_at": updated,
    }


def _mtime(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    except OSError:
        return None


def _tail_json_lines(path: Path, *, limit: int = 8 * 1024 * 1024) -> list[dict]:
    """Read a bounded transcript tail without following links or partial lines."""
    if path.is_symlink() or not path.is_file():
        return []
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            if size > limit:
                stream.seek(size - limit)
                stream.readline()
            payload = stream.read(limit)
    except OSError:
        return []
    rows = []
    for line in payload.splitlines():
        if len(line) > 2 * 1024 * 1024:
            continue
        try:
            value = json.loads(line)
        except (UnicodeError, ValueError):
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _tool_blocks(value: object):
    """Yield provider-native tool calls/results from a captured event."""
    if isinstance(value, dict):
        kind = value.get("type")
        if kind in {"tool_use", "tool_result"}:
            yield value
            return
        for child in value.values():
            yield from _tool_blocks(child)
    elif isinstance(value, list):
        for child in value:
            yield from _tool_blocks(child)


def _tool_phase(tool: object, content: object) -> tuple[str, str | None]:
    name = str(tool or "unknown")
    rendered = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    text = f"{name} {rendered}".casefold()
    if any(marker in text for marker in ("wiki-query", "wiki_query", "/v1/wiki/", "gpu wiki")):
        return "waiting_wiki", name
    match = KIND_PATTERN.search(rendered)
    kind = match.group(1).casefold() if match else ""
    if kind in AGATE_KINDS or re.search(r"(?:^|[ /])agate(?: |$).*(?:eval|profile|check|disassemble|dev|get)", text):
        return "waiting_agate", kind or name
    if "tools/sandbox.py" in text or "runtime tool" in text or "/v1/journal/" in text:
        return "executing_runtime_tool", kind or name
    return "executing_tool", name


def _conversation_phase(path: Path) -> tuple[str, str | None, str | None]:
    pending: dict[str, tuple[str, object, str | None]] = {}
    last_observed = _mtime(path)
    for row in _tail_json_lines(path):
        event = row.get("event") if row.get("type") == "provider_event" else row
        if not isinstance(event, dict):
            continue
        timestamp = event.get("timestamp") or row.get("timestamp") or last_observed
        # Codex emits command_execution items instead of Claude tool blocks.
        item = event.get("item")
        if isinstance(item, dict) and item.get("type") == "command_execution":
            identifier = str(item.get("id") or "codex-command")
            if event.get("type") in {"item.completed", "item.failed"} or item.get("status") in {"completed", "failed"}:
                pending.pop(identifier, None)
            else:
                pending[identifier] = ("command", item.get("command"), timestamp)
        for block in _tool_blocks(event):
            if block.get("type") == "tool_use":
                identifier = str(block.get("id") or f"anonymous-{len(pending)}")
                pending[identifier] = (str(block.get("name") or "unknown"), block.get("input"), timestamp)
            else:
                identifier = str(block.get("tool_use_id") or block.get("id") or "")
                pending.pop(identifier, None)
    if not pending:
        return "thinking", None, last_observed
    tool, content, timestamp = next(reversed(pending.values()))
    phase, label = _tool_phase(tool, content)
    return phase, label, timestamp or last_observed


def _active_session_rows(scope, *, current_episode: int | None, writer_active: bool, stale_after: float) -> list[dict]:
    if not writer_active:
        return []
    root = scope.workspace / ".atrex_long_horizon" / "episodes"
    rows = []
    for usage_path in sorted(root.glob("e*/sessions/run-*/token-usage.json")):
        value = read_json(usage_path, default={})
        if not isinstance(value, dict) or value.get("state") != "running":
            continue
        try:
            episode = int(usage_path.parents[2].name[1:])
        except (ValueError, IndexError):
            continue
        if current_episode is not None and episode != current_episode:
            continue
        conversation = usage_path.with_name("conversation.jsonl")
        phase, tool, phase_at = _conversation_phase(conversation)
        updated_at = max(
            (stamp for stamp in (_mtime(usage_path), _mtime(conversation)) if stamp is not None),
            default=value.get("started_at"),
        )
        stale = age_seconds(updated_at)
        rows.append(
            {
                "scope": scope.label,
                "trajectory": scope.trajectory,
                "episode": episode,
                "stage": "episode",
                "capture_run": usage_path.parent.name,
                "session_id": value.get("session_id") or usage_path.parent.name,
                "backend": value.get("backend"),
                "state": value.get("state"),
                "phase": phase,
                "tool": tool,
                "phase_observed_at": phase_at,
                "phase_seconds": age_seconds(phase_at),
                "started_at": value.get("started_at"),
                "updated_at": updated_at,
                "stagnation_seconds": stale,
                "health": "stalled" if stale is not None and stale > stale_after else "active",
            }
        )
    return rows


def _active_startup_session_rows(scope, *, stale_after: float) -> list[dict]:
    """Read pre-Episode bootstrap captures and legacy baseline captures."""
    root = scope.workspace.parent / ".atrex-session-traces" / scope.workspace.name
    rows = []
    for usage_path in sorted(root.glob("run-*/token-usage.json")):
        value = read_json(usage_path, default={})
        if not isinstance(value, dict) or value.get("state") != "running":
            continue
        context = value.get("context") if isinstance(value.get("context"), dict) else {}
        captured_workspace = context.get("workspace")
        if isinstance(captured_workspace, str):
            try:
                if Path(captured_workspace).resolve() != scope.workspace.resolve():
                    continue
            except OSError:
                continue
        conversation = usage_path.with_name("conversation.jsonl")
        phase, tool, phase_at = _conversation_phase(conversation)
        updated_at = max(
            (stamp for stamp in (_mtime(usage_path), _mtime(conversation)) if stamp is not None),
            default=value.get("started_at"),
        )
        stale = age_seconds(updated_at)
        rows.append(
            {
                "scope": scope.label,
                "trajectory": scope.trajectory,
                "episode": 0,
                "stage": startup_stage(scope.workspace),
                "capture_run": usage_path.parent.name,
                "session_id": value.get("session_id") or usage_path.parent.name,
                "backend": value.get("backend"),
                "state": value.get("state"),
                "phase": phase,
                "tool": tool,
                "phase_observed_at": phase_at,
                "phase_seconds": age_seconds(phase_at),
                "started_at": value.get("started_at"),
                "updated_at": updated_at,
                "stagnation_seconds": stale,
                "health": "stalled" if stale is not None and stale > stale_after else "active",
            }
        )
    return rows


def snapshot(workspace: Path, *, stale_after: float = 600.0) -> dict:
    scopes = discover_scopes(workspace)
    rows = []
    for scope in scopes:
        progress = _progress(scope)
        episodes = _episode_rows(attempt_documents(scope), progress["episode"])
        active_sessions = _active_session_rows(
            scope,
            current_episode=progress["episode"],
            writer_active=progress["writer_active"],
            stale_after=stale_after,
        )
        if not active_sessions:
            active_sessions = _active_startup_session_rows(
                scope,
                stale_after=stale_after,
            )
        activity_points = [
            (age_seconds(progress["updated_at"]), progress["updated_at"]),
            *[
                (session["stagnation_seconds"], session["updated_at"])
                for session in active_sessions
            ],
        ]
        latest_activity = min(
            (point for point in activity_points if point[0] is not None),
            default=(None, None),
            key=lambda point: point[0],
        )
        age, latest_activity_at = latest_activity
        if progress["phase"] in TERMINAL_CAMPAIGN_PHASES:
            health = progress["phase"]
        elif (progress["writer_active"] or active_sessions) and (age is None or age <= stale_after):
            health = "healthy"
        elif progress["writer_active"] or active_sessions:
            health = "stalled"
        else:
            health = "inactive"
        recorded_sessions = _session_rows(scope)
        active_ids = {row["session_id"] for row in active_sessions}
        for session in recorded_sessions:
            session["active"] = session["session_id"] in active_ids
        rows.append(
            {
                "scope": scope.label,
                "trajectory": scope.trajectory,
                "workspace": str(scope.workspace),
                "campaign_id": progress["campaign_id"],
                "phase": progress["phase"],
                "writer_active": progress["writer_active"],
                "current_episode": progress["episode"],
                "completed_episodes": progress["completed_episodes"],
                "latest_activity_at": latest_activity_at,
                "stagnation_seconds": age,
                "health": health,
                "episode_counts": dict(Counter(row["category"] for row in episodes)),
                "episodes": episodes,
                "active_session_count": len(active_sessions),
                "active_sessions": active_sessions,
                "sessions": recorded_sessions,
            }
        )

    search = search_state(scopes[0].workspace)
    search_summary = None
    if search is not None:
        active = search.get("active") if isinstance(search.get("active"), dict) else {}
        search_summary = {
            "epoch": search.get("epoch"),
            "completed_epochs": search.get("completed_epochs"),
            "phase": active.get("phase"),
            "stop_reason": search.get("stop_reason"),
            "winner": active.get("winner"),
            "jobs": [
                {
                    key: job.get(key)
                    for key in ("epoch", "trajectory", "episode", "step", "status")
                }
                for job in active.get("jobs", [])
                if isinstance(job, dict)
            ],
        }
    return {
        "schema_version": 3,
        "observed_at": observed_at(),
        "campaign_workspace": str(scopes[0].workspace),
        "stale_after_seconds": stale_after,
        "scopes": rows,
        "search": search_summary,
    }


def render(value: dict) -> str:
    lines = ["AKA Light 运行状态", ""]
    lines.append(
        table(
            ["Scope", "Campaign Phase", "Episode", "Done", "Health", "Stale(s)", "Active Sessions"],
            [
                [
                    row["scope"],
                    row["phase"],
                    row["current_episode"],
                    row["completed_episodes"],
                    row["health"],
                    round(row["stagnation_seconds"], 1) if row["stagnation_seconds"] is not None else None,
                    row["active_session_count"],
                ]
                for row in value["scopes"]
            ],
        )
    )
    for scope in value["scopes"]:
        if not scope["episodes"]:
            continue
        lines += ["", f"[{scope['scope']}] Episodes"]
        lines.append(
            table(
                ["Episode", "Status", "Category", "Agent HEAD", "Session"],
                [
                    [row["episode"], row["status"], row["category"], row["agent_head_commit_id"], row["session_id"]]
                    for row in scope["episodes"]
                ],
            )
        )
    active_sessions = [session for scope in value["scopes"] for session in scope["active_sessions"]]
    lines += ["", "Active Agent Sessions"]
    lines.append(
        table(
            ["Scope", "Stage", "Episode", "Session", "Backend", "Agent Phase", "Tool", "Phase(s)", "Health"],
            [
                [
                    row["scope"], row["stage"], row["episode"], row["session_id"], row["backend"], row["phase"],
                    row["tool"], round(row["phase_seconds"], 1) if row["phase_seconds"] is not None else None,
                    row["health"],
                ]
                for row in active_sessions
            ],
        )
    )
    if value["search"]:
        search = value["search"]
        lines += ["", f"Search: epoch={search['epoch']} completed={search['completed_epochs']} phase={search['phase']} stop={search['stop_reason'] or '—'}"]
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--stale-after", type=float, default=600.0, metavar="SECONDS")
    add_output_arguments(parser)
    options = parser.parse_args(argv)
    if options.stale_after <= 0:
        parser.error("--stale-after must be positive")
    return options


def _main(argv=None) -> int:
    options = parse_args(argv)
    value = snapshot(options.workspace, stale_after=options.stale_after)
    emit(value, as_json=options.json, compact=options.compact, text=render(value))
    return 0


def main(argv=None) -> int:
    return main_error("light-runtime-status", _main, argv)


if __name__ == "__main__":
    raise SystemExit(main())
