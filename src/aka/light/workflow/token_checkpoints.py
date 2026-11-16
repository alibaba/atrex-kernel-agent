"""Immutable cumulative token snapshots at meaningful optimization milestones."""
from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping

from aka.light.agent.home import open_private_directory
from aka.light.agent.session_tail import read_regular_bytes


TOKEN_KEYS = (
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "total_tokens",
)
MAX_USAGE_BYTES = 64 * 1024 * 1024


def _json(path: Path) -> dict:
    raw = read_regular_bytes(path, limit=MAX_USAGE_BYTES + 1)
    if len(raw) > MAX_USAGE_BYTES:
        raise ValueError("Token usage evidence exceeds 64 MiB")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Token usage evidence must be an object")
    return value


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    except ValueError:
        return None


def _usage(value: object) -> dict:
    value = value if isinstance(value, Mapping) else {}
    return {
        key: value.get(key) if type(value.get(key)) is int else None
        for key in TOKEN_KEYS
    } | {
        "measurement": (
            value.get("measurement")
            if value.get("measurement") in {"exact", "estimated", "partial"}
            else "unavailable"
        )
    }


def _summary(rows: list[dict], *, missing_expected: bool = False) -> dict:
    available = {
        key: sum(type(row["usage"].get(key)) is int for row in rows)
        for key in TOKEN_KEYS
    }
    usage = {
        key: (
            sum(
                row["usage"][key]
                for row in rows
                if type(row["usage"].get(key)) is int
            )
            if available[key]
            else None
        )
        for key in TOKEN_KEYS
    }
    measurements = [row["usage"].get("measurement") for row in rows]
    totals_complete = bool(rows) and available["total_tokens"] == len(rows)
    usage["measurement"] = (
        "unavailable"
        if not rows or not available["total_tokens"]
        else "partial"
        if missing_expected or not totals_complete
        else "exact"
        if all(value == "exact" for value in measurements)
        else "estimated"
        if all(value in {"exact", "estimated"} for value in measurements)
        else "partial"
    )
    return {
        "invocation_count": len(rows),
        "usage_available_invocations": available,
        "usage": usage,
    }


def _main_rows(session_capture_root: Path | None, occurred_at: datetime) -> tuple[list[dict], list[str]]:
    if session_capture_root is None:
        return [], ["main_agent_capture_not_configured"]
    episodes_root = session_capture_root.parent.parent
    rows, warnings = [], []
    current_seen = False
    for path in sorted(episodes_root.glob("e*/sessions/run-*/token-usage.json")):
        try:
            value = _json(path)
        except (OSError, ValueError, RecursionError) as error:
            warnings.append(f"unreadable_main_agent_usage:{path.parent.name}:{type(error).__name__}")
            continue
        started = _timestamp(value.get("started_at"))
        if started is not None and started > occurred_at:
            continue
        if path.parent.parent == session_capture_root:
            current_seen = True
        rows.append({
            "episode": path.parents[2].name,
            "run": path.parent.name,
            "session_id": value.get("session_id"),
            "started_at": value.get("started_at"),
            "finished_at": value.get("finished_at"),
            "state": value.get("state"),
            "usage": _usage(value.get("total")),
            "accounting_basis": value.get("accounting_basis"),
            "source": str(path),
        })
    if not current_seen:
        warnings.append("current_main_agent_capture_unavailable")
    return rows, warnings


def _review_rows(audit_root: Path, occurred_at: datetime) -> tuple[list[dict], list[str]]:
    rows, warnings = [], []
    lineage = None
    campaign_root = audit_root
    if audit_root.parent.name == "lineages":
        campaign_root = audit_root.parent.parent
        suffix = audit_root.name.removeprefix("t")
        if suffix.isdigit() and int(suffix) > 0:
            lineage = f"T{int(suffix)}"
    from aka.light.workflow.review_archive import bound_review_usage_rows

    for value in bound_review_usage_rows(campaign_root, lineage=lineage):
        recorded = _timestamp(value.get("recorded_at"))
        if recorded is not None and recorded > occurred_at:
            continue
        rows.append({
            "review_kind": value.get("review_kind"),
            "label": value.get("label"),
            "session_id": value.get("session_id"),
            "recorded_at": value.get("recorded_at"),
            "usage": _usage(value.get("usage")),
            "source": value.get("source"),
        })
    # Read-only compatibility for Campaigns created before Reviews were bound
    # directly to Direction/Experiment revisions and submissions.
    for path in sorted((audit_root / "review-usage").glob("*/*.json")):
        try:
            value = _json(path)
        except (OSError, ValueError, RecursionError) as error:
            warnings.append(f"unreadable_review_usage:{path.name}:{type(error).__name__}")
            continue
        recorded = _timestamp(value.get("recorded_at"))
        if recorded is not None and recorded > occurred_at:
            continue
        rows.append({
            "review_kind": value.get("review_kind"),
            "label": value.get("label"),
            "session_id": value.get("session_id"),
            "recorded_at": value.get("recorded_at"),
            "usage": _usage(value.get("usage")),
            "source": str(path),
        })
    return rows, warnings


def _write_once(path: Path, value: dict) -> dict:
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    with open_private_directory(path.parent) as parent:
        temporary = ".token-checkpoint-" + uuid.uuid4().hex
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent,
        )
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(encoded)
                output.flush()
                os.fsync(output.fileno())
            try:
                os.link(
                    temporary,
                    path.name,
                    src_dir_fd=parent,
                    dst_dir_fd=parent,
                    follow_symlinks=False,
                )
            except FileExistsError:
                existing = _json(path)
                if (
                    existing.get("schema") != value.get("schema")
                    or existing.get("checkpoint_id") != value.get("checkpoint_id")
                    or existing.get("event") != value.get("event")
                    or existing.get("identity") != value.get("identity")
                ):
                    raise ValueError("Token checkpoint identity collision")
                return existing
            os.fsync(parent)
        finally:
            try:
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass
    return value


def capture_token_checkpoint(
    *,
    audit_root: Path,
    session_capture_root: Path | None,
    event: str,
    checkpoint_id: str,
    occurred_at: str,
    identity: Mapping[str, object],
) -> tuple[dict, dict]:
    """Persist one cumulative main-Agent plus independent-review token snapshot."""
    moment = _timestamp(occurred_at)
    if moment is None:
        raise ValueError("Token checkpoint requires an ISO-8601 event timestamp")
    main_rows, main_warnings = _main_rows(session_capture_root, moment)
    review_rows, review_warnings = _review_rows(audit_root, moment)
    missing_main = "current_main_agent_capture_unavailable" in main_warnings
    main = _summary(main_rows, missing_expected=missing_main)
    reviews = _summary(review_rows)
    reviews_by_kind = {
        kind: _summary([
            row for row in review_rows if row.get("review_kind") == kind
        ])
        for kind in sorted({
            row.get("review_kind")
            for row in review_rows
            if isinstance(row.get("review_kind"), str)
        })
    }
    all_paths = _summary(
        [*main_rows, *review_rows], missing_expected=missing_main
    )
    all_paths["usage"]["main_agent_total_tokens"] = main["usage"]["total_tokens"]
    all_paths["usage"]["review_total_tokens"] = reviews["usage"]["total_tokens"]
    value = {
        "schema": "atrex.light-token-checkpoint.v1",
        "checkpoint_id": checkpoint_id,
        "event": event,
        "recorded_at": datetime.now(moment.tzinfo).isoformat(),
        "occurred_at": occurred_at,
        "identity": dict(identity),
        "main_agent": main,
        "reviews": reviews,
        "reviews_by_kind": reviews_by_kind,
        "all_paths": all_paths,
        "warnings": [*main_warnings, *review_warnings],
        "sources": {
            "main_agent_invocations": main_rows,
            "review_invocations": review_rows,
        },
    }
    relative = Path("token-checkpoints") / event / f"{checkpoint_id}.json"
    stored = _write_once(audit_root / relative, value)
    usage = stored["all_paths"]["usage"]
    reference = {
        "checkpoint_id": checkpoint_id,
        "path": relative.as_posix(),
        "occurred_at": stored["occurred_at"],
        "measurement": usage.get("measurement"),
        "main_agent_total_tokens": usage.get("main_agent_total_tokens"),
        "review_total_tokens": usage.get("review_total_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "reasoning_tokens": usage.get("reasoning_tokens"),
    }
    return reference, stored
