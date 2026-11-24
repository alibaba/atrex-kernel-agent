"""Versioned Review evidence and native resume state in the Measurement Store."""

from __future__ import annotations

import json
import os
import re
import shutil
import fcntl
from contextlib import contextmanager
from collections.abc import Iterator, Mapping
from pathlib import Path

from aka.agent.execution.home import open_private_directory
from aka.agent.session_tail import read_regular_bytes
from aka.agent.usage_summary import select_session_usage
from aka.experiment.measurement_store import private_write, private_write_bytes

MAX_REVIEW_FILE_BYTES = 64 * 1024 * 1024


def review_root(measurement_root: Path, identity: str) -> Path:
    """One private Review family in the Campaign's Measurement Store."""
    if not isinstance(identity, str) or re.fullmatch(r"[A-Za-z0-9_-]+", identity) is None:
        raise ValueError("Invalid Review identity")
    return Path(measurement_root) / "reviews" / identity


def review_revision_root(measurement_root: Path, identity: str, revision: int) -> Path:
    if type(revision) is not int or revision < 1:
        raise ValueError("Invalid Review revision")
    return review_root(measurement_root, identity) / f"v{revision}"


@contextmanager
def reviewer_workspace(root: Path):
    """Keep the native project path stable; replace only the supplied evidence.

    Home, captures and versioned verdicts are siblings, outside the Agent's
    evidence workspace. Serialize revisions of this Review family.
    """
    with open_private_directory(root) as directory:
        descriptor = os.open("session.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
                             0o600, dir_fd=directory)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        workspace = root / "workspace"
        if workspace.is_symlink():
            raise ValueError("Reviewer workspace must not be a symlink")
        if workspace.exists():
            shutil.rmtree(workspace)
        with open_private_directory(workspace):
            pass
        yield workspace
    finally:
        os.close(descriptor)


def reviewer_environment(root: Path) -> dict[str, str]:
    from aka.agent.execution.home import SESSION_HOME_ROOT_ENV, SESSION_HOME_ID_ENV
    return {SESSION_HOME_ROOT_ENV: str(root / "home"), SESSION_HOME_ID_ENV: root.name}


def _copy_regular(source: Path, destination: Path) -> None:
    if source.is_symlink():
        raise ValueError(f"Review archive source is not a regular file: {source.name}")
    source = source.resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"Review archive source is not a regular file: {source.name}")
    data = read_regular_bytes(source, limit=MAX_REVIEW_FILE_BYTES + 1)
    if len(data) > MAX_REVIEW_FILE_BYTES:
        raise ValueError(f"Review archive source exceeds 64 MiB: {source.name}")
    private_write_bytes(destination, data)


def archive_review(
    root: Path,
    *,
    result: Mapping[str, object],
    conversation: Path | None,
    token_usage: Path | None,
) -> Path:
    """Write one self-contained revision under its stable Review identity."""
    if (conversation is None) != (token_usage is None):
        raise ValueError(
            "Review conversation and token usage must be archived together"
        )
    with open_private_directory(root):
        pass
    if conversation is not None and token_usage is not None:
        _copy_regular(conversation, root / "conversation.jsonl")
        _copy_regular(token_usage, root / "token-usage.json")
    private_write(root / "result.json", dict(result))
    return root


def archive_attempts(
    root: Path,
    *,
    session_audit: Mapping[str, object] | None,
    source_root: Path,
) -> tuple[Path | None, Path | None]:
    """Copy every reviewer attempt and return the accepted/latest complete pair."""
    attempts = (
        session_audit.get("attempts")
        if isinstance(session_audit, Mapping)
        else None
    )
    if not isinstance(attempts, list):
        return None, None
    selected: tuple[Path, Path] | None = None
    for row in attempts:
        if not isinstance(row, Mapping):
            continue
        number = row.get("attempt")
        conversation_path = row.get("conversation_path")
        token_usage_path = row.get("token_usage_path")
        if (
            type(number) is not int
            or number <= 0
            or not isinstance(conversation_path, str)
            or not isinstance(token_usage_path, str)
        ):
            continue
        conversation = (source_root / conversation_path).resolve(strict=True)
        token_usage = (source_root / token_usage_path).resolve(strict=True)
        source_boundary = source_root.resolve(strict=True)
        if (
            not conversation.is_relative_to(source_boundary)
            or not token_usage.is_relative_to(source_boundary)
        ):
            raise ValueError("Review Session archive escaped its private root")
        attempt_root = root / "attempts" / f"a{number}"
        _copy_regular(conversation, attempt_root / "conversation.jsonl")
        _copy_regular(token_usage, attempt_root / "token-usage.json")
        if row.get("status") == "accepted" or selected is None:
            selected = (conversation, token_usage)
    return selected if selected is not None else (None, None)


def move_review(source: Path, destination: Path) -> None:
    """Atomically attach a staged Review to its canonical object when possible."""
    if not source.is_dir() or source.is_symlink():
        raise ValueError("Staged Review archive is unavailable")
    if destination.exists():
        raise ValueError("Canonical Review archive already exists")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.replace(source, destination)
    except OSError:
        shutil.copytree(source, destination, symlinks=False)
        shutil.rmtree(source)


def remove_empty_parents(path: Path, *, stop: Path) -> None:
    """Remove empty private staging parents without touching canonical evidence."""
    stop = stop.resolve()
    current = path.resolve()
    while current != stop and current.is_relative_to(stop):
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def read_review_result(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def bound_review_usage_rows(
    campaign_root: Path,
    *,
    lineage: str | None = None,
) -> Iterator[dict[str, object]]:
    """Yield one row per executed object-attached Reviewer invocation."""

    def token_value(path: Path) -> dict[str, object] | None:
        try:
            value = json.loads(
                read_regular_bytes(
                    path.resolve(strict=True),
                    limit=MAX_REVIEW_FILE_BYTES + 1,
                )
            )
        except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    library = campaign_root / "measurements" / "reviews"
    for result_path in sorted(library.glob("submission-*/v*/result.json")):
        result = read_review_result(result_path)
        if result is None or result.get("review_execution") == "reused":
            continue
        submission = read_review_result(campaign_root / "submissions" /
                                       result_path.parent.parent.name / result_path.parent.name / "submission.json")
        if submission is None:
            continue
        row_lineage = submission.get("lineage")
        if lineage is not None and row_lineage != lineage:
            continue
        # Include every executed attempt, including failed resumes and invalid
        # verdicts. The top-level receipt is only a convenient final projection.
        token_paths = sorted(result_path.parent.glob("captures/*/run-*/token-usage.json"))
        for token_path in token_paths or [result_path.parent / "token-usage.json"]:
            token = token_value(token_path)
            if token is None:
                continue
            yield {
                "review_kind": "final_source_policy_review",
                "label": "independent final-source policy review",
                "lineage": row_lineage,
                "recorded_at": token.get("finished_at", result.get("reviewed_at")),
                "session_id": token.get("session_id", result.get("session_id")),
                "exit_status": token.get("exit_status"),
                "timed_out": token.get("timed_out") is True,
                "usage": select_session_usage(token)["usage"],
                "source": str(token_path),
            }

    claim_results = [
        *library.glob("direction_*/v*/result.json"),
        *library.glob("direction_*/attempts/*/result.json"),
        *library.glob("experiment_*/v*/result.json"),
        *library.glob("experiment_*/attempts/*/result.json"),
    ]
    for result_path in sorted(claim_results):
        result = read_review_result(result_path)
        if result is None:
            continue
        origin = result.get("origin")
        trajectory = origin.get("trajectory") if isinstance(origin, Mapping) else None
        row_lineage = (
            f"T{trajectory}" if type(trajectory) is int and trajectory > 0 else None
        )
        if lineage is not None and row_lineage != lineage:
            continue
        claim_type = result.get("claim_type")
        if claim_type not in {"direction", "experiment"}:
            continue
        review = result.get("review")
        review = review if isinstance(review, Mapping) else {}
        session_audit = review.get("session_audit")
        attempts = (
            session_audit.get("attempts")
            if isinstance(session_audit, Mapping)
            else None
        )
        attempt_rows = {
            row.get("attempt"): row
            for row in (attempts if isinstance(attempts, list) else [])
            if isinstance(row, Mapping)
            and type(row.get("attempt")) is int
        }
        token_paths = sorted(result_path.parent.glob("attempts/a*/token-usage.json"))
        if not token_paths:
            token_paths = [result_path.parent / "token-usage.json"]
        for token_path in token_paths:
            token = token_value(token_path)
            if token is None:
                continue
            attempt = None
            if token_path.parent.name.startswith("a"):
                try:
                    attempt = int(token_path.parent.name[1:])
                except ValueError:
                    attempt = None
            audit_row = attempt_rows.get(attempt, {})
            yield {
                "review_kind": f"{claim_type}_claim_review",
                "label": f"independent {claim_type} claim review",
                "lineage": row_lineage,
                "recorded_at": token.get("finished_at", result.get("archived_at", result.get("finished_at"))),
                "session_id": audit_row.get(
                    "session_id", token.get("session_id")
                ),
                "exit_status": audit_row.get(
                    "exit_status", token.get("exit_status")
                ),
                "timed_out": bool(
                    audit_row.get("timed_out", token.get("timed_out", False))
                ),
                "usage": select_session_usage(token)["usage"],
                "source": str(token_path),
            }
