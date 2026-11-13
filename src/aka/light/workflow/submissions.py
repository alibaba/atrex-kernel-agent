"""Campaign-wide immutable audit records for ``submit-kernel`` calls."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

from aka.light.experiment.store import private_write
from aka.light.runtime.request_errors import RuntimeStateError

COMMIT_ID_RE = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
SUBMISSION_ID_RE = re.compile(r"submission-[0-9a-f]{32}")


def _records(root: Path) -> Iterable[Path]:
    current = [
        path / "submission.json"
        for path in root.glob("submission-*")
        if path.is_dir() and not path.is_symlink()
    ]
    # Flat JSON records are a read-only compatibility input. New writes always
    # co-locate the submission and its final-source Review in one directory.
    legacy = list(root.glob("submission-*.json"))
    return [*sorted(current), *sorted(legacy)]


def read_submission(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeStateError(
            "A private submit-kernel audit record is unreadable."
        ) from error
    if (
        not isinstance(value, dict)
        or value.get("schema") != "atrex.kernel-submission.v1"
        or not isinstance(value.get("submission_id"), str)
        or SUBMISSION_ID_RE.fullmatch(value["submission_id"]) is None
        or (
            path.stem != value["submission_id"]
            and path.parent.name != value["submission_id"]
        )
        or not isinstance(value.get("kernel_commit_id"), str)
        or COMMIT_ID_RE.fullmatch(value["kernel_commit_id"]) is None
        or type(value.get("accepted")) is not bool
        or not isinstance(value.get("lineage"), str)
        or re.fullmatch(r"T[1-9][0-9]*", value["lineage"]) is None
        or type(value.get("episode")) is not int
        or value["episode"] < 0
    ):
        raise RuntimeStateError(
            "A private submit-kernel audit record has an unsupported schema."
        )
    return value


def append_submission(root: Path, value: Mapping[str, object]) -> dict[str, object]:
    record = dict(value)
    submission_id = record.get("submission_id")
    if (
        not isinstance(submission_id, str)
        or SUBMISSION_ID_RE.fullmatch(submission_id) is None
    ):
        raise ValueError("Kernel submission requires a valid submission_id")
    # The schema is Runtime-owned.  Do not allow a migrated or caller-supplied
    # field to weaken the immutable record contract.
    record["schema"] = "atrex.kernel-submission.v1"
    path = root / submission_id / "submission.json"
    if path.exists():
        existing = read_submission(path)
        if existing != record:
            raise RuntimeStateError(
                "Kernel submission identity already belongs to different audit data"
            )
        return existing
    legacy_path = root / f"{submission_id}.json"
    if legacy_path.exists():
        existing = read_submission(legacy_path)
        if existing != record:
            raise RuntimeStateError(
                "Kernel submission identity already belongs to different audit data"
            )
    private_write(path, record)
    return record


def submissions(
    root: Path,
    *,
    lineage: str | None = None,
    through_episode: int | None = None,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for path in _records(root):
        value = read_submission(path)
        submission_id = str(value["submission_id"])
        if submission_id in seen:
            continue
        seen.add(submission_id)
        if lineage is not None and value["lineage"] != lineage:
            continue
        if through_episode is not None and value["episode"] > through_episode:
            continue
        result.append(value)
    result.sort(
        key=lambda value: (
            str(value.get("submitted_at") or ""),
            str(value["submission_id"]),
        )
    )
    return result


def accepted_submission_kernel_ids(
    root: Path,
    *,
    lineage: str | None = None,
    through_episode: int | None = None,
) -> set[str]:
    return {
        str(value["kernel_commit_id"])
        for value in submissions(
            root, lineage=lineage, through_episode=through_episode
        )
        if value["accepted"] is True
    }


def migrate_legacy_journals(
    journals_root: Path,
    submissions_root: Path,
    *,
    lineage: str,
) -> int:
    """Import old Episode submission rows once without reviving Journal authority."""
    migrated = 0
    for path in sorted((journals_root / "episodes").glob("e*/journal.json")):
        try:
            episode = int(path.parent.name[1:])
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
            continue
        rows = value.get("kernel_submissions", []) if isinstance(value, dict) else []
        if not isinstance(rows, list):
            continue
        for index, raw in enumerate(rows):
            if not isinstance(raw, Mapping):
                continue
            commit_id = raw.get("kernel_commit_id")
            if (
                not isinstance(commit_id, str)
                or COMMIT_ID_RE.fullmatch(commit_id) is None
            ):
                continue
            # Do not include the Campaign's absolute path. Campaign roots may
            # be moved or restored elsewhere; the same legacy row must still
            # map to the same immutable submission identity.
            seed = json.dumps(
                {
                    "lineage": lineage,
                    "episode": episode,
                    "index": index,
                    "submission": dict(raw),
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                default=str,
            ).encode("utf-8")
            submission_id = "submission-" + hashlib.sha256(seed).hexdigest()[:32]
            record = {
                "submission_id": submission_id,
                "lineage": lineage,
                "episode": episode,
                **dict(raw),
            }
            record["submission_id"] = submission_id
            record["kernel_commit_id"] = commit_id
            record["accepted"] = raw.get("accepted") is True
            existed = (
                submissions_root / submission_id / "submission.json"
            ).exists()
            append_submission(submissions_root, record)
            if not existed:
                migrated += 1
    return migrated
