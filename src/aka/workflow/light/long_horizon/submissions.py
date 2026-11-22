"""Campaign-wide versioned audit records for ``submit-kernel`` calls.

One stable Submission identity may contain multiple immutable revisions.  A
revision binds one exact Agent Git commit, its saved full-Evaluate evidence and
the final-source Review that inspected it.  Revisions never replace or revoke
an earlier accepted revision.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

from aka.experiment.measurement_store import private_write
from aka.workflow.light.host.errors import AgentRequestError, RuntimeStateError

COMMIT_ID_RE = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
SUBMISSION_ID_RE = re.compile(r"submission-[0-9a-f]{32}")
SUBMISSION_SCHEMA = "atrex.kernel-submission.v2"
SUBMISSION_VERSION_RE = re.compile(r"v([1-9][0-9]*)")


def _records(root: Path) -> Iterable[Path]:
    versioned = [
        path / "submission.json"
        for path in root.glob("submission-*/v*")
        if path.is_dir()
        and not path.is_symlink()
        and SUBMISSION_VERSION_RE.fullmatch(path.name) is not None
    ]
    current = [
        path / "submission.json"
        for path in root.glob("submission-*")
        if path.is_dir()
        and not path.is_symlink()
        and (path / "submission.json").is_file()
    ]
    # Flat JSON records are a read-only compatibility input. New writes always
    # co-locate the submission and its final-source Review in one directory.
    legacy = list(root.glob("submission-*.json"))
    return [*sorted(versioned), *sorted(current), *sorted(legacy)]


def submission_revision_root(root: Path, submission_id: str, revision: int) -> Path:
    """Return the private directory for one immutable Submission revision."""
    if SUBMISSION_ID_RE.fullmatch(submission_id) is None:
        raise ValueError("Kernel submission requires a valid submission_id")
    if type(revision) is not int or revision < 1:
        raise ValueError("Kernel submission revision must be a positive integer")
    return root / submission_id / f"v{revision}"


def read_submission(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeStateError(
            "A private submit-kernel audit record is unreadable."
        ) from error
    schema = value.get("schema") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or schema not in {"atrex.kernel-submission.v1", SUBMISSION_SCHEMA}
        or not isinstance(value.get("submission_id"), str)
        or SUBMISSION_ID_RE.fullmatch(value["submission_id"]) is None
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
    submission_id = str(value["submission_id"])
    if schema == SUBMISSION_SCHEMA:
        revision = value.get("revision")
        if (
            type(revision) is not int
            or revision < 1
            or path.name != "submission.json"
            or path.parent.name != f"v{revision}"
            or path.parent.parent.name != submission_id
        ):
            raise RuntimeStateError(
                "A private submit-kernel revision has an unsupported identity."
            )
    elif (
        path.stem != submission_id
        and path.parent.name != submission_id
    ):
        raise RuntimeStateError(
            "A private submit-kernel audit record has an unsupported identity."
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
    revision = record.get("revision")
    if type(revision) is not int or revision < 1:
        raise ValueError("Kernel submission requires a positive revision")
    # The schema is Runtime-owned. Do not allow caller-supplied data to weaken
    # the immutable version contract.
    record["schema"] = SUBMISSION_SCHEMA
    path = submission_revision_root(root, submission_id, revision) / "submission.json"
    if path.exists():
        existing = read_submission(path)
        if existing != record:
            raise RuntimeStateError(
                "Kernel submission identity already belongs to different audit data"
            )
        return existing
    private_write(path, record)
    return record


def submission_versions(root: Path, submission_id: str) -> list[dict[str, object]]:
    """Return every immutable revision for one stable Submission identity."""
    if SUBMISSION_ID_RE.fullmatch(submission_id) is None:
        raise ValueError("Kernel submission requires a valid submission_id")
    result = [
        value
        for value in submissions(root)
        if value.get("submission_id") == submission_id
    ]
    result.sort(key=lambda value: int(value.get("revision", 1)))
    return result


def next_submission_revision(
    root: Path,
    submission_id: str,
    *,
    lineage: str,
) -> tuple[int, dict[str, object] | None]:
    """Validate ownership and allocate the next logical revision number."""
    versions = submission_versions(root, submission_id)
    if not versions:
        return 1, None
    if any(value.get("lineage") != lineage for value in versions):
        raise AgentRequestError(
            "Kernel submission identity belongs to another lineage",
            code="submission_read_only",
            submission_id=submission_id,
            next_action=(
                "Create a new local Submission for your commit. Only the owner "
                "lineage may revise an existing Submission identity."
            ),
        )
    latest = versions[-1]
    return int(latest.get("revision", 1)) + 1, latest


def submission_for_commit(
    root: Path,
    *,
    lineage: str,
    kernel_commit_id: str,
    submission_id: str | None = None,
) -> dict[str, object] | None:
    """Find an existing exact-commit decision for idempotent submission."""
    matches = [
        value
        for value in submissions(root, lineage=lineage)
        if value.get("kernel_commit_id") == kernel_commit_id
        and (submission_id is None or value.get("submission_id") == submission_id)
    ]
    return matches[-1] if matches else None


def submissions(
    root: Path,
    *,
    lineage: str | None = None,
    through_episode: int | None = None,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    for path in _records(root):
        value = read_submission(path)
        submission_id = str(value["submission_id"])
        revision = int(value.get("revision", 1))
        identity = (submission_id, revision)
        if identity in seen:
            continue
        seen.add(identity)
        if lineage is not None and value["lineage"] != lineage:
            continue
        if through_episode is not None and value["episode"] > through_episode:
            continue
        result.append(value)
    result.sort(
        key=lambda value: (
            str(value.get("submitted_at") or ""),
            str(value["submission_id"]),
            int(value.get("revision", 1)),
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
                "revision": 1,
                "lineage": lineage,
                "episode": episode,
                **dict(raw),
            }
            record["submission_id"] = submission_id
            record["kernel_commit_id"] = commit_id
            record["accepted"] = raw.get("accepted") is True
            existed = (
                submission_revision_root(submissions_root, submission_id, 1)
                / "submission.json"
            ).exists()
            append_submission(submissions_root, record)
            if not existed:
                migrated += 1
    return migrated
