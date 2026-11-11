"""Supervisor-owned Direction, Experiment, and terminal-report Journal.

The Agent submits interpretation, intent, and explicit Gateway Record references.
The private Gateway store owns their exact Kernel bindings; the Journal does not
duplicate source identity or require a before/after comparison structure.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import inspect
import json
import math
import os
import re
import stat
import subprocess
import tempfile
import threading
import uuid
from collections import ChainMap
from collections.abc import Callable, Mapping, MutableMapping, Set as AbstractSet
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aka.light.agent.home import open_private_directory

from aka.light.workflow.direction_genealogy import (
    RELATIONSHIP_FIELDS,
    relationship_fields,
    validate_relationship,
)
from aka.light.workflow.direction_lifecycle import (CLOSED_STATUSES as _CLOSED_STATUSES)
from aka.light.workflow.direction_lifecycle import (CLOSURES as _CLOSURES)
from aka.light.workflow.direction_lifecycle import (DIRECTION_ACTIONS as _DIRECTION_ACTIONS)
from aka.light.workflow.direction_lifecycle import (DIRECTION_STATUS as _DIRECTION_STATUS)
from aka.light.workflow.direction_lifecycle import (DIRECTION_REVISION_ACTION as _DIRECTION_REVISION_ACTION)
from aka.light.workflow.direction_lifecycle import (advance_direction)
from aka.light.runtime.request_errors import AgentRequestError, RuntimeStateError, field_error, require_fields
from aka.light.experiment.identifiers import (
    DIRECTION_ID_RE,
    EXPERIMENT_ID_RE,
    GATEWAY_RECORD_ID_RE,
    KERNEL_RECORD_ID_RE,
)
from aka.light.experiment.store import (
    MeasurementStore,
    private_write,
    private_write_bytes,
    read_json,
)
from aka.light.evaluation.provider import result_from_response
from aka.light.workspace.files import (
    MAX_FILE_BYTES,
    publish,
    publish_if_changed,
    relative_path,
)
from aka.light.agent.session_tail import read_regular_bytes
from aka.light.research import ResearchStore

DIRECTION_PROPOSAL_FIELDS = {
    "action",
    "name",
    "change_scope",
    "objective",
    "applicability",
    "expected_benefit",
    "plan",
}
DIRECTION_PROPOSAL_OPTIONAL_FIELDS = {"stop_conditions"}
DIRECTION_REVISION_FIELDS = DIRECTION_PROPOSAL_FIELDS | {"direction_id"}
DIRECTION_CHANGE_SCOPES = {"incremental", "major_redesign"}
DIRECTION_UPDATE_FIELDS = {"action", "direction_id", "analysis"}
DIRECTION_CLOSURE_FIELDS = DIRECTION_UPDATE_FIELDS
DIRECTION_CLOSURE_OPTIONAL_FIELDS = {"reopen_condition", "supporting_experiment_ids"}
_EVIDENCE_KINDS = {"run", "same_allocation_abba", "profile", "dev", "check", "disassemble"}
EXPERIMENT_FIELDS = {"name", "intent", "interpretation"}
EXPERIMENT_OPTIONAL_FIELDS = {"experiment_id", "gateway_record_ids", "references"}
LEGACY_EXPERIMENT_FIELDS = {
    "name", "hypothesis", "change", "analysis",
}
JOURNAL_MODULES = frozenset({"directions", "experiments"})
HANDOFF_REASONS = frozenset({
    "direction_terminal",
    "cross_trajectory_share",
    "blocked",
})
_DIRECTION_PROJECTION_STATUSES = (
    "proposed",
    "in_progress",
    "deferred",
    "blocked",
    "completed",
    "abandoned",
)
MAX_EXPERIMENT_REFERENCE_PATHS = 32
MAX_EXPERIMENT_REFERENCE_FILES = 256
MAX_EXPERIMENT_REFERENCE_DIRECTORIES = 256
MAX_EXPERIMENT_REFERENCE_FILE_BYTES = 16 * 1024 * 1024
MAX_EXPERIMENT_REFERENCE_TOTAL_BYTES = 64 * 1024 * 1024
MAX_PROGRESS_FRONTIER_ROWS = 20

_REVIEW_SESSION_AUDIT_KEY = "_review_session_audit"
_ACTIVE_DIRECTION_EXPERIMENT_RUNTIME_ACTION = (
    "Keep the Direction active unless its objective, planned variants, and stop "
    "conditions are genuinely exhausted."
)


def _tool_modules(journal: Mapping[str, object]) -> tuple[str, ...]:
    """Return the immutable per-Episode Journal module selection."""
    raw = journal.get("tool_modules")
    if raw is None:
        return ("directions", "experiments") if journal.get("research_records", True) else ()
    if (
        not isinstance(raw, list)
        or any(not isinstance(item, str) for item in raw)
        or len(set(raw)) != len(raw)
        or set(raw) - JOURNAL_MODULES
    ):
        raise RuntimeStateError("The Runtime research-module policy is invalid.")
    return tuple(raw)


def _agent_claim_review(review: Mapping[str, object]) -> dict[str, object]:
    """Project decision feedback without exposing Supervisor review-bundle metadata."""
    return {
        key: deepcopy(value)
        for key, value in review.items()
        if key not in {"evidence_manifest", _REVIEW_SESSION_AUDIT_KEY}
    }


def _private_claim_review(review: Mapping[str, object]) -> dict[str, object]:
    """Return the private receipt with a readable name for Session audit metadata."""
    result = deepcopy(dict(review))
    session_audit = result.pop(_REVIEW_SESSION_AUDIT_KEY, None)
    if session_audit is not None:
        result["session_audit"] = session_audit
    return result


def _review_projection(review: Mapping[str, object] | None) -> dict[str, object]:
    """Keep only durable decision state from an independent claim review."""
    if not isinstance(review, Mapping):
        return {}
    quality = review.get("claim_review_quality")
    if quality is None:  # Read pre-v2 private receipts during migration.
        quality = review.get("verdict")
    if quality is None and review.get("status") == "unavailable":
        quality = "unreviewed"
    if quality is None:
        return {}
    projected: dict[str, object] = {"claim_review_quality": quality}
    for source, target in (
        ("scope_safe_interpretation", "scope_safe_interpretation"),
        ("scientific_next_check", "scientific_next_check"),
        ("next_check", "scientific_next_check"),  # legacy private receipt
    ):
        value = review.get(source)
        if isinstance(value, str) and value.strip():
            projected[target] = value.strip()
    return projected


def _durable_review_fields(value: Mapping[str, object]) -> dict[str, object]:
    """Canonicalize durable Review fields while reading pre-rename history."""
    result: dict[str, object] = {}
    quality = value.get("claim_review_quality", value.get("review_quality"))
    if quality is not None:
        result["claim_review_quality"] = quality
    safe = value.get(
        "scope_safe_interpretation", value.get("review_safe_interpretation")
    )
    if safe is not None:
        result["scope_safe_interpretation"] = safe
    next_check = value.get(
        "scientific_next_check",
        value.get("review_scientific_next_check", value.get("review_next_check")),
    )
    if next_check is not None:
        result["scientific_next_check"] = next_check
    return result


def _claim_review_session_audit(value: object) -> dict[str, object] | None:
    """Validate bounded Supervisor-owned links to captured reviewer conversations."""
    if value is None:
        return None
    if not isinstance(value, Mapping) or value.get("schema") != (
        "atrex.research-claim-review-session.v1"
    ):
        raise ValueError("research review Session audit metadata is invalid")
    attempts = value.get("attempts")
    if not isinstance(attempts, list) or not 1 <= len(attempts) <= 4:
        raise ValueError("research review Session audit attempts are invalid")
    normalized = []
    allowed_statuses = {"accepted", "invalid", "capture_missing"}
    for index, row in enumerate(attempts):
        if not isinstance(row, Mapping):
            raise ValueError(f"research review Session attempt {index} is invalid")
        attempt = row.get("attempt")
        session_id = row.get("session_id")
        status = row.get("status")
        conversation_path = row.get("conversation_path")
        if type(attempt) is not int or attempt < 1 or attempt > 4:
            raise ValueError(f"research review Session attempt {index} number is invalid")
        if not isinstance(session_id, str) or len(session_id) > 512:
            raise ValueError(f"research review Session attempt {index} id is invalid")
        if status not in allowed_statuses:
            raise ValueError(f"research review Session attempt {index} status is invalid")
        if conversation_path is not None:
            if (
                not isinstance(conversation_path, str)
                or not conversation_path
                or len(conversation_path) > 2048
                or conversation_path.startswith("/")
                or ".." in conversation_path.split("/")
                or not conversation_path.endswith("/conversation.jsonl")
            ):
                raise ValueError(
                    f"research review Session attempt {index} conversation path is invalid"
                )
        error = row.get("error")
        if error is not None and (
            not isinstance(error, Mapping)
            or not isinstance(error.get("type"), str)
            or not isinstance(error.get("message"), str)
            or len(error["type"]) > 256
            or len(error["message"]) > 2000
        ):
            raise ValueError(f"research review Session attempt {index} error is invalid")
        normalized_row: dict[str, object] = {
            "attempt": attempt,
            "session_id": session_id,
            "status": status,
            "exit_status": row.get("exit_status"),
            "timed_out": bool(row.get("timed_out", False)),
            "conversation_path": conversation_path,
            "resumed": bool(row.get("resumed", False)),
        }
        if error is not None:
            normalized_row["error"] = {
                "type": error["type"],
                "message": error["message"],
            }
        normalized.append(normalized_row)
    return {
        "schema": "atrex.research-claim-review-session.v1",
        "attempts": normalized,
    }


def _module_enabled(journal: Mapping[str, object], module: str) -> bool:
    return module in _tool_modules(journal)


class _HistoryIndex:
    """Process-local projection of finalized, immutable Episode Journals."""

    def __init__(self) -> None:
        self.next_episode = 0
        self.shared_revision: str | None = None
        self.shared_direction_ids: set[str] = set()
        self.shared_experiment_ids: set[str] = set()
        self.directions: dict[str, dict[str, Any]] = {}
        self.statuses: dict[str, str] = {}
        self.experiments: list[dict[str, Any]] = []
        self.experiments_by_id: dict[str, dict[str, Any]] = {}
        self.experiment_ids: set[str] = set()
        self.pending: set[int] = set()

    def fork(self) -> "_HistoryIndex":
        """Return an unpublished copy that can ingest the next Episode safely."""
        result = _HistoryIndex()
        result.next_episode = self.next_episode
        result.shared_revision = self.shared_revision
        result.shared_direction_ids = set(self.shared_direction_ids)
        result.shared_experiment_ids = set(self.shared_experiment_ids)
        # Direction events and Experiment association update nested lists, so
        # these views require a deep copy. Existing Experiment rows are
        # immutable after finalization and can be shared between snapshots.
        result.directions = deepcopy(self.directions)
        result.statuses = dict(self.statuses)
        result.experiments = list(self.experiments)
        result.experiments_by_id = dict(self.experiments_by_id)
        result.experiment_ids = set(self.experiment_ids)
        result.pending = set(self.pending)
        return result


_HISTORY_LOCK = threading.RLock()
_HISTORY_INDEXES: dict[Path, _HistoryIndex] = {}


@dataclass(frozen=True)
class _VisibleHistory:
    current: dict[str, Any]
    directions: dict[str, dict[str, Any]]
    statuses: dict[str, str]
    experiments: list[dict[str, Any]]
    experiments_by_id: dict[str, dict[str, Any]]
    shared_direction_ids: frozenset[str]
    shared_experiment_ids: frozenset[str]



def _now() -> str:
    return datetime.now(UTC).isoformat()


@contextmanager
def journal_lock(path: Path):
    """Serialize processes as well as Sessions; no blocking global Runtime lock."""
    with open_private_directory(path.parent) as parent:
        descriptor = os.open(
            "journal.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600, dir_fd=parent
        )
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise AgentRequestError(
                    "Another controller is updating this Episode's private Runtime state; "
                    "this request made no changes",
                    code="journal_busy",
                    next_action=(
                        "Wait briefly and retry with backoff; do not edit projected research files"
                    ),
                ) from error
            yield
        finally:
            os.close(descriptor)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 16000:
        field = {"Direction ID": "direction_id", "Experiment Direction ID": "direction_id",
                 "Experiment ID": "experiment_id"}.get(label, label.split()[-1])
        raise field_error(field, f"{label} must be non-empty text of at most 16000 characters")
    return value.strip()


def _text_list(value: object, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or len(value) > 64
        or any(not isinstance(item, str) or not item.strip() or len(item) > 4000 for item in value)
    ):
        raise field_error(label.split()[-1],
            f"{label} must contain at most 64 non-empty strings, "
            "each at most 4000 characters"
        )
    return [item.strip() for item in value]


def _change_scope(value: object) -> str:
    if value == "local":  # Import old journals without exposing the ambiguous term.
        value = "incremental"
    if not isinstance(value, str) or value not in DIRECTION_CHANGE_SCOPES:
        raise field_error(
            "change_scope",
            "Direction change_scope must be incremental or major_redesign",
        )
    return value


def initialize_journal(
    path: Path,
    *,
    episode: int,
    base_commit: str,
    branch: str,
    memory_version: int | None = None,
    tool_modules: tuple[str, ...] | list[str] = ("directions", "experiments"),
) -> dict[str, Any]:
    """Create one private Runtime-managed Episode Journal."""
    modules = tuple(tool_modules)
    if len(set(modules)) != len(modules) or set(modules) - JOURNAL_MODULES:
        raise ValueError("tool_modules may contain only unique directions and experiments")
    value: dict[str, Any] = {
        "schema_version": 2,
        "runtime_managed": True,
        "research_records": bool(modules),
        "tool_modules": list(modules),
        "episode": episode,
        "memory_version": memory_version,
        "base_commit": base_commit,
        # Public identity for the same source in Agent-owned Git. The
        # controller commit above remains private archival state.
        "base_kernel_id": None,
        "episode_branch": branch,
        "state": "in_progress",
        "direction_events": [],
        "experiments": [],
        "outcome": None,
        "candidate_commit": None,
        # Private recovery state only. When an evidence-complete
        # candidate_ready report cannot be copied into the Supervisor archive,
        # keep the Episode open and prevent a fallback blocked/pivot report from
        # permanently discarding that candidate.
        "pending_candidate_archive": None,
        "created_at": _now(),
        "finalized_at": None,
    }
    private_write(path, value)
    return value


def load_journal(path: Path) -> dict[str, Any]:
    try:
        value = read_json(path, limit=MAX_FILE_BYTES)
    except (OSError, ValueError) as exc:
        raise RuntimeStateError("The private Runtime Episode state is missing or unreadable.") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 2
        or value.get("runtime_managed") is not True
    ):
        raise RuntimeStateError("The private Runtime Episode state has an unsupported schema.")
    if not isinstance(value.get("direction_events"), list) or not isinstance(
        value.get("experiments"), list
    ):
        raise RuntimeStateError("The private Runtime Episode state has invalid collections.")
    if type(value.get("research_records", True)) is not bool:
        raise RuntimeStateError("The private Runtime Episode state has an invalid research policy.")
    base_kernel_id = value.get("base_kernel_id")
    if base_kernel_id is not None and (
        not isinstance(base_kernel_id, str)
        or KERNEL_RECORD_ID_RE.fullmatch(base_kernel_id) is None
    ):
        raise RuntimeStateError(
            "The private Runtime Episode state has an invalid Agent baseline identity."
        )
    modules = _tool_modules(value)
    if bool(modules) != value.get("research_records", True):
        raise RuntimeStateError(
            "The private Runtime Episode state's research policy disagrees with tool_modules."
        )
    if "directions" not in modules and value["direction_events"]:
        raise RuntimeStateError(
            "Direction-disabled Runtime Episode state contains unexpected Direction records."
        )
    if "experiments" not in modules and value["experiments"]:
        raise RuntimeStateError(
            "Experiment-disabled Runtime Episode state contains unexpected Experiment records."
        )
    if not modules and (value["direction_events"] or value["experiments"]):
        raise RuntimeStateError(
            "Research-disabled Runtime Episode state contains unexpected research records."
        )
    return value


def _index_historical_journal(index: _HistoryIndex, path: Path, episode: int) -> None:
    try:
        journal = load_journal(path)
    except (ValueError, RuntimeStateError) as error:
        raise RuntimeStateError(
            "Historical private Runtime Episode state is unreadable; restore it before continuing."
        ) from error
    if journal["episode"] != episode:
        raise RuntimeStateError(
            "Historical private Runtime state has a mismatched Episode identity"
        )
    if not journal.get("finalized_at"):
        index.pending.add(episode)
        return
    # Shared Epoch Directions belong to peer trajectories. A peer may keep its
    # Direction active across Episodes, but that read-only lease is not a local
    # active Direction while private historical Journals are replayed. Preserve the published peer status in
    # ``index.statuses``; ignore it only in the lifecycle validation copy.
    _apply_direction_events(
        index.directions,
        index.statuses,
        journal,
        ignored_active_ids=frozenset(index.shared_direction_ids),
    )
    for experiment in journal["experiments"]:
        if not isinstance(experiment, dict):
            continue
        experiment_id = experiment.get("experiment_id")
        if not isinstance(experiment_id, str):
            continue
        previous = index.experiments_by_id.get(experiment_id)
        if previous is not None:
            # Shared peer Experiments are immutable in this trajectory. Local
            # Experiments may publish a newer reviewed revision in a later Episode.
            if experiment_id in index.shared_experiment_ids:
                continue
            if _experiment_revision(experiment) <= _experiment_revision(previous):
                continue
            indexed = dict(experiment)
            index.experiments_by_id[experiment_id] = indexed
            index.experiments = [
                indexed if item.get("experiment_id") == experiment_id else item
                for item in index.experiments
            ]
            continue
        index.experiment_ids.add(experiment_id)
        indexed = dict(experiment)
        index.experiments.append(indexed)
        index.experiments_by_id[experiment_id] = indexed


def shared_research_path(campaign_root: Path) -> Path:
    """Private, controller-published research visible after an Epoch barrier."""
    return campaign_root / "shared" / "research.json"


def _shared_research(campaign_root: Path) -> dict[str, Any]:
    path = shared_research_path(campaign_root)
    try:
        value = read_json(path, limit=MAX_FILE_BYTES)
    except FileNotFoundError:
        return {
            "schema_version": 1,
            "revision": "",
            "through_epoch": 0,
            "directions": {},
            "statuses": {},
            "experiments": [],
            "experiments_by_id": {},
            "research_queue": [],
        }
    except (OSError, ValueError) as error:
        raise RuntimeStateError("Shared Epoch research evidence is unreadable.") from error
    if (
        value.get("schema_version") != 1
        or not isinstance(value.get("revision"), str)
        or type(value.get("through_epoch")) is not int
        or value["through_epoch"] < 0
        or not isinstance(value.get("directions"), dict)
        or not isinstance(value.get("experiments"), list)
        or not isinstance(value.get("research_queue", []), list)
    ):
        raise RuntimeStateError("Shared Epoch research evidence has an unsupported schema.")
    directions: dict[str, dict[str, Any]] = {}
    statuses: dict[str, str] = {}
    for direction_id, raw in value["directions"].items():
        if (
            not isinstance(direction_id, str)
            or DIRECTION_ID_RE.fullmatch(direction_id) is None
            or not isinstance(raw, dict)
            or raw.get("direction_id") != direction_id
            or raw.get("status") not in set(_DIRECTION_STATUS.values())
        ):
            raise RuntimeStateError("Shared Epoch Direction evidence is invalid.")
        item = deepcopy(raw)
        if "objective" not in item and isinstance(item.get("hypothesis"), str):
            item["objective"] = item["hypothesis"]
        item.pop("hypothesis", None)
        item.pop("hypothesis_status", None)
        directions[direction_id] = item
        statuses[direction_id] = raw["status"]
    experiments: list[dict[str, Any]] = []
    experiments_by_id: dict[str, dict[str, Any]] = {}
    for raw in value["experiments"]:
        experiment_id = raw.get("experiment_id") if isinstance(raw, dict) else None
        if (
            not isinstance(experiment_id, str)
            or EXPERIMENT_ID_RE.fullmatch(experiment_id) is None
            or experiment_id in experiments_by_id
        ):
            raise RuntimeStateError("Shared Epoch Experiment evidence is invalid.")
        item = deepcopy(raw)
        experiments.append(item)
        experiments_by_id[experiment_id] = item
    return {
        **value,
        "directions": directions,
        "statuses": statuses,
        "experiments": experiments,
        "experiments_by_id": experiments_by_id,
    }


def shared_research_queue(campaign_root: Path) -> list[str]:
    value = _shared_research(campaign_root)
    queue = value.get("research_queue", [])
    if (
        not isinstance(queue, list)
        or len(set(queue)) != len(queue)
        or any(not isinstance(item, str) or DIRECTION_ID_RE.fullmatch(item) is None for item in queue)
    ):
        raise RuntimeStateError("Shared Epoch Direction queue is invalid.")
    return list(queue)


def research_projection(current: Path, campaign_root: Path) -> dict[str, Any]:
    """Export the complete visible research view for an Epoch barrier."""
    history = _visible_history(current, campaign_root)
    if not history.current.get("finalized_at"):
        raise RuntimeStateError("Only finalized Runtime Episode state may cross an Epoch barrier.")
    directions = {
        key: deepcopy(value)
        for key, value in _direction_views(current, campaign_root, history=history).items()
    }
    experiments = _visible_experiments(current, campaign_root, history=history)
    return {"directions": directions, "experiments": experiments}


def _visible_history(current: Path, campaign_root: Path) -> _VisibleHistory:
    """Replay finalized history once, then ingest only newly completed Episodes.

    A restarted Supervisor rebuilds the projection from private Journals. During
    a running Campaign finalized Journals are immutable; the current Journal is
    still read on every request. An older unfinished Episode is rechecked and
    forces a rebuild if it is finalized later than expected.
    """
    current_value = load_journal(current)
    episode = current_value.get("episode")
    if not isinstance(episode, int) or isinstance(episode, bool) or episode < 0:
        raise RuntimeStateError("Private Runtime state has an invalid Episode identity")
    root = campaign_root.resolve()
    shared = _shared_research(root)
    with _HISTORY_LOCK:
        index = _HISTORY_INDEXES.get(root)
        # A straggling old-Episode binding may still issue a read while a newer
        # Episode owns the Campaign. Build that older view privately; never
        # replace the newest published Campaign snapshot with an older one.
        older_snapshot = index is not None and episode < index.next_episode
        rebuild = (
            index is None
            or older_snapshot
            or index.shared_revision != shared["revision"]
        )
        if index is not None and not rebuild:
            for pending in index.pending:
                path = root / "episodes" / f"e{pending:08d}" / "journal.json"
                try:
                    pending_journal = load_journal(path)
                except RuntimeStateError:
                    # Rebuild via the directory index. A deleted abandoned
                    # Episode disappears as it did in the former glob-based
                    # implementation; an existing corrupt file still fails
                    # closed when _index_historical_journal reads it below.
                    rebuild = True
                    break
                if pending_journal.get("finalized_at"):
                    rebuild = True
                    break
        if rebuild:
            rebuilt = _HistoryIndex()
            rebuilt.shared_revision = shared["revision"]
            rebuilt.shared_direction_ids = set(shared["directions"])
            rebuilt.shared_experiment_ids = set(shared["experiments_by_id"])
            rebuilt.directions = deepcopy(shared["directions"])
            rebuilt.statuses = dict(shared["statuses"])
            rebuilt.experiments = list(shared["experiments"])
            rebuilt.experiments_by_id = dict(shared["experiments_by_id"])
            rebuilt.experiment_ids = set(shared["experiments_by_id"])
            for path in sorted((root / "episodes").glob("e*/journal.json")):
                identity = re.fullmatch(r"e([0-9]{8,})", path.parent.name)
                if identity is not None and int(identity[1]) < episode:
                    _index_historical_journal(rebuilt, path, int(identity[1]))
            rebuilt.next_episode = episode
            index = rebuilt
            if not older_snapshot:
                _HISTORY_INDEXES[root] = index
                if len(_HISTORY_INDEXES) > 16:
                    _HISTORY_INDEXES.pop(next(iter(_HISTORY_INDEXES)))
        elif episode > index.next_episode:
            # Published indexes are immutable. Build the next snapshot in
            # private, then replace the cache entry atomically while holding
            # _HISTORY_LOCK. Readers that already hold the old snapshot can
            # continue iterating it without observing partial ingestion.
            updated = index.fork()
            # Sequential Episode numbers avoid re-globbing the entire Campaign.
            # A sparse jump falls back to one directory scan, not an unbounded range.
            if episode - index.next_episode > 4096:
                paths = sorted((root / "episodes").glob("e*/journal.json"))
                numbered = (
                    (path, re.fullmatch(r"e([0-9]{8,})", path.parent.name))
                    for path in paths
                )
                for path, identity in numbered:
                    if identity is not None and index.next_episode <= int(identity[1]) < episode:
                        _index_historical_journal(updated, path, int(identity[1]))
            else:
                for previous in range(index.next_episode, episode):
                    path = root / "episodes" / f"e{previous:08d}" / "journal.json"
                    if path.exists() or path.is_symlink():
                        _index_historical_journal(updated, path, previous)
            updated.next_episode = episode
            _HISTORY_INDEXES[root] = updated
            index = updated
        return _VisibleHistory(
            current=current_value,
            directions=index.directions,
            statuses=index.statuses,
            experiments=index.experiments,
            experiments_by_id=index.experiments_by_id,
            shared_direction_ids=frozenset(index.shared_direction_ids),
            shared_experiment_ids=frozenset(index.shared_experiment_ids),
        )


def _editable_direction(
    views: MutableMapping[str, dict[str, Any]], direction_id: str
) -> dict[str, Any]:
    if isinstance(views, ChainMap) and direction_id not in views.maps[0]:
        views[direction_id] = deepcopy(views[direction_id])
    return views[direction_id]


def _apply_direction_events(
    views: MutableMapping[str, dict[str, Any]], statuses: MutableMapping[str, str],
    journal: dict[str, Any],
    *,
    ignored_active_ids: set[str] | frozenset[str] = frozenset(),
) -> None:
    lifecycle_statuses = dict(statuses)
    for direction_id in ignored_active_ids:
        if lifecycle_statuses.get(direction_id) == "in_progress":
            lifecycle_statuses[direction_id] = "deferred"
    for event in journal["direction_events"]:
        if not isinstance(event, dict):
            raise RuntimeStateError("Private Runtime Episode state contains an invalid Direction event")
        direction_id = event.get("direction_id")
        action = event.get("action")
        try:
            advance_direction(lifecycle_statuses, direction_id, action)
            statuses[direction_id] = lifecycle_statuses[direction_id]
            if action in {"propose", _DIRECTION_REVISION_ACTION}:
                for field in ("name", "recorded_at"):
                    _text(event.get(field), f"Direction {field}")
                objective = event.get("objective", event.get("hypothesis"))
                _text(objective, "Direction objective")
                scope = event.get("change_scope", "unspecified")
                if scope != "unspecified":
                    scope = _change_scope(scope)
                _text_list(event.get("plan"), "Direction plan")
                _text_list(event.get("stop_conditions", []), "Direction stop_conditions")
                # Required for new proposals; optional while replaying older
                # finalized Journals so Runtime does not invent prior claims.
                if "applicability" in event:
                    _text(event.get("applicability"), "Direction applicability")
                if "expected_benefit" in event:
                    _text(event.get("expected_benefit"), "Direction expected_benefit")
                # Keep finalized Journals from the former, more repetitive
                # proposal contract readable without requiring new proposals
                # to reproduce the same claim in three fields.
                if "rationale" in event:
                    _text(event.get("rationale"), "Direction rationale")
                if "success_criteria" in event:
                    _text_list(event.get("success_criteria"), "Direction success_criteria")
                revision = event.get("revision", 1)
                if type(revision) is not int or revision < 1:
                    raise ValueError("Direction revision must be a positive integer")
            _text_list(event.get("supporting_experiment_ids", []), "supporting_experiment_ids")
            _text_list(event.get("gateway_record_ids", []), "gateway_record_ids")
        except ValueError as error:
            raise RuntimeStateError(
                f"Invalid Direction history in Episode {journal['episode']}: {error}"
            ) from error
        if action == "propose":
            direction = {
                "direction_id": direction_id,
                "revision": event.get("revision", 1),
                "name": event["name"],
                "change_scope": scope,
                "objective": objective,
                **(
                    {"applicability": event["applicability"]}
                    if "applicability" in event else {}
                ),
                **(
                    {"expected_benefit": event["expected_benefit"]}
                    if "expected_benefit" in event else {}
                ),
                "plan": event["plan"],
                "stop_conditions": event.get("stop_conditions", []),
                "status": "proposed",
                "analysis": None,
                "recorded_at": event["recorded_at"],
                "supporting_experiment_ids": [],
                "gateway_record_ids": list(event.get("gateway_record_ids", [])),
                "reopen_condition": None,
                **_durable_review_fields(event),
                **relationship_fields(event),
            }
            # Legacy fields remain visible on legacy Directions; they are not
            # invented for the compact contract.
            for field in ("rationale", "success_criteria"):
                if field in event:
                    direction[field] = event[field]
            views[direction_id] = direction
        elif action == _DIRECTION_REVISION_ACTION and direction_id in views:
            direction = _editable_direction(views, direction_id)
            for field in (
                "name",
                "objective",
                "applicability",
                "expected_benefit",
                "plan",
                "stop_conditions",
            ):
                direction[field] = deepcopy(
                    event.get("objective", event.get("hypothesis"))
                    if field == "objective" else event[field]
                )
            direction["change_scope"] = _change_scope(event["change_scope"])
            direction["revision"] = event["revision"]
            direction["updated_at"] = event["recorded_at"]
            if "gateway_record_ids" in event:
                direction["gateway_record_ids"] = list(
                    event["gateway_record_ids"]
                )
            direction.update(_durable_review_fields(event))
        elif direction_id in views:
            direction = _editable_direction(views, direction_id)
            direction["status"] = _DIRECTION_STATUS[action]
            direction["analysis"] = event.get("analysis")
            direction["updated_at"] = event.get("recorded_at")
            direction["reopen_condition"] = event.get("reopen_condition")
            support = event.get("supporting_experiment_ids", [])
            direction["supporting_experiment_ids"] = (
                list(support) if action in _CLOSURES else []
            )
            if action in _CLOSURES and "gateway_record_ids" in event:
                direction["gateway_record_ids"] = list(event["gateway_record_ids"])
            direction.update(_durable_review_fields(event))


def _direction_state(
    current: Path, campaign_root: Path, *, history: _VisibleHistory | None = None
) -> tuple[ChainMap[str, dict[str, Any]], ChainMap[str, str]]:
    if history is None:
        history = _visible_history(current, campaign_root)
    views: ChainMap[str, dict[str, Any]] = ChainMap({}, history.directions)
    statuses: ChainMap[str, str] = ChainMap({}, history.statuses)
    _apply_direction_events(
        views,
        statuses,
        history.current,
        ignored_active_ids=frozenset(history.shared_direction_ids),
    )
    return views, statuses


def _direction_views(
    current: Path, campaign_root: Path, *, history: _VisibleHistory | None = None
) -> Mapping[str, dict[str, Any]]:
    return _direction_state(current, campaign_root, history=history)[0]


def _visible_experiments(
    current: Path, campaign_root: Path, *, history: _VisibleHistory | None = None
) -> list[dict[str, Any]]:
    if history is None:
        history = _visible_history(current, campaign_root)
    # Cached history snapshots live for the Supervisor process lifetime. Keep
    # their nested Experiment payloads private from callers just as Direction
    # loads do; list construction is already proportional to the output size.
    values = [deepcopy(item) for item in history.experiments]
    positions = {item["experiment_id"]: index for index, item in enumerate(values)}
    for experiment in history.current["experiments"]:
        if not isinstance(experiment, dict):
            continue
        experiment_id = experiment.get("experiment_id")
        if not isinstance(experiment_id, str):
            continue
        item = deepcopy(experiment)
        if experiment_id in positions:
            values[positions[experiment_id]] = item
        else:
            positions[experiment_id] = len(values)
            values.append(item)
    return values


def _public_direction(direction: Mapping[str, object]) -> dict[str, Any]:
    """Return the complete decision-useful Direction workspace projection."""
    visible = deepcopy(dict(direction))
    if "objective" not in visible and isinstance(visible.get("hypothesis"), str):
        # Read old finalized Journals without preserving the retired
        # single-hypothesis contract in the Agent workspace.
        visible["objective"] = visible["hypothesis"]
    visible.pop("hypothesis", None)
    visible.pop("hypothesis_status", None)
    visible.update(_durable_review_fields(visible))
    visible.pop("review_quality", None)
    visible.pop("review_next_check", None)
    visible.pop("review_safe_interpretation", None)
    visible.pop("review_scientific_next_check", None)
    for field in ("recorded_at", "updated_at", "origin", "associated_experiment_ids"):
        visible.pop(field, None)
    return visible


def _public_experiment(experiment: Mapping[str, object]) -> dict[str, Any]:
    """Return the complete decision-useful Experiment workspace projection."""
    visible = deepcopy(dict(experiment))
    visible.update(_durable_review_fields(visible))
    visible.pop("review_quality", None)
    visible.pop("review_next_check", None)
    visible.pop("review_safe_interpretation", None)
    visible.pop("review_scientific_next_check", None)
    for field in (
        "sequence",
        "recorded_at",
        "before_kernel_id",
        "after_kernel_id",
        "source_diff_summary",
        "operations",
        "correctness",
        "latency_summary",
        "per_regime_summary",
        "failure_class",
        "measurement_digests",
        "reference_revision",
    ):
        visible.pop(field, None)
    # Older Journals may contain the former duplicated summary. Gateway
    # Records remain the only public source of measurement facts.
    visible.pop("evidence", None)
    visible.pop("action", None)
    visible.pop("origin", None)
    for row in visible.get("wiki_usage", []):
        if isinstance(row, dict):
            row.pop("use", None)
            row.pop("evidence", None)
    return visible


def _experiment_revision(experiment: Mapping[str, object]) -> int:
    revision = experiment.get("revision", 1)
    if type(revision) is not int or revision < 1:
        raise RuntimeStateError("Experiment has an invalid revision")
    return revision


def _projection_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode()


def _public_gateway_record(store: MeasurementStore, record_id: str) -> dict[str, Any]:
    """Return one standardized Agent-visible measurement projection."""
    from aka.light.evaluation.public_result import seal_response

    record = store.read(record_id)
    return seal_response(
        record["response"],
        record["request"],
        record_id=record_id,
        kernels=record["kernels"],
    )["agent_result"]


def _git_visible_record_ids(
    store: MeasurementStore,
    workspace: Path,
    *,
    commit_reachable: Callable[[str], bool] | None = None,
) -> set[str]:
    """Expose records only when every referenced Kernel is exact, reachable Git."""
    from aka.light.workspace.agent_git import AgentGit
    from aka.light.workspace.sources import from_git

    git = AgentGit(workspace, workspace / ".git" / "unused-runtime-bundle")
    commit_reachable = commit_reachable or git.commit_reachable
    visible: set[str] = set()
    checked: dict[str, bool] = {}
    for record_id in store.record_ids():
        try:
            record = store.read(record_id)
            if record.get("request", {}).get("visibility") == "supervisor_private_v0":
                continue
            kernels = record.get("kernels")
            if not isinstance(kernels, Mapping) or not kernels:
                continue
            valid = True
            for identity in kernels.values():
                commit_id = identity.get("kernel_id") if isinstance(identity, Mapping) else None
                if not isinstance(commit_id, str):
                    valid = False
                    break
                if commit_id not in checked:
                    checked[commit_id] = bool(
                        commit_reachable(commit_id)
                        and from_git(workspace, commit_id) == store.read_kernel(commit_id)
                    )
                if not checked[commit_id]:
                    valid = False
                    break
            if valid:
                visible.add(record_id)
        except (OSError, UnicodeError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
            # Historical private snapshots remain available to Supervisor
            # recovery but are not projected without their original Git object.
            continue
    return visible


def _git_reachable_projection(
    value: object,
    commit_reachable: Callable[[str], bool],
) -> object:
    """Drop Kernel identities that are not addressable in Agent-visible Git.

    Private Measurement/Research stores may retain old source snapshots for
    Supervisor recovery.  A normal-file projection must never advertise one
    of those snapshots as a usable Kernel commit unless the corresponding Git
    object is reachable from HEAD or an imported local/shared Runtime ref.
    """
    if isinstance(value, Mapping):
        projected: dict[str, object] = {}
        for key, item in value.items():
            is_kernel_commit = (
                key in {"kernel_id", "kernel_commit_id"}
                or key.endswith("_kernel_id")
                or key.endswith("_kernel_commit_id")
            )
            if is_kernel_commit:
                if item is None or (
                    isinstance(item, str) and commit_reachable(item)
                ):
                    projected[key] = item
                continue
            if key in {"kernel_ids", "kernel_commit_ids", "winner_kernel_commit_ids"}:
                if isinstance(item, list):
                    projected[key] = [
                        commit_id
                        for commit_id in item
                        if isinstance(commit_id, str) and commit_reachable(commit_id)
                    ]
                elif isinstance(item, Mapping):
                    projected[key] = {
                        side: commit_id
                        for side, commit_id in item.items()
                        if isinstance(commit_id, str) and commit_reachable(commit_id)
                    }
                continue
            projected[key] = _git_reachable_projection(item, commit_reachable)
        return projected
    if isinstance(value, list):
        return [_git_reachable_projection(item, commit_reachable) for item in value]
    return deepcopy(value)


def _experiment_reference_paths(raw: object, *, allow_empty: bool = False) -> list[Path]:
    if not isinstance(raw, list):
        raise field_error(
            "references",
            "Experiment references must be an array of workspace-relative file or directory paths.",
            code="reference_list_invalid",
        )
    minimum = 0 if allow_empty else 1
    if not minimum <= len(raw) <= MAX_EXPERIMENT_REFERENCE_PATHS:
        raise field_error(
            "references",
            (
                f"Experiment references may contain at most {MAX_EXPERIMENT_REFERENCE_PATHS} paths."
                if allow_empty
                else f"Experiment references must contain 1-{MAX_EXPERIMENT_REFERENCE_PATHS} paths."
            ),
            code="reference_list_invalid",
        )
    paths: list[Path] = []
    for index, value in enumerate(raw):
        if not isinstance(value, str) or len(value) > 1024:
            raise field_error(
                f"references.{index}",
                "Reference must be a workspace-relative path of at most 1024 characters.",
                code="reference_path_invalid",
            )
        try:
            path = relative_path(value)
        except ValueError as error:
            raise field_error(
                f"references.{index}",
                str(error),
                code="reference_path_invalid",
            ) from error
        if path in paths:
            raise field_error(
                f"references.{index}",
                "Experiment references must not contain duplicate paths.",
                code="reference_path_duplicate",
            )
        paths.append(path)
    for index, path in enumerate(paths):
        for other_index, other in enumerate(paths):
            if index >= other_index:
                continue
            if path in other.parents or other in path.parents:
                raise field_error(
                    f"references.{other_index}",
                    "Experiment references must not overlap; cite the containing directory once.",
                    code="reference_path_overlap",
                )
    return paths


def _capture_experiment_references(
    workspace: Path, raw: object
) -> tuple[list[Path], set[str], dict[str, bytes]]:
    """Capture a bounded, no-follow snapshot of Agent-selected material."""
    references = _experiment_reference_paths(raw)
    directories: set[str] = set()
    files: dict[str, bytes] = {}
    total_bytes = 0

    def limit_error(path: str, message: str) -> AgentRequestError:
        return field_error(
            "references",
            f"Reference {path!r} {message}",
            code="reference_archive_limit_exceeded",
            next_action=(
                "Reduce the referenced material, keep only files needed to support the Experiment, "
                "then resubmit record-experiment with the same Gateway Record IDs."
            ),
        )

    def read_file(parent: int, name: str, archive_path: str) -> None:
        nonlocal total_bytes
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=parent,
        )
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise field_error(
                    "references",
                    f"Reference {archive_path!r} is not a regular file.",
                    code="reference_type_invalid",
                )
            chunks = bytearray()
            while len(chunks) <= MAX_EXPERIMENT_REFERENCE_FILE_BYTES:
                chunk = os.read(
                    descriptor,
                    min(1024 * 1024, MAX_EXPERIMENT_REFERENCE_FILE_BYTES + 1 - len(chunks)),
                )
                if not chunk:
                    break
                chunks.extend(chunk)
            after = os.fstat(descriptor)
            if (
                before.st_dev != after.st_dev
                or before.st_ino != after.st_ino
                or before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
            ):
                raise field_error(
                    "references",
                    f"Reference {archive_path!r} changed while Runtime was archiving it.",
                    code="reference_changed",
                    next_action=(
                        "Wait for the file writer to finish, then resubmit record-experiment with "
                        "the same Gateway Record IDs."
                    ),
                )
            data = bytes(chunks)
            if len(data) > MAX_EXPERIMENT_REFERENCE_FILE_BYTES:
                raise limit_error(
                    archive_path,
                    f"exceeds the {MAX_EXPERIMENT_REFERENCE_FILE_BYTES}-byte per-file limit.",
                )
            if len(files) >= MAX_EXPERIMENT_REFERENCE_FILES:
                raise limit_error(
                    archive_path,
                    f"would exceed the {MAX_EXPERIMENT_REFERENCE_FILES}-file archive limit.",
                )
            if total_bytes + len(data) > MAX_EXPERIMENT_REFERENCE_TOTAL_BYTES:
                raise limit_error(
                    archive_path,
                    f"would exceed the {MAX_EXPERIMENT_REFERENCE_TOTAL_BYTES}-byte total limit.",
                )
            files[archive_path] = data
            total_bytes += len(data)
        finally:
            os.close(descriptor)

    def walk(directory: int, prefix: str) -> None:
        if prefix not in directories and len(directories) >= MAX_EXPERIMENT_REFERENCE_DIRECTORIES:
            raise limit_error(
                prefix,
                f"would exceed the {MAX_EXPERIMENT_REFERENCE_DIRECTORIES}-directory archive limit.",
            )
        directories.add(prefix)
        for name in sorted(os.listdir(directory)):
            child_path = f"{prefix}/{name}" if prefix else name
            info = os.stat(name, dir_fd=directory, follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode):
                raise field_error(
                    "references",
                    f"Reference {child_path!r} is a symbolic link; links cannot be archived.",
                    code="reference_symlink_forbidden",
                )
            if stat.S_ISDIR(info.st_mode):
                child = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory,
                )
                try:
                    walk(child, child_path)
                finally:
                    os.close(child)
            elif stat.S_ISREG(info.st_mode):
                read_file(directory, name, child_path)
            else:
                raise field_error(
                    "references",
                    f"Reference {child_path!r} is not a regular file or directory.",
                    code="reference_type_invalid",
                )

    root = os.open(
        workspace,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        for index, path in enumerate(references):
            parent = os.dup(root)
            try:
                for component in path.parts[:-1]:
                    child = os.open(
                        component,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=parent,
                    )
                    os.close(parent)
                    parent = child
                name = path.parts[-1]
                try:
                    info = os.stat(name, dir_fd=parent, follow_symlinks=False)
                except FileNotFoundError as error:
                    raise field_error(
                        f"references.{index}",
                        f"Referenced path does not exist: {path.as_posix()}",
                        code="reference_not_found",
                        next_action=(
                            "Correct or remove this references entry, then resubmit "
                            "record-experiment with the same Gateway Record IDs."
                        ),
                    ) from error
                if stat.S_ISLNK(info.st_mode):
                    raise field_error(
                        f"references.{index}",
                        f"Referenced path is a symbolic link: {path.as_posix()}",
                        code="reference_symlink_forbidden",
                    )
                if stat.S_ISDIR(info.st_mode):
                    child = os.open(
                        name,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=parent,
                    )
                    try:
                        walk(child, path.as_posix())
                    finally:
                        os.close(child)
                elif stat.S_ISREG(info.st_mode):
                    read_file(parent, name, path.as_posix())
                else:
                    raise field_error(
                        f"references.{index}",
                        f"Referenced path is not a regular file or directory: {path.as_posix()}",
                        code="reference_type_invalid",
                    )
            except FileNotFoundError as error:
                raise field_error(
                    f"references.{index}",
                    f"Referenced path does not exist: {path.as_posix()}",
                    code="reference_not_found",
                    next_action=(
                        "Correct or remove this references entry, then resubmit "
                        "record-experiment with the same Gateway Record IDs."
                    ),
                ) from error
            except OSError as error:
                if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise field_error(
                        f"references.{index}",
                        f"Referenced path contains a symbolic link or non-directory component: {path.as_posix()}",
                        code="reference_path_unsafe",
                    ) from error
                raise
            finally:
                os.close(parent)
    finally:
        os.close(root)
    return references, directories, files


def _archive_experiment_references(
    workspace: Path,
    campaign_root: Path,
    experiment_id: str,
    reference_revision: int,
    raw: object,
) -> list[str]:
    references, directories, files = _capture_experiment_references(workspace, raw)
    archive_root = _experiment_reference_archive_root(
        campaign_root, experiment_id, reference_revision
    )
    archived_references = [
        (Path("experiments") / experiment_id / "references" / path).as_posix()
        for path in references
    ]
    for directory in sorted(directories):
        with open_private_directory(archive_root / "files" / directory):
            pass
    file_rows = []
    for path, data in sorted(files.items()):
        private_write_bytes(archive_root / "files" / path, data)
        file_rows.append({
            "path": path,
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        })
    manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "reference_revision": reference_revision,
        "source_references": [path.as_posix() for path in references],
        "archived_references": archived_references,
        "directories": sorted(directories),
        "files": file_rows,
    }
    manifest_path = archive_root / "manifest.json"
    try:
        existing = read_json(manifest_path, limit=MAX_FILE_BYTES)
    except FileNotFoundError:
        private_write(manifest_path, manifest)
    else:
        if existing != manifest:
            raise RuntimeStateError(
                "Experiment reference archive identity collided with different material."
            )
    return archived_references


def _experiment_reference_archive_root(
    campaign_root: Path, experiment_id: str, reference_revision: int
) -> Path:
    if type(reference_revision) is not int or reference_revision < 1:
        raise RuntimeStateError(
            f"Experiment {experiment_id} has an invalid reference revision."
        )
    root = campaign_root / "experiment-archives" / experiment_id
    # Keep revision 1 at its historical path so existing Campaigns remain readable.
    return (
        root
        if reference_revision == 1
        else root / "revisions" / f"revision-{reference_revision:08d}"
    )


def _read_experiment_reference_archive(
    campaign_root: Path, experiment_id: str, reference_revision: int = 1
) -> tuple[dict[str, Any], dict[str, bytes]]:
    archive_root = _experiment_reference_archive_root(
        campaign_root, experiment_id, reference_revision
    )
    try:
        manifest = read_json(archive_root / "manifest.json", limit=MAX_FILE_BYTES)
    except (FileNotFoundError, OSError, ValueError) as error:
        raise RuntimeStateError(
            f"Experiment {experiment_id} reference archive is unavailable."
        ) from error
    if (
        manifest.get("schema_version") != 1
        or manifest.get("experiment_id") != experiment_id
        or manifest.get("reference_revision", 1) != reference_revision
        or not isinstance(manifest.get("source_references"), list)
        or not isinstance(manifest.get("archived_references"), list)
        or not isinstance(manifest.get("directories"), list)
        or not isinstance(manifest.get("files"), list)
    ):
        raise RuntimeStateError(
            f"Experiment {experiment_id} reference archive manifest is invalid."
        )
    if (
        len(manifest["files"]) > MAX_EXPERIMENT_REFERENCE_FILES
        or len(manifest["directories"]) > MAX_EXPERIMENT_REFERENCE_DIRECTORIES
    ):
        raise RuntimeStateError(f"Experiment {experiment_id} reference archive is oversized.")
    try:
        source_references = _experiment_reference_paths(manifest["source_references"])
    except AgentRequestError as error:
        raise RuntimeStateError(
            f"Experiment {experiment_id} reference archive manifest is invalid."
        ) from error
    expected_archived = [
        (Path("experiments") / experiment_id / "references" / path).as_posix()
        for path in source_references
    ]
    if manifest["archived_references"] != expected_archived:
        raise RuntimeStateError(
            f"Experiment {experiment_id} reference archive paths are inconsistent."
        )
    seen_directories = set()
    for value in manifest["directories"]:
        try:
            directory = relative_path(value)
        except (TypeError, ValueError) as error:
            raise RuntimeStateError(
                f"Experiment {experiment_id} reference archive contains an unsafe directory."
            ) from error
        if directory.as_posix() in seen_directories:
            raise RuntimeStateError(
                f"Experiment {experiment_id} reference archive repeats a directory."
            )
        seen_directories.add(directory.as_posix())
    payloads: dict[str, bytes] = {}
    total = 0
    for row in manifest["files"]:
        if not isinstance(row, dict):
            raise RuntimeStateError(f"Experiment {experiment_id} reference archive is invalid.")
        path, size, checksum = row.get("path"), row.get("size"), row.get("sha256")
        try:
            relative = relative_path(path)
        except (TypeError, ValueError) as error:
            raise RuntimeStateError(
                f"Experiment {experiment_id} reference archive contains an unsafe path."
            ) from error
        if (
            type(size) is not int
            or not 0 <= size <= MAX_EXPERIMENT_REFERENCE_FILE_BYTES
            or not isinstance(checksum, str)
            or re.fullmatch(r"[0-9a-f]{64}", checksum) is None
        ):
            raise RuntimeStateError(f"Experiment {experiment_id} reference archive is invalid.")
        try:
            data = read_regular_bytes(
                archive_root / "files" / relative,
                limit=MAX_EXPERIMENT_REFERENCE_FILE_BYTES + 1,
            )
        except (OSError, ValueError) as error:
            raise RuntimeStateError(
                f"Experiment {experiment_id} archived reference {path!r} is unavailable."
            ) from error
        if len(data) != size or hashlib.sha256(data).hexdigest() != checksum:
            raise RuntimeStateError(
                f"Experiment {experiment_id} archived reference {path!r} failed integrity validation."
            )
        total += len(data)
        if total > MAX_EXPERIMENT_REFERENCE_TOTAL_BYTES or path in payloads:
            raise RuntimeStateError(f"Experiment {experiment_id} reference archive is invalid.")
        payloads[path] = data
    return manifest, payloads


def copy_experiment_reference_archive(
    source_campaign_root: Path,
    destination_campaign_root: Path,
    experiment_id: str,
    reference_revision: int = 1,
) -> None:
    """Copy one immutable Experiment attachment revision across trajectory stores."""
    manifest, payloads = _read_experiment_reference_archive(
        source_campaign_root, experiment_id, reference_revision
    )
    destination = _experiment_reference_archive_root(
        destination_campaign_root, experiment_id, reference_revision
    )
    try:
        existing = read_json(destination / "manifest.json", limit=MAX_FILE_BYTES)
    except FileNotFoundError:
        existing = None
    except (OSError, ValueError) as error:
        raise RuntimeStateError(
            f"Destination archive for Experiment {experiment_id} is unreadable."
        ) from error
    if existing is not None and existing != manifest:
        raise RuntimeStateError(
            f"Destination archive for Experiment {experiment_id} conflicts with shared evidence."
        )
    for directory in manifest["directories"]:
        try:
            relative = relative_path(directory)
        except (TypeError, ValueError) as error:
            raise RuntimeStateError(
                f"Experiment {experiment_id} reference archive contains an unsafe directory."
            ) from error
        with open_private_directory(destination / "files" / relative):
            pass
    for path, data in payloads.items():
        private_write_bytes(destination / "files" / path, data)
    private_write(destination / "manifest.json", manifest)


def _project_experiment_references(
    campaign_root: Path,
    workspace: Path,
    experiment: Mapping[str, object],
) -> None:
    references = experiment.get("references")
    if not references:
        # Journals recorded before reference archives were introduced remain readable.
        # An explicitly empty optional list has no archive to project.
        return
    experiment_id = experiment.get("experiment_id")
    if not isinstance(experiment_id, str):
        raise RuntimeStateError("Visible Experiment has no valid identity.")
    reference_revision = experiment.get("reference_revision", 1)
    if type(reference_revision) is not int or reference_revision < 1:
        raise RuntimeStateError(
            f"Experiment {experiment_id} has an invalid reference revision."
        )
    manifest, payloads = _read_experiment_reference_archive(
        campaign_root, experiment_id, reference_revision
    )
    if references != manifest["archived_references"]:
        raise RuntimeStateError(
            f"Experiment {experiment_id} references do not match its immutable archive."
        )
    prefix = Path("experiments") / experiment_id / "references"
    for directory in manifest["directories"]:
        with open_private_directory(workspace / prefix / directory):
            pass
    for path, data in payloads.items():
        publish_if_changed(workspace, (prefix / path).as_posix(), data)


def _progress_markdown(
    board: Mapping[str, object],
    *,
    modules: AbstractSet[str] = frozenset({"directions", "experiments"}),
) -> bytes:
    """Render a concise Ruibo-style current-state page from trusted projections."""
    modules = frozenset(modules)
    frontier = board.get("pareto_frontier")
    frontier = frontier if isinstance(frontier, Mapping) else {}
    rows = frontier.get("best_by_shape")
    rows = rows if isinstance(rows, Mapping) else {}
    direction_groups: dict[str, list[dict[str, object]]] = {
        status: [] for status in _DIRECTION_PROJECTION_STATUSES
    }
    seen_directions: set[str] = set()
    for collection in (
        board.get("open_queue"),
        board.get("paused_directions"),
        board.get("shared_direction_pool"),
        board.get("shared_paused_directions"),
        board.get("closed_directions"),
    ):
        for raw in collection if isinstance(collection, list) else ():
            if not isinstance(raw, Mapping):
                continue
            item = deepcopy(dict(raw))
            direction_id = item.get("direction_id")
            raw_status = item.get("status")
            status = (
                "in_progress"
                if isinstance(raw_status, str) and raw_status.startswith("in_progress(")
                else raw_status
            )
            if (
                not isinstance(direction_id, str)
                or direction_id in seen_directions
                or status not in direction_groups
            ):
                continue
            seen_directions.add(direction_id)
            direction_groups[status].append(item)

    def text(value: object, *, empty: str = "none") -> str:
        if value is None or value == "":
            return empty
        return " ".join(str(value).split()).replace("|", "\\|")

    def commit(value: object) -> str:
        return f"`{value}`" if isinstance(value, str) and value else "none"

    incumbent = board.get("incumbent")
    incumbent = incumbent if isinstance(incumbent, Mapping) else {}
    measured_best = frontier.get("best_full_evaluation")
    measured_best = measured_best if isinstance(measured_best, Mapping) else {}
    latest = board.get("latest_result")
    latest = latest if isinstance(latest, Mapping) else {}
    baseline = board.get("v0_baseline")
    baseline = baseline if isinstance(baseline, Mapping) else {}
    current_lineage = board.get("current_lineage")
    current_lineage = current_lineage if isinstance(current_lineage, str) else "T1"
    lineages = board.get("lineages")
    lineages = lineages if isinstance(lineages, Mapping) else {}
    current = lineages.get(current_lineage)
    current = current if isinstance(current, Mapping) else {}
    raw_active_ids = current.get("active_direction_ids")
    active_ids = (
        [item for item in raw_active_ids if isinstance(item, str)]
        if isinstance(raw_active_ids, list)
        else [current["active_direction_id"]]
        if isinstance(current.get("active_direction_id"), str)
        else []
    )
    local_queue = [
        deepcopy(dict(item))
        for item in board.get("open_queue", [])
        if isinstance(item, Mapping)
    ]
    local_queue.sort(key=lambda item: (
        item.get("local_rank") if type(item.get("local_rank")) is int else 1 << 30,
        str(item.get("direction_id", "")),
    ))
    local_paused = [
        deepcopy(dict(item))
        for item in board.get("paused_directions", [])
        if isinstance(item, Mapping)
    ]
    local_paused.sort(key=lambda item: (
        item.get("local_rank") if type(item.get("local_rank")) is int else 1 << 30,
        str(item.get("direction_id", "")),
    ))
    peer_queue = [
        deepcopy(dict(item))
        for item in board.get("shared_direction_pool", [])
        if isinstance(item, Mapping)
    ]
    peer_queue.sort(key=lambda item: (
        item.get("shared_rank") if type(item.get("shared_rank")) is int else 1 << 30,
        str(item.get("direction_id", "")),
    ))
    peer_paused = [
        deepcopy(dict(item))
        for item in board.get("shared_paused_directions", [])
        if isinstance(item, Mapping)
    ]
    peer_paused.sort(key=lambda item: (
        item.get("shared_rank") if type(item.get("shared_rank")) is int else 1 << 30,
        str(item.get("direction_id", "")),
    ))
    active = [
        item
        for direction_id in active_ids
        for item in local_queue
        if item.get("direction_id") == direction_id
    ]
    focus_direction_id = current.get("focus_direction_id")
    focus = next(
        (
            item
            for item in active
            if item.get("direction_id") == focus_direction_id
        ),
        None,
    )
    selected = (
        focus
        or (active[0] if active else None)
        or (local_queue[0] if local_queue else None)
        or (peer_queue[0] if peer_queue else {})
    ) if "directions" in modules else {}
    blocked = (
        [item for item in local_paused if item.get("status") == "blocked"]
        if "directions" in modules else []
    )
    blocked.sort(key=lambda item: (
        item.get("local_rank") if type(item.get("local_rank")) is int else 1 << 30,
        str(item.get("direction_id", "")),
    ))
    best_latency = measured_best.get("latency_us_geomean")
    best_latency_text = (
        f"{best_latency:g} us"
        if type(best_latency) in {int, float} and math.isfinite(float(best_latency))
        else "not measured"
    )
    best_commit = measured_best.get("kernel_commit_id")
    incumbent_commit = incumbent.get("kernel_commit_id")
    baseline_latency = baseline.get("latency_us_geomean")
    baseline_latency_text = (
        f"{baseline_latency:g} us"
        if type(baseline_latency) in {int, float}
        and math.isfinite(float(baseline_latency))
        else "not measured"
    )
    baseline_line = (
        f"- **V0 baseline source commit ID:** {commit(baseline.get('kernel_commit_id'))}; "
        f"GeoMean: **{baseline_latency_text}**; correctness: "
        f"`{text(baseline.get('correctness_status'), empty='unknown')}`; measured Shapes: "
        f"{text(baseline.get('measured_shape_count'), empty='unknown')}; private Gateway Record: "
        f"{commit(baseline.get('gateway_record_id'))}."
        if baseline else "- **V0 baseline:** private evidence is unavailable."
    )
    best_line = (
        f"- **Best verified Kernel commit ID:** {commit(best_commit)}; "
        f"GeoMean: **{best_latency_text}**; evidence: "
        f"{commit(measured_best.get('gateway_record_id'))}."
    )
    if isinstance(best_commit, str) and best_commit == incumbent_commit:
        best_line = best_line[:-1] + "; also the promoted incumbent."
        incumbent_line = None
    else:
        incumbent_line = (
            f"- **Promoted incumbent commit ID:** {commit(incumbent_commit)}."
        )

    def finite(value: object) -> float | None:
        return (
            float(value)
            if type(value) in {int, float} and math.isfinite(float(value))
            else None
        )

    latest_scope = latest.get("scope")
    latest_scope = latest_scope if isinstance(latest_scope, Mapping) else {}
    scope_parts = [
        str(value)
        for value in (latest_scope.get("mode"), latest_scope.get("input_scope"))
        if isinstance(value, str) and value
    ]
    shape_count = latest_scope.get("measured_shape_count")
    if type(shape_count) is int:
        scope_parts.append(f"{shape_count} measured Shapes")
    latest_facts = [text(latest.get("status"), empty="unknown status")]
    correctness = latest.get("correctness_status")
    if isinstance(correctness, str):
        latest_facts.append(correctness)
    latency = finite(latest.get("latency_us_geomean"))
    if latency is not None and latency > 0:
        latest_facts.append(f"GeoMean {latency:g} us")
    speedup = finite(latest.get("speedup"))
    if speedup is not None and speedup > 0:
        latest_facts.append(f"speedup {speedup:g}x")
    latest_result_text = (
        f"{text(latest.get('operation'), empty='unknown operation')} "
        f"({'; '.join(latest_facts)}); scope: "
        f"{', '.join(scope_parts) if scope_parts else 'operation-defined'}; "
        f"Kernel commit ID {commit(latest.get('kernel_commit_id'))}; evidence "
        f"{commit(latest.get('gateway_record_id'))}."
        if latest else "none recorded."
    )
    suggested_next_work = (
        (
            f"{text(selected.get('name'), empty='unnamed Direction')} — "
            "choose its next concrete probe from the Direction plan, latest code, "
            "Experiments and Gateway Records."
        )
        if selected and selected.get("access") == "local_writable"
        else (
            f"Review the Shared direction pool, starting with {commit(selected.get('direction_id'))}; "
            "absorb compatible peer mechanisms and evidence into one local Direction, cite every "
            "source ID in `derived_from_direction_ids`, then start that local identity."
        )
        if selected and selected.get("access") == "peer_read_only"
        else (
            "review the visible Experiments and measurement evidence, then choose the highest-value next probe."
            if "experiments" in modules
            else "choose the highest-value next probe from the current Kernel and measurement evidence."
        )
    )
    if blocked:
        blocker = blocked[0]
        blocker_text = (
            f"{text(blocker.get('name'), empty='unnamed blocked Direction')} — "
            f"{text(blocker.get('analysis'), empty='no concrete blocker analysis recorded').rstrip('.;')}"
        )
        if blocker.get("reopen_condition"):
            blocker_text += (
                f"; reopen when: {text(blocker.get('reopen_condition')).rstrip('.;')}"
            )
        blocker_text += "."
    else:
        latest_error = latest.get("error")
        latest_error = latest_error if isinstance(latest_error, Mapping) else {}
        error_class = latest_error.get("error_class")
        if latest.get("status") in {"failed", "cancelled"} and error_class in {
            "infra", "timeout"
        }:
            blocker_text = (
                f"latest Gateway operation is blocked by {text(error_class)}: "
                f"{text(latest_error.get('message') or latest_error.get('reason'))}; "
                f"evidence {commit(latest.get('gateway_record_id'))}."
            )
        else:
            blocker_text = "none recorded."
    research_locations = ["`records/<gateway_record_id>.json`"]
    if "directions" in modules:
        research_locations.append("`directions/<status>/<direction_id>.json`")
    if "experiments" in modules:
        research_locations.append("`experiments/<experiment_id>/experiment.json`")
        if "directions" not in modules:
            research_locations.append("`experiments/index.json`")
    lines = [
        "# Current state",
        "",
        "> Runtime-managed, read-only projection. Do not edit this file; use the documented",
        "> Runtime mutation operations and treat Gateway Records as measurement truth.",
        "",
        best_line,
        *([incumbent_line] if incumbent_line is not None else []),
        baseline_line,
        f"- **Current lineage:** `{current_lineage}`.",
        f"- **Last measured working Kernel commit ID:** {commit(current.get('working_kernel_commit_id'))}.",
        f"- **Best verified lineage Kernel commit ID:** {commit(current.get('lineage_best_kernel_commit_id'))}.",
        f"- **Latest visible Gateway result and scope:** {latest_result_text}",
        f"- **Suggested next work/probe:** {suggested_next_work}",
        f"- **Current real blocker:** {blocker_text}",
        *([
            (
            "- **Active Directions:** " + "; ".join(
                f"{text(item.get('name'))} (`{item.get('direction_id')}`)" for item in active
            ) + "."
            if active else "- **Active Directions:** none."
            ),
            f"- **Focus Direction:** {commit(focus_direction_id)}.",
            f"- **Objective:** {text(selected.get('objective'), empty='none established')}",
        ] if "directions" in modules else []),
        "",
        "Visible evidence and research projections: " + ", ".join(research_locations) + ".",
        *([
            "Experiment files declare `recorded_by` and `access`; revise only "
            "`local_writable` Experiments and extend `peer_read_only` evidence with a new local record."
        ] if "experiments" in modules else []),
        "",
        "## Lineages",
        "",
        (
            "| Lineage | Last measured Kernel commit ID | Best verified Kernel commit ID | Focus Direction | Active Directions |"
            if "directions" in modules
            else "| Lineage | Last measured Kernel commit ID | Best verified Kernel commit ID |"
        ),
        "| --- | --- | --- | --- | --- |" if "directions" in modules else "| --- | --- | --- |",
    ]
    for lineage, raw in sorted(lineages.items()):
        item = raw if isinstance(raw, Mapping) else {}
        row = (
            f"| `{text(lineage)}` | {commit(item.get('working_kernel_commit_id'))} | "
            f"{commit(item.get('lineage_best_kernel_commit_id'))}"
        )
        if "directions" in modules:
            raw_ids = item.get("active_direction_ids")
            ids = (
                [value for value in raw_ids if isinstance(value, str)]
                if isinstance(raw_ids, list)
                else [item["active_direction_id"]]
                if isinstance(item.get("active_direction_id"), str)
                else []
            )
            row += " | " + commit(item.get("focus_direction_id"))
            row += " | " + (", ".join(commit(value) for value in ids) or "none")
        lines.append(row + " |")
    if not lineages:
        lines.append(
            "| _none_ | none | none | none | none |"
            if "directions" in modules else "| _none_ | none | none |"
        )
    if "directions" in modules:
        def append_directions(
            heading, entries, *, rank_field=None, rank_label="", total=None
        ):
            lines.extend([f"### {heading}", ""])
            if type(total) is int and total > len(entries):
                lines.extend([
                    f"_Showing {len(entries)} of {total}; inspect the Direction directories for the full catalog._",
                    "",
                ])
            if not entries:
                lines.extend(["_None._", ""])
                return
            for index, item in enumerate(entries, 1):
                rank = item.get(rank_field) if rank_field else None
                rank_text = (
                    f"{rank_label} #{rank}; " if type(rank) is int else ""
                )
                raw_status = item.get("status")
                owner = ""
                if isinstance(raw_status, str) and raw_status.startswith("in_progress("):
                    owner = (
                        f"owner `{raw_status.removeprefix('in_progress(').removesuffix(')')}`; "
                    )
                proposer = item.get("proposed_by")
                proposer_text = (
                    f"proposed by `{text(proposer)}`; "
                    if isinstance(proposer, str) and proposer else ""
                )
                access = item.get("access")
                access_text = (
                    f"access `{access}`; "
                    if access in {"local_writable", "peer_read_only"} else ""
                )
                status = (
                    raw_status.split("(", 1)[0]
                    if isinstance(raw_status, str) else "unknown"
                )
                lines.append(
                    f"{index}. **{text(item.get('name'), empty='unnamed')}** "
                    f"(`{text(item.get('direction_id'))}`) — "
                    f"{rank_text}status `{status}`; {proposer_text}{owner}{access_text}"
                    f"scope `{text(item.get('change_scope'), empty='incremental')}`."
                )
                for label, field in (
                    ("Objective", "objective"),
                    ("Applicability", "applicability"),
                    ("Expected benefit", "expected_benefit"),
                    ("Reopen when", "reopen_condition"),
                ):
                    if item.get(field):
                        lines.append(f"   - {label}: {text(item.get(field))}")
                if item.get("claim_review_quality"):
                    lines.append(
                        f"   - Claim review quality: {text(item.get('claim_review_quality'))}"
                    )
            lines.append("")

        closed = [
            item
            for status in ("completed", "abandoned")
            for item in direction_groups[status]
        ]
        closed.sort(key=lambda item: str(item.get("direction_id", "")))
        lines.extend(["", "## Directions", ""])
        append_directions(
            "Local work queue", local_queue,
            rank_field="local_rank", rank_label="local rank",
            total=board.get("open_queue_total"),
        )
        append_directions(
            "Paused local directions", local_paused,
            rank_field="local_rank", rank_label="local rank",
            total=board.get("paused_directions_total"),
        )
        append_directions(
            "Shared direction pool", peer_queue,
            rank_field="shared_rank", rank_label="shared rank",
            total=board.get("shared_direction_pool_total"),
        )
        append_directions(
            "Paused shared directions", peer_paused,
            rank_field="shared_rank", rank_label="shared rank",
            total=board.get("shared_paused_directions_total"),
        )
        append_directions(
            "Terminal directions", closed, total=board.get("closed_directions_total")
        )
    lines.extend([
        "## Visible-history per-Shape Pareto frontier",
        "",
        (
            f"Eligible visible full evaluations across local and imported peer history: "
            f"**{frontier.get('evaluation_count', 0)}**; "
            f"covered Shapes: **{frontier.get('shape_count', 0)}**."
        ),
        "",
    ])
    winners = frontier.get("winner_kernel_commit_ids")
    if isinstance(winners, list) and winners:
        lines.extend([
            "Winning Kernel commits: "
            + ", ".join(f"`{value}`" for value in winners if isinstance(value, str)),
            "",
        ])
    visible_rows = list(sorted(rows.items()))[:MAX_PROGRESS_FRONTIER_ROWS]
    if rows:
        if len(rows) > len(visible_rows):
            lines.extend([
                f"Showing {len(visible_rows)} of {len(rows)} Shapes; read `frontier.json` for the complete frontier.",
                "",
            ])
        lines.extend([
            "| Shape ID | Latency (us) | Kernel commit ID | Gateway Record ID |",
            "| --- | ---: | --- | --- |",
        ])
        for shape, raw in visible_rows:
            row = raw if isinstance(raw, Mapping) else {}
            lines.append(
                f"| `{shape}` | {row.get('latency_us', '')} | "
                f"`{row.get('kernel_commit_id', '')}` | "
                f"`{row.get('gateway_record_id', '')}` |"
            )
    else:
        lines.append("_No eligible full-Evaluate history yet._")
    lines.extend([
        "",
        "Each row is an observed per-Shape minimum, not a measured composite Kernel.",
        "",
    ])
    return "\n".join(lines).encode()


def _remove_stale_direction_projection(
    workspace: Path,
    direction_id: str,
    *,
    current_status: str,
) -> None:
    """Remove old status copies without replacing the unchanged current projection."""
    filename = f"{direction_id}.json"
    directories = [workspace / "directions"]
    directories.extend(
        workspace / "directions" / status
        for status in _DIRECTION_PROJECTION_STATUSES
        if status != current_status
    )
    for directory in directories:
        with open_private_directory(directory) as parent:
            try:
                metadata = os.stat(filename, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise RuntimeStateError(
                    f"Direction projection path is not a regular file: {directory / filename}"
                )
            os.unlink(filename, dir_fd=parent)


def _save_journal(path: Path, campaign_root: Path, journal: dict[str, Any]) -> None:
    if (
        len(json.dumps(journal, indent=2, sort_keys=True, allow_nan=False).encode()) + 1
        > MAX_FILE_BYTES
    ):
        raise AgentRequestError(
            "Runtime-managed Episode state exceeds 16 MiB; finish this Episode before adding more evidence",
            code="journal_size_limit",
        )
    private_write(path, journal)


def _mutable_journal(path: Path, module: str | None = None) -> dict[str, Any]:
    journal = load_journal(path)
    if module is not None and not _module_enabled(journal, module):
        label = "Direction" if module == "directions" else "Experiment"
        raise AgentRequestError(
            f"{label} recording is disabled for this Episode",
            code=f"{module}_disabled",
            next_action=(
                f"Do not use {label} mutation tools in this Episode. Continue with the enabled "
                "modules and saved Gateway Records, then follow the current episode-report schema."
            ),
        )
    if journal.get("state") != "in_progress":
        raise AgentRequestError(
            "cannot mutate finalized Runtime-managed Episode state",
            code="journal_finalized",
            repairable=False,
            next_action=(
                "This Episode has already submitted its report. Do not add more research records."
            ),
        )
    return journal


def _append_event(path: Path, campaign_root: Path, event: dict[str, Any]) -> None:
    journal = _mutable_journal(path, "directions")
    events = journal["direction_events"]
    events.append(event)
    _save_journal(path, campaign_root, journal)


def update_direction(
    path: Path,
    campaign_root: Path,
    request: Mapping[str, object],
    *,
    review_claim: Callable[[str, Mapping[str, object]], dict[str, object]] | None = None,
) -> dict[str, object]:
    current = _mutable_journal(path, "directions")
    experiments_enabled = _module_enabled(current, "experiments")
    value = dict(request)
    action = value.get("action")
    status_before_mutation: str | None = None
    history = _visible_history(path, campaign_root)
    views, statuses = _direction_state(path, campaign_root, history=history)
    if action == "propose":
        relationship_fields = (
            RELATIONSHIP_FIELDS
            if experiments_enabled
            else RELATIONSHIP_FIELDS - {"derived_from_experiment_ids"}
        )
        proposal_optional = DIRECTION_PROPOSAL_OPTIONAL_FIELDS | relationship_fields
        if not experiments_enabled:
            proposal_optional = proposal_optional | {"gateway_record_ids"}
        require_fields(
            value,
            DIRECTION_PROPOSAL_FIELDS,
            optional=proposal_optional,
            label="Direction proposal",
        )
        direction_id = f"direction_{uuid.uuid4().hex}"
        advance_direction(statuses, direction_id, action)
        ancestry: dict[str, Any] = {}
        if set(value) & RELATIONSHIP_FIELDS:
            ancestry = validate_relationship(
                direction_id,
                value,
                views,
                ChainMap(
                    history.experiments_by_id,
                    {
                        str(item["experiment_id"]): item
                        for item in history.current["experiments"]
                        if isinstance(item, dict)
                        and isinstance(item.get("experiment_id"), str)
                    },
                ),
            )
        event = {
            "direction_event_id": f"directionevent_{uuid.uuid4().hex}",
            "direction_id": direction_id,
            "action": "propose",
            "revision": 1,
            "name": _text(value["name"], "Direction name"),
            "change_scope": _change_scope(value["change_scope"]),
            "objective": _text(value["objective"], "Direction objective"),
            "applicability": _text(
                value["applicability"], "Direction applicability"
            ),
            "expected_benefit": _text(
                value["expected_benefit"], "Direction expected_benefit"
            ),
            "plan": _text_list(value["plan"], "Direction plan"),
            "stop_conditions": _text_list(
                value.get("stop_conditions", []), "Direction stop_conditions"
            ),
            "recorded_at": _now(),
            # A proposal is an exploration draft, not an established research
            # claim.  Independent review is intentionally deferred until the
            # trajectory commits resources with action=start.
            "claim_review_quality": "not_reviewed",
            **ancestry,
        }
        if not experiments_enabled:
            event["gateway_record_ids"] = _validate_record_ids(
                campaign_root,
                value.get("gateway_record_ids", []),
                allow_empty=True,
                label="Direction",
            )
    elif action == _DIRECTION_REVISION_ACTION:
        if set(value) & RELATIONSHIP_FIELDS:
            raise ValueError(
                "Direction genealogy is immutable; propose a new derived Direction instead of "
                "changing ancestry in a revision"
            )
        revision_optional = set(DIRECTION_PROPOSAL_OPTIONAL_FIELDS)
        if not experiments_enabled:
            revision_optional.add("gateway_record_ids")
        require_fields(
            value,
            DIRECTION_REVISION_FIELDS,
            optional=revision_optional,
            label="Direction revision",
        )
        direction_id = _text(value["direction_id"], "Direction ID")
        if DIRECTION_ID_RE.fullmatch(direction_id) is None:
            raise field_error(
                "direction_id",
                "Direction ID has an invalid format",
                code="direction_id_invalid",
            )
        if direction_id in history.shared_direction_ids:
            raise AgentRequestError(
                "This shared peer Direction is read-only and cannot be revised locally.",
                code="shared_direction_read_only",
                direction_id=direction_id,
                next_action=(
                    "Review compatible Shared-pool ideas, propose one local Direction citing every "
                    "source ID in derived_from_direction_ids, then revise that local identity."
                ),
            )
        if direction_id not in views:
            raise field_error(
                "direction_id",
                "Direction is outside this trajectory's editable history",
                code="direction_not_found",
                next_action=(
                    "Use a direction_id from the current directions/<status>/ projection. "
                    "If this is a new optimization claim, submit action=propose without a "
                    "direction_id instead."
                ),
            )
        advance_direction(dict(statuses), direction_id, action)
        previous = views[direction_id]
        status_before_mutation = str(previous.get("status"))
        previous_revision = previous.get("revision", 1)
        if type(previous_revision) is not int or previous_revision < 1:
            raise RuntimeStateError("Visible Direction has an invalid revision")
        event = {
            "direction_event_id": f"directionevent_{uuid.uuid4().hex}",
            "direction_id": direction_id,
            "action": action,
            "revision": previous_revision + 1,
            "name": _text(value["name"], "Direction name"),
            "change_scope": _change_scope(value["change_scope"]),
            "objective": _text(value["objective"], "Direction objective"),
            "applicability": _text(
                value["applicability"], "Direction applicability"
            ),
            "expected_benefit": _text(
                value["expected_benefit"], "Direction expected_benefit"
            ),
            "plan": _text_list(value["plan"], "Direction plan"),
            "stop_conditions": _text_list(
                value.get("stop_conditions", []), "Direction stop_conditions"
            ),
            "recorded_at": _now(),
        }
        if status_before_mutation == "proposed":
            # Editing an unstarted draft does not spend a reviewer Session.
            event["claim_review_quality"] = "not_reviewed"
        if not experiments_enabled:
            event["gateway_record_ids"] = _validate_record_ids(
                campaign_root,
                (
                    value["gateway_record_ids"]
                    if "gateway_record_ids" in value
                    else previous.get("gateway_record_ids", [])
                ),
                allow_empty=True,
                label="Direction",
            )
    else:
        if set(value) & RELATIONSHIP_FIELDS:
            raise ValueError(
                "Direction genealogy is immutable; propose a new derived Direction "
                "instead of changing ancestry in a lifecycle update"
            )
        if not isinstance(action, str) or action not in (
            _DIRECTION_ACTIONS - {"propose", _DIRECTION_REVISION_ACTION}
        ):
            raise ValueError(
                "Direction action must be propose, revise, start, complete, abandon, block, or defer"
            )
        fields = DIRECTION_CLOSURE_FIELDS
        closure_optional = set(DIRECTION_CLOSURE_OPTIONAL_FIELDS)
        if not experiments_enabled:
            closure_optional.remove("supporting_experiment_ids")
            closure_optional.add("gateway_record_ids")
        require_fields(
            value,
            fields if action in _CLOSURES else DIRECTION_UPDATE_FIELDS,
            optional=closure_optional if action in _CLOSURES else None,
            label=f"Direction {action}",
        )
        direction_id = _text(value["direction_id"], "Direction ID")
        if direction_id in history.shared_direction_ids:
            raise AgentRequestError(
                "This Direction belongs to a completed shared Epoch and is read-only.",
                code="shared_direction_read_only",
                direction_id=direction_id,
                next_action=(
                    "Review compatible Shared-pool ideas, propose one local Direction citing every "
                    "source ID in derived_from_direction_ids, then start and evaluate it."
                ),
            )
        # Peer Directions imported at an Epoch boundary retain their owner and
        # may legitimately remain in progress. They are read-only context, not
        # part of this trajectory's local active set.
        lifecycle_statuses = dict(statuses)
        for shared_id in history.shared_direction_ids:
            if lifecycle_statuses.get(shared_id) == "in_progress":
                lifecycle_statuses[shared_id] = "deferred"
        advance_direction(lifecycle_statuses, direction_id, action)
        status_before_mutation = str(statuses.get(direction_id))
        supporting = (
            _closure_support(
                path,
                campaign_root,
                direction_id,
                value,
                history=history,
                experiments_enabled=experiments_enabled,
            )
            if action in _CLOSURES
            else []
        )
        reopen_condition = value.get("reopen_condition")
        if (
            action in _CLOSURES
            and action in {"defer", "block"}
            and not isinstance(reopen_condition, str)
        ):
            raise field_error(
                "reopen_condition",
                "A deferred or blocked Direction requires reopen_condition",
            )
        event = {
            "direction_event_id": f"directionevent_{uuid.uuid4().hex}",
            "direction_id": direction_id,
            "action": action,
            "analysis": _text(value["analysis"], "Direction analysis"),
            "recorded_at": _now(),
            "supporting_experiment_ids": supporting,
            "reopen_condition": (
                _text(reopen_condition, "Direction reopen_condition")
                if reopen_condition is not None else None
            ),
        }
        if not experiments_enabled and action in _CLOSURES:
            records = _validate_record_ids(
                campaign_root,
                (
                    value["gateway_record_ids"]
                    if "gateway_record_ids" in value
                    else views[direction_id].get("gateway_record_ids", [])
                ),
                allow_empty=True,
                label="Direction",
            )
            event["gateway_record_ids"] = records
    review_required = (
        action == "start"
        or action in _CLOSURES
        or (
            action == _DIRECTION_REVISION_ACTION
            and status_before_mutation == "in_progress"
        )
    )
    review = (
        review_claim("direction", event)
        if review_claim is not None
        and review_required
        else None
    )
    if isinstance(review, Mapping) and review.get("claim_review_quality") == "unsupported":
        error = AgentRequestError(
            "Independent research review rejected this Direction; nothing was archived.",
            code="research_claim_unsupported",
            review=_agent_claim_review(review),
            next_action=(
                "Use review.scope_safe_interpretation and review.scientific_next_check to repair the claim, "
                "then submit a corrected Direction. Do not cite this rejected submission as a "
                "recorded Direction."
            ),
        )
        error.private_review = _private_claim_review(review)
        raise error
    review_projection = _review_projection(review)
    event.update(review_projection)
    _append_event(path, campaign_root, event)
    direction_status = (
        views[direction_id]["status"]
        if action == _DIRECTION_REVISION_ACTION else _DIRECTION_STATUS[action]
    )
    result: dict[str, object] = {
        "status": "updated" if action == _DIRECTION_REVISION_ACTION else "recorded",
        "direction_id": direction_id,
        "archive_path": f"directions/{direction_status}/{direction_id}.json",
    }
    if action in {"propose", _DIRECTION_REVISION_ACTION}:
        result["revision"] = event["revision"]
    if review is not None:
        result["review"] = dict(review)
    return result


def _read_private_gateway_evidence(journals_root: Path, record_id: str) -> dict[str, Any]:
    """Read a Journal-internal evidence view from the private Measurement Store.

    ``journals_root`` is the Runtime-owned ``<scope>/journals`` collection;
    its sibling ``<scope>/measurements`` stores the referenced Gateway Records.
    This is not an Agent response projection. Raw request/options/inputs stay in
    the private store; only a derived full-evaluation eligibility flag is carried
    into report validation. Agent output is assembled separately by the Journal
    handlers and compatibility publisher.
    """
    if GATEWAY_RECORD_ID_RE.fullmatch(record_id) is None:
        raise ValueError("Gateway Record ID must be a valid gateway-... ID returned by the Runtime")
    store = MeasurementStore(journals_root.parent / "measurements")
    try:
        record = store.read(record_id)
    except FileNotFoundError as error:
        raise ValueError(
            "Gateway Record is not available in this Campaign; check its exact records/<ID>.json path"
        ) from error
    except (ValueError, KeyError, TypeError, OSError) as error:
        raise RuntimeStateError(
            "Gateway Record evidence is unreadable or inconsistent; ask the operator to restore it"
        ) from error
    identity = record["kernels"].get("candidate", {})
    incumbent = record["kernels"].get("incumbent", {})
    response = record["response"]
    result = result_from_response(response, record["operation"])
    options = record["request"]["options"]
    standard_full_evaluation = (
        record["operation"] == "evaluate"
        and (options.get("evaluation_mode") or "full") == "full"
        and not options.get("evaluation_input_path")
        and not options.get("evaluation_shapes_path")
        and not any(
            arg == "--shape-id" or arg.startswith("--shape-id=")
            for arg in options.get("command", [])
        )
    )
    return {
        "gateway_record_id": record_id,
        "gateway_kind": "run" if record["operation"] == "evaluate" else record["operation"],
        **identity,
        "incumbent_kernel_id": incumbent.get("kernel_id"),
        "result": result or {},
        "execution_status": "completed" if record.get("cacheable") else "unknown",
        "exit_code": response["exit_code"],
        "standard_full_evaluation": standard_full_evaluation,
    }


def _has_passing_evaluate(
    campaign_root: Path,
    record_ids: object,
    kernel_digest: str,
    kernel_id: str | None = None,
) -> bool:
    if not isinstance(record_ids, list):
        return False
    for record_id in record_ids:
        if not isinstance(record_id, str):
            continue
        record = _read_private_gateway_evidence(campaign_root, record_id)
        if _passing_evaluate(record, kernel_digest, kernel_id=kernel_id):
            return True
    return False


def _passing_evaluate(
    record: dict[str, Any], kernel_digest: str, *, kernel_id: str | None = None
) -> bool:
    return (
        record.get("gateway_kind") == "run"
        and record.get("kernel_artifact_digest") == kernel_digest
        and (kernel_id is None or record.get("kernel_id") == kernel_id)
        and record["result"].get("all_pass") is True
        and not record["result"].get("error")
        and _completed_record(record)
        and record.get("exit_code") == 0
        and record["standard_full_evaluation"]
        and record["result"].get("mode") != "correctness_only"
        and bool(record["result"].get("latency_us_by_shape"))
    )


def selected_evaluation(path: Path, journals_root: Path, source: bytes) -> dict[str, Any]:
    """Controller-only lookup; revalidate exact source against the sealed report.

    Pass ``SupervisorRuntime.journals_root`` explicitly, rather than inferring
    the collection root from the Episode journal file's directory depth.
    """
    current = load_journal(path)
    digest = "sha256:" + hashlib.sha256(source).hexdigest()
    if current.get("state") != "candidate_ready" or current.get("candidate_kernel_digest") != digest:
        raise ValueError("Selected report does not match the acceptance Kernel")
    if not _module_enabled(current, "experiments"):
        record_ids = _validate_record_ids(journals_root, current["outcome"].get("gateway_record_ids"),
                                          label="Episode Report")
    else:
        selected_id = current["outcome"]["selected_experiment_id"]
        matches = [item for item in current["experiments"] if item["experiment_id"] == selected_id]
        if len(matches) != 1:
            raise ValueError("Selected report has no unique Experiment")
        record_ids = matches[0]["gateway_record_ids"]
    for record_id in reversed(record_ids):
        record = _read_private_gateway_evidence(journals_root, record_id)
        if _passing_evaluate(
            record, digest, kernel_id=current.get("candidate_kernel_id")
        ):
            return dict(record["result"], gateway_record_id=record_id)
    raise ValueError("Selected report evidence has no passing standard Evaluate for this Kernel")


def sealed_candidate_source(journal: dict[str, Any], git_workspace: Path) -> bytes | None:
    """Read committed source only when it matches a finalized private report."""
    if journal.get("state") != "candidate_ready":
        return None
    commit = journal.get("candidate_commit")
    if (not journal.get("finalized_at") or not isinstance(commit, str)
            or re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", commit) is None):
        raise RuntimeError("Private candidate report has no valid sealed commit")
    from aka.light.workspace.sources import from_git
    source = from_git(git_workspace, commit)
    if journal.get("candidate_kernel_digest") != "sha256:" + hashlib.sha256(source).hexdigest():
        raise RuntimeError("Sealed candidate commit does not match the private report digest")
    return source


def _validate_record_ids(
    campaign_root: Path,
    raw: object,
    *,
    label: str = "Experiment",
    allow_empty: bool = False,
) -> list[str]:
    records = _text_list(raw, f"{label} gateway_record_ids")
    if not records:
        if allow_empty:
            return []
        if label == "Episode Report":
            raise field_error("gateway_record_ids", "candidate_ready requires non-empty gateway_record_ids citing a passing full Evaluate")
        raise AgentRequestError(
            "This use of the Experiment as measured evidence requires at least one real "
            "Kernel-bound Gateway Record in gateway_record_ids. An unmeasured Experiment is "
            "allowed as a research note, but it cannot support a Direction closure or "
            "candidate promotion.",
            code="experiment_evidence_required",
            next_action=(
                "Cite an actual Evaluate, ABBA, Profile, Dev, Check, or Disassemble result in "
                "this Experiment, or select/create a measured Experiment for this operation. "
                "A failed diagnostic can document an unresolved blocker. Env/Wiki responses "
                "and transport errors without a Gateway Record do not qualify; never invent "
                "evidence."
            ),
        )
    if len(set(records)) != len(records):
        raise field_error("gateway_record_ids", f"{label} gateway_record_ids must not contain duplicates")
    for index, record_id in enumerate(records):
        try:
            record = _read_private_gateway_evidence(campaign_root, record_id)
        except ValueError as error:
            raise field_error(f"gateway_record_ids.{index}", str(error),
                              next_action="Check records/<gateway_record_id>.json, correct the citation, then resubmit.") from error
        digest = record.get("kernel_artifact_digest")
        kernel_id = record.get("kernel_id")
        if (
            not isinstance(record.get("gateway_kind"), str)
            or record["gateway_kind"] not in _EVIDENCE_KINDS
            or not isinstance(digest, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
            or not isinstance(kernel_id, str)
            or KERNEL_RECORD_ID_RE.fullmatch(kernel_id) is None
        ):
            raise ValueError(
                f"Gateway Record {record_id} must bind a real Kernel and a GPU operation; "
                "Env, Health, Wiki and unbound Dev records do not qualify"
            )
    return records


def _completed_record(record: Mapping[str, Any]) -> bool:
    """Completion is an execution fact, independent of correctness or interpretation."""
    return record["execution_status"] == "completed"


def _closure_support(
    path: Path, campaign_root: Path, direction_id: str, value: Mapping[str, object],
    *, history: _VisibleHistory | None = None, experiments_enabled: bool = True,
) -> list[str]:
    if not experiments_enabled:
        if value.get("supporting_experiment_ids") not in (None, []):
            raise ValueError("Experiment recording is disabled; omit supporting_experiment_ids")
        return []
    selected = _text_list(
        value.get("supporting_experiment_ids", []), "supporting_experiment_ids"
    )
    if len(selected) > 32 or len(set(selected)) != len(selected):
        raise AgentRequestError(
            "Direction supporting_experiment_ids must contain at most 32 unique visible "
            "Experiments owned by this Direction.",
            code="direction_support_required",
            next_action=(
                "Inspect experiments/*/experiment.json and retain only entries whose direction_id "
                "matches this Direction, or omit supporting_experiment_ids when the lifecycle "
                "decision does not rely on a saved Experiment."
            ),
        )
    if history is None:
        history = _visible_history(path, campaign_root)
    current_experiments = {
        item["experiment_id"]: item
        for item in history.current["experiments"]
        if isinstance(item, dict) and isinstance(item.get("experiment_id"), str)
    }
    # A local revision recorded in the current Episode supersedes the finalized
    # historical projection with the same stable Experiment identity.
    experiments = ChainMap(current_experiments, history.experiments_by_id)
    for experiment_id in selected:
        experiment = experiments.get(experiment_id)
        if experiment is None:
            raise ValueError(f"Supporting Experiment {experiment_id} is outside visible history")
        if experiment.get("direction_id") != direction_id:
            raise ValueError("Supporting Experiment must belong to the Direction being closed")
        _validate_record_ids(
            campaign_root,
            experiment.get("gateway_record_ids", []),
            allow_empty=True,
        )
    return selected


def record_experiment(
    path: Path,
    campaign_root: Path,
    request: Mapping[str, object],
    *,
    archive_references: Callable[[str, int, object], list[str]] | None = None,
    review_claim: Callable[[str, Mapping[str, object]], dict[str, object]] | None = None,
) -> dict[str, object]:
    value = dict(request)
    current = _mutable_journal(path, "experiments")
    directions_enabled = _module_enabled(current, "directions")
    legacy = not ({"intent", "interpretation"} & set(value)) and bool(
        {"hypothesis", "change", "analysis"} & set(value)
    )
    required = set(LEGACY_EXPERIMENT_FIELDS if legacy else EXPERIMENT_FIELDS)
    optional = {
        "direction_id",
        "wiki_usage",
        "wiki_query_ids",
        "wiki_usage_status",
        *EXPERIMENT_OPTIONAL_FIELDS,
    }
    require_fields(
        value,
        required,
        optional=optional - required,
        label="Experiment",
    )
    history = _visible_history(path, campaign_root)
    requested_experiment_id = value.get("experiment_id")
    existing: dict[str, Any] | None = None
    existing_index: int | None = None
    if requested_experiment_id is not None:
        experiment_id = _text(requested_experiment_id, "Experiment ID")
        if EXPERIMENT_ID_RE.fullmatch(experiment_id) is None:
            raise field_error(
                "experiment_id",
                "Experiment ID has an invalid format",
                code="experiment_id_invalid",
            )
        if experiment_id in history.shared_experiment_ids:
            raise AgentRequestError(
                "A shared peer Experiment is read-only and cannot be revised locally.",
                code="shared_experiment_read_only",
                experiment_id=experiment_id,
                next_action=(
                    "Record a new local Experiment that cites the relevant Gateway Records and "
                    "explains how it extends or corrects the shared Experiment."
                ),
            )
        visible = {
            item.get("experiment_id"): item
            for item in _visible_experiments(path, campaign_root, history=history)
            if isinstance(item, dict)
        }
        candidate = visible.get(experiment_id)
        if not isinstance(candidate, dict):
            raise field_error(
                "experiment_id",
                "Experiment is outside this trajectory's editable history",
                code="experiment_not_found",
                next_action=(
                    "Use an experiment_id from this trajectory's experiments/ projection, or "
                    "omit experiment_id to create a new Experiment."
                ),
            )
        existing = deepcopy(candidate)
        for index, item in enumerate(current["experiments"]):
            if isinstance(item, dict) and item.get("experiment_id") == experiment_id:
                existing_index = index
                break
    else:
        experiment_id = f"experiment_{uuid.uuid4().hex}"
    direction_id = None
    direction_status: str | None = None
    if directions_enabled:
        views = _direction_views(path, campaign_root, history=history)
        if existing is not None:
            direction_id = existing.get("direction_id")
            supplied_direction_id = value.get("direction_id")
            if supplied_direction_id is not None:
                supplied_direction_id = _text(
                    supplied_direction_id, "Experiment Direction ID"
                )
                if supplied_direction_id != direction_id:
                    raise field_error(
                        "direction_id",
                        "An Experiment revision cannot change its Direction",
                        code="experiment_direction_immutable",
                        existing_direction_id=direction_id,
                    )
        elif value.get("direction_id") is None:
            active = [
                item["direction_id"] for item in views.values()
                if item.get("status") == "in_progress"
                and item.get("direction_id") not in history.shared_direction_ids
            ]
            if len(active) != 1:
                raise field_error(
                    "direction_id",
                    "Omit direction_id only when exactly one local Direction is in progress",
                    code="direction_ambiguous",
                    active_direction_ids=active,
                )
            direction_id = active[0]
        else:
            direction_id = _text(value["direction_id"], "Experiment Direction ID")
        if direction_id is not None:
            if not isinstance(direction_id, str) or DIRECTION_ID_RE.fullmatch(direction_id) is None:
                raise ValueError("Experiment Direction ID has an invalid format")
            direction = views.get(direction_id)
            if direction is None:
                raise ValueError("Experiment Direction is outside visible history")
            direction_status = str(direction["status"])
            if direction_id in history.shared_direction_ids and existing is None:
                raise AgentRequestError(
                    "A shared Direction is read-only and cannot own a local Experiment.",
                    code="shared_direction_read_only",
                    direction_id=direction_id,
                    next_action=(
                        "Propose and start a new local Direction derived from the shared Direction, "
                        "then record this Experiment against the local Direction."
                    ),
                )
            if direction["status"] not in {"in_progress", *_CLOSED_STATUSES}:
                raise ValueError(
                    "Experiment Direction must be in progress or closed; start a proposed Direction "
                    "before recording evidence. Late evidence does not reopen exploration"
                )
    elif "direction_id" in value:
        raise ValueError("Direction recording is disabled; omit direction_id from Experiment")
    elif existing is not None and isinstance(existing.get("direction_id"), str):
        # The association remains part of the stable Experiment identity even
        # if this Episode intentionally disables Direction tools.
        direction_id = existing["direction_id"]
    record_ids = _validate_record_ids(
        campaign_root,
        (
            value["gateway_record_ids"]
            if "gateway_record_ids" in value
            else existing.get("gateway_record_ids", []) if existing is not None else []
        ),
        allow_empty=True,
    )
    intent = (
        _text(value["intent"], "Experiment intent")
        if not legacy else _text(value["hypothesis"], "Experiment hypothesis")
    )
    interpretation = (
        _text(value["interpretation"], "Experiment interpretation")
        if not legacy else _text(value["analysis"], "Experiment analysis")
    )
    revision = _experiment_revision(existing) + 1 if existing is not None else 1
    experiment = {
        "experiment_id": experiment_id,
        "revision": revision,
        "sequence": (
            existing.get("sequence", len(current["experiments"]) + 1)
            if existing is not None else len(current["experiments"]) + 1
        ),
        "name": _text(value["name"], "Experiment name"),
        "intent": intent,
        "gateway_record_ids": record_ids,
        "interpretation": interpretation,
    }
    if direction_id is not None:
        experiment["direction_id"] = direction_id
    if existing is not None and not any(
        key in value for key in ("wiki_usage", "wiki_query_ids", "wiki_usage_status")
    ):
        for key in ("wiki_usage", "wiki_query_ids", "wiki_usage_status"):
            if key in existing:
                experiment[key] = deepcopy(existing[key])
    if any(key in value for key in ("wiki_usage", "wiki_query_ids", "wiki_usage_status")):
        from aka.light.workflow.journal_projection import normalize_wiki_attribution

        experiment.update(
            {
                key: value[key]
                for key in ("wiki_usage", "wiki_query_ids", "wiki_usage_status")
                if key in value
            }
        )
        issues = []
        for field in ("wiki_usage", "wiki_query_ids"):
            rows = value.get(field)
            if field in value and not isinstance(rows, list):
                issues.append({"path": field, "code": "list_type", "message": "Expected an array."})
            if isinstance(rows, list) and len(rows) > 64:
                raise field_error(field, "Wiki attribution arrays may contain at most 64 entries.",
                                  code="wiki_attribution_invalid")
        if "wiki_usage_status" in value and not isinstance(value["wiki_usage_status"], str):
            issues.append({"path": "wiki_usage_status", "code": "string_type", "message": "Expected an attribution status."})
        usage = value.get("wiki_usage")
        for index, row in enumerate(usage if isinstance(usage, list) else []):
            if not isinstance(row, dict):
                continue
            label = f"wiki_usage[{index}]"
            for field in sorted(set(row) - {"query_id", "wiki_id", "cite", "disposition"})[:32]:
                issues.append({"path": f"{label}.{field[:128]}", "code": "extra_forbidden", "message": "Unexpected attribution field."})
        normalize_wiki_attribution(experiment)
        for row in experiment.get("wiki_usage", []):
            if isinstance(row, dict):
                # The Experiment already records the change, analysis and
                # Gateway evidence. Attribution only needs identity and use
                # disposition.
                row.pop("use", None)
                row.pop("evidence", None)
        for message in experiment.get("wiki_usage_errors", []):
            field = re.match(r"wiki_usage_status|wiki_query_ids|wiki_usage(?:\[[0-9]+\](?:\.[a-z_]+)?)?", message)
            issues.append({"path": field[0] if field else "wiki_usage", "code": "wiki_attribution_invalid",
                           "message": message})
        if issues:
            raise AgentRequestError(
                "Wiki attribution is invalid; no Experiment was recorded.", code="wiki_attribution_invalid",
                issues=issues[:32],
                next_action="Correct the indicated Wiki fields using request_schema and resubmit record-experiment. "
                            "Use applied, partially_applied, reference_only or rejected for disposition. "
                            "Reuse the existing Gateway Record IDs; do not rerun GPU measurements.",
            )
    if review_claim is not None:
        review_target = dict(experiment)
        if "references" in value:
            review_target["references"] = list(
                str(path) for path in _experiment_reference_paths(
                    value["references"], allow_empty=True
                )
            )
        review = review_claim("experiment", review_target)
        if review.get("claim_review_quality") == "unsupported":
            error = AgentRequestError(
                "Independent research review rejected this Experiment; nothing was archived.",
                code="research_claim_unsupported",
                review=_agent_claim_review(review),
                next_action=(
                    "Use review.scope_safe_interpretation and review.scientific_next_check to repair the "
                    "claim, then submit a corrected Experiment. Reuse existing Gateway Record "
                    "IDs; do not rerun measurements solely because the interpretation was rejected."
                ),
            )
            error.private_review = _private_claim_review(review)
            raise error
        if (
            review.get("claim_review_quality") == "supported"
            and direction_status == "in_progress"
        ):
            review = dict(review)
            review["direction_status"] = direction_status
            review["runtime_action"] = _ACTIVE_DIRECTION_EXPERIMENT_RUNTIME_ACTION
    else:
        review = None
    experiment.update(_review_projection(review))
    if "references" in value:
        reference_paths = _experiment_reference_paths(
            value["references"],
            allow_empty=True,
        )
        if reference_paths:
            if archive_references is None:
                raise RuntimeStateError(
                    "Experiment reference archiving is unavailable; no Experiment was recorded."
                )
            experiment["references"] = archive_references(
                experiment_id, revision, value["references"]
            )
            experiment["reference_revision"] = revision
        else:
            experiment["references"] = []
    elif existing is not None and "references" in existing:
        experiment["references"] = deepcopy(existing["references"])
        experiment["reference_revision"] = existing.get("reference_revision", 1)
    if existing_index is None:
        current["experiments"].append(experiment)
    else:
        current["experiments"][existing_index] = experiment
    _save_journal(path, campaign_root, current)
    result: dict[str, object] = {
        "status": "updated" if existing is not None else "recorded",
        "experiment_id": experiment["experiment_id"],
        "revision": revision,
        "archive_path": (
            f"experiments/{experiment['experiment_id']}/experiment.json"
        ),
    }
    if review is not None:
        result["review"] = dict(review)
    return result


def validate_report_evidence(
    path: Path,
    campaign_root: Path,
    workspace: Path,
    *,
    status: str,
    selected_id: object,
    gateway_record_ids: object = None,
    source_policy: dict | None = None,
    kernel_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], bytes | None]:
    """Read-only preflight shared by HTTP submission and Supervisor terminal rechecks."""
    current = load_journal(path)
    directions_enabled = _module_enabled(current, "directions")
    experiments_enabled = _module_enabled(current, "experiments")
    if not experiments_enabled:
        if selected_id is not None:
            raise ValueError(
                "The Experiment module is disabled; cite gateway_record_ids instead of "
                "selected_experiment_id"
            )
        if status != "candidate_ready":
            if gateway_record_ids is not None:
                raise ValueError(f"{status} cannot include gateway_record_ids")
            directions = (_direction_views(path, campaign_root) if directions_enabled else {})
            return current, dict(directions), None
        records = _validate_record_ids(campaign_root, gateway_record_ids, label="Episode Report")
        from aka.light.workspace.sources import snapshot as source_snapshot
        source = source_snapshot(workspace, policy=source_policy)
        digest = "sha256:" + hashlib.sha256(source).hexdigest()
        if not _has_passing_evaluate(campaign_root, records, digest, kernel_id):
            raise field_error("gateway_record_ids",
                "Episode Report must cite a passing standard full Evaluate Gateway Record for "
                "byte-identical current candidate source; inspect records/<ID>.json, repair the Kernel or "
                "record IDs, then resubmit episode-report", code="candidate_evidence_mismatch",
                recovery=[{"path": f"records/{record_id}.json",
                           "instruction": "Check the result and its Kernel binding before correcting the report."}
                          for record_id in records[:8]],
            )
        directions = (_direction_views(path, campaign_root) if directions_enabled else {})
        return current, dict(directions), source
    if gateway_record_ids is not None:
        raise ValueError(
            "The Experiment module is enabled; select an Experiment with "
            "selected_experiment_id instead of gateway_record_ids"
        )
    history = _visible_history(path, campaign_root)
    current = history.current
    directions = (_direction_views(path, campaign_root, history=history)
                  if directions_enabled else {})
    if status != "candidate_ready":
        if selected_id is not None and selected_id != "":
            raise ValueError(f"{status} cannot include selected_experiment_id")
        return current, directions, None
    if not isinstance(selected_id, str) or EXPERIMENT_ID_RE.fullmatch(selected_id) is None:
        raise ValueError("candidate_ready requires a valid selected_experiment_id")
    current_matches = [
        item
        for item in current["experiments"]
        if isinstance(item, dict) and item.get("experiment_id") == selected_id
    ]
    if len(current_matches) > 1:
        raise RuntimeStateError(
            "The current Episode contains duplicate Experiment identities."
        )
    selected = (
        current_matches[0]
        if current_matches
        else history.experiments_by_id.get(selected_id)
    )
    if not isinstance(selected, dict):
        raise ValueError(
            "selected_experiment_id must identify a visible Experiment"
        )
    selected_review_quality = selected.get(
        "claim_review_quality", selected.get("review_quality")
    )
    if selected_review_quality != "supported":
        raise AgentRequestError(
            "candidate_ready requires a visible Experiment with a supported measured decision.",
            code="candidate_experiment_review_incomplete",
            experiment_id=selected_id,
            claim_review_quality=selected_review_quality,
            next_action=(
                "Revise a local Experiment using its stable experiment_id and address its projected "
                "scope_safe_interpretation/scientific_next_check before nominating the candidate. "
                "A historical Experiment may be reused without revision when its measured decision is supported "
                "and still cites a passing full Evaluate of the exact current commit."
            ),
        )
    if directions_enabled:
        direction_id = selected.get("direction_id")
        if not isinstance(direction_id, str) or direction_id not in directions:
            raise ValueError("selected Experiment Direction is outside visible history")
        if directions[direction_id]["status"] == "proposed":
            raise ValueError("selected Experiment Direction must have been started")
    records = _validate_record_ids(campaign_root, selected.get("gateway_record_ids"))
    from aka.light.workspace.sources import snapshot as source_snapshot
    kernel_source = source_snapshot(workspace, policy=source_policy)
    kernel_digest = "sha256:" + hashlib.sha256(kernel_source).hexdigest()
    if not _has_passing_evaluate(campaign_root, records, kernel_digest, kernel_id):
        raise ValueError(
            "selected Experiment must cite a passing Evaluate Gateway record whose "
            "Kernel exactly matches the current candidate source; inspect the cited records "
            "under records/<gateway_record_id>.json"
        )
    return current, directions, kernel_source


class SupervisorJournalService:
    """Execute scoped Journal operations over one private Episode document."""

    def __init__(
        self,
        *,
        workspace: Path,
        campaign_root: Path,
        evidence_root: Path,
        git_workspace: Path | None = None,
        supervisor_git: bool = False,
        research_root: Path | None = None,
        origin: Mapping[str, object] | None = None,
        claim_reviewer: Callable[..., Mapping[str, object]] | None = None,
    ) -> None:
        self.workspace = workspace
        self.git_workspace = git_workspace or workspace
        self.campaign_root = campaign_root
        self.evidence_root = evidence_root
        self.path = evidence_root / "journal.json"
        self.supervisor_git = supervisor_git
        self.origin = dict(origin or {"trajectory": 1})
        self.claim_reviewer = claim_reviewer
        self.research = ResearchStore(
            research_root or campaign_root.parent / "research"
        )

    @property
    def lineage(self) -> str:
        trajectory = self.origin.get("trajectory", 1)
        return f"T{trajectory}" if type(trajectory) is int and trajectory > 0 else "T1"

    @staticmethod
    def _reviewed_claim_identity(
        claim_type: str, target: Mapping[str, object]
    ) -> tuple[str, int] | None:
        if claim_type == "experiment":
            claim_id = target.get("experiment_id")
            revision = target.get("revision", 1)
            pattern = EXPERIMENT_ID_RE
        elif claim_type == "direction":
            claim_id = target.get("direction_id")
            revision = target.get("revision", 1)
            pattern = DIRECTION_ID_RE
        else:
            return None
        if (
            not isinstance(claim_id, str)
            or pattern.fullmatch(claim_id) is None
            or type(revision) is not int
            or revision < 1
        ):
            return None
        return claim_id, revision

    def _claim_review_index_path(self, claim_type: str, claim_id: str) -> Path:
        return self.campaign_root / "claim-review-index" / claim_type / f"{claim_id}.json"

    def _prior_claim_review_session(
        self, claim_type: str, target: Mapping[str, object]
    ) -> str | None:
        identity = self._reviewed_claim_identity(claim_type, target)
        if identity is None:
            return None
        claim_id, _revision = identity
        try:
            value = read_json(
                self._claim_review_index_path(claim_type, claim_id),
                limit=MAX_FILE_BYTES,
            )
        except (FileNotFoundError, OSError, ValueError):
            return None
        session_id = value.get("session_id")
        if (
            value.get("schema") != "atrex.research-claim-review-session-index.v1"
            or value.get("claim_type") != claim_type
            or value.get("claim_id") != claim_id
            or not isinstance(session_id, str)
            or not session_id
            or len(session_id) > 512
        ):
            return None
        return session_id

    def _remember_claim_review_session(
        self,
        claim_type: str,
        target: Mapping[str, object],
        review: Mapping[str, object],
    ) -> None:
        identity = self._reviewed_claim_identity(claim_type, target)
        audit = review.get(_REVIEW_SESSION_AUDIT_KEY)
        if identity is None or not isinstance(audit, Mapping):
            return
        attempts = audit.get("attempts")
        if not isinstance(attempts, list):
            return
        accepted = next(
            (
                row for row in reversed(attempts)
                if isinstance(row, Mapping)
                and row.get("status") == "accepted"
                and isinstance(row.get("session_id"), str)
                and row.get("session_id")
            ),
            None,
        )
        if accepted is None:
            return
        claim_id, revision = identity
        private_write(
            self._claim_review_index_path(claim_type, claim_id),
            {
                "schema": "atrex.research-claim-review-session-index.v1",
                "claim_type": claim_type,
                "claim_id": claim_id,
                "claim_revision": revision,
                "session_id": accepted["session_id"],
                "conversation_path": accepted.get("conversation_path"),
                "updated_at": _now(),
            },
        )

    def _begin_claim_audit(self, claim_type: str, submission: object) -> str:
        """Durably capture every Agent claim submission before validation."""
        audit_id = "claimaudit_" + uuid.uuid4().hex
        audit_root = self.evidence_root / "claim-audit" / audit_id
        private_write(
            audit_root / "submission.json",
            {
                "schema": "atrex.research-claim-audit.v1",
                "audit_id": audit_id,
                "claim_type": claim_type,
                "received_at": _now(),
                "origin": deepcopy(self.origin),
                "submission": deepcopy(submission),
            },
        )
        if claim_type == "experiment" and isinstance(submission, Mapping):
            raw_references = submission.get("references")
            if raw_references is None:
                private_write(
                    audit_root / "references" / "manifest.json",
                    {
                        "schema": "atrex.research-claim-audit-references.v1",
                        "requested": False,
                        "source_references": [],
                        "directories": [],
                        "files": [],
                    },
                )
            elif raw_references == []:
                private_write(
                    audit_root / "references" / "manifest.json",
                    {
                        "schema": "atrex.research-claim-audit-references.v1",
                        "requested": True,
                        "source_references": [],
                        "directories": [],
                        "files": [],
                    },
                )
            else:
                try:
                    references, directories, files = _capture_experiment_references(
                        self.workspace, raw_references
                    )
                except Exception as error:
                    response = getattr(error, "response", None)
                    private_write(
                        audit_root / "references" / "manifest.json",
                        {
                            "schema": "atrex.research-claim-audit-references.v1",
                            "requested": True,
                            "capture_status": "failed",
                            "source_references": deepcopy(raw_references),
                            "error": (
                                deepcopy(dict(response))
                                if isinstance(response, Mapping) else
                                {
                                    "type": type(error).__name__,
                                    "message": str(error)[:2000],
                                }
                            ),
                        },
                    )
                else:
                    for directory in sorted(directories):
                        with open_private_directory(
                            audit_root / "references" / "files" / directory
                        ):
                            pass
                    rows = []
                    for path, data in sorted(files.items()):
                        private_write_bytes(
                            audit_root / "references" / "files" / path,
                            data,
                        )
                        rows.append({
                            "path": path,
                            "size": len(data),
                            "sha256": hashlib.sha256(data).hexdigest(),
                        })
                    private_write(
                        audit_root / "references" / "manifest.json",
                        {
                            "schema": "atrex.research-claim-audit-references.v1",
                            "requested": True,
                            "capture_status": "captured",
                            "source_references": [
                                path.as_posix() for path in references
                            ],
                            "directories": sorted(directories),
                            "files": rows,
                        },
                    )
        return audit_id

    def _finish_claim_audit(
        self,
        audit_id: str,
        *,
        status: str,
        result: Mapping[str, object] | None = None,
        error: BaseException | None = None,
    ) -> None:
        """Record the final disposition without exposing private audit identity."""
        outcome: dict[str, object] = {
            "schema": "atrex.research-claim-audit-outcome.v1",
            "audit_id": audit_id,
            "finished_at": _now(),
            "status": status,
        }
        if result is not None:
            outcome["result"] = deepcopy(dict(result))
        if error is not None:
            response = getattr(error, "response", None)
            if isinstance(response, Mapping):
                outcome["error"] = deepcopy(dict(response))
            else:
                outcome["error"] = {
                    "code": "invalid_request" if isinstance(error, ValueError) else "runtime_error",
                    "message": str(error)[:2000],
                    "type": type(error).__name__,
                }
            private_review = getattr(error, "private_review", None)
            if isinstance(private_review, Mapping):
                outcome["private_review"] = deepcopy(dict(private_review))
        private_write(
            self.evidence_root / "claim-audit" / audit_id / "outcome.json",
            outcome,
        )

    def _observe_direction(self, direction_id: str) -> None:
        direction = _direction_views(self.path, self.campaign_root).get(direction_id)
        if direction is None:
            raise RuntimeStateError("Recorded Direction is missing from the visible projection")
        visible = deepcopy(direction)
        visible.setdefault("proposed_by", self.lineage)
        visible["access"] = "local_writable"
        if visible.get("status") == "in_progress" and not visible.get("owner"):
            visible["owner"] = self.lineage
        self.research.observe_direction(visible, origin=self.origin)

    def _observe_experiment(self, experiment_id: str) -> None:
        experiment = next(
            (
                item
                for item in _visible_experiments(self.path, self.campaign_root)
                if item.get("experiment_id") == experiment_id
            ),
            None,
        )
        if experiment is None:
            raise RuntimeStateError("Recorded Experiment is missing from the visible projection")
        visible = deepcopy(experiment)
        visible.setdefault("recorded_by", self.lineage)
        visible["access"] = "local_writable"
        self.research.observe_experiment(visible, origin=self.origin)

    def _sync_research(self) -> None:
        history = _visible_history(self.path, self.campaign_root)
        for direction_id, direction in _direction_views(
            self.path, self.campaign_root, history=history
        ).items():
            visible = deepcopy(direction)
            if direction_id in history.shared_direction_ids:
                # Current shared projections include proposed_by. Keep legacy
                # shared data honest instead of attributing it to this reader.
                visible.setdefault("proposed_by", visible.get("owner") or "unknown")
                visible["access"] = "peer_read_only"
            else:
                visible.setdefault("proposed_by", self.lineage)
                visible["access"] = "local_writable"
            if visible.get("status") == "in_progress" and not visible.get("owner"):
                visible["owner"] = self.lineage
            self.research.observe_direction(visible, origin=self.origin)
        for experiment in _visible_experiments(self.path, self.campaign_root):
            visible = deepcopy(experiment)
            experiment_id = visible.get("experiment_id")
            if experiment_id in history.shared_experiment_ids:
                visible.setdefault("recorded_by", "unknown")
                visible["access"] = "peer_read_only"
            else:
                visible.setdefault("recorded_by", self.lineage)
                visible["access"] = "local_writable"
            self.research.observe_experiment(visible, origin=self.origin)
        queue = shared_research_queue(self.campaign_root)
        self.research.synchronize_shared_queue(queue, origin=self.origin)

    def observe_measurement(self, agent_result: Mapping[str, object]) -> None:
        """Checkpoint a Kernel-bound operation in the current lineage.

        Gateway Records remain the measurement truth. This projection records
        only which committed source was most recently measured and which passing
        standard full Evaluate is the best one observed by this lineage. ABBA
        has two source identities and deliberately does not move the single
        working-HEAD pointer.
        """
        kernel_commit_id = agent_result.get("kernel_commit_id")
        if (
            not isinstance(kernel_commit_id, str)
            or KERNEL_RECORD_ID_RE.fullmatch(kernel_commit_id) is None
        ):
            return
        payload: dict[str, object] = {"working_kernel_id": kernel_commit_id}
        result = agent_result.get("result")
        result = result if isinstance(result, Mapping) else {}
        passing_evaluate = (
            agent_result.get("operation") == "evaluate"
            and agent_result.get("status") == "completed"
            and result.get("correct") is True
        )
        if passing_evaluate:
            payload["last_correct_kernel_id"] = kernel_commit_id
        latency = result.get("latency_us_geomean")
        standard_full = (
            passing_evaluate
            and result.get("mode") == "full"
            and result.get("input_scope") == "contract"
            and type(latency) in {int, float}
            and math.isfinite(float(latency))
            and float(latency) > 0
        )
        if standard_full:
            state = self.research.read()
            lineages = state.get("lineages")
            lineages = lineages if isinstance(lineages, Mapping) else {}
            current = lineages.get(self.lineage)
            current = current if isinstance(current, Mapping) else {}
            prior_latency = current.get("lineage_best_latency_us")
            if (
                type(prior_latency) not in {int, float}
                or not math.isfinite(float(prior_latency))
                or float(prior_latency) <= 0
                or float(latency) < float(prior_latency)
            ):
                payload.update(
                    lineage_best_kernel_id=kernel_commit_id,
                    lineage_best_latency_us=float(latency),
                )
        self.research.checkpoint_lineage(payload, origin=self.origin)

    def publish_research_files(
        self,
        *,
        synchronize: bool = True,
        direction_ids: AbstractSet[str] | None = None,
        experiment_ids: AbstractSet[str] | None = None,
        record_ids: AbstractSet[str] | None = None,
    ) -> None:
        """Materialize the Agent-visible research history in the workspace.

        The private Journal, Research Store, Measurement Store, and Git remain
        authoritative. PROGRESS.md and these JSON files are normal-file projections for
        search, remote ``command`` probes, and long-running engineering work;
        Agent edits are never read back as Runtime state and are overwritten on
        the next refresh.
        """
        journal = load_journal(self.path)
        modules = set(_tool_modules(journal))

        if synchronize:
            self._sync_research()
        research = self.research.read()
        from aka.light.experiment.frontier import kernel_pareto_frontier, latest_gateway_result
        from aka.light.workspace.agent_git import AgentGit
        store = MeasurementStore(self.campaign_root.parent / "measurements")
        trajectory = self.origin.get("trajectory", 1)
        git = AgentGit(
            self.workspace,
            self.workspace / ".git" / "unused-runtime-bundle",
        )
        reachable_cache: dict[str, bool] = {}

        def commit_reachable(commit_id: str) -> bool:
            if commit_id not in reachable_cache:
                reachable_cache[commit_id] = git.commit_reachable(commit_id)
            return reachable_cache[commit_id]

        visible_record_ids = _git_visible_record_ids(
            store,
            self.workspace,
            commit_reachable=commit_reachable,
        )
        board = self.research.board(
            trajectory=trajectory if type(trajectory) is int and trajectory > 0 else 1,
            pareto_frontier=kernel_pareto_frontier(
                store, record_ids=visible_record_ids
            ),
        )
        board["latest_result"] = latest_gateway_result(
            store, record_ids=visible_record_ids
        )
        from aka.light.workspace.baseline import (
            ensure_baseline_gateway_record,
            read_baseline_record,
        )
        # V0 facts live beside the private Controller repository, not inside
        # the projected Agent workspace.  Expose only the Agent-visible base
        # commit identity from the Runtime Journal; never leak the Controller
        # commit stored in the private V0 receipt.
        ensure_baseline_gateway_record(self.git_workspace)
        baseline = read_baseline_record(self.git_workspace) or {}
        performance = baseline.get("performance")
        performance = performance if isinstance(performance, Mapping) else {}
        correctness = baseline.get("correctness")
        correctness = correctness if isinstance(correctness, Mapping) else {}
        board["v0_baseline"] = {
            "kernel_commit_id": journal.get("base_kernel_id"),
            "latency_us_geomean": performance.get("latency_us_geomean"),
            "measured_shape_count": performance.get("measured_shape_count"),
            "correctness_status": correctness.get("status"),
            "gateway_record_id": baseline.get("gateway_record_id"),
        }
        board = _git_reachable_projection(board, commit_reachable)
        publish_if_changed(
            self.workspace,
            "PROGRESS.md",
            _progress_markdown(board, modules=modules),
        )
        publish_if_changed(
            self.workspace,
            "frontier.json",
            _projection_json(board.get("pareto_frontier", {})),
        )

        records_root = self.workspace / "records"
        with open_private_directory(records_root) as parent:
            if record_ids is None:
                for name in os.listdir(parent):
                    if not name.endswith(".json"):
                        continue
                    if name.removesuffix(".json") not in visible_record_ids:
                        metadata = os.stat(name, dir_fd=parent, follow_symlinks=False)
                        if not stat.S_ISREG(metadata.st_mode):
                            raise RuntimeStateError(
                                f"Record projection path is not a regular file: {records_root / name}"
                            )
                        os.unlink(name, dir_fd=parent)
        selected_record_ids = (
            sorted(visible_record_ids)
            if record_ids is None
            else sorted(set(record_ids) & visible_record_ids)
        )
        for record_id in selected_record_ids:
            publish_if_changed(
                self.workspace,
                f"records/{record_id}.json",
                _projection_json(_public_gateway_record(store, record_id)),
            )

        if not modules:
            return

        if "directions" in modules and (direction_ids is None or direction_ids):
            with open_private_directory(self.workspace / "directions"):
                pass
            local_queue_ranks = {
                direction_id: rank
                for rank, direction_id in enumerate(research.get("queue", []), 1)
                if isinstance(direction_id, str)
            }
            shared_queue_ranks = {
                direction_id: rank
                for rank, direction_id in enumerate(research.get("shared_queue", []), 1)
                if isinstance(direction_id, str)
            }
            research_directions = research.get("directions", {})
            research_directions = (
                research_directions if isinstance(research_directions, Mapping) else {}
            )
            for direction_id, direction in _direction_views(
                self.path, self.campaign_root
            ).items():
                if direction_ids is not None and direction_id not in direction_ids:
                    continue
                visible = _git_reachable_projection(
                    _public_direction(direction), commit_reachable
                )
                research_direction = research_directions.get(direction_id)
                if isinstance(research_direction, Mapping):
                    for field in ("owner", "proposed_by", "access"):
                        if field in research_direction:
                            visible[field] = deepcopy(research_direction[field])
                if direction_id in local_queue_ranks:
                    visible["local_rank"] = local_queue_ranks[direction_id]
                if direction_id in shared_queue_ranks:
                    visible["shared_rank"] = shared_queue_ranks[direction_id]
                status = visible.get("status")
                if status not in _DIRECTION_PROJECTION_STATUSES:
                    raise RuntimeStateError(
                        f"Visible Direction {direction_id} has invalid status {status!r}."
                    )
                _remove_stale_direction_projection(
                    self.workspace,
                    direction_id,
                    current_status=status,
                )
                publish_if_changed(
                    self.workspace,
                    f"directions/{status}/{direction_id}.json",
                    _projection_json(visible),
                )

        if "experiments" in modules and (experiment_ids is None or experiment_ids):
            with open_private_directory(self.workspace / "experiments"):
                pass

            research_experiments = research.get("experiments", {})
            research_experiments = (
                research_experiments if isinstance(research_experiments, Mapping) else {}
            )
            visible_experiments = _visible_experiments(self.path, self.campaign_root)
            if "directions" not in modules:
                index = []
                for experiment in visible_experiments:
                    experiment_id = experiment.get("experiment_id")
                    if not isinstance(experiment_id, str):
                        continue
                    research_experiment = research_experiments.get(experiment_id, {})
                    index.append({
                        "experiment_id": experiment_id,
                        "name": experiment.get("name"),
                        "recorded_by": (
                            research_experiment.get("recorded_by")
                            if isinstance(research_experiment, Mapping) else None
                        ),
                        "access": (
                            research_experiment.get("access")
                            if isinstance(research_experiment, Mapping) else None
                        ),
                        "claim_review_quality": experiment.get(
                            "claim_review_quality", experiment.get("review_quality")
                        ),
                        "gateway_record_count": len(
                            experiment.get("gateway_record_ids", [])
                        ),
                    })
                publish_if_changed(
                    self.workspace,
                    "experiments/index.json",
                    _projection_json(index),
                )
            else:
                # A stable workspace may be reused after the research-module
                # configuration changes.  The compact index belongs only to
                # Experiment-only mode; leaving an older copy visible would
                # present two conflicting navigation contracts to the Agent.
                index_path = self.workspace / "experiments" / "index.json"
                if index_path.exists():
                    index_path.unlink()
            for experiment in visible_experiments:
                experiment_id = experiment.get("experiment_id")
                if (
                    not isinstance(experiment_id, str)
                    or EXPERIMENT_ID_RE.fullmatch(experiment_id) is None
                ):
                    raise RuntimeStateError(
                        "Visible Experiment has an invalid identity; restore private Runtime research state."
                    )
                if experiment_ids is not None and experiment_id not in experiment_ids:
                    continue
                visible = _git_reachable_projection(
                    _public_experiment(experiment), commit_reachable
                )
                research_experiment = research_experiments.get(experiment_id)
                if isinstance(research_experiment, Mapping):
                    for field in ("recorded_by", "access"):
                        if field in research_experiment:
                            visible[field] = deepcopy(research_experiment[field])
                publish_if_changed(
                    self.workspace,
                    f"experiments/{experiment_id}/experiment.json",
                    _projection_json(visible),
                )
                _project_experiment_references(
                    self.campaign_root,
                    self.workspace,
                    experiment,
                )
                for record_id in experiment.get("gateway_record_ids", []):
                    if (
                        not isinstance(record_id, str)
                        or GATEWAY_RECORD_ID_RE.fullmatch(record_id) is None
                    ):
                        raise RuntimeStateError(
                            "Visible Experiment cites an invalid Gateway Record identity."
                        )
                    try:
                        _public_gateway_record(store, record_id)
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        raise RuntimeStateError(
                            "Visible Experiment evidence is unavailable; restore its Gateway Record."
                        ) from error

    def _publish_after_mutation(
        self,
        result: dict[str, object],
        *,
        direction_ids: AbstractSet[str] | None = frozenset(),
        experiment_ids: AbstractSet[str] | None = frozenset(),
        record_ids: AbstractSet[str] | None = None,
    ) -> dict[str, object]:
        """Refresh a derived view without making a durable mutation ambiguous."""
        try:
            # Each mutation branch has already observed/synchronized the
            # affected state. Re-synchronizing here could overwrite an explicit
            # local priority order with the last shared Epoch queue.
            self.publish_research_files(
                synchronize=False,
                direction_ids=direction_ids,
                experiment_ids=experiment_ids,
                record_ids=record_ids,
            )
        except (OSError, ValueError, RuntimeStateError) as error:
            # The Journal mutation is already durable. Returning an ordinary
            # failure would invite the Agent to repeat a non-idempotent propose
            # or record operation. Report the stale convenience projection in
            # the successful mutation receipt instead.
            return {
                **result,
                "workspace_projection": {
                    "status": "stale",
                    "message": str(error),
                    "next_action": (
                        "Keep the returned ID. Do not repeat the mutation. Ask the operator to "
                        "repair the workspace projection; Runtime will regenerate PROGRESS.md "
                        "from authoritative state."
                    ),
                },
            }
        return result

    @staticmethod
    def _claim_gateway_ids(value: object) -> set[str]:
        found: set[str] = set()
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key == "gateway_record_ids" and isinstance(item, list):
                    found.update(
                        record_id for record_id in item
                        if isinstance(record_id, str)
                        and GATEWAY_RECORD_ID_RE.fullmatch(record_id) is not None
                    )
                else:
                    found.update(SupervisorJournalService._claim_gateway_ids(item))
        elif isinstance(value, list):
            for item in value:
                found.update(SupervisorJournalService._claim_gateway_ids(item))
        return found

    @staticmethod
    def _claim_kernel_ids(value: object) -> set[str]:
        found: set[str] = set()
        if isinstance(value, Mapping):
            for key, item in value.items():
                if (
                    key in {
                        "kernel_commit_id", "baseline_kernel_commit_id",
                        "candidate_kernel_commit_id", "kernel_id",
                    }
                    and isinstance(item, str)
                    and KERNEL_RECORD_ID_RE.fullmatch(item) is not None
                ):
                    found.add(item)
                else:
                    found.update(SupervisorJournalService._claim_kernel_ids(item))
        elif isinstance(value, list):
            for item in value:
                found.update(SupervisorJournalService._claim_kernel_ids(item))
        return found

    def _copy_claim_reference(self, reference: str, destination: Path) -> bool:
        try:
            relative = relative_path(reference)
        except ValueError:
            return False
        source = self.workspace / relative
        if source.is_file() and not source.is_symlink():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(
                read_regular_bytes(source, limit=MAX_EXPERIMENT_REFERENCE_FILE_BYTES + 1)
            )
            return True
        if not source.is_dir() or source.is_symlink():
            return False
        copied = False
        for child in sorted(source.rglob("*")):
            if child.is_symlink() or not child.is_file():
                continue
            target = destination / child.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(
                read_regular_bytes(child, limit=MAX_EXPERIMENT_REFERENCE_FILE_BYTES + 1)
            )
            copied = True
        return copied

    def _build_claim_review_bundle(
        self,
        root: Path,
        *,
        claim_type: str,
        target: Mapping[str, object],
    ) -> dict[str, object]:
        target = deepcopy(dict(target))
        publish(
            root / "submissions",
            "target.json",
            _projection_json(target),
        )
        documents: list[dict[str, Any]] = [target]
        if claim_type == "direction":
            direction_id = target.get("direction_id")
            existing = _direction_views(self.path, self.campaign_root).get(direction_id)
            if target.get("action") == "propose":
                proposal = target
            elif (
                target.get("action") == _DIRECTION_REVISION_ACTION
                and isinstance(existing, Mapping)
            ):
                # A revision deliberately omits immutable genealogy.  Review
                # the effective Direction rather than the sparse revision
                # event so its Experiment and Gateway evidence remains
                # available to the reviewer.
                proposal = {**deepcopy(dict(existing)), **target}
            else:
                proposal = existing
            proposal = _public_direction(proposal) if isinstance(proposal, Mapping) else target
            documents.append(proposal)
            publish(
                root / "submissions",
                "direction-proposal.json",
                _projection_json(proposal),
            )
            # Direction evidence may arrive through terminal-support edges or
            # through the Experiments from which a new Direction was derived.
            # Resolve both before collecting Gateway Records; copying only the
            # Direction event leaves the reviewer with IDs but no measurements.
            experiment_ids: list[str] = []
            for document in (proposal, target):
                if not isinstance(document, Mapping):
                    continue
                for field in (
                    "supporting_experiment_ids",
                    "derived_from_experiment_ids",
                ):
                    values = document.get(field, [])
                    if isinstance(values, list):
                        experiment_ids.extend(
                            value for value in values if isinstance(value, str)
                        )
            experiments = {
                item.get("experiment_id"): item
                for item in _visible_experiments(self.path, self.campaign_root)
                if isinstance(item, dict)
            }
            for experiment_id in dict.fromkeys(experiment_ids):
                experiment = experiments.get(experiment_id)
                if not isinstance(experiment, dict):
                    continue
                public = _public_experiment(experiment)
                documents.append(public)
                publish(
                    root / "submissions" / "supporting-experiments",
                    f"{experiment_id}.json",
                    _projection_json(public),
                )
        elif claim_type == "experiment":
            direction_id = target.get("direction_id")
            if isinstance(direction_id, str):
                direction = _direction_views(self.path, self.campaign_root).get(direction_id)
                if isinstance(direction, Mapping):
                    publish(
                        root / "submissions",
                        "direction-proposal.json",
                        _projection_json(_public_direction(direction)),
                    )
        else:
            raise RuntimeStateError("Research claim type is invalid")

        store = MeasurementStore(self.campaign_root.parent / "measurements")
        public_records: list[dict[str, Any]] = []
        missing_records: list[str] = []
        for record_id in sorted(self._claim_gateway_ids(documents)):
            try:
                record = _public_gateway_record(store, record_id)
            except (OSError, ValueError, KeyError, TypeError):
                missing_records.append(record_id)
                continue
            public_records.append(record)
            publish(
                root / "gateway-records" / record_id,
                "record.json",
                _projection_json(record),
            )

        missing_kernels: list[str] = []
        from aka.light.workspace.sources import unpack
        for kernel_id in sorted(self._claim_kernel_ids(public_records)):
            try:
                _policy, files = unpack(store.read_kernel(kernel_id))
            except (OSError, ValueError, KeyError, TypeError):
                missing_kernels.append(kernel_id)
                continue
            for name, data in files.items():
                publish(root / "kernels" / kernel_id, name, data)

        missing_references: list[str] = []
        for document in documents:
            references = document.get("references", [])
            if not isinstance(references, list):
                continue
            for reference in references:
                if not isinstance(reference, str):
                    continue
                if not self._copy_claim_reference(
                    reference,
                    root / "references" / reference,
                ):
                    missing_references.append(reference)
        manifest = {
            "schema": "atrex.research-claim-review-input.v1",
            "claim_type": claim_type,
            "gateway_record_count": len(public_records),
            "missing_gateway_records": missing_records,
            "missing_kernels": missing_kernels,
            "missing_references": sorted(set(missing_references)),
        }
        publish(root, "evidence-manifest.json", _projection_json(manifest))
        return manifest

    def _review_claim_before_mutation(
        self,
        claim_type: str,
        target: Mapping[str, object],
    ) -> dict[str, object]:
        """Review one validated claim before it becomes canonical research state."""
        if self.claim_reviewer is None:
            return {}
        try:
            with tempfile.TemporaryDirectory(
                prefix="claim-review-", dir=self.evidence_root
            ) as directory:
                root = Path(directory)
                manifest = self._build_claim_review_bundle(
                    root, claim_type=claim_type, target=target
                )
                from aka.light.workflow.claim_review import validate_review
                resume_session_id = self._prior_claim_review_session(
                    claim_type, target
                )
                try:
                    reviewer_parameters = inspect.signature(
                        self.claim_reviewer
                    ).parameters.values()
                except (TypeError, ValueError):
                    reviewer_parameters = ()
                supports_resume = any(
                    parameter.name == "resume_session_id"
                    or parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in reviewer_parameters
                )
                raw_review = (
                    self.claim_reviewer(
                        root, resume_session_id=resume_session_id
                    )
                    if supports_resume else self.claim_reviewer(root)
                )
                if not isinstance(raw_review, Mapping):
                    raise ValueError("research review must be a JSON object")
                review_value = dict(raw_review)
                session_audit = _claim_review_session_audit(
                    review_value.pop(_REVIEW_SESSION_AUDIT_KEY, None)
                )
                review = validate_review(review_value)
        except Exception as error:
            session_audit = _claim_review_session_audit(
                getattr(error, "review_session_audit", None)
            )
            review_receipt: dict[str, object] = {
                "archive_status": "archived",
                "claim_review_quality": "unreviewed",
                "evidence_eligible": False,
                "runtime_action": (
                    "The reviewer is unavailable. Runtime will preserve this record as unreviewed; "
                    "do not treat it as established research. Continue with bounded evidence or "
                    "ask the operator to restore the claim reviewer."
                ),
            }
            if session_audit is not None:
                review_receipt[_REVIEW_SESSION_AUDIT_KEY] = session_audit
        else:
            quality = review["verdict"]
            evidence_eligible = quality == "supported"
            review_receipt = {
                "archive_status": (
                    "not_archived" if quality == "unsupported" else "archived"
                ),
                "claim_review_quality": quality,
                "evidence_eligible": evidence_eligible,
                "primary_issue": review["primary_issue"],
                "findings": review["findings"],
                "scope_safe_interpretation": review["scope_safe_interpretation"],
                "scientific_next_check": review["next_check"],
                "confidence": review["confidence"],
                "runtime_action": (
                    "Use the archived record as reviewed research evidence."
                    if evidence_eligible else
                    (
                        "This submission is unsupported and will not be archived. Follow "
                        "scope_safe_interpretation, run scientific_next_check when useful, and submit a "
                        "corrected claim."
                        if quality == "unsupported" else
                        "This partially supported record will be archived, but is not established "
                        "research. Follow scope_safe_interpretation, run scientific_next_check when useful, "
                        "and resubmit a corrected claim through the same Runtime operation. For "
                        "an Experiment, include its stable experiment_id to create a new revision."
                    )
                ),
            }
            review_receipt["evidence_manifest"] = manifest
            if session_audit is not None:
                review_receipt[_REVIEW_SESSION_AUDIT_KEY] = session_audit
            self._remember_claim_review_session(
                claim_type, target, review_receipt
            )
        return review_receipt

    def _persist_claim_review(
        self,
        result: dict[str, object],
        *,
        claim_type: str,
        audit_id: str,
    ) -> dict[str, object]:
        """Archive a completed/unavailable review after its claim is durable."""
        review = result.get("review")
        if not isinstance(review, Mapping):
            return result
        review_receipt = deepcopy(dict(review))
        review_id = "review_" + uuid.uuid4().hex
        archive_path = str(result["archive_path"])
        private_review = _private_claim_review(review_receipt)
        review_archive_path = f"claim-reviews/{review_id}.json"
        claim_revision = result.get("revision")
        revision_row = (
            {"claim_revision": claim_revision}
            if type(claim_revision) is int and claim_revision > 0 else {}
        )
        private_write(
            self.evidence_root / review_archive_path,
            {
                "review_id": review_id,
                "claim_type": claim_type,
                "claim_archive_path": archive_path,
                **revision_row,
                **private_review,
            },
        )
        link: dict[str, object] = {
            "schema": "atrex.research-claim-audit-review-link.v1",
            "review_id": review_id,
            "review_archive_path": review_archive_path,
            "claim_archive_path": archive_path,
            **revision_row,
        }
        if "session_audit" in private_review:
            link["session_audit"] = deepcopy(private_review["session_audit"])
        private_write(
            self.evidence_root / "claim-audit" / audit_id / "review.json",
            link,
        )
        return {
            **result,
            "review": _agent_claim_review(review_receipt),
        }

    def sealed_source(self) -> bytes | None:
        if not self.supervisor_git:
            return None
        return sealed_candidate_source(load_journal(self.path), self.git_workspace)

    def execute(
        self, request: Mapping[str, object], *,
        validate_candidate: Callable[[bytes], dict[str, Any]] | None = None,
        agent_kernel_id: str | None = None,
        seal_candidate: Callable[[str, bytes], str] | None = None,
    ) -> dict[str, object]:
        operation = request.get("operation")
        if not isinstance(operation, str):
            raise ValueError("Runtime research/report operation must be text")
        fields = {
            "direction_update": "request",
            "experiment_record": "request",
            "episode_report": "request",
            "direction_prioritize": "request",
        }
        if operation not in fields:
            raise ValueError(f"unsupported Runtime research/report operation: {operation}")
        journal = load_journal(self.path)
        module = ("directions" if operation in {"direction_update",
                                                  "direction_prioritize"}
                  else "experiments" if operation == "experiment_record"
                  else None)
        if module is not None and not _module_enabled(journal, module):
            raise AgentRequestError(
                f"{module.capitalize()} tools are disabled for this Episode",
                code=f"{module}_disabled",
                next_action="Use only the enabled Runtime tools and follow the current episode-report schema.",
            )
        required = {"operation"}
        required.add(fields[operation])
        require_fields(dict(request), required, label="Runtime research/report operation")
        if operation in {"direction_update", "experiment_record", "episode_report"}:
            self._check_legacy_journal()
        if operation == "direction_update":
            body = request.get("request")
            audit_id = self._begin_claim_audit("direction", body)
            try:
                if not isinstance(body, Mapping):
                    raise ValueError("direction_update requires a request object")
                result = update_direction(
                    self.path,
                    self.campaign_root,
                    body,
                    review_claim=(
                        self._review_claim_before_mutation
                        if self.claim_reviewer is not None else None
                    ),
                )
                direction_id = str(result["direction_id"])
                self._observe_direction(direction_id)
                if body.get("action") == "start":
                    self.research.focus_direction(direction_id, origin=self.origin)
                published = self._publish_after_mutation(
                    result, direction_ids={direction_id}
                )
                final = self._persist_claim_review(
                    published, claim_type="direction", audit_id=audit_id
                )
            except Exception as error:
                response = getattr(error, "response", {})
                code = (
                    response.get("error", {}).get("code")
                    if isinstance(response, Mapping) else None
                )
                self._finish_claim_audit(
                    audit_id,
                    status=(
                        "unsupported"
                        if code == "research_claim_unsupported" else
                        "rejected"
                        if isinstance(error, (AgentRequestError, ValueError)) else
                        "runtime_error"
                    ),
                    error=error,
                )
                raise
            self._finish_claim_audit(audit_id, status="archived", result=final)
            return final
        if operation == "direction_prioritize":
            body = request.get("request")
            audit_id = self._begin_claim_audit("direction_priority", body)
            try:
                if not isinstance(body, Mapping):
                    raise ValueError("direction_prioritize requires a request object")
                require_fields(dict(body), {"direction_id", "reason"}, optional={"before_direction_id", "after_direction_id"},
                               label="Direction priority")
                self._sync_research()
                direction_id = _text(body.get("direction_id"), "Direction ID")
                current = self.research.read().get("directions", {}).get(direction_id)
                if isinstance(current, Mapping) and current.get("access") == "peer_read_only":
                    raise AgentRequestError(
                        "A shared peer Direction is not part of the local work queue.",
                        code="shared_direction_read_only",
                        direction_id=direction_id,
                        next_action=(
                            "Keep peer ordering in the Shared direction pool. To prioritize execution, "
                            "synthesize compatible peer ideas into a local Direction, cite their IDs, "
                            "and reorder that local identity."
                        ),
                    )
                state = self.research.prioritize(
                    direction_id,
                    before_direction_id=(
                        _text(body["before_direction_id"], "before_direction_id")
                        if body.get("before_direction_id") is not None else None
                    ),
                    after_direction_id=(
                        _text(body["after_direction_id"], "after_direction_id")
                        if body.get("after_direction_id") is not None else None
                    ),
                    reason=_text(body.get("reason"), "Direction priority reason"),
                    origin=self.origin,
                )
                final = self._publish_after_mutation(
                    {"status": "recorded", "queue": state["queue"]},
                    direction_ids=None,
                )
            except Exception as error:
                self._finish_claim_audit(
                    audit_id,
                    status=(
                        "rejected"
                        if isinstance(error, (AgentRequestError, ValueError)) else
                        "runtime_error"
                    ),
                    error=error,
                )
                raise
            self._finish_claim_audit(audit_id, status="archived", result=final)
            return final
        if operation == "experiment_record":
            body = request.get("request")
            audit_id = self._begin_claim_audit("experiment", body)
            try:
                if not isinstance(body, Mapping):
                    raise ValueError("experiment_record requires a request object")
                body = dict(body)
                result = record_experiment(
                    self.path,
                    self.campaign_root,
                    body,
                    archive_references=lambda experiment_id, revision, references: (
                        _archive_experiment_references(
                            self.workspace,
                            self.campaign_root,
                            experiment_id,
                            revision,
                            references,
                        )
                    ),
                    review_claim=(
                        self._review_claim_before_mutation
                        if self.claim_reviewer is not None else None
                    ),
                )
                experiment_id = str(result["experiment_id"])
                self._observe_experiment(experiment_id)
                observed = self.research.read().get("experiments", {}).get(
                    experiment_id, {}
                )
                direction_id = (
                    observed.get("direction_id")
                    if isinstance(observed, Mapping)
                    else None
                )
                affected_directions = (
                    {direction_id} if isinstance(direction_id, str) else frozenset()
                )
                published = self._publish_after_mutation(
                    result,
                    direction_ids=affected_directions,
                    experiment_ids={experiment_id},
                )
                final = self._persist_claim_review(
                    published, claim_type="experiment", audit_id=audit_id
                )
            except Exception as error:
                response = getattr(error, "response", {})
                code = (
                    response.get("error", {}).get("code")
                    if isinstance(response, Mapping) else None
                )
                self._finish_claim_audit(
                    audit_id,
                    status=(
                        "unsupported"
                        if code == "research_claim_unsupported" else
                        "rejected"
                        if isinstance(error, (AgentRequestError, ValueError)) else
                        "runtime_error"
                    ),
                    error=error,
                )
                raise
            self._finish_claim_audit(audit_id, status="archived", result=final)
            return final
        if operation == "episode_report":
            body = request.get("request")
            if not isinstance(body, Mapping):
                raise ValueError("episode_report requires a request object")
            return self._submit_report(
                body,
                validate_candidate=validate_candidate,
                agent_kernel_id=agent_kernel_id,
                seal_candidate=seal_candidate,
            )
        raise ValueError(f"unsupported Runtime research/report operation: {operation}")

    def _check_legacy_journal(self) -> None:
        from aka.light.agent.session_tail import read_regular_bytes

        try:
            raw = read_regular_bytes(
                self.git_workspace / ".atrex_long_horizon/journal.json", limit=MAX_FILE_BYTES + 1
            )
        except FileNotFoundError:
            return
        except (OSError, ValueError) as error:
            if isinstance(error, OSError) and error.errno not in {
                errno.ELOOP,
                errno.ENOTDIR,
                errno.EISDIR,
                errno.ENAMETOOLONG,
            }:
                raise RuntimeStateError(
                    "Legacy .atrex_long_horizon/journal.json cannot be read because storage access failed",
                    code="legacy_journal_unavailable",
                ) from error
            raise AgentRequestError(
                "Legacy .atrex_long_horizon/journal.json is invalid; repair it before using "
                "Runtime research/report tools",
                code="legacy_journal_invalid",
                next_action=(
                    "Make .atrex_long_horizon a real workspace directory and journal.json "
                    "a regular JSON file, with no symlinks in either path. Preserve existing "
                    "file content. No research/report mutation was applied; retry the request "
                    "after repairing the path."
                ),
            ) from error
        if len(raw) > MAX_FILE_BYTES:
            raise ValueError("Legacy .atrex_long_horizon/journal.json exceeds its size limit")
        try:
            legacy = json.loads(raw)
        except ValueError as error:
            raise ValueError(
                "Legacy .atrex_long_horizon/journal.json is invalid; repair it before using "
                "Runtime research/report tools"
            ) from error
        if not isinstance(legacy, dict):
            raise ValueError("Legacy .atrex_long_horizon/journal.json must contain a JSON object")
        current = load_journal(self.path)
        if legacy.get("runtime_journal_projection") is True:
            expected_public_base = (
                current.get("base_kernel_id") or current["base_commit"]
            )
            if (
                legacy.get("episode") != current["episode"]
                or legacy.get("base_commit") != expected_public_base
                or legacy.get("episode_branch") != current["episode_branch"]
            ):
                raise ValueError(
                    "Legacy .atrex_long_horizon/journal.json projection belongs to a different Episode"
                )
        elif legacy.get("experiments") or legacy.get("state", "in_progress") != "in_progress":
            raise AgentRequestError(
                "This Episode already uses the legacy .atrex_long_horizon/journal.json protocol; "
                "current Runtime research/report mutations "
                "cannot be used in the same Episode",
                code="journal_protocol_conflict",
                repairable=False,
                next_action=(
                    "Preserve the workspace and legacy .atrex_long_horizon/journal.json unchanged. "
                    "This Light Session no "
                    "longer exposes the legacy append/finalize protocol, so the operator must "
                    "either resume the Episode with a compatible legacy Runtime or start a fresh "
                    "Campaign and import the retained Kernel source. Do not retry current Runtime "
                    "research/report mutations, erase Experiments, or reset the legacy file to "
                    "switch protocols."
                ),
            )

    def _publish_report(self, current: dict[str, Any]) -> None:
        try:
            self._publish_legacy_report(current)
        except OSError as error:
            raise RuntimeStateError(
                "Report is durably recorded but legacy handoff publication failed. "
                "Restore storage access, then resubmit the identical episode-report; "
                "do not rerun measurements",
                code="report_publication_failed",
            ) from error
        except Exception as error:
            raise RuntimeStateError(
                "Report is durably recorded but its legacy handoff projection failed. "
                "This is a Supervisor compatibility error; preserve the candidate and "
                "evidence and ask the operator to repair the Runtime before resubmitting "
                "the identical episode-report",
                code="report_projection_failed",
            ) from error

    def _publish_legacy_report(self, current: dict[str, Any]) -> None:
        """Adapt the private report to existing Long Horizon consumers, without Git writes."""
        projection = dict(current, schema_version=1, runtime_journal_projection=True)
        public_candidate_commit = current.get("candidate_kernel_id")
        if not public_candidate_commit and not self.supervisor_git:
            # Legacy Agent-owned Journals already store their public Git ID in
            # candidate_commit and have no separate Supervisor archive.
            public_candidate_commit = current.get("candidate_commit")
        if "candidate_kernel_id" in projection:
            projection["candidate_kernel_commit_id"] = projection.pop(
                "candidate_kernel_id"
            )
        # ``current.candidate_commit`` is a Supervisor-private archival commit.
        # The public projection and handoff use only the Agent's own Git ID.
        projection["candidate_commit"] = public_candidate_commit
        projection["base_commit"] = (
            current.get("base_kernel_id") or current.get("base_commit")
        )
        projection.pop("base_kernel_id", None)
        projection.pop("runtime_managed", None)
        projection.pop("submitted_report", None)
        projection.pop("pending_candidate_archive", None)
        experiments = []
        selected = current["outcome"].get("selected_experiment_id")
        projection["outcome"] = dict(current["outcome"])
        for index, item in enumerate(current["experiments"], 1):
            row = dict(
                {
                    key: value
                    for key, value in item.items()
                    if key not in {"evidence", "action"}
                },
                # Compact Experiments renamed the Agent-authored conclusion to
                # ``interpretation``.  Keep old finalized Journals readable while
                # projecting every current Experiment to the legacy ``result`` key.
                result=(
                    item["interpretation"]
                    if "interpretation" in item
                    else item["analysis"]
                ),
                # Historical Experiments carried their own timestamp. Compact
                # Experiments rely on the authoritative Journal/event audit
                # time instead of duplicating it in every semantic record.
                timestamp=item.get("recorded_at", current["created_at"]),
                decision=(
                    "keep_as_best" if item["experiment_id"] == selected else "discard"
                ),
            )
            # Legacy reports retain the IDs and analysis. Measurement truth is
            # read from the private store, never parsed from the Agent's prose.
            for record_id in reversed(item["gateway_record_ids"]):
                record = _read_private_gateway_evidence(self.campaign_root, record_id)
                if item["experiment_id"] == selected and not _passing_evaluate(
                    record,
                    current["candidate_kernel_digest"],
                    kernel_id=current.get("candidate_kernel_id"),
                ):
                    continue
                if record["gateway_kind"] == "run" and _completed_record(record):
                    result = record["result"]
                    row["evaluation"] = {
                        "correctness": "pass" if result.get("all_pass") is True else "fail",
                        "latency_us_geomean": result.get("latency_us_geomean"),
                        "latency_us_by_shape": result.get("latency_us_by_shape", {}),
                    }
                    break
            experiments.append(row)
            if item["experiment_id"] == selected:
                projection["outcome"]["selected_experiment_index"] = index
        projection["experiments"] = experiments
        publish(
            self.git_workspace,
            ".atrex_long_horizon/journal.json",
            (json.dumps(projection, ensure_ascii=False, allow_nan=False) + "\n").encode(),
        )
        handoff = {"status": current["state"]}
        if public_candidate_commit:
            handoff["candidate_commit"] = public_candidate_commit
        # Handoff is the final publication: a failed journal write must not
        # signal completion. Retrying the same report republishes these files.
        publish(
            self.git_workspace,
            ".atrex_long_horizon/handoff.json",
            (json.dumps(handoff) + "\n").encode(),
        )

    def _submit_report(
        self, raw: Mapping[str, object], *,
        validate_candidate: Callable[[bytes], dict[str, Any]] | None = None,
        agent_kernel_id: str | None = None,
        seal_candidate: Callable[[str, bytes], str] | None = None,
    ) -> dict[str, object]:
        persisted = load_journal(self.path)
        experiments_enabled = _module_enabled(persisted, "experiments")
        directions_enabled = _module_enabled(persisted, "directions")
        # An accepted report is immutable and exactly replayable for publication
        # recovery.  Check this before the current request schema so a report
        # accepted by an older Runtime remains recoverable after an upgrade.
        if persisted.get("state") != "in_progress":
            if persisted.get("submitted_report") != dict(raw):
                raise AgentRequestError(
                    "This Episode already accepted a different report",
                    code="journal_finalized",
                    repairable=False,
                )
            self._publish_report(persisted)
            return {"status": "accepted", "message": "Report already recorded; handoff republished"}
        allowed = {
            "status",
            "summary",
            "handoff_reason",
            "selected_experiment_id",
            "blocker",
            "accepted_ppu_diagnostics",
            "candidate_commit",
        }
        if self.supervisor_git:
            allowed.remove("candidate_commit")
        if not experiments_enabled:
            allowed.remove("selected_experiment_id")
            allowed.add("gateway_record_ids")
        require_fields(
            dict(raw),
            {"status", "summary", "handoff_reason"},
            optional=allowed,
            label="Episode Report",
        )
        status = raw.get("status")
        if not isinstance(status, str) or status not in {
            "candidate_ready",
            "pivot",
            "blocked",
        }:
            raise ValueError("Episode Report status must be candidate_ready, pivot, or blocked")
        pending_archive = persisted.get("pending_candidate_archive")
        if status != "candidate_ready" and isinstance(pending_archive, Mapping):
            pending_kernel_id = pending_archive.get("agent_kernel_id")
            raise RuntimeStateError(
                "A previously submitted candidate_ready report passed evidence validation, "
                "but the Supervisor has not yet sealed its private Kernel archive. The "
                "Episode remains open and cannot be finalized as blocked or pivoted while "
                "that candidate is pending archival recovery.",
                code="candidate_archive_pending",
                next_action=(
                    "Preserve the clean candidate HEAD and its Gateway Records. After the "
                    "Supervisor archival fault is repaired, resubmit the same candidate_ready "
                    "episode-report; do not submit blocked or pivot to work around a Runtime "
                    "archive failure."
                    + (
                        f" Pending Agent Kernel commit: {pending_kernel_id}."
                        if isinstance(pending_kernel_id, str) else ""
                    )
                ),
            )
        summary = _text(raw.get("summary"), "Episode Report summary")
        handoff_reason = raw.get("handoff_reason")
        if not isinstance(handoff_reason, str) or handoff_reason not in HANDOFF_REASONS:
            raise field_error(
                "handoff_reason",
                "Episode Report handoff_reason must be direction_terminal, "
                "cross_trajectory_share, or blocked",
                code="handoff_reason_invalid",
            )
        if (status == "blocked") != (handoff_reason == "blocked"):
            raise field_error(
                "handoff_reason",
                "blocked reports require handoff_reason=blocked; non-blocked reports "
                "must use direction_terminal or cross_trajectory_share",
                code="handoff_reason_mismatch",
            )
        blocker = raw.get("blocker")
        if status == "blocked":
            _text(blocker, "Episode Report blocker")
        elif blocker is not None and blocker != "":
            raise ValueError(f"Episode Report status {status} cannot include blocker")
        selected_id = raw.get("selected_experiment_id")
        if status == "candidate_ready" and (
            not isinstance(agent_kernel_id, str)
            or KERNEL_RECORD_ID_RE.fullmatch(agent_kernel_id) is None
        ):
            raise AgentRequestError(
                "candidate_ready requires a clean committed Agent Kernel; HEAD is its Kernel commit ID.",
                code="kernel_commit_required",
                next_action=(
                    "Commit the complete Kernel source tree, verify git status is clean, "
                    "then resubmit episode-report."
                ),
            )
        from aka.light.workspace.sources import policy_at
        current, directions, kernel_source = validate_report_evidence(
            self.path,
            self.campaign_root,
            self.workspace,
            status=status,
            selected_id=selected_id,
            gateway_record_ids=raw.get("gateway_record_ids"),
            source_policy=policy_at(self.git_workspace, committed=True),
            kernel_id=agent_kernel_id,
        )
        if (
            status == "candidate_ready"
            and kernel_source is not None
            and (self.git_workspace / ".git").exists()
        ):
            # A no-change outcome is scientifically valid, but it is a pivot,
            # not a new candidate.  Reject it before correctness GPU work or
            # commit sealing, and keep the Episode open so the Agent can submit
            # the already-supported no-candidate terminal report.
            from aka.light.workspace.sources import from_git

            incumbent_source = from_git(
                self.git_workspace, current["base_commit"]
            )
            if kernel_source == incumbent_source:
                raise AgentRequestError(
                    "candidate_ready source is identical to the incumbent comparison base.",
                    code="candidate_matches_incumbent",
                    next_action=(
                        "Do not make a nominal source edit to manufacture a difference. If the "
                        "incumbent remains best after the recorded work, submit "
                        "{\"status\":\"pivot\",\"handoff_reason\":\"direction_terminal\","
                        "\"summary\":\"<evidence-backed conclusion>\"}. A pivot nominates no "
                        "candidate and needs no selected_experiment_id or gateway_record_ids; "
                        "local Directions may remain in_progress for a later Episode."
                    ),
                )
        if status == "candidate_ready" and experiments_enabled and not any(
            isinstance(item, Mapping)
            and item.get("experiment_id") == selected_id
            for item in current["experiments"]
        ):
            # Evidence preflight deliberately permits a decision-supported visible
            # Experiment from an earlier Episode when its passing full Evaluate
            # still binds the exact candidate commit.  Finalized Journals and the
            # compatibility handoff are Episode-local, however, so retain the
            # selected immutable Experiment in this Episode before publication.
            # This is a projection of already validated evidence, not a revision
            # or a second claim-review submission.
            reusable = [
                item
                for item in _visible_experiments(self.path, self.campaign_root)
                if item.get("experiment_id") == selected_id
            ]
            if len(reusable) != 1:
                raise RuntimeStateError(
                    "Validated historical Experiment is no longer uniquely visible; "
                    "the report was not accepted"
                )
            current["experiments"].append(deepcopy(reusable[0]))
        if handoff_reason == "cross_trajectory_share":
            if self.origin.get("cross_trajectory_sharing") is not True:
                raise AgentRequestError(
                    "This Episode is not running inside a cross-trajectory sharing search.",
                    code="cross_trajectory_sharing_unavailable",
                    handoff_reason=handoff_reason,
                    next_action=(
                        "Continue the current work in this Episode, use direction_terminal for an "
                        "ordinary measured handoff, or report a real blocker."
                    ),
                )
            if status != "candidate_ready":
                raise field_error(
                    "handoff_reason",
                    "cross_trajectory_share requires status=candidate_ready and a reusable "
                    "checkpoint backed by a passing standard full Evaluate of the exact "
                    "committed source",
                    code="handoff_reason_mismatch",
                )
        if directions_enabled:
            history = _visible_history(self.path, self.campaign_root)
            local_active = sorted(
                direction_id
                for direction_id, item in directions.items()
                if direction_id not in history.shared_direction_ids
                and item.get("status") == "in_progress"
            )
            selected_direction_id = None
            if status == "candidate_ready" and experiments_enabled:
                visible_experiments = {
                    item.get("experiment_id"): item
                    for item in _visible_experiments(
                        self.path, self.campaign_root, history=history
                    )
                    if isinstance(item, Mapping)
                    and isinstance(item.get("experiment_id"), str)
                }
                visible_experiments.update({
                    item.get("experiment_id"): item
                    for item in current.get("experiments", [])
                    if isinstance(item, Mapping)
                    and isinstance(item.get("experiment_id"), str)
                })
                selected_experiment = visible_experiments.get(selected_id)
                selected_direction_id = (
                    selected_experiment.get("direction_id")
                    if isinstance(selected_experiment, Mapping) else None
                )
            if handoff_reason == "direction_terminal":
                if (
                    status == "candidate_ready"
                    and experiments_enabled
                    and (
                        not isinstance(selected_direction_id, str)
                        or directions.get(selected_direction_id, {}).get("status")
                        not in {"in_progress", "completed", "abandoned"}
                    )
                ):
                    raise AgentRequestError(
                        "The nominated Experiment must belong to an in-progress, completed, "
                        "or abandoned visible Direction for this handoff.",
                        code="handoff_direction_mismatch",
                        selected_experiment_id=selected_id,
                        selected_direction_id=selected_direction_id,
                        next_action=(
                            "Select a supported visible Experiment attached to an in-progress, "
                            "completed, or abandoned Direction and the exact candidate HEAD. "
                            "Historical evidence is reusable when its full Evaluate still matches "
                            "that commit."
                        ),
                    )
            elif handoff_reason == "blocked" and local_active:
                raise AgentRequestError(
                    "A blocked handoff requires all local active Directions to be closed with block.",
                    code="handoff_not_ready",
                    handoff_reason=handoff_reason,
                    active_direction_ids=local_active,
                    next_action=(
                        "Record the concrete blocker by closing every local active Direction with "
                        "block, then resubmit the blocked episode-report."
                    ),
                )
            elif (
                handoff_reason == "cross_trajectory_share"
                and status == "candidate_ready"
                and experiments_enabled
                and (
                    not isinstance(selected_direction_id, str)
                    or directions.get(selected_direction_id, {}).get("status")
                    != "in_progress"
                )
            ):
                raise AgentRequestError(
                    "A cross-trajectory checkpoint must be owned by an in-progress Direction.",
                    code="handoff_direction_mismatch",
                    selected_experiment_id=selected_id,
                    selected_direction_id=selected_direction_id,
                    next_action=(
                        "Select the measured Experiment for the active Direction being shared, "
                        "or use direction_terminal for an ordinary handoff."
                    ),
                )
        candidate_commit = raw.get("candidate_commit") or ""
        if status == "candidate_ready" and not self.supervisor_git:
            if (
                not isinstance(candidate_commit, str)
                or re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", candidate_commit) is None
            ):
                raise ValueError(
                    "candidate_ready requires the full candidate_commit from git rev-parse HEAD; "
                    "commit the complete Kernel source tree before reporting"
                )
            from aka.light.workspace.git import EpisodeWorktree

            violation, _ = EpisodeWorktree(
                current["episode"],
                current["base_commit"],
                current["episode_branch"],
                self.git_workspace,
            ).validate_candidate(candidate_commit)
            if violation:
                raise ValueError(violation)
        elif candidate_commit:
            raise ValueError(f"{status} cannot include candidate_commit")
        next_directions = [
            item["name"]
            for item in directions.values()
            if item["status"] in {"proposed", "deferred"}
        ]
        accepted_ppu_diagnostics: list[dict[str, Any]] = []
        if "accepted_ppu_diagnostics" in raw:
            from aka.light.workflow.journal_projection import (
                normalize_accepted_ppu_diagnostics,
                validate_accepted_ppu_evidence,
            )

            accepted_ppu_diagnostics, errors = normalize_accepted_ppu_diagnostics(
                raw["accepted_ppu_diagnostics"]
            )
            if not errors:
                errors.extend(
                    validate_accepted_ppu_evidence(accepted_ppu_diagnostics, self.workspace)
                )
            if errors:
                raise ValueError("invalid accepted_ppu_diagnostics: " + "; ".join(errors))
        if kernel_source is not None:
            if validate_candidate is None:
                raise RuntimeStateError("Supervisor candidate validation is unavailable; report was not accepted")
            # All report/evidence checks precede GPU work. The callback measures
            # these frozen bytes; only a pass may finalize or commit the report.
            current["acceptance_checks"] = {
                "five_case_correctness": validate_candidate(kernel_source),
            }
        if kernel_source is not None and self.supervisor_git:
            if seal_candidate is None or agent_kernel_id is None:
                raise RuntimeStateError(
                    "Supervisor Agent-commit sealing is unavailable; report was not accepted"
                )
            try:
                candidate_commit = seal_candidate(agent_kernel_id, kernel_source)
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                current["pending_candidate_archive"] = {
                    "agent_kernel_id": agent_kernel_id,
                    "candidate_kernel_digest": (
                        "sha256:" + hashlib.sha256(kernel_source).hexdigest()
                    ),
                    "submitted_report": dict(raw),
                    "failed_at": _now(),
                    "failure": f"{type(error).__name__}: {error}"[:2000],
                }
                _save_journal(self.path, self.campaign_root, current)
                raise RuntimeStateError(
                    "Supervisor could not seal the evidence-complete candidate into its "
                    "private Kernel archive. The Episode remains open for an identical "
                    "candidate_ready retry.",
                    code="candidate_archive_unavailable",
                    next_action=(
                        "Preserve the clean candidate HEAD and Gateway Records. Retry the same "
                        "candidate_ready episode-report after the Supervisor archive is "
                        "available; do not submit blocked or pivot as a fallback."
                    ),
                ) from error
            from aka.light.workspace.git import EpisodeWorktree

            violation, _ = EpisodeWorktree(
                current["episode"], current["base_commit"], current["episode_branch"],
                self.git_workspace,
            ).validate_candidate(candidate_commit)
            if violation:
                from aka.light.workspace.git import UNCHANGED_CANDIDATE

                if violation == UNCHANGED_CANDIDATE:
                    raise AgentRequestError(
                        "candidate_ready source is identical to the incumbent comparison base.",
                        code="candidate_matches_incumbent",
                        next_action=(
                            "Do not report an unchanged source as candidate_ready or make a "
                            "nominal edit. Submit status=pivot with "
                            "handoff_reason=direction_terminal; local Directions may remain "
                            "in_progress, and candidate evidence fields must be omitted."
                        ),
                    )
                current["pending_candidate_archive"] = {
                    "agent_kernel_id": agent_kernel_id,
                    "candidate_kernel_digest": (
                        "sha256:" + hashlib.sha256(kernel_source).hexdigest()
                    ),
                    "submitted_report": dict(raw),
                    "failed_at": _now(),
                    "failure": violation[:2000],
                }
                _save_journal(self.path, self.campaign_root, current)
                raise RuntimeStateError(
                    "Supervisor rejected its private Kernel archive: " + violation,
                    code="candidate_archive_unavailable",
                    next_action=(
                        "Preserve the clean candidate HEAD and Gateway Records. Retry the same "
                        "candidate_ready episode-report after the Supervisor archive is "
                        "repaired; do not submit blocked or pivot as a fallback."
                    ),
                )
            current.pop("pending_candidate_archive", None)
        current["state"] = status
        current["outcome"] = {
            "summary": summary,
            "handoff_reason": handoff_reason,
            "next_directions": next_directions,
            "selected_experiment_id": selected_id if status == "candidate_ready" else None,
            "blocker": str(blocker).strip() if status == "blocked" else None,
        }
        if not experiments_enabled:
            current["outcome"]["gateway_record_ids"] = (
                _text_list(raw["gateway_record_ids"], "Episode Report gateway_record_ids")
                if status == "candidate_ready" else []
            )
        if accepted_ppu_diagnostics:
            current["outcome"]["accepted_ppu_diagnostics"] = accepted_ppu_diagnostics
        current["candidate_commit"] = candidate_commit or None
        if kernel_source is not None:
            current["candidate_kernel_digest"] = (
                "sha256:" + hashlib.sha256(kernel_source).hexdigest()
            )
            current["candidate_kernel_id"] = agent_kernel_id
        current["finalized_at"] = _now()
        current["submitted_report"] = dict(raw)
        _save_journal(self.path, self.campaign_root, current)
        self._publish_report(current)
        return {
            "status": "accepted",
            "kernel_commit_id": agent_kernel_id,
            "message": (
                "Report accepted; Supervisor archived the exact Agent source for verification"
                if candidate_commit and self.supervisor_git else
                "Report accepted; Agent commit queued for existing Supervisor verification"
                if candidate_commit
                else "Report accepted and recorded"
            ),
        }


__all__ = [
    "DIRECTION_CHANGE_SCOPES",
    "DIRECTION_CLOSURE_FIELDS",
    "DIRECTION_CLOSURE_OPTIONAL_FIELDS",
    "DIRECTION_PROPOSAL_FIELDS",
    "DIRECTION_PROPOSAL_OPTIONAL_FIELDS",
    "DIRECTION_REVISION_FIELDS",
    "DIRECTION_UPDATE_FIELDS",
    "EXPERIMENT_FIELDS",
    "EXPERIMENT_OPTIONAL_FIELDS",
    "SupervisorJournalService",
    "copy_experiment_reference_archive",
    "initialize_journal",
    "journal_lock",
    "load_journal",
    "sealed_candidate_source",
    "selected_evaluation",
    "shared_research_queue",
]
