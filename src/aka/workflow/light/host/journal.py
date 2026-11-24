"""Supervisor-owned research transactions and exact-commit submission processing.

The Agent submits interpretation, intent, and explicit Gateway Record references.
The private Gateway store owns their exact Kernel bindings; this transaction layer does not
duplicate source identity or require a before/after comparison structure.
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import math
import os
import re
import shutil
import stat
import subprocess
import tempfile
import threading
import uuid
from collections import ChainMap
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Set as AbstractSet
from copy import copy, deepcopy
from functools import wraps
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from typing import Any

from aka.agent.execution.home import open_private_directory

from aka.workflow.light.host.direction_genealogy import (
    RELATIONSHIP_FIELDS,
    relationship_fields,
    validate_relationship,
)
from aka.workflow.light.host.direction_lifecycle import (CLOSED_STATUSES as _CLOSED_STATUSES)
from aka.workflow.light.host.direction_lifecycle import (CLOSURES as _CLOSURES)
from aka.workflow.light.host.direction_lifecycle import (DIRECTION_ACTIONS as _DIRECTION_ACTIONS)
from aka.workflow.light.host.direction_lifecycle import (DIRECTION_STATUS as _DIRECTION_STATUS)
from aka.workflow.light.host.direction_lifecycle import (DIRECTION_REVISION_ACTION as _DIRECTION_REVISION_ACTION)
from aka.workflow.light.host.direction_lifecycle import (advance_direction)
from aka.workflow.light.host.errors import AgentRequestError, RuntimeStateError, field_error, require_fields
from aka.workflow.light.host.identifiers import (
    DIRECTION_ID_RE,
    EXPERIMENT_ID_RE,
    GATEWAY_RECORD_ID_RE,
    KERNEL_RECORD_ID_RE,
)
from aka.experiment.measurement_store import (
    MeasurementStore,
    digest as evidence_digest,
    private_write,
    private_write_bytes,
    read_json,
)
from aka.evaluation.response import result_from_response
from aka.workflow.light.long_horizon.claim_review import canonical_review_quality
from aka.workflow.light.host.workspace import (
    MAX_FILE_BYTES,
    publish,
    publish_if_changed,
    publish_pinned_if_changed,
    relative_path,
)
from aka.agent.session_tail import read_regular_bytes
from aka.workflow.light.research import ResearchStore

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
DIRECTION_REVISION_OPTIONAL_FIELDS = DIRECTION_PROPOSAL_OPTIONAL_FIELDS | {"analysis"}
DIRECTION_CHANGE_SCOPES = {"incremental", "major_redesign"}
DIRECTION_UPDATE_FIELDS = {"action", "direction_id"}
DIRECTION_CLOSURE_FIELDS = DIRECTION_UPDATE_FIELDS | {"analysis"}
DIRECTION_CLOSURE_OPTIONAL_FIELDS = {"reopen_condition", "supporting_experiment_ids"}
_SHARED_GATEWAY_RECORD_REASONS = frozenset({
    "visible_direction_reference",
    "visible_experiment_reference",
    "passing_full_evaluate",
    "accepted_submission_full_evaluate",
    "lineage_winner_full_evaluate",
    "global_winner_full_evaluate",
})
EXPERIMENT_FIELDS = {"name", "intent", "interpretation"}
EXPERIMENT_OPTIONAL_FIELDS = {
    "experiment_id", "gateway_record_ids", "references", "implementation", "expectations", "next_step",
}
EXPERIMENT_EXPECTATION_FIELDS = {"scope", "prediction", "observation", "assessment"}
EXPERIMENT_EXPECTATION_ASSESSMENTS = frozenset({"met", "missed", "unresolved"})
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
_REVIEW_STAGING_PATH_KEY = "_review_staging_path"
_REVIEW_CLAIM_ID_KEY = "_review_claim_id"
_REVIEW_CLAIM_REVISION_KEY = "_review_claim_revision"
_PRIVATE_REVIEW_KEYS = {
    "evidence_manifest",
    _REVIEW_SESSION_AUDIT_KEY,
    _REVIEW_STAGING_PATH_KEY,
    _REVIEW_CLAIM_ID_KEY,
    _REVIEW_CLAIM_REVISION_KEY,
}
_ACTIVE_DIRECTION_EXPERIMENT_RUNTIME_ACTION = (
    "Keep the Direction active unless its objective, planned variants, and stop "
    "conditions are genuinely exhausted. Choose the next high-value candidate from its "
    "remaining branches; a supported Experiment requires neither a mechanism proof nor "
    "an immediate Direction revision."
)


_FANOUT_LOCK = threading.Lock()
_FANOUT_RUNNING: set[Path] = set()
_FANOUT_PENDING: set[Path] = set()


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
        if key not in _PRIVATE_REVIEW_KEYS
    }


def _private_claim_review(review: Mapping[str, object]) -> dict[str, object]:
    """Return the private receipt with a readable name for Session audit metadata."""
    result = deepcopy(dict(review))
    session_audit = result.pop(_REVIEW_SESSION_AUDIT_KEY, None)
    if session_audit is not None:
        result["session_audit"] = session_audit
    return result


def _durable_private_claim_review(
    review: Mapping[str, object],
) -> dict[str, object]:
    """Return private Review evidence without transient attachment metadata."""
    result = _private_claim_review(review)
    for key in (
        _REVIEW_STAGING_PATH_KEY,
        _REVIEW_CLAIM_ID_KEY,
        _REVIEW_CLAIM_REVISION_KEY,
    ):
        result.pop(key, None)
    return result


def _review_projection(review: Mapping[str, object] | None) -> dict[str, object]:
    """Keep only durable decision state from an independent claim review."""
    if not isinstance(review, Mapping):
        return {}
    quality = review.get("claim_review_quality")
    if quality is None:  # Read pre-v2 private receipts during migration.
        quality = review.get("verdict")
    if quality is None and review.get("status") == "unavailable":
        quality = "review_unavailable"
    quality = canonical_review_quality(quality)
    if quality is None:
        return {}
    projected: dict[str, object] = {
        "claim_review_quality": quality,
        "review_skipped": review.get("review_skipped") is True,
    }
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
    if value.get("review_skipped") is True:
        result["review_skipped"] = True
    quality = canonical_review_quality(
        value.get("claim_review_quality", value.get("review_quality"))
    )
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
    if not isinstance(value, Mapping) or value.get("schema") not in {
        "atrex.research-claim-review-session.v1",
        "atrex.research-claim-review-session.v2",
    }:
        raise ValueError("research review Session audit metadata is invalid")
    schema = value["schema"]
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
        token_usage_path = row.get("token_usage_path")
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
        if token_usage_path is not None:
            if (
                not isinstance(token_usage_path, str)
                or not token_usage_path
                or len(token_usage_path) > 2048
                or token_usage_path.startswith("/")
                or ".." in token_usage_path.split("/")
                or not token_usage_path.endswith("/token-usage.json")
            ):
                raise ValueError(
                    f"research review Session attempt {index} token usage path is invalid"
                )
        if (
            schema == "atrex.research-claim-review-session.v2"
            and (conversation_path is None) != (token_usage_path is None)
        ):
            raise ValueError(
                f"research review Session attempt {index} archive is incomplete"
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
            "token_usage_path": token_usage_path,
            "resumed": bool(row.get("resumed", False)),
        }
        if error is not None:
            normalized_row["error"] = {
                "type": error["type"],
                "message": error["message"],
            }
        normalized.append(normalized_row)
    return {
        "schema": schema,
        "attempts": normalized,
    }


def _module_enabled(journal: Mapping[str, object], module: str) -> bool:
    return module in _tool_modules(journal)






@dataclass(frozen=True)
class _VisibleHistory:
    current: dict[str, Any]
    directions: dict[str, dict[str, Any]]
    statuses: dict[str, str]
    experiments: list[dict[str, Any]]
    experiments_by_id: dict[str, dict[str, Any]]
    shared_direction_ids: frozenset[str]
    shared_experiment_ids: frozenset[str]
    shared_queue: tuple[str, ...]



def _now() -> str:
    return datetime.now(UTC).isoformat()




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
            "research_catalog_through_sequence": 0,
            "directions": {},
            "statuses": {},
            "experiments": [],
            "experiments_by_id": {},
            "gateway_record_ids": [],
            "research_queue": [],
            "accepted_submission_kernel_ids": {},
            "lineage_winners": {},
            "global_best": None,
        }
    except (OSError, ValueError) as error:
        raise RuntimeStateError("Shared Epoch research evidence is unreadable.") from error
    if (
        value.get("schema_version") != 1
        or not isinstance(value.get("revision"), str)
        or type(value.get("through_epoch")) is not int
        or value["through_epoch"] < 0
        or type(value.get("research_catalog_through_sequence", 0)) is not int
        or value.get("research_catalog_through_sequence", 0) < 0
        or not isinstance(value.get("directions"), dict)
        or not isinstance(value.get("experiments"), list)
        or not isinstance(value.get("gateway_record_ids", []), list)
        or not isinstance(value.get("research_queue", []), list)
    ):
        raise RuntimeStateError("Shared Epoch research evidence has an unsupported schema.")
    gateway_record_ids = value.get("gateway_record_ids", [])
    if (
        any(
            not isinstance(record_id, str)
            or GATEWAY_RECORD_ID_RE.fullmatch(record_id) is None
            for record_id in gateway_record_ids
        )
        or len(set(gateway_record_ids)) != len(gateway_record_ids)
    ):
        raise RuntimeStateError("Shared Gateway Record identities are invalid.")
    gateway_record_reasons = value.get("gateway_record_reasons", {})
    if (
        not isinstance(gateway_record_reasons, dict)
        or set(gateway_record_reasons) - set(gateway_record_ids)
        or any(
            reason not in _SHARED_GATEWAY_RECORD_REASONS
            for reason in gateway_record_reasons.values()
        )
    ):
        raise RuntimeStateError("Shared Gateway Record publication reasons are invalid.")
    lineage_winners = value.get("lineage_winners", {})
    global_best = value.get("global_best")
    if not isinstance(lineage_winners, dict) or (
        global_best is not None and not isinstance(global_best, dict)
    ):
        raise RuntimeStateError("Shared submitted-winner evidence is invalid.")

    def validate_winner(raw: object, lineage: str | None = None) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise RuntimeStateError("Shared submitted-winner evidence is invalid.")
        commit_id = raw.get("kernel_commit_id")
        record_id = raw.get("gateway_record_id")
        latency = raw.get("latency_us_geomean")
        raw_lineage = raw.get("lineage")
        if (
            not isinstance(commit_id, str)
            or KERNEL_RECORD_ID_RE.fullmatch(commit_id) is None
            or not isinstance(record_id, str)
            or GATEWAY_RECORD_ID_RE.fullmatch(record_id) is None
            or type(latency) not in {int, float}
            or not math.isfinite(float(latency))
            or float(latency) <= 0
            or not isinstance(raw_lineage, str)
            or re.fullmatch(r"T[1-9][0-9]*", raw_lineage) is None
            or (lineage is not None and raw_lineage != lineage)
        ):
            raise RuntimeStateError("Shared submitted-winner evidence is invalid.")
        return deepcopy(raw)

    validated_winners = {
        lineage: validate_winner(raw, lineage)
        for lineage, raw in lineage_winners.items()
        if isinstance(lineage, str) and re.fullmatch(r"T[1-9][0-9]*", lineage)
    }
    if len(validated_winners) != len(lineage_winners):
        raise RuntimeStateError("Shared submitted-winner evidence is invalid.")
    validated_global = validate_winner(global_best) if global_best is not None else None
    if validated_global is not None and not any(
        item == validated_global for item in validated_winners.values()
    ):
        raise RuntimeStateError("Shared global best is not a lineage winner.")
    accepted_by_lineage = value.get("accepted_submission_kernel_ids")
    if accepted_by_lineage is None:
        # A v1 publication from before accepted-submission Pareto tracking can
        # prove only its published winner. Preserve that safe subset.
        accepted_by_lineage = {
            lineage: [winner["kernel_commit_id"]]
            for lineage, winner in validated_winners.items()
        }
    if not isinstance(accepted_by_lineage, dict):
        raise RuntimeStateError("Shared accepted-submission evidence is invalid.")
    validated_accepted: dict[str, list[str]] = {}
    for lineage, commit_ids in accepted_by_lineage.items():
        if (
            not isinstance(lineage, str)
            or re.fullmatch(r"T[1-9][0-9]*", lineage) is None
            or not isinstance(commit_ids, list)
            or len(set(commit_ids)) != len(commit_ids)
            or any(
                not isinstance(commit_id, str)
                or KERNEL_RECORD_ID_RE.fullmatch(commit_id) is None
                for commit_id in commit_ids
            )
        ):
            raise RuntimeStateError("Shared accepted-submission evidence is invalid.")
        validated_accepted[lineage] = sorted(commit_ids)
    for lineage, winner in validated_winners.items():
        if winner["kernel_commit_id"] not in validated_accepted.get(lineage, []):
            raise RuntimeStateError(
                "Shared lineage winner is not an accepted submission."
            )
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
        "gateway_record_ids": sorted(gateway_record_ids),
        "accepted_submission_kernel_ids": validated_accepted,
        "lineage_winners": validated_winners,
        "global_best": validated_global,
    }


def shared_research_queue(campaign_root: Path) -> list[str]:
    return _shared_queue(_shared_research(campaign_root))


def _shared_queue(value: Mapping[str, object]) -> list[str]:
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


def _visible_history(
    current: dict[str, Any], campaign_root: Path
) -> _VisibleHistory:
    """Build the visible research state for one validated mutation.

    Mutations use an in-memory snapshot read directly from the central Research
    Store. Episode files are not research authority.
    """
    current_value = current
    episode = current_value.get("episode")
    if not isinstance(episode, int) or isinstance(episode, bool) or episode < 0:
        raise RuntimeStateError("Private Runtime state has an invalid Episode identity")
    central = current_value.get("central_research")
    if isinstance(central, Mapping):
        # Current Light keeps research authority in the central append-only
        # Research Store. The caller-owned mapping is one bounded validation
        # snapshot and is never persisted as a second authority.
        raw_directions = central.get("directions", {})
        raw_directions = (
            raw_directions if isinstance(raw_directions, Mapping) else {}
        )
        raw_queue = central.get("queue", [])
        queue = raw_queue if isinstance(raw_queue, list) else []
        ordered_direction_ids: list[str] = []
        seen_direction_ids: set[str] = set()
        for direction_id in queue:
            if (
                isinstance(direction_id, str)
                and isinstance(raw_directions.get(direction_id), Mapping)
                and direction_id not in seen_direction_ids
            ):
                ordered_direction_ids.append(direction_id)
                seen_direction_ids.add(direction_id)
        for direction_id, direction in raw_directions.items():
            if (
                isinstance(direction_id, str)
                and isinstance(direction, Mapping)
                and direction_id not in seen_direction_ids
            ):
                ordered_direction_ids.append(direction_id)
                seen_direction_ids.add(direction_id)
        local_directions = {
            direction_id: deepcopy(raw_directions[direction_id])
            for direction_id in ordered_direction_ids
        }
        local_experiments = {
            key: deepcopy(value)
            for key, value in central.get("experiments", {}).items()
            if isinstance(key, str) and isinstance(value, Mapping)
        }
        shared = _shared_research(campaign_root)
        directions = {
            key: deepcopy(value) for key, value in shared["directions"].items()
        }
        directions.update(local_directions)
        experiments_by_id = {
            key: deepcopy(value)
            for key, value in shared["experiments_by_id"].items()
        }
        experiments_by_id.update(local_experiments)
        shared_direction_ids = {
            key for key in shared["directions"] if key not in local_directions
        } | {
            key
            for key, value in local_directions.items()
            if value.get("access") == "peer_read_only"
        }
        shared_experiment_ids = {
            key for key in shared["experiments_by_id"] if key not in local_experiments
        } | {
            key
            for key, value in local_experiments.items()
            if value.get("access") == "peer_read_only"
        }
        return _VisibleHistory(
            current=current_value,
            directions=directions,
            statuses={
                key: str(value.get("status"))
                for key, value in directions.items()
            },
            experiments=list(experiments_by_id.values()),
            experiments_by_id=experiments_by_id,
            shared_direction_ids=frozenset(shared_direction_ids),
            shared_experiment_ids=frozenset(shared_experiment_ids),
            shared_queue=tuple(_shared_queue(shared)),
        )
    raise RuntimeStateError("Research mutation requires a central Research Store snapshot")


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
                if "analysis" in event:
                    _text(event["analysis"], "Direction analysis")
                revision = event.get("revision", 0)
                if type(revision) is not int or revision < 0:
                    raise ValueError("Direction revision must be a non-negative integer")
            _text_list(event.get("supporting_experiment_ids", []), "supporting_experiment_ids")
            _text_list(event.get("gateway_record_ids", []), "gateway_record_ids")
        except ValueError as error:
            raise RuntimeStateError(
                f"Invalid Direction history in Episode {journal['episode']}: {error}"
            ) from error
        if action == "propose":
            direction = {
                "direction_id": direction_id,
                "revision": event.get("revision", 0),
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
            if "analysis" in event:
                direction["analysis"] = event["analysis"]
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
            previous_revision = direction.get("revision", 0)
            revision = event.get(
                "revision",
                previous_revision + 1
                if type(previous_revision) is int and previous_revision >= 0
                else None,
            )
            if type(revision) is not int or revision < 1:
                raise RuntimeStateError(
                    "Direction state has an invalid revision"
                )
            direction["revision"] = revision
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
    current: dict[str, Any],
    campaign_root: Path,
    *,
    history: _VisibleHistory | None = None,
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
    current: dict[str, Any],
    campaign_root: Path,
    *,
    history: _VisibleHistory | None = None,
) -> Mapping[str, dict[str, Any]]:
    return _direction_state(current, campaign_root, history=history)[0]


def _visible_experiments(
    current: dict[str, Any],
    campaign_root: Path,
    *,
    history: _VisibleHistory | None = None,
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
    if "owner_lineage" not in visible and isinstance(visible.get("proposed_by"), str):
        visible["owner_lineage"] = visible["proposed_by"]
    visible.pop("proposed_by", None)
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
    if "owner_lineage" not in visible and isinstance(visible.get("recorded_by"), str):
        visible["owner_lineage"] = visible["recorded_by"]
    visible.pop("recorded_by", None)
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


def _catalog_claim_event(event: Mapping[str, object]) -> tuple[str, dict[str, Any]] | None:
    kind = event.get("type")
    payload = event.get("payload")
    payload = payload if isinstance(payload, Mapping) else {}
    if kind == "direction.observed" and isinstance(payload.get("direction"), Mapping):
        return "directions", deepcopy(dict(payload["direction"]))
    if kind == "experiment.observed" and isinstance(payload.get("experiment"), Mapping):
        return "experiments", deepcopy(dict(payload["experiment"]))
    return None


def _catalog_visible_events(
    catalog: ResearchStore,
    *,
    trajectory: int,
    synchronization: str,
    peer_through_sequence: int | None,
) -> tuple[list[dict[str, Any]], int]:
    events = catalog.events()
    latest_sequence = events[-1]["sequence"] if events else 0
    visible = []
    for event in events:
        claim = _catalog_claim_event(event)
        if claim is None:
            continue
        origin = event.get("origin")
        owner = origin.get("trajectory") if isinstance(origin, Mapping) else None
        if owner == trajectory:
            visible.append(event)
            continue
        if synchronization == "asynchronous":
            visible.append(event)
            continue
        if (
            synchronization == "synchronous"
            and type(peer_through_sequence) is int
            and event["sequence"] <= peer_through_sequence
        ):
            visible.append(event)
    return visible, latest_sequence


def _catalog_claim_version(
    events: list[dict[str, Any]],
    *,
    claim_type: str,
    claim_id: str,
    through_sequence: int | None = None,
) -> int:
    """Return the stable Agent-visible version number at a catalog position.

    Directions and Experiments are reviewed on first submission and start at
    immutable ``v1``. Lifecycle-only Direction updates retain their prior review
    and create the next version. Historical unreviewed drafts remain ``v0``.
    """
    identity_field = "direction_id" if claim_type == "directions" else "experiment_id"
    version = 0
    for event in events:
        sequence = event.get("sequence")
        if through_sequence is not None and type(sequence) is int and sequence > through_sequence:
            break
        claim = _catalog_claim_event(event)
        if claim is None or claim[0] != claim_type:
            continue
        if claim[1].get(identity_field) != claim_id:
            continue
        if claim_type == "directions" and canonical_review_quality(
            claim[1].get("claim_review_quality", claim[1].get("review_quality"))
        ) == "review_not_requested" and claim[1].get("review_skipped") is not True:
            continue
        version += 1
    return version


def _catalog_versions_by_sequence(
    events: list[dict[str, Any]],
) -> dict[int, int]:
    """Index per-identity visible versions in one linear catalog pass."""
    reviewed_counts: dict[tuple[str, str], int] = {}
    result: dict[int, int] = {}
    for event in events:
        claim = _catalog_claim_event(event)
        sequence = event.get("sequence")
        if claim is None or type(sequence) is not int:
            continue
        claim_type, value = claim
        identity_field = (
            "direction_id" if claim_type == "directions" else "experiment_id"
        )
        claim_id = value.get(identity_field)
        if not isinstance(claim_id, str):
            continue
        key = (claim_type, claim_id)
        if claim_type == "directions" and canonical_review_quality(
            value.get("claim_review_quality", value.get("review_quality"))
        ) == "review_not_requested" and value.get("review_skipped") is not True:
            result[sequence] = 0
            continue
        reviewed_counts[key] = reviewed_counts.get(key, 0) + 1
        result[sequence] = reviewed_counts[key]
    return result


def _kernel_submission_result(
    record: Mapping[str, object],
    *,
    idempotent: bool = False,
) -> dict[str, object]:
    """Project one immutable Submission revision into the Agent receipt."""
    reviewed = dict(record)
    evaluation_evidence = deepcopy(
        reviewed.get("evaluation_evidence", {"status": "missing"})
    )
    review_status = reviewed.get("review", {})
    source_contract = reviewed.get("source_contract")
    if (
        source_contract not in {"pass", "fail"}
        and isinstance(review_status, Mapping)
    ):
        source_contract = review_status.get("source_contract")
    review_verdict = (
        review_status.get("verdict")
        if isinstance(review_status, Mapping)
        else None
    )
    review_accepted = (
        review_verdict in {"allow", "not_applicable"}
        if review_verdict is not None
        else isinstance(review_status, Mapping)
        and "fail" not in review_status.values()
    )
    source_accepted = (
        source_contract == "pass"
        and not bool(reviewed.get("diagnostics"))
        and review_accepted
    )
    accepted = bool(reviewed.get("accepted")) and (
        isinstance(evaluation_evidence, Mapping)
        and evaluation_evidence.get("status") == "passed"
    )
    evidence_status = (
        evaluation_evidence.get("status")
        if isinstance(evaluation_evidence, Mapping)
        else "missing"
    )
    if accepted:
        status = "reviewed"
        runtime_action = (
            "The exact commit has a passing full Evaluate and passed the applicable "
            "final-source checks. Continue development or end the Session."
        )
    elif not source_accepted:
        status = "rejected"
        runtime_action = (
            "Repair the cited final-source policy violations, commit the repair, run a "
            "full Evaluate for that exact commit, then revise this Submission by supplying "
            "its submission_id with the repaired kernel_commit_id."
        )
    elif evidence_status == "failed":
        status = "rejected"
        if (
            isinstance(evaluation_evidence, Mapping)
            and evaluation_evidence.get("reason_code")
            == "performance_result_incomplete"
        ):
            runtime_action = (
                "The saved full Evaluate passed correctness but its performance result is "
                "incomplete or unparseable. Rerun a standard full Evaluate for this same "
                "commit, then retry this exact Submission revision."
            )
        else:
            runtime_action = (
                "The saved full Evaluate for this exact commit did not pass correctness or "
                "could not be parsed. Repair and commit the candidate when needed, run a "
                "complete passing full Evaluate, then revise this Submission."
            )
    else:
        status = "needs_evaluation"
        runtime_action = (
            "Run a full Evaluate for this exact commit, then retry this exact Submission "
            "revision."
        )
    result = {
        "status": status,
        "submission_id": reviewed.get("submission_id"),
        "revision": int(reviewed.get("revision", 1)),
        "kernel_commit_id": reviewed.get("kernel_commit_id"),
        "accepted": accepted,
        "source_contract": source_contract,
        "review": deepcopy(reviewed.get("review", {})),
        "diagnostics": deepcopy(reviewed.get("diagnostics", [])),
        "evaluation_evidence": evaluation_evidence,
        "runtime_action": runtime_action,
    }
    if idempotent:
        result["idempotent"] = True
    return result


def _prune_managed_projection(
    root: Path,
    expected_files: AbstractSet[str],
    expected_directories: AbstractSet[str] = frozenset(),
    *,
    preserve_private_reviews: bool = False,
) -> None:
    """Remove stale files from one Runtime-owned Agent projection tree."""
    if not root.exists():
        return
    if root.is_symlink() or not root.is_dir():
        raise RuntimeStateError(f"Runtime projection root is not a directory: {root}")
    for current, directories, files in os.walk(root, topdown=False, followlinks=False):
        current_path = Path(current)
        for name in files:
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode):
                raise RuntimeStateError(
                    f"Runtime projection contains a non-regular file: {path}"
                )
            parts = Path(relative).parts
            if preserve_private_reviews and (
                "review" in parts or "attempts" in parts
            ):
                continue
            if relative not in expected_files:
                path.unlink()
        for name in directories:
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                raise RuntimeStateError(
                    f"Runtime projection contains a symbolic link: {path}"
                )
            parts = Path(relative).parts
            if preserve_private_reviews and (
                "review" in parts or "attempts" in parts
            ):
                continue
            if relative in expected_directories:
                continue
            try:
                path.rmdir()
            except OSError as error:
                if error.errno not in {errno.ENOTEMPTY, errno.EEXIST}:
                    raise


def _retire_legacy_research_projection(workspace: Path) -> None:
    """Remove superseded top-level Direction/Experiment convenience views."""
    for name in ("directions", "experiments"):
        root = workspace / name
        _prune_managed_projection(root, frozenset())
        try:
            root.rmdir()
        except FileNotFoundError:
            pass
        except OSError as error:
            if error.errno not in {errno.ENOTEMPTY, errno.EEXIST}:
                raise


def _catalog_version_entries(
    *,
    catalog_root: Path,
    catalog_events: list[dict[str, Any]],
    events: list[dict[str, Any]],
    trajectory: int | None,
) -> dict[
    tuple[str, str, int],
    tuple[str, dict[str, Any], dict[str, bytes], set[str]],
]:
    """Build stable version entries from selected Catalog events.

    Version numbers are always computed against the complete Campaign Catalog;
    filtering changes visibility, never identity.  ``trajectory=None`` builds
    the canonical Campaign Library.  A concrete trajectory adds only the
    Agent-facing ownership/access projection.
    """
    versions_by_sequence = _catalog_versions_by_sequence(catalog_events)
    versions: dict[
        tuple[str, str, int],
        tuple[str, dict[str, Any], dict[str, bytes], set[str]],
    ] = {}
    for event in events:
        claim = _catalog_claim_event(event)
        if claim is None:
            continue
        claim_type, value = claim
        identity_field = (
            "direction_id" if claim_type == "directions" else "experiment_id"
        )
        claim_id = value.get(identity_field)
        if not isinstance(claim_id, str):
            raise RuntimeStateError("Central Research Catalog contains an invalid claim ID")
        origin = event.get("origin")
        owner = origin.get("trajectory") if isinstance(origin, Mapping) else None
        if trajectory is not None:
            value["access"] = (
                "local_writable" if owner == trajectory else "peer_read_only"
            )
        else:
            # Access is a lineage projection property, not canonical claim
            # content.  The Campaign Library stores ownership once and each
            # workspace copy derives local_writable/peer_read_only.
            value.pop("access", None)
        value.setdefault(
            "owner_lineage",
            f"T{owner}" if type(owner) is int and owner > 0 else "unknown",
        )
        public = (
            _public_direction(value)
            if claim_type == "directions"
            else _public_experiment(value)
        )
        sequence = int(event["sequence"])
        version = versions_by_sequence.get(sequence, -1)
        if version < 0 or (claim_type == "experiments" and version == 0):
            raise RuntimeStateError(
                "Central Research Catalog contains an invalid claim version"
            )
        version_root = (
            Path("research-library")
            / claim_type
            / claim_id
            / f"v{version}"
        )
        reference_payloads: dict[str, bytes] = {}
        reference_directories: set[str] = set()
        if claim_type == "experiments" and value.get("references"):
            reference_revision = value.get("reference_revision", 1)
            manifest, payloads = _read_experiment_reference_archive(
                catalog_root.parent / "research-library",
                claim_id,
                reference_revision,
            )
            reference_root = version_root / "references"
            public["references"] = [
                (reference_root / relative).as_posix()
                for relative in manifest["source_references"]
            ]
            for relative, data in payloads.items():
                relative_path = (reference_root / relative).as_posix()
                reference_payloads[relative_path] = data
            for relative in manifest["directories"]:
                reference_directories.add((reference_root / relative).as_posix())
        claim_filename = "direction.json" if claim_type == "directions" else "experiment.json"
        claim_path = (version_root / claim_filename).as_posix()
        versions[(claim_type, claim_id, version)] = (
            claim_path,
            public,
            reference_payloads,
            reference_directories,
        )
    return versions



def _direction_attempt_views(versions) -> dict[str, bytes]:
    """Derive histories exclusively from the supplied visibility snapshot.

    One row per expectation, one latest visible version per Experiment. Review
    eligibility and Direction supporting IDs do not filter investigative history.
    """
    latest = {}
    for (kind, identity, version), entry in sorted(versions.items()):
        latest[(kind, identity)] = (version, entry[1])

    def cell(value):
        if value is None or value == "":
            return "Not recorded"
        value = str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return " ".join(value.split()).replace("|", "&#124;").replace("`", "&#96;")

    views = {}
    for (kind, identity), (_, direction) in sorted(latest.items()):
        if kind != "directions":
            continue
        lines = [f"# Attempts: {cell(direction.get('name', identity))}", "",
                 "Latest visible version of every linked Experiment, including failed and unresolved attempts. "
                 "Historical versions remain in the Experiment directories. Predictions may be retrospective; "
                 "this view does not establish when they were recorded or infer that an idea is ruled out.", "",
                 "| Attempt | Method / difference | Scope and prediction | Observation | Assessment | Prediction origin | Explicitly unresolved | Interpretation | Next action | Reason |",
                 "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        count = 0
        for (experiment_kind, experiment_id), (version, experiment) in sorted(latest.items()):
            if experiment_kind != "experiments" or experiment.get("direction_id") != identity:
                continue
            count += 1
            link = f"../../experiments/{experiment_id}/v{version}/experiment.json"
            attempt = f"[{cell(experiment_id)} v{version}]({link}): {cell(experiment.get('name'))}"
            expectations = experiment.get("expectations") or [{}]
            for expectation in expectations:
                assessment = expectation.get("assessment")
                prediction = expectation.get("prediction")
                scope = expectation.get("scope")
                unresolved = prediction if assessment == "unresolved" else (
                    "Not recorded" if not expectation else "Not marked unresolved")
                row = [attempt, cell(experiment.get("implementation")),
                       cell(f"{scope}: {prediction}" if scope and prediction else prediction),
                       cell(expectation.get("observation")), cell(assessment),
                       cell(expectation.get("prediction_origin", "retrospective")), cell(unresolved),
                       cell(experiment.get("interpretation")),
                       cell((experiment.get("next_step") or {}).get("action")),
                       cell((experiment.get("next_step") or {}).get("reason"))]
                lines.append("| " + " | ".join(row) + " |")
        if not count:
            lines.extend(["", "No linked Experiments are visible yet."])
        views[f"research-library/directions/{identity}/attempts.md"] = ("\n".join(lines) + "\n").encode()
    return views


def _publish_research_library_tree(
    root: Path,
    versions: Mapping[
        tuple[str, str, int],
        tuple[str, dict[str, Any], dict[str, bytes], set[str]],
    ],
    *,
    canonical: bool,
) -> tuple[set[str], set[str]]:
    """Atomically publish one canonical Library or disposable Agent copy."""
    expected_files: set[str] = set()
    expected_directories: set[str] = set()

    def target(relative: str | Path) -> Path:
        path = Path(relative)
        return path.relative_to("research-library") if canonical else path

    for (claim_type, _claim_id, version), (
        claim_path,
        public,
        reference_payloads,
        reference_directories,
    ) in sorted(versions.items()):
        for relative_path in reference_directories:
            with open_private_directory(root / target(relative_path)):
                pass
            expected_directories.add(
                Path(relative_path).relative_to("research-library").as_posix()
            )
        for relative_path, data in reference_payloads.items():
            publish_if_changed(root, target(relative_path).as_posix(), data)
            expected_files.add(
                Path(relative_path).relative_to("research-library").as_posix()
            )
        claim_data = _projection_json(public)
        target_claim_path = target(claim_path)
        absolute_claim_path = root / target_claim_path
        if canonical and absolute_claim_path.exists() and not (
            claim_type == "directions" and version == 0
        ):
            existing = read_regular_bytes(absolute_claim_path, limit=MAX_FILE_BYTES + 1)
            if existing != claim_data:
                raise RuntimeStateError(
                    f"Immutable Research Library version changed: {claim_path}"
                )
        publish_if_changed(root, target_claim_path.as_posix(), claim_data)
        expected_files.add(Path(claim_path).relative_to("research-library").as_posix())
        if canonical and claim_type == "experiments" and public.get("references"):
            reference_manifest = Path(claim_path).parent / ".references.json"
            if (root / target(reference_manifest)).is_file():
                expected_files.add(
                    reference_manifest.relative_to("research-library").as_posix()
                )
    if not canonical:
        for relative_path, data in _direction_attempt_views(versions).items():
            publish_if_changed(root, relative_path, data)
            expected_files.add(Path(relative_path).relative_to("research-library").as_posix())
        _prune_managed_projection(
            root / "research-library",
            expected_files,
            expected_directories,
        )
    # The canonical Campaign Library is an append-only authority, not a
    # disposable projection.  In particular, record-experiment archives its
    # reference files before appending the corresponding Catalog event.  An
    # asynchronous peer may refresh the Catalog in that small interval.  If a
    # canonical refresh prunes paths absent from its snapshot, it can delete
    # the new Experiment's .references.json and payloads before publication.
    # Claim versions never disappear from the append-only Catalog, so there is
    # nothing legitimate to prune here; only Agent workspace projections are
    # regenerated destructively.
    return expected_files, expected_directories


def project_campaign_research_library(
    *,
    catalog_root: Path,
    workspace: Path,
    trajectory: int,
    synchronization: str,
    peer_through_sequence: int | None = None,
    library_root: Path | None = None,
) -> dict[str, object]:
    """Publish the canonical Library and one visibility-filtered Agent copy.

    The Campaign Library owns the complete versioned content.  Agent workspaces
    contain ordinary, disposable files so native and Bubblewrap profiles share
    one projection contract.  Local revisions are always current; synchronous
    peer revisions stop at the sealed Epoch watermark and asynchronous peers
    follow the Catalog head.
    """
    if synchronization not in {"independent", "synchronous", "asynchronous"}:
        raise ValueError("Invalid research synchronization mode")
    if type(trajectory) is not int or trajectory <= 0:
        raise ValueError("Research projection requires a positive trajectory")
    catalog = ResearchStore(catalog_root / "events")
    events, latest_sequence = _catalog_visible_events(
        catalog,
        trajectory=trajectory,
        synchronization=synchronization,
        peer_through_sequence=peer_through_sequence,
    )
    catalog_events = catalog.events()
    library_root = library_root or catalog_root.parent / "research-library"
    canonical_versions = _catalog_version_entries(
        catalog_root=catalog_root,
        catalog_events=catalog_events,
        events=catalog_events,
        trajectory=None,
    )
    _publish_research_library_tree(
        library_root,
        canonical_versions,
        canonical=True,
    )
    visible_versions = _catalog_version_entries(
        catalog_root=catalog_root,
        catalog_events=catalog_events,
        events=events,
        trajectory=trajectory,
    )
    # Immutable reviewed versions are copied from the canonical Campaign
    # Library, not reconstructed independently in each Agent workspace.  Only
    # ``access`` is lineage-specific.  Mutable v0 drafts retain the selected
    # Catalog snapshot so synchronous peers do not observe a post-watermark
    # rewrite before the next Epoch boundary.
    for key, entry in list(visible_versions.items()):
        claim_type, _claim_id, version = key
        if claim_type == "directions" and version == 0:
            continue
        claim_path, projected, references, directories = entry
        canonical_path = library_root / Path(claim_path).relative_to(
            "research-library"
        )
        try:
            canonical_public = read_json(canonical_path, limit=MAX_FILE_BYTES)
        except (FileNotFoundError, OSError, ValueError) as error:
            raise RuntimeStateError(
                f"Canonical Research Library version is unavailable: {claim_path}"
            ) from error
        if not isinstance(canonical_public, dict):
            raise RuntimeStateError(
                f"Canonical Research Library version is invalid: {claim_path}"
            )
        access = projected.get("access")
        if isinstance(access, str):
            canonical_public["access"] = access
        visible_versions[key] = (
            claim_path,
            canonical_public,
            references,
            directories,
        )
    _publish_research_library_tree(
        workspace,
        visible_versions,
        canonical=False,
    )
    counts = {
        claim_type: sum(1 for key in visible_versions if key[0] == claim_type)
        for claim_type in ("directions", "experiments")
    }
    _retire_legacy_research_projection(workspace)
    manifest = {
        "schema_version": 1,
        "trajectory": trajectory,
        "synchronization": synchronization,
        "peer_through_sequence": (
            peer_through_sequence if synchronization == "synchronous" else None
        ),
        "catalog_head_sequence": latest_sequence,
        "visible_version_counts": counts,
        "ownership": (
            "Only the original owner may revise an identity. Peer claims are read-only; "
            "derive a new local identity to continue them."
        ),
    }
    return manifest


def publish_catalog_shared_research(
    *,
    catalog_root: Path,
    research_root: Path,
    trajectory: int,
    synchronization: str,
    peer_through_sequence: int | None = None,
) -> None:
    """Refresh the Runtime-readable peer view from the central catalog."""
    catalog = ResearchStore(catalog_root / "events")
    events, _latest_sequence = _catalog_visible_events(
        catalog,
        trajectory=trajectory,
        synchronization=synchronization,
        peer_through_sequence=peer_through_sequence,
    )
    directions: dict[str, dict[str, Any]] = {}
    experiments: dict[str, dict[str, Any]] = {}
    for event in events:
        origin = event.get("origin")
        owner = origin.get("trajectory") if isinstance(origin, Mapping) else None
        if owner == trajectory:
            continue
        claim = _catalog_claim_event(event)
        if claim is None:
            continue
        claim_type, value = claim
        if claim_type == "directions":
            direction_id = value.get("direction_id")
            if not isinstance(direction_id, str):
                raise RuntimeStateError("Central Direction has an invalid identity")
            value["owner_lineage"] = f"T{owner}"
            if value.get("status") == "in_progress":
                value["owner"] = f"T{owner}"
            directions[direction_id] = value
        else:
            experiment_id = value.get("experiment_id")
            if not isinstance(experiment_id, str):
                raise RuntimeStateError("Central Experiment has an invalid identity")
            value["owner_lineage"] = f"T{owner}"
            experiments[experiment_id] = value
    # An unreadable peer view is not an empty view. Leave the published winners
    # untouched and let the next publication retry after the read is repaired.
    previous = _shared_research(research_root)
    queue = [
        direction_id
        for direction_id in previous.get("research_queue", [])
        if direction_id in directions
    ]
    queue.extend(
        direction_id
        for direction_id, direction in directions.items()
        if direction.get("status") in {"proposed", "in_progress", "deferred", "blocked"}
        and direction_id not in queue
    )
    previous_reasons = previous.get("gateway_record_reasons", {})
    persistent_record_reasons = {
        "passing_full_evaluate",
        "accepted_submission_full_evaluate",
        "lineage_winner_full_evaluate",
        "global_winner_full_evaluate",
    }
    record_reasons = {
        record_id: reason
        for record_id, reason in previous_reasons.items()
        if isinstance(record_id, str)
        and isinstance(reason, str)
        and reason in persistent_record_reasons
    } if isinstance(previous_reasons, Mapping) else {}

    def share_claim_records(value: Mapping[str, object], reason: str) -> None:
        record_ids = value.get("gateway_record_ids", [])
        if not isinstance(record_ids, list):
            return
        for record_id in record_ids:
            if isinstance(record_id, str):
                record_reasons.setdefault(record_id, reason)

    for direction in directions.values():
        share_claim_records(direction, "visible_direction_reference")
    for experiment in experiments.values():
        share_claim_records(experiment, "visible_experiment_reference")

    # Passing full Evaluates also cross trajectory boundaries independently of
    # claim citations. An accepted submission necessarily points at one of
    # these Records.
    from aka.experiment.frontier import (
        kernel_full_evaluation_evidence,
        standard_full_evaluation,
    )

    measurement_store = MeasurementStore(catalog_root.parent / "measurements")
    for registration_path in sorted((catalog_root / "lineages").glob("T*.json")):
        try:
            registration = read_json(registration_path, limit=MAX_FILE_BYTES)
        except (FileNotFoundError, OSError, ValueError):
            continue
        owner = registration.get("trajectory")
        owner_research_value = registration.get("research_root")
        legacy_journals_value = registration.get("journals_root")
        if (
            type(owner) is not int
            or owner <= 0
            or owner == trajectory
            or not isinstance(
                owner_research_value or legacy_journals_value, str
            )
        ):
            continue
        try:
            owner_research_root = (
                Path(owner_research_value).resolve(strict=True)
                if isinstance(owner_research_value, str)
                else Path(legacy_journals_value).resolve(strict=True).parent
                / "research"
            )
        except (FileNotFoundError, OSError):
            continue
        owner_research = ResearchStore(owner_research_root).read()
        lineage = owner_research.get("lineages", {}).get(f"T{owner}", {})
        local_record_ids = set(
            lineage.get("gateway_record_ids", [])
            if isinstance(lineage, Mapping)
            and isinstance(lineage.get("gateway_record_ids", []), list)
            else []
        )
        for record_id in local_record_ids:
            try:
                parsed = standard_full_evaluation(
                    measurement_store.read(record_id)
                )
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if isinstance(parsed, Mapping) and parsed.get("passed") is True:
                record_reasons.setdefault(record_id, "passing_full_evaluate")
        from aka.workflow.light.long_horizon.submissions import (
            accepted_submission_kernel_ids as central_accepted_submission_kernel_ids,
        )

        for commit_id in central_accepted_submission_kernel_ids(
            catalog_root.parent / "submissions", lineage=f"T{owner}"
        ):
            evidence = kernel_full_evaluation_evidence(measurement_store, commit_id)
            if evidence.get("status") == "passed":
                record_id = evidence.get("gateway_record_id")
                if isinstance(record_id, str):
                    record_reasons[record_id] = (
                        "accepted_submission_full_evaluate"
                    )

    # Epoch publications already selected exact winner evidence. Retain those
    # immutable IDs when refreshing the asynchronous claim projection.
    for winner in previous.get("lineage_winners", {}).values():
        if isinstance(winner, Mapping) and isinstance(
            winner.get("gateway_record_id"), str
        ):
            record_reasons.setdefault(
                winner["gateway_record_id"], "lineage_winner_full_evaluate"
            )
    global_best = previous.get("global_best")
    if isinstance(global_best, Mapping) and isinstance(
        global_best.get("gateway_record_id"), str
    ):
        record_reasons.setdefault(
            global_best["gateway_record_id"], "global_winner_full_evaluate"
        )

    for record_id in sorted(record_reasons):
        try:
            measurement_store.read(record_id)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise RuntimeStateError(
                f"Shared Gateway Record is unavailable: {record_id}"
            ) from error
    value = {
        "schema_version": 1,
        "revision": evidence_digest({
            "directions": directions,
            "experiments": experiments,
            "gateway_record_reasons": record_reasons,
            "synchronization": synchronization,
            "peer_through_sequence": peer_through_sequence,
        }),
        "through_epoch": previous.get("through_epoch", 0),
        "research_catalog_through_sequence": (
            peer_through_sequence
            if synchronization == "synchronous" and type(peer_through_sequence) is int
            else _latest_sequence
        ),
        "directions": directions,
        "experiments": [experiments[key] for key in sorted(experiments)],
        "gateway_record_ids": sorted(record_reasons),
        "gateway_record_reasons": record_reasons,
        "research_queue": queue,
        "accepted_submission_kernel_ids": previous.get(
            "accepted_submission_kernel_ids", {}
        ),
        "lineage_winners": previous.get("lineage_winners", {}),
        "global_best": previous.get("global_best"),
    }
    private_write(shared_research_path(research_root), value)


def _public_gateway_record(store: MeasurementStore, record_id: str) -> dict[str, Any]:
    """Return one standardized Agent-visible measurement projection."""
    from aka.evaluation.tool_results import seal_response
    from aka.workflow.light.host.projection import credential_values, redact_diagnostics

    record = store.read(record_id)
    public = seal_response(
        record["response"],
        record["request"],
        record_id=record_id,
        kernels=record["kernels"],
    )["agent_result"]
    return redact_diagnostics(public, private_paths=(str(store.root),),
                              private_values=credential_values(os.environ))


def _git_visible_record_ids(
    store: MeasurementStore,
    workspace: Path,
    *,
    commit_reachable: Callable[[str], bool] | None = None,
    record_ids: Iterable[str] | None = None,
) -> set[str]:
    """Expose records only when every referenced Kernel is exact, reachable Git."""
    from aka.task.candidate_workspace.agent_git import AgentGit
    from aka.task.source.formats import from_git

    git = AgentGit(workspace, workspace / ".git" / "unused-runtime-bundle")
    commit_reachable = commit_reachable or git.commit_reachable
    visible: set[str] = set()
    checked: dict[str, bool] = {}
    for record_id in (
        sorted(set(record_ids)) if record_ids is not None else store.record_ids()
    ):
        try:
            # Listing is header-only; consumers validate the immutable body
            # before using or publishing evidence. Do not parse every prior
            # stdout/correctness payload on each new measurement.
            from aka.experiment.measurement_store import read_record_header
            record = read_record_header(store.root / "records" / record_id / "record.json")
            kernels = record.get("kernels")
            if not isinstance(kernels, Mapping):
                continue
            # Dev/Command/Env probes have no committed Kernel dependency.
            # Their visibility is controlled by the lineage Record index.
            if not kernels:
                visible.add(record_id)
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
    library_root: Path,
    experiment_id: str,
    reference_revision: int,
    raw: object,
) -> list[str]:
    references, directories, files = _capture_experiment_references(workspace, raw)
    archive_root = _experiment_reference_archive_root(
        library_root, experiment_id, reference_revision
    )
    archived_references = [
        (
            Path("research-library")
            / "experiments"
            / experiment_id
            / f"v{reference_revision}"
            / "references"
            / path
        ).as_posix()
        for path in references
    ]
    for directory in sorted(directories):
        with open_private_directory(archive_root / "references" / directory):
            pass
    file_rows = []
    for path, data in sorted(files.items()):
        private_write_bytes(archive_root / "references" / path, data)
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
    manifest_path = archive_root / ".references.json"
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
    library_root: Path, experiment_id: str, reference_revision: int
) -> Path:
    if type(reference_revision) is not int or reference_revision < 1:
        raise RuntimeStateError(
            f"Experiment {experiment_id} has an invalid reference revision."
        )
    return (
        library_root
        / "experiments"
        / experiment_id
        / f"v{reference_revision}"
    )


def _read_experiment_reference_archive(
    library_root: Path, experiment_id: str, reference_revision: int = 1
) -> tuple[dict[str, Any], dict[str, bytes]]:
    archive_root = _experiment_reference_archive_root(
        library_root, experiment_id, reference_revision
    )
    try:
        manifest = read_json(archive_root / ".references.json", limit=MAX_FILE_BYTES)
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
        (
            Path("research-library")
            / "experiments"
            / experiment_id
            / f"v{reference_revision}"
            / "references"
            / path
        ).as_posix()
        for path in source_references
    ]
    legacy_archived = [
        (Path("experiments") / experiment_id / "references" / path).as_posix()
        for path in source_references
    ]
    if (
        manifest["archived_references"] != expected_archived
        and manifest["archived_references"] != legacy_archived
    ):
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
                archive_root / "references" / relative,
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

    latest = board.get("latest_result")
    latest = latest if isinstance(latest, Mapping) else {}
    current_lineage = board.get("current_lineage")
    current_lineage = current_lineage if isinstance(current_lineage, str) else "T1"
    lineages = board.get("lineages")
    lineages = lineages if isinstance(lineages, Mapping) else {}
    current = lineages.get(current_lineage)
    current = current if isinstance(current, Mapping) else {}
    submitted_winners = board.get("submitted_winners")
    submitted_winners = (
        submitted_winners if isinstance(submitted_winners, Mapping) else {}
    )
    lineage_winners = submitted_winners.get("lineages")
    lineage_winners = lineage_winners if isinstance(lineage_winners, Mapping) else {}
    global_best = submitted_winners.get("global_best")
    global_best = global_best if isinstance(global_best, Mapping) else {}
    lineage_winner = lineage_winners.get(current_lineage)
    lineage_winner = (
        lineage_winner if isinstance(lineage_winner, Mapping) else {}
    )
    visible_lineage_ids = set(lineages) | set(lineage_winners)
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
    def finite(value: object) -> float | None:
        return (
            float(value)
            if type(value) in {int, float} and math.isfinite(float(value))
            else None
        )

    submitted_best_line = (
        (
            f"- **Global best accepted submission commit ID:** "
            f"{commit(global_best.get('kernel_commit_id'))}; GeoMean: "
            f"**{finite(global_best.get('latency_us_geomean')):g} us**; lineage: "
            f"`{text(global_best.get('lineage'))}`; evidence: "
            f"{commit(global_best.get('gateway_record_id'))}. This is an observed winner, "
            "not an automatic checkout or promotion."
            if finite(global_best.get("latency_us_geomean")) is not None
            else "- **Global best accepted submission:** none observed."
        )
        if len(visible_lineage_ids) > 1
        else None
    )
    lineage_winner_line = (
        f"- **Best accepted submission for `{current_lineage}`:** "
        f"{commit(lineage_winner.get('kernel_commit_id'))}; GeoMean: "
        f"**{finite(lineage_winner.get('latency_us_geomean')):g} us**; evidence: "
        f"{commit(lineage_winner.get('gateway_record_id'))}."
        if finite(lineage_winner.get("latency_us_geomean")) is not None
        else f"- **Best accepted submission for `{current_lineage}`:** none observed."
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
        research_locations.append(
            "`research-library/directions/<direction_id>/vN/direction.json`"
        )
    if "experiments" in modules:
        research_locations.append(
            "`research-library/experiments/<experiment_id>/vN/experiment.json`"
        )
    lines = [
        "# Current state",
        "",
        "> Runtime-managed, read-only projection. Do not edit this file; use the documented",
        "> Runtime mutation operations and treat Gateway Records as measurement truth.",
        "",
        *([submitted_best_line] if submitted_best_line is not None else []),
        lineage_winner_line,
        f"- **Current lineage:** `{current_lineage}`.",
        f"- **Last measured working Kernel commit ID:** {commit(current.get('working_kernel_commit_id'))}.",
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
            "Experiment files declare `owner_lineage` and `access`; revise only "
            "`local_writable` Experiments and extend `peer_read_only` evidence with a new local record."
        ] if "experiments" in modules else []),
        "",
        "## Lineages",
        "",
        (
            "| Lineage | Last measured Kernel commit ID | Best accepted submission | Focus Direction | Active Directions |"
            if "directions" in modules
            else "| Lineage | Last measured Kernel commit ID | Best accepted submission |"
        ),
        "| --- | --- | --- | --- | --- |" if "directions" in modules else "| --- | --- | --- |",
    ]
    for lineage, raw in sorted(lineages.items()):
        item = raw if isinstance(raw, Mapping) else {}
        row = (
            f"| `{text(lineage)}` | {commit(item.get('working_kernel_commit_id'))} | "
            f"{commit((lineage_winners.get(lineage) or {}).get('kernel_commit_id'))}"
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
                proposer = item.get("owner_lineage", item.get("proposed_by"))
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
                if item.get("direction_id"):
                    lines.append(
                        f"   - [Attempt history](research-library/directions/{item['direction_id']}/attempts.md)"
                    )
                for label, field in (
                    ("Objective", "objective"),
                    ("Applicability", "applicability"),
                    ("Expected benefit", "expected_benefit"),
                    ("Current assessment", "analysis"),
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
        "## Accepted-submission per-Shape Pareto frontier",
        "",
        (
            f"Eligible full evaluations for accepted local and imported peer submissions: "
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
                f"Showing {len(visible_rows)} of {len(rows)} Shapes; read `frontier.json` for the complete accepted-submission frontier.",
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
        lines.append("_No accepted submission has eligible full-Evaluate evidence yet._")
    lines.extend([
        "",
        "Each row is an observed per-Shape minimum, not a measured composite Kernel.",
        "",
    ])
    return "\n".join(lines).encode()


def _save_journal(
    target: dict[str, Any],
    campaign_root: Path,
    journal: dict[str, Any],
) -> None:
    if not isinstance(target, dict):
        raise TypeError("Research transaction target must be a dict")
    if target is not journal:
        target.clear()
        target.update(journal)


def _mutable_journal(
    source: dict[str, Any], module: str | None = None
) -> dict[str, Any]:
    journal = source
    if module is not None and not _module_enabled(journal, module):
        label = "Direction" if module == "directions" else "Experiment"
        raise AgentRequestError(
            f"{label} recording is disabled for this Episode",
            code=f"{module}_disabled",
            next_action=(
                f"Do not use {label} mutation tools in this Episode. Continue with the enabled "
                "modules, saved Gateway Records, and exact-commit submit-kernel when appropriate."
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


def _append_event(
    state: dict[str, Any], campaign_root: Path, event: dict[str, Any]
) -> None:
    journal = _mutable_journal(state, "directions")
    events = journal["direction_events"]
    events.append(event)
    _save_journal(state, campaign_root, journal)


def update_direction(
    path: dict[str, Any],
    campaign_root: Path,
    request: Mapping[str, object],
    *,
    measurement_root: Path | None = None,
    review_claim: Callable[[str, Mapping[str, object]], dict[str, object]] | None = None,
) -> dict[str, object]:
    current = _mutable_journal(path, "directions")
    experiments_enabled = _module_enabled(current, "experiments")
    value = dict(request)
    action = value.get("action")
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
            **ancestry,
        }
        if not experiments_enabled:
            event["gateway_record_ids"] = _validate_record_ids(
                campaign_root,
                value.get("gateway_record_ids", []),
                measurement_root=measurement_root,
                allow_empty=True,
                label="Direction",
            )
    elif action == _DIRECTION_REVISION_ACTION:
        if set(value) & RELATIONSHIP_FIELDS:
            raise ValueError(
                "Direction genealogy is immutable; propose a new derived Direction instead of "
                "changing ancestry in a revision"
            )
        revision_optional = set(DIRECTION_REVISION_OPTIONAL_FIELDS)
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
                    "Use a direction_id from the highest visible vN under "
                    "research-library/directions/<direction_id>/. "
                    "If this is a new optimization claim, submit action=propose without a "
                    "direction_id instead."
                ),
            )
        advance_direction(dict(statuses), direction_id, action)
        previous = views[direction_id]
        previous_revision = previous.get("revision", 0)
        if type(previous_revision) is not int or previous_revision < 0:
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
        if "analysis" in value:
            event["analysis"] = _text(value["analysis"], "Direction analysis")
        elif previous.get("analysis") is not None:
            event["analysis"] = previous["analysis"]
        if not experiments_enabled:
            event["gateway_record_ids"] = _validate_record_ids(
                campaign_root,
                (
                    value["gateway_record_ids"]
                    if "gateway_record_ids" in value
                    else previous.get("gateway_record_ids", [])
                ),
                measurement_root=measurement_root,
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
            optional=closure_optional if action in _CLOSURES else {"analysis"},
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
        supporting = (
            _closure_support(
                path,
                campaign_root,
                direction_id,
                value,
                history=history,
                experiments_enabled=experiments_enabled,
                measurement_root=measurement_root,
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
        previous_revision = views[direction_id].get("revision", 0)
        if type(previous_revision) is not int or previous_revision < 0:
            raise RuntimeStateError("Visible Direction has an invalid revision")
        event = {
            "direction_event_id": f"directionevent_{uuid.uuid4().hex}",
            "direction_id": direction_id,
            "action": action,
            "revision": previous_revision + 1,
            "recorded_at": _now(),
            "supporting_experiment_ids": supporting,
            "reopen_condition": (
                _text(reopen_condition, "Direction reopen_condition")
                if reopen_condition is not None else None
            ),
        }
        if "analysis" in value:
            event["analysis"] = _text(value["analysis"], "Direction analysis")
        elif views[direction_id].get("analysis") is not None:
            event["analysis"] = views[direction_id]["analysis"]
        if not experiments_enabled and action in _CLOSURES:
            records = _validate_record_ids(
                campaign_root,
                (
                    value["gateway_record_ids"]
                    if "gateway_record_ids" in value
                    else views[direction_id].get("gateway_record_ids", [])
                ),
                measurement_root=measurement_root,
                allow_empty=True,
                label="Direction",
            )
            event["gateway_record_ids"] = records
    # Admission is reviewed before archiving the proposal. Revisions must also
    # be reviewed while proposed, so editing a plan cannot bypass admission.
    # Starting/restarting work changes lifecycle only and retains that review.
    review_required = action in {"propose", _DIRECTION_REVISION_ACTION} or action in _CLOSURES
    review = (
        review_claim("direction", event)
        if review_claim is not None
        and review_required
        else None
    )
    if review_required and review_claim is None:
        event["claim_review_quality"] = "review_unavailable"
    if isinstance(review, Mapping) and review.get("claim_review_quality") == "unsupported":
        error = AgentRequestError(
            "Independent research review rejected this Direction; nothing was archived.",
            code="research_claim_unsupported",
            review=_agent_claim_review(review),
            next_action=(
                "Narrow the claim using review.scope_safe_interpretation before resubmitting. "
                "Run review.scientific_next_check only if its uncertainty affects the next "
                "implementation choice or the submitted lifecycle decision. Do not cite this "
                "rejected submission as a recorded Direction."
            ),
        )
        error.private_review = _private_claim_review(review)
        raise error
    review_projection = _review_projection(review)
    if (action in {"complete", "abandon"}
            and review_projection.get("claim_review_quality") != "supported"
            and review_projection.get("review_skipped") is not True):
        # Check this terminal decision's review, not an older supported revision.
        rejected_review = dict(review or {})
        quality = review_projection.get("claim_review_quality", "review_unavailable")
        recovery = (
            f"Direction {action} requires supported Review; the Direction remains in_progress. "
            "Address the decision-critical findings and resubmit when the terminal decision "
            "is justified. Continue work or use defer with a reopen_condition when appropriate. "
            "Reuse existing evidence unless a decision-critical gap needs a new measurement."
        )
        rejected_review.update(
            archive_status="not_archived",
            claim_review_quality=quality,
            evidence_eligible=False,
            runtime_action=recovery,
        )
        error = AgentRequestError(
            f"Direction {action} requires supported Review; nothing was archived.",
            code="direction_review_insufficient",
            direction_id=direction_id,
            action=action,
            direction_status="in_progress",
            required_claim_review_quality="supported",
            review=_agent_claim_review(rejected_review),
            next_action=recovery,
        )
        error.private_review = _private_claim_review(rejected_review)
        raise error
    event.update(review_projection)
    _append_event(path, campaign_root, event)
    result: dict[str, object] = {
        "status": "updated" if action == _DIRECTION_REVISION_ACTION else "recorded",
        "direction_id": direction_id,
    }
    result["revision"] = event["revision"]
    if review is not None:
        result["review"] = dict(review)
    return result


def _read_private_gateway_evidence(
    journals_root: Path,
    record_id: str,
    *,
    measurement_root: Path | None = None,
) -> dict[str, Any]:
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
    store = MeasurementStore(
        measurement_root or journals_root.parent / "measurements"
    )
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
        "frozen_predictions": deepcopy(response.get("agent_result", {}).get("frozen_predictions")),
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
    *,
    measurement_root: Path | None = None,
) -> bool:
    if not isinstance(record_ids, list):
        return False
    for record_id in record_ids:
        if not isinstance(record_id, str):
            continue
        record = _read_private_gateway_evidence(
            campaign_root, record_id, measurement_root=measurement_root
        )
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






def _validate_record_ids(
    campaign_root: Path,
    raw: object,
    *,
    measurement_root: Path | None = None,
    label: str = "Experiment",
    allow_empty: bool = False,
) -> list[str]:
    """Validate real Record citations; Kernel binding is not required for research."""
    records = _text_list(raw, f"{label} gateway_record_ids")
    if not records:
        if allow_empty:
            return []
        raise AgentRequestError(
            "This operation requires at least one real Gateway Record in gateway_record_ids.",
            code="experiment_evidence_required",
            next_action=(
                "Cite an existing records/<gateway_record_id>.json relevant to the claim. "
                "Records need not bind a Kernel; Review assesses what their contents support."
            ),
        )
    if len(set(records)) != len(records):
        raise field_error("gateway_record_ids", f"{label} gateway_record_ids must not contain duplicates")
    for index, record_id in enumerate(records):
        try:
            _read_private_gateway_evidence(
                campaign_root, record_id, measurement_root=measurement_root
            )
        except ValueError as error:
            raise field_error(f"gateway_record_ids.{index}", str(error),
                              next_action="Check records/<gateway_record_id>.json, correct the citation, then resubmit.") from error
    return records


def _completed_record(record: Mapping[str, Any]) -> bool:
    """Completion is an execution fact, independent of correctness or interpretation."""
    return record["execution_status"] == "completed"


def _closure_support(
    path: dict[str, Any], campaign_root: Path, direction_id: str, value: Mapping[str, object],
    *, history: _VisibleHistory | None = None, experiments_enabled: bool = True,
    measurement_root: Path | None = None,
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
                "Inspect the highest visible vN under research-library/experiments/*/ and retain "
                "only entries whose direction_id "
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
            measurement_root=measurement_root,
            allow_empty=True,
        )
    return selected


def _experiment_next_step(raw: object) -> dict[str, str] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict) or set(raw) != {"action", "reason"}:
        raise field_error("next_step", "Use {action, reason}, or null to clear the next step")
    return {key: _text(raw[key], f"Experiment next_step.{key}") for key in ("action", "reason")}


def _experiment_expectations(raw: object) -> list[dict[str, Any]]:
    """Validate Agent-authored comparisons; measured facts remain in cited Records."""
    if not isinstance(raw, list):
        raise field_error("expectations", "Experiment expectations must be an array")
    result = []
    for index, item in enumerate(raw):
        label = f"expectations[{index}]"
        if not isinstance(item, dict):
            raise field_error(label, "Each expectation must be an object")
        missing = EXPERIMENT_EXPECTATION_FIELDS - item.keys()
        extra = item.keys() - EXPERIMENT_EXPECTATION_FIELDS
        if missing or extra:
            raise field_error(label, f"Expected scope, prediction, observation, assessment; "
                              f"missing={sorted(missing)}, unexpected={sorted(extra)}")
        row = {key: _text(item[key], f"Experiment {label}.{key}")
               for key in sorted(EXPERIMENT_EXPECTATION_FIELDS)}
        if row["assessment"] not in EXPERIMENT_EXPECTATION_ASSESSMENTS:
            raise field_error(f"{label}.assessment", "Use met, missed, or unresolved")
        result.append(row)
    return result


def _bind_experiment_predictions(experiment, existing, campaign_root, measurement_root):
    """Import all cited forecasts; references are resolved by Runtime, not the Agent."""
    supplied = deepcopy(experiment.get("expectations", []))
    forecasts = {}
    # Preserve previously cited forecasts even if a revision changes its citations.
    for row in (existing or {}).get("expectations", []):
        if row.get("prediction_origin") == "before_execution":
            key = (row["gateway_record_id"], row["scope"], row["prediction"])
            forecasts[key] = deepcopy(row)
    for record_id in experiment["gateway_record_ids"]:
        evidence = _read_private_gateway_evidence(campaign_root, record_id, measurement_root=measurement_root)
        frozen = evidence.get("frozen_predictions") or {}
        for prediction in frozen.get("predictions", []):
            key = (record_id, prediction["scope"], prediction["prediction"])
            forecasts.setdefault(key, {**prediction, "gateway_record_id": record_id,
                "prediction_origin": "before_execution", "prediction_recorded_at": frozen["recorded_at"],
                "observation": "Not recorded", "assessment": "unresolved"})
    matched = set()
    for key, forecast in forecasts.items():
        candidates = [(i, row) for i, row in enumerate(supplied)
                      if row["scope"] == forecast["scope"] and row["prediction"] == forecast["prediction"]
                      and row.get("gateway_record_id", key[0]) == key[0]]
        # Never guess correspondence when the same wording belongs to multiple Records.
        same_wording = sum(k[1:] == key[1:] for k in forecasts)
        if len(candidates) == 1 and (same_wording == 1 or candidates[0][1].get("gateway_record_id") == key[0]):
            i, row = candidates[0]
            forecast.update(observation=row["observation"], assessment=row["assessment"])
            matched.add(i)
    retrospective = []
    for i, row in enumerate(supplied):
        if i not in matched and row.get("prediction_origin") != "before_execution":
            row["prediction_origin"] = "retrospective"
            retrospective.append(row)
    if supplied or forecasts or "expectations" in experiment:
        experiment["expectations"] = [*forecasts.values(), *retrospective]


def record_experiment(
    path: dict[str, Any],
    campaign_root: Path,
    request: Mapping[str, object],
    *,
    measurement_root: Path | None = None,
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
                    "Use an experiment_id from the highest visible vN under this trajectory's "
                    "research-library/experiments/ projection, or "
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
                if direction_id is not None and supplied_direction_id != direction_id:
                    raise field_error(
                        "direction_id",
                        "An Experiment revision cannot change its Direction",
                        code="experiment_direction_immutable",
                        existing_direction_id=direction_id,
                    )
                direction_id = supplied_direction_id
        elif value.get("direction_id") is None:
            active = [
                item["direction_id"] for item in views.values()
                if item.get("status") == "in_progress"
                and item.get("direction_id") not in history.shared_direction_ids
            ]
            direction_id = active[0] if len(active) == 1 else None
        else:
            direction_id = _text(value["direction_id"], "Experiment Direction ID")
        if direction_id is not None:
            if not isinstance(direction_id, str) or DIRECTION_ID_RE.fullmatch(direction_id) is None:
                raise ValueError("Experiment Direction ID has an invalid format")
            direction = views.get(direction_id)
            if direction is None:
                raise ValueError("Experiment Direction is outside visible history")
            direction_status = str(direction["status"])
            if direction_id in history.shared_direction_ids and (
                existing is None or existing.get("direction_id") != direction_id
            ):
                raise AgentRequestError(
                    "A shared Direction is read-only and cannot own a local Experiment.",
                    code="shared_direction_read_only",
                    direction_id=direction_id,
                    next_action=(
                        "Propose and start a new local Direction derived from the shared Direction, "
                        "then record this Experiment against the local Direction."
                    ),
                )
            if direction["status"] not in {
                "proposed",
                "in_progress",
                *_CLOSED_STATUSES,
            }:
                raise ValueError(
                    "Experiment Direction must be a local proposed, in-progress, or closed "
                    "Direction. Evidence may be recorded against a proposed Direction before "
                    "its admission Review; late evidence does not reopen exploration"
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
        measurement_root=measurement_root,
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
    for key in ("implementation", "expectations", "next_step"):
        if key in value:
            experiment[key] = (
                _text(value[key], "Experiment implementation") if key == "implementation"
                else _experiment_next_step(value[key]) if key == "next_step"
                else _experiment_expectations(value[key])
            )
        elif existing is not None and key in existing:
            experiment[key] = deepcopy(existing[key])
    _bind_experiment_predictions(experiment, existing, campaign_root, measurement_root)
    if direction_id is not None:
        experiment["direction_id"] = direction_id
    if existing is not None and not any(
        key in value for key in ("wiki_usage", "wiki_query_ids", "wiki_usage_status")
    ):
        for key in ("wiki_usage", "wiki_query_ids", "wiki_usage_status"):
            if key in existing:
                experiment[key] = deepcopy(existing[key])
    if any(key in value for key in ("wiki_usage", "wiki_query_ids", "wiki_usage_status")):
        from aka.evaluation.evidence_fields import normalize_wiki_attribution

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
                    "Narrow the claim using review.scope_safe_interpretation before resubmitting. "
                    "Run review.scientific_next_check only if its uncertainty affects the next "
                    "implementation choice. Reuse existing Gateway Record IDs; do not rerun "
                    "measurements solely because the interpretation was rejected."
                ),
            )
            error.private_review = _private_claim_review(review)
            raise error
        if (
            (review.get("claim_review_quality") == "supported" or review.get("review_skipped") is True)
            and direction_status == "in_progress"
        ):
            review = dict(review)
            review["direction_status"] = direction_status
            review["runtime_action"] = (
                (str(review["runtime_action"]) + " " if review.get("review_skipped") else "")
                + _ACTIVE_DIRECTION_EXPERIMENT_RUNTIME_ACTION
            )
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
    }
    if review is not None:
        result["review"] = dict(review)
    return result




def _serialize_research_publication(method):
    @wraps(method)
    def locked(self, *args, **kwargs):
        with self._projection_lock:
            return method(self, *args, **kwargs)
    return locked


class SupervisorJournalService:
    """Execute scoped research and submission operations for one audit segment.

    Direction/Experiment authority is the central Research Store and
    submit-kernel authority is the Campaign Submission Store. Validation uses
    an in-memory snapshot of those authorities; Light creates no Episode
    Journal or compatibility transaction document.
    """

    def __init__(
        self,
        *,
        workspace: Path,
        campaign_root: Path,
        evidence_root: Path,
        git_workspace: Path | None = None,
        research_root: Path | None = None,
        research_synchronization: str = "independent",
        measurement_root: Path | None = None,
        submissions_root: Path | None = None,
        episode: int = 0,
        base_commit: str = "",
        base_kernel_id: str | None = None,
        episode_branch: str = "work",
        tool_modules: tuple[str, ...] | list[str] = (),
        session_capture_root: Path | None = None,
        origin: Mapping[str, object] | None = None,
        claim_reviewer: Callable[..., Mapping[str, object]] | None = None,
        claim_review_enabled: bool = False,
    ) -> None:
        self.workspace = workspace
        self.git_workspace = git_workspace or workspace
        self.campaign_root = campaign_root
        self.evidence_root = evidence_root
        self.evidence_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.session_capture_root = (
            session_capture_root.resolve() if session_capture_root is not None else None
        )
        self.episode = episode
        self.base_commit = base_commit
        self.base_kernel_id = base_kernel_id
        self.episode_branch = episode_branch
        self.tool_modules = tuple(tool_modules)
        if len(set(self.tool_modules)) != len(self.tool_modules) or set(
            self.tool_modules
        ) - JOURNAL_MODULES:
            raise ValueError(
                "tool_modules may contain only unique directions and experiments"
            )
        self.origin = dict(origin or {"trajectory": 1})
        if type(claim_review_enabled) is not bool:
            raise ValueError("claim_review_enabled must be boolean")
        self.claim_review_enabled = claim_review_enabled
        self.claim_reviewer = claim_reviewer
        self.measurement_root = (
            measurement_root or campaign_root.parent / "measurements"
        )
        self.submissions_root = submissions_root or campaign_root.parent / "submissions"
        self.research = ResearchStore(
            research_root or campaign_root.parent / "research"
        )
        if research_synchronization not in {
            "independent", "synchronous", "asynchronous"
        }:
            raise ValueError("Invalid research synchronization mode")
        self.research_synchronization = research_synchronization
        from aka.workflow.light.support.paths import (
            supervisor_research_catalog_root,
            supervisor_research_library_root,
        )
        self.research_catalog_root = supervisor_research_catalog_root(self.git_workspace)
        self.research_library_root = supervisor_research_library_root(
            self.git_workspace
        )
        from aka.workflow.light.support.paths import supervisor_review_root

        self.review_staging_root = supervisor_review_root(self.git_workspace)
        self._claim_audits: dict[str, dict[str, object]] = {}
        # Reviews use request-local snapshots. Only publication and mutations
        # of the same identity serialize; unrelated tools/claims keep running.
        self._projection_lock = threading.RLock()
        self._claim_locks_guard = threading.Lock()
        self._claim_locks: dict[tuple[str, str], tuple[threading.Lock, int]] = {}
        self.research_catalog = ResearchStore(self.research_catalog_root / "events")
        trajectory = self.origin.get("trajectory", 1)
        if type(trajectory) is not int or trajectory <= 0:
            raise ValueError("Research catalog registration requires a positive trajectory")
        private_write(
            self.research_catalog_root / "lineages" / f"T{trajectory}.json",
            {
                "schema_version": 1,
                "trajectory": trajectory,
                "workspace": str(self.workspace.resolve()),
                "research_root": str(self.campaign_root.resolve()),
                "synchronization": self.research_synchronization,
            },
        )
        if self.research_synchronization == "asynchronous":
            publish_catalog_shared_research(
                catalog_root=self.research_catalog_root,
                research_root=self.campaign_root,
                trajectory=trajectory,
                synchronization="asynchronous",
            )
        self._refresh_research_state()

    def _refresh_research_state(self) -> None:
        """Read one bounded in-memory validation snapshot from central authority."""
        self._research_state: dict[str, Any] = {
            "schema_version": 2,
            "runtime_managed": True,
            "research_records": bool(self.tool_modules),
            "tool_modules": list(self.tool_modules),
            "episode": self.episode,
            "memory_version": None,
            "base_commit": self.base_commit,
            "base_kernel_id": self.base_kernel_id,
            "episode_branch": self.episode_branch,
            "state": "in_progress",
            "direction_events": [],
            "experiments": [],
            "kernel_submissions": [],
            "central_research": self.research.read(),
            "created_at": _now(),
        }

    def _catalog_peer_watermark(self) -> int | None:
        if self.research_synchronization != "synchronous":
            return None
        value = _shared_research(self.campaign_root).get(
            "research_catalog_through_sequence"
        )
        return value if type(value) is int and value >= 0 else 0

    def _project_research_catalog(self) -> None:
        trajectory = int(self.origin.get("trajectory", 1))
        project_campaign_research_library(
            catalog_root=self.research_catalog_root,
            library_root=self.research_library_root,
            workspace=self.workspace,
            trajectory=trajectory,
            synchronization=self.research_synchronization,
            peer_through_sequence=self._catalog_peer_watermark(),
        )

    def _fanout_asynchronous_catalog(self) -> None:
        if self.research_synchronization != "asynchronous":
            return
        key = self.research_catalog_root.resolve()
        # Fold overlapping publications into the active projector's next pass.
        # No timer or stale watermark: the owner drains updates before exiting.
        with _FANOUT_LOCK:
            _FANOUT_PENDING.add(key)
            if key in _FANOUT_RUNNING:
                return
            _FANOUT_RUNNING.add(key)
        try:
            while True:
                with _FANOUT_LOCK:
                    _FANOUT_PENDING.discard(key)
                self._project_asynchronous_catalog()
                with _FANOUT_LOCK:
                    if key not in _FANOUT_PENDING:
                        _FANOUT_RUNNING.remove(key)
                        return
        except BaseException:
            with _FANOUT_LOCK:
                _FANOUT_RUNNING.discard(key)
                # A peer may already have returned after coalescing its update.
                # Retain the dirty watermark for the next publication attempt.
                _FANOUT_PENDING.add(key)
            raise

    def _project_asynchronous_catalog(self) -> None:
        registrations = self.research_catalog_root / "lineages"
        failed = False
        for path in sorted(registrations.glob("T*.json")):
            try:
                registration = read_json(path, limit=MAX_FILE_BYTES)
                trajectory = registration.get("trajectory")
                workspace_value = registration.get("workspace")
                research_value = registration.get("research_root")
                synchronization = registration.get("synchronization")
                if (
                    type(trajectory) is not int
                    or trajectory <= 0
                    or not isinstance(workspace_value, str)
                    or not isinstance(research_value, str)
                    or synchronization != "asynchronous"
                ):
                    continue
                workspace = Path(workspace_value).resolve(strict=True)
                if not workspace.is_dir():
                    continue
                research_root = Path(research_value).resolve(strict=True)
                publish_catalog_shared_research(
                    catalog_root=self.research_catalog_root,
                    research_root=research_root,
                    trajectory=trajectory,
                    synchronization="asynchronous",
                )
                project_campaign_research_library(
                    catalog_root=self.research_catalog_root,
                    workspace=workspace,
                    trajectory=trajectory,
                    synchronization="asynchronous",
                )
            except (FileNotFoundError, OSError, ValueError, RuntimeStateError):
                failed = True
                logging.getLogger(__name__).exception(
                    "Could not refresh asynchronous central Research projection %s",
                    path,
                )
        if failed:
            raise RuntimeStateError("Asynchronous Research projection is incomplete; pending publication is retained")

    def record_token_checkpoint(
        self,
        *,
        event: str,
        checkpoint_id: str,
        occurred_at: str,
        identity: Mapping[str, object],
    ) -> dict | None:
        """Best-effort private milestone accounting; never changes Agent results."""
        if self.session_capture_root is None:
            return None
        try:
            from aka.workflow.light.long_horizon.token_checkpoints import capture_token_checkpoint

            reference, _stored = capture_token_checkpoint(
                audit_root=self.campaign_root.parent,
                session_capture_root=self.session_capture_root,
                event=event,
                checkpoint_id=checkpoint_id,
                occurred_at=occurred_at,
                identity=identity,
            )
            return reference
        except (OSError, TypeError, ValueError, RecursionError):
            logging.getLogger(__name__).exception(
                "Could not persist %s token checkpoint %s",
                event,
                checkpoint_id,
            )
            return None

    def record_passing_full_evaluate_checkpoint(
        self, record: Mapping[str, object]
    ) -> dict | None:
        """Checkpoint one newly completed, passing standard full Evaluate."""
        from aka.experiment.frontier import standard_full_evaluation

        parsed = standard_full_evaluation(dict(record))
        if parsed is None or not parsed["passed"]:
            return None
        record_id = record.get("gateway_record_id")
        created_at = record.get("created_at")
        kernels = record.get("kernels")
        candidate = (
            kernels.get("candidate", {}) if isinstance(kernels, Mapping) else {}
        )
        if not isinstance(record_id, str) or not isinstance(created_at, str):
            raise ValueError("Passing full Evaluate has no durable Record identity")
        return self.record_token_checkpoint(
            event="passing-full-evaluate",
            checkpoint_id=record_id,
            occurred_at=created_at,
            identity={
                "gateway_record_id": record_id,
                "kernel_commit_id": (
                    candidate.get("kernel_id")
                    if isinstance(candidate, Mapping)
                    else None
                ),
                "latency_us_geomean": parsed.get("latency_us_geomean"),
                "measured_shape_count": len(parsed.get("latencies", {})),
            },
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

    @staticmethod
    def _claim_collection(claim_type: str) -> str:
        if claim_type == "direction":
            return "directions"
        if claim_type == "experiment":
            return "experiments"
        raise ValueError(f"Unsupported reviewed claim type: {claim_type}")

    def _claim_root(self, claim_type: str, claim_id: str) -> Path:
        return self.research_library_root / self._claim_collection(claim_type) / claim_id

    @staticmethod
    def _accepted_review_session(value: Mapping[str, object]) -> str | None:
        review = value.get("review")
        if not isinstance(review, Mapping):
            review = value
        audit = review.get("session_audit")
        if not isinstance(audit, Mapping):
            return None
        attempts = audit.get("attempts")
        if not isinstance(attempts, list):
            return None
        for row in reversed(attempts):
            if (
                isinstance(row, Mapping)
                and row.get("status") == "accepted"
                and isinstance(row.get("session_id"), str)
                and row.get("session_id")
            ):
                return str(row["session_id"])
        return None

    def _prior_claim_review_session(
        self, claim_type: str, target: Mapping[str, object]
    ) -> str | None:
        identity = self._reviewed_claim_identity(claim_type, target)
        if identity is None:
            return None
        claim_id, _revision = identity
        from aka.workflow.light.long_horizon.review_archive import review_root
        root = review_root(self.measurement_root, claim_id)
        candidates = [
            *root.glob("v*/result.json"),
            *root.glob("attempts/*/result.json"),
        ]
        candidates.sort(
            key=lambda path: path.stat().st_mtime_ns if path.exists() else 0,
            reverse=True,
        )
        for path in candidates:
            try:
                value = read_json(path, limit=MAX_FILE_BYTES)
            except (FileNotFoundError, OSError, ValueError):
                continue
            session_id = self._accepted_review_session(value)
            if session_id is not None and len(session_id) <= 512:
                return session_id
        # Read-only migration input for Campaigns created before Reviews were
        # attached to immutable claim revisions. Never update this index.
        legacy_index = (
            self.campaign_root
            / "claim-review-index"
            / claim_type
            / f"{claim_id}.json"
        )
        try:
            legacy = read_json(legacy_index, limit=MAX_FILE_BYTES)
        except (FileNotFoundError, OSError, ValueError):
            return None
        session_id = legacy.get("session_id")
        if (
            legacy.get("schema") == "atrex.research-claim-review-session-index.v1"
            and legacy.get("claim_type") == claim_type
            and legacy.get("claim_id") == claim_id
            and isinstance(session_id, str)
            and 0 < len(session_id) <= 512
        ):
            return session_id
        return None

    def _begin_claim_audit(self, claim_type: str, submission: object) -> str:
        """Track one request until it is bound to its claim or Runtime audit."""
        audit_id = "claimaudit_" + uuid.uuid4().hex
        pending: dict[str, object] = {
            "claim_type": claim_type,
            "received_at": _now(),
            "origin": deepcopy(self.origin),
            "submission": deepcopy(submission),
        }
        if claim_type == "experiment" and isinstance(submission, Mapping):
            references = submission.get("references")
            if references not in (None, []):
                snapshot_root = (
                    self.review_staging_root
                    / "claim-requests"
                    / audit_id
                    / "references"
                )
                try:
                    source_paths, directories, files = _capture_experiment_references(
                        self.workspace, references
                    )
                except Exception as error:  # Validation returns the useful error separately.
                    pending["reference_capture_error"] = {
                        "type": type(error).__name__,
                        "message": str(error)[:2000],
                    }
                else:
                    for directory in sorted(directories):
                        with open_private_directory(snapshot_root / "files" / directory):
                            pass
                    rows = []
                    for relative, data in sorted(files.items()):
                        private_write_bytes(snapshot_root / "files" / relative, data)
                        rows.append({
                            "path": relative,
                            "size": len(data),
                            "sha256": hashlib.sha256(data).hexdigest(),
                        })
                    private_write(
                        snapshot_root / "manifest.json",
                        {
                            "schema": "atrex.bound-claim-reference-snapshot.v1",
                            "requested": True,
                            "capture_status": "captured",
                            "source_references": [path.as_posix() for path in source_paths],
                            "directories": sorted(directories),
                            "files": rows,
                        },
                    )
                    pending["reference_snapshot_root"] = str(snapshot_root)
        self._claim_audits[audit_id] = pending
        return audit_id

    def _finish_claim_audit(
        self,
        audit_id: str,
        *,
        status: str,
        result: Mapping[str, object] | None = None,
        error: BaseException | None = None,
    ) -> None:
        """Attach rejected reviewed attempts; Runtime audit owns other failures."""
        pending = self._claim_audits.pop(audit_id, {})
        reference_value = pending.get("reference_snapshot_root")
        reference_snapshot = (
            Path(reference_value)
            if isinstance(reference_value, str) and reference_value
            else None
        )
        outcome: dict[str, object] = {
            "schema": "atrex.bound-claim-review.v1",
            "audit_id": audit_id,
            "finished_at": _now(),
            "status": status,
            "claim_type": pending.get("claim_type"),
            "submission": deepcopy(pending.get("submission")),
            "origin": deepcopy(pending.get("origin")),
        }
        if result is not None:
            outcome["result"] = deepcopy(dict(result))
        if error is None:
            if reference_snapshot is not None and reference_snapshot.is_dir():
                shutil.rmtree(reference_snapshot.parent)
            return
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
                outcome["review"] = _durable_private_claim_review(private_review)
                claim_id = private_review.get(_REVIEW_CLAIM_ID_KEY)
                claim_revision = private_review.get(_REVIEW_CLAIM_REVISION_KEY)
                staging_value = private_review.get(_REVIEW_STAGING_PATH_KEY)
                claim_type = pending.get("claim_type")
                if (
                    claim_type in {"direction", "experiment"}
                    and isinstance(claim_id, str)
                    and type(claim_revision) is int
                    and claim_revision >= 1
                ):
                    attempt_root = (
                        self._claim_root(str(claim_type), claim_id)
                        / "attempts"
                        / audit_id
                    )
                    from aka.workflow.light.long_horizon.review_archive import review_root as stored_review_root
                    review_root = stored_review_root(self.measurement_root, claim_id) / "attempts" / audit_id
                    staging = (
                        Path(staging_value)
                        if isinstance(staging_value, str) and staging_value
                        else None
                    )
                    if staging is not None and staging.is_dir():
                        private_write(staging / "result.json", outcome)
                        from aka.workflow.light.long_horizon.review_archive import (
                            move_review,
                            remove_empty_parents,
                        )

                        move_review(staging, review_root)
                        remove_empty_parents(
                            staging.parent, stop=self.review_staging_root
                        )
                    else:
                        private_write(review_root / "result.json", outcome)
                    if reference_snapshot is not None and reference_snapshot.is_dir():
                        shutil.copytree(
                            reference_snapshot,
                            attempt_root / "references",
                            symlinks=False,
                        )
                        shutil.rmtree(reference_snapshot.parent)
                    return
        # Validation errors without a stable reviewed identity are already
        # preserved by the Runtime request audit. Experiment files need their
        # own immutable bytes even when validation failed before an identity
        # could be allocated, so retain those as an explicitly unbound attempt.
        if reference_snapshot is not None and reference_snapshot.is_dir():
            unbound = (
                self.research_library_root
                / "experiments"
                / "unbound-attempts"
                / audit_id
            )
            private_write(unbound / "result.json", outcome)
            shutil.copytree(
                reference_snapshot,
                unbound / "references",
                symlinks=False,
            )
            shutil.rmtree(reference_snapshot.parent)

    @_serialize_research_publication
    def _observe_direction(self, direction_id: str) -> int:
        direction = _direction_views(self._research_state, self.campaign_root).get(direction_id)
        if direction is None:
            raise RuntimeStateError("Recorded Direction is missing from the visible projection")
        visible = deepcopy(direction)
        visible.setdefault("owner_lineage", self.lineage)
        visible.pop("proposed_by", None)
        visible["access"] = "local_writable"
        if visible.get("status") == "in_progress" and not visible.get("owner"):
            visible["owner"] = self.lineage
        self.research.observe_direction(visible, origin=self.origin)
        catalog_state = self.research_catalog.observe_direction(
            visible, origin=self.origin
        )
        self._fanout_asynchronous_catalog()
        return _catalog_claim_version(
            self.research_catalog.events(),
            claim_type="directions",
            claim_id=direction_id,
            through_sequence=int(catalog_state["sequence"]),
        )

    @_serialize_research_publication
    def _observe_experiment(self, experiment_id: str) -> int:
        experiment = next(
            (
                item
                for item in _visible_experiments(self._research_state, self.campaign_root)
                if item.get("experiment_id") == experiment_id
            ),
            None,
        )
        if experiment is None:
            raise RuntimeStateError("Recorded Experiment is missing from the visible projection")
        visible = deepcopy(experiment)
        visible.setdefault("owner_lineage", self.lineage)
        visible.pop("recorded_by", None)
        visible["access"] = "local_writable"
        self.research.observe_experiment(visible, origin=self.origin)
        catalog_state = self.research_catalog.observe_experiment(
            visible, origin=self.origin
        )
        self._fanout_asynchronous_catalog()
        return _catalog_claim_version(
            self.research_catalog.events(),
            claim_type="experiments",
            claim_id=experiment_id,
            through_sequence=int(catalog_state["sequence"]),
        )

    def _sync_research(self) -> None:
        history = _visible_history(self._research_state, self.campaign_root)
        for direction_id, direction in _direction_views(
            self._research_state, self.campaign_root, history=history
        ).items():
            visible = deepcopy(direction)
            if direction_id in history.shared_direction_ids:
                # Current shared projections include owner identity. Keep legacy
                # shared data honest instead of attributing it to this reader.
                visible.setdefault(
                    "owner_lineage",
                    visible.get("proposed_by") or visible.get("owner") or "unknown",
                )
                visible.pop("proposed_by", None)
                visible["access"] = "peer_read_only"
            else:
                visible.setdefault("owner_lineage", self.lineage)
                visible.pop("proposed_by", None)
                visible["access"] = "local_writable"
            if visible.get("status") == "in_progress" and not visible.get("owner"):
                visible["owner"] = self.lineage
            self.research.observe_direction(visible, origin=self.origin)
        for experiment in _visible_experiments(self._research_state, self.campaign_root, history=history):
            visible = deepcopy(experiment)
            experiment_id = visible.get("experiment_id")
            if experiment_id in history.shared_experiment_ids:
                visible.setdefault(
                    "owner_lineage", visible.get("recorded_by") or "unknown"
                )
                visible.pop("recorded_by", None)
                visible["access"] = "peer_read_only"
            else:
                visible.setdefault("owner_lineage", self.lineage)
                visible.pop("recorded_by", None)
                visible["access"] = "local_writable"
            self.research.observe_experiment(visible, origin=self.origin)
        # A peer may publish between reads. Use the same frozen view as the
        # imported claims so its queue cannot reference a newer unseen ID.
        self.research.synchronize_shared_queue(list(history.shared_queue), origin=self.origin)

    @_serialize_research_publication
    def observe_measurement(self, agent_result: Mapping[str, object]) -> None:
        """Checkpoint a Kernel-bound operation in the current lineage.

        Gateway Records remain the measurement truth. This projection records
        only which committed source was most recently measured and which source
        most recently passed Evaluate. ABBA has two source identities and
        deliberately does not move the single working-HEAD pointer. Passing
        Evaluate evidence remains in MeasurementStore; it enters the Pareto
        frontier and winner set only after an accepted submit-kernel review.
        """
        kernel_commit_id = agent_result.get("kernel_commit_id")
        record_id = agent_result.get("gateway_record_id")
        trajectory = self.origin.get("trajectory", 1)
        lineage = f"T{trajectory if type(trajectory) is int and trajectory > 0 else 1}"
        state = self.research.read()
        current = state.get("lineages", {}).get(lineage, {})
        gateway_record_ids = list(
            current.get("gateway_record_ids", [])
            if isinstance(current, Mapping)
            and isinstance(current.get("gateway_record_ids", []), list)
            else []
        )
        if (
            isinstance(record_id, str)
            and GATEWAY_RECORD_ID_RE.fullmatch(record_id) is not None
            and record_id not in gateway_record_ids
        ):
            gateway_record_ids.append(record_id)
            self.research.checkpoint_lineage(
                {"gateway_record_ids": [record_id]}, origin=self.origin
            )
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
        self.research.checkpoint_lineage(payload, origin=self.origin)

    def experiment_recording_hint(
        self, agent_result: Mapping[str, object]
    ) -> dict[str, object] | None:
        """Return one compact post-measurement reminder for enabled Experiments.

        Measurements remain useful without an Experiment, so Runtime never
        auto-generates scientific interpretation. Remind the Agent to preserve
        observations at a decision checkpoint without requiring a reviewed
        Experiment between related candidate trials.
        """
        if "experiments" not in self.tool_modules:
            return None
        record_id = agent_result.get("gateway_record_id")
        if (
            not isinstance(record_id, str)
            or GATEWAY_RECORD_ID_RE.fullmatch(record_id) is None
        ):
            return None
        try:
            _validate_record_ids(
                self.campaign_root,
                [record_id],
                measurement_root=self.measurement_root,
            )
        except (
            AgentRequestError,
            RuntimeStateError,
            OSError,
            ValueError,
            KeyError,
            TypeError,
        ):
            # Do not recommend a citation whose Record is missing or invalid.
            return None
        hint: dict[str, object] = {
            "operation": "record-experiment",
            "gateway_record_ids": [record_id],
            "instruction": (
                "When these results determine whether to keep or discard a candidate, select a "
                "parameter or workload strategy, rule out a branch, or switch paths, record that "
                "decision as an Experiment with observations, measured scope, and the next action. "
                "Related candidate trials may continue and be recorded together. Cite the relevant "
                "Records; do not wait for a mechanism proof or Direction closure. Not every "
                "Evaluate or disposable probe needs an Experiment."
            ),
        }
        if "directions" in self.tool_modules:
            try:
                current = copy(self)
                current._refresh_research_state()
                history = _visible_history(current._research_state, self.campaign_root)
                local = [
                    direction
                    for direction in _direction_views(
                        current._research_state,
                        self.campaign_root,
                        history=history,
                    ).values()
                    if direction.get("direction_id")
                    not in history.shared_direction_ids
                    and direction.get("status") in {"proposed", "in_progress"}
                ]
            except (RuntimeStateError, OSError, ValueError, KeyError, TypeError):
                # The hint must remain best-effort: a damaged Research view
                # must not turn an otherwise successful GPU measurement into
                # a failed Runtime request.
                return hint
            if len(local) == 1 and isinstance(local[0].get("direction_id"), str):
                hint["direction_id"] = local[0]["direction_id"]
        return hint

    @_serialize_research_publication
    def publish_research_files(self, **kwargs) -> None:
        # Never replace a request's in-flight transaction when a measurement
        # completes or an early failure publishes its Record concurrently.
        copy(self)._publish_research_files(**kwargs)

    def _publish_research_files(
        self,
        *,
        synchronize: bool = True,
        direction_ids: AbstractSet[str] | None = None,
        experiment_ids: AbstractSet[str] | None = None,
        record_ids: AbstractSet[str] | None = None,
    ) -> None:
        """Materialize the Agent-visible research history in the workspace.

        The Research Store, Submission Store, Measurement Store, and Git remain
        authoritative. PROGRESS.md and these JSON files are normal-file projections for
        search, remote ``command`` probes, and long-running engineering work. Bubblewrap
        exposes PROGRESS.md and records/ read-only; native-mode edits are never read back
        as Runtime state and are overwritten on the next refresh.
        """
        self._project_research_catalog()
        self._refresh_research_state()
        modules = set(self.tool_modules)

        if synchronize:
            self._sync_research()
        research = self.research.read()
        from aka.experiment.frontier import kernel_pareto_frontier, latest_gateway_result
        from aka.task.candidate_workspace.agent_git import AgentGit
        from aka.experiment.measurement_store import MeasurementReadView
        store = MeasurementReadView(self.measurement_root)
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

        current_lineage = f"T{trajectory if type(trajectory) is int and trajectory > 0 else 1}"
        lineage_state = research.get("lineages", {}).get(current_lineage, {})
        local_record_ids = set(
            value
            for value in (
                lineage_state.get("gateway_record_ids", [])
                if isinstance(lineage_state, Mapping)
                else []
            )
            if isinstance(value, str)
        )
        shared = _shared_research(self.campaign_root)
        shared_record_ids = {
            value
            for value in shared.get("gateway_record_ids", [])
            if isinstance(value, str)
        }
        visible_record_ids = _git_visible_record_ids(
            store,
            self.workspace,
            commit_reachable=commit_reachable,
            record_ids=local_record_ids | shared_record_ids,
        )
        from aka.workflow.light.long_horizon.submissions import (
            accepted_submission_kernel_ids as central_accepted_submission_kernel_ids,
        )

        local_accepted_kernel_ids = central_accepted_submission_kernel_ids(
            self.submissions_root, lineage=self.lineage
        )
        peer_accepted_kernel_ids = {
            commit_id
            for commit_ids in shared.get(
                "accepted_submission_kernel_ids", {}
            ).values()
            if isinstance(commit_ids, list)
            for commit_id in commit_ids
            if isinstance(commit_id, str)
        }
        accepted_kernel_ids = local_accepted_kernel_ids | peer_accepted_kernel_ids
        board = self.research.board(
            trajectory=trajectory if type(trajectory) is int and trajectory > 0 else 1,
            pareto_frontier=kernel_pareto_frontier(
                store,
                record_ids=visible_record_ids,
                kernel_ids=accepted_kernel_ids,
            ),
        )
        board["latest_result"] = latest_gateway_result(
            store, record_ids=visible_record_ids
        )
        visible_lineage_winners = {
            lineage: deepcopy(winner)
            for lineage, winner in shared.get("lineage_winners", {}).items()
            if isinstance(winner, Mapping)
            and isinstance(winner.get("kernel_commit_id"), str)
            and commit_reachable(winner["kernel_commit_id"])
        }
        local_winner = kernel_pareto_frontier(
            store,
            record_ids=visible_record_ids,
            kernel_ids=local_accepted_kernel_ids,
        ).get("best_full_evaluation")
        if isinstance(local_winner, Mapping):
            visible_lineage_winners[f"T{trajectory}"] = dict(
                local_winner,
                trajectory=trajectory,
                lineage=f"T{trajectory}",
            )
        board["submitted_winners"] = {
            "lineages": visible_lineage_winners,
            "global_best": (
                min(
                    visible_lineage_winners.values(),
                    key=lambda item: float(item["latency_us_geomean"]),
                )
                if visible_lineage_winners
                else None
            ),
        }
        board = _git_reachable_projection(board, commit_reachable)
        publish_pinned_if_changed(
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

        if "experiments" in modules:
            # Claims themselves are projected only by the versioned central
            # Research Library. Keep evidence validation here so a readable
            # claim can never silently cite a missing private Gateway Record.
            for experiment in _visible_experiments(self._research_state, self.campaign_root):
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
            # The Research/Submission mutation is already durable. Returning an ordinary
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
            existing = _direction_views(self._research_state, self.campaign_root).get(direction_id)
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
                for item in _visible_experiments(self._research_state, self.campaign_root)
                if isinstance(item, dict)
            }
            # Interim analysis may refer to Experiments already bound to this
            # Direction. Include them for revisions without requiring any
            # Experiment for admission of a new proposal.
            if (
                isinstance(direction_id, str)
                and target.get("action") == _DIRECTION_REVISION_ACTION
            ):
                experiment_ids.extend(
                    str(experiment_id)
                    for experiment_id, experiment in experiments.items()
                    if isinstance(experiment_id, str)
                    and isinstance(experiment, Mapping)
                    and experiment.get("direction_id") == direction_id
                )
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
                direction = _direction_views(self._research_state, self.campaign_root).get(direction_id)
                if isinstance(direction, Mapping):
                    publish(
                        root / "submissions",
                        "direction-proposal.json",
                        _projection_json(_public_direction(direction)),
                    )
        else:
            raise RuntimeStateError("Research claim type is invalid")

        store = MeasurementStore(self.measurement_root)
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
        from aka.task.source.formats import unpack
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
        if not self.claim_review_enabled:
            return {
                "archive_status": "archived",
                "claim_review_quality": "review_not_requested",
                "evidence_eligible": False,
                "review_skipped": True,
                "runtime_action": "Recorded. Claim Review was skipped because workflow.claim_review=false; no independent review was performed.",
            }
        if self.claim_reviewer is None:
            return {}
        identity = self._reviewed_claim_identity(claim_type, target)
        if identity is None:
            raise RuntimeStateError(
                "A reviewed research claim has no stable identity and revision"
            )
        claim_id, claim_revision = identity
        staging_root = (
            self.review_staging_root
            / "claims"
            / self._claim_collection(claim_type)
            / claim_id
            / ("pending_" + uuid.uuid4().hex)
            / "review"
        )
        try:
            with tempfile.TemporaryDirectory(
                prefix="claim-review-", dir=self.evidence_root
            ) as directory:
                root = Path(directory)
                manifest = self._build_claim_review_bundle(
                    root, claim_type=claim_type, target=target
                )
                from aka.workflow.light.long_horizon.claim_review import validate_review
                resume_session_id = self._prior_claim_review_session(
                    claim_type, target
                )
                raw_review = self.claim_reviewer(
                    root, resume_session_id=resume_session_id, archive_root=staging_root
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
                "claim_review_quality": "review_unavailable",
                "evidence_eligible": False,
                "runtime_action": (
                    "The reviewer remained unavailable after one automatic retry. Runtime will "
                    "preserve this record as review_unavailable; do not treat it as established "
                    "research. Continue high-value trials using the saved measurements; do not "
                    "rerun measurements solely to obtain a review label."
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
                    "Use the archived observations within scope_safe_interpretation to choose "
                    "the next high-value implementation. Investigate mechanism details only "
                    "when they affect that choice."
                    if evidence_eligible else
                    (
                        "This submission is unsupported and will not be archived. Follow "
                        "scope_safe_interpretation to narrow the claim before resubmitting. Run "
                        "scientific_next_check only if the uncertainty affects the next implementation "
                        "choice or the submitted lifecycle decision; reuse existing measurements."
                        if quality == "unsupported" else
                        "This partially supported record will be archived. Use observations within "
                        "scope_safe_interpretation to continue high-value candidate trials; unresolved "
                        "claims are not established evidence. Run scientific_next_check or revise "
                        "only when the uncertainty affects the next implementation choice or the "
                        "submitted lifecycle decision. Do not repeat measurements or resubmit "
                        "solely to obtain supported."
                    )
                ),
            }
            review_receipt["evidence_manifest"] = manifest
            if session_audit is not None:
                review_receipt[_REVIEW_SESSION_AUDIT_KEY] = session_audit
        review_receipt[_REVIEW_STAGING_PATH_KEY] = str(staging_root)
        review_receipt[_REVIEW_CLAIM_ID_KEY] = claim_id
        review_receipt[_REVIEW_CLAIM_REVISION_KEY] = claim_revision
        return review_receipt

    def _persist_claim_review(
        self,
        result: dict[str, object],
        *,
        claim_type: str,
        audit_id: str,
    ) -> dict[str, object]:
        """Attach the private Review archive to the immutable claim revision."""
        review = result.get("review")
        if not isinstance(review, Mapping):
            return result
        review_receipt = deepcopy(dict(review))
        archive_path = str(result["archive_path"])
        private_review = _durable_private_claim_review(review_receipt)
        claim_revision = result.get("revision")
        revision_row = (
            {"claim_revision": claim_revision}
            if type(claim_revision) is int and claim_revision > 0 else {}
        )
        bound_result: dict[str, object] = {
            "schema": "atrex.bound-claim-review.v1",
            "claim_type": claim_type,
            "claim_archive_path": archive_path,
            **revision_row,
            "submission": deepcopy(
                self._claim_audits.get(audit_id, {}).get("submission")
            ),
            "origin": deepcopy(
                self._claim_audits.get(audit_id, {}).get("origin")
            ),
            "review": private_review,
            "archived_at": _now(),
        }
        claim_file = self.research_library_root.parent / archive_path
        if not claim_file.is_file():
            raise RuntimeStateError(
                "Canonical research claim is unavailable while attaching its Review"
            )
        from aka.workflow.light.long_horizon.review_archive import review_revision_root
        review_root = review_revision_root(self.measurement_root, claim_file.parent.parent.name, int(claim_revision))
        staging_value = review_receipt.get(_REVIEW_STAGING_PATH_KEY)
        staging = (
            Path(staging_value)
            if isinstance(staging_value, str) and staging_value
            else None
        )
        if staging is not None and staging.is_dir():
            private_write(staging / "result.json", bound_result)
            from aka.workflow.light.long_horizon.review_archive import (
                move_review,
                remove_empty_parents,
            )

            move_review(staging, review_root)
            remove_empty_parents(staging.parent, stop=self.review_staging_root)
        else:
            private_write(review_root / "result.json", bound_result)
        return {
            **result,
            "review": _agent_claim_review(review_receipt),
        }

    def execute(
        self, request: Mapping[str, object], *,
        review_kernel: Callable[[str, str, int, str | None], dict[str, Any]] | None = None,
        check_wait: Callable[[], None] | None = None,
    ) -> dict[str, object]:
        body = request.get("request")
        operation = request.get("operation")
        fields = {"direction_update": ("direction_id",),
                  "experiment_record": ("experiment_id",),
                  "kernel_submit": ("kernel_commit_id", "submission_id")}.get(
                      operation if isinstance(operation, str) else "", ())
        keys = sorted((field, body[field]) for field in fields
                      if isinstance(body, Mapping) and isinstance(body.get(field), str)
                      and body[field])
        locks = []
        # Count waiters as well as owners, so releasing an owner cannot replace
        # a lock while another caller is still waiting on that same identity.
        with self._claim_locks_guard:
            for key in keys:
                lock, users = self._claim_locks.get(key, (threading.Lock(), 0))
                self._claim_locks[key] = (lock, users + 1)
                locks.append(lock)
        if operation == "direction_prioritize":
            locks.append(self._projection_lock)
        acquired = []
        try:
            for lock in locks:
                while True:
                    if check_wait is not None:
                        check_wait()
                    if lock.acquire(timeout=0.1):
                        acquired.append(lock)
                        break
            if check_wait is not None:
                check_wait()
            worker = copy(self)
            worker._claim_audits = {}
            return worker._execute(request, review_kernel=review_kernel)
        finally:
            for lock in reversed(acquired):
                lock.release()
            with self._claim_locks_guard:
                for key in keys:
                    lock, users = self._claim_locks[key]
                    if users == 1:
                        del self._claim_locks[key]
                    else:
                        self._claim_locks[key] = (lock, users - 1)

    def _execute(
        self, request: Mapping[str, object], *,
        review_kernel: Callable[[str, str, int, str | None], dict[str, Any]] | None = None,
    ) -> dict[str, object]:
        operation = request.get("operation")
        if not isinstance(operation, str):
            raise ValueError("Runtime research/report operation must be text")
        fields = {
            "direction_update": "request",
            "experiment_record": "request",
            "kernel_submit": "request",
            "direction_prioritize": "request",
        }
        if operation not in fields:
            raise ValueError(f"unsupported Runtime research/report operation: {operation}")
        self._refresh_research_state()
        journal = self._research_state
        module = ("directions" if operation in {"direction_update",
                                                  "direction_prioritize"}
                  else "experiments" if operation == "experiment_record"
                  else None)
        if module is not None and not _module_enabled(journal, module):
            raise AgentRequestError(
                f"{module.capitalize()} tools are disabled for this Episode",
                code=f"{module}_disabled",
                next_action="Use only the enabled Runtime research and submit-kernel tools.",
            )
        required = {"operation"}
        required.add(fields[operation])
        require_fields(dict(request), required, label="Runtime research/report operation")
        if operation == "direction_update":
            body = request.get("request")
            audit_id = self._begin_claim_audit("direction", body)
            try:
                if not isinstance(body, Mapping):
                    raise ValueError("direction_update requires a request object")
                result = update_direction(
                    self._research_state,
                    self.campaign_root,
                    body,
                    measurement_root=self.measurement_root,
                    review_claim=(
                        self._review_claim_before_mutation
                        if self.claim_reviewer is not None or not self.claim_review_enabled else None
                    ),
                )
                direction_id = str(result["direction_id"])
                library_version = self._observe_direction(direction_id)
                result["archive_path"] = (
                    f"research-library/directions/{direction_id}/v{library_version}/direction.json"
                )
                if body.get("action") == "propose" and result.get("review", {}).get(
                    "claim_review_quality"
                ) in {"supported", "partially_supported"}:
                    result["runtime_action"] = (
                        "To begin this Direction, call update-direction with action=start, "
                        f"direction_id={direction_id}. analysis is optional; omit it to preserve "
                        "the current assessment. start requires no Experiment and does not run another Review."
                    )
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
                    self._research_state,
                    self.campaign_root,
                    body,
                    measurement_root=self.measurement_root,
                    archive_references=lambda experiment_id, revision, references: (
                        _archive_experiment_references(
                            self.workspace,
                            self.research_library_root,
                            experiment_id,
                            revision,
                            references,
                        )
                    ),
                    review_claim=(
                        self._review_claim_before_mutation
                        if self.claim_reviewer is not None or not self.claim_review_enabled else None
                    ),
                )
                experiment_id = str(result["experiment_id"])
                library_version = self._observe_experiment(experiment_id)
                result["archive_path"] = (
                    f"research-library/experiments/{experiment_id}/v{library_version}/experiment.json"
                )
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
        if operation == "kernel_submit":
            body = request.get("request")
            if not isinstance(body, Mapping):
                raise ValueError("kernel_submit requires a request object")
            require_fields(
                dict(body),
                {"kernel_commit_id"},
                optional={"submission_id"},
                label="Kernel submission",
            )
            commit_id = body.get("kernel_commit_id")
            if (
                not isinstance(commit_id, str)
                or KERNEL_RECORD_ID_RE.fullmatch(commit_id) is None
            ):
                raise field_error(
                    "kernel_commit_id",
                    "kernel_commit_id must be one full Agent Git commit ID",
                )
            if review_kernel is None:
                raise RuntimeStateError("Kernel submission review is unavailable")
            from aka.workflow.light.long_horizon.submissions import (
                SUBMISSION_ID_RE,
                append_submission,
                next_submission_revision,
                submission_for_commit,
                submission_revision_root,
                submission_versions,
            )

            requested_submission_id = body.get("submission_id")
            if (
                requested_submission_id is not None
                and (
                    not isinstance(requested_submission_id, str)
                    or SUBMISSION_ID_RE.fullmatch(requested_submission_id) is None
                )
            ):
                raise field_error(
                    "submission_id",
                    "submission_id must be one Runtime-issued Submission ID",
                )
            existing = submission_for_commit(
                self.submissions_root,
                lineage=self.lineage,
                kernel_commit_id=commit_id,
                submission_id=requested_submission_id,
            )
            if existing is not None and requested_submission_id is None:
                return _kernel_submission_result(existing, idempotent=True)
            if existing is not None:
                from aka.experiment.frontier import (
                    kernel_full_evaluation_evidence,
                )

                current_evidence = kernel_full_evaluation_evidence(
                    MeasurementStore(self.measurement_root), commit_id
                )
                if current_evidence == existing.get("evaluation_evidence"):
                    return _kernel_submission_result(existing, idempotent=True)

            if requested_submission_id is None:
                submission_id = "submission-" + uuid.uuid4().hex
                revision = 1
                previous = None
            else:
                submission_id = requested_submission_id
                versions = submission_versions(
                    self.submissions_root, submission_id
                )
                if not versions:
                    raise AgentRequestError(
                        "submission_id does not identify an existing Submission",
                        code="submission_not_found",
                        submission_id=submission_id,
                        next_action=(
                            "Omit submission_id to create a new Submission, or copy the exact "
                            "submission_id returned by an earlier submit-kernel call."
                        ),
                    )
                revision, previous = next_submission_revision(
                    self.submissions_root,
                    submission_id,
                    lineage=self.lineage,
                )
            resume_session_id = None
            if previous is not None:
                previous_revision = int(previous.get("revision", 1))
                from aka.workflow.light.long_horizon.review_archive import review_revision_root
                previous_review = review_revision_root(
                    self.measurement_root, submission_id, previous_revision,
                ) / "result.json"
                try:
                    previous_review_value = json.loads(
                        previous_review.read_text(encoding="utf-8")
                    )
                except (
                    FileNotFoundError,
                    OSError,
                    UnicodeError,
                    json.JSONDecodeError,
                ):
                    previous_review_value = None
                if isinstance(previous_review_value, Mapping):
                    candidate_session_id = previous_review_value.get("session_id")
                    if isinstance(candidate_session_id, str) and candidate_session_id:
                        resume_session_id = candidate_session_id
            reviewed = review_kernel(commit_id, submission_id, revision, resume_session_id)
            if not isinstance(reviewed, dict):
                raise RuntimeStateError(
                    "Kernel submission review returned invalid state"
                )
            evaluation_evidence = deepcopy(
                reviewed.get("evaluation_evidence", {"status": "missing"})
            )
            accepted = bool(reviewed.get("accepted")) and (
                evaluation_evidence.get("status") == "passed"
            )
            reviewed = {**reviewed, "accepted": accepted}
            submitted_at = _now()
            audit = {
                "submission_id": submission_id,
                "revision": revision,
                "lineage": self.lineage,
                "episode": self.episode,
                "kernel_commit_id": commit_id,
                "submitted_at": submitted_at,
                **deepcopy(reviewed),
            }
            if accepted:
                checkpoint = self.record_token_checkpoint(
                    event="accepted-submit-kernel",
                    checkpoint_id=f"{submission_id}-v{revision}",
                    occurred_at=submitted_at,
                    identity={
                        "kernel_commit_id": commit_id,
                        "gateway_record_id": evaluation_evidence.get(
                            "gateway_record_id"
                        ),
                    },
                )
                if checkpoint is not None:
                    audit["token_checkpoint"] = checkpoint
            result = _kernel_submission_result(audit)
            audit.update({
                "status": result["status"],
                "source_contract": result["source_contract"],
                "runtime_action": result["runtime_action"],
            })
            append_submission(self.submissions_root, audit)
            if accepted:
                self._fanout_asynchronous_catalog()
                return self._publish_after_mutation(
                    result,
                    direction_ids=frozenset(),
                    experiment_ids=frozenset(),
                    record_ids=frozenset(),
                )
            return result
        raise ValueError(f"unsupported Runtime research/report operation: {operation}")

__all__ = [
    "DIRECTION_CHANGE_SCOPES",
    "DIRECTION_CLOSURE_FIELDS",
    "DIRECTION_CLOSURE_OPTIONAL_FIELDS",
    "DIRECTION_PROPOSAL_FIELDS",
    "DIRECTION_PROPOSAL_OPTIONAL_FIELDS",
    "DIRECTION_REVISION_FIELDS",
    "DIRECTION_REVISION_OPTIONAL_FIELDS",
    "DIRECTION_UPDATE_FIELDS",
    "EXPERIMENT_FIELDS",
    "EXPERIMENT_OPTIONAL_FIELDS",
    "SupervisorJournalService",
    "shared_research_queue",
]
