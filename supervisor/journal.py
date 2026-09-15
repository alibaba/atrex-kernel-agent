"""Supervisor-owned Direction, Experiment, and terminal-report Journal.

The Agent submits interpretation, intent, and explicit Gateway Record references.
The private Gateway store owns their exact Kernel bindings; the Journal does not
duplicate source identity or require a before/after comparison structure.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from long_horizon.protocol import atomic_write_json

from supervisor.direction_genealogy import (
    RELATIONSHIP_FIELDS,
    relationship_fields,
    validate_relationship,
)
from supervisor.direction_lifecycle import (
    CLOSED_STATUSES as _CLOSED_STATUSES,
    CLOSURES as _CLOSURES,
    DIRECTION_STATUS as _DIRECTION_STATUS,
    advance_direction,
)
from supervisor.errors import AgentRequestError, RuntimeStateError, require_fields
from supervisor.identifiers import (
    DIRECTION_ID_RE,
    EXPERIMENT_ID_RE,
    GATEWAY_RECORD_ID_RE,
    KERNEL_RECORD_ID_RE,
)

_PROPOSAL_FIELDS = {
    "action",
    "name",
    "hypothesis",
    "rationale",
    "plan",
    "success_criteria",
    "stop_conditions",
}
_UPDATE_FIELDS = {"action", "direction_id", "analysis"}
_CLOSURE_FIELDS = _UPDATE_FIELDS | {"hypothesis_status", "supporting_experiment_ids"}
_HYPOTHESIS_STATUSES = {"unresolved", "supported", "refuted"}
_EVIDENCE_KINDS = {"run", "same_allocation_abba", "profile", "dev", "check", "disassemble"}
_EXPERIMENT_FIELDS = {
    "direction_id",
    "name",
    "hypothesis",
    "change",
    "gateway_record_ids",
    "evidence",
    "analysis",
    "action",
}
_EXPERIMENT_ACTIONS = {
    "baseline",
    "keep_after",
    "restore_before",
    "abandon_direction",
    "adopt",
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def _text_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{label} must be an array of non-empty text")
    return [item.strip() for item in value]


def initialize_journal(
    path: Path,
    *,
    episode: int,
    base_commit: str,
    branch: str,
    memory_version: int | None = None,
    live_path: Path | None = None,
) -> dict[str, Any]:
    """Create one private Runtime-managed Episode Journal."""
    value: dict[str, Any] = {
        "schema_version": 2,
        "runtime_managed": True,
        "episode": episode,
        "memory_version": memory_version,
        "base_commit": base_commit,
        "episode_branch": branch,
        "state": "in_progress",
        "direction_events": [],
        "experiments": [],
        "outcome": None,
        "candidate_commit": None,
        "created_at": _now(),
        "finalized_at": None,
    }
    atomic_write_json(path, value)
    if live_path is not None:
        from long_horizon.journal import sync_live_memory

        sync_live_memory(live_path, value)
    return value


def load_journal(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeStateError("The Supervisor-owned Journal is missing or unreadable.") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 2
        or value.get("runtime_managed") is not True
    ):
        raise RuntimeStateError("The Supervisor-owned Journal has an unsupported schema.")
    if not isinstance(value.get("direction_events"), list) or not isinstance(
        value.get("experiments"), list
    ):
        raise RuntimeStateError("The Supervisor-owned Journal collections are invalid.")
    return value


def _journal_files(current: Path, campaign_root: Path) -> list[Path]:
    archived = sorted(
        (campaign_root / ".atrex_long_horizon" / "episodes").glob(
            "e*/supervisor_runtime/journal.json"
        )
    )
    return [*archived, current]


def _visible_journals(current: Path, campaign_root: Path) -> list[dict[str, Any]]:
    current_value = load_journal(current)
    values: list[dict[str, Any]] = []
    for path in _journal_files(current, campaign_root):
        if path.resolve() == current.resolve():
            continue
        try:
            value = load_journal(path)
        except (ValueError, RuntimeStateError):
            continue
        if value["episode"] < current_value["episode"]:
            values.append(value)
    values.sort(key=lambda value: value["episode"])
    values.append(current_value)
    return values


def _direction_views(current: Path, campaign_root: Path) -> dict[str, dict[str, Any]]:
    views: dict[str, dict[str, Any]] = {}
    statuses: dict[str, str] = {}
    for journal in _visible_journals(current, campaign_root):
        started: set[str] = set()
        for event in journal["direction_events"]:
            if not isinstance(event, dict):
                raise RuntimeStateError("Runtime Journal contains an invalid Direction event")
            direction_id = event.get("direction_id")
            action = event.get("action")
            try:
                advance_direction(statuses, started, direction_id, action)
                if action == "propose":
                    for field in ("name", "hypothesis", "rationale", "recorded_at"):
                        _text(event.get(field), f"Direction {field}")
                    for field in ("plan", "success_criteria", "stop_conditions"):
                        _text_list(event.get(field), f"Direction {field}")
                _text_list(event.get("supporting_experiment_ids", []), "supporting_experiment_ids")
            except ValueError as error:
                raise RuntimeStateError(
                    f"Invalid Direction history in Episode {journal['episode']}: {error}"
                ) from error
            if action == "propose":
                views[direction_id] = {
                    "direction_id": direction_id,
                    "name": event["name"],
                    "hypothesis": event["hypothesis"],
                    "rationale": event["rationale"],
                    "plan": event["plan"],
                    "success_criteria": event["success_criteria"],
                    "stop_conditions": event["stop_conditions"],
                    "status": "proposed",
                    "analysis": None,
                    "recorded_at": event["recorded_at"],
                    "supporting_experiment_ids": [],
                    "associated_experiment_ids": [],
                    "hypothesis_status": "unresolved",
                    **relationship_fields(event),
                }
            elif direction_id in views:
                views[direction_id]["status"] = _DIRECTION_STATUS[action]
                views[direction_id]["analysis"] = event.get("analysis")
                views[direction_id]["updated_at"] = event.get("recorded_at")
                direction = views[direction_id]
                assessment = event.get("hypothesis_status")
                direction["hypothesis_status"] = assessment or "unresolved"
                # Old automatic support lists are associations, not explicit judgments.
                support = event.get("supporting_experiment_ids", [])
                direction["supporting_experiment_ids"] = (
                    list(support) if assessment is not None and action in _CLOSURES else []
                )
                for experiment_id in support:
                    if experiment_id not in direction["associated_experiment_ids"]:
                        direction["associated_experiment_ids"].append(experiment_id)
    for experiment in _visible_experiments(current, campaign_root):
        direction = views.get(str(experiment.get("direction_id")))
        if direction is None:
            continue
        experiment_id = experiment.get("experiment_id")
        if isinstance(experiment_id, str):
            if experiment_id not in direction["associated_experiment_ids"]:
                direction["associated_experiment_ids"].append(experiment_id)
    return views


def _visible_experiments(current: Path, campaign_root: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    seen: set[str] = set()
    for journal in _visible_journals(current, campaign_root):
        experiments = journal.get("experiments", [])
        if not isinstance(experiments, list):
            continue
        for experiment in experiments:
            if not isinstance(experiment, dict):
                continue
            experiment_id = experiment.get("experiment_id")
            if not isinstance(experiment_id, str) or experiment_id in seen:
                continue
            seen.add(experiment_id)
            values.append(dict(experiment))
    return values


def _save_journal(path: Path, campaign_root: Path, journal: dict[str, Any]) -> None:
    atomic_write_json(path, journal)
    try:
        from long_horizon.journal import sync_live_memory

        sync_live_memory(campaign_root / "memory" / "live.json", journal)
    except OSError:
        pass


def _mutable_journal(path: Path) -> dict[str, Any]:
    journal = load_journal(path)
    if journal.get("state") != "in_progress":
        raise AgentRequestError(
            "cannot mutate a finalized Runtime Journal",
            code="journal_finalized", repairable=False,
            next_action=(
                "This Episode has already submitted its report. Do not append more Journal entries."
            ),
        )
    return journal


def _append_event(path: Path, campaign_root: Path, event: dict[str, Any]) -> None:
    journal = _mutable_journal(path)
    events = journal["direction_events"]
    events.append(event)
    _save_journal(path, campaign_root, journal)


def update_direction(
    path: Path, campaign_root: Path, request: Mapping[str, object]
) -> dict[str, object]:
    _mutable_journal(path)
    value = dict(request)
    action = value.get("action")
    views = _direction_views(path, campaign_root)
    statuses = {key: direction["status"] for key, direction in views.items()}
    started = {
        event["direction_id"] for event in load_journal(path)["direction_events"]
        if event["action"] == "start"
    }
    if action == "propose":
        require_fields(
            value, _PROPOSAL_FIELDS, optional=RELATIONSHIP_FIELDS, label="Direction proposal"
        )
        direction_id = f"direction_{uuid.uuid4().hex}"
        advance_direction(statuses, started, direction_id, action)
        ancestry: dict[str, Any] = {}
        if set(value) & RELATIONSHIP_FIELDS:
            ancestry = validate_relationship(
                direction_id,
                value,
                views,
                {
                    str(item["experiment_id"]): item
                    for item in _visible_experiments(path, campaign_root)
                },
            )
        event = {
            "direction_event_id": f"directionevent_{uuid.uuid4().hex}",
            "direction_id": direction_id,
            "action": "propose",
            "name": _text(value["name"], "Direction name"),
            "hypothesis": _text(value["hypothesis"], "Direction hypothesis"),
            "rationale": _text(value["rationale"], "Direction rationale"),
            "plan": _text_list(value["plan"], "Direction plan"),
            "success_criteria": _text_list(value["success_criteria"], "Direction success_criteria"),
            "stop_conditions": _text_list(value["stop_conditions"], "Direction stop_conditions"),
            "recorded_at": _now(),
            **ancestry,
        }
    else:
        if set(value) & RELATIONSHIP_FIELDS:
            raise ValueError(
                "Direction genealogy is immutable; propose a new derived Direction "
                "instead of changing ancestry in a lifecycle update"
            )
        if not isinstance(action, str) or action not in (set(_DIRECTION_STATUS) - {"propose"}):
            raise ValueError(
                "Direction action must be propose, start, complete, abandon, block, or defer"
            )
        require_fields(
            value, _CLOSURE_FIELDS if action in _CLOSURES else _UPDATE_FIELDS,
            label=f"Direction {action}",
        )
        direction_id = _text(value["direction_id"], "Direction ID")
        advance_direction(statuses, started, direction_id, action)
        supporting = (
            _closure_support(path, campaign_root, direction_id, value)
            if action in _CLOSURES else []
        )
        event = {
            "direction_event_id": f"directionevent_{uuid.uuid4().hex}",
            "direction_id": direction_id,
            "action": action,
            "analysis": _text(value["analysis"], "Direction analysis"),
            "recorded_at": _now(),
            "supporting_experiment_ids": supporting,
            "hypothesis_status": value.get("hypothesis_status"),
        }
    _append_event(path, campaign_root, event)
    return {"status": "recorded", "direction_id": direction_id}


def _visible_evidence_roots(evidence_root: Path, campaign_root: Path) -> list[Path]:
    archived = sorted(
        (campaign_root / ".atrex_long_horizon" / "episodes").glob("e*/supervisor_runtime")
    )
    current_path = evidence_root / "journal.json"
    if current_path.exists():
        episode = load_journal(current_path)["episode"]
        visible = []
        for root in archived:
            try:
                if load_journal(root / "journal.json")["episode"] < episode:
                    visible.append(root)
            except RuntimeStateError:
                continue
        archived = visible
    return [*archived, evidence_root]


def _gateway_record(evidence_root: Path, campaign_root: Path, record_id: str) -> dict[str, Any]:
    if GATEWAY_RECORD_ID_RE.fullmatch(record_id) is None:
        raise ValueError("Gateway Record ID must be a valid gateway-... ID returned by the Runtime")
    for evidence in _visible_evidence_roots(evidence_root, campaign_root):
        root = evidence / "gateway-records"
        directory = root / record_id
        path = directory / "result.json"
        if any(item.is_symlink() for item in (evidence, root, directory, path)):
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(record, dict) and isinstance(record.get("result"), dict):
            # Read legacy execution status without rewriting the immutable record.
            raw_path = directory / "raw-result.json"
            if "execution_status" not in record and not raw_path.is_symlink():
                try:
                    raw = json.loads(raw_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError):
                    raw = None
                if isinstance(raw, dict) and isinstance(raw.get("status"), str):
                    record["execution_status"] = raw["status"]
            return record
    raise ValueError(
        f"Gateway Record {record_id} is not visible or is invalid. Use an ID returned by "
        "the Runtime in this Campaign; inspect it with tools/sandbox.py --kind record-read "
        f"--record-id {record_id}"
    )


def _has_passing_evaluate(
    evidence_root: Path, campaign_root: Path, record_ids: object, kernel_digest: str
) -> bool:
    if not isinstance(record_ids, list):
        return False
    for record_id in record_ids:
        if not isinstance(record_id, str):
            continue
        record = _gateway_record(evidence_root, campaign_root, record_id)
        if (
            record.get("gateway_kind") == "run"
            and record.get("kernel_artifact_digest") == kernel_digest
            and record["result"].get("all_pass") is True
            and not record.get("error")
            and not record["result"].get("error")
            and _completed_record(record)
        ):
            return True
    return False


def _validate_record_ids(evidence_root: Path, campaign_root: Path, raw: object) -> list[str]:
    records = _text_list(raw, "Experiment gateway_record_ids")
    if not records:
        raise AgentRequestError(
            "Every Experiment requires at least one real Kernel-bound Gateway Record in "
            "gateway_record_ids, including abandon_direction; historical unmeasured notes "
            "cannot support a new Direction closure.",
            code="experiment_evidence_required",
            next_action=(
                "Cite an actual Evaluate, ABBA, Profile, Dev, Check, or Disassemble result. "
                "A failed diagnostic can document an unresolved blocker. Env/Wiki responses "
                "and transport errors without a Gateway Record do not qualify; never invent evidence."
            ),
        )
    if len(set(records)) != len(records):
        raise ValueError("Experiment gateway_record_ids must not contain duplicates")
    for record_id in records:
        record = _gateway_record(evidence_root, campaign_root, record_id)
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
    """Completion is a measurement fact, not correctness or a hypothesis verdict."""
    result = record["result"]
    execution_status = record.get("execution_status")
    if execution_status is not None:
        return isinstance(execution_status, str) and execution_status in {"completed", "succeeded"}
    if record.get("error") or result.get("error"):
        return False
    status = result.get("status")
    if status is not None:
        return isinstance(status, str) and status in {"completed", "succeeded", "success", "ok"}
    # Legacy successful evaluator/profiler payloads have no status. Require an
    # actual completion signal, never infer completion from an empty dictionary.
    return (
        isinstance(result.get("all_pass"), bool)
        or isinstance(result.get("correct"), bool)
        or isinstance(result.get("passed"), bool)
        or isinstance(result.get("kernels"), list)
        or (isinstance(result.get("exit_code"), int) and not isinstance(result["exit_code"], bool))
    )


def _closure_support(
    path: Path, campaign_root: Path, direction_id: str, value: Mapping[str, object]
) -> list[str]:
    assessment = value.get("hypothesis_status")
    if not isinstance(assessment, str) or assessment not in _HYPOTHESIS_STATUSES:
        raise ValueError("hypothesis_status must be unresolved, supported, or refuted")
    selected = _text_list(value.get("supporting_experiment_ids"), "supporting_experiment_ids")
    if not 1 <= len(selected) <= 32 or len(set(selected)) != len(selected):
        raise AgentRequestError(
            "Direction closure requires 1–32 unique supporting_experiment_ids from this "
            "Direction's visible Experiments.",
            code="direction_support_required",
            next_action=(
                "Use load-direction to find associated_experiment_ids; record-experiment first "
                "if necessary. Select the relevant IDs explicitly and declare hypothesis_status. "
                "Each Experiment must cite real Kernel-bound Gateway evidence."
            ),
        )
    experiments = {item["experiment_id"]: item for item in _visible_experiments(path, campaign_root)}
    for experiment_id in selected:
        experiment = experiments.get(experiment_id)
        if experiment is None:
            raise ValueError(f"Supporting Experiment {experiment_id} is outside visible history")
        if experiment.get("direction_id") != direction_id:
            raise ValueError("Supporting Experiment must belong to the Direction being closed")
        records = _validate_record_ids(path.parent, campaign_root, experiment.get("gateway_record_ids"))
        if assessment != "unresolved" and not any(
            _completed_record(_gateway_record(path.parent, campaign_root, record_id))
            for record_id in records
        ):
            raise AgentRequestError(
                "supported/refuted requires a completed Kernel-bound Gateway observation in "
                "every selected Experiment; failed or unfinished jobs do not establish a verdict.",
                code="direction_assessment_requires_completed_result",
                next_action=(
                    "Use hypothesis_status=unresolved for infrastructure failures or an untested "
                    "claim. Cite a completed result only if it bears on the stated hypothesis; "
                    "Runtime validates record bindings, not causal relevance or scientific truth."
                ),
            )
    return selected


def record_experiment(
    path: Path,
    campaign_root: Path,
    evidence_root: Path,
    request: Mapping[str, object],
) -> dict[str, object]:
    value = dict(request)
    current = _mutable_journal(path)
    require_fields(value, _EXPERIMENT_FIELDS, label="Experiment")
    direction_id = _text(value["direction_id"], "Experiment Direction ID")
    if DIRECTION_ID_RE.fullmatch(direction_id) is None:
        raise ValueError("Experiment Direction ID has an invalid format")
    direction = _direction_views(path, campaign_root).get(direction_id)
    if direction is None:
        raise ValueError("Experiment Direction is outside visible history")
    if direction["status"] not in {"in_progress", *_CLOSED_STATUSES}:
        raise ValueError(
            "Experiment Direction must be in progress or closed; start a proposed Direction "
            "before recording evidence. Late evidence does not reopen exploration"
        )
    action = value.get("action")
    if not isinstance(action, str) or action not in _EXPERIMENT_ACTIONS:
        raise ValueError(f"Experiment action must be one of {sorted(_EXPERIMENT_ACTIONS)}")
    record_ids = _validate_record_ids(evidence_root, campaign_root, value["gateway_record_ids"])
    if action == "baseline" and any(
        item.get("action") == "baseline" for item in current["experiments"]
    ):
        raise ValueError("an Episode may record baseline only once")
    experiment = {
        "experiment_id": f"experiment_{uuid.uuid4().hex}",
        "direction_id": direction_id,
        "sequence": len(current["experiments"]) + 1,
        "recorded_at": _now(),
        "name": _text(value["name"], "Experiment name"),
        "hypothesis": _text(value["hypothesis"], "Experiment hypothesis"),
        "change": _text(value["change"], "Experiment change"),
        "gateway_record_ids": record_ids,
        "evidence": _text(value["evidence"], "Experiment evidence"),
        "analysis": _text(value["analysis"], "Experiment analysis"),
        "action": action,
    }
    current["experiments"].append(experiment)
    _save_journal(path, campaign_root, current)
    return {"status": "recorded", "experiment_id": experiment["experiment_id"]}


def _safe_scratch_path(workspace: Path, value: object) -> tuple[Path, str]:
    text = _text(value, "Journal output file")
    relative = PurePosixPath(text)
    if relative.is_absolute() or not relative.parts or relative.parts[0] != "scratch":
        raise ValueError("Journal output file must be a workspace-relative path under scratch/")
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("Journal output file contains an unsafe path component")
    target = workspace.joinpath(*relative.parts)
    current = workspace / "scratch"
    if current.exists() and (current.is_symlink() or not current.is_dir()):
        raise ValueError("workspace scratch path is unsafe")
    current.mkdir(mode=0o700, exist_ok=True)
    for part in relative.parts[1:-1]:
        current /= part
        if current.exists() and (current.is_symlink() or not current.is_dir()):
            raise ValueError("Journal output file traverses an unsafe directory")
        current.mkdir(mode=0o700, exist_ok=True)
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise ValueError("Journal output file must name a regular file, not a symlink or directory")
    return target, relative.as_posix()


def _write_index(workspace: Path, file: object, payload: dict[str, Any]) -> dict[str, object]:
    target, public = _safe_scratch_path(workspace, file)
    atomic_write_json(target, payload)
    count = len(next(iter(payload.values())))
    return {"status": "written", "file": public, "count": count}


def validate_report_evidence(
    path: Path, campaign_root: Path, workspace: Path, *, status: str, selected_id: object,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], bytes | None]:
    """Read-only preflight shared by HTTP submission and Supervisor terminal rechecks."""
    current = load_journal(path)
    directions = _direction_views(path, campaign_root)
    in_progress = [
        item["direction_id"] for item in directions.values() if item["status"] == "in_progress"
    ]
    if in_progress:
        raise ValueError(
            "Episode Report cannot leave a Direction in progress; record its real Gateway "
            "evidence as an Experiment, then close with hypothesis_status and "
            "supporting_experiment_ids: " + ", ".join(in_progress)
        )
    for event in current["direction_events"]:
        if event.get("hypothesis_status") is not None:
            _closure_support(path, campaign_root, event["direction_id"], event)
    if status != "candidate_ready":
        if selected_id is not None and selected_id != "":
            raise ValueError(f"{status} cannot include selected_experiment_id")
        return current, directions, None
    if not isinstance(selected_id, str) or EXPERIMENT_ID_RE.fullmatch(selected_id) is None:
        raise ValueError("candidate_ready requires a valid selected_experiment_id")
    matches = [
        item for item in current["experiments"]
        if isinstance(item, dict) and item.get("experiment_id") == selected_id
    ]
    if len(matches) != 1:
        raise ValueError("selected_experiment_id must identify exactly one Experiment in the current Episode")
    selected = matches[0]
    action = selected.get("action")
    if not isinstance(action, str) or action not in {"keep_after", "adopt", "baseline"}:
        raise ValueError("selected Experiment must use baseline, keep_after, or adopt")
    direction_id = selected.get("direction_id")
    if not isinstance(direction_id, str) or direction_id not in directions:
        raise ValueError("selected Experiment Direction is outside visible history")
    if directions[direction_id]["status"] not in _CLOSED_STATUSES:
        raise ValueError("selected Experiment Direction must have been started and closed")
    records = _validate_record_ids(path.parent, campaign_root, selected.get("gateway_record_ids"))
    kernel = workspace / "kernel.py"
    if kernel.is_symlink() or not kernel.is_file():
        raise ValueError("candidate kernel.py must be a regular file")
    kernel_source = kernel.read_bytes()
    kernel_digest = "sha256:" + hashlib.sha256(kernel_source).hexdigest()
    if not _has_passing_evaluate(path.parent, campaign_root, records, kernel_digest):
        raise ValueError(
            "selected Experiment must cite a passing Evaluate Gateway record whose "
            "Kernel exactly matches current kernel.py; inspect the cited records "
            "with tools/sandbox.py --kind record-read --record-id gateway-..."
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
    ) -> None:
        self.workspace = workspace
        self.git_workspace = git_workspace or workspace
        self.campaign_root = campaign_root
        self.evidence_root = evidence_root
        self.path = evidence_root / "journal.json"

    def execute(self, request: Mapping[str, object]) -> dict[str, object]:
        operation = request.get("operation")
        if not isinstance(operation, str):
            raise ValueError("Journal operation must be text")
        if operation == "direction_update":
            body = request.get("request")
            if not isinstance(body, Mapping):
                raise ValueError("direction_update requires a request object")
            return update_direction(self.path, self.campaign_root, body)
        if operation == "directions_list":
            directions = [
                {
                    "direction_id": item["direction_id"],
                    "name": item["name"],
                    "status": item["status"],
                    "hypothesis_status": item["hypothesis_status"],
                    **relationship_fields(item),
                }
                for item in _direction_views(self.path, self.campaign_root).values()
            ]
            return _write_index(self.workspace, request.get("file"), {"directions": directions})
        if operation == "direction_load":
            direction_id = _text(request.get("direction_id"), "Direction ID")
            direction = _direction_views(self.path, self.campaign_root).get(direction_id)
            if direction is None:
                raise ValueError("Direction ID is outside visible history")
            return direction
        if operation == "experiment_record":
            body = request.get("request")
            if not isinstance(body, Mapping):
                raise ValueError("experiment_record requires a request object")
            return record_experiment(
                self.path,
                self.campaign_root,
                self.evidence_root,
                body,
            )
        if operation == "experiments_list":
            experiments = [
                {
                    key: item[key]
                    for key in (
                        "experiment_id",
                        "name",
                        "hypothesis",
                        "change",
                        "gateway_record_ids",
                        "evidence",
                        "analysis",
                        "action",
                    )
                }
                for item in _visible_experiments(self.path, self.campaign_root)
            ]
            return _write_index(self.workspace, request.get("file"), {"experiments": experiments})
        if operation == "experiment_load":
            experiment_id = _text(request.get("experiment_id"), "Experiment ID")
            for experiment in _visible_experiments(self.path, self.campaign_root):
                if experiment.get("experiment_id") == experiment_id:
                    visible = dict(experiment)
                    visible.pop("sequence", None)
                    return visible
            raise ValueError("Experiment ID is outside visible history")
        if operation == "episode_report":
            body = request.get("request")
            if not isinstance(body, Mapping):
                raise ValueError("episode_report requires a request object")
            return self._submit_report(body)
        raise ValueError(f"unsupported Journal operation: {operation}")

    def _submit_report(self, raw: Mapping[str, object]) -> dict[str, object]:
        allowed = {
            "status",
            "summary",
            "selected_experiment_id",
            "blocker",
            "accepted_ppu_diagnostics",
        }
        require_fields(dict(raw), {"status", "summary"}, optional=allowed, label="Episode Report")
        status = raw.get("status")
        if not isinstance(status, str) or status not in {
            "candidate_ready",
            "pivot",
            "blocked",
        }:
            raise ValueError("Episode Report status must be candidate_ready, pivot, or blocked")
        summary = _text(raw.get("summary"), "Episode Report summary")
        blocker = raw.get("blocker")
        if status == "blocked":
            _text(blocker, "Episode Report blocker")
        elif blocker is not None and blocker != "":
            raise ValueError(f"Episode Report status {status} cannot include blocker")
        selected_id = raw.get("selected_experiment_id")
        current, directions, kernel_source = validate_report_evidence(
            self.path, self.campaign_root, self.workspace, status=status, selected_id=selected_id,
        )
        candidate_commit = ""
        next_directions = [
            item["name"]
            for item in directions.values()
            if item["status"] in {"proposed", "deferred"}
        ]
        accepted_ppu_diagnostics: list[dict[str, Any]] = []
        if "accepted_ppu_diagnostics" in raw:
            from long_horizon.journal import (
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
            from long_horizon.git_episode import EpisodeWorktree

            candidate_commit = EpisodeWorktree(
                episode=current["episode"],
                base_commit=current["base_commit"],
                branch=current["episode_branch"],
                path=self.git_workspace,
            ).commit_candidate(kernel_source)
        current["state"] = status
        current["outcome"] = {
            "summary": summary,
            "next_directions": next_directions,
            "selected_experiment_id": selected_id if status == "candidate_ready" else None,
            "blocker": str(blocker).strip() if status == "blocked" else None,
        }
        if accepted_ppu_diagnostics:
            current["outcome"]["accepted_ppu_diagnostics"] = accepted_ppu_diagnostics
        current["candidate_commit"] = candidate_commit or None
        current["finalized_at"] = _now()
        _save_journal(self.path, self.campaign_root, current)
        handoff: dict[str, str] = {"status": str(status)}
        if candidate_commit:
            handoff["candidate_commit"] = candidate_commit
        handoff_path = self.git_workspace / ".atrex_long_horizon" / "handoff.json"
        atomic_write_json(handoff_path, handoff)
        return {
            "status": "accepted",
            "message": (
                "Report accepted; Supervisor committed the measured Kernel for verification"
                if candidate_commit else "Report accepted and recorded"
            ),
        }


__all__ = ["SupervisorJournalService", "initialize_journal", "load_journal"]
