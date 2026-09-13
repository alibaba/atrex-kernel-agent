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
from supervisor.errors import AgentRequestError, RuntimeStateError, require_fields

GATEWAY_RECORD_ID_RE = re.compile(r"gateway-[0-9]+-[0-9a-f]{12}")
DIRECTION_ID_RE = re.compile(r"direction_[0-9a-f]{32}")
EXPERIMENT_ID_RE = re.compile(r"experiment_[0-9a-f]{32}")

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
_DIRECTION_STATUS = {
    "propose": "proposed",
    "start": "in_progress",
    "complete": "completed",
    "abandon": "abandoned",
    "block": "blocked",
    "defer": "deferred",
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
    for journal in _visible_journals(current, campaign_root):
        events = journal.get("direction_events", [])
        if not isinstance(events, list):
            continue
        for event in events:
            if not isinstance(event, dict):
                continue
            direction_id = event.get("direction_id")
            action = event.get("action")
            if not isinstance(direction_id, str) or action not in _DIRECTION_STATUS:
                continue
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
                    **relationship_fields(event),
                }
            elif direction_id in views:
                views[direction_id]["status"] = _DIRECTION_STATUS[action]
                views[direction_id]["analysis"] = event.get("analysis")
                views[direction_id]["updated_at"] = event.get("recorded_at")
    for experiment in _visible_experiments(current, campaign_root):
        direction = views.get(str(experiment.get("direction_id")))
        if direction is None:
            continue
        experiment_id = experiment.get("experiment_id")
        if isinstance(experiment_id, str):
            direction["supporting_experiment_ids"].append(experiment_id)
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


def _append_event(path: Path, campaign_root: Path, event: dict[str, Any]) -> None:
    journal = load_journal(path)
    if journal.get("state") != "in_progress":
        raise AgentRequestError(
            "cannot mutate a finalized Runtime Journal",
            code="journal_finalized", repairable=False,
            next_action=(
                "This Episode has already submitted its report. Do not append more Journal entries."
            ),
        )
    events = journal["direction_events"]
    events.append(event)
    _save_journal(path, campaign_root, journal)


def update_direction(
    path: Path, campaign_root: Path, request: Mapping[str, object]
) -> dict[str, object]:
    value = dict(request)
    action = value.get("action")
    if action == "propose":
        require_fields(
            value, _PROPOSAL_FIELDS, optional=RELATIONSHIP_FIELDS, label="Direction proposal"
        )
        direction_id = f"direction_{uuid.uuid4().hex}"
        ancestry: dict[str, Any] = {}
        if set(value) & RELATIONSHIP_FIELDS:
            ancestry = validate_relationship(
                direction_id,
                value,
                _direction_views(path, campaign_root),
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
        require_fields(value, _UPDATE_FIELDS, label="Direction update")
        if not isinstance(action, str) or action not in (set(_DIRECTION_STATUS) - {"propose"}):
            raise ValueError(
                "Direction update requires exactly action, direction_id, and analysis; "
                "action must be start, complete, abandon, block, or defer"
            )
        direction_id = _text(value["direction_id"], "Direction ID")
        if DIRECTION_ID_RE.fullmatch(direction_id) is None:
            raise ValueError("Direction ID has an invalid format")
        views = _direction_views(path, campaign_root)
        direction = views.get(direction_id)
        if direction is None:
            raise ValueError("Direction ID is outside visible history")
        status = direction["status"]
        if action == "start":
            active = [
                item["direction_id"]
                for item in views.values()
                if item["status"] == "in_progress" and item["direction_id"] != direction_id
            ]
            if active:
                raise AgentRequestError(
                    "only one Direction may be in progress; close or defer "
                    f"{active[0]} before starting another",
                    code="direction_in_progress",
                    direction_id=active[0],
                    next_action=(
                        "Use update-direction to complete, abandon, block, or defer that Direction "
                        "before starting another. complete/abandon require a supporting Experiment."
                    ),
                )
            journal = load_journal(path)
            started = {
                event.get("direction_id")
                for event in journal["direction_events"]
                if isinstance(event, dict) and event.get("action") == "start"
            }
            if direction_id not in started and len(started) >= 3:
                raise AgentRequestError(
                    "Direction advancement limit exceeded: at most three Directions "
                    "may be started in one Episode",
                    code="direction_limit_exceeded",
                    started_direction_ids=sorted(started),
                    next_action=(
                        "Do not start a fourth Direction. Continue an already-started Direction "
                        "if it is in_progress, or resume it if deferred; otherwise close remaining "
                        "in_progress Directions and submit episode-report. You may still propose "
                        "new Directions for later Episodes."
                    ),
                )
            if status not in {"proposed", "deferred"}:
                raise ValueError(f"Direction cannot start from status {status}")
        elif action in {"complete", "abandon"}:
            if status != "in_progress":
                raise ValueError(f"Direction cannot {action} from status {status}")
            supporting = [
                item.get("experiment_id")
                for item in load_journal(path)["experiments"]
                if isinstance(item, dict) and item.get("direction_id") == direction_id
            ]
            if not supporting:
                raise ValueError(
                    f"Direction cannot {action} before recording a supporting Experiment"
                )
        elif action in {"block", "defer"} and status != "in_progress":
            raise ValueError(f"Direction cannot {action} from status {status}")
        event = {
            "direction_event_id": f"directionevent_{uuid.uuid4().hex}",
            "direction_id": direction_id,
            "action": action,
            "analysis": _text(value["analysis"], "Direction analysis"),
            "recorded_at": _now(),
        }
    _append_event(path, campaign_root, event)
    return {"status": "recorded", "direction_id": direction_id}


def _visible_evidence_roots(evidence_root: Path, campaign_root: Path) -> list[Path]:
    archived = sorted(
        (campaign_root / ".atrex_long_horizon" / "episodes").glob("e*/supervisor_runtime")
    )
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
        ):
            return True
    return False


def _validate_record_ids(evidence_root: Path, campaign_root: Path, raw: object) -> list[str]:
    records = _text_list(raw, "Experiment gateway_record_ids")
    if len(set(records)) != len(records):
        raise ValueError("Experiment gateway_record_ids must not contain duplicates")
    for record_id in records:
        _gateway_record(evidence_root, campaign_root, record_id)
    return records


def record_experiment(
    path: Path,
    campaign_root: Path,
    evidence_root: Path,
    request: Mapping[str, object],
) -> dict[str, object]:
    value = dict(request)
    require_fields(value, _EXPERIMENT_FIELDS, label="Experiment")
    direction_id = _text(value["direction_id"], "Experiment Direction ID")
    if DIRECTION_ID_RE.fullmatch(direction_id) is None:
        raise ValueError("Experiment Direction ID has an invalid format")
    direction = _direction_views(path, campaign_root).get(direction_id)
    if direction is None:
        raise ValueError("Experiment Direction is outside visible history")
    if direction["status"] != "in_progress":
        raise ValueError(
            "Experiment Direction must be in_progress; start it before recording evidence"
        )
    action = value.get("action")
    if not isinstance(action, str) or action not in _EXPERIMENT_ACTIONS:
        raise ValueError(f"Experiment action must be one of {sorted(_EXPERIMENT_ACTIONS)}")
    record_ids = _validate_record_ids(evidence_root, campaign_root, value["gateway_record_ids"])
    if not record_ids and action != "abandon_direction":
        raise ValueError(
            f"{action} requires at least one gateway_record_id in gateway_record_ids; "
            "cite the measured results used in this Experiment"
        )
    current = load_journal(path)
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
        directions = _direction_views(self.path, self.campaign_root)
        in_progress = [
            item["direction_id"] for item in directions.values() if item["status"] == "in_progress"
        ]
        if in_progress:
            raise ValueError(
                "Episode Report cannot leave a Direction in progress; block or defer "
                + ", ".join(in_progress)
            )
        current = load_journal(self.path)
        if status == "pivot" and not current["experiments"]:
            raise ValueError("pivot requires at least one current-Episode Experiment")
        selected_id = raw.get("selected_experiment_id")
        candidate_commit = ""
        kernel_source: bytes | None = None
        if status == "candidate_ready":
            if not isinstance(selected_id, str) or EXPERIMENT_ID_RE.fullmatch(selected_id) is None:
                raise ValueError("candidate_ready requires a valid selected_experiment_id")
            selected = next(
                (
                    item
                    for item in current["experiments"]
                    if item.get("experiment_id") == selected_id
                ),
                None,
            )
            if selected is None:
                raise ValueError("selected_experiment_id must belong to the current Episode")
            if selected.get("action") not in {"keep_after", "adopt", "baseline"}:
                raise ValueError("selected Experiment must use baseline, keep_after, or adopt")
            kernel = self.workspace / "kernel.py"
            if kernel.is_symlink() or not kernel.is_file():
                raise ValueError("candidate kernel.py must be a regular file")
            kernel_source = kernel.read_bytes()
            kernel_digest = "sha256:" + hashlib.sha256(kernel_source).hexdigest()
            if not _has_passing_evaluate(
                self.evidence_root,
                self.campaign_root,
                selected.get("gateway_record_ids"),
                kernel_digest,
            ):
                raise ValueError(
                    "selected Experiment must cite a passing Evaluate Gateway record whose "
                    "Kernel exactly matches current kernel.py; inspect the cited records "
                    "with tools/sandbox.py --kind record-read --record-id gateway-..."
                )
        else:
            if selected_id is not None and selected_id != "":
                raise ValueError(f"{status} cannot include selected_experiment_id")
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
