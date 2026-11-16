#!/usr/bin/env python3
"""Shared, read-only helpers for the public AKA Light observability commands."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for entry in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from aka.workflow.legacy.export import RetainedWorkflowExport  # noqa: E402
from aka.evaluation.response import result_from_response  # noqa: E402
from aka.experiment.measurement_store import (  # noqa: E402
    MeasurementStore,
    RECORD_ID,
)
from aka.workflow.light.support.paths import (  # noqa: E402
    campaign_runtime_configuration,
    legacy_supervisor_campaign_root,
    supervisor_campaign_base_root,
    supervisor_campaign_root,
    supervisor_measurement_root,
)
from aka.workflow.light.search import (  # noqa: E402
    COORDINATOR_STATE,
    read_state as read_search_state,
)


TERMINAL_EPISODE_STATUSES = frozenset(
    {
        "accepted",
        "candidate_ready",
        "pivot",
        "blocked",
        "invalid_handoff",
        "interrupted",
        "failed",
        "completed",
        "rejected",
    }
)
TERMINAL_CAMPAIGN_PHASES = frozenset({"completed", "failed", "interrupted", "stopped"})
KIND_PATTERN = re.compile(r"--kind(?:=|\s+)([a-zA-Z0-9_-]+)")


@dataclass(frozen=True)
class Scope:
    """One controller workspace and its private Supervisor state."""

    workspace: Path
    campaign_root: Path
    label: str
    trajectory: int | None

    @property
    def measurement_root(self) -> Path:
        canonical = supervisor_measurement_root(self.workspace)
        if canonical.exists() or canonical.is_symlink():
            return canonical
        legacy = self.campaign_root / "measurements"
        if (legacy / "identity.key").is_file():
            return legacy
        return canonical

    @property
    def audit_root(self) -> Path:
        return self.campaign_root

    @property
    def agent_workspace(self) -> Path:
        """Return the persistent workspace actually exposed to the Agent.

        The controller repository and Agent repository are the same directory
        only in compatibility layouts.  Current Light keeps the Agent checkout
        below private Supervisor state, including for pooled trajectories.
        Observability projections such as ``PROGRESS.md`` must follow that
        boundary instead of assuming they live in the controller repository.
        """
        candidate = self.audit_root / "agent-workspace"
        return candidate if candidate.is_dir() else self.workspace

    @property
    def segment_root(self) -> Path:
        return self.audit_root / "segments"


def resolve_workspace(value: Path) -> Path:
    workspace = value.expanduser().resolve()
    if not workspace.is_dir():
        raise ValueError(f"Campaign workspace does not exist: {workspace}")
    return workspace


def read_json(path: Path, *, default: Any = None, limit: int = 64 * 1024 * 1024) -> Any:
    try:
        if path.is_symlink() or not path.is_file():
            return default
        if path.stat().st_size > limit:
            raise ValueError(f"JSON exceeds {limit} bytes: {path}")
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def discover_scopes(workspace: Path) -> list[Scope]:
    workspace = resolve_workspace(workspace)
    campaign_root = supervisor_campaign_base_root(workspace)
    scopes = [Scope(workspace, campaign_root, "campaign", None)]
    lineages = campaign_root / "lineages"
    if lineages.is_dir():
        for lineage in sorted(p for p in lineages.glob("t*") if p.is_dir()):
            match = re.fullmatch(r"t0*([1-9][0-9]*)", lineage.name)
            number = int(match.group(1)) if match else None
            agent_workspace = lineage / "agent-workspace"
            if number is None or not agent_workspace.is_dir():
                continue
            scopes.append(
                Scope(
                    agent_workspace.resolve(),
                    lineage.resolve(),
                    lineage.name,
                    number,
                )
            )
        return scopes
    trajectories = campaign_root / "search" / "trajectories"
    if trajectories.is_dir():
        for path in sorted(p for p in trajectories.glob("t*") if p.is_dir()):
            match = re.fullmatch(r"t0*([1-9][0-9]*)", path.name)
            number = int(match.group(1)) if match else None
            lineage_root = supervisor_campaign_root(path)
            legacy_root = legacy_supervisor_campaign_root(path)
            audit_root = (
                legacy_root
                if not lineage_root.exists() and legacy_root.exists()
                else lineage_root
            )
            scopes.append(Scope(path.resolve(), audit_root, path.name, number))
    return scopes


def load_snapshot(scope: Scope):
    return RetainedWorkflowExport().read(scope.workspace)


def existing_store(scope: Scope) -> MeasurementStore | None:
    root = scope.measurement_root
    return MeasurementStore(root) if (root / "identity.key").is_file() else None


def iter_record_directories(scope: Scope) -> Iterable[Path]:
    root = scope.measurement_root / "records"
    if root.is_dir():
        yield from sorted(
            path for path in root.glob("gateway-*")
            if path.is_dir() and RECORD_ID.fullmatch(path.name)
        )


def scope_record_ids(scope: Scope) -> set[str] | None:
    """Return Records created by one lineage; ``None`` means all Records.

    The canonical bytes are Campaign-wide. Runtime request audits retain the
    lineage that issued each call, so lineage views do not duplicate every
    central Record. A serial Campaign has no lineage partition and owns the
    complete Store.
    """
    if scope.trajectory is None:
        if (scope.campaign_root / "lineages").is_dir() or (
            scope.campaign_root / "search" / "trajectories"
        ).is_dir():
            return set()
        return None
    record_ids: set[str] = set()
    for path in runtime_audit_paths(scope):
        value = read_json(path, default={})
        if not isinstance(value, dict):
            continue
        record_id = value.get("gateway_record_id")
        if isinstance(record_id, str) and RECORD_ID.fullmatch(record_id):
            record_ids.add(record_id)
    return record_ids


def runtime_audit_paths(scope: Scope) -> list[Path]:
    """Return current request/plugin audits plus legacy flat receipts once."""
    paths = [
        *scope.audit_root.glob("runtime-audit/requests/request-*.json"),
        *scope.audit_root.glob("runtime-audit/plugin-calls/request-*.json"),
        *scope.audit_root.glob("request-*.json"),
    ]
    return sorted({path.resolve(): path for path in paths}.values())


def iter_records(scope: Scope, *, include_incomplete: bool = False) -> Iterable[dict[str, Any]]:
    store = existing_store(scope)
    selected = scope_record_ids(scope)
    for directory in iter_record_directories(scope):
        if selected is not None and directory.name not in selected:
            continue
        request = read_json(directory / "request.json", default={})
        if not isinstance(request, dict):
            request = {}
        row: dict[str, Any] = {
            "scope": scope.label,
            "trajectory": scope.trajectory,
            "workspace": str(scope.workspace),
            "gateway_record_id": directory.name,
            "directory": directory,
            "request": request,
            "complete": False,
        }
        if store is not None and (directory / "record.json").is_file():
            try:
                record = store.read(directory.name)
            except (OSError, ValueError, KeyError, TypeError, RecursionError) as error:
                row["read_error"] = str(error)
            else:
                row.update(record=record, request=record.get("request", request), complete=True)
        if row["complete"] or include_incomplete:
            yield row


def operation_name(request_or_record: dict[str, Any]) -> str:
    operation = request_or_record.get("operation")
    if operation is None and isinstance(request_or_record.get("request"), dict):
        operation = request_or_record["request"].get("operation")
    mapping = {
        "eval": "evaluate",
        "evaluate": "evaluate",
        "abba": "abba",
        "same_allocation_abba": "abba",
        "profile": "profile",
        "check": "check",
        "disassemble": "disassemble",
        "dev": "dev",
    }
    return mapping.get(str(operation), str(operation or "unknown"))


def evaluation_result(record: dict[str, Any]) -> dict[str, Any] | None:
    response = record.get("response")
    if not isinstance(response, dict):
        return None
    operation = str(record.get("operation") or "")
    try:
        value = result_from_response(response, operation)
    except (ValueError, KeyError, TypeError, AttributeError):
        return None
    return value if isinstance(value, dict) else None


def finite_positive(value: object) -> float | None:
    if type(value) not in {int, float}:
        return None
    number = float(value)
    return number if math.isfinite(number) and number > 0 else None


def parse_timestamp(value: object) -> datetime | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        try:
            return datetime.fromtimestamp(float(value), timezone.utc)
        except (ValueError, OverflowError, OSError):
            return None
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def elapsed_seconds(start: object, finish: object) -> float | None:
    left, right = parse_timestamp(start), parse_timestamp(finish)
    if left is None or right is None:
        return None
    return max(0.0, (right - left).total_seconds())


def age_seconds(value: object, *, now: datetime | None = None) -> float | None:
    stamp = parse_timestamp(value)
    if stamp is None:
        return None
    return max(0.0, ((now or datetime.now(timezone.utc)) - stamp).total_seconds())


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item) for item in value]
    return value


def observed_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def search_state(workspace: Path) -> dict[str, Any] | None:
    campaign_root = supervisor_campaign_base_root(workspace)
    path = campaign_root / COORDINATOR_STATE
    if not path.is_file():
        return None
    value = read_search_state(path)
    if not isinstance(value, dict):
        return None
    # Coordinator state intentionally contains no per-lineage mutable facts.
    # Compose those facts only in this read-only operator view.
    lineages = {}
    for lineage in sorted((campaign_root / "lineages").glob("t*")):
        match = re.fullmatch(r"t0*([1-9][0-9]*)", lineage.name)
        state_path = lineage / "state.json"
        if match is None or not state_path.is_file():
            continue
        state = read_json(state_path, default=None)
        if isinstance(state, dict):
            lineages[str(int(match.group(1)))] = state
    return {**value, "lineages": lineages}


def startup_stage(workspace: Path) -> str:
    """Current Light starts directly in the optimization Session."""
    _ = workspace
    return "bootstrap"


def attempt_documents(scope: Scope) -> list[dict[str, Any]]:
    runtime = scope.segment_root
    by_episode: dict[int, dict[str, Any]] = {}
    state = read_json(runtime / "state.json", default={})
    if isinstance(state, dict):
        rows = state.get("attempts")
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict) and type(row.get("episode")) is int:
                    by_episode[row["episode"]] = row
    for path in sorted(runtime.glob("episodes/e*/attempt.json")):
        row = read_json(path, default={})
        if isinstance(row, dict) and type(row.get("episode")) is int:
            by_episode[row["episode"]] = row
    return [by_episode[key] for key in sorted(by_episode)]


def telemetry_documents(scope: Scope) -> list[dict[str, Any]]:
    runtime = scope.segment_root / "episodes"
    rows = []
    for path in sorted(runtime.glob("e*/telemetry.summary.json")):
        value = read_json(path, default={})
        if isinstance(value, dict):
            rows.append(value)
    return rows


def job_state_rows(record_directory: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(record_directory.glob("**/jobs/*/state.json")):
        state = read_json(path, default={})
        if not isinstance(state, dict):
            continue
        process = state.get("process") if isinstance(state.get("process"), dict) else {}
        job = None
        try:
            candidate = json.loads(process.get("stdout") or "")
            if isinstance(candidate, dict):
                job = candidate
        except (ValueError, TypeError):
            pass
        rows.append(
            {
                "path": str(path),
                "checkpoint_id": path.parent.name,
                "checkpoint_updated_at": (
                    datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
                    if path.exists()
                    else None
                ),
                "phase": state.get("phase"),
                "submission": state.get("submission"),
                "retries": state.get("retries") if isinstance(state.get("retries"), dict) else {},
                "identity": state.get("identity"),
                "runtime_request_ids": [
                    value
                    for value in state.get("runtime_request_ids", [])
                    if isinstance(value, str) and value
                ],
                "phase_observations": [
                    dict(value)
                    for value in state.get("phase_observations", [])
                    if isinstance(value, dict)
                ],
                "process": process,
                "job": job,
            }
        )
    return rows


_JOB_SNAPSHOT = re.compile(r"submission-(\d+)-([a-z0-9_-]+)\.json")
_JOB_PHASE_PRIORITY = {
    "submitting": 10,
    "retry_pending": 20,
    "accepted": 30,
    "rejected": 40,
    "terminal": 50,
}


def job_attempt_rows(record_directory: Path) -> list[dict[str, Any]]:
    """Return every durably captured physical submission below one Record.

    ``state.json`` contains only the latest state of a checkpoint.  The
    adjacent ``submission-N-PHASE.json`` files retain earlier failed/retried
    Agate Jobs.  Collapse the accepted and terminal snapshots of the same
    submission into one row, preferring the most advanced phase, while still
    supporting legacy checkpoints that only contain ``state.json``.
    """
    rows: list[dict[str, Any]] = []
    for state_path in sorted(record_directory.glob("**/jobs/*/state.json")):
        job_directory = state_path.parent
        snapshots = sorted(job_directory.glob("submission-*.json"))
        # Always consider state.json as well.  New checkpoints write matching
        # submission snapshots, while partially migrated checkpoints may have
        # old snapshots plus a newer state that was never snapshotted.
        candidates = [*snapshots, state_path]
        by_submission: dict[int, tuple[int, dict[str, Any]]] = {}
        for path in candidates:
            state = read_json(path, default={})
            if not isinstance(state, dict):
                continue
            process = (
                state.get("process")
                if isinstance(state.get("process"), dict)
                else {}
            )
            if not process:
                continue
            submission = state.get("submission")
            match = _JOB_SNAPSHOT.fullmatch(path.name)
            if type(submission) is not int and match is not None:
                submission = int(match.group(1))
            if type(submission) is not int:
                submission = 0
            phase = state.get("phase")
            if not isinstance(phase, str) and match is not None:
                phase = match.group(2)
            phase = phase if isinstance(phase, str) else "unknown"
            job = None
            try:
                candidate = json.loads(process.get("stdout") or "")
                if isinstance(candidate, dict):
                    job = candidate
            except (ValueError, TypeError):
                pass
            try:
                relative_checkpoint = job_directory.relative_to(
                    record_directory
                ).as_posix()
            except ValueError:
                relative_checkpoint = str(job_directory)
            row = {
                "path": str(path),
                "checkpoint_path": relative_checkpoint,
                "checkpoint_id": job_directory.name,
                "checkpoint_updated_at": (
                    datetime.fromtimestamp(
                        path.stat().st_mtime, timezone.utc
                    ).isoformat()
                    if path.exists()
                    else None
                ),
                "phase": phase,
                "submission": submission,
                "retries": (
                    state.get("retries")
                    if isinstance(state.get("retries"), dict)
                    else {}
                ),
                "identity": state.get("identity"),
                "runtime_request_ids": [
                    value
                    for value in state.get("runtime_request_ids", [])
                    if isinstance(value, str) and value
                ],
                "phase_observations": [
                    dict(value)
                    for value in state.get("phase_observations", [])
                    if isinstance(value, dict)
                ],
                "process": process,
                "job": job,
            }
            priority = _JOB_PHASE_PRIORITY.get(phase, 0)
            existing = by_submission.get(submission)
            if existing is None or priority > existing[0]:
                by_submission[submission] = (priority, row)

        rows.extend(value[1] for _, value in sorted(by_submission.items()))
    return rows


def job_identifier(job: object) -> str | None:
    if not isinstance(job, dict):
        return None
    for key in ("job_id", "id", "task_id"):
        value = job.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def normalized_job_status(state: dict[str, Any]) -> str:
    """Normalize one durable Agate Job snapshot for all observability tools."""
    job = state.get("job")
    raw = str(job.get("status") or "").casefold() if isinstance(job, dict) else ""
    aliases = {
        "success": "succeeded", "succeeded": "succeeded", "completed": "succeeded",
        "failure": "failed", "failed": "failed", "error": "failed",
        "canceled": "cancelled", "cancelled": "cancelled",
        "queued": "queued", "pending": "queued", "running": "running",
    }
    if raw in aliases:
        return aliases[raw]
    phase = str(state.get("phase") or "").casefold()
    if phase == "rejected":
        return "rejected"
    process = state.get("process")
    returncode = process.get("returncode") if isinstance(process, dict) else None
    if phase == "terminal" and type(returncode) is int and returncode != 0:
        return "failed"
    return "unknown"


def job_error_details(job: object, response: object = None) -> dict[str, Any] | None:
    """Extract and classify the structured error of one Agate attempt."""
    candidates: list[object] = []
    if isinstance(job, dict):
        candidates.append(job.get("error"))
        result = job.get("result")
        if isinstance(result, dict):
            candidates.append(result.get("error"))
    if isinstance(response, dict):
        for stream in (response.get("stderr"), response.get("stdout")):
            if not isinstance(stream, str):
                continue
            try:
                parsed = json.loads(stream.strip())
            except ValueError:
                continue
            if isinstance(parsed, dict):
                candidates.append(parsed.get("error"))
                if parsed.get("error_class") or parsed.get("code"):
                    candidates.append(parsed)
    value = next((item for item in candidates if isinstance(item, dict)), None)
    if value is None:
        return None
    error_class = value.get("error_class")
    reason = value.get("reason") or value.get("code")
    if error_class in {"candidate", "compilation", "correctness"}:
        category = "candidate"
    elif error_class == "infra" or str(reason) in {"oss_unavailable", "gateway_unavailable"}:
        category = "infrastructure"
    elif error_class in {"validation", "policy"} or str(reason).startswith("source_"):
        category = "policy"
    else:
        category = "protocol" if error_class or reason else "unknown"
    return {"category": category, "error_class": error_class, "reason": reason,
            "message": value.get("message") or value.get("detail")}


def add_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="emit structured JSON")
    parser.add_argument("--compact", action="store_true", help="emit compact JSON (implies --json)")


def emit(value: dict[str, Any], *, as_json: bool, compact: bool, text: str) -> None:
    if as_json or compact:
        print(
            json.dumps(
                jsonable(value),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":") if compact else None,
                indent=None if compact else 2,
            )
        )
    else:
        print(text)


def table(headers: list[str], rows: list[list[object]]) -> str:
    rendered = [["—" if value is None else str(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in rendered:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    line = "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    divider = "  ".join("-" * widths[index] for index in range(len(headers)))
    body = ["  ".join(value.ljust(widths[index]) for index, value in enumerate(row)) for row in rendered]
    return "\n".join([line, divider, *body])


def main_error(name: str, function, argv: list[str] | None = None) -> int:
    try:
        return function(argv)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        print(f"{name}: {error}", file=sys.stderr)
        return 1
