#!/usr/bin/env python3
"""Shared, read-only helpers for AKA Light observability commands."""
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


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for entry in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from aka.workflow.legacy.export import RetainedWorkflowExport  # noqa: E402
from aka.task.candidate_workspace.provider import GitCandidateWorkspace
from aka.evaluation.tool_results import result_from_response  # noqa: E402
from observability.measurement_reader import (  # noqa: E402
    MeasurementStore,
)
from aka.workflow.light.support.paths import supervisor_campaign_root  # noqa: E402
from aka.workflow.light.search import read_state as read_search_state  # noqa: E402


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
        return MeasurementStore(self.workspace).root if MeasurementStore.available(self.workspace) else supervisor_campaign_root(self.workspace) / "measurements"

    @property
    def audit_root(self) -> Path:
        return supervisor_campaign_root(self.workspace)


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
    campaign_root = supervisor_campaign_root(workspace)
    scopes = [Scope(workspace, campaign_root, "campaign", None)]
    trajectories = campaign_root / "search" / "trajectories"
    if trajectories.is_dir():
        for path in sorted(p for p in trajectories.glob("t*") if p.is_dir()):
            match = re.fullmatch(r"t0*([1-9][0-9]*)", path.name)
            number = int(match.group(1)) if match else None
            scopes.append(Scope(path.resolve(), supervisor_campaign_root(path), path.name, number))
    return scopes


def load_snapshot(scope: Scope):
    return RetainedWorkflowExport(candidate_workspace=GitCandidateWorkspace()).read(scope.workspace)


def existing_store(scope: Scope) -> MeasurementStore | None:
    return MeasurementStore(scope.workspace) if MeasurementStore.available(scope.workspace) else None


def iter_record_directories(scope: Scope) -> Iterable[Path]:
    store = existing_store(scope)
    if store is not None:
        yield from (store.record_directory(record_id) for record_id in store.record_ids())


def iter_records(scope: Scope, *, include_incomplete: bool = False) -> Iterable[dict[str, Any]]:
    store = existing_store(scope)
    for directory in iter_record_directories(scope):
        request = read_json(directory / "request.json", default={})
        if not isinstance(request, dict):
            request = {}
        row: dict[str, Any] = {
            "scope": scope.label,
            "trajectory": scope.trajectory,
            "workspace": str(scope.workspace),
            "gateway_record_id": directory.name if directory.name.startswith("gateway-") else "gateway-" + directory.name,
            "directory": directory,
            "request": request,
            "complete": False,
        }
        if store is not None and (directory / "record.json").is_file():
            try:
                row["execution_record"] = store.execution_record(row["gateway_record_id"])
                record = store.read(row["gateway_record_id"])
            except (OSError, ValueError, KeyError, TypeError, RuntimeError, RecursionError) as error:
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
    path = supervisor_campaign_root(workspace) / "search" / "state.json"
    if not path.is_file():
        return None
    value = read_search_state(path)
    return value if isinstance(value, dict) else None


def startup_stage(workspace: Path) -> str:
    """Return the durable pre-Episode stage for session-trace attribution."""
    live = read_json(workspace / "memory/live.json", default={})
    if isinstance(live, dict):
        if live.get("episode_branch") == "framework-baseline":
            return "framework_baseline"
        phase = live.get("phase")
        if isinstance(phase, str) and phase:
            return phase
    if (workspace / ".atrex_long_horizon/framework_baseline").exists():
        return "framework_baseline"
    return "bootstrap"


def attempt_documents(scope: Scope) -> list[dict[str, Any]]:
    runtime = scope.workspace / ".atrex_long_horizon"
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
    runtime = scope.workspace / ".atrex_long_horizon" / "episodes"
    rows = []
    for path in sorted(runtime.glob("e*/telemetry.summary.json")):
        value = read_json(path, default={})
        if isinstance(value, dict):
            rows.append(value)
    return rows


def job_state_rows(record_directory: Path, *, record=None) -> list[dict[str, Any]]:
    rows = []
    envelope = read_json(record_directory / "record.json", default={})
    if isinstance(envelope, dict) and envelope.get("format") == "aka-experiment-files/1":
        if record is None:
            from aka.contracts.experiment import ExperimentRef, ExperimentRecordError
            from aka.experiment.file_records import FileExperimentRecords
            try:
                ref = ExperimentRef(**envelope["record"]["ref"])
                reader = FileExperimentRecords(record_directory.parent.parent, ref.run_id, create=False)
                if reader.root != record_directory.parent:
                    raise ValueError('record namespace mismatch')
                record = reader.get(ref).as_dict()
            except (LookupError, ExperimentRecordError, ValueError, TypeError, KeyError) as error:
                print(f"measurement-reader: {record_directory.name}: {getattr(error, 'code', 'record_corrupt')}; evidence unavailable", file=sys.stderr)
                return []
        for attempt in record.get("attempts", []):
            result = attempt.get("result") or {}
            for reference in result.get("job_references", []):
                facts = result.get("facts", {})
                job = {"job_id": reference.get("identifier"), "status": result.get("transport_status", "unknown"),
                       "execution_facts": result.get("facts", {}), "budgets": result.get("budgets", []),
                       "causes": result.get("causes", []), "source": "recorded_execution_facts"}
                rows.append({"path": str(record_directory / "record.json"), "checkpoint_id": attempt.get("attempt_id"),
                             "checkpoint_updated_at": record.get("updated_at"),
                             "phase": "terminal" if facts.get("disposition") in {"terminal", "not_submitted"} else record.get("phase"),
                             "submission": {"job_id": job["job_id"]}, "retries": {}, "identity": reference,
                             "process": {}, "job": job})
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
                "process": process,
                "job": job,
            }
        )
    return rows


def job_identifier(job: object) -> str | None:
    if not isinstance(job, dict):
        return None
    for key in ("job_id", "id", "task_id"):
        value = job.get(key)
        if isinstance(value, str) and value:
            return value
    return None


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
