"""One-shot crash capture for the framework-native V1 stage.

V1 does not continuously journal normal work. When a coding-agent process exits
unexpectedly, the supervisor snapshots its current candidate and records enough terminal/Git
context for a fresh agent to resume from the worktree instead of starting over.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "atrex_framework_baseline_crash_v1"
RECOVERY_SCHEMA_VERSION = "atrex_framework_baseline_recovery_v1"
REJECTION_SCHEMA_VERSION = "atrex_framework_baseline_rejection_v1"
RUNTIME_RELATIVE_DIR = Path(".atrex_long_horizon/framework_baseline")
PROGRESS_RELATIVE_PATH = RUNTIME_RELATIVE_DIR / "progress.json"
LATEST_REJECTION_RELATIVE_PATH = RUNTIME_RELATIVE_DIR / "latest-rejection.json"
_MAX_CAPTURE_BYTES = 16 * 1024 * 1024
_MAX_TERMINAL_CHARS = 32_000
_MAX_PUBLIC_MESSAGE_CHARS = 4_000


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def progress_path(workspace: Path) -> Path:
    return workspace.resolve() / PROGRESS_RELATIVE_PATH


def load_progress(workspace: Path) -> dict[str, Any]:
    path = progress_path(workspace)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read V1 crash record {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"V1 crash record is not a JSON object: {path}")
    return value


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"
    _atomic_write_bytes(path, payload)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _capture_paths(workspace: Path) -> Iterable[Path]:
    for relative in ("kernel.py", "solution.json", "memory/v1.json"):
        yield workspace / relative
    yield from sorted((workspace / "plans").glob("v1_*.md"))
    for pattern in ("debug*.py", "probe*.py", "smoke*.py"):
        yield from sorted(workspace.glob(pattern))


def _public_diagnostic(value: object) -> dict[str, Any]:
    """Project evaluator details without private inputs, paths, or raw streams."""
    if not isinstance(value, dict):
        return {}

    def text(item: object) -> str:
        return str(item or "")[:_MAX_PUBLIC_MESSAGE_CHARS]

    result: dict[str, Any] = {}
    if isinstance(value.get("gate"), str):
        result["gate"] = text(value["gate"])
    if isinstance(value.get("message"), str):
        result["message"] = text(value["message"])
    failure = value.get("failure")
    if isinstance(failure, dict):
        public_failure = {
            key: failure[key]
            for key in ("category", "code", "candidate_repairable")
            if isinstance(failure.get(key), (str, bool))
        }
        if isinstance(failure.get("message"), str):
            public_failure["message"] = text(failure["message"])
        if public_failure:
            result["failure"] = public_failure
    public_runs = []
    for run in value.get("runs", []) if isinstance(value.get("runs"), list) else []:
        if not isinstance(run, dict):
            continue
        public_run = {
            key: run[key]
            for key in ("exit_code", "all_pass", "evaluation_id")
            if isinstance(run.get(key), (str, int, bool))
        }
        diagnostics = []
        for diagnostic in (
            run.get("actionable_diagnostics", [])
            if isinstance(run.get("actionable_diagnostics"), list)
            else []
        )[:8]:
            if not isinstance(diagnostic, dict):
                continue
            item = {}
            if isinstance(diagnostic.get("stage"), str):
                item["stage"] = text(diagnostic["stage"])
            if isinstance(diagnostic.get("message"), str):
                item["message"] = text(diagnostic["message"])
            if item:
                diagnostics.append(item)
        if diagnostics:
            public_run["actionable_diagnostics"] = diagnostics
        if public_run:
            public_runs.append(public_run)
    if public_runs:
        result["runs"] = public_runs[:4]
    return result


def rejection_path(workspace: Path) -> Path:
    return workspace.resolve() / LATEST_REJECTION_RELATIVE_PATH


def capture_terminal_rejection(
    workspace: Path,
    *,
    version: int,
    status: str,
    reason: str,
    diagnostic: object = None,
) -> dict[str, Any]:
    """Persist the exact public candidate whenever the supervisor rejects V1."""
    if type(version) is not int or version < 1:
        raise ValueError("framework-baseline rejection requires a positive version")
    if status not in {"rejected", "validation_blocked"}:
        raise ValueError("unsupported framework-baseline rejection status")
    if not reason or len(reason) > 64 or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in reason
    ):
        raise ValueError("invalid framework-baseline rejection reason")

    workspace = workspace.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = workspace / RUNTIME_RELATIVE_DIR / "rejections" / f"v{version}-{stamp}"
    files: dict[str, dict[str, object]] = {}
    for filename in ("kernel.py", "solution.json"):
        source = workspace / filename
        if not source.is_file() or source.is_symlink():
            continue
        data = source.read_bytes()
        if len(data) > _MAX_CAPTURE_BYTES:
            continue
        _atomic_write_bytes(root / filename, data)
        files[filename] = {
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    record = {
        "schema_version": REJECTION_SCHEMA_VERSION,
        "version": version,
        "status": status,
        "reason": reason,
        "captured_at": _utc_now(),
        "snapshot_root": str(root.relative_to(workspace)),
        "files": files,
        "diagnostic": _public_diagnostic(diagnostic),
    }
    _atomic_write_json(root / "diagnostic.json", record)
    _atomic_write_json(rejection_path(workspace), record)
    return record


def _snapshot_after_exit(
    workspace: Path, *, exit_number: int, agent_cli: str
) -> dict[str, Any]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = (
        workspace / RUNTIME_RELATIVE_DIR / f"exit-{exit_number:02d}-{agent_cli}-{stamp}"
    )
    files: list[str] = []
    hashes: dict[str, str] = {}
    seen: set[Path] = set()
    for source in _capture_paths(workspace):
        try:
            resolved = source.resolve()
            relative = resolved.relative_to(workspace)
        except (OSError, ValueError):
            continue
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        try:
            if resolved.stat().st_size > _MAX_CAPTURE_BYTES:
                continue
            data = resolved.read_bytes()
        except OSError:
            continue
        destination = root / relative
        _atomic_write_bytes(destination, data)
        files.append(str(destination.relative_to(workspace)))
        hashes[str(relative)] = hashlib.sha256(data).hexdigest()
    return {
        "root": str(root.relative_to(workspace)),
        "files": files,
        "sha256": hashes,
    }


def _git_text(workspace: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(workspace),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def capture_unexpected_exit(
    workspace: Path,
    *,
    agent_cli: str,
    exit_status: int,
    timed_out: bool,
    tokens: int,
    session_id: str,
    stdout_tail: str,
    stderr_tail: str,
    error: str = "",
    supervisor_order: Iterable[str] = (),
) -> dict[str, Any]:
    """Persist one terminal failure and candidate snapshot after an unexpected exit."""
    workspace = workspace.resolve()
    try:
        value = load_progress(workspace)
    except RuntimeError as exc:
        path = progress_path(workspace)
        if path.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            shutil.copy2(path, path.with_name(f"progress.corrupt-{stamp}.json"))
        value = {"prior_record_error": str(exc)}
    exits = value.get("unexpected_exits")
    if not isinstance(exits, list):
        exits = []
        value["unexpected_exits"] = exits
    exit_number = len(exits) + 1
    snapshot = _snapshot_after_exit(
        workspace, exit_number=exit_number, agent_cli=agent_cli
    )
    record = {
        "number": exit_number,
        "captured_at": _utc_now(),
        "agent_cli": agent_cli,
        "exit_status": exit_status,
        "timed_out": timed_out,
        "tokens": tokens,
        "session_id": session_id,
        "error": error,
        "stdout_tail": stdout_tail[-_MAX_TERMINAL_CHARS:],
        "stderr_tail": stderr_tail[-_MAX_TERMINAL_CHARS:],
        "git_head": _git_text(workspace, "rev-parse", "HEAD"),
        "git_status": _git_text(
            workspace, "status", "--short", "--untracked-files=all"
        ),
        "snapshot": snapshot,
    }
    exits.append(record)
    value.update(
        {
            "schema_version": SCHEMA_VERSION,
            "version": "v1",
            "workspace": str(workspace),
            "status": "interrupted",
            "last_agent_cli": agent_cli,
            "progress_supervisor_order": list(supervisor_order),
            "latest_snapshot": snapshot,
            "updated_at": _utc_now(),
        }
    )
    _atomic_write_json(progress_path(workspace), value)
    _atomic_write_json(
        workspace / RUNTIME_RELATIVE_DIR / f"exit-{exit_number:02d}.json", record
    )
    return value


def mark_accepted(workspace: Path, *, commit: str, latency_us: object) -> None:
    """Close a pre-existing crash record after a fallback agent succeeds."""
    value = load_progress(workspace)
    if not value:
        return
    value.update(
        {
            "status": "accepted",
            "accepted_commit": commit,
            "accepted_latency_us": latency_us,
            "completed_at": _utc_now(),
            "updated_at": _utc_now(),
        }
    )
    _atomic_write_json(progress_path(workspace), value)


def save_supervisor_recovery(
    workspace: Path, payload: object, *, agent_cli: str
) -> Path:
    """Validate and persist the exit-supervisor's concise V1 handoff."""
    if not isinstance(payload, dict):
        raise ValueError("V1 recovery handoff must be a JSON object")
    required_strings = ("summary", "current_state", "known_failure", "next_step")
    for key in required_strings:
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise ValueError(f"V1 recovery handoff requires non-empty {key!r}")
    for key in ("completed_work", "files_to_resume"):
        value = payload.get(key)
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item.strip() for item in value
        ):
            raise ValueError(f"V1 recovery handoff {key!r} must be a string list")
    experiments = payload.get("experiments")
    if not isinstance(experiments, list) or not all(
        isinstance(item, dict) for item in experiments
    ):
        raise ValueError("V1 recovery handoff 'experiments' must be an object list")

    recovery = dict(payload)
    recovery.update(
        {
            "schema_version": RECOVERY_SCHEMA_VERSION,
            "supervisor_agent_cli": agent_cli,
            "recorded_at": _utc_now(),
        }
    )
    destination = workspace.resolve() / RUNTIME_RELATIVE_DIR / "resume.json"
    _atomic_write_json(destination, recovery)
    progress = load_progress(workspace)
    progress.update(
        {
            "recovery_status": "ready",
            "recovery_agent_cli": agent_cli,
            "recovery_record": str(destination.relative_to(workspace.resolve())),
            "updated_at": _utc_now(),
        }
    )
    _atomic_write_json(progress_path(workspace), progress)
    return destination


def restore_latest_candidate(workspace: Path) -> list[str]:
    """Restore only kernel.py/solution.json from the newest crash snapshot."""
    workspace = workspace.resolve()
    value = load_progress(workspace)
    snapshot = value.get("latest_snapshot")
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("files"), list):
        return []
    restored: list[str] = []
    for relative_snapshot in snapshot["files"]:
        if not isinstance(relative_snapshot, str):
            continue
        source = (workspace / relative_snapshot).resolve()
        try:
            source.relative_to(workspace)
        except ValueError:
            continue
        if not source.is_file():
            continue
        destination_name = source.name
        if destination_name not in {"kernel.py", "solution.json"}:
            continue
        destination = workspace / destination_name
        _atomic_write_bytes(destination, source.read_bytes())
        restored.append(destination_name)
    return restored


def candidate_sha256(workspace: Path) -> str:
    path = workspace / "kernel.py"
    return _sha256(path) if path.is_file() else ""
