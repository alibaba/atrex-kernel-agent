"""Experiment-owned episode mirrors and retrieved measurement publication."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from .file_records import _atomic
EPISODE_EVALUATIONS_PATH = ".atrex_long_horizon/evaluations.jsonl"

def record_episode_evaluation(
    workspace: Path,
    result: dict[str, Any],
    *,
    gateway_kind: str,
    job_id: object = None,
) -> None:
    """Persist exact optimizer-facing results for terminal memory construction.

    Long-horizon workers use ``--no-memory`` because canonical memory belongs to the
    supervisor.  Keep their complete evaluator results in excluded episode runtime
    state so a pivot or interruption can still record this round's real per-shape
    performance without parsing model-authored journal prose.
    """
    runtime = workspace / ".atrex_long_horizon"
    if not (runtime / "journal.json").is_file():
        return
    try:
        kernel_sha256 = hashlib.sha256(
            (workspace / "kernel.py").read_bytes()
        ).hexdigest()
    except OSError:
        kernel_sha256 = None
    manifest = workspace / "solution.json"
    solution_sha256 = (
        hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else None
    )
    payload = {
        "schema_version": 3,
        "solution_sha256": solution_sha256,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gateway_kind": gateway_kind,
        "job_id": str(job_id) if job_id else None,
        "kernel_sha256": kernel_sha256,
        "result": result,
    }
    path = workspace / EPISODE_EVALUATIONS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    if path.is_symlink() or path.parent.resolve() != path.parent:
        raise ValueError("episode result path cannot traverse symlinks")
    import fcntl
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        offset = 0
        while offset < len(line):
            offset += os.write(descriptor, line[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def record_result_lines(workspace: Path, stdout: str, *, gateway_kind: str, job_id=None) -> None:
    """Record the last ordinary RESULT_JSON emitted by a dev evaluator command."""
    for line in reversed(stdout.splitlines()):
        if not line.startswith("[test_kernel] RESULT_JSON="):
            continue
        try:
            result = json.loads(line[len("[test_kernel] RESULT_JSON=") :])
        except json.JSONDecodeError:
            return
        if isinstance(result, dict):
            record_episode_evaluation(workspace, result, gateway_kind=gateway_kind, job_id=job_id)
        return

