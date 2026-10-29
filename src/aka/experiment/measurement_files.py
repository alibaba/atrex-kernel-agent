"""Experiment-owned episode mirrors and retrieved measurement publication."""
import hashlib
import json
import os
from datetime import datetime, timezone
from aka.contracts.evaluation import EpisodeEvaluationContent
from aka.contracts.experiment import EpisodeEvidenceScope
EPISODE_EVALUATIONS_PATH = ".atrex_long_horizon/evaluations.jsonl"

def record_episode_evaluation(scope: EpisodeEvidenceScope | None, content: EpisodeEvaluationContent) -> None:
    """Append host-selected episode evidence after its final outputs are published."""
    if scope is None:
        return
    workspace = scope.workspace
    try:
        kernel = scope.candidate_workspace.read_working(workspace, ("kernel.py",)).files["kernel.py"]
        kernel_sha256 = hashlib.sha256(kernel).hexdigest() if kernel is not None else None
    except OSError:
        kernel_sha256 = None
    manifest = scope.candidate_workspace.read_working(workspace, ("solution.json",)).files["solution.json"]
    solution_sha256 = hashlib.sha256(manifest).hexdigest() if manifest is not None else None
    result = json.loads(content.result_json)
    payload = {
        "schema_version": 3,
        "solution_sha256": solution_sha256,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gateway_kind": content.operation,
        "job_id": content.job_id,
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
