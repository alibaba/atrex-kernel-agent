"""Agent-writable episode messages, separate from private Supervisor state."""
import json
from pathlib import Path

from aka.core.filesystem import durable_write_text, read_regular_bytes
from .runtime_paths import EXCHANGE_DIR, RUNTIME_DIR

POLICY_REVIEW_REQUEST = Path(EXCHANGE_DIR) / "policy_review_request.json"
LEGACY_FILES = ("journal.json", "handoff.json", "telemetry.jsonl")
FILES = (*LEGACY_FILES, POLICY_REVIEW_REQUEST.name)


def prepare(workspace):
    workspace = Path(workspace).resolve(strict=True)
    directory = workspace / EXCHANGE_DIR
    if directory.is_symlink():
        raise ValueError("Episode exchange must not be a symlink")
    try:
        directory.mkdir(mode=0o700)
        created = True
    except FileExistsError:
        if not directory.is_dir():
            raise ValueError("Episode exchange must be a directory")
        created = False
    # Existing episodes retain their messages; private state is never copied out.
    for name in FILES:
        target = directory / name
        if target.is_symlink():
            raise ValueError("Episode exchange message must not be a symlink")
        source = workspace / RUNTIME_DIR / name
        if created and name in LEGACY_FILES and source.is_file():
            if source.is_symlink() or source.parent.is_symlink():
                raise ValueError("Legacy episode message must not be a symlink")
            durable_write_text(target, source.read_text(encoding="utf-8"))
    return directory


def policy_review_candidate(workspace, *, episode, base_commit):
    """Read one bounded, atomically published request for this episode only."""
    try:
        raw = read_regular_bytes(Path(workspace) / POLICY_REVIEW_REQUEST, limit=4097)
        if len(raw) > 4096:
            return None
        payload = json.loads(raw)
    except (OSError, ValueError, UnicodeError):
        return None
    if not isinstance(payload, dict):
        return None
    candidate = payload.get("candidate_commit")
    if (payload.get("schema_version") != 2
            or type(payload.get("episode")) is not int or payload["episode"] != episode
            or payload.get("base_commit") != base_commit
            or not isinstance(candidate, str) or len(candidate) not in (40, 64)
            or any(char not in "0123456789abcdef" for char in candidate)):
        return None
    return candidate
