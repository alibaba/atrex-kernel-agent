"""Frozen, read-only Episode evidence; never mount the controller's archives.

Only terminal Episodes before the current one are eligible. Conversations come
from finalized Session capture; neither Agent workspace files nor canonical
memory are substitutes for that source. Episode Reports remain controller-private.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from pathlib import Path

from aka.light.orchestrator.agent_home import open_private_directory
from aka.light.orchestrator.session_tail import read_regular_bytes
from aka.light.supervisor.workspace import publish

MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_CONVERSATION_BYTES = 256 * 1024 * 1024
MAX_VIEW_BYTES = 512 * 1024 * 1024
MAX_DIRECTORIES = 4096
LOGGER = logging.getLogger(__name__)


def _json(path: Path) -> dict:
    raw = read_regular_bytes(path, limit=MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError("History metadata exceeds its size limit")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("History metadata must be an object")
    return value


def _directories(root: Path, pattern: str) -> list[Path]:
    if not root.exists():
        return []
    # Pin every component; do not follow an archive/session symlink.
    with open_private_directory(root) as directory:
        values = []
        with os.scandir(directory) as entries:
            for count, entry in enumerate(entries, 1):
                if count > MAX_DIRECTORIES:
                    raise ValueError("History directory count exceeds its limit")
                if re.fullmatch(pattern, entry.name) and entry.is_dir(follow_symlinks=False):
                    values.append(root / entry.name)
    return values


class EpisodeHistory:
    def __init__(self, *, episode: int, episodes_root: Path):
        if type(episode) is not int or episode < 1:
            raise ValueError("History requires a controller Episode number")
        self.episode = episode
        self.episodes_root = episodes_root

    def _conversation(self, directory: Path, destination: Path, remaining: int) -> dict:
        candidates = []
        for session in _directories(directory / "sessions", r"run-[0-9a-f]{32}"):
            try:
                usage = _json(session / "token-usage.json")
                finished = usage.get("finished_at")
                if isinstance(finished, str) and finished and usage.get("state") in {
                    "completed", "failed", "timed_out", "interrupted"
                }:
                    candidates.append((finished, session.name, session, usage))
            except (OSError, ValueError, RecursionError):
                # An unfinalized capture is never treated as the latest sealed one.
                continue
        if not candidates:
            return {"available": False, "reason": "no_finalized_capture"}
        _, _, session, usage = max(candidates, key=lambda row: row[:2])
        limit = min(MAX_CONVERSATION_BYTES, remaining)
        raw = read_regular_bytes(session / "conversation.jsonl", limit=limit + 1)
        if len(raw) > limit:
            return {"available": False, "reason": "history_size_limit"}
        # Capture's final atomic projection always has these two envelope records.
        # Do not publish a partial/live prefix, or silently fall back to an older run.
        first = raw.find(b"\n")
        end = len(raw)
        while end and raw[end - 1] in b" \r\n\t":
            end -= 1
        last = raw.rfind(b"\n", 0, end) + 1
        if first < 0 or first > MAX_JSON_BYTES or end - last > MAX_JSON_BYTES:
            raise ValueError("History conversation has invalid envelope records")
        head = json.loads(raw[:first])
        tail = json.loads(raw[last:end])
        if (not isinstance(head, dict) or head.get("type") != "session_start"
                or not isinstance(tail, dict) or tail.get("type") != "session_end"):
            raise ValueError("History conversation is not a finalized reading projection")
        publish(destination, "conversation.jsonl", raw)
        return {"available": True, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                "state": usage["state"], "finished_at": usage["finished_at"],
                "capture_complete": usage.get("capture_complete") is True}

    def materialize(self, destination: Path) -> None:
        """Build a Session snapshot; gaps are explicit and never block optimization."""
        index = {"schema_version": 1, "before_episode": self.episode, "episodes": [], "warnings": []}
        try:
            directories = _directories(self.episodes_root, r"e[0-9]{4,}")
        except (OSError, ValueError):
            LOGGER.warning("Historical Episode archive unavailable", exc_info=True)
            directories = []
            index["warnings"].append("episode_archive_unavailable")
        total = 0
        visible = []
        for directory in sorted(directories, key=lambda path: int(path.name[1:])):
            number = int(directory.name[1:])
            if number >= self.episode:
                continue
            try:
                terminal = _json(directory / "attempt.json")
                if terminal.get("episode") != number:
                    raise ValueError("Terminal Episode identity differs")
            except FileNotFoundError:
                continue  # Running/abandoned Episode, not a terminal archive.
            except (OSError, ValueError, RecursionError):
                LOGGER.warning("Historical Episode %s terminal record unavailable", number, exc_info=True)
                index["warnings"].append(f"episode_{number}_terminal_unavailable")
                continue
            relative = f"episodes/{number:08d}"
            row = {"episode": number, "status": terminal.get("status"),
                   "accepted": terminal.get("accepted") is True}
            visible.append((directory, relative, row))
        # If the bounded view fills, prefer recent conversations.
        for directory, relative, row in reversed(visible):
            try:
                result = self._conversation(directory, destination / relative, MAX_VIEW_BYTES - total)
                if result["available"]:
                    total += result["bytes"]
                    result["file"] = relative + "/conversation.jsonl"
            except (OSError, ValueError, RecursionError):
                LOGGER.warning("Historical Episode %s conversation unavailable", row["episode"], exc_info=True)
                result = {"available": False, "reason": "capture_unreadable_or_unsealed"}
            row["conversation"] = result
        index["episodes"] = [row for _, _, row in visible]
        publish(destination, "index.json", (json.dumps(index, ensure_ascii=False, indent=2) + "\n").encode())


def refresh_history(workspace: Path, private_directory: Path, history: EpisodeHistory | None) -> None:
    """Replace a derived view between Sessions, not its authoritative sources.

    The input directory is a read-only mount in Bubblewrap. Native mode remains
    cooperative; rebuild the view so native edits never become later evidence.
    """
    with tempfile.TemporaryDirectory(prefix="history-view-", dir=private_directory) as temporary:
        staging = Path(temporary)
        evidence = staging / "evidence"
        evidence.mkdir(mode=0o700)
        if history is not None:
            history.materialize(evidence)
        else:
            publish(evidence, "index.json", b'{"episodes":[],"warnings":["history_not_configured"]}\n')
        for parent, _, names in os.walk(evidence):
            for name in names:
                (Path(parent) / name).chmod(0o400)
        with open_private_directory(workspace / "input") as target, open_private_directory(staging) as source:
            # Pinned parents + rename do not follow an Agent-controlled destination.
            try:
                os.rename("evidence", "previous", src_dir_fd=target, dst_dir_fd=source)
            except FileNotFoundError:
                pass
            try:
                os.rename("evidence", "evidence", src_dir_fd=source, dst_dir_fd=target)
            except BaseException:
                if (staging / "previous").exists() or (staging / "previous").is_symlink():
                    os.rename("previous", "evidence", src_dir_fd=source, dst_dir_fd=target)
                raise
