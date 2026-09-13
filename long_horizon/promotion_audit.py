"""Private promotion evidence, bound to its Git commit for crash recovery."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from orchestrator.durable_state import durable_write_text

AUDIT_FILENAME_RE = re.compile(r"long_horizon_e([0-9]+)\.json")
AUDIT_TRAILER = "AKA-Promotion-Audit: "


def promotion_audit_path(workspace: Path, episode: int) -> Path:
    from orchestrator.supervisor_runtime import supervisor_campaign_root

    if isinstance(episode, bool) or not isinstance(episode, int) or episode < 1:
        raise ValueError("promotion audit requires a positive Episode number")
    return supervisor_campaign_root(workspace) / "promotions" / f"long_horizon_e{episode:04d}.json"


def _git(workspace: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=workspace, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Cannot read promotion Git evidence: {result.stderr.strip()}")
    return result.stdout


def write_promotion_audit(workspace: Path, episode: int, evidence: dict[str, Any]) -> str:
    """Write before Git commit; file existence alone never proves promotion."""
    text = json.dumps(evidence, indent=2, ensure_ascii=False) + "\n"
    path = promotion_audit_path(workspace, episode)
    if path.is_symlink() or path.parent.is_symlink():
        raise RuntimeError("Promotion audit path cannot be a symlink")
    durable_write_text(path, text)
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def committed_promotion_audit(
    workspace: Path,
    *,
    episode: int,
    base_commit: str,
    branch: str,
    version: int,
) -> dict[str, Any] | None:
    """Recover only an exact committed promotion, never an uncommitted workspace file."""
    message = _git(workspace, "log", "-1", "--format=%B")
    if message.splitlines()[0] != f"episode {episode}: promote verified long-horizon candidate":
        return None
    if _git(workspace, "rev-parse", "HEAD^").strip() != base_commit:
        return None
    digests = [
        line.removeprefix(AUDIT_TRAILER)
        for line in message.splitlines()
        if line.startswith(AUDIT_TRAILER)
    ]
    path = promotion_audit_path(workspace, episode)
    if digests:
        try:
            if path.is_symlink() or path.parent.is_symlink():
                raise ValueError("audit path is a symlink")
            raw = path.read_bytes()
            digest = "sha256:" + hashlib.sha256(raw).hexdigest()
            if digests != [digest]:
                raise ValueError("audit digest does not match the promotion commit")
            record = json.loads(raw)
        except (OSError, ValueError) as error:
            raise RuntimeError(f"Cannot recover private promotion audit: {error}") from error
    else:
        # Older commits sealed the audit inside memory/. Read Git, not the mutable checkout.
        raw_text = _git(workspace, "show", f"HEAD:memory/long_horizon_e{episode:04d}.json")
        try:
            record = json.loads(raw_text)
        except ValueError as error:
            raise RuntimeError("Invalid committed legacy promotion audit") from error
    expected = {
        "episode": episode,
        "version": version,
        "base_commit": base_commit,
        "episode_branch": branch,
        "accepted": True,
    }
    if not isinstance(record, dict) or any(
        record.get(key) != value for key, value in expected.items()
    ):
        raise RuntimeError("Promotion audit does not match the recovering Episode")
    if not digests:
        write_promotion_audit(workspace, episode, record)
    return record


def preserve_legacy_promotion_audits(workspace: Path) -> None:
    """Copy committed legacy snapshots into private storage without rewriting history."""
    if not (workspace / ".git").exists():
        return
    head = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"], cwd=workspace, capture_output=True, text=True
    )
    if head.returncode:
        return  # Initial workspace, before V0 is committed.
    for relative in _git(
        workspace, "ls-tree", "-r", "--name-only", "HEAD", "--", "memory/"
    ).splitlines():
        entry = Path(relative)
        match = AUDIT_FILENAME_RE.fullmatch(entry.name) if entry.parent == Path("memory") else None
        if match is None:
            continue
        episode = int(match[1])
        path = promotion_audit_path(workspace, episode)
        if path.exists():
            continue
        raw = _git(workspace, "show", f"HEAD:{relative}")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise RuntimeError(f"Invalid committed legacy audit: {relative}")
        write_promotion_audit(workspace, episode, value)
