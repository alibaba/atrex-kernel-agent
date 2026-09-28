"""Private promotion evidence, bound to its Git commit for crash recovery."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from orchestrator.durable_state import durable_write_text
from orchestrator.session_tail import read_regular_bytes

AUDIT_TRAILER = "AKA-Promotion-Audit: "


class PromotionAuditUnverifiable(RuntimeError):
    """The Git promotion exists, but its supporting audit cannot be verified."""

    def __init__(self, reason: str, message: str):
        super().__init__(f"Cannot recover private promotion audit: {message}")
        self.reason = reason


def promotion_audit_path(workspace: Path, episode: int) -> Path:
    from orchestrator.supervisor_runtime import supervisor_campaign_root

    if isinstance(episode, bool) or not isinstance(episode, int) or episode < 1:
        raise ValueError("promotion audit requires a positive Episode number")
    return supervisor_campaign_root(workspace) / "promotions" / f"long_horizon_e{episode:04d}.json"


def promotion_git(workspace: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=workspace, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Cannot read promotion Git evidence: {result.stderr.strip()}")
    return result.stdout


def write_promotion_audit(workspace: Path, episode: int, evidence: dict[str, Any]) -> str:
    """Write before Git commit; file existence alone never proves promotion."""
    text = json.dumps(evidence, indent=2, ensure_ascii=False) + "\n"
    if len(text.encode()) > 16 * 1024 * 1024:
        raise ValueError("Promotion audit exceeds 16 MiB; no promotion was committed")
    path = promotion_audit_path(workspace, episode)
    if path.is_symlink() or path.parent.is_symlink():
        raise RuntimeError("Promotion audit path cannot be a symlink")
    durable_write_text(path, text)
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def promotion_binding(
    workspace: Path,
    *,
    episode: int,
    base_commit: str,
    branch: str,
    version: int,
) -> dict[str, Any] | None:
    """Bind recovery to the current promotion commit and its Episode checkpoint."""
    message = promotion_git(workspace, "log", "-1", "--format=%B")
    if message.splitlines()[0] != f"episode {episode}: promote verified long-horizon candidate":
        return None
    if promotion_git(workspace, "rev-parse", "HEAD^").strip() != base_commit:
        return None
    return {
        "episode": episode,
        "version": version,
        "base_commit": base_commit,
        "episode_branch": branch,
        "promotion_commit": promotion_git(workspace, "rev-parse", "HEAD").strip(),
        "audit_digests": [
            line.removeprefix(AUDIT_TRAILER)
            for line in message.splitlines()
            if line.startswith(AUDIT_TRAILER)
        ],
    }


def validate_audit_record(record: object, binding: dict[str, Any]) -> dict[str, Any]:
    expected = {key: binding[key] for key in ("episode", "version", "base_commit", "episode_branch")}
    expected["accepted"] = True
    if not isinstance(record, dict) or any(
        record.get(key) != value for key, value in expected.items()
    ):
        raise PromotionAuditUnverifiable("identity_mismatch", "audit does not match the recovering Episode")
    return record


def validate_private_audit(raw: bytes, binding: dict[str, Any]) -> dict[str, Any]:
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    if binding["audit_digests"] != [digest]:
        raise PromotionAuditUnverifiable("digest_mismatch", "audit digest does not match the promotion commit")
    try:
        record = json.loads(raw)
    except ValueError as error:
        raise PromotionAuditUnverifiable("invalid_json", str(error)) from error
    return validate_audit_record(record, binding)


def committed_promotion_audit(
    workspace: Path,
    *,
    episode: int,
    base_commit: str,
    branch: str,
    version: int,
) -> dict[str, Any] | None:
    """Recover only an exact committed promotion, never an uncommitted workspace file."""
    binding = promotion_binding(
        workspace, episode=episode, base_commit=base_commit, branch=branch, version=version,
    )
    if binding is None:
        return None
    digests = binding["audit_digests"]
    path = promotion_audit_path(workspace, episode)
    if digests:
        if len(digests) != 1 or re.fullmatch(r"sha256:[0-9a-f]{64}", digests[0]) is None:
            raise PromotionAuditUnverifiable("invalid_trailer", "invalid promotion audit digest trailer")
        if path.is_symlink() or path.parent.is_symlink():
            raise PromotionAuditUnverifiable("unsafe_path", "audit path is a symlink")
        try:
            raw = read_regular_bytes(path, limit=16 * 1024 * 1024 + 1)
        except FileNotFoundError as error:
            raise PromotionAuditUnverifiable("missing", str(error)) from error
        except OSError as error:
            raise PromotionAuditUnverifiable("unreadable", str(error)) from error
        return validate_private_audit(raw, binding)
    else:
        # Older commits sealed the audit inside memory/. Read Git, not the mutable checkout.
        try:
            raw_text = promotion_git(workspace, "show", f"HEAD:memory/long_horizon_e{episode:04d}.json")
            record = json.loads(raw_text)
        except (ValueError, RuntimeError) as error:
            raise PromotionAuditUnverifiable("legacy_unavailable", str(error)) from error
        record = validate_audit_record(record, binding)
        write_promotion_audit(workspace, episode, record)
    return record
