"""Operator-only recovery of an already committed promotion; never reruns a Gate."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orchestrator.durable_state import durable_write_json, durable_write_text

from .promotion_audit import (
    PromotionAuditUnverifiable,
    _git,
    committed_promotion_audit,
    promotion_audit_path,
    promotion_binding,
    validate_private_audit,
)
from .store import CampaignStore


class PromotionAuditRecoveryRequired(RuntimeError):
    """Pause recovery without discarding either the promotion or active checkpoint."""


def _receipt_path(workspace: Path, episode: int) -> Path:
    return promotion_audit_path(workspace, episode).with_suffix(".recovery.json")


def _validate_committed_files(workspace: Path, version: int) -> None:
    # Acknowledging an audit loss cannot repair a missing Kernel/report or an edited checkout.
    for relative in ("kernel.py", f"memory/v{version}.json"):
        if _git(workspace, "hash-object", "--", relative).strip() != _git(
            workspace, "rev-parse", f"HEAD:{relative}",
        ).strip():
            raise ValueError(f"{relative} differs from the committed promotion")
    memory = json.loads(_git(workspace, "show", f"HEAD:memory/v{version}.json"))
    if not isinstance(memory, dict) or memory.get("version") != f"v{version}":
        raise ValueError("Committed canonical memory has an invalid version")


def recover_promotion_audit(workspace: Path, **identity: Any) -> dict[str, Any] | None:
    try:
        record = committed_promotion_audit(workspace, **identity)
    except PromotionAuditUnverifiable as error:
        # A missing file can be acknowledged, but corruption/identity failures cannot.
        if error.reason != "missing":
            raise
        binding = promotion_binding(workspace, **identity)
        if binding is None:
            raise
        path = _receipt_path(workspace, identity["episode"])
        if path.is_symlink() or path.parent.is_symlink():
            raise PromotionAuditUnverifiable("unsafe_path", "recovery receipt path is a symlink") from error
        try:
            receipt = json.loads(path.read_bytes())
        except FileNotFoundError:
            raise error
        except (OSError, ValueError) as failure:
            raise PromotionAuditUnverifiable("invalid_recovery", str(failure)) from failure
        if (
            not isinstance(receipt, dict)
            or receipt.get("binding") != binding
            or receipt.get("status") != "audit_unverifiable"
            or receipt.get("resolution") != "operator_acknowledged_missing"
            or not isinstance(receipt.get("reason"), str)
            or not receipt["reason"].strip()
            or not isinstance(receipt.get("recorded_at"), str)
        ):
            raise PromotionAuditUnverifiable("invalid_recovery", "recovery receipt does not match this promotion") from error
        try:
            _validate_committed_files(workspace, identity["version"])
        except (OSError, ValueError, RuntimeError) as failure:
            raise PromotionAuditUnverifiable("invalid_recovery", str(failure)) from failure
        return receipt
    return {"status": "verified"} if record is not None else None


def pause_for_audit_repair(
    workspace: Path, store: CampaignStore, active: dict[str, Any], error: PromotionAuditUnverifiable,
) -> PromotionAuditRecoveryRequired:
    binding = promotion_binding(
        workspace, episode=int(active["episode"]), version=int(active["memory_version"]),
        base_commit=active["base_commit"], branch=active["episode_branch"],
    )
    active["promotion_audit"] = {
        "status": "audit_unverifiable", "reason": error.reason,
        "error": str(error), "binding": binding,
    }
    # Keep phase/worktree/counters intact. A later resume always verifies again.
    store.save_active(active)
    command = shlex.join([
        "python3", "-m", "long_horizon.audit_recovery", "--workspace", str(workspace),
        "--promotion-commit", _git(workspace, "rev-parse", "HEAD").strip(),
    ])
    message = (
        f"audit_unverifiable: {error}\nGit promotion and active checkpoint are preserved. "
        "Stop this Campaign before repair. From the repository root, restore the exact original audit:\n"
        f"  {command} restore --file /path/to/original-audit.json"
    )
    if error.reason == "missing":
        message += (
            "\nIf no backup exists, explicitly acknowledge the missing evidence (not a verified Gate):\n"
            f"  {command} acknowledge-missing --reason 'Explain why continuing is acceptable'"
        )
    return PromotionAuditRecoveryRequired(message + "\nThen rerun the original Campaign command.")


def repair_promotion_audit(
    workspace: Path, *, promotion_commit: str, source: Path | None = None, reason: str = "",
) -> dict[str, Any]:
    """Restore exact bytes or acknowledge loss, scoped to one stopped Campaign's HEAD."""
    workspace = workspace.resolve()
    if not workspace.is_dir() or not (workspace / ".git").exists():
        raise ValueError("--workspace must be the existing Campaign Git workspace")
    store = CampaignStore(workspace)
    active = store.load_active()
    if not active or active.get("phase") not in {"promoting", "promoted"}:
        raise ValueError("No interrupted promotion checkpoint to repair")
    identity = {
        "episode": int(active["episode"]), "version": int(active["memory_version"]),
        "base_commit": active["base_commit"], "branch": active["episode_branch"],
    }
    binding = promotion_binding(workspace, **identity)
    if binding is None or binding["promotion_commit"] != promotion_commit:
        raise ValueError("--promotion-commit must exactly match the checkpoint's committed promotion at HEAD")
    _validate_committed_files(workspace, identity["version"])
    if source is None and not reason.strip():
        raise ValueError("Acknowledging lost audit evidence requires a non-empty --reason")
    path = promotion_audit_path(workspace, identity["episode"])
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError("Promotion audit path cannot be a symlink")
    if source is not None:
        raw = source.read_bytes()
        validate_private_audit(raw, binding)
        # Preserve whitespace and formatting: the commit binds bytes, not equivalent JSON.
        text = raw.decode("utf-8")
        if _git(workspace, "rev-parse", "HEAD").strip() != promotion_commit:
            raise ValueError("Campaign advanced during audit repair; stop it before retrying")
        durable_write_text(path, text)
        return {"status": "restored", "promotion_commit": promotion_commit, "audit_path": str(path)}
    try:
        committed_promotion_audit(workspace, **identity)
    except PromotionAuditUnverifiable as error:
        if error.reason != "missing":
            raise ValueError("Only missing audits can be acknowledged; restore/investigate corrupt evidence") from error
    else:
        raise ValueError("Promotion audit is already verifiable; resume the Campaign normally")
    receipt_path = _receipt_path(workspace, identity["episode"])
    if receipt_path.is_symlink() or receipt_path.parent.is_symlink():
        raise ValueError("Recovery receipt path cannot be a symlink")
    if receipt_path.exists():
        # Idempotent repair cannot overwrite a conflicting or damaged operator decision.
        receipt = recover_promotion_audit(workspace, **identity)
        assert receipt is not None
        return receipt
    receipt = {
        "status": "audit_unverifiable", "resolution": "operator_acknowledged_missing",
        "binding": binding, "reason": reason.strip(),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    if _git(workspace, "rev-parse", "HEAD").strip() != promotion_commit:
        raise ValueError("Campaign advanced during audit repair; stop it before retrying")
    durable_write_json(receipt_path, receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--promotion-commit", required=True, help="Exact full HEAD commit ID; stop the Campaign first")
    commands = parser.add_subparsers(dest="command", required=True)
    restore = commands.add_parser("restore", allow_abbrev=False)
    restore.add_argument("--file", type=Path, required=True, help="Original audit bytes from backup")
    acknowledge = commands.add_parser("acknowledge-missing", allow_abbrev=False)
    acknowledge.add_argument("--reason", required=True, help="Operator's justification for accepting lost audit evidence")
    args = parser.parse_args(argv)
    try:
        result = repair_promotion_audit(
            args.workspace, promotion_commit=args.promotion_commit,
            source=args.file if args.command == "restore" else None,
            reason=args.reason if args.command == "acknowledge-missing" else "",
        )
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"Audit repair refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("Repair recorded. Resume using the original Campaign command.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
