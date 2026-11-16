"""Durable, per-invocation token accounting for independent Light reviewers."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import sys
import uuid

from aka.core.filesystem import durable_write_json
from aka.workflow.light.support.paths import supervisor_campaign_root


REVIEW_KINDS = frozenset({
    "final_source_policy_review",
    "direction_claim_review",
    "experiment_claim_review",
    "supplemental_numerical_review",
})
LEGACY_REVIEW_KINDS = frozenset({"research_claim_review"})


def record_review_usage(
    workspace: Path,
    result: object,
    *,
    review_kind: str,
    label: str,
    backend: str,
    model: str = "",
) -> Path:
    """Append one immutable Supervisor-private reviewer invocation receipt."""
    if review_kind not in REVIEW_KINDS | LEGACY_REVIEW_KINDS:
        raise ValueError(f"unsupported review kind: {review_kind!r}")
    usage = getattr(result, "terminal_usage", None)
    payload = {
        "schema": "atrex.light-review-usage.v1",
        "review_kind": review_kind,
        "label": str(label),
        "backend": str(backend),
        "model": str(model) or None,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "session_id": str(getattr(result, "session_id", "") or "") or None,
        "exit_status": getattr(result, "exit_status", None),
        "timed_out": getattr(result, "timed_out", None),
        "usage": asdict(usage) if usage is not None else {
            "input_tokens": None,
            "output_tokens": None,
            "cache_read_tokens": None,
            "cache_write_tokens": None,
            "reasoning_tokens": None,
            "total_tokens": None,
            "measurement": "unavailable",
        },
        "observation_errors": list(
            getattr(result, "observation_errors", ()) or ()
        ),
    }
    root = supervisor_campaign_root(Path(workspace)) / "review-usage" / review_kind
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = root / f"{uuid.uuid4().hex}.json"
    durable_write_json(path, payload, indent=2)
    return path


def account_review(
    campaign: object,
    result: object,
    *,
    review_kind: str,
    label: str,
    backend: str,
    model: str = "",
    persist_separate_receipt: bool = True,
) -> None:
    """Account one reviewer without mutating the main-Agent token counter."""
    usage = getattr(result, "terminal_usage", None)
    total = getattr(usage, "total_tokens", None)
    reasoning = getattr(usage, "reasoning_tokens", None)
    print(
        f"[orchestrator] {label}: review_kind={review_kind} "
        f"exit={getattr(result, 'exit_status', None)} "
        f"timed_out={getattr(result, 'timed_out', None)} "
        f"tokens={total if total is not None else 'unavailable'} "
        f"reasoning_tokens={reasoning if reasoning is not None else 'unavailable'}",
        flush=True,
    )
    if getattr(result, "exit_status", 0) != 0 or getattr(result, "timed_out", False):
        stderr_tail = str(getattr(result, "stderr_tail", "") or "")
        if stderr_tail:
            print(
                f"[orchestrator] review stderr tail:\n{stderr_tail}",
                file=sys.stderr,
                flush=True,
            )
    if not persist_separate_receipt:
        return
    try:
        workspace = getattr(campaign, "workspace", None)
        if workspace is None:
            raise ValueError("review accounting requires a Campaign workspace")
        record_review_usage(
            Path(workspace),
            result,
            review_kind=review_kind,
            label=label,
            backend=backend,
            model=model,
        )
    except (OSError, TypeError, ValueError) as error:
        print(
            f"[orchestrator] WARNING: could not persist {review_kind} usage: {error}",
            file=sys.stderr,
            flush=True,
        )
