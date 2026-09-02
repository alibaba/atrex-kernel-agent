from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import TERMINAL_STATUSES
from .protocol import atomic_write_json


WIKI_USAGE_DISPOSITIONS = {
    "applied",
    "partially_applied",
    "reference_only",
    "rejected",
}
WIKI_USAGE_STATUSES = {"declared", "no_material_use", "not_queried"}
EVALUATION_CORRECTNESS = {"pass", "fail", "unknown"}
EVALUATION_PERFORMANCE = {"improved", "not_improved", "unknown"}
PPU_DIAGNOSTIC_ROUTES = {"acu", "timeline", "joint"}
PPU_EVIDENCE_SCHEMAS = {
    "acu": {"ppu-acu-extraction/v2"},
    "timeline": {
        "ppu-fixed-slot-receipt/v4",
        "ppu-critical-path-report/v2",
    },
    "joint": {"ppu-joint-profile/v3"},
}
PPU_DIAGNOSTIC_TEXT_FIELDS = (
    "question",
    "kernel_specialization",
    "workload_identity",
    "device_identity",
    "launch_topology",
    "control_pipeline_identity",
    "finding",
    "decision_impact",
)
QUERY_ID_RE = re.compile(r"wiki-query-[0-9a-f]{32}")
WIKI_ID_RE = re.compile(
    r"(?:gpu_wiki|internal_gpu_wiki)::[A-Za-z0-9][A-Za-z0-9._:-]*"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def infer_live_memory_path(path: Path) -> Path | None:
    """Resolve the incumbent workspace from a regular or linked Git worktree."""
    worktree = path.resolve().parent.parent
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=str(worktree), capture_output=True, text=True,
        )
    except OSError:
        return None
    if result.returncode:
        return None
    common_dir = Path(result.stdout.strip())
    if not common_dir.is_absolute():
        common_dir = (worktree / common_dir).resolve()
    if common_dir.name != ".git":
        return None
    return common_dir.parent / "memory" / "live.json"


def sync_live_memory(
    path: Path,
    value: dict[str, Any],
    *,
    phase: str = "",
    canonical_memory: str = "",
    accepted: bool | None = None,
    memory_version: int | None = None,
    episode: int | None = None,
) -> dict[str, Any]:
    """Write a non-canonical progress view without changing version history."""
    experiments = value.get("experiments")
    if not isinstance(experiments, list):
        experiments = []
    journal_state = str(value.get("state", "in_progress"))
    effective_phase = phase or (
        "exploring" if journal_state == "in_progress" else "awaiting_verification"
    )
    raw_version = value.get("memory_version") if memory_version is None else memory_version
    version = (
        int(raw_version)
        if isinstance(raw_version, int) and not isinstance(raw_version, bool)
        else None
    )
    raw_episode = value.get("episode") if episode is None else episode
    live = {
        "schema_version": "atrex_long_horizon_live_v1",
        "canonical": False,
        "canonical_memory_recorded": effective_phase == "recorded",
        "note": (
            "Live optimization progress only; memory/vN.json is authoritative after "
            "the supervisor records verification or an interrupted recovery outcome."
        ),
        "version": f"v{version}" if version is not None else None,
        "episode": raw_episode,
        "phase": effective_phase,
        "journal_state": journal_state,
        "experiment_count": len(experiments),
        "latest_experiment": experiments[-1] if experiments else None,
        "outcome": value.get("outcome"),
        "candidate_commit": value.get("candidate_commit"),
        "base_commit": value.get("base_commit"),
        "episode_branch": value.get("episode_branch"),
        "created_at": value.get("created_at"),
        "updated_at": utc_now(),
        "canonical_memory": canonical_memory or None,
        "accepted": accepted,
    }
    atomic_write_json(path, live)
    return live


def _sync_live_best_effort(
    journal_path: Path,
    value: dict[str, Any],
    live_path: Path | None,
) -> None:
    inferred = infer_live_memory_path(journal_path)
    destination = inferred or live_path
    if destination is None:
        return
    overrides: dict[str, Any] = {}
    if not isinstance(value.get("memory_version"), int):
        active_path = destination.parent.parent / ".atrex_long_horizon" / "active_episode.json"
        try:
            active = json.loads(active_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            active = None
        if (
            isinstance(active, dict)
            and active.get("episode") == value.get("episode")
            and isinstance(active.get("memory_version"), int)
        ):
            overrides["memory_version"] = active["memory_version"]
            overrides["episode"] = active["episode"]
            overrides["phase"] = str(active.get("phase", "exploring"))
    try:
        sync_live_memory(destination, value, **overrides)
    except OSError:
        # A diagnostic mirror must never invalidate the authoritative journal.
        pass


def initialize(
    path: Path,
    *,
    episode: int,
    base_commit: str,
    branch: str,
    memory_version: int | None = None,
    live_path: Path | None = None,
) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "episode": episode,
        "memory_version": memory_version,
        "base_commit": base_commit,
        "episode_branch": branch,
        "state": "in_progress",
        "experiments": [],
        "outcome": None,
        "created_at": utc_now(),
        "finalized_at": None,
    }
    atomic_write_json(path, value)
    _sync_live_best_effort(path, value, live_path)
    return value


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("unsupported episode journal")
    return value


def normalize_wiki_usage(raw: object) -> tuple[list[dict[str, str]], list[str]]:
    """Validate Wiki attribution without putting telemetry on the promotion path."""
    if not isinstance(raw, list):
        return [], ["wiki_usage must be a list"]
    normalized: list[dict[str, str]] = []
    errors: list[str] = []
    for index, row in enumerate(raw):
        label = f"wiki_usage[{index}]"
        if not isinstance(row, dict):
            errors.append(f"{label} must be an object")
            continue
        query_id = row.get("query_id")
        wiki_id = row.get("wiki_id")
        disposition = row.get("disposition")
        use = row.get("use", "")
        evidence = row.get("evidence", "")
        if not isinstance(query_id, str) or not QUERY_ID_RE.fullmatch(query_id.strip()):
            errors.append(f"{label}.query_id must be an emitted Wiki query_id")
            continue
        if not isinstance(wiki_id, str) or not WIKI_ID_RE.fullmatch(wiki_id.strip()):
            errors.append(f"{label}.wiki_id must be an emitted canonical store::record id")
            continue
        if disposition not in WIKI_USAGE_DISPOSITIONS:
            errors.append(
                f"{label}.disposition must be one of "
                + ", ".join(sorted(WIKI_USAGE_DISPOSITIONS))
            )
            continue
        if not isinstance(use, str) or not isinstance(evidence, str):
            errors.append(f"{label}.use and {label}.evidence must be strings")
            continue
        normalized.append({
            "query_id": query_id.strip(),
            "wiki_id": wiki_id.strip(),
            "disposition": disposition,
            "use": use.strip(),
            "evidence": evidence.strip(),
        })
    return normalized, errors


def normalize_experiment_evaluation(raw: object) -> tuple[dict[str, Any] | None, list[str]]:
    """Normalize the bounded outcome fields used for Wiki effectiveness joins."""
    if not isinstance(raw, dict):
        return None, ["evaluation must be an object"]
    correctness = raw.get("correctness")
    performance = raw.get("performance")
    errors: list[str] = []
    if correctness not in EVALUATION_CORRECTNESS:
        errors.append(
            "evaluation.correctness must be one of "
            + ", ".join(sorted(EVALUATION_CORRECTNESS))
        )
    if performance not in EVALUATION_PERFORMANCE:
        errors.append(
            "evaluation.performance must be one of "
            + ", ".join(sorted(EVALUATION_PERFORMANCE))
        )
    latency_us = raw.get("latency_us")
    if latency_us is not None and (
        isinstance(latency_us, bool) or not isinstance(latency_us, (int, float))
        or latency_us < 0
        or (isinstance(latency_us, float) and not math.isfinite(latency_us))
    ):
        errors.append("evaluation.latency_us must be a non-negative number or null")
    kernel_hash = raw.get("kernel_hash")
    if kernel_hash is not None and not isinstance(kernel_hash, str):
        errors.append("evaluation.kernel_hash must be a string or null")
    if errors:
        return None, errors
    return {
        "correctness": correctness,
        "performance": performance,
        "latency_us": latency_us,
        "kernel_hash": kernel_hash.strip() if isinstance(kernel_hash, str) else None,
    }, []


def normalize_accepted_ppu_diagnostics(
    raw: object,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate reusable PPU evidence selected at terminal handoff."""
    if not isinstance(raw, list):
        return [], ["accepted_ppu_diagnostics must be a list"]
    normalized: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, row in enumerate(raw):
        label = f"accepted_ppu_diagnostics[{index}]"
        if not isinstance(row, dict):
            errors.append(f"{label} must be an object")
            continue
        route = row.get("route")
        if route not in PPU_DIAGNOSTIC_ROUTES:
            errors.append(
                f"{label}.route must be one of "
                + ", ".join(sorted(PPU_DIAGNOSTIC_ROUTES))
            )
            continue
        item: dict[str, Any] = {"route": route}
        item_errors: list[str] = []
        for field in PPU_DIAGNOSTIC_TEXT_FIELDS:
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                item_errors.append(f"{label}.{field} must be a non-empty string")
            else:
                item[field] = value.strip()
        invalidation_conditions = row.get("invalidation_conditions")
        if (
            not isinstance(invalidation_conditions, list)
            or not invalidation_conditions
            or any(
                not isinstance(value, str) or not value.strip()
                for value in invalidation_conditions
            )
        ):
            item_errors.append(
                f"{label}.invalidation_conditions must be a non-empty string list"
            )
        else:
            item["invalidation_conditions"] = [
                value.strip() for value in invalidation_conditions
            ]
        evidence = row.get("evidence")
        if not isinstance(evidence, dict):
            item_errors.append(f"{label}.evidence must be an object")
        else:
            normalized_evidence: dict[str, str] = {}
            for field in ("artifact", "sha256", "schema", "evidence_id"):
                value = evidence.get(field)
                if not isinstance(value, str) or not value.strip():
                    item_errors.append(
                        f"{label}.evidence.{field} must be a non-empty string"
                    )
                else:
                    normalized_evidence[field] = value.strip()
            if (
                normalized_evidence.get("schema")
                and normalized_evidence["schema"] not in PPU_EVIDENCE_SCHEMAS[route]
            ):
                item_errors.append(
                    f"{label}.evidence.schema is not valid for route {route}"
                )
            digest = normalized_evidence.get("sha256", "")
            if digest and not re.fullmatch(r"[0-9a-f]{64}", digest):
                item_errors.append(
                    f"{label}.evidence.sha256 must be a lowercase SHA-256 digest"
                )
            item["evidence"] = normalized_evidence
        if item_errors:
            errors.extend(item_errors)
            continue
        normalized.append(item)
    return normalized, errors


def validate_accepted_ppu_evidence(
    rows: list[dict[str, Any]], workspace: Path
) -> list[str]:
    """Verify that reusable conclusions bind to decision-grade artifacts."""
    errors: list[str] = []
    workspace = workspace.resolve()
    for index, row in enumerate(rows):
        label = f"accepted_ppu_diagnostics[{index}].evidence"
        evidence = row["evidence"]
        artifact = Path(evidence["artifact"])
        if artifact.is_absolute():
            errors.append(f"{label}.artifact must be workspace-relative")
            continue
        resolved = (workspace / artifact).resolve()
        if not resolved.is_relative_to(workspace) or not resolved.is_file():
            errors.append(f"{label}.artifact is outside the workspace or missing")
            continue
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        if digest != evidence["sha256"]:
            errors.append(f"{label} content hash mismatch")
            continue
        try:
            document = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            errors.append(f"{label} is not readable JSON: {error}")
            continue
        if not isinstance(document, dict):
            errors.append(f"{label} must reference a JSON object")
            continue
        if document.get("schema") != evidence["schema"]:
            errors.append(f"{label} schema mismatch")
        if document.get("evidence_id") != evidence["evidence_id"]:
            errors.append(f"{label} evidence_id mismatch")
        if document.get("evidence_grade") != "decision":
            errors.append(f"{label} is not decision-grade")
        validation = document.get("validation")
        accepted = validation == "accepted" or (
            isinstance(validation, dict) and validation.get("status") == "accepted"
        )
        if not accepted:
            errors.append(f"{label} validation is not accepted")
    return errors


def normalize_wiki_attribution(entry: dict[str, Any]) -> None:
    """Apply the explicit use/no-use contract while preserving legacy declarations."""
    raw_usage_present = "wiki_usage" in entry
    errors: list[str] = []
    if raw_usage_present:
        entry["wiki_usage"], usage_errors = normalize_wiki_usage(entry["wiki_usage"])
        errors.extend(usage_errors)
    raw_query_ids = entry.get("wiki_query_ids")
    query_ids: list[str] = []
    if raw_query_ids is not None:
        if not isinstance(raw_query_ids, list) or any(
            not isinstance(item, str) or not QUERY_ID_RE.fullmatch(item.strip())
            for item in raw_query_ids
        ):
            errors.append("wiki_query_ids must contain only emitted Wiki query_ids")
        else:
            query_ids = list(dict.fromkeys(item.strip() for item in raw_query_ids))
            entry["wiki_query_ids"] = query_ids
    usage_query_ids = list(dict.fromkeys(
        row["query_id"] for row in entry.get("wiki_usage", [])
        if isinstance(row, dict) and row.get("query_id")
    ))
    status = entry.get("wiki_usage_status")
    if status is None and entry.get("wiki_usage"):
        # Compatibility for journals written before the explicit status contract.
        status = "declared"
        entry["wiki_usage_status"] = status
        entry["wiki_usage_status_inferred"] = True
    elif status is not None and status not in WIKI_USAGE_STATUSES:
        errors.append(
            "wiki_usage_status must be one of "
            + ", ".join(sorted(WIKI_USAGE_STATUSES))
        )
    if status == "declared" and not entry.get("wiki_usage"):
        errors.append("wiki_usage_status=declared requires at least one valid wiki_usage row")
    if status == "declared":
        if not query_ids:
            query_ids = usage_query_ids
            entry["wiki_query_ids"] = query_ids
        elif not set(usage_query_ids).issubset(query_ids):
            errors.append("wiki_query_ids must include every wiki_usage query_id")
    if status == "no_material_use" and not query_ids:
        errors.append("wiki_usage_status=no_material_use requires non-empty wiki_query_ids")
    if status in {"no_material_use", "not_queried"} and entry.get("wiki_usage"):
        errors.append(f"wiki_usage_status={status} requires an empty or omitted wiki_usage")
    if status == "not_queried" and query_ids:
        errors.append("wiki_usage_status=not_queried requires empty or omitted wiki_query_ids")
    if errors:
        existing = entry.get("wiki_usage_errors")
        entry["wiki_usage_errors"] = [
            *([str(item) for item in existing] if isinstance(existing, list) else []),
            *errors,
        ]


def append_experiment(
    path: Path,
    experiment: dict[str, Any],
    *,
    live_path: Path | None = None,
) -> dict[str, Any]:
    value = load(path)
    if value.get("state") != "in_progress":
        raise ValueError("cannot append to a finalized episode journal")
    if not isinstance(experiment, dict) or not experiment:
        raise ValueError("experiment must be a non-empty JSON object")
    entry = dict(experiment)
    normalize_wiki_attribution(entry)
    if "evaluation" in entry:
        entry["evaluation"], errors = normalize_experiment_evaluation(entry["evaluation"])
        if errors:
            entry["evaluation_errors"] = errors
    entry.setdefault("timestamp", utc_now())
    experiments = value.setdefault("experiments", [])
    if not isinstance(experiments, list):
        raise ValueError("journal experiments must be a list")
    experiments.append(entry)
    atomic_write_json(path, value)
    _sync_live_best_effort(path, value, live_path)
    return value


def finalize(
    path: Path,
    *,
    state: str,
    outcome: dict[str, Any],
    candidate_commit: str = "",
    live_path: Path | None = None,
) -> dict[str, Any]:
    value = load(path)
    if state not in TERMINAL_STATUSES:
        raise ValueError("state must be candidate_ready, pivot, or blocked")
    if value.get("state") not in {"in_progress", state}:
        raise ValueError(f"cannot change finalized state to {state}")
    if not isinstance(outcome, dict) or not str(outcome.get("summary", "")).strip():
        raise ValueError("outcome.summary must be non-empty")
    outcome = dict(outcome)
    directions = outcome.get("next_directions", [])
    if not isinstance(directions, list) or any(not isinstance(item, str) for item in directions):
        raise ValueError("outcome.next_directions must be a list of strings")
    experiments = value.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("a terminal journal requires at least one experiment")
    if state == "candidate_ready" and not candidate_commit.strip():
        raise ValueError("candidate_ready requires candidate_commit")
    if "accepted_ppu_diagnostics" in outcome:
        normalized, errors = normalize_accepted_ppu_diagnostics(
            outcome["accepted_ppu_diagnostics"]
        )
        if not errors:
            errors.extend(
                validate_accepted_ppu_evidence(normalized, path.resolve().parent.parent)
            )
        if errors:
            raise ValueError("; ".join(errors))
        outcome["accepted_ppu_diagnostics"] = normalized
    value["state"] = state
    value["outcome"] = outcome
    value["candidate_commit"] = candidate_commit.strip() or None
    value["finalized_at"] = utc_now()
    atomic_write_json(path, value)
    _sync_live_best_effort(path, value, live_path)
    return value


def validate_terminal(
    path: Path,
    *,
    expected_episode: int,
    base_commit: str,
    branch: str,
    state: str,
    candidate_commit: str = "",
) -> str:
    try:
        value = load(path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return f"episode journal is missing or invalid: {exc}"
    if value.get("episode") != expected_episode:
        return "episode journal identity does not match the active episode"
    if value.get("base_commit") != base_commit or value.get("episode_branch") != branch:
        return "episode journal base commit or branch does not match"
    if value.get("state") != state:
        return "episode journal state does not match handoff"
    experiments = value.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        return "episode journal has no structured experiments"
    outcome = value.get("outcome")
    if not isinstance(outcome, dict) or not str(outcome.get("summary", "")).strip():
        return "episode journal has no terminal outcome summary"
    directions = outcome.get("next_directions", [])
    if not isinstance(directions, list) or any(not isinstance(item, str) for item in directions):
        return "episode journal next_directions is invalid"
    if "accepted_ppu_diagnostics" in outcome:
        normalized, errors = normalize_accepted_ppu_diagnostics(
            outcome["accepted_ppu_diagnostics"]
        )
        if not errors:
            errors.extend(
                validate_accepted_ppu_evidence(normalized, path.resolve().parent.parent)
            )
        if errors:
            return "; ".join(errors)
    if not value.get("finalized_at"):
        return "episode journal is not finalized"
    if state == "candidate_ready" and value.get("candidate_commit") != candidate_commit:
        return "episode journal candidate_commit does not match handoff"
    return ""


def _json_object(raw: str, label: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage one long-horizon episode journal.")
    parser.add_argument(
        "--live-path",
        default="",
        help="Optional non-canonical memory/live.json progress mirror.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    append = sub.add_parser("append")
    append.add_argument("--path", required=True)
    append.add_argument("--experiment-json", required=True)
    finish = sub.add_parser("finalize")
    finish.add_argument("--path", required=True)
    finish.add_argument("--state", choices=sorted(TERMINAL_STATUSES), required=True)
    finish.add_argument("--outcome-json", required=True)
    finish.add_argument("--candidate-commit", default="")
    args = parser.parse_args(argv)
    try:
        if args.command == "append":
            value = append_experiment(
                Path(args.path),
                _json_object(args.experiment_json, "--experiment-json"),
                live_path=Path(args.live_path) if args.live_path else None,
            )
        else:
            value = finalize(
                Path(args.path),
                state=args.state,
                outcome=_json_object(args.outcome_json, "--outcome-json"),
                candidate_commit=args.candidate_commit,
                live_path=Path(args.live_path) if args.live_path else None,
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(value, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
