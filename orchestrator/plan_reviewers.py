"""Campaign-scoped startup discovery for optional plan reviewers."""

from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
PLAN_REVIEWER_CACHE = Path(".atrex_long_horizon/plan_reviewer_availability.json")
PLAN_REVIEWER_CACHE_SCHEMA_VERSION = 1
DEFAULT_PLAN_REVIEWER_PROBE_TIMEOUT_S = 120

REVIEWER_ENVIRONMENT = {
    "codex": (
        "ATREX_PLAN_REVIEW_CODEX_ENABLED",
        "ATREX_PLAN_REVIEW_CODEX_REASON",
    ),
    "qoder": (
        "ATREX_PLAN_REVIEW_QODER_ENABLED",
        "ATREX_PLAN_REVIEW_QODER_REASON",
    ),
}

_REVIEWER_SPECS = {
    "codex": ("ask-codex.sh", "codex"),
    "qoder": ("ask-qoder.sh", "qodercli"),
}


def _single_line(value: str, limit: int = 500) -> str:
    return " ".join(value.split())[:limit]


def _probe_timeout() -> int:
    raw = os.environ.get("ATREX_PLAN_REVIEW_PROBE_TIMEOUT", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return DEFAULT_PLAN_REVIEWER_PROBE_TIMEOUT_S


def _failure_reason(completed: subprocess.CompletedProcess[str]) -> str:
    lines = [
        line.strip()
        for line in (completed.stderr or "").splitlines()
        if line.strip()
        and "consultation failed with exit code" not in line
        and "running read-only consultation" not in line
        and "running isolated read-only consultation" not in line
        and "running campaign-persistent read-only consultation" not in line
    ]
    if lines:
        return _single_line(lines[-1])
    return f"startup probe exited with code {completed.returncode}"


def _probe_reviewer(
    reviewer: str,
    draft: Path,
    proposal: Path,
    workspace: Path,
    agent_cli: str,
    timeout_s: int,
) -> dict[str, Any]:
    helper_name, matching_backend = _REVIEWER_SPECS[reviewer]
    if agent_cli == matching_backend:
        return {
            "available": True,
            "status": "current_backend",
            "reason": f"{reviewer} is the active campaign backend",
        }

    helper = REPO_ROOT / "skills" / "gen-plan" / "scripts" / helper_name
    environment = os.environ.copy()
    environment["ATREX_AGENT_CLI"] = agent_cli
    # A parent campaign's decision must never suppress a fresh campaign's one-time probe.
    for enabled_name, reason_name in REVIEWER_ENVIRONMENT.values():
        environment.pop(enabled_name, None)
        environment.pop(reason_name, None)
    try:
        completed = subprocess.run(
            [
                "bash",
                str(helper),
                "--input",
                str(draft),
                "--proposal",
                str(proposal),
                "--timeout",
                str(timeout_s),
            ],
            cwd=str(workspace),
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_s + 15,
        )
    except subprocess.TimeoutExpired:
        return {
            "available": False,
            "status": "timeout",
            "reason": f"startup probe exceeded {timeout_s} seconds",
        }
    except OSError as exc:
        return {
            "available": False,
            "status": "unavailable",
            "reason": _single_line(f"startup probe could not start: {exc}"),
        }

    if completed.returncode == 0:
        return {
            "available": True,
            "status": "available",
            "reason": "startup probe completed",
        }
    return {
        "available": False,
        "status": f"unavailable_exit_{completed.returncode}",
        "reason": _failure_reason(completed),
    }


def _valid_cache(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if value.get("schema_version") != PLAN_REVIEWER_CACHE_SCHEMA_VERSION:
        return False
    reviewers = value.get("reviewers")
    if not isinstance(reviewers, dict):
        return False
    for name in _REVIEWER_SPECS:
        record = reviewers.get(name)
        if not isinstance(record, dict) or not isinstance(
            record.get("available"), bool
        ):
            return False
        if not isinstance(record.get("status"), str) or not isinstance(
            record.get("reason"), str
        ):
            return False
    return True


def _load_cache(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if _valid_cache(value) else None


def _write_cache(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def discover_plan_reviewers(
    workspace: Path,
    *,
    agent_cli: str,
) -> tuple[dict[str, Any], bool]:
    """Load a campaign's reviewer decision or probe each reviewer exactly once."""
    workspace = workspace.resolve()
    cache_path = workspace / PLAN_REVIEWER_CACHE
    cached = _load_cache(cache_path)
    if cached is not None:
        return cached, True

    timeout_s = _probe_timeout()
    with tempfile.TemporaryDirectory(prefix="atrex-plan-reviewer-probe-") as directory:
        draft = Path(directory) / "availability_probe.md"
        draft.write_text(
            "# Plan reviewer availability probe\n\n"
            "Confirm that this reviewer can receive a bounded GPU-kernel plan draft and return "
            "the requested structured review sections. No repository inspection is needed.\n",
            encoding="utf-8",
        )
        proposal = Path(directory) / "availability_proposal.md"
        proposal.write_text(
            "# Candidate Proposal\n\n"
            "- Evidence: the reviewer availability probe draft requests a bounded response.\n"
            "- Inference: a successful structured response confirms the consultation path.\n"
            "- Optimization category: reviewer availability validation.\n"
            "- Action: return the required review sections without repository inspection.\n"
            "- Validation: every required response marker is present.\n",
            encoding="utf-8",
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                name: executor.submit(
                    _probe_reviewer,
                    name,
                    draft,
                    proposal,
                    workspace,
                    agent_cli,
                    timeout_s,
                )
                for name in _REVIEWER_SPECS
            }
            reviewers = {name: future.result() for name, future in futures.items()}

    value = {
        "schema_version": PLAN_REVIEWER_CACHE_SCHEMA_VERSION,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "agent_cli": agent_cli,
        "probe_timeout_seconds": timeout_s,
        "reviewers": reviewers,
    }
    _write_cache(cache_path, value)
    return value, False


def plan_reviewer_environment(value: dict[str, Any]) -> dict[str, str]:
    """Translate a validated discovery record into the episode helper contract."""
    reviewers = value["reviewers"]
    environment: dict[str, str] = {}
    for name, (enabled_name, reason_name) in REVIEWER_ENVIRONMENT.items():
        record = reviewers[name]
        environment[enabled_name] = "1" if record["available"] else "0"
        environment[reason_name] = _single_line(record["reason"])
    return environment
