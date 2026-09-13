"""Declare minimal workspace and Supervisor-private evidence for Wiki mining.

This module does not archive or upload anything.  It publishes an exact list of
small, semantically relevant files in separate manifests for a completion hook. The
hook remains authoritative for path validation, secret scanning and size limits.
"""
from __future__ import annotations

import json
import os
import re
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from .constants import FRAMEWORK_BASELINE_FILE
from .supervisor_runtime import supervisor_campaign_root
from long_horizon.promotion_audit import AUDIT_FILENAME_RE

MANIFEST_NAME = "trace-retention-manifest.json"
MANIFEST_SCHEMA = "atrex-trace-retention-manifest-v1"
TERMINAL_STATUSES = {"completed", "interrupted", "failed"}
MEMORY_RE = re.compile(r"v[0-9]+\.json")
ROOT_FILES = ("kernel.py", "definition.json", "solution.json", "workload.jsonl")


def _safe_relative(workspace: Path, value: str) -> str | None:
    raw = value.split("#", 1)[0].replace("\\", "/").strip()
    path = PurePosixPath(raw)
    if (
        not raw
        or path.is_absolute()
        or ".." in path.parts
        or ".git" in path.parts
    ):
        return None
    source = workspace / path.as_posix()
    try:
        resolved = source.resolve(strict=True)
        resolved.relative_to(workspace.resolve(strict=True))
    except (OSError, ValueError):
        return None
    if source.is_symlink() or not source.is_file():
        return None
    return path.as_posix()


def _solution_sources(workspace: Path) -> list[str]:
    try:
        document = json.loads((workspace / "solution.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return []
    rows = document.get("sources") if isinstance(document, dict) else None
    if not isinstance(rows, list):
        return []
    result = []
    for row in rows:
        value = row.get("path") if isinstance(row, dict) else row
        if isinstance(value, str):
            relative = _safe_relative(workspace, value)
            if relative:
                result.append(relative)
    return result


def collect_evidence_files(workspace: Path) -> list[dict[str, str]]:
    """Collect known semantic files without walking caches or session trees."""
    workspace = workspace.resolve()
    files: dict[str, str] = {}

    def add(path: Path, role: str) -> None:
        try:
            relative = path.relative_to(workspace).as_posix()
        except ValueError:
            return
        safe = _safe_relative(workspace, relative)
        if safe:
            files.setdefault(safe, role)

    for name in ROOT_FILES:
        add(workspace / name, "extraction-core")
    for relative in _solution_sources(workspace):
        add(workspace / relative, "candidate-source")

    memory = workspace / "memory"
    if memory.is_dir():
        for path in memory.glob("*.json"):
            if MEMORY_RE.fullmatch(path.name):
                add(path, "canonical-memory")

    runtime = workspace / ".atrex_long_horizon"
    add(runtime / "state.json", "long-horizon-state")
    add(runtime / "evaluations.jsonl", "authoritative-evaluation")
    episodes = runtime / "episodes"
    if episodes.is_dir():
        for episode_dir in episodes.glob("e*"):
            add(
                episode_dir / "supervisor_runtime/journal.json",
                "runtime-journal",
            )
            add(
                episode_dir / "supervisor_runtime/evaluations.jsonl",
                "episode-evaluation",
            )

    return [
        {"path": path, "role": files[path]}
        for path in sorted(files)
    ]


def collect_private_evidence_files(root: Path) -> list[dict[str, str]]:
    """Collect baseline, promotion, and Wiki evidence outside the workspace."""
    if root.is_symlink():
        return []
    candidates = [(root / FRAMEWORK_BASELINE_FILE, "framework-baseline-pin")]
    promotions = root / "promotions"
    if not promotions.is_symlink() and promotions.is_dir():
        candidates.extend(
            (path, "promotion-audit") for path in sorted(promotions.iterdir())
            if AUDIT_FILENAME_RE.fullmatch(path.name)
        )
    profile = root / "wiki-profile"
    if not profile.is_symlink() and profile.is_dir():
        candidates.append((profile / "run.json", "wiki-run-identity"))
        candidates.extend(
            (path, "wiki-query-event")
            for path in sorted((profile / "raw" / "query_events").glob("*/*.json"))
        )
    files: list[dict[str, str]] = []
    for path, role in candidates:
        relative = _safe_relative(root, path.relative_to(root).as_posix())
        if relative:
            files.append({"path": relative, "role": role})
    return files


def _write_manifest(
    root: Path,
    status: str,
    files: list[dict[str, str]],
    *,
    hardware: dict[str, str] | None = None,
) -> Path | None:
    path = root / MANIFEST_NAME
    temporary = path.with_suffix(f".json.tmp-{os.getpid()}-{uuid.uuid4()}")
    try:
        if root.is_symlink() or path.is_symlink():
            raise ValueError("trace retention destination cannot be a symlink")
        root.mkdir(parents=True, mode=0o700, exist_ok=True)
        document: dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA,
            "producer": "atrex-kernel-agent",
            "status": status,
            "generated_at": datetime.now(UTC).isoformat(),
            "files": files,
            "excluded_families": [
                "coding-agent-session-transcripts",
                "stdout-stderr-logs",
                "temporary-episode-worktrees",
                "bulk-profiler-captures",
                "dependency-caches",
            ],
        }
        normalized_hardware = {
            key: str((hardware or {}).get(key) or "").strip()
            for key in ("platform", "arch", "sandbox_hardware")
            if str((hardware or {}).get(key) or "").strip()
        }
        if normalized_hardware:
            document["hardware"] = normalized_hardware
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.chmod(0o600)
        temporary.replace(path)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        print(
            f"WARNING trace retention manifest could not be written: {exc}",
            file=sys.stderr,
        )
        return None
    return path


def write_trace_retention_manifest(
    workspace: Path,
    status: str,
    *,
    hardware: dict[str, str] | None = None,
) -> Path | None:
    """Publish two root-relative manifests, even after the Runtime has stopped.

    The workspace manifest contains no private paths or Wiki audit events.
    The private Campaign manifest declares baseline, promotion, and Wiki evidence in place; completion
    hooks must collect it from that root without restoring it into the workspace.
    """
    if status not in TERMINAL_STATUSES:
        raise ValueError(f"unsupported trace retention status: {status}")
    workspace = Path(workspace).expanduser().resolve()
    if not workspace.is_dir():
        return None
    try:
        private_root = supervisor_campaign_root(workspace)
        _write_manifest(
            private_root, status, collect_private_evidence_files(private_root), hardware=hardware
        )
    except Exception as exc:
        print(
            f"WARNING private trace retention manifest could not be written: {exc}",
            file=sys.stderr,
        )
    try:
        return _write_manifest(
            workspace, status, collect_evidence_files(workspace), hardware=hardware
        )
    except Exception as exc:
        print(f"WARNING workspace evidence could not be collected: {exc}", file=sys.stderr)
        return None
