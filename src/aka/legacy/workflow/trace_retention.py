"""Declare the minimal workspace evidence needed for offline Wiki mining.

This module does not archive or upload anything.  It publishes an exact list of
small, semantically relevant files for a consumer-owned completion hook.  The
hook remains authoritative for path validation, secret scanning and size limits.
"""
from __future__ import annotations

import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


MANIFEST_NAME = "trace-retention-manifest.json"
MANIFEST_SCHEMA = "atrex-trace-retention-manifest-v1"
TERMINAL_STATUSES = {"completed", "interrupted", "failed"}
MEMORY_RE = re.compile(r"v[0-9]+\.json")
LONG_MEMORY_RE = re.compile(r"long_horizon_e[0-9]+\.json")
PROFILE_RE = re.compile(r"v[0-9]+(?:[_-].+)?")
ROOT_FILES = ("kernel.py", "definition.json", "solution.json", "workload.jsonl")
PROFILE_FILES = (
    "summary.txt", "REPORT.md", "iteration_report.md", "evidence.md",
    "analysis/metrics_key_run.json",
)


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
            elif LONG_MEMORY_RE.fullmatch(path.name):
                add(path, "long-horizon-outcome")

    runtime = workspace / ".atrex_long_horizon"
    add(runtime / "state.json", "long-horizon-state")
    add(runtime / "evaluations.jsonl", "authoritative-evaluation")
    episodes = runtime / "episodes"
    if episodes.is_dir():
        for episode in episodes.glob("e*/episode_runtime"):
            add(episode / "journal.json", "episode-journal")
            add(episode / "evaluations.jsonl", "episode-evaluation")

    profile = workspace / ".gpu_wiki_profile"
    add(profile / "run.json", "wiki-run-identity")
    query_root = profile / "raw" / "query_events"
    if query_root.is_dir():
        for path in query_root.glob("*/*.json"):
            add(path, "wiki-query-event")

    profiles = workspace / "profiles"
    if profiles.is_dir():
        for directory in profiles.iterdir():
            if directory.is_dir() and PROFILE_RE.fullmatch(directory.name):
                for relative in PROFILE_FILES:
                    add(directory / relative, "compact-profiler-evidence")

    return [
        {"path": path, "role": files[path]}
        for path in sorted(files)
    ]


def write_trace_retention_manifest(
    workspace: Path,
    status: str,
    *,
    hardware: dict[str, str] | None = None,
) -> Path | None:
    """Atomically publish terminal evidence; never change optimization outcome."""
    if status not in TERMINAL_STATUSES:
        raise ValueError(f"unsupported trace retention status: {status}")
    workspace = Path(workspace).expanduser().resolve()
    if not workspace.is_dir():
        return None
    path = workspace / MANIFEST_NAME
    temporary = path.with_suffix(f".json.tmp-{os.getpid()}-{uuid.uuid4()}")
    try:
        document: dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA,
            "producer": "atrex-kernel-agent",
            "status": status,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "files": collect_evidence_files(workspace),
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
        temporary.replace(path)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        print(
            f"WARNING trace retention manifest could not be written: {exc}",
            file=sys.stderr,
        )
        return None
    return path
