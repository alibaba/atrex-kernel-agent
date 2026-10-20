"""Controller version facts, independent of optional Agent memory summaries.

These Git-tracked facts retain atomic Kernel/version promotion and recovery when
memory generation is disabled. They are never projected into Agent drafts or
GPU bundles. Existing memory-only campaigns remain readable without migration.
"""
from pathlib import Path

from .durable_state import durable_write_json

STATE_DIRECTORY = ".atrex_versions"
FACT_FIELDS = frozenset({
    "version", "timestamp", "performance", "correctness", "quality_gate",
    "git_commit_hash", "long_horizon", "framework_baseline",
})


def version_record_path(workspace: Path, version: int, *, prefer_summary: bool = False) -> Path:
    if isinstance(version, bool) or not isinstance(version, int) or version < 0:
        raise ValueError("Version must be a non-negative integer")
    private = workspace / STATE_DIRECTORY / f"v{version}.json"
    memory = workspace / "memory" / f"v{version}.json"
    if prefer_summary and memory.exists():
        return memory
    return private if private.exists() or not memory.exists() else memory


def save_version_record(workspace: Path, version: int, record: dict, *, generate_memory: bool = False) -> Path:
    if record.get("version") != f"v{version}":
        raise ValueError("Version record identity does not match")
    version_record_path(workspace, version)  # Validate before constructing paths.
    path = workspace / STATE_DIRECTORY / f"v{version}.json"
    durable_write_json(path, {key: value for key, value in record.items() if key in FACT_FIELDS})
    if generate_memory:
        durable_write_json(workspace / "memory" / f"v{version}.json", record)
    return path


def version_commit_paths(workspace: Path, version: int, *, generate_memory: bool) -> list[str]:
    paths = [str(version_record_path(workspace, version).relative_to(workspace))]
    summary = f"memory/v{version}.json"
    if generate_memory and summary not in paths and (workspace / summary).exists():
        paths.append(summary)
    return paths
