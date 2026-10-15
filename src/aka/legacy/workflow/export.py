"""Project retained Workflow state while owning its private file layout."""
import json
from pathlib import Path
import re
import subprocess
from aka.contracts.workflow_export import WorkflowExportSnapshot, VersionSnapshot
from aka.core.keys import ServiceKey

WORKFLOW_EXPORT = ServiceKey("workflow_export", "WorkflowExportReader", module="aka.contracts.workflow_export")
name = "export"
provide = ("workflow_export",)
identity_files = ("export.py",)
Config = {"type": "object", "properties": {}, "additionalProperties": False}
Defaults = {}


def document(path):
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def verified_memory(workspace, memory):
    """Accept a successful full measurement bound to an existing source commit."""
    import math
    if not isinstance(memory, dict):
        return False
    performance = memory.get("performance") or {}
    latency = performance.get("latency_us_geomean", performance.get("latency_us"))
    if (memory.get("correctness", {}).get("status") != "PASS"
            or memory.get("quality_gate", {}).get("result") != "PASS"
            or isinstance(latency, bool) or not isinstance(latency, (int, float))
            or not math.isfinite(latency) or latency <= 0
            or performance.get("measurement_status") != "complete"):
        return False
    revision = memory.get("git_commit_hash")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        return False
    result = subprocess.run(["git", "cat-file", "-e", f"{revision}:kernel.py"],
                            cwd=workspace, capture_output=True)
    return result.returncode == 0


class RetainedWorkflowExport:
    def read(self, workspace):
        workspace = Path(workspace).resolve()
        versions = []
        for path in sorted((workspace / "memory").glob("v*.json"),
                           key=lambda path: int(path.stem[1:]) if path.stem[1:].isdigit() else 2**31):
            if not re.fullmatch(r"v\d+", path.stem):
                continue
            try:
                content = path.read_bytes()
            except FileNotFoundError:
                continue
            try:
                memory = json.loads(content)
            except (UnicodeDecodeError, json.JSONDecodeError):
                memory = None
            if not isinstance(memory, dict):
                memory = None
            revision = ""
            if memory is not None:
                result = subprocess.run(["git", "log", "-1", "--format=%H", "HEAD", "--",
                                         path.relative_to(workspace).as_posix()], cwd=workspace,
                                        capture_output=True, text=True)
                revision = result.stdout.strip() if result.returncode == 0 else ""
                if revision:
                    blob = subprocess.run(["git", "show", f"{revision}:{path.relative_to(workspace).as_posix()}"],
                                          cwd=workspace, capture_output=True)
                    if blob.returncode or blob.stdout != content:
                        revision = ""
            versions.append(VersionSnapshot(int(path.stem[1:]), memory, revision))
        runtime = workspace / ".atrex_long_horizon"
        journal_paths = list(runtime.glob("episodes/e*/episode_runtime/journal.json"))
        active = document(runtime / "active_episode.json")
        if active and isinstance(active.get("worktree"), str):
            journal_paths.append(Path(active["worktree"]) / ".atrex_long_horizon/journal.json")
        journal_paths.append(runtime / "journal.json")
        for path in journal_paths:
            if path.is_file() and document(path) is None:
                raise ValueError("retained Workflow journal is malformed")
        journals = tuple(value for path in dict.fromkeys(journal_paths)
                         if (value := document(path)) is not None)
        state = document(runtime / "state.json") or {}
        attempts = list(state.get("attempts") or [])
        attempts.extend(value for path in runtime.glob("episodes/e*/attempt.json")
                        if (value := document(path)) is not None)
        if any(not isinstance(value, dict) for value in attempts):
            raise ValueError("retained Workflow attempts must be objects")
        framework = (document(workspace / ".orchestrator_mode.json") or {}).get("framework", "")
        baseline = document(workspace / "framework_baseline.json") or {}
        warning_path = runtime / "warnings.jsonl"
        warnings = []
        if warning_path.is_file():
            for line in warning_path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    warnings.append({key:row[key] for key in ("timestamp", "severity", "code", "stop_reason", "gate_failed") if key in row})
        plans = {}
        for version in versions:
            paths = [workspace / "plans" / f"v{version.version}_plan.md"]
            paths.extend(runtime.glob(f"episodes/e*/archive/worktree_files/plans/v{version.version}_plan.md"))
            path = next((path for path in paths if path.is_file()), None)
            if path:
                plans[version.version] = path.read_text()
        sessions = tuple({key:row[key] for key in ("session_id", "episode", "version", "status", "accepted") if key in row}
                         for row in attempts if row.get("session_id"))
        observed = [path.stat().st_mtime for path in (runtime / "active_episode.json", runtime / "state.json", workspace / "memory/live.json") if path.is_file()]
        live = workspace / "memory/live.json"
        progress = {"name":workspace.name, "latestVersion":max((v.version for v in versions), default=None),
            "episode":(active or {}).get("episode"), "memoryVersion":(active or {}).get("memory_version"),
            "phase":(active or {}).get("phase"), "completedEpisodes":state.get("episodes"),
            "observedTokens":state.get("tokens"), "liveMemoryBytes":live.stat().st_size if live.is_file() else 0,
            "observed_mtimes":observed}
        ready = any(v.version == 0 and v.committed_revision and verified_memory(workspace, v.memory) for v in versions)
        return WorkflowExportSnapshot(2, str(framework), tuple(versions), journals, tuple(attempts),
            baseline, tuple(warnings[-100:]), plans, sessions, progress, ready)


def apply(ctx, config):
    ctx.provide(WORKFLOW_EXPORT, RetainedWorkflowExport())
