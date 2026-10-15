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
        return WorkflowExportSnapshot(1, str(framework), tuple(versions), journals, tuple(attempts))


def apply(ctx, config):
    ctx.provide(WORKFLOW_EXPORT, RetainedWorkflowExport())
