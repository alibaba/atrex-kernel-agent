"""Export declared workspace outputs and task-level observations, without writes."""
from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from aka.contracts.observation import Artifact, Campaign

OUTPUTS = (("kernel", "kernel.py", "Kernel source", "text/x-python"),
           ("solution", "solution.json", "Solution manifest", "application/json"))
SCOPE = "Current workspace files and a snapshot of recorded optimization results."


def _files(workspace: Path):
    for identifier, filename, label, content_type in OUTPUTS:
        path = workspace / filename
        if path.is_file() and not path.is_symlink():
            yield path, {"id": identifier, "filename": filename, "label": label,
                         "content_type": content_type, "kind": "file", "size_bytes": path.stat().st_size}


def describe_artifacts(workspace: Path) -> list[Artifact]:
    return [{"id": "bundle", "filename": f"{workspace.name}-results.zip", "label": "Result bundle",
             "content_type": "application/zip", "kind": "bundle", "size_bytes": None},
            *(metadata for _, metadata in _files(workspace)),
            {"id": "summary", "filename": "result-summary.json", "label": "Optimization summary",
             "content_type": "application/json", "kind": "file", "size_bytes": None}]


def artifact_bytes(workspace: Path, artifact_id: str, snapshot: Campaign) -> bytes:
    """Expose two exact source files and generated summaries, never directory trees."""
    if artifact_id not in {artifact["id"] for artifact in snapshot["artifacts"]}:
        raise KeyError(artifact_id)
    files = list(_files(workspace))
    if artifact_id not in {"bundle", "summary"}:
        path = {metadata["id"]: path for path, metadata in files}[artifact_id]
        try:
            return path.read_bytes()
        except FileNotFoundError as exc:
            raise KeyError(artifact_id) from exc
    report = {"observation_version": snapshot["observation_version"],
              "exported_at": datetime.now(timezone.utc).isoformat(),
              "artifact_scope": "current_workspace",
              "files": [metadata["filename"] for _, metadata in files],
              "campaign": {key: snapshot[key] for key in ("name", "display_name", "framework", "platform", "arch")},
              **{key: snapshot[key] for key in ("updated_at", "canonical", "counts", "tokens", "progress", "issues")}}
    summary = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    if artifact_id == "summary":
        return summary
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for path, metadata in files:
            try:
                data = path.read_bytes()
            except FileNotFoundError as exc:
                raise KeyError(artifact_id) from exc
            archive.writestr(metadata["filename"], data)
        archive.writestr("result-summary.json", summary)
    return buffer.getvalue()
