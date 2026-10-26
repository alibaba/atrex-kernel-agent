"""Record non-promoted episode outcome through the selected Workspace."""
from pathlib import Path
from typing import Any
import json
from aka.contracts.content import WorkspaceFile
from aka.contracts.workspace import CandidateWorkspace


def record_episode_outcome(
    incumbent_workspace: Path,
    *,
    candidate_workspace: CandidateWorkspace,
    base_commit: str,
    version: int,
    episode: int,
    status: str,
    memory_record: dict[str, Any],
) -> str:
    """Advance canonical history without changing the incumbent kernel."""
    if type(version) is not int or version < 0:
        raise ValueError("invalid canonical record version")
    content = json.dumps(memory_record, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    return candidate_workspace.record(
        incumbent_workspace,
        expected_revision=base_commit,
        files=(WorkspaceFile(path=f"memory/v{version}.json", text=content),),
        message=f"v{version}: long-horizon episode {episode} {status}",
    )
