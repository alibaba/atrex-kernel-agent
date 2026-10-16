"""Problem content and explicit roles, independent of the input bench layout."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from .content import WorkspaceFile


@dataclass(frozen=True)
class AgentProblem:
    """Only files approved for installation in the Agent workspace."""
    files: tuple[WorkspaceFile, ...]


@dataclass(frozen=True)
class InitialSource:
    """Selected implementation; its entry need not be the semantic reference."""
    kind: str
    entry: Path
    files: tuple[WorkspaceFile, ...]


@dataclass(frozen=True)
class EvaluationInput:
    """Host-only content interpreted by the selected evaluation adapter.

    The legacy adapter uses a flat bundle and a separately copied runtime.
    These paths describe owned content, not the original benchmark location.
    """
    adapter: str
    directory: Path
    files: tuple[WorkspaceFile, ...]
    runtime_root: Path | None = None
    private_reference_dir: Path | None = None
    case_ids: tuple[str, ...] = ()
    harness: WorkspaceFile | None = None
    worker_requirements: WorkspaceFile | None = None


@dataclass(frozen=True)
class LoadedProblem:
    name: str
    directory: Path
    reference: Path
    bench_root: Path | None
    public_contract: Path | None
    public_contract_source: Literal["none", "provided", "auto"]
    agent: AgentProblem | None = None
    initial_source: InitialSource | None = None
    evaluation: EvaluationInput | None = None
    snapshot_directory: Path | None = None



@dataclass(frozen=True)
class ProblemSnapshot:
    """Opaque Task snapshot location and content identity, including authored input."""
    directory: Path
    identity: str
