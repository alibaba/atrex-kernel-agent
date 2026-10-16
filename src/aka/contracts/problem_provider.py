"""Typed content handoff; Problem never installs the Agent workspace."""
from pathlib import Path
from typing import Protocol

from .problem import LoadedProblem, EvaluationInput, ProblemSnapshot
from .content import PublicInputContent, WorkspaceFile


class ProblemContent(Protocol):
    """Public preparation and host validation against the retained Problem."""
    evaluation: EvaluationInput
    generated_digest: str

    @property
    def private_reference_dir(self) -> Path | None: ...
    def prepare_public(self) -> PublicInputContent: ...
    def operator_files(self) -> tuple[WorkspaceFile, ...]: ...
    def authoring_assets(self) -> tuple[WorkspaceFile, ...]: ...
    def validate_generated(self, generated: Path) -> None: ...
    def generated_file(self, generated: Path) -> WorkspaceFile: ...
    def record_generated(self) -> None: ...
    def assert_private(self) -> None: ...


class ProblemProvider(Protocol):
    def load(self, directory: Path, *, optimization_mode: str | None = None,
             snapshot_directory: Path | None = None) -> LoadedProblem:
        """Load explicit roles, or restore a validated run snapshot.

        With snapshot_directory, copy required content before returning. Reuse
        that snapshot on recovery even if the original source no longer exists.
        A different problem/mode at an existing snapshot must fail closed.
        """
        ...


    def content(self, problem: LoadedProblem, workspace: Path, *,
                generated_digest: str = "") -> ProblemContent:
        """Return typed public contributions and host checks using owned content."""
        ...

    def runtime_files(self, problem: LoadedProblem) -> tuple[WorkspaceFile, ...]:
        """Describe retained evaluator code for copying from a validated frozen Task.

        These files are Task checkpoint content, not external Resources roots.
        Private cases and benchmark data are never included in this projection.
        """
        ...

    def snapshot(self, problem: LoadedProblem) -> ProblemSnapshot:
        """Validate an already frozen Problem and return its current content identity."""
        ...

    def export_snapshot(self, problem: LoadedProblem, destination: Path) -> ProblemSnapshot:
        """Copy all retained Task roles and authored input, preserving content identity."""
        ...

    def validate_snapshot(self, snapshot: ProblemSnapshot, *,
                          destination: Path | None = None) -> LoadedProblem:
        """Read-only validation of identity, roles, all bytes and modes; never writes."""
        ...

    def restore_snapshot(self, snapshot: ProblemSnapshot, destination: Path) -> LoadedProblem:
        """Validate first, atomically copy to a new destination or require its same identity."""
        ...
