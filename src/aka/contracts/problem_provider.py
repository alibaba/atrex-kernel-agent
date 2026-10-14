"""Typed content handoff; Problem never installs the Agent workspace."""
from pathlib import Path
from typing import Protocol

from .problem import LoadedProblem, EvaluationInput
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
