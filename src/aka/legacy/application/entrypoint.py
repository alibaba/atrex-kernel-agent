"""Bind the legacy checkout entry without importing or running it during boot."""
from __future__ import annotations

from pathlib import Path

from .application import LegacyApplication

class LegacyEntrypoint:
    def __init__(self, repo_root: Path, dependencies=None) -> None:
        self._repo_root = Path(repo_root).resolve()
        self._dependencies = dependencies

    def __call__(self, argv: list[str] | None) -> int:
        from aka.legacy.compat.optimize import _run_application as entry
        if self._dependencies is None:
            return entry(argv)
        from .workspace import candidate_workspace
        with candidate_workspace(self._dependencies["workspace"]):
            return entry(argv, dependencies=self._dependencies)


def create_legacy_application(repo_root: Path, *, problem=None, source=None, workspace=None, workflow=None) -> LegacyApplication:
    """Construct with a concrete lazy entry; leave composition ownership here."""
    dependencies = None
    if workflow is not None and any(value is None for value in (problem, source, workspace)):
        raise ValueError("selected Workflow requires Problem, Source and Workspace together")
    if any(value is not None for value in (problem, source, workspace)):
        if any(value is None for value in (problem, source, workspace)):
            raise ValueError("legacy application requires Problem, Source and Workspace together")
        dependencies = dict(problem=problem, source=source, workspace=workspace)
        if workflow is not None:
            dependencies["workflow"] = workflow
    return LegacyApplication(repo_root=repo_root, entry=LegacyEntrypoint(repo_root, dependencies))
