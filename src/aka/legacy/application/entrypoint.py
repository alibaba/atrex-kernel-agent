"""Bind the legacy checkout entry without importing or running it during boot."""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

from .application import LegacyApplication

class LegacyEntrypoint:
    def __init__(self, repo_root: Path, dependencies=None) -> None:
        self._repo_root = Path(repo_root).resolve()
        self._dependencies = dependencies

    def __call__(self, argv: list[str] | None) -> int:
        expected = self._repo_root / "orchestrator" / "optimize.py"
        # Direct script execution aliases __main__ under this name and has no
        # module spec. Reuse that live object instead of importing a second copy.
        module = sys.modules.get("orchestrator.optimize")
        if module is None:
            try:
                spec = importlib.util.find_spec("orchestrator.optimize")
            except ModuleNotFoundError as exc:
                raise RuntimeError("make the matching AKA checkout importable before invocation") from exc
            if spec is None or spec.origin is None or Path(spec.origin).resolve() != expected:
                raise RuntimeError(f"application import does not match the requested checkout: {expected}")
            module = importlib.import_module("orchestrator.optimize")
        if Path(module.__file__).resolve() != expected:
            raise RuntimeError("a different AKA checkout is already loaded")
        entry = getattr(module, "_run_application", None)
        if not callable(entry):
            raise RuntimeError("checkout lacks the non-dispatching application entry; use a matching revision")
        # Never call public main: it will become the Core dispatcher. The internal
        # entry retains the full recovery wrapper, not just _run_main's campaign loop.
        if self._dependencies is None:
            return entry(argv)
        from .workspace import candidate_workspace
        with candidate_workspace(self._dependencies["workspace"]):
            return entry(argv, dependencies=self._dependencies)


def create_legacy_application(repo_root: Path, *, problem=None, source=None, workspace=None) -> LegacyApplication:
    """Construct with a concrete lazy entry; leave composition ownership here."""
    dependencies = None
    if any(value is not None for value in (problem, source, workspace)):
        if any(value is None for value in (problem, source, workspace)):
            raise ValueError("legacy application requires Problem, Source and Workspace together")
        dependencies = dict(problem=problem, source=source, workspace=workspace)
    return LegacyApplication(repo_root=repo_root, entry=LegacyEntrypoint(repo_root, dependencies))
