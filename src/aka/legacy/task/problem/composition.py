"""Own a selected Problem provider and release its plugin effects."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from aka.core.boot import boot, compose
from .plugin import PROBLEM

PROFILES_DIR = Path(__file__).with_name("profiles")


class ProblemComposition:
    def __init__(self, report):
        self.report = report
        self.disposed = False

    @property
    def provider(self):
        if self.disposed:
            raise RuntimeError("problem composition has been disposed")
        return self.report.service(PROBLEM.name)

    def dispose(self):
        if not self.disposed:
            self.disposed = True
            self.report.dispose()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.dispose()


def compose_problem(*, optimization_mode: str = "leaderboard", profile: str = "legacy",
                    profiles_dir: Path = PROFILES_DIR,
                    patches: Sequence[Mapping[str, Any]] = (), patch_files: Sequence[Path] = ()) -> ProblemComposition:
    composition = compose(profile, profiles_dir=profiles_dir, patch_files=patch_files,
        patches=({"id": "problem", "config": {"optimization_mode": optimization_mode}}, *patches))
    return ProblemComposition(boot(composition, tokens=(PROBLEM,), required_services=(PROBLEM.name,)))

