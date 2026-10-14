"""Load the application framework from this checkout or its installed packages."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional


def task_modules():
    source = Path(__file__).resolve().parent.parent / "src"
    if (source / "aka" / "legacy" / "task" / "problem").is_dir() and str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from aka.legacy.task import problem
    return problem


def load_problem(directory: str, optimization_mode: str = "leaderboard", *, provider=None,
                 snapshot_directory: Path | None = None):
    task_modules()
    from aka.legacy.application.task import load_operator_problem
    return load_operator_problem(directory, optimization_mode, provider=provider,
                                 snapshot_directory=snapshot_directory)


def execution_modules():
    source = Path(__file__).resolve().parent.parent / "src"
    if (source / "aka" / "legacy" / "application" / "execution.py").is_file() and str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from aka.legacy.application import execution
    from aka.legacy.execution.legacy import runtime
    return execution, runtime


def run(argv: Optional[list[str]] = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    source = repo_root / "src"
    # A source checkout supports the historical command without an install step.
    # Exported application-only checkouts use the installed adapter distributions.
    if (source / "aka" / "bootstrap" / "host.py").is_file():
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
    from aka.legacy.application.host import run_application
    from aka.contracts.application import ApplicationRequest

    return run_application(ApplicationRequest(repo_root, argv))
