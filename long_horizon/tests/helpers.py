from __future__ import annotations

import subprocess
from pathlib import Path


def run_git(path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=str(path), check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def init_repo(path: Path) -> str:
    path.mkdir(parents=True, exist_ok=True)
    run_git(path, "init")
    run_git(path, "config", "user.name", "test")
    run_git(path, "config", "user.email", "test@example.com")
    (path / "kernel.py").write_text("VALUE = 10\n", encoding="utf-8")
    (path / "test_kernel.py").write_text("# immutable evaluator\n", encoding="utf-8")
    (path / "reference.py").write_text("# immutable reference\n", encoding="utf-8")
    (path / ".gitignore").write_text("/tools\n/reference\n/skills\n", encoding="utf-8")
    run_git(path, "add", ".")
    run_git(path, "commit", "-m", "baseline")
    return run_git(path, "rev-parse", "HEAD")
