"""Explicit input/output views for auxiliary sessions; legacy optimizer stays Git-backed."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

from .session_tail import _regular_bytes

WORKSPACE_ROLE_ENV = "ATREX_AGENT_WORKSPACE_ROLE"
WORKSPACE_LAYOUTS = {
    "plan-review-probe": (("availability_probe.md", "availability_proposal.md"), ()),
    "production-review": (("review_request.json", "candidate"), ("dependency_review.json",)),
    "problem-generation": (
        ("reference.py", "input.py", "shapes.json", "metadata.json"),
        ("agent_problem.json",),
    ),
    "baseline-exit-review": (("crash_record.json", "candidate"), ("resume.json",)),
    "baseline-correctness-review": (("context",), ("correctness_review.md",)),
}


def reset_episode_scratch(workspace: Path) -> None:
    """Only at a new Episode boundary, never on a same-Episode retry/resume."""
    scratch = workspace.resolve() / "scratch"
    if scratch.is_symlink() or scratch.is_file():
        scratch.unlink()
    elif scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(mode=0o700)


class AuxiliaryWorkspace:
    def __init__(
        self, workspace: Path, home: Path, role: str,
        *, input_files: dict[str, Path] | None = None,
    ):
        self.workspace = workspace
        self.inputs, self.outputs = WORKSPACE_LAYOUTS[role]
        input_files = dict(input_files or {})
        if set(input_files) - set(self.inputs):
            raise ValueError("Explicit auxiliary files must use declared input names")
        self.root = Path(tempfile.mkdtemp(prefix="view-", dir=home.parent))
        try:
            count, total = 0, 0

            def copy_file(source: Path, destination: Path) -> None:
                nonlocal count, total
                content = _regular_bytes(source, limit=16 * 1024 * 1024 + 1)
                count, total = count + 1, total + len(content)
                if count > 4096 or total > 16 * 1024 * 1024:
                    raise ValueError("Auxiliary input view exceeds 4096 files / 16 MiB")
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)

            for name in (*self.inputs, *self.outputs):
                if name in input_files:
                    # Supervisor-supplied files may live outside the Campaign.
                    # Snapshot only these regular files, never their directory.
                    copy_file(input_files[name], self.root / name)
                    continue
                source = workspace / name
                if not source.exists() and not source.is_symlink():
                    continue
                if source.is_symlink():
                    raise ValueError("Auxiliary inputs/outputs cannot be symlinks")
                paths = sorted(source.rglob("*")) if source.is_dir() else [source]
                if source.is_dir():
                    (self.root / name).mkdir()
                for path in paths:
                    if path.is_symlink():
                        raise ValueError("Auxiliary inputs cannot contain symlinks")
                    destination = self.root / path.relative_to(workspace)
                    if path.is_dir():
                        destination.mkdir(parents=True, exist_ok=True)
                        continue
                    copy_file(path, destination)
            (self.root / "scratch").mkdir(exist_ok=True)
        except BaseException:
            self.close()
            raise

    def publish(self) -> None:
        for name in self.outputs:
            source = self.root / name
            if not source.exists() and not source.is_symlink():
                continue
            content = _regular_bytes(source, limit=8 * 1024 * 1024 + 1)
            if len(content) > 8 * 1024 * 1024:
                raise ValueError(f"Auxiliary output exceeds 8 MiB: {name}")
            descriptor, temporary = tempfile.mkstemp(prefix=".report-", dir=self.workspace)
            try:
                with os.fdopen(descriptor, "wb") as output:
                    output.write(content)
                os.replace(temporary, self.workspace / name)
            finally:
                Path(temporary).unlink(missing_ok=True)

    def close(self) -> None:
        primary_error = sys.exception()
        try:
            shutil.rmtree(self.root)
        except Exception as cleanup_error:
            if primary_error is None:
                raise
            # Cleanup is still attempted on interruption, failed publication and
            # failed construction, but must not replace the original exception.
            primary_error.add_note(
                f"Auxiliary workspace cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}"
            )
