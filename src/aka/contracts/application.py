"""Synchronous application invocation; no execution/backend contract."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class ApplicationRequest:
    """Invoke using the current process's cwd, environment and signal ownership.

    ``repo_root`` explicitly identifies the matching application checkout; it is
    not a request to chdir or modify imports. Make that checkout importable before
    invocation. ``argv=None`` retains the original application's sys.argv behavior.
    An explicit tuple, including an empty tuple, is passed without adding flags.
    """

    repo_root: Path
    argv: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "repo_root", Path(self.repo_root).resolve())
        if self.argv is not None:
            if (
                isinstance(self.argv, (str, bytes))
                or not isinstance(self.argv, Sequence)
                or any(not isinstance(arg, str) for arg in self.argv)
            ):
                raise TypeError("argv must be a sequence of strings or None")
            object.__setattr__(self, "argv", tuple(self.argv))
