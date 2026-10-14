"""Run the preserved optimization body through a borrowed execution entry."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from aka.contracts.application import ApplicationRequest


class LegacyApplication:
    def __init__(
        self,
        *,
        repo_root: Path,
        entry: Callable[[list[str] | None], int],
    ) -> None:
        """Borrow the matching checkout's non-dispatching execution entry.

        The constructor does not run it. The assembly owner supplies the entry
        and owns its lifetime, compatibility checks and process reconstruction.
        No plugin context or composition is retained by the Application.
        """
        if not callable(entry):
            raise TypeError("legacy application entry must be callable")
        self._repo_root = Path(repo_root).resolve()
        self._entry = entry

    def run(self, request: ApplicationRequest) -> int:
        if request.repo_root != self._repo_root:
            raise RuntimeError("a different AKA checkout is requested")
        return self._entry(None if request.argv is None else list(request.argv))
