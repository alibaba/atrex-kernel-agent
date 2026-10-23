"""Pluggable Wiki knowledge source behind the Supervisor's Tool Runtime.

A provider owns one knowledge source: a local store, a remote service, or anything
else that answers read-only questions. The Tool Runtime keeps authorization,
queueing, single-flight caching, timing receipts, redaction and revision pinning;
the provider only describes its Agent-visible operations and executes them.

Operations are declared by a local plugin catalog (``catalog_root/plugin.json``).
A tool with ``"runtime_tool": "wiki"`` is executed through :meth:`WikiProvider.invoke`
with its input already validated against the declared input schema; its result is
validated against the declared output schema before the Agent receives it. The
catalog may also contribute Skills and ``wiki_<slot>`` instruction files that replace
the Runtime's built-in Wiki prompt fragments (``episode``, ``conversion``,
``framework_baseline`` and ``hardware_evidence``).
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class WikiCall:
    operation: str
    payload: Mapping[str, Any]
    #: The Campaign-pinned identity returned earlier by :meth:`WikiProvider.revision`.
    revision: str
    #: ``time.monotonic()`` deadline. Providers must not outlive it.
    deadline: float
    is_live: Callable[[], bool]
    #: Private, request-scoped directory removed after the call.
    scratch: Path
    #: Trusted Supervisor environment. Never echo it in results.
    environment: Mapping[str, str]
    #: Bounded, cancellable subprocess execution under the request deadline.
    run: Callable[[Sequence[str], Path, Mapping[str, str]], subprocess.CompletedProcess]


class WikiProviderError(Exception):
    """A classified, Agent-safe failure. ``message`` must not contain credentials or private paths."""

    def __init__(self, code: str, message: str, *, retryable: bool = False,
                 repairable: bool = False, next_action: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.repairable = repairable
        self.next_action = next_action


@runtime_checkable
class WikiProvider(Protocol):
    #: Plugin directory declaring the Agent-visible operations, schemas, Skills and instructions.
    catalog_root: Path

    @property
    def cli_root(self) -> Path | None:
        """Directory with ``query_nl.py``/``query_wiki.py``/``query_hardware.py``, or ``None``.

        Only stores implementing the GPU Wiki command-line contract support the
        ``tools/sandbox.py --kind wiki-*`` requests.
        """

    def revision(self) -> str:
        """Identity of the knowledge currently served. The Runtime pins the first value per Campaign."""

    def invoke(self, call: WikiCall) -> Mapping[str, Any]:
        """Execute one declared operation against ``call.revision``.

        Raise :class:`WikiProviderError` with code ``wiki_revision_changed`` when the
        pinned revision can no longer be served.
        """


__all__ = ["WikiCall", "WikiProvider", "WikiProviderError"]
