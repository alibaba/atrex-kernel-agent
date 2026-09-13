"""Transparent GPU Wiki proxy used by Supervisor-managed Agent sessions."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def maybe_proxy(tool: str, argv: list[str]) -> int | None:
    """Proxy when scoped Runtime authority is present; otherwise use local Wiki."""
    if not os.environ.get("ATREX_AKA_RUNTIME_URL"):
        return None
    repository_root = Path(__file__).resolve().parents[2]
    tools = str(repository_root / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    from sandbox import proxy_wiki

    return proxy_wiki(tool, argv)


__all__ = ["maybe_proxy"]
