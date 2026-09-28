"""Keep the existing Wiki commands as scoped HTTP clients during Agent sessions."""
import os
import sys
from pathlib import Path


def maybe_proxy(tool: str, argv: list[str]) -> int | None:
    if not os.environ.get("ATREX_AKA_RUNTIME_URL"):
        return None
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    from sandbox import proxy_wiki

    return proxy_wiki(tool, argv)
