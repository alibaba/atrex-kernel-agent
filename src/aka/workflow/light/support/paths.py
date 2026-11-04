"""Private Campaign storage layout, independent of the HTTP server lifecycle."""
import hashlib
from pathlib import Path


def supervisor_campaign_root(workspace: Path) -> Path:
    workspace = workspace.resolve()
    scope = hashlib.sha256(str(workspace).encode()).hexdigest()[:16]
    return workspace.parent / ".atrex-supervisor-runtime" / scope
