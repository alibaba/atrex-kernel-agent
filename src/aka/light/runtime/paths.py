"""Private Campaign storage layout, independent of the HTTP server lifecycle."""
import hashlib
import os
from pathlib import Path


SESSION_SHELL_GUARD = Path(__file__).resolve().parents[1] / "agent" / "session_shell_guard.sh"


def supervisor_campaign_root(workspace: Path) -> Path:
    workspace = workspace.resolve()
    scope = hashlib.sha256(str(workspace).encode()).hexdigest()[:16]
    return workspace.parent / ".atrex-supervisor-runtime" / scope


def legacy_resource_root() -> Path:
    """Use an explicit resource checkout or the installed package's asset root."""
    configured = os.environ.get("AKA_LEGACY_REPO_ROOT")
    package = Path(__file__).resolve().parents[2]
    checkout = Path(__file__).resolve().parents[4]
    root = (Path(configured).expanduser().resolve() if configured else
            checkout if (checkout / "src/aka").is_dir() else package)
    if not (root / "src" / "aka").is_dir() and root != package:
        raise RuntimeError(
            "AKA_LEGACY_REPO_ROOT must name a matching resource checkout or the installed AKA package"
        )
    return root
