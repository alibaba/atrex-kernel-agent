"""Private Campaign storage layout, independent of the HTTP server lifecycle."""
import hashlib
import os
from pathlib import Path


def supervisor_campaign_root(workspace: Path) -> Path:
    workspace = workspace.resolve()
    scope = hashlib.sha256(str(workspace).encode()).hexdigest()[:16]
    return workspace.parent / ".atrex-supervisor-runtime" / scope


def legacy_resource_root() -> Path:
    """Return the profile-selected public checkout used by legacy resource adapters.

    Installed launches bind this value through the launch profile. The source-tree
    fallback keeps direct developer imports usable without recreating the deleted
    ``aka.legacy.application.paths`` compatibility module.
    """
    configured = os.environ.get("AKA_LEGACY_REPO_ROOT")
    root = (Path(configured).expanduser().resolve() if configured
            else Path(__file__).resolve().parents[4])
    if not (root / "src" / "aka").is_dir():
        raise RuntimeError(
            "AKA_LEGACY_REPO_ROOT must name the matching AKA checkout selected by the profile"
        )
    return root
