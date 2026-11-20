"""Locate every measurement workspace belonging to a Campaign."""
from pathlib import Path

from observability._internal.common import discover_scopes


def measurement_workspaces(workspace: Path, campaign_root: Path | None = None) -> list[Path]:
    """Return every Campaign measurement scope."""
    del campaign_root
    return [scope.workspace for scope in discover_scopes(workspace)]
