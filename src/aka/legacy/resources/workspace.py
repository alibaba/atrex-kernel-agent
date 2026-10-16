"""Install only the resources declared by the selected modules and tool plugins."""
from pathlib import Path


def install_workspace_resources(workspace: Path, *, plugin_registry):
    workspace = Path(workspace)
    plugin_registry.check_lock(workspace)
    plugin_registry.environment(workspace)
    return plugin_registry.install(workspace)
