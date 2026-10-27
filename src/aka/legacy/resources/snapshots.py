"""Owner facade for retained resource data and exact original-root restoration.

Standalone copy is a data operation. Its caller places output outside active
Agent grants, or restores its result and obtains new filesystem_access before
preparing a view. Existing prepared views are never changed retroactively.
"""
from pathlib import Path
from aka.contracts.agent_resources import (
    AgentSkill, ResourcesRegistry, ResourcesSnapshot, WorkspaceResourceContribution,
)


def _source_roots(registry):
    roots = []
    if registry.plugin_dir is not None:
        roots.append(Path(registry.plugin_dir))
    for plugin in registry.plugins:
        roots.append(plugin.root)
        roots.extend(path for path, _mount in plugin.resources.values() if path.exists())
        roots.extend(plugin.root / skill["path"] for skill in plugin.manifest.get("skills", {}).values()
                     if (plugin.root / skill["path"]).exists())
    roots.extend(value.source for value in registry._agent_skills)
    roots.extend(value.source for value in registry._workspace_resources)
    return tuple(dict.fromkeys(Path(path).resolve(strict=True) for path in roots))


def _snapshot_paths(snapshot):
    return (Path(snapshot.directory).absolute(), *((Path(snapshot.content_directory).absolute(),)
             if snapshot.content_directory is not None else ()))


def export_snapshot(registry, destination, *, content_directory=None) -> "ResourcesSnapshot":
    """Retain the registry's explicit selected roots and register their access boundary."""
    from . import snapshot_store
    roots = _source_roots(registry)
    selection = registry.resource_snapshot()
    paths = (Path(destination).absolute(), *((Path(content_directory).absolute(),)
             if content_directory is not None else ()))
    registry._check_snapshot_paths(paths, sources=roots)
    def before_publish(manifest, read):
        from .snapshot_selection import validate_selection
        validate_selection(manifest, read)
        if _source_roots(registry) != roots:
            raise ValueError("selected Resources source roots changed during capture")
        for row in manifest["roots"]:
            if snapshot_store.inspect_root(Path(row["source"])) != row:
                raise ValueError("selected Resources closure changed during capture")
        registry.check_resource_snapshot(selection)
        registry._check_snapshot_paths(paths, sources=roots)
    result = snapshot_store.capture_snapshot(selection, roots,
        destination, content_directory=content_directory, before_publish=before_publish)
    registry._retain_snapshot_paths(_snapshot_paths(result), sources=roots)
    return result


def copy_snapshot(snapshot, destination, *, content_directory=None) -> "ResourcesSnapshot":
    """Copy retained data without original roots, boot, registry or tool construction."""
    from . import snapshot_store
    validate_snapshot(snapshot, check_original_destinations=False)
    return snapshot_store.copy_snapshot(snapshot, destination, content_directory=content_directory)


def validate_snapshot(snapshot, *, check_original_destinations=True) -> None:
    """Read retained data and optional exact targets; never construct a registry or boot."""
    from . import snapshot_store
    manifest = snapshot_store.load_snapshot(snapshot)
    from .snapshot_selection import validate_selection
    absent = validate_selection(manifest, snapshot_store.object_reader(snapshot, manifest))
    if check_original_destinations:
        from .snapshot_restore import validate_original_destinations
        validate_original_destinations(manifest, absent=absent)


def _registry(selection):
    from .plugins import PluginRegistry
    plugins = selection["plugins"]
    # Explicit empty string means no catalog; None would select ambient defaults.
    catalog = plugins.get("plugin_dir") or ""
    paths = tuple(Path(path) for path in plugins.get("plugin_paths", ()))
    skills = tuple(AgentSkill(value["name"], Path(value["source"]), value["description"])
                   for value in selection["agent_skills"]["skills"])
    resources = tuple(WorkspaceResourceContribution(value["name"], Path(value["source"]),
        value["destination"], visibility=value["visibility"], version=value["version"])
        for value in selection["workspace_resources"]["resources"])
    registry = PluginRegistry(plugin_dir=catalog, plugins=paths, resources=resources, agent_skills=skills)
    if registry.resource_snapshot() != selection:
        raise ValueError("restored Resources selection or runtime identity differs")
    return registry


def restore_snapshot(snapshot) -> "ResourcesRegistry":
    """Restore exact original roots before boot and return a usable public registry.

    This does not supply Workflow's separate all-owner checkpoint validation or
    change any existing prepared Agent view.
    """
    from . import snapshot_store
    from .snapshot_restore import restore_original_roots
    manifest = snapshot_store.load_snapshot(snapshot)
    from .snapshot_selection import validate_selection
    absent = validate_selection(manifest, snapshot_store.object_reader(snapshot, manifest))
    def finish():
        registry = _registry(manifest["selection"])
        registry._retain_snapshot_paths(_snapshot_paths(snapshot))
        return registry
    return restore_original_roots(snapshot, manifest, finish=finish, absent=absent)
