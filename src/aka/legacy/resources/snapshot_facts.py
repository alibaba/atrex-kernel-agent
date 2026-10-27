"""Expose verified retained Resources facts without constructing their runtime."""
from pathlib import Path

from aka.contracts.agent_resources import (
    AgentResourceFile, ResourceDirectoryFact, ResourceRootFacts,
    ResourcesSnapshot, ResourcesSnapshotFacts,
)


def _verified_snapshot(snapshot: ResourcesSnapshot):
    """Bind the verified retained closure for Resources-owned pure consumers."""
    from . import snapshot_store
    from .snapshot_selection import validate_selection

    manifest = snapshot_store.load_snapshot(snapshot)
    reader = snapshot_store.object_reader(snapshot, manifest)
    absent = validate_selection(manifest, reader)
    return manifest, reader, absent


def inspect_snapshot(snapshot: ResourcesSnapshot) -> ResourcesSnapshotFacts:
    """Return original selection and content facts without inspecting source roots."""
    manifest, _reader, absent = _verified_snapshot(snapshot)
    roots = tuple(ResourceRootFacts(
        root=Path(row["source"]), kind=row["kind"], mode=row["mode"],
        files=tuple(AgentResourceFile(entry["path"], entry["mode"], entry["sha256"])
                    for entry in row["entries"] if entry["kind"] == "file"),
        directories=tuple(ResourceDirectoryFact(entry["path"], entry["mode"])
                          for entry in row["entries"] if entry["kind"] == "directory"),
    ) for row in manifest["roots"])
    return ResourcesSnapshotFacts(snapshot.identity, manifest["selection"], roots,
                                  tuple(sorted(absent)))
