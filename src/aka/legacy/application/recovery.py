"""Commit legacy launch metadata with a replayable, validated migration record."""
from __future__ import annotations

import json
from pathlib import Path

from aka.bootstrap.continuation import DIGEST_ENV, SELECTION_ENV, digest, restore

MIGRATION_FILE = "launch-migration.json"


def has_selection(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _validate(directory, record):
    if (not isinstance(record, dict) or set(record) != {"version", "previous", "next", "selection"}
            or type(record["version"]) is not int or record["version"] != 1):
        raise RuntimeError("invalid recovery migration record")
    previous, following, payload = record["previous"], record["next"], record["selection"]
    if (not isinstance(previous, dict) or previous.get("schema_version") != 3
            or "launch_environment" in previous or not isinstance(following, dict)
            or following.get("schema_version") != 4 or not isinstance(payload, str)):
        raise RuntimeError("invalid recovery migration metadata")
    # Only these fields can change in the already-validated owner upgrade.
    allowed = {"schema_version", "created_at", "launch_environment"}
    if "runtime_health_command" not in previous:
        allowed.add("runtime_health_command")
    if ({key: value for key, value in previous.items() if key not in allowed}
            != {key: value for key, value in following.items() if key not in allowed}
            or previous.get("environment_state_file") != str(directory / "failure.json")):
        raise RuntimeError("recovery migration changes the original configuration")
    selection = restore(payload)
    environment = {**selection.environment, SELECTION_ENV: str(directory / "launch-selection.json"),
                   DIGEST_ENV: digest(payload)}
    if following.get("launch_environment") != environment:
        raise RuntimeError("recovery migration launch environment mismatch")
    return previous, following, payload


def complete_migration(directory: Path, *, owner: bool, repo_root=None, environment=None, configuration=None):
    """Replay only a verified owner transaction; never infer one from an orphan."""
    directory = directory.resolve()
    path = directory / MIGRATION_FILE
    if not has_selection(path):
        return None
    if not owner:
        raise RuntimeError("only the recovery owner can complete a pending launch migration")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        previous, following, payload = _validate(directory, record)
        current = json.loads((directory / "restart.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"cannot validate recovery migration: {exc}") from exc
    if current != previous and current != following:
        raise RuntimeError("recovery migration does not match active restart metadata")
    if configuration is not None and any(following.get(key) != value for key, value in configuration.items()):
        raise RuntimeError("recovery migration has different resolved configuration")
    if repo_root is not None:
        selection = restore(payload)
        if selection.composition.variables.get("repo_root") != str(Path(repo_root).resolve()):
            raise RuntimeError("recovery migration selected a different AKA checkout")
    if environment is not None:
        from aka.bootstrap.continuation import read
        supplied = read(environment)
        if supplied is not None and supplied != payload:
            raise RuntimeError("cannot replace a pending recovery launch selection")
    selection_path = directory / "launch-selection.json"
    if has_selection(selection_path):
        if selection_path.is_symlink() or selection_path.read_text(encoding="utf-8") != payload:
            raise RuntimeError("recovery migration does not match the durable launch selection")
    elif current == following:
        # A committed schema-4 record losing its selection is corruption, not
        # an interrupted write that may be repaired from legacy state.
        raise RuntimeError("committed recovery migration lacks its launch selection")
    from orchestrator.durable_state import durable_unlink, durable_write_json, durable_write_text
    if not selection_path.exists():
        durable_write_text(selection_path, payload)
    if current == previous:
        durable_write_json(directory / "restart.json", following, indent=2, ensure_ascii=False)
    durable_unlink(path, missing_ok=True)
    return following["launch_environment"]


def begin_migration(directory: Path, previous, following, payload):
    """Record intent after configuration validation and before either new write."""
    from orchestrator.durable_state import durable_write_json
    record = {"version": 1, "previous": previous, "next": following, "selection": payload}
    _validate(directory.resolve(), record)
    if has_selection(directory / MIGRATION_FILE):
        raise RuntimeError("a recovery launch migration is already pending")
    durable_write_json(directory / MIGRATION_FILE, record)
    complete_migration(directory, owner=True)
