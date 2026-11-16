"""Supervisor-private candidate source contract and file-system selection."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from threading import RLock

from aka.core.filesystem import durable_write_json, open_private_directory, read_regular_bytes

_bindings = {}
_binding_lock = RLock()


def bind_namespace(namespace: Path, *, file_system: str, contract: dict | None) -> None:
    """Inject a host-owned immutable source binding without a Workflow import.

    The host persists this in its manifest and rebinds it on process startup.
    No contract metadata is added to an Agent repository or to its Git history.
    """
    from .registry import select

    select(file_system).validate_policy(contract)
    namespace = Path(namespace).resolve()
    value = (file_system, json.loads(json.dumps(contract, allow_nan=False)))
    with _binding_lock:
        existing = _bindings.get(namespace)
        if existing is not None and existing != value:
            raise ValueError("Source namespace contract cannot change")
        _bindings[namespace] = value


def _namespace_binding(workspace: Path):
    workspace = Path(workspace).resolve()
    with _binding_lock:
        for path in (workspace, *workspace.parents):
            if path in _bindings:
                name, contract = _bindings[path]
                return name, json.loads(json.dumps(contract))
    return None


def root(workspace: Path) -> Path:
    """Standalone Task storage; Campaign bindings never write this directory."""
    workspace = Path(workspace).resolve()
    return workspace.parent / ".aka-source-state" / hashlib.sha256(str(workspace).encode()).hexdigest()


def bind(workspace: Path, *, file_system: str, contract: dict | None) -> None:
    """Verify a host-injected binding, or persist standalone Task state."""
    campaign_binding = _namespace_binding(workspace)
    if campaign_binding is not None:
        if campaign_binding != (file_system, contract):
            raise ValueError(
                "Campaign source contract cannot change while resuming a Campaign"
            )
        return
    existing = read(workspace)
    if existing is not None and existing != (file_system, contract):
        raise ValueError(
            "Private source contract cannot change while resuming a Campaign"
        )
    state_root = root(workspace)
    with open_private_directory(state_root):
        pass
    durable_write_json(
        state_root / "file-system.json",
        {"schema_version": 1, "name": file_system},
    )
    durable_write_json(
        state_root / "source-contract.json",
        {"schema_version": 1, "contract": contract},
    )


def read(workspace: Path) -> tuple[str, dict | None] | None:
    """Read the host binding, then standalone Task state."""
    campaign_binding = _namespace_binding(workspace)
    if campaign_binding is not None:
        return campaign_binding
    state_root = root(workspace)
    try:
        selected = json.loads(
            read_regular_bytes(state_root / "file-system.json", limit=64 * 1024)
        )
        source = json.loads(
            read_regular_bytes(state_root / "source-contract.json", limit=1024 * 1024)
        )
    except FileNotFoundError:
        if (state_root / "file-system.json").exists() or (
            state_root / "source-contract.json"
        ).exists():
            raise ValueError("Private source state is incomplete") from None
        return None
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Private source state is unreadable") from error
    if (
        not isinstance(selected, dict)
        or selected.get("schema_version") != 1
        or not isinstance(selected.get("name"), str)
        or not isinstance(source, dict)
        or source.get("schema_version") != 1
        or set(source) != {"schema_version", "contract"}
    ):
        raise ValueError("Private source state is invalid")
    contract = source["contract"]
    if contract is not None and not isinstance(contract, dict):
        raise ValueError("Private source contract is invalid")
    return selected["name"], contract


def retire_visible_metadata(workspace: Path) -> None:
    """Remove obsolete public metadata after its contents have been bound."""
    for name in (".atrex_source.json", ".atrex_file_system.json"):
        (Path(workspace) / name).unlink(missing_ok=True)
