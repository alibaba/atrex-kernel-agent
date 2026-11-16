"""Supervisor-private candidate source contract and file-system selection."""

from __future__ import annotations

import json
from pathlib import Path

from aka.legacy.task.persistence import durable_write_json
from aka.light.agent.home import open_private_directory
from aka.light.agent.session_tail import read_regular_bytes
from aka.light.runtime.paths import supervisor_campaign_root


def root(workspace: Path) -> Path:
    """Return source state that is private to this Campaign lineage."""
    return supervisor_campaign_root(Path(workspace)) / "source"


def bind(workspace: Path, *, file_system: str, contract: dict | None) -> None:
    """Verify the Campaign binding, or persist legacy standalone state."""
    from aka.light.runtime.paths import campaign_source_configuration

    campaign_binding = campaign_source_configuration(workspace)
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
    """Read the Campaign manifest binding, then legacy standalone state."""
    from aka.light.runtime.paths import campaign_source_configuration

    campaign_binding = campaign_source_configuration(workspace)
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
