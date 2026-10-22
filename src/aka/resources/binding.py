"""Bind a Workflow request to its frozen Core selection before side effects."""

import json
import os
from pathlib import Path


def bind_campaign_resources(owner, workspace: Path, campaign_id: str):
    if owner is None:
        return None  # Explicit legacy compatibility without a Core resource host.
    from aka.bootstrap.continuation import base_payload, read

    from .host import restore_bound

    payload = read(os.environ)
    value = json.loads(payload) if payload else None
    if value and value.get("version") == 3:
        binding = value["agent_resources"]
        if (
            binding["workspace"] != str(workspace.resolve())
            or binding["campaign_id"] != campaign_id
        ):
            raise ValueError("campaign differs from its frozen agent resources")
        bound = restore_bound(binding, base_selection=json.loads(base_payload(payload)))
        if owner.freeze() != bound.resource_snapshot():
            raise ValueError("selected agent resource contributions changed")
        return bound
    return owner.bind(workspace, campaign_id, base_selection=value)
