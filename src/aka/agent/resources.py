"""Consume selected runtime files without knowing their provider's package layout."""

import hashlib
from pathlib import Path
from aka.contracts.agent_resources import InstalledWorkspaceResource
from .execution.home import read_regular_file


def shell_guard_path(resource):
    if resource is None:
        return None
    if not isinstance(resource, InstalledWorkspaceResource):
        raise TypeError("Agent shell guard must be a selected resource handle")
    if (
        len(resource.files) != 1
        or resource.files[0].path != "."
        or resource.visibility != "runtime"
    ):
        raise ValueError("Agent shell guard must be one runtime-only file")
    data, info = read_regular_file(resource.source)
    if hashlib.sha256(data).hexdigest() != resource.files[0].sha256:
        raise ValueError("Selected Agent shell guard identity changed")
    if info.st_mode & 0o777 != resource.files[0].mode:
        raise ValueError("Selected Agent shell guard permissions changed")
    return Path(resource.source)


def discover_resources(installation):
    return {"tools": installation.catalog(), "skills": installation.skill_catalog()}


def selected_runtime_file(contribution):
    """Capture the selected runtime file's content and permissions at first use."""
    from dataclasses import asdict
    import json
    import stat
    from aka.contracts.agent_resources import (
        AgentResourceFile,
        WorkspaceResourceContribution,
    )

    if (
        not isinstance(contribution, WorkspaceResourceContribution)
        or contribution.visibility != "runtime"
    ):
        raise TypeError(
            "Agent runtime resource must be a selected runtime contribution"
        )
    data, info = read_regular_file(contribution.source)
    mode = info.st_mode & 0o777
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("Agent runtime resource must be a regular file")
    file = AgentResourceFile(".", mode, hashlib.sha256(data).hexdigest())
    identity = {
        "files": [asdict(file)],
        "directories": [],
        "kind": "file",
        "mode": mode,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return InstalledWorkspaceResource(
        contribution.name,
        contribution.source,
        None,
        "runtime",
        contribution.version,
        digest,
        (file,),
    )
