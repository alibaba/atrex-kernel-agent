"""Read explicitly selected module resources from the current composition scope."""
from contextvars import ContextVar
from dataclasses import asdict
import json
import os
from pathlib import Path

from aka.contracts.agent_resources import AgentSkill, WorkspaceResourceContribution

_selected_scope = ContextVar("aka_workspace_resource_scope", default=None)
SELECTION_ENV = "ATREX_WORKSPACE_RESOURCES"


def selected_resources():
    scope = _selected_scope.get()
    if scope is not None:
        values = []
        for kind, expected in (("workspace_resources", WorkspaceResourceContribution), ("agent_skills", AgentSkill)):
            entries = scope.resolve(kind)
            if any(not isinstance(entry.value, expected) or entry.name != entry.value.name for entry in entries):
                raise TypeError(f"invalid {kind} contribution")
            values.append(tuple(entry.value for entry in entries))
        return tuple(values)
    try:
        value = json.loads(os.environ.get(SELECTION_ENV, '{"resources":[],"skills":[]}'))
        if not isinstance(value, dict) or set(value) != {"resources", "skills"}:
            raise ValueError("invalid inherited workspace resource selection")
        if not all(isinstance(value[k], list) for k in value):
            raise ValueError("workspace resource selection must contain arrays")
        return (tuple(WorkspaceResourceContribution(**{**row, "source": Path(row["source"])}) for row in value["resources"]),
                tuple(AgentSkill(**{**row, "source": Path(row["source"])}) for row in value["skills"]))
    except (TypeError, KeyError, ValueError) as exc:
        raise ValueError("invalid inherited workspace resource declaration") from exc


def selection_environment(resources, skills):
    def row(value):
        return {**asdict(value), "source": str(value.source)}
    return json.dumps({"resources": [row(r) for r in resources], "skills": [row(s) for s in skills]}, sort_keys=True)
