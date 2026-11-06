"""Harness projections of public contributions; independent of feature modules."""
from __future__ import annotations

import json
from pathlib import Path

from aka.contracts.contributions import CATALOG_SCHEMA, render_instructions


def codex_instructions(catalog):
    bindings = {row["id"]: f"`python3 tools/aka.py call {row['id']} --input scratch/request.json`"
                for row in catalog["tools"]}
    instructions = render_instructions(catalog, bindings)
    if not catalog["tools"] and not instructions:
        return ""
    return (
        "## Selected AKA module contributions\n\n"
        "Use `python3 tools/aka.py list` to inspect the current catalog. Calls exchange JSON; "
        "a successful transport can return a failed measurement, so inspect the result status.\n\n"
        + json.dumps(catalog["tools"], sort_keys=True) + "\n\n" + instructions
    )


def project_command(command, environment):
    """Deliver one frozen contribution snapshot before launching a managed Session."""
    raw = environment.get("ATREX_AKA_CONTRIBUTIONS")
    if not raw or not command:
        return command
    catalog = json.loads(raw)
    if catalog.get("schema") != CATALOG_SCHEMA:
        raise ValueError("unsupported contribution catalog")
    # DSH's plugin projects this same snapshot through its native registries.
    if Path(command[0]).name == "codex" and command[1:2] == ["exec"]:
        text = codex_instructions(catalog)
        if text:
            return [*command[:-1], command[-1] + "\n\n" + text]
    return command
