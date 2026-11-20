"""Scope measurement guidance to a Workflow's public Runtime surface."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from aka.contracts.agent_resources import AgentSkill
from aka.contracts.agate_tools import AGATE_TOOLS, normalize_agate_tools
from aka.core.filesystem import durable_write_text


def scoped_measurement_skill(skill: AgentSkill, directory: Path, enabled_kinds) -> AgentSkill:
    """Install only enabled operation sections; never mount the full provider manual."""
    enabled = normalize_agate_tools(enabled_kinds)
    requests = (skill.source / "references/requests.md").read_text()
    sections = re.split(r"(?=^## )", requests, flags=re.MULTILINE)
    requests = "".join(section for section in sections
                       if section.splitlines()[0].removeprefix("## ") not in AGATE_TOOLS
                       or section.splitlines()[0].removeprefix("## ") in enabled)
    selection = ("\n## Enabled GPU measurement tools\n\n" + "\n".join(
        f"- `{kind}`: [request example](references/requests.md#{kind})." for kind in enabled)
        if enabled else "\nNo GPU measurement tools are enabled in this Campaign.")
    selection += "\nUse `python3 tools/sandbox.py --help` for the complete enabled Runtime tool list."
    content = {"SKILL.md": (skill.source / "SKILL.md").read_text() + selection + "\n",
               "references/requests.md": requests}
    # Immutable guidance supports resumed Sessions and discovery in separate
    # processes. Existing workspace resource locks still apply.
    digest = hashlib.sha256()
    for name, text in sorted(content.items()):
        digest.update(name.encode() + b"\0" + text.encode() + b"\0")
    root = directory / ("gpu-measurement-" + digest.hexdigest()[:16])
    for name, text in content.items():
        target = root / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            durable_write_text(target, text)
        elif target.read_text() != text:
            raise ValueError("Scoped measurement Skill content changed")
    return AgentSkill(skill.name, root, "Use the Campaign-enabled Runtime measurement tools and their request help.")
