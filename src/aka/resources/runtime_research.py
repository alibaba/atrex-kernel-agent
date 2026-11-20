"""Render research Skills for the Campaign's enabled mutation surface."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from aka.contracts.agent_resources import AgentSkill
from aka.core.filesystem import durable_write_text


_SECTIONS = {
    "Directions": "directions",
    "Propose and select a Direction": "directions",
    "Close, defer, or preserve a Direction": "directions",
    "Experiments": "experiments",
    "Record an Experiment": "experiments",
}


def scoped_research_skill(skill: AgentSkill, directory: Path, modules) -> AgentSkill:
    enabled = frozenset(modules)
    if enabled - {"directions", "experiments"}:
        raise ValueError("Unknown research module")
    content = {}
    for path in sorted(skill.source.rglob("*.md")):
        text = path.read_text()
        sections = re.split(r"(?=^## )", text, flags=re.MULTILINE)
        text = "".join(section for section in sections
                       if (_SECTIONS.get(section.splitlines()[0].removeprefix("## ")) in enabled
                           or section.splitlines()[0].removeprefix("## ") not in _SECTIONS)
                       and (enabled or section.splitlines()[0] != "## Research review"))
        for module in ("directions", "experiments"):
            pattern = rf"<!-- {module}:start -->(.*?)<!-- {module}:end -->"
            text = re.sub(pattern, lambda m: m[1] if module in enabled else "", text, flags=re.DOTALL)
        if path.name == "SKILL.md":
            text += "\nEnabled research modules: " + (", ".join(sorted(enabled)) or "none") + ".\n"
        content[path.relative_to(skill.source).as_posix()] = text
    digest = hashlib.sha256()
    for name, text in sorted(content.items()):
        digest.update(name.encode() + b"\0" + text.encode() + b"\0")
    root = directory / ("runtime-records-" + digest.hexdigest()[:16])
    for name, text in content.items():
        target = root / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            durable_write_text(target, text)
        elif target.read_text() != text:
            raise ValueError("Scoped research Skill content changed")
    return AgentSkill(skill.name, root, "Use enabled research operations and exact-commit submission review.")
