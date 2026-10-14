"""Fingerprint a complete module contribution independently of its installer."""
from __future__ import annotations

import hashlib

from aka.contracts.agent_resources import AgentSkill, AgentResourceFile, InstalledAgentSkill


def skill_files(skill: AgentSkill):
    # Revalidate: a source may have changed since the composition was created.
    AgentSkill(skill.name, skill.source, skill.description)
    return tuple(path for path in sorted(skill.source.rglob("*"))
                 if path.is_file()
                 and not {".git", "__pycache__", ".pytest_cache"}.intersection(path.parts)
                 and path.suffix not in {".pyc", ".pyo"})


def fingerprint(skill: AgentSkill) -> str:
    digest = hashlib.sha256()
    for path in skill_files(skill):
        digest.update(path.relative_to(skill.source).as_posix().encode() + b"\0")
        digest.update(str(path.stat().st_mode & 0o777).encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def describe_skill(skill: AgentSkill) -> InstalledAgentSkill:
    return InstalledAgentSkill(skill.name, skill.source.resolve(), skill.description,
        fingerprint(skill), tuple(AgentResourceFile(
            path.relative_to(skill.source).as_posix(), path.stat().st_mode & 0o777,
            hashlib.sha256(path.read_bytes()).hexdigest()) for path in skill_files(skill)))
