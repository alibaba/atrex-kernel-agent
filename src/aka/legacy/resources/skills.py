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


def describe_resource(resource):
    """Fingerprint the complete selected immutable resource, including modes."""
    from dataclasses import asdict
    import json
    from aka.contracts.agent_resources import WorkspaceResourceContribution, InstalledWorkspaceResource
    resource = WorkspaceResourceContribution(**asdict(resource))
    source = resource.source
    paths = (source,) if source.is_file() else tuple(p for p in sorted(source.rglob("*"))
        if not {".git", "__pycache__", ".pytest_cache"}.intersection(p.relative_to(source).parts)
        and p.suffix not in {".pyc", ".pyo"})
    files = tuple(AgentResourceFile("." if p == source else p.relative_to(source).as_posix(),
        p.stat().st_mode & 0o777, hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths if p.is_file())
    if not files:
        raise ValueError(f"workspace resource has no files: {resource.name}")
    directories = [(p.relative_to(source).as_posix(), p.stat().st_mode & 0o777)
                   for p in paths if p.is_dir()]
    identity = {"files": [asdict(f) for f in files], "directories": directories,
                "kind": "file" if source.is_file() else "directory", "mode": source.stat().st_mode & 0o777}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return InstalledWorkspaceResource(resource.name, source, resource.destination,
        resource.visibility, resource.version, digest, files)
