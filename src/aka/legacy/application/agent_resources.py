"""Adapt selected Core Skill contributions to the existing workspace installer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from aka.contracts.agent_resources import AgentSkill


def _fingerprint(skill: AgentSkill) -> str:
    # Revalidate before install/lock checks: sources may have changed since boot.
    AgentSkill(skill.name, skill.source, skill.description)
    digest = hashlib.sha256()
    for path in sorted(skill.source.rglob("*")):
        if not path.is_file() or {".git", "__pycache__", ".pytest_cache"}.intersection(path.parts):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        digest.update(path.relative_to(skill.source).as_posix().encode() + b"\0")
        digest.update(str(path.stat().st_mode & 0o777).encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


class AgentSkillsRegistryMixin:
    """Keep old plugin locks intact while adding selected module-owned Skills."""

    _agent_skills = ()

    def bind_agent_skills(self, skills) -> None:
        from plugin_runtime.schema import PluginError
        values = tuple(skills)
        for skill in values:
            if not isinstance(skill, AgentSkill):
                raise TypeError("invalid Agent Skill contribution")
            _fingerprint(skill)
        names = [skill.name for skill in values]
        if len(names) != len(set(names)):
            raise PluginError("invalid_config", "duplicate module Agent Skill")
        previous = self._agent_skills
        self._agent_skills = values
        try:
            self.mounts()  # Validate legacy/module conflicts before installation.
        except BaseException:
            self._agent_skills = previous
            raise

    def mounts(self):
        from plugin_runtime.registry import SKILL_ROOTS
        from plugin_runtime.schema import PluginError
        mounts = super().mounts()
        for skill in self._agent_skills:
            for root in SKILL_ROOTS:
                name = f"{root}/{skill.name}"
                candidate = Path(name)
                if any(candidate.is_relative_to(Path(existing)) or Path(existing).is_relative_to(candidate)
                       for existing in mounts):
                    raise PluginError("invalid_config", f"conflicting Agent Skill: {skill.name}")
                mounts[name] = skill.source.resolve()
        return mounts

    def _agent_skills_snapshot(self):
        return {"api_version": 1, "skills": [
            {"name": skill.name, "source": str(skill.source.resolve()),
             "description": skill.description, "fingerprint": _fingerprint(skill)}
            for skill in self._agent_skills
        ]}

    def check_lock(self, workspace):
        from plugin_runtime.registry import STATE_DIR
        from plugin_runtime.schema import PluginError, read_json
        super().check_lock(workspace)
        lock = Path(workspace) / STATE_DIR / "agent-skills.json"
        if lock.exists() and read_json(lock) != self._agent_skills_snapshot():
            raise PluginError("plugin_changed", "selected Agent Skills changed; restore locked resources or use a new workspace")
        self._check_workspace_mounts(Path(workspace))

    def _check_workspace_mounts(self, workspace):
        from plugin_runtime.registry import STATE_DIR
        from plugin_runtime.schema import PluginError
        # link_runtime calls check_lock before touching discovery directories.
        # Reject redirects and occupied destinations at that initial barrier.
        state = workspace / STATE_DIR
        if state.is_symlink() or (state.exists() and not state.is_dir()):
            raise PluginError("mount_conflict", "Agent resource state must be a workspace directory")
        for name, source in self.mounts().items():
            destination = workspace / name
            if any(parent.is_symlink() for parent in destination.parents
                   if parent != workspace and workspace in parent.parents):
                raise PluginError("mount_conflict", f"Agent resource parent is a symlink: {name}")
            if any(parent.exists() and not parent.is_dir() for parent in destination.parents
                   if parent != workspace and workspace in parent.parents):
                raise PluginError("mount_conflict", f"Agent resource parent is not a directory: {name}")
            if (destination.exists() or destination.is_symlink()) and not (
                    destination.is_symlink() and destination.resolve() == source):
                raise PluginError("mount_conflict", f"Agent resource would replace {destination}")

    def install(self, workspace):
        from plugin_runtime.registry import STATE_DIR
        workspace = Path(workspace)
        self.check_lock(workspace)
        super().install(workspace)
        lock = workspace / STATE_DIR / "agent-skills.json"
        if self._agent_skills and not lock.exists():
            lock.write_text(json.dumps(self._agent_skills_snapshot(), indent=2) + "\n")

    def skill_catalog(self):
        return [*super().skill_catalog(), *(
            {"id": f"module.{skill.name}", "plugin": "selected-modules", "name": skill.name,
             "version": "1", "path": str(skill.source.resolve()), "description": skill.description}
            for skill in self._agent_skills
        )]

    def instructions(self, phase, **values):
        text = super().instructions(phase, **values)
        if self._agent_skills:
            lines = ["## Selected module Skills"]
            lines.extend(f"- Skill `{skill.name}`: {skill.description} Read `{skill.source.resolve()}/SKILL.md`."
                         for skill in self._agent_skills)
            text += "\n\n" + "\n".join(lines)
        return text
