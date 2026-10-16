"""Adapt selected Core Skill contributions to the existing workspace installer."""
from __future__ import annotations

from pathlib import Path

from aka.contracts.agent_resources import AgentSkill, AgentResourceInstallation
from aka.legacy.task.durable import durable_write_json
from .skills import fingerprint as _fingerprint
from .skills import describe_skill


class AgentSkillsRegistryMixin:
    """Keep old plugin locks intact while adding selected module-owned Skills."""

    _agent_skills = ()
    _installation = None

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
        self._installation = None

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

    def resource_snapshot(self):
        """Export the selected plugin and module Skill locks without workspace IO."""
        return {"plugins": self.snapshot(), "agent_skills": self._agent_skills_snapshot()}

    def check_resource_snapshot(self, snapshot):
        """Validate frozen resources before a checkpoint creates its destination."""
        from plugin_runtime.schema import PluginError
        from plugin_runtime.registry import PluginRegistry
        current = {"plugins": PluginRegistry(self.plugin_dir, plugins=self.plugin_paths).snapshot(),
                   "agent_skills": self._agent_skills_snapshot()}
        if snapshot != current:
            raise PluginError("plugin_changed", "selected tool or Skill resources differ from the frozen inputs")

    def check_lock(self, workspace):
        from plugin_runtime.registry import STATE_DIR
        from plugin_runtime.schema import PluginError, read_json
        super().check_lock(workspace)
        lock = Path(workspace) / STATE_DIR / "agent-skills.json"
        if lock.is_symlink() or (lock.exists() and not lock.is_file()):
            raise PluginError("mount_conflict", "Agent Skill lock must be a local file")
        if lock.exists() and read_json(lock) != self._agent_skills_snapshot():
            raise PluginError("plugin_changed", "selected Agent Skills changed; restore locked resources or use a new workspace")

    def install(self, workspace):
        from plugin_runtime.registry import STATE_DIR
        workspace = Path(workspace)
        self.check_lock(workspace)
        super().install(workspace)
        lock = workspace / STATE_DIR / "agent-skills.json"
        if self._agent_skills and not lock.exists():
            durable_write_json(lock, self._agent_skills_snapshot(), indent=2)
        self._installation = self._resource_result(workspace.resolve())
        return self._installation

    def _resource_result(self, workspace=None):
        skills = [*super().skill_catalog(), *(
            {"id": f"module.{skill.name}", "plugin": "selected-modules", "name": skill.name,
             "version": "1", "path": str(skill.source.resolve()), "description": skill.description}
            for skill in self._agent_skills
        )]
        instructions = ""
        if self._agent_skills:
            lines = ["## Selected module Skills"]
            lines.extend(f"- Skill `{skill.name}`: {skill.description} Read `{skill.source.resolve()}/SKILL.md`."
                         for skill in self._agent_skills)
            instructions = "\n\n" + "\n".join(lines)
        return AgentResourceInstallation(workspace, tuple(super().catalog()), tuple(skills), instructions,
                                         tuple(describe_skill(skill) for skill in self._agent_skills))

    def resource_result(self, workspace=None):
        """Return discovery without mutating a workspace; preserve pre-install list."""
        if workspace is not None:
            workspace = Path(workspace)
            self.check_lock(workspace)
            if self._installation is not None and self._installation.workspace == workspace.resolve():
                return self._installation
            # A new CLI process has checked the existing locks and can reconstruct
            # the same discovery result without re-installing or modifying files.
            from plugin_runtime.registry import STATE_DIR
            if (workspace / STATE_DIR / "lock.json").is_file():
                return self._resource_result(workspace.resolve())
            return self._resource_result()
        return self._installation or self._resource_result()

    def catalog(self):
        return self.resource_result().catalog()

    def skill_catalog(self):
        return self.resource_result().skill_catalog()

    def instructions(self, phase, **values):
        return super().instructions(phase, **values) + self.resource_result().module_instructions
