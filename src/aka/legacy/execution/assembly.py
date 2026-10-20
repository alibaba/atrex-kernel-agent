"""Own selected provider scopes; injected executors retain caller ownership."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from aka.contracts.execution import Executor, validate_executor
from aka.contracts.agent_resources import AgentSkill, WorkspaceResourceContribution
from aka.core.boot import boot, compose
from aka.core.errors import BootFailure
from aka.core.keys import ServiceKey

EXECUTOR = ServiceKey("executor", "Executor", module="aka.contracts.execution", protocol_version=2)
PROFILES_DIR = Path(__file__).with_name("profiles")


class ExecutionComposition:
    def __init__(self, report, executor: Executor):
        self.report = report
        self._executor = executor
        self.disposed = False

    @property
    def executor(self) -> Executor:
        if self.disposed:
            raise RuntimeError("execution composition has been disposed")
        return self._executor

    @property
    def agent_skills(self) -> tuple[AgentSkill, ...]:
        if self.disposed:
            raise RuntimeError("execution composition has been disposed")
        entries = self.report.root.scope.resolve("agent_skills")
        skills = tuple(entry.value for entry in entries)
        if any(not isinstance(skill, AgentSkill) or entry.name != skill.name
               for entry, skill in zip(entries, skills)):
            raise TypeError("invalid Execution Agent Skill contribution")
        return skills

    def dispose(self) -> None:
        if not self.disposed:
            self.disposed = True
            self.report.dispose()

    @property
    def workspace_resources(self) -> tuple[WorkspaceResourceContribution, ...]:
        if self.disposed:
            raise RuntimeError("execution composition has been disposed")
        entries = self.report.root.scope.resolve("workspace_resources")
        values = tuple(entry.value for entry in entries)
        if any(not isinstance(value, WorkspaceResourceContribution) or entry.name != value.name
               for entry, value in zip(entries, values)):
            raise TypeError("invalid Execution workspace resource contribution")
        return values

    def __enter__(self) -> ExecutionComposition:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.dispose()


def compose_execution(*, profile: str = "legacy", profiles_dir: Path = PROFILES_DIR,
                      patch_files: Sequence[Path] = (),
                      patches: Sequence[Mapping[str, Any]] = ()) -> ExecutionComposition:
    composition = compose(profile, profiles_dir=profiles_dir,
        patch_files=patch_files, patches=patches)
    report = boot(composition, tokens=(EXECUTOR,), required_services=(EXECUTOR.name,))
    executor = report.service(EXECUTOR.name)
    try:
        try:
            validate_executor(executor)
        except TypeError as error:
            raise BootFailure(((EXECUTOR.name, str(error)),)) from error
    except BaseException:
        report.dispose()
        raise
    return ExecutionComposition(report, executor)
