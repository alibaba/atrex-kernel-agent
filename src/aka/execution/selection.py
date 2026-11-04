"""Own selection, nesting and disposal of the active Executor."""
from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
from pathlib import Path
import subprocess

from aka.execution.assembly import compose_execution
from aka.contracts.execution import (
    CommandSpec, JobResult, Executor, UnsupportedExecution,
)

_current_executor = ContextVar("aka_execution_executor", default=None)
_current_selection = ContextVar("aka_execution_scope_selection", default=None)
_current_agent_skills = ContextVar("aka_execution_agent_skills", default=None)
_current_workspace_resources = ContextVar("aka_execution_workspace_resources", default=None)
SELECTION_ENV = "AKA_EXECUTION_SELECTION"


def read_execution_patches(paths):
    patches = []
    for path in paths:
        value = json.loads(Path(path).expanduser().resolve().read_text())
        if (not isinstance(value, dict) or value.get("api_version") != 1
                or not isinstance(value.get("patches"), list)):
            raise ValueError("execution patch must contain api_version 1 and a patches list")
        patches.extend(value["patches"])
    return patches


def selected_execution():
    value = _current_selection.get()
    if value is not None:
        return value
    serialized = os.environ.get(SELECTION_ENV)
    if not serialized:
        return {"profile": "legacy"}
    value = json.loads(serialized)
    if (not isinstance(value, dict) or set(value) - {"profile", "profiles_dir", "patches"}
            or not isinstance(value.get("profile"), str)
            or ("profiles_dir" in value and not isinstance(value["profiles_dir"], str))
            or ("patches" in value and not isinstance(value["patches"], list))):
        raise ValueError("invalid inherited execution selection")
    return value


def execution_environment(environment):
    environment = dict(environment)
    selection = selected_execution()
    if selection != {"profile": "legacy"} or SELECTION_ENV in environment:
        environment[SELECTION_ENV] = json.dumps(selection, sort_keys=True)
    return environment


@contextmanager
def execution_scope(*, selection=None):
    current = _current_executor.get()
    if current is not None:
        if selection is not None and dict(selection) != selected_execution():
            raise ValueError("cannot change execution selection inside an active execution scope")
        yield current
        return
    selected = dict(selected_execution() if selection is None else selection)
    arguments = dict(selected)
    if "profiles_dir" in arguments:
        arguments["profiles_dir"] = Path(arguments["profiles_dir"]).expanduser().resolve()
        selected["profiles_dir"] = str(arguments["profiles_dir"])
    composition = compose_execution(**arguments)
    try:
        executor = composition.executor
        skills = tuple(composition.agent_skills)
        resources = tuple(composition.workspace_resources)
    except BaseException:
        composition.dispose()
        raise
    executor_token = _current_executor.set(executor)
    selection_token = _current_selection.set(selected)
    skills_token = _current_agent_skills.set(skills)
    resources_token = _current_workspace_resources.set(resources)
    try:
        yield executor
    finally:
        _current_executor.reset(executor_token)
        _current_selection.reset(selection_token)
        _current_agent_skills.reset(skills_token)
        _current_workspace_resources.reset(resources_token)
        composition.dispose()


def selected_agent_skills():
    """Borrow contributions from the active Executor lifetime."""
    return _current_agent_skills.get() or ()


def selected_workspace_resources():
    return _current_workspace_resources.get() or ()


def _completed_process(command, result):
    if not isinstance(result, JobResult):
        raise TypeError("Executor.run must return JobResult")
    return subprocess.CompletedProcess(list(command.command), result.returncode, result.stdout, result.stderr)


def run_sandbox(executor: Executor, request, *, hooks) -> subprocess.CompletedProcess[str]:
    command = CommandSpec(workspace=request.workspace, command=request.command,
        timeout=request.timeout, environment=request.environment, hardware=request.hardware,
        artifacts=request.sync, operation="preflight" if request.preflight else request.gateway_kind,
        cancel_event=request.cancel_event)
    # CLI-only endpoint adaptation is owned by the backend, not by its consumers.
    binder = getattr(executor, "bind_sandbox", None)
    if binder is not None:
        executor = binder(request, hooks)
    elif any((request.profile, request.url, request.ssh, request.ssh_init,
              request.health_command, request.wall_timeout is not None)):
        raise UnsupportedExecution("selected executor owns endpoint configuration; omit legacy sandbox options")
    result = executor.run(command)
    result.require_transport()
    return _completed_process(command, result)
