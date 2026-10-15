"""Own selection, nesting and disposal of the active Executor."""
from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
from pathlib import Path
import subprocess

from aka.legacy.execution.assembly import compose_execution
from aka.legacy.execution.legacy.executor import LegacyExecutor
from aka.legacy.execution.legacy.runtime import sandbox_invocation
from aka.contracts.execution import (
    CommandSpec, JobResult, Executor, UnsupportedExecution,
)

_current_executor = ContextVar("aka_execution_executor", default=None)
_current_selection = ContextVar("aka_execution_scope_selection", default=None)
_current_agent_skills = ContextVar("aka_execution_agent_skills", default=None)
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
    except BaseException:
        composition.dispose()
        raise
    executor_token = _current_executor.set(executor)
    selection_token = _current_selection.set(selected)
    skills_token = _current_agent_skills.set(skills)
    try:
        yield executor
    finally:
        _current_executor.reset(executor_token)
        _current_selection.reset(selection_token)
        _current_agent_skills.reset(skills_token)
        composition.dispose()


def selected_agent_skills():
    """Borrow the active composition; direct host callers use their selection."""
    skills = _current_agent_skills.get()
    if skills is not None:
        return skills
    arguments = dict(selected_execution())
    if "profiles_dir" in arguments:
        arguments["profiles_dir"] = Path(arguments["profiles_dir"]).expanduser().resolve()
    with compose_execution(**arguments) as composition:
        return composition.agent_skills


def _completed_process(command, result):
    if not isinstance(result, JobResult):
        raise TypeError("Executor.run must return JobResult")
    if type(result.transport_returncode) is not int or (
        result.exit_code is not None and type(result.exit_code) is not int
    ):
        raise TypeError("execution return codes must be integers or an unknown command exit")
    if result.output_scope not in ("transport", "command"):
        raise ValueError("unknown execution output scope")
    if result.transport_returncode != 0:
        returncode = result.transport_returncode
    elif result.exit_code is not None:
        returncode = result.exit_code
    elif result.output_scope == "command":
        raise ValueError("command-scoped output requires a known command exit code")
    else:
        returncode = result.transport_returncode
    return subprocess.CompletedProcess(list(command.command), returncode, result.stdout, result.stderr)


def run_sandbox(executor: Executor, request, *, hooks) -> subprocess.CompletedProcess[str]:
    command = CommandSpec(workspace=request.workspace, command=request.command,
        timeout=request.timeout, environment=request.environment, hardware=request.hardware,
        artifacts=request.sync, operation="preflight" if request.preflight else request.gateway_kind,
        cancel_event=request.cancel_event)
    if not isinstance(executor, LegacyExecutor):
        if any((request.profile, request.url, request.ssh, request.ssh_init,
                request.health_command, request.wall_timeout is not None)):
            raise UnsupportedExecution(
                "selected executor owns endpoint and recovery configuration; "
                "omit legacy sandbox options"
            )
        executor.capabilities.require(command)
        return _completed_process(command, executor.run(command))
    with sandbox_invocation(request, hooks) as invocation:
        result = executor.run(command)
        if not isinstance(result, JobResult):
            raise TypeError("Executor.run must return JobResult")
        if invocation.result is not None:
            original = invocation.result
            if (original.returncode, original.stdout, original.stderr) == (
                result.transport_returncode, result.stdout, result.stderr
            ):
                return original
        return _completed_process(command, result)
