"""Application binding and explicit inheritance of the selected Executor."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import time

from aka.legacy.execution.assembly import compose_execution
from aka.legacy.execution.legacy.executor import LegacyExecutor
from aka.legacy.execution.legacy.runtime import sandbox_invocation
from aka.contracts.execution import (
    CommandSpec, ExecutionBudget, Executor, JobResult, UnsupportedExecution, validate_executor,
)

_current_executor = ContextVar("aka_application_executor", default=None)
_current_selection = ContextVar("aka_application_execution_selection", default=None)
_current_agent_skills = ContextVar("aka_application_agent_skills", default=None)
_explicit_health_configuration = ContextVar("aka_explicit_health_configuration", default=None)
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


def explicit_health_configuration():
    return _explicit_health_configuration.get()


def record_explicit_health_configuration():
    if _explicit_health_configuration.get() is not None:
        _explicit_health_configuration.set(True)


@contextmanager
def application_transport_options(*, health_command_explicit):
    token = _explicit_health_configuration.set(health_command_explicit)
    try:
        yield
    finally:
        _explicit_health_configuration.reset(token)


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
def application_execution(executor=None, selection=None, *, agent_skills=()):
    current = _current_executor.get()
    if executor is None and current is not None:
        if selection is not None and dict(selection) != selected_execution():
            raise ValueError("cannot change execution selection inside an active application scope")
        yield current
        return
    selected = dict(selected_execution() if selection is None else selection)
    arguments = dict(selected)
    if "profiles_dir" in arguments:
        arguments["profiles_dir"] = Path(arguments["profiles_dir"]).expanduser().resolve()
        selected["profiles_dir"] = str(arguments["profiles_dir"])
    composition = None
    if executor is None:
        composition = compose_execution(**arguments)
        executor = composition.executor
        try:
            agent_skills = composition.agent_skills
        except BaseException:
            composition.dispose()
            raise
    else:
        validate_executor(executor)
    # Materialize caller resources before changing the active context.
    skills = tuple(agent_skills)
    executor_token = _current_executor.set(executor)
    selection_token = _current_selection.set(selected)
    skills_token = _current_agent_skills.set(skills)
    try:
        yield executor
    finally:
        _current_executor.reset(executor_token)
        _current_selection.reset(selection_token)
        _current_agent_skills.reset(skills_token)
        if composition is not None:
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


class _HostExecutionControl:
    def __init__(self, command, hooks, budget):
        self.command = command
        self.hooks = hooks
        self.budget = budget
        self.deadline = time.monotonic() + budget.wall_timeout
        self.failure = None
        self.checkpoints = 0

    def checkpoint(self):
        self.checkpoints += 1
        if self.failure is not None:
            raise self.failure
        try:
            self.hooks.check_environment()
            if self.command.cancel_event is not None and self.command.cancel_event.is_set():
                raise subprocess.SubprocessError("sandbox command cancelled")
            if time.monotonic() >= self.deadline:
                raise subprocess.TimeoutExpired(list(self.command.command), self.budget.wall_timeout)
        except BaseException as error:
            self.failure = error
            raise


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
    if isinstance(executor, LegacyExecutor):
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
    executor.capabilities.require(command)
    explicit_health = request.health_command_explicit
    if explicit_health is None:
        explicit_health = bool(request.health_command) or "ATREX_SANDBOX_HEALTH_COMMAND" in request.environment
    elif not explicit_health and "ATREX_SANDBOX_HEALTH_COMMAND" in request.environment:
        # A host-generated default is implicit; an effective per-call override is not.
        explicit_health = request.environment["ATREX_SANDBOX_HEALTH_COMMAND"] != request.health_command
    environment = request.environment
    endpoint = request.ssh or request.profile or request.url
    inherited_endpoint = not endpoint and any(environment.get(name) for name in (
        "ATREX_SANDBOX_SSH", "ATREX_SANDBOX_PROFILE", "ATREX_SANDBOX_URL"))
    if (endpoint or inherited_endpoint or request.ssh_init
            or environment.get("ATREX_SANDBOX_SSH_INIT") or explicit_health):
        raise UnsupportedExecution("legacy launcher settings require a legacy-compatible execution provider")
    wall_timeout = request.wall_timeout
    if wall_timeout is None:
        wall_timeout = request.timeout + hooks.queue_wait_grace(request.environment) + 240
    budget = ExecutionBudget(wall_timeout)
    control = _HostExecutionControl(command, hooks, budget)
    command = replace(command, budget=budget, control=control)
    executor.capabilities.require(command)
    control.checkpoint()
    initial_checkpoints = control.checkpoints
    try:
        result = executor.run(command)
    except BaseException:
        if control.failure is not None:
            raise control.failure
        raise
    if control.checkpoints == initial_checkpoints:
        raise TypeError("executor advertised host controls but did not checkpoint")
    control.checkpoint()
    return _completed_process(command, result)
