"""Adapt Workflow recovery and dependency protection to the shared Agent."""

from pathlib import Path
from aka.agent.provider import Agent, AgentOptions
from . import agent_policy


def unavailable(environment):
    # Recovery state is live host state; a process environment is not authority.
    from aka.execution.recovery import environment_is_blocked
    if environment_is_blocked():
        return (
            75,
            "Remote GPU environment became unavailable; terminated Agent for durable recovery",
        )
    return None


def process_policy(environment):
    from aka.contracts.agent import AgentProcessPolicy
    screen, state = agent_policy.protected_gateway_identity(environment)
    protected = {"ATREX_PROTECTED_GATEWAY_SCREEN": screen, "ATREX_PROTECTED_GATEWAY_STATE_DIR": state}
    return AgentProcessPolicy(
        unavailable=unavailable,
        violation=lambda argv, values: violation(argv, {**values, **protected}),
        environment=protected,
    )


def violation(argv, environment):
    cwd = environment.get("AKA_INSPECTED_PROCESS_CWD")
    return agent_policy.dependency_process_violation(argv, environment, cwd=Path(cwd) if cwd else None)


def prepare_environment(values, explicit_environment=()):
    from importlib.resources import files

    values = dict(values)
    # The matching Workflow runtime is an explicit code dependency. Its parent
    # is a search path, not a grant to that parent's files or sibling checkouts.
    values["PYTHONPATH"] = str(Path(str(files("aka"))).resolve().parent)
    for name in (
        "ATREX_PRIVATE_REFERENCE_DIR",
        "ATREX_SANDBOX_REQUIREMENTS_FILE",
        "AKA_GATEWAY_DEV_DEPENDENCIES",
    ):
        if name not in explicit_environment:
            values.pop(name, None)
    return values


def default_guard():
    # The compatibility constructor explicitly selects the default Execution contribution.
    from aka.execution.resources import contribute
    from aka.agent.resources import selected_runtime_file

    class Selection:
        def __init__(self):
            self.resources = []

        def contribute(self, family, name, value):
            if family == "workspace_resources" and name == "execution-shell-guard":
                self.resources.append(value)

    selection = Selection()
    contribute(selection)
    return selected_runtime_file(selection.resources[0])


def create_agent(options=AgentOptions(), *, shell_guard=None, **kwargs):
    import os
    from aka.execution.processes import spawn_owned_session
    environment = kwargs.get("environment")
    policy = process_policy(os.environ if environment is None else environment)

    return Agent(
        options,
        spawn=spawn_owned_session,
        unavailable=policy.unavailable,
        violation=policy.violation,
        host_environment=prepare_environment,
        policy_environment=lambda: policy.environment,
        shell_guard=default_guard() if shell_guard is None else shell_guard,
        **kwargs,
    )
