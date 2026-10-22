"""Adapt Workflow recovery and dependency protection to the shared Agent."""

from pathlib import Path
from aka.agent.provider import Agent, AgentOptions
from . import agent_policy


def unavailable(environment):
    state = environment.get("ATREX_ENVIRONMENT_STATE_FILE", "")
    if state and Path(state).is_file():
        return (
            75,
            "Remote GPU environment became unavailable; terminated Agent for durable recovery",
        )
    return None


def violation(argv, environment):
    return agent_policy.dependency_process_violation(argv, environment)


def prepare_environment(values, explicit_environment=()):
    values = dict(values)
    for name in (
        "ATREX_PRIVATE_REFERENCE_DIR",
        "ATREX_SANDBOX_REQUIREMENTS_FILE",
        "AKA_GATEWAY_DEV_DEPENDENCIES",
    ):
        if name not in explicit_environment:
            values.pop(name, None)
    screen, state = agent_policy.protected_gateway_identity(values)
    values.update(
        ATREX_PROTECTED_GATEWAY_SCREEN=screen, ATREX_PROTECTED_GATEWAY_STATE_DIR=state
    )
    return values


def default_guard():
    # The compatibility constructor explicitly selects the default Execution contribution.
    from aka.legacy.execution.resources import contribute
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
    from aka.legacy.execution.processes import spawn_owned_session

    return Agent(
        options,
        spawn=spawn_owned_session,
        unavailable=unavailable,
        violation=violation,
        host_environment=prepare_environment,
        shell_guard=default_guard() if shell_guard is None else shell_guard,
        **kwargs,
    )
