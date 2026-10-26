"""Compatibility constructor for the shared Agent with the existing Workflow host."""
from aka.agent.provider import Agent, AgentOptions

class LegacyAgent(Agent):
    def __init__(self, *, process_runner=None, registry=None, session_name="", options=AgentOptions()):
        from aka.legacy.workflow.agent_host import default_guard, unavailable, violation, prepare_environment
        from aka.legacy.execution.processes import spawn_owned_session
        super().__init__(options, process_runner=process_runner, registry=registry,
                         session_name=session_name, shell_guard=default_guard(),
                         spawn=spawn_owned_session, unavailable=unavailable, violation=violation,
                         host_environment=prepare_environment)
