import json


from aka.agent.provider import Agent
from aka.contracts.agent import AgentToolBinding
from aka.contracts.filesystem import FilesystemAccess, FilesystemMount
from aka.workflow.light.long_horizon.session import LongSessionRunner


def test_light_per_turn_fields_are_declared_before_scoped_invocation(tmp_path):
    calls = []
    def process(command, *, cwd, timeout, env):
        calls.append(env)
        return json.dumps({'type': 'result', 'subtype': 'success', 'usage': {
            'input_tokens': 1, 'output_tokens': 1}}) + '\n', '', 0, False

    agent = Agent(process_runner=process, environment={'HOME': str(tmp_path)})
    class Handler:
        def handle(self, payload, context):
            return {}
    try:
        with agent.tool_session(tmp_path, access=FilesystemAccess((FilesystemMount(tmp_path, tmp_path, True),)),
                                bindings=(AgentToolBinding('test.probe', Handler()),)) as grant:
            runner = LongSessionRunner(agent=agent, usage_policy=lambda _: 2)
            result = runner.run(tmp_path, 'test', telemetry_environment=dict(grant.environment))
            assert result.exit_status == 0
            assert calls[0]['ATREX_AGENT_STAGE_NAME'] == 'Episode'
            assert calls[0]['ATREX_OUTPUT_LIMIT_RECOVERY_SCOPE']
    finally:
        agent.close()
