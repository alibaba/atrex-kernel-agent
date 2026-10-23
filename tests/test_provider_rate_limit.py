"""Provider failures must not turn infrastructure retries into optimization rounds."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch
from concurrent.futures import CancelledError

from aka.contracts.agent import TokenUsage
from aka.light.agent.runtime.adapter import PiAdapter, ClaudeAdapter
from aka.light.agent.runtime.provider_errors import rate_limit_code
from aka.light.execution.retry import InfrastructureUnavailable
from aka.light.workflow.session import LongSessionRunner

THROTTLED = '429 event:error\ndata:{"code":"Throttling","message":"You exceeded your current quota, please check your plan and billing details."}'


def pi_message(error=None, role="assistant"):
    return json.dumps({"type": "message_end", "message": {
        "role": role, "stopReason": "error" if error else "stop",
        "errorMessage": error, "content": []}})


class ClassificationTests(TestCase):
    def test_structured_throttling_overrides_generic_billing_prose(self):
        self.assertEqual(rate_limit_code(THROTTLED), "Throttling")
        self.assertEqual(rate_limit_code('429 Too many requests'), "rate_limit")

    def test_permanent_quota_is_not_retried(self):
        self.assertIsNone(rate_limit_code('429 {"code":"insufficient_quota"}'))
        self.assertIsNone(rate_limit_code('429 billing balance exhausted'))
        self.assertIsNone(rate_limit_code('401 invalid api key'))

    def test_pi_terminal_error_is_detected_despite_exit_zero(self):
        self.assertEqual(PiAdapter().classify_terminal_failure(pi_message(THROTTLED), "id"),
                         ("rate_limit", "Throttling"))

    def test_pi_recovered_error_and_tool_output_do_not_fail_session(self):
        adapter = PiAdapter()
        self.assertEqual(adapter.classify_terminal_failure(pi_message(THROTTLED) + '\n' + pi_message(), "id"), (None, None))
        self.assertEqual(adapter.classify_terminal_failure(pi_message(THROTTLED, "toolResult"), "id"), (None, None))

    def test_pi_other_terminal_errors_are_not_invalid_handoffs(self):
        self.assertEqual(PiAdapter().classify_terminal_failure(pi_message('401 unauthorized'), "id")[0], "provider_error")

    def test_pi_aborted_response_is_never_retried(self):
        event = json.loads(pi_message(THROTTLED))
        event['message']['stopReason'] = 'aborted'
        self.assertEqual(PiAdapter().classify_terminal_failure(json.dumps(event), "id"),
                         ("provider_error", "pi_assistant_aborted"))

    def test_claude_terminal_error_and_recovery(self):
        error = {"type": "assistant", "session_id": "id", "isApiErrorMessage": True,
                 "message": {"content": [{"type": "text", "text": THROTTLED}]}}
        failure = {"type": "result", "session_id": "id", "is_error": True}
        stream = json.dumps(error) + '\n' + json.dumps(failure)
        self.assertEqual(ClaudeAdapter().classify_terminal_failure(stream, "id"), ("rate_limit", "Throttling"))
        success = {"type": "assistant", "session_id": "id", "message": {"content": []}}
        self.assertEqual(ClaudeAdapter().classify_terminal_failure(stream + '\n' + json.dumps(success), "id"), (None, None))
        error['parent_tool_use_id'] = 'child'
        self.assertEqual(ClaudeAdapter().classify_terminal_failure(json.dumps(error), "id"), (None, None))


class FakeAgent:
    def __init__(self, workspace, errors, backend="pi"):
        self.workspace, self.errors, self.backend = workspace, list(errors), backend
        self.calls, self.sessions = [], []

    def open_session(self, runtime_id, workspace, *, session_id, environment):
        agent = self
        class Session:
            supports_resume = runtime_id == "claude"
            closed = False
            def __init__(self):
                self.environment = environment or {}
            def close(self):
                self.closed = True
            def invoke(self, prompt, **kwargs):
                agent.calls.append((session_id, prompt, kwargs))
                self.assert_draft()
                error = agent.errors.pop(0)
                if error is None:
                    (workspace / 'handoff.json').write_text('{"status":"pivot"}')
                if runtime_id == 'pi':
                    stdout = pi_message(error)
                else:
                    stdout = json.dumps({"type": "assistant", "session_id": session_id,
                        "isApiErrorMessage": error is not None,
                        "message": {"content": [{"type": "text", "text": error or "done"}]}})
                return SimpleNamespace(stdout=stdout, stderr='', exit_status=0, timed_out=False,
                    session_id=session_id, terminal_usage=TokenUsage(7,3,0,0,10,'exact'),
                    events=(), capabilities=None, observation_errors=(), resume_usage_qualified=False)
            def assert_draft(self):
                assert (workspace / 'kernel.py').read_text() == 'saved draft'
        native = Session()
        self.sessions.append(native)
        return native


class RecoveryTests(TestCase):
    def run_case(self, errors, backend="pi", token_budget=None, cancelled=False):
        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / 'kernel.py').write_text('saved draft')
            agent = FakeAgent(workspace, errors, backend)
            runner = LongSessionRunner(agent=agent, usage_policy=lambda turn: 10, agent_cli=backend)
            usage = []
            with patch('aka.light.workflow.session.Event.wait', return_value=cancelled) as wait:
                result = runner.run(workspace, 'original task', handoff_path=workspace/'handoff.json',
                    handoff_resumes=0, completion_check=lambda handoff: '',
                    remaining_token_budget=token_budget, record_usage=usage.append)
            return result, agent, usage, wait.call_args_list

    def test_pi_retries_same_draft_without_handoff_resume_budget(self):
        result, agent, usage, waits = self.run_case([THROTTLED] * 7 + [None])
        self.assertEqual(result.exit_status, 0)
        self.assertEqual(result.handoff.status, 'pivot')
        self.assertEqual(result.resume_count, 0)
        self.assertEqual(result.tokens, 80)
        self.assertEqual(usage, list(range(10, 81, 10)))
        self.assertEqual([call.args[0] for call in waits], [5,10,20,40,60,60,60])
        self.assertTrue(all(s.closed for s in agent.sessions))
        self.assertEqual({i.failure_kind for i in result.invocations[:-1]}, {'rate_limit'})
        self.assertEqual(result.invocations[-1].purpose, 'rate_limit_recovery')

    def test_claude_preserves_native_session_without_consuming_resume_count(self):
        result, agent, _, _ = self.run_case([THROTTLED, None], 'claude')
        self.assertEqual(len(agent.sessions), 1)
        self.assertTrue(agent.calls[1][2]['resume'])
        self.assertEqual(result.resume_count, 0)

    def test_quota_failure_is_terminal_with_precise_diagnosis(self):
        result, agent, _, waits = self.run_case(['429 {"code":"insufficient_quota"}'])
        self.assertEqual(result.exit_status, 1)
        self.assertIn('Provider session failed', result.completion_diagnosis)
        self.assertEqual(len(agent.calls), 1)
        self.assertFalse(waits)

    def test_token_budget_stops_without_an_episode_result(self):
        with self.assertRaises(InfrastructureUnavailable):
            self.run_case([THROTTLED], token_budget=10)

    def test_cancellation_stops_without_an_episode_result(self):
        with self.assertRaises(CancelledError):
            self.run_case([THROTTLED], cancelled=True)


class HistoricalCreditTests(TestCase):
    def test_credit_preserves_history_and_token_accounting(self):
        from aka.light.workflow.models import SupervisorState
        state = SupervisorState.from_dict({"episodes": 29, "protocol_failures": 29,
            "tokens": 1234, "rate_limit_credits": [2, 3]})
        restored = SupervisorState.from_dict(state.as_dict())
        self.assertEqual(restored.tokens, 1234)
        self.assertEqual(restored.protocol_failures, 29)
        self.assertEqual(restored.rate_limit_credits, [2, 3])
        self.assertEqual(SupervisorState.from_dict({}).rate_limit_credits, [])

    def test_completed_version_limit_resumes_only_with_audited_credit(self):
        from aka.light.workflow.episode import LongHorizonCampaign
        from aka.light.workflow.models import SupervisorState
        from unittest.mock import Mock
        with TemporaryDirectory() as directory:
            base = Mock()
            base.candidate_workspace.plan.side_effect = RuntimeError("new episode planned")
            engine = LongHorizonCampaign(base_campaign=base, max_version=30,
                                         verifier=Mock(), session_runner=Mock())
            # workspace is derived from the campaign by the real engine.
            base.workspace = Path(directory)
            for credits in ([], [2]):
                state = SupervisorState(episodes=29, rate_limit_credits=credits)
                with patch('aka.light.workflow.episode.main_adapter.prepare_campaign'), \
                     patch('aka.light.workflow.episode.CampaignStore.ensure_excluded'), \
                     patch('aka.light.workflow.episode.git_head', return_value='base'), \
                     patch('aka.light.workflow.episode.main_adapter.conversion_required', return_value=False), \
                     patch('aka.light.workflow.episode.main_adapter.latest_version', return_value=30), \
                     patch('aka.light.workflow.episode.CampaignStore.load_state', return_value=state), \
                     patch.object(LongHorizonCampaign, '_recover_interrupted', return_value=None), \
                     patch.object(LongHorizonCampaign, '_episode_mode', return_value='episode'):
                    if credits:
                        with self.assertRaisesRegex(RuntimeError, "new episode planned"):
                            engine.run()
                    else:
                        self.assertEqual(engine.run(), 'budget: max-iters')


if __name__ == '__main__':
    main()
