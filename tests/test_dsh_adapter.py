"""DSH's public Agent contract: sessions, accounting, evidence and recovery."""
from dataclasses import asdict
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from aka.contracts.agent import AgentRunRequest
from aka.legacy.agent.adapter import DshAdapter
from aka.legacy.agent.dsh import normalize_stream, session_root, settings_values, terminal_failure, token_usage
from aka.legacy.agent.native import resume_session_command
from aka.light.agent.provider import AgentOptions, LightAgent
from aka.light.agent.session_capture import SessionCapture, clear_capture
from aka.light.agent.session_native import HostSessionTranscripts


def stream(*events):
    return "".join(json.dumps(event) + "\n" for event in events)


def invocation(session="session-native", *, tokens=30, reason=None, extra=()):
    return stream(
        {"type": "session", "sessionId": session},
        {"type": "status", "phase": "step_start", "turn": 1, "step": 1},
        *extra,
        {"type": "status", "phase": "step_end", "turn": 1, "step": 1,
         "usage": {"inputTokens": tokens - 10, "outputTokens": 10, "totalTokens": tokens}},
        {"type": "final", "sessionId": session, "reason": reason or {"kind": "completed"}, "text": "done"},
    )


def native_log(home, identity, *, parent=None, version=3):
    path = session_root({"DSH_HOME": str(home)}) / "--fixture--" / identity / f"session.v{version}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {"type": "session", "id": identity, "version": version}
    if parent:
        header["parentSession"] = parent
    path.write_text(stream(header))
    return path


class DshProtocolTest(unittest.TestCase):
    def setUp(self):
        values = {'ATREX_DSH_SESSION_SETTINGS': '{}', 'ATREX_SESSION_SETTINGS': '',
                  'ATREX_AKA_RUNTIME_URL': '', 'ATREX_AKA_RUNTIME_TOKEN': ''}
        self.environment = patch.dict(os.environ, values)
        self.environment.start()

    def tearDown(self):
        clear_capture()
        self.environment.stop()

    def test_settings_and_prompt_are_data_not_cli_options(self):
        adapter = DshAdapter()
        command = adapter.build_command("--session-id attacker", "caller", "max", '{"model":"configured-model"}')
        self.assertEqual(command[-2:], ["--", "--session-id attacker"])
        self.assertNotIn("caller", command)
        resumed = resume_session_command("continue", "session-real", "high", "dsh")
        self.assertEqual(resumed[resumed.index("--session-id") + 1], "session-real")
        environment = adapter.prepare_environment(
            {"ATREX_DSH_SESSION_SETTINGS": '{"provider":"route","model":"model"}'}, reasoning_effort="high")
        self.assertEqual(environment["AKA_DSH_PROVIDER"], "route")
        self.assertEqual(environment["AKA_DSH_MODEL"], "model")
        self.assertEqual(environment["AKA_DSH_REASONING_EFFORT"], "high")
        defaults = adapter.prepare_environment(
            {"ATREX_DSH_SESSION_SETTINGS": '{"provider":"route","reasoning_effort":"default"}'},
            reasoning_effort="max")
        self.assertEqual(defaults["AKA_DSH_REASONING_EFFORT"], "")
        override = adapter.prepare_environment(
            {"ATREX_DSH_SESSION_SETTINGS": '{"reasoning_effort":"high"}'}, reasoning_effort="max")
        self.assertEqual(override["AKA_DSH_REASONING_EFFORT"], "high")
        configured = adapter.build_command('work', 'caller', 'high',
            '{"profile":"custom-headless","provider":"configured-provider","model":"configured-model"}')
        self.assertEqual(configured[configured.index('--profile') + 1], 'custom-headless')
        self.assertEqual(adapter.prepare_environment({}, reasoning_effort='xhigh')['AKA_DSH_REASONING_EFFORT'], 'max')
        for invalid in ('[]', '{"plugins":[]}', '{"provider":false}', '{"model":""}',
                        '{"profile":"../web"}', '{"reasoning_effort":null}', '{'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                settings_values(invalid)

    def test_committed_steps_cache_buckets_and_resume_are_not_double_counted(self):
        usage = {"inputTokens": 20, "outputTokens": 10, "cacheReadTokens": 100,
                 "cacheWriteTokens": 5, "totalTokens": 135}
        end = {"type": "status", "phase": "step_end", "turn": 1, "step": 1, "usage": usage}
        marker = 'ATREX_TRACE_EVENT=' + json.dumps({"schema": "atrex.iteration_trace.v1",
                  "kind": "phase_marker", "action": "start", "phase": "research", "marker_id": "m1"})
        text = stream({"type": "session", "sessionId": "session-real"}, end, end,
                      {"type": "tool_result", "status": "completed", "result": marker},
                      {"type": "tool_result", "status": "error", "result": marker},
                      {"type": "final", "sessionId": "session-real", "reason": {"kind": "completed"}})
        events, total = normalize_stream(text)
        self.assertEqual(total.total_tokens, 135)
        self.assertEqual(total.input_tokens, 20)
        self.assertEqual(total.cache_read_tokens, 100)
        self.assertEqual(total.measurement, "exact")
        self.assertEqual(sum(event.kind == "usage_delta" for event in events), 1)
        self.assertEqual(sum(event.kind == "phase_marker" for event in events), 1)
        self.assertEqual([event.sequence for event in events], list(range(len(events))))
        self.assertEqual(normalize_stream(text)[1], total)

    def test_missing_steps_and_interruption_preserve_partial_usage(self):
        text = invocation().splitlines()
        self.assertEqual(normalize_stream("\n".join(text[:-1]))[1].measurement, "partial")
        text.insert(2, json.dumps({"type": "status", "phase": "step_end", "turn": 1, "step": 0}))
        self.assertEqual(normalize_stream("\n".join(text))[1].measurement, "partial")
        self.assertEqual(normalize_stream('{"type":"final"}')[1].measurement, "unavailable")

    def test_invalid_usage_counters_remain_unavailable(self):
        for field in ('inputTokens', 'outputTokens', 'cacheReadTokens', 'cacheWriteTokens', 'totalTokens'):
            for value in (-1, True, 1.5, '10'):
                with self.subTest(field=field, value=value):
                    raw = {'inputTokens': 20, 'outputTokens': 10, 'totalTokens': 30, field: value}
                    self.assertIsNone(token_usage(raw).total_tokens)
        self.assertEqual(token_usage({'inputTokens': 20, 'outputTokens': 10}).measurement, 'partial')

    def test_terminal_success_requires_identity_and_completed_reason(self):
        self.assertEqual(terminal_failure(invocation()), (None, None))
        self.assertEqual(terminal_failure(invocation(reason={"kind": "max-tokens"})),
                         ("output_limit", "max_output_tokens"))
        self.assertEqual(terminal_failure(invocation(reason={"kind": "error", "code": "rate_limit_error"})),
                         ("rate_limit", "rate_limit_error"))
        self.assertEqual(terminal_failure(invocation(reason={"kind": "error", "error": {"code": "RATE_LIMIT"}})),
                         ("rate_limit", "RATE_LIMIT"))
        self.assertEqual(terminal_failure(invocation(), "another")[1], "dsh_incomplete_stream")
        self.assertEqual(DshAdapter().validated_exit_status('{"type":"text","text":"done"}', 0), 1)
        self.assertEqual(DshAdapter().validated_exit_status(invocation(), -15), -15)
        self.assertEqual(terminal_failure(invocation(extra=({"type": "text", "text": "rate_limit_error"},))),
                         (None, None))

    def test_agent_runtime_and_resume_publish_native_identity_and_local_usage(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            commands = []

            def process(command, cwd, timeout, env):
                commands.append(command)
                self.assertEqual(env["AKA_DSH_REASONING_EFFORT"], "high")
                native_log(root, "session-native")
                return invocation(extra=({"type": "tool_call", "tool": "bash", "callId": "one",
                                          "input": {"command": "true"}},)), "", 0, False

            agent = LightAgent(AgentOptions(capture=False), process_runner=process)
            with patch.dict(os.environ, {"DSH_HOME": str(root)}):
                run = agent.runtime("dsh").run(AgentRunRequest(root, "work", 10, "high"))
                native = agent.open_session("dsh", root)
                self.assertEqual(native.session_id, "")
                first = native.invoke("work", reasoning_effort="high")
                second = native.invoke("continue", resume=True, reasoning_effort="high")
                self.assertTrue(native.supports_resume)
                self.assertEqual(first.session_id, "session-native")
                self.assertEqual(second.terminal_usage, first.terminal_usage)
                self.assertTrue(second.resume_usage_qualified)
                self.assertNotIn("--session-id", commands[1])
                self.assertIn("--session-id", commands[2])
                agent.close()
            self.assertEqual(run.session_id, "session-native")
            self.assertEqual(run.terminal_usage.total_tokens, 30)
            observations = [json.loads(path.read_bytes()) for path in (root / '.aka-agent-observations').glob('*.json')]
            self.assertEqual(len(observations), 3)
            self.assertTrue(any(row['kind'] == 'tool_call' for row in observations[0]['events']))
            self.assertFalse((root / '.aka-agent-session-errors').exists())

    def test_capture_matches_adapter_without_charging_native_history(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            transcript = native_log(root / 'home', 'session-native')
            capture = SessionCapture(root / 'capture', backend='dsh',
                                     command=DshAdapter().build_command('work', 'caller', 'high', ''),
                                     context={}, native_environment={'DSH_HOME': str(root / 'home')})
            capture._chunks['stdout'].append(invocation())
            report = capture._write_usage(True)
            self.assertEqual(report['total'], asdict(normalize_stream(invocation())[1]))
            self.assertEqual(capture.session_id, 'session-native')
            self.assertEqual(report['response_count'], 1)
            self.assertIn(transcript, [path for path, _ in capture._native_paths(force=True).values()])

    def test_checkpoint_restores_only_selected_session_family_and_latest_generation(self):
        from aka.legacy.agent.checkpoint import capture_native_session, export_checkpoint, restore_checkpoint
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            home, output = root / 'home', root / 'output'
            older = native_log(home, 'session-native', version=2)
            parent = native_log(home, 'session-native')
            child = native_log(home, 'session-child', parent='session-native')
            foreign = native_log(home, 'session-foreign')
            capture_native_session(output, output, 'dsh', 'session-native', {'DSH_HOME': str(home)})
            snapshot = export_checkpoint(output, root / 'snapshot')
            saved = json.loads((snapshot.directory / 'manifest.json').read_bytes())
            self.assertEqual({row['target'] for row in saved['files']}, {str(parent), str(child)})
            parent.unlink()
            child.unlink()
            restore_checkpoint(snapshot)
            self.assertTrue(parent.is_file() and child.is_file() and older.is_file() and foreign.is_file())
            reader = HostSessionTranscripts('dsh', {'DSH_HOME': str(home)}, ['dsh', '--session-id', 'session-native'])
            self.assertEqual({path for path, _ in reader.selected(force=True).values()}, {parent, child})
            commands = []

            def process(command, cwd, timeout, env):
                commands.append(command)
                return invocation(), '', 0, False

            agent = LightAgent(AgentOptions(capture=False), process_runner=process)
            with patch.dict(os.environ, {'DSH_HOME': str(home)}):
                native = agent.open_session('dsh', root, session_id='session-native', output_directory=output)
                self.assertEqual(native.invoke('continue', resume=True).session_id, 'session-native')
                self.assertIn('--session-id', commands[0])
                agent.close()

    def test_capture_counts_native_child_attempts_once_and_excludes_seeded_context(self):
        from aka.light.agent.session_usage import UsageAccumulator
        usage = UsageAccumulator('dsh')
        usage.feed('provider/stdout.stream-json', invocation(tokens=30))
        sample = {"inputTokens": 4, "outputTokens": 6, "totalTokens": 10}
        path = 'provider/native/.dsh/aka-sessions/project/session-child/session.v3.jsonl'
        usage.feed(path, stream(
            {"type": "session", "id": "session-child", "isSeeded": True},
            {"type": "assistant/message", "seq": 0, "data": {"usage": sample}},
            {"type": "session/end-seed", "seq": 1, "data": {"inherited": True}},
            {"type": "assistant/attempt", "seq": 2, "data": {"stream": [
                {"type": "chunk", "chunk": {"type": "usage", "usage": sample}}]}},
            {"type": "assistant/message", "seq": 3, "data": {"usage": sample}},
        ))
        root_path = 'provider/native/.dsh/aka-sessions/project/session-native/session.v3.jsonl'
        usage.feed(root_path, stream({"type": "assistant/message", "seq": 4, "data": {"usage": sample}}))
        report = usage.report(finished=True)
        self.assertEqual(report['total']['total_tokens'], 50)
        self.assertEqual(report['response_count'], 3)
        self.assertEqual(report['total']['measurement'], 'exact')

    def test_native_log_migration_excludes_history_and_deduplicates_current_attempts(self):
        from aka.light.agent.session_usage import UsageAccumulator
        usage = UsageAccumulator('dsh')
        usage.feed('provider/stdout.stream-json', invocation(tokens=30))
        sample = {'inputTokens': 4, 'outputTokens': 6, 'totalTokens': 10}
        header = {'type': 'session', 'id': 'session-child', 'parentSession': 'session-native'}
        def settled(sequence):
            return {'type': 'assistant/message', 'seq': sequence, 'data': {'usage': sample}}
        base = 'provider/native/.dsh/aka-sessions/project/session-child/'
        usage.feed(base + 'session.v3.jsonl', stream(header, settled(1)), previous=True)
        usage.feed(base + 'session.v4.jsonl', stream(header, settled(1), settled(2)))
        self.assertEqual(usage.report(finished=True)['total']['total_tokens'], 40)
        usage.feed(base + 'session.v5.jsonl', stream(header, settled(1), settled(2), settled(3)))
        report = usage.report(finished=True)
        self.assertEqual(report['total']['total_tokens'], 50)
        self.assertEqual(report['response_count'], 3)
        self.assertEqual(report['total']['measurement'], 'exact')

    def test_isolated_home_seeds_selected_profile_and_preserves_session_auth(self):
        from aka.light.agent.home import _seed_provider
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            provider, home = root / 'configured-home', root / 'session-home'
            fixtures = {
                '.env': 'FIXTURE_SETTING=example\n',
                '.credentials.yaml': 'fixture: synthetic-auth\n',
                'cordis.patch.yml': '[]\n',
                'profiles/custom-headless/package.json': '{"private":true}\n',
                'profiles/custom-headless/cordis.patch.yml': '[]\n',
                'sessions/unrelated.jsonl': 'unrelated history\n',
            }
            for name, content in fixtures.items():
                path = provider / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            environment = {'DSH_HOME': str(provider),
                           'ATREX_DSH_SESSION_SETTINGS': '{"profile":"custom-headless"}'}
            _seed_provider('dsh', root / 'host', home, environment)
            for name, content in fixtures.items():
                path = home / '.dsh' / name
                if name.startswith('sessions/'):
                    self.assertFalse(path.exists())
                    continue
                self.assertEqual(path.read_text(), content)
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            auth = home / '.dsh/.credentials.yaml'
            auth.write_text('fixture: session-refresh\n')
            _seed_provider('dsh', root / 'host', home, environment)
            self.assertEqual(auth.read_text(), 'fixture: session-refresh\n')
            self.assertEqual((provider / '.credentials.yaml').read_text(), fixtures['.credentials.yaml'])

    def test_connection_file_is_private_and_cleaned_when_process_raises(self):
        from aka.legacy.agent.dsh import CONNECTION_ENV
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            connections = []

            def process(command, cwd, timeout, env):
                path = Path(env[CONNECTION_ENV])
                connections.append(path)
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                self.assertEqual(json.loads(path.read_bytes()), {'url': 'http://127.0.0.1:1234', 'token': 'fixture'})
                raise RuntimeError('fixture failure')

            agent = LightAgent(AgentOptions(capture=False), process_runner=process)
            with patch.dict(os.environ, {'DSH_HOME': str(root), 'ATREX_AKA_RUNTIME_TOKEN': 'fixture',
                                        'ATREX_AKA_RUNTIME_URL': 'http://127.0.0.1:1234'}):
                with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                    agent.runtime('dsh').run(AgentRunRequest(root, 'work', 10))
                native = agent.open_session('dsh', root)
                with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                    native.invoke('work')
            self.assertEqual(len(connections), 2)
            self.assertTrue(all(not path.exists() for path in connections))
            agent.close()

    def test_managed_connection_is_bound_after_runtime_auth_and_cleaned_on_error(self):
        from aka.legacy.agent.dsh import CONNECTION_ENV
        from aka.light.agent.runtime.process import run_bounded
        from aka.light.runtime.server import scrub_environment
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            connections = []

            @contextlib.contextmanager
            def late_auth(workspace, environment, **kwargs):
                self.assertNotIn(CONNECTION_ENV, environment)
                yield {**environment, 'ATREX_AKA_RUNTIME_URL': 'http://127.0.0.1:1234',
                       'ATREX_AKA_RUNTIME_TOKEN': 'fixture'}

            def process(command, cwd, timeout, env, **kwargs):
                path = Path(env[CONNECTION_ENV])
                connections.append(path)
                self.assertEqual(json.loads(path.read_bytes()),
                                 {'url': 'http://127.0.0.1:1234', 'token': 'fixture'})
                self.assertNotIn(CONNECTION_ENV, scrub_environment(env))
                raise RuntimeError('late fixture failure')

            with patch('aka.light.runtime.server.session_environment', late_auth), patch(
                    'aka.light.agent.runtime.process._run_bounded', process):
                with self.assertRaisesRegex(RuntimeError, 'late fixture failure'):
                    run_bounded(['dsh', '--profile', 'headless'], root, 10,
                                {'DSH_HOME': str(root), 'ATREX_AKA_RUNTIME_OWNER': 'fixture-owner'})
            self.assertEqual(len(connections), 1)
            self.assertFalse(connections[0].exists())

    def test_long_session_repairs_handoff_without_resetting_native_identity_or_budget(self):
        from aka.light.workflow.session import LongSessionRunner
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            handoff = root / 'handoff.json'
            commands = []

            def process(command, cwd, timeout, env):
                commands.append(command)
                native_log(root, 'session-native')
                if len(commands) == 2:
                    handoff.write_text(json.dumps({'status': 'candidate_ready', 'candidate_commit': 'fixture-seal'}))
                return invocation(), '', 0, False

            agent = LightAgent(AgentOptions(capture=False), process_runner=process)
            with patch.dict(os.environ, {'DSH_HOME': str(root)}):
                result = LongSessionRunner(agent=agent, agent_cli='dsh',
                    usage_policy=lambda turn: turn.terminal_usage.total_tokens or 0).run(
                    root, 'work', handoff_path=handoff, handoff_resumes=1, completion_check=lambda value: '',
                    episode_timeout_seconds=30)
            agent.close()
            self.assertEqual(result.exit_status, 0)
            self.assertEqual(result.resume_count, 1)
            self.assertEqual(result.session_id, 'session-native')
            self.assertEqual(result.tokens, 60)
            self.assertEqual(commands[1][commands[1].index('--session-id') + 1], 'session-native')

    def test_client_refuses_invalid_connection_files_before_network_access(self):
        path = Path(__file__).resolve().parents[1] / 'src/aka/light/runtime/client/sandbox.py'
        spec = importlib.util.spec_from_file_location('dsh_client_fixture', path)
        client = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(client)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            connection = root / 'connection.json'
            connection.write_text(json.dumps({'url': 'https://example.invalid', 'token': 'fixture'}))
            fifo = root / 'fifo'
            os.mkfifo(fifo)
            for candidate in (connection, fifo):
                with self.subTest(candidate=candidate), patch.dict(os.environ, {
                    'ATREX_AKA_RUNTIME_CONNECTION_FILE': str(candidate),
                }), patch.object(client.urllib.request, 'build_opener') as opener, contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(client.proxy_command('/v1/plugins/execute', {'action': 'list'}), 75)
                    opener.assert_not_called()

    def test_replacement_agent_remains_independent_of_builtin_backend_registry(self):
        from aka.contracts.agent import NativeInvocation, TokenUsage, AgentRuntimeCapabilities
        from aka.light.workflow.session import LongSessionRunner
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            handoff = root / 'handoff.json'

            def invoke(*args, **kwargs):
                handoff.write_text(json.dumps({'status': 'pivot'}))
                return NativeInvocation(0, False, 'custom-session', '', '', TokenUsage.zero(), (),
                                        AgentRuntimeCapabilities(False, False, False), (), False)

            native = SimpleNamespace(environment={}, supports_resume=False, invoke=invoke, close=lambda: None)
            replacement = SimpleNamespace(open_session=lambda *args, **kwargs: native)
            result = LongSessionRunner(agent=replacement, agent_cli='custom-provider', usage_policy=lambda _: 0).run(
                root, 'work', handoff_path=handoff, handoff_resumes=0, completion_check=lambda _: '',
                episode_timeout_seconds=30)
            self.assertEqual(result.exit_status, 0)
            self.assertEqual(result.handoff.status, 'pivot')


if __name__ == '__main__':
    unittest.main()
