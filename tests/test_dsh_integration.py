"""Opt-in real DSH/headless test with a local model and Runtime wire fixture.

AKA_RUN_DSH_INTEGRATION=1 PYTHONPATH=src python -m unittest discover -s tests -v
No model credentials, remote model requests or GPU allocations are needed.
"""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from aka.contracts.agent import AgentRunRequest
from aka.light.agent.provider import AgentOptions, LightAgent


ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def fixture_server():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append((self.path, body, self.headers.get('Authorization')))
            if self.path in {'/v1/plugins/execute', '/v1/gateway/execute', '/v1/wiki/query', '/v1/journal/execute'}:
                value = json.dumps({'exit_code': 0, 'stdout': 'fixture-tool-ok\n', 'stderr': ''}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(value)))
                self.end_headers()
                self.wfile.write(value)
                return
            last = body['messages'][-1]['content']
            tool_result = isinstance(last, list) and any(row.get('type') == 'tool_result' for row in last)
            if tool_result:
                block = {'type': 'text', 'text': ''}
                delta = {'type': 'text_delta', 'text': 'fixture-complete'}
                reason = 'end_turn'
            else:
                block = {'type': 'tool_use', 'id': 'fixture-call', 'name': 'bash', 'input': {}}
                command = (
                    f'{shlex.quote(sys.executable)} work.py && python3 tools/plugin.py list'
                    ' && python3 tools/sandbox.py --kind run --no-sync -- python3 kernel.py'
                    ' && python3 tools/sandbox.py --kind profile --no-sync -- python3 kernel.py'
                    ' && python3 tools/sandbox.py --kind wiki-search --query fixture'
                    ' && python3 tools/sandbox.py --kind episode-report --request-file report.json'
                )
                delta = {'type': 'input_json_delta', 'partial_json': json.dumps({
                    'command': command, 'description': 'Exercise the AKA fixture tool client'})}
                reason = 'tool_use'
            frames = [
                ('message_start', {'type': 'message_start', 'message': {
                    'id': 'fixture-message', 'type': 'message', 'role': 'assistant',
                    'content': [], 'model': body['model'], 'stop_reason': None, 'stop_sequence': None,
                    'usage': {'input_tokens': 20, 'output_tokens': 0,
                              'cache_read_input_tokens': 100, 'cache_creation_input_tokens': 5}}}),
                ('content_block_start', {'type': 'content_block_start', 'index': 0, 'content_block': block}),
                ('content_block_delta', {'type': 'content_block_delta', 'index': 0, 'delta': delta}),
                ('content_block_stop', {'type': 'content_block_stop', 'index': 0}),
                ('message_delta', {'type': 'message_delta', 'delta': {'stop_reason': reason, 'stop_sequence': None},
                                   'usage': {'output_tokens': 10}}),
                ('message_stop', {'type': 'message_stop'}),
            ]
            value = ''.join(f'event: {kind}\ndata: {json.dumps(data)}\n\n' for kind, data in frames).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Content-Length', str(len(value)))
            self.end_headers()
            self.wfile.write(value)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', requests
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


@unittest.skipUnless(os.environ.get('AKA_RUN_DSH_INTEGRATION') == '1' and shutil.which('dsh'),
                     'set AKA_RUN_DSH_INTEGRATION=1 with DSH installed')
class RealDshTest(unittest.TestCase):
    def test_real_headless_edits_calls_existing_client_captures_and_resumes(self):
        with tempfile.TemporaryDirectory() as raw, fixture_server() as (url, requests):
            root = Path(raw).resolve()
            workspace = root / 'workspace'
            workspace.mkdir()
            (workspace / 'tools').mkdir()
            for name in ('plugin.py', 'sandbox.py'):
                shutil.copyfile(ROOT / 'src/aka/light/runtime/client' / name, workspace / 'tools' / name)
            skill = workspace / 'skills' / 'fixture-skill'
            skill.mkdir(parents=True)
            (skill / 'SKILL.md').write_text('---\nname: fixture-skill\ndescription: Fixture skill catalog entry.\n---\nUse the fixture.\n')
            marker = {'schema': 'atrex.iteration_trace.v1', 'kind': 'phase_marker',
                      'action': 'start', 'phase': 'research', 'marker_id': 'fixture-marker'}
            (workspace / 'work.py').write_text(
                'import os\n'
                'from pathlib import Path\n'
                'assert "ATREX_AKA_RUNTIME_TOKEN" not in os.environ\n'
                'Path("kernel.py").write_text("candidate = True\\n")\n'
                'Path("report.json").write_text("{}")\n'
                f'print({"ATREX_TRACE_EVENT=" + json.dumps(marker)!r})\n')
            values = {
                'DSH_HOME': str(root / 'dsh-home'), 'DEEPSEEK_API_KEY': 'fixture-key',
                'DEEPSEEK_BASE_URL': url, 'ATREX_DSH_SESSION_SETTINGS': '{}',
                'ATREX_AKA_RUNTIME_OWNER': 'fixture-owner',
                'ATREX_AKA_RUNTIME_URL': '', 'ATREX_AKA_RUNTIME_TOKEN': '',
                'ATREX_SESSION_CAPTURE_DIR': str(root / 'captures'),
            }
            @contextmanager
            def late_runtime_environment(workspace, environment, **kwargs):
                self.assertFalse(environment.get('ATREX_AKA_RUNTIME_CONNECTION_FILE'))
                # Match the Campaign order: auth is minted in the process
                # runner, after Agent/NativeSession environment preparation.
                yield {**environment, 'ATREX_AKA_RUNTIME_URL': url,
                       'ATREX_AKA_RUNTIME_TOKEN': 'fixture-capability'}

            agent = LightAgent(AgentOptions(capture=True, timeout_seconds=45))
            with patch.dict(os.environ, values), patch(
                    'aka.light.runtime.server.session_environment', late_runtime_environment):
                try:
                    result = agent.runtime('dsh').run(AgentRunRequest(workspace, 'Run fixture tools.', 45, 'max'))
                    self.assertEqual(result.exit_status, 0, result.stderr_tail + result.stdout_tail)
                    self.assertEqual(result.terminal_usage.total_tokens, 270)
                    self.assertEqual(result.terminal_usage.measurement, 'exact')
                    self.assertTrue(result.session_id.startswith('session-'))
                    self.assertTrue(any(row.kind == 'phase_marker' for row in result.events), result)
                    native = agent.open_session('dsh', workspace, output_directory=root / 'observations')
                    first = native.invoke('Run fixture tools.', reasoning_effort='high')
                    self.assertEqual(first.exit_status, 0, first.stderr + first.stdout)
                    second = native.invoke('Run fixture tools again.', resume=True, reasoning_effort='high')
                    self.assertEqual(second.exit_status, 0, second.stderr + second.stdout)
                    self.assertEqual(first.session_id, second.session_id)
                    self.assertEqual(first.terminal_usage, second.terminal_usage)
                    self.assertEqual(second.terminal_usage.total_tokens, 270)
                    native.close()
                finally:
                    agent.close()
            self.assertEqual((workspace / 'kernel.py').read_text(), 'candidate = True\n')
            tool_requests = [(path, body, authorization) for path, body, authorization in requests
                             if path in {'/v1/plugins/execute', '/v1/gateway/execute', '/v1/wiki/query', '/v1/journal/execute'}]
            self.assertEqual(len(tool_requests), 15)
            self.assertTrue(all(authorization == 'Bearer fixture-capability'
                                for _, _, authorization in tool_requests))
            self.assertEqual({path for path, _, _ in tool_requests},
                             {'/v1/plugins/execute', '/v1/gateway/execute', '/v1/wiki/query', '/v1/journal/execute'})
            self.assertEqual(sum(body == {'action': 'list'} for _, body, _ in tool_requests), 3)
            model_requests = [body for path, body, _ in requests if path.endswith('/messages')]
            self.assertTrue(any('fixture-skill' in json.dumps(body) for body in model_requests))
            self.assertTrue(any('fixture-tool-ok' in json.dumps(body) for body in model_requests))
            reports = list((root / 'captures').glob('run-*/token-usage.json'))
            self.assertEqual(len(reports), 3)
            for report in reports:
                value = json.loads(report.read_bytes())
                self.assertEqual(value['total']['total_tokens'], 270)
                self.assertEqual(value['total']['measurement'], 'exact')
            self.assertTrue(list((root / 'captures').glob('run-*/provider/native/.dsh/aka-sessions/*/*/session*.jsonl')))
            self.assertFalse(list((root / 'observations' / '.aka-agent-session-errors').glob('*')))
            self.assertTrue(list((root / 'observations' / '.aka-agent-sessions').glob('*/session.json')))
            self.assertEqual(list((root / 'dsh-home' / 'aka-connections').glob('*')), [])


if __name__ == '__main__':
    unittest.main()
