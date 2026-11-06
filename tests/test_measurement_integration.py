"""Real Runtime and Harness transports with local Measurement/model fixtures."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
from importlib.resources import files
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

from aka.contracts.agent import AgentRunRequest
from aka.light.agent.provider import AgentOptions, LightAgent
from aka.light.runtime.contribution_composition import compose
from aka.light.runtime.server import RuntimeConfig, SupervisorRuntime, OWNER_ENV

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def fixture_server(*, tool_name, arguments):
    """An offline model route that requests one native tool, then reads its result."""
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append((self.path, body, self.headers.get('Authorization')))
            last = body['messages'][-1]['content']
            complete = isinstance(last, list) and any(row.get('type') == 'tool_result' for row in last)
            block = ({'type': 'text', 'text': ''} if complete else
                     {'type': 'tool_use', 'id': 'fixture-call', 'name': tool_name, 'input': {}})
            delta = ({'type': 'text_delta', 'text': 'fixture-complete'} if complete else
                     {'type': 'input_json_delta', 'partial_json': json.dumps(arguments)})
            frames = [
                ('message_start', {'type': 'message_start', 'message': {
                    'id': 'fixture-message', 'type': 'message', 'role': 'assistant',
                    'content': [], 'model': body['model'], 'stop_reason': None, 'stop_sequence': None,
                    'usage': {'input_tokens': 20, 'output_tokens': 0,
                              'cache_read_input_tokens': 100, 'cache_creation_input_tokens': 5}}}),
                ('content_block_start', {'type': 'content_block_start', 'index': 0, 'content_block': block}),
                ('content_block_delta', {'type': 'content_block_delta', 'index': 0, 'delta': delta}),
                ('content_block_stop', {'type': 'content_block_stop', 'index': 0}),
                ('message_delta', {'type': 'message_delta', 'delta': {
                    'stop_reason': 'end_turn' if complete else 'tool_use', 'stop_sequence': None},
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


class MeasurementFixture:
    def __init__(self):
        self.executed = []
        self.prepared = []
        self.records = {}

    def execute(self, session, route, request, *, prepare_identity_only=False):
        argv = request['argv']
        if prepare_identity_only:
            self.prepared.append(argv)
            return None
        self.executed.append(argv)
        if argv[1] == 'record-read':
            return {'agent_result': self.records[argv[-1]]}
        if argv[1] == 'kernel-records':
            return {'agent_result': {'records': list(self.records.values())}}
        source = (session.workspace / 'kernel.py').read_bytes()
        digest = hashlib.sha256(source).hexdigest()
        value = {'operation': 'Evaluate', 'status': 'completed', 'record_id': 'fixture-' + digest,
                 'result': {'correct': True, 'kernel_id': 'kernel-' + digest}}
        self.records[value['record_id']] = value
        (session.workspace / 'retained-record.json').write_text(json.dumps(value))
        return {'agent_result': value}


@contextmanager
def runtime_fixture(*, modules=('measurement',), overrides=None):
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        workspace = root / 'workspace'
        (workspace / 'tools').mkdir(parents=True)
        (workspace / 'kernel.py').write_text('candidate = True\n')
        (workspace / 'request.json').write_text('{}')
        for name in ('sandbox.py', 'aka.py'):
            shutil.copyfile(Path(str(files('aka.light'))) / 'runtime/client' / name, workspace / 'tools' / name)
        backend = MeasurementFixture()
        registry = compose(backend.execute, modules=modules, instruction_overrides=overrides)
        runtime = SupervisorRuntime(RuntimeConfig('fixture-gpu', wiki_enabled=False,
                                    pause_agent_during_requests=True, request_timeout=10, queue_timeout=0.05), contributions=registry)
        try:
            yield root, workspace, runtime, backend
        finally:
            runtime.close()


def post(url, token, route, body):
    request = urllib.request.Request(url + route, json.dumps(body).encode(),
                                     {'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


class MeasurementRuntimeTest(unittest.TestCase):
    def test_campaign_resume_pins_effective_module_instructions(self):
        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw).resolve()
            config = RuntimeConfig('fixture-gpu', workspace=workspace, wiki_enabled=False)
            first = SupervisorRuntime(config)
            identity = first.contribution_catalog['identity']
            first.close()
            second = SupervisorRuntime(config)
            self.assertEqual(second.contribution_catalog['identity'], identity)
            second.close()
            with self.assertRaisesRegex(ValueError, 'selection changed'):
                SupervisorRuntime(RuntimeConfig('fixture-gpu', workspace=workspace, wiki_enabled=False,
                    instruction_overrides={'measurement.usage': 'A different instruction.'}))

    def test_cli_catalog_evaluate_and_read_share_the_selected_provider(self):
        with runtime_fixture() as (_root, workspace, runtime, backend), runtime.session(workspace, {}) as env:
            values = {**os.environ, **env}
            def cli(*args):
                result = subprocess.run([sys.executable, 'tools/aka.py', *args], cwd=workspace,
                                        env=values, capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
                return json.loads(result.stdout)
            catalog = cli('list')
            self.assertEqual(catalog, runtime.contribution_catalog)
            value = cli('call', 'measurement.evaluate', '--input', 'request.json')
            self.assertEqual(value, json.loads((workspace / 'retained-record.json').read_text()))
            (workspace / 'read.json').write_text(json.dumps({'record_id': value['record_id']}))
            self.assertEqual(cli('call', 'measurement.record_read', '--input', 'read.json'), value)
            self.assertEqual(len(backend.executed), 2)
            self.assertEqual(backend.prepared, [])

    def test_busy_session_prepares_identity_without_dispatching_another_job(self):
        with runtime_fixture() as (_root, workspace, runtime, backend), runtime.session(workspace, {}) as env:
            token = env['ATREX_AKA_RUNTIME_TOKEN']
            lock = runtime.capabilities[token].lock
            lock.acquire()
            try:
                status, response = post(runtime.url, token, '/v1/contributions/invoke', {
                    'catalog_id': runtime.contribution_catalog['identity'],
                    'tool': 'measurement.evaluate', 'arguments': {}})
            finally:
                lock.release()
            self.assertEqual(status, 429, response)
            self.assertEqual(response['error']['code'], 'request_not_started')
            self.assertEqual(len(backend.prepared), 1)
            self.assertEqual(backend.executed, [])

    def test_stale_unknown_invalid_and_revoked_requests_never_execute(self):
        with runtime_fixture() as (_root, workspace, runtime, backend):
            with runtime.session(workspace, {}) as env:
                token = env['ATREX_AKA_RUNTIME_TOKEN']
                valid = {'catalog_id': runtime.contribution_catalog['identity'],
                         'tool': 'measurement.evaluate', 'arguments': {}}
                for body in ({**valid, 'catalog_id': 'old'}, {**valid, 'tool': 'unknown.call'},
                             {**valid, 'arguments': {'hardware': 'override'}}):
                    status, response = post(runtime.url, token, '/v1/contributions/invoke', body)
                    self.assertEqual(status, 400, response)
                self.assertEqual(post(runtime.url, 'invalid', '/v1/contributions/catalog', {})[0], 401)
            self.assertEqual(post(runtime.url, token, '/v1/contributions/invoke', valid)[0], 401)
            self.assertEqual(backend.executed, [])

    def test_disabling_module_removes_its_tools_and_instructions(self):
        with runtime_fixture(modules=()) as (_root, workspace, runtime, backend), runtime.session(workspace, {}) as env:
            catalog = runtime.contribution_catalog
            self.assertEqual(catalog['tools'], [])
            self.assertEqual(catalog['instructions'], [])
            status, _ = post(runtime.url, env['ATREX_AKA_RUNTIME_TOKEN'], '/v1/contributions/invoke', {
                'catalog_id': catalog['identity'], 'tool': 'measurement.evaluate', 'arguments': {}})
            self.assertEqual(status, 400)
            self.assertEqual(backend.executed, [])

    def test_codex_launch_receives_projected_prompt_and_calls_the_same_runtime(self):
        with runtime_fixture(overrides={'measurement.usage': 'CUSTOM: use {{tool:measurement.evaluate}}.'}) as (root, workspace, runtime, backend):
            binary = root / 'bin'
            binary.mkdir()
            script = binary / 'codex'
            script.write_text(f'#!{sys.executable}\n' + '''import json, pathlib, subprocess, sys
prompt = sys.argv[-1]
assert 'CUSTOM: use `python3 tools/aka.py call measurement.evaluate' in prompt
assert '{{tool:' not in prompt
result = subprocess.run([sys.executable, 'tools/aka.py', 'call', 'measurement.evaluate', '--input', 'request.json'], capture_output=True, text=True)
assert result.returncode == 0, result.stderr
pathlib.Path('codex-result.json').write_text(result.stdout)
print(json.dumps({'type': 'turn.completed', 'usage': {'input_tokens': 3, 'output_tokens': 2}}))
''')
            script.chmod(0o755)
            with patch.dict(os.environ, {'PATH': str(binary) + os.pathsep + os.environ['PATH']}):
                agent = LightAgent(AgentOptions(timeout_seconds=20))
                try:
                    result = agent.runtime('codex').run(AgentRunRequest(workspace, 'Optimize.', 20,
                        environment={OWNER_ENV: runtime.owner_id}))
                finally:
                    agent.close()
            self.assertEqual(result.exit_status, 0, result.stderr_tail + result.stdout_tail)
            self.assertEqual(json.loads((workspace / 'codex-result.json').read_text()),
                             json.loads((workspace / 'retained-record.json').read_text()))
            self.assertEqual(len(backend.executed), 1)

    @unittest.skipUnless(os.environ.get('AKA_RUN_DSH_INTEGRATION') == '1' and shutil.which('dsh'),
                         'set AKA_RUN_DSH_INTEGRATION=1 with DSH installed')
    def test_real_dsh_registers_native_tool_and_consumes_shared_instruction(self):
        with runtime_fixture(overrides={'measurement.usage': 'CUSTOM: use {{tool:measurement.evaluate}}.'}) as (root, workspace, runtime, backend), fixture_server(
                tool_name='aka_measurement_evaluate', arguments={'mode': 'full'}) as (url, requests):
            values = {'DSH_HOME': str(root / 'dsh-home'), 'DEEPSEEK_API_KEY': 'fixture-key',
                      'DEEPSEEK_BASE_URL': url, 'ATREX_DSH_SESSION_SETTINGS': '{}'}
            with patch.dict(os.environ, values):
                agent = LightAgent(AgentOptions(capture=True, timeout_seconds=45))
                try:
                    native = agent.open_session('dsh', workspace, environment={OWNER_ENV: runtime.owner_id})
                    result = native.invoke('Evaluate the candidate.')
                    continued = native.invoke('Evaluate the candidate again.', resume=True)
                    self.assertEqual(continued.exit_status, 0, continued.stderr + continued.stdout)
                    self.assertEqual(result.session_id, continued.session_id)
                    native.close()
                finally:
                    agent.close()
            self.assertEqual(result.exit_status, 0, result.stderr + result.stdout)
            self.assertEqual(len(backend.executed), 2)
            self.assertEqual(backend.executed[0], ['--kind', 'run', '--mode', 'full'])
            messages = json.dumps([body for path, body, _ in requests if path.endswith('/messages')])
            self.assertIn('CUSTOM: use aka_measurement_evaluate.', messages)
            self.assertIn(next(iter(backend.records)), messages)
            self.assertNotIn('{{tool:', messages)
            self.assertEqual(list((root / 'dsh-home' / 'aka-connections').glob('*')), [])


if __name__ == '__main__':
    unittest.main()
