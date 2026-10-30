"""Real HTTP/client/scheduler regressions; a CPU evaluator records exact inputs."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from aka.legacy.execution.runtime_tools import local_eval, local_gateway
from aka.light.evaluation.source_transport import ARCHIVES, uploaded_request, upload_bytes


EVALUATOR = '''import argparse, json, pathlib, runpy
p = argparse.ArgumentParser()
p.add_argument('--config')
p.add_argument('--sdk-result-path-output')
a = p.parse_args()
c = json.loads(pathlib.Path(a.config).read_text())
candidate = runpy.run_path(c['input'])['Model']
assert candidate == 42
assert c['warmup_iters'] == 5 and c['perf_timeout_s'] == 300
out = pathlib.Path(c['output'])
out.mkdir(parents=True)
result = {'passed': {'correctness': {'0': {'status': 'failed'}}}, 'observed': c}
if c.get('eval_mode') == 'abba':
    baseline = runpy.run_path(c['baseline_input'])['Model']
    assert baseline == 41
    result['abba'] = {'runs': [dict(index=i, revision=r, label=l, repeat=n,
        result={'measured': candidate if r == 'candidate' else baseline,
                'passed': {'correctness': {'0': {'status': 'failed'}}}})
        for i, (r,l,n) in enumerate([('baseline','A',0), ('candidate','B',0),
                                    ('candidate','B',1), ('baseline','A',1)])]}
(out / 'eval_result.json').write_text(json.dumps(result))
pathlib.Path(a.sdk_result_path_output).write_text(str(out / 'eval_result.json'))
raise SystemExit(1)  # Candidate failure is complete evaluator evidence.
'''


class LocalNativeEvalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        bench = self.root / 'bench'
        (bench / 'scripts').mkdir(parents=True)
        (bench / 'scripts/run_eval.py').write_text(EVALUATOR)
        self.gateway = local_gateway.LocalGateway(self.root / 'state', atrex_bench_root=bench)
        self.server = local_gateway.create_server(self.gateway, '127.0.0.1', 0)
        self.url = 'http://127.0.0.1:' + str(self.server.server_port)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.gateway.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.gateway.close()
        self.temp.cleanup()

    def call(self, method, path, payload=None, timeout=30):
        request = urllib.request.Request(self.url + path, method=method,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    def request(self):
        return {'spec': {'target_hardware': ['local'], 'num_gpus': 1},
                'candidate': {'archive': 'archives/candidate.tar.gz', 'entry_point': 'kernel.py'},
                'reference': {'operator': 'fixture', 'reference_py': 'Model = 42',
                              'input_py': 'def get_inputs(): return []', 'shapes': {'0': {}}},
                'options': {'timeout_s': 1230},
                'runner_overrides': {'warmup_iters': 5, 'perf_timeout_s': 300},
                ARCHIVES: {'archives/candidate.tar.gz': {'kernel.py': 'from helper import Model',
                                                        'helper.py': 'Model = 42'}}}

    def evaluate(self, request, key):
        wire = uploaded_request(request, key, json_call=self.call, upload=upload_bytes)
        accepted = self.call('POST', '/v1/jobs/eval', wire)
        job = self.call('GET', '/v1/jobs/' + accepted['job_id'] + '?wait=true&timeout=20')
        return wire, job

    def test_public_upload_client_executes_exact_archive_and_preserves_failed_verdict(self):
        wire, job = self.evaluate(self.request(), 'candidate')
        self.assertEqual(job['status'], 'succeeded', job)
        self.assertEqual(job['result']['passed']['correctness']['0']['status'], 'failed')
        self.assertEqual(job['result']['observed']['perf_timeout_s'], 300)
        again = self.call('POST', '/v1/jobs/eval', wire)
        self.assertEqual(again['job_id'], job['job_id'])

    def test_native_abba_keeps_all_sdk_blocks_and_both_exact_sources(self):
        request = self.request()
        request['abba'] = {'baseline': {'archive': 'archives/baseline.tar.gz', 'entry_point': 'kernel.py'},
                           'repeats': 2}
        request[ARCHIVES]['archives/baseline.tar.gz'] = {'kernel.py': 'Model = 41'}
        _, job = self.evaluate(request, 'abba')
        self.assertEqual(job['status'], 'succeeded', job)
        blocks = job['result']['abba']['sdk_results']
        self.assertEqual(len(blocks), 2)
        for block in blocks:
            self.assertEqual([row['result']['measured'] for row in block['abba']['runs']], [41,42,42,41])
            self.assertTrue(all(row['result']['passed']['correctness']['0']['status'] == 'failed'
                                for row in block['abba']['runs']))

    def test_partial_abba_is_an_infrastructure_failure_not_a_partial_success(self):
        runner = self.root / 'bench/scripts/run_eval.py'
        runner.write_text(EVALUATOR.replace("('baseline','A',1)", "('candidate','B',2)"))
        request = self.request()
        request['abba'] = {'baseline': 'Model = 41', 'repeats': 1}
        _, job = self.evaluate(request, 'partial')
        self.assertEqual(job['status'], 'failed', job)
        self.assertEqual(job['error']['reason'], 'evaluator_failed')
        self.assertIn('incomplete native ABBA evidence', job['result']['stderr'])

    def test_upload_digest_mismatch_is_rejected_before_eval(self):
        data = b'original'
        reservation = self.call('POST', '/v1/uploads', {'kind': 'eval', 'spec': self.request()['spec'],
            'files': [{'path': 'source.tar.gz', 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}]})
        with self.assertRaisesRegex(Exception, 'HTTP 400'):
            upload_bytes(reservation['uploads'][0]['put_url'], b'modified')
        self.assertEqual(self.call('GET', '/v1/jobs')['jobs'], [])

    def test_uploaded_archive_rejects_traversal_and_links(self):
        for name, kind in [('../escape.py', tarfile.REGTYPE), ('link', tarfile.SYMTYPE)]:
            with self.subTest(name=name):
                stream = io.BytesIO()
                with tarfile.open(fileobj=stream, mode='w:gz') as bundle:
                    info = tarfile.TarInfo(name)
                    info.type = kind
                    info.linkname = '../escape.py' if kind == tarfile.SYMTYPE else ''
                    bundle.addfile(info, io.BytesIO())
                data = stream.getvalue()
                request = self.request()
                reservation = self.gateway.uploads.reserve({'kind': 'eval', 'spec': request['spec'],
                    'files': [{'path': 'archives/candidate.tar.gz', 'bytes': len(data),
                               'sha256': hashlib.sha256(data).hexdigest()}]})
                item = reservation['uploads'][0]
                self.gateway.uploads.put(item['upload_ref'], data)
                request['oss_files'] = [item]
                with self.assertRaises(ValueError):
                    local_eval.materialize_source(request['candidate'], self.root / ('source-' + str(kind)),
                                                  request, self.gateway.uploads, 'eval')
        self.assertFalse((self.root / 'escape.py').exists())


if __name__ == '__main__':
    unittest.main()
