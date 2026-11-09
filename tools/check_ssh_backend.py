"""Offline regressions and opt-in GPU smoke tests for the Light SSH backend.

Run with PYTHONPATH=src python3 tools/check_ssh_backend.py. Pass --host, --gpu,
--runtime-bind and --init for a real isolated GPU run; no host is implicit.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from aka.light.evaluation import gateway, ssh_devices, ssh_transport, ssh_worker, ssh_native


class SSHTests(unittest.TestCase):
    def test_abba_requires_exact_order_and_every_shape(self):
        import copy
        from aka.light.evaluation.native_abba import parse_native_abba
        raw = {'passed': {'compile': {'status': 'passed'},
                          'correctness': {'0': {'status': 'passed'}}},
               'correctness': {'shapes': {'0': {}}},
               'performance': {'shapes': {'0': {'samples': [{'end_to_end_time_ms': .01}]}}}}
        runs = [{'index': i, 'revision': side, 'label': label, 'repeat': repeat, 'result': copy.deepcopy(raw)}
                for i, (side, label, repeat) in enumerate([
                    ('baseline', 'A', 0), ('candidate', 'B', 0),
                    ('candidate', 'B', 1), ('baseline', 'A', 1)])]
        job = {'status': 'succeeded', 'result': {'abba': {'sdk_results': [{'abba': {'runs': runs}}]}}}
        self.assertEqual(len(parse_native_abba(job, 2, ['0'])['runs']), 4)
        with self.assertRaises(ValueError):
            parse_native_abba(job, 2, ['0', '1'])
        runs[2]['label'] = 'A'
        with self.assertRaises(ValueError):
            parse_native_abba(job, 2, ['0'])

    def test_device_assignment_reaches_namespace(self):
        command = gateway._ssh_bwrap_command([], ['true'], gpu_index=5, gpu_uuid={
            'uuid': 'GPU-1234',
            'devices': ['/dev/kfd', '/dev/dri/renderD168'],
            'environment': {'ROCR_VISIBLE_DEVICES': 'GPU-1234'}})
        self.assertIn('--dev-bind /dev/dri/renderD168 /dev/dri/renderD168', command)
        self.assertIn('ROCR_VISIBLE_DEVICES=GPU-1234', command)
        self.assertNotIn('/dev/nvidia', command)
        self.assertIn('--unshare-net', command)

    def test_nvidia_minor_is_not_index(self):
        response = SimpleNamespace(stdout='5, GPU-12345678, 9, Disabled\n')
        real_exists = Path.exists
        with patch.object(Path, 'exists', lambda p: False if str(p) == '/dev/kfd' else real_exists(p)), \
             patch.object(ssh_devices.subprocess, 'run', return_value=response):
            assignment = ssh_devices.resolve(5)
        self.assertEqual(assignment['devices'][0], '/dev/nvidia9')
        self.assertEqual(assignment['environment'], {'CUDA_VISIBLE_DEVICES': 'GPU-12345678'})

    def test_runtime_selection_overrides_activation_and_preserves_vendor(self):
        script = gateway._ssh_shell_script(
            'export CUDA_VISIBLE_DEVICES=0; export ROCR_VISIBLE_DEVICES=0',
            gateway._ssh_visibility({'vendor': 'ppu', 'environment': {'CUDA_VISIBLE_DEVICES': '6'}})
            + '\nprintf "%s/%s/%s" "$CUDA_VISIBLE_DEVICES" "${ROCR_VISIBLE_DEVICES-unset}" "$ATREX_SSH_GPU_VENDOR"')
        result = subprocess.run(['bash', '-c', script], capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout, '6/unset/ppu')

    def test_endpoint_changes_invalidate_evidence(self):
        args = SimpleNamespace(ssh='worker', ssh_gpu=5, ssh_init='', ssh_runtime_bind=[])
        identity = ssh_transport.endpoint_identity(args, {})
        args.ssh_gpu = 6
        self.assertNotEqual(identity, ssh_transport.endpoint_identity(args, {}))
        args.ssh_gpu = 5
        self.assertNotEqual(identity, ssh_transport.endpoint_identity(args,
            {'ATREX_SANDBOX_SSH_GPU_ISOLATION': 'runtime'}))

    def test_source_tree_materialization_and_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = ssh_worker.materialize({'archive': 'a', 'entry_point': 'kernel.py'},
                {'__atrex_eval_archives': {'a': {'kernel.py': 'from impl.helper import Model',
                                               'impl/helper.py': 'class Model: pass'}}}, root)
            self.assertTrue(entry.is_file())
            self.assertTrue((root / 'impl/helper.py').is_file())
            with self.assertRaises(ValueError):
                ssh_worker.write(root, '../outside.py', '')

    def test_runtime_bind_allows_root_venv_but_not_home(self):
        self.assertEqual(gateway._ssh_runtime_bind('/root/job/env=/opt/env'), ('/root/job/env', '/opt/env'))
        for path in ('/root', '/root/.ssh', '/root/job'):
            with self.assertRaises(ValueError):
                gateway._ssh_runtime_bind(path)

    def test_amd_resolves_gaps_and_explicit_runtime_visibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for node, minor, unique in ((0, 0, 0), (3, 160, 16), (7, 168, 32)):
                path = root / f'sys/class/kfd/kfd/topology/nodes/{node}/properties'
                path.parent.mkdir(parents=True)
                path.write_text(f'drm_render_minor {minor}\nunique_id {unique}\n')
            (root / 'dev/dri').mkdir(parents=True)
            for name in ('kfd', 'dri/renderD160', 'dri/renderD168'):
                (root / 'dev' / name).touch()
            def remote_path(value):
                return root / value.lstrip('/')
            with patch.object(ssh_devices, 'Path', side_effect=remote_path):
                strict = ssh_devices.resolve(1)
                runtime = ssh_devices.resolve(1, 'runtime')
            self.assertEqual(strict['uuid'], 'GPU-0000000000000020')
            self.assertEqual(strict['devices'][1:], [str(root / 'dev/dri/renderD168')])
            self.assertEqual(len(runtime['devices']), 3)
            self.assertEqual(runtime['environment'], strict['environment'])

    def test_lost_health_probe_is_missing_evidence(self):
        args = SimpleNamespace(ssh='worker', ssh_init='', health_command='true',
                               ssh_runtime_bind=[], ssh_gpu=5)
        with patch.object(gateway, '_run_ssh_health', side_effect=gateway.SSHTransportError('lost')), \
             patch.object(gateway, '_record_environment_failure') as record:
            with self.assertRaises(gateway.EvidenceUnavailable):
                ssh_transport._check_health(gateway, args)
            record.assert_called_once()

    def test_native_rejects_wrong_targets_and_source_digest_before_build(self):
        import hashlib
        source = 'extern "C" void kernel() {}'
        manifest = {'targets': {'hardware': ['H20']},
                    'source': {'archive': 'a', 'files': {'kernel.cu': hashlib.sha256(source.encode()).hexdigest()}}}
        request = {'spec': {'target_hardware': ['A100']},
                   '__atrex_eval_archives': {'a': {'kernel.cu': source + 'changed'}}}
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {'ssh_worker': ssh_worker}):
            with self.assertRaisesRegex(ValueError, 'targets'):
                ssh_native.materialize(manifest, request, Path(directory))
            request['spec']['target_hardware'] = ['H20']
            with self.assertRaisesRegex(ValueError, 'digest'):
                ssh_native.materialize(manifest, request, Path(directory))

    def test_evaluator_version_pin_is_enforced(self):
        with patch.dict(sys.modules, {'local_eval': SimpleNamespace(),
                                     'atrex_bench': SimpleNamespace(__version__='0.1.0')}), \
             patch.object(ssh_worker, 'requirements'):
            with self.assertRaisesRegex(RuntimeError, 'requested pin'):
                ssh_worker.evaluate({'atrex_bench_version': '9.9.9'}, Path('.'))


REFERENCE = '''import torch
class Model(torch.nn.Module):
    def forward(self, x):
        return x + 1
'''
KERNEL = '''import torch
from impl.add import add
class Model(torch.nn.Module):
    def forward(self, x):
        return add(x)
'''
HELPER = '''import torch
import triton
import triton.language as tl
@triton.jit
def kernel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + i, tl.load(X + i, i < N, other=0) + 1, i < N)
def add(x):
    y = torch.empty_like(x)
    kernel[(triton.cdiv(x.numel(), 256),)](x, y, x.numel(), 256)
    return y
'''


def smoke(args):
    from aka.light.workspace.sources.source_tree import default_tree
    from aka.light.workspace.sources import snapshot
    with tempfile.TemporaryDirectory(prefix='aka-ssh-smoke-') as directory:
        root = Path(directory).resolve()
        files = {'kernel.py': KERNEL, 'impl/add.py': HELPER, 'reference.py': REFERENCE,
                 'input.py': 'import torch\ndef _make_inputs(n):\n    return {"x": torch.randn(n, device="cuda")}\n',
                 'shapes.json': json.dumps({'0': {'input_kwargs': {'n': 1024}},
                                           '1': {'input_kwargs': {'n': 8192}}})}
        for name, text in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        policy, _ = default_tree({'kernel.py': KERNEL.encode()})
        (root / '.atrex_source.json').write_text(json.dumps(policy))
        (root / 'scratch').mkdir()
        (root / 'scratch/baseline.bin').write_bytes(snapshot(root))
        command = [sys.executable, '-m', 'aka.light.evaluation.gateway', '--workspace', str(root),
                   '--hardware', args.hardware, '--ssh', args.host, '--ssh-gpu', str(args.gpu),
                   '--ssh-init', args.init, '--timeout', '300']
        for value in args.runtime_bind:
            command += ['--ssh-runtime-bind', value]
        env = dict(os.environ, ATREX_BENCH_RUNTIME_ROOT=str(Path(args.bench).resolve()))
        operations = [('env', ['--kind', 'env']), ('health', ['--check-health']),
                      ('evaluate', ['--kind', 'run']),
                      ('abba', ['--kind', 'run', '--baseline-path', 'scratch/baseline.bin', '--comparison-repeats', '2']),
                      ('check', ['--kind', 'check']), ('profile', ['--kind', 'profile']),
                      ('disassemble', ['--kind', 'disassemble']),
                      ('dev', ['--kind', 'dev', '--sync', 'scratch/ssh-check', '--', 'python3', '-c',
                               'import torch; from pathlib import Path; assert torch.cuda.device_count()==1; '
                               'p=Path("scratch/ssh-check/result.txt"); p.parent.mkdir(parents=True,exist_ok=True); '
                               'p.write_text(str(torch.ones(4,device="cuda").sum().item()))']),
                      ('bad-evaluate', ['--kind', 'run']),
                      ('timeout', ['--kind', 'dev', '--timeout', '2', '--no-sync', '--',
                                   'python3', '-c', 'import time; time.sleep(10)'])]
        for name, flags in operations:
            if args.operations and name not in args.operations.split(','):
                continue
            print(f'=== {name} ===', flush=True)
            if name == 'bad-evaluate':
                (root / 'kernel.py').write_text(REFERENCE.replace('x + 1', 'x + 2'))
            try:
                result = subprocess.run(command + flags, env=env, capture_output=True, text=True, timeout=420)
            finally:
                (root / 'kernel.py').write_text(KERNEL)
            if args.output:
                output = Path(args.output)
                output.mkdir(parents=True, exist_ok=True)
                (output / (name + '.json')).write_text(json.dumps({
                    'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}))
            expected_failure = name in {'bad-evaluate', 'timeout'}
            if bool(result.returncode) != expected_failure:
                print(result.stdout[-10000:], flush=True)
                print(result.stderr[-6000:], file=sys.stderr)
                raise RuntimeError(f'{name} failed with {result.returncode}')
            if name == 'bad-evaluate':
                assert 'RESULT_JSON=' in result.stdout and '"all_pass": false' in result.stdout
            if name == 'dev':
                assert (root / 'scratch/ssh-check/result.txt').read_text() == '4.0'
            print(name + ': PASS', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--hardware', default='MI308X')
    parser.add_argument('--runtime-bind', action='append', default=[])
    parser.add_argument('--init', default='')
    parser.add_argument('--bench')
    parser.add_argument('--operations')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.host:
        if args.gpu is None or not args.bench:
            parser.error('--host requires --gpu and --bench')
        smoke(args)
    else:
        unittest.main(argv=[sys.argv[0]])
