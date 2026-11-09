"""Carry Light's typed requests over the isolated OpenSSH executor."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile


def endpoint_identity(args, environment=None):
    from aka.legacy.execution.runtime_tools import local_eval
    environment = os.environ if environment is None else environment
    binds = args.ssh_runtime_bind
    if binds is None:
        binds = json.loads(environment.get('ATREX_SANDBOX_SSH_RUNTIME_BINDS', '[]'))
    settings = [args.ssh, args.ssh_gpu, args.ssh_init, binds,
                *(environment.get(name) for name in ('ATREX_SANDBOX_SSH_GPU_ISOLATION',
                  'ATREX_SANDBOX_SSH_DEVICE_MAP', 'ATREX_SANDBOX_SSH_BWRAP'))]
    digest = hashlib.sha256()
    files = [Path(__file__).with_name(name) for name in (
        'ssh_worker.py', 'ssh_native.py', 'ssh_devices.py', 'source_diagnostics.py', 'abba_remote.py')]
    files.append(Path(local_eval.__file__))
    runtime = Path(environment.get('ATREX_BENCH_RUNTIME_ROOT',
                                  str(Path(getattr(args, 'workspace', '.')) / 'atrex-bench')))
    evaluator = sorted((runtime / 'src/atrex_bench').rglob('*.py'))
    if (runtime / 'scripts/run_eval.py').is_file():
        evaluator.append(runtime / 'scripts/run_eval.py')
    for path in files + evaluator:
        name = path.name if path in files else path.relative_to(runtime).as_posix()
        digest.update(name.encode() + b'\0' + path.read_bytes())
    return {'transport': 'ssh_native_eval', 'runtime_sha256': digest.hexdigest(),
            'endpoint_sha256': hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()}


def run(gateway, args, workspace, request, kind):
    from aka.legacy.execution.runtime_tools import local_eval
    from aka.legacy.evaluation.resources import ncu_profile_script
    from aka.legacy.execution import resources
    from . import ssh_worker, ssh_native, source_diagnostics, abba_remote

    if request.get('placement') or request.get('options', {}).get('shape_shards') not in (None, 1):
        raise ValueError('SSH owns one fixed GPU; Gateway placement and distributed sharding do not apply')
    if kind in {'check', 'profile', 'disassemble'}:
        request = dict(request, options=dict(request.get('options', {}), timeout_s=args.timeout))

    with tempfile.TemporaryDirectory(prefix='aka-ssh-typed-') as directory:
        root = Path(directory)
        uploads = root / 'upload'
        uploads.mkdir()
        files = {
            'request.json': json.dumps({'kind': kind, 'request': request}),
            'ssh_worker.py': Path(ssh_worker.__file__).read_text(),
            'ssh_native.py': Path(ssh_native.__file__).read_text(),
            'local_eval.py': Path(local_eval.__file__).read_text(),
            '__atrex_diagnostic.py': Path(source_diagnostics.__file__).read_text(),
            '__atrex_abba.py': Path(abba_remote.__file__).read_text(),
            'ncu_profile.py': ncu_profile_script().read_text(),
            'profile_ppu.py': (Path(resources.__file__).parent / 'profiling/ppu/profile_ppu.py').read_text(),
        }
        archive = uploads / 'payload.tar.gz'
        with tarfile.open(archive, 'w:gz') as bundle:
            for name, text in files.items():
                data = text.encode()
                entry = tarfile.TarInfo(name)
                entry.size, entry.mode = len(data), 0o600
                bundle.addfile(entry, io.BytesIO(data))
        runtime = gateway._make_atrex_bench_runtime_bundle(workspace) if kind == 'run' else None
        if runtime:
            (uploads / 'runtime.tar.gz').write_bytes(base64.b64decode(runtime))
        runner = uploads / '__atrex_runner.sh'
        runner.write_text(
            'set -eu\n'
            'tar -xzf payload.tar.gz\n'
            + ('tar -xzf runtime.tar.gz\nexport PYTHONPATH="$PWD/atrex-bench/src"\n' if runtime else '')
            + 'mkdir -p scratch\n'
            'python3 ssh_worker.py request.json\n'
            f'tar -czf {gateway.OSS_OUTPUT_ARCHIVE} scratch\n')
        try:
            process = gateway._run_ssh_job(
                target=args.ssh, init_command=args.ssh_init,
                runtime_binds=args.ssh_runtime_bind, gpu_index=args.ssh_gpu,
                timeout=args.timeout, env_items=args.env,
                upload_paths=list(uploads.iterdir()), temp=root, workspace=root,
                sync_outputs=True, sync_paths=['scratch'],
                max_output_file_bytes=args.max_output_file_mb * 1024 * 1024)
        except gateway.SSHTransportError as exc:
            gateway._record_environment_failure(target=args.ssh, stage='transport', detail=str(exc))
            raise gateway.EvidenceUnavailable('SSH request transport failed') from exc
        receipt = root / 'scratch' / 'job.json'
        if not receipt.is_file():
            _check_health(gateway, args)
            raise gateway.EvidenceUnavailable('SSH request returned no complete result: ' + process.stderr[-1000:])
        job = json.loads(receipt.read_text())
        if job['status'] != 'succeeded':
            _check_health(gateway, args)
        job['transport'] = endpoint_identity(args)
        return subprocess.CompletedProcess(process.args, 0 if job['status'] == 'succeeded' else 1,
                                           json.dumps(job), process.stderr)


def _check_health(gateway, args):
    try:
        health = gateway._run_ssh_health(args.ssh, args.ssh_init, args.health_command,
                                         args.ssh_runtime_bind, args.ssh_gpu)
    except gateway.SSHTransportError as exc:
        gateway._record_environment_failure(target=args.ssh, stage='health', detail=str(exc))
        raise gateway.EvidenceUnavailable('SSH health probe unavailable after a failed request') from exc
    if health.returncode:
        gateway._record_environment_failure(target=args.ssh, stage='health',
                                            detail=health.stderr, health_status=health.returncode)
        raise gateway.EvidenceUnavailable('SSH GPU environment unavailable after a failed request')


def environment(gateway, args):
    if args.env_gpu and args.env_gpu != args.hardware:
        raise ValueError('SSH environment queries must name the assigned hardware')
    request = {
        'hardware': args.hardware, 'frameworks': args.frameworks or args.env_capabilities,
        'counters': args.counters, 'limits': args.limits or args.env_capabilities,
        'capabilities': args.env_capabilities,
    }
    if args.dry_run:
        print(json.dumps({'kind': 'env', 'transport': 'ssh', **request}))
        return 0
    process = run(gateway, args, Path(args.workspace).resolve(), request, 'env')
    job = json.loads(process.stdout)
    if job['status'] != 'succeeded':
        from .projection import emit_gateway_error
        emit_gateway_error(process, 'env')
        return process.returncode
    print('[sandbox] ENV_JSON=' + json.dumps(job['result']))
    return 0
