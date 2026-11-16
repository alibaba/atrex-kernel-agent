"""Carry prepared measurements over the isolated OpenSSH executor."""
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
from aka.contracts.execution import ExecutionUnavailable


def endpoint_identity(args, environment=None):
    from aka.execution.runtime_tools import local_eval
    environment = os.environ if environment is None else environment
    binds = args.ssh_runtime_bind
    if binds is None:
        binds = json.loads(environment.get('ATREX_SANDBOX_SSH_RUNTIME_BINDS', '[]'))
    settings = [args.ssh, args.ssh_gpu, args.ssh_init, binds,
                *(environment.get(name) for name in ('ATREX_SANDBOX_SSH_GPU_ISOLATION',
                  'ATREX_SANDBOX_SSH_DEVICE_MAP', 'ATREX_SANDBOX_SSH_BWRAP'))]
    digest = hashlib.sha256()
    from aka.evaluation.drivers import ssh_worker, ssh_native
    from aka.evaluation import source_diagnostics, diagnostic_clocks
    from . import devices, environment as environment_probe
    files = [Path(module.__file__) for module in (ssh_worker, ssh_native, devices, environment_probe, source_diagnostics, diagnostic_clocks)]
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
    from aka.execution.runtime_tools import local_eval
    from aka.evaluation.resources import ncu_profile_script
    from aka.execution.resources import profiling_resource
    from aka.evaluation.drivers import ssh_worker, ssh_native
    from aka.evaluation import source_diagnostics, diagnostic_clocks as abba_remote

    if request.get('placement') or request.get('options', {}).get('shape_shards') not in (None, 1):
        raise ValueError('SSH owns one fixed GPU; Gateway placement and distributed sharding do not apply')
    if request.get('spec', {}).get('num_gpus', 1) != 1:
        raise ValueError('SSH owns exactly one GPU')
    from aka.execution.gateway.native_options import catalog, COMMAND_BY_KIND
    supplied = {option.dest: request[option.dest] for option in catalog().values()
                if COMMAND_BY_KIND.get(kind) in option.commands and option.dest in request}
    validate_native(supplied, kind)
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
            'profile_ppu.py': profiling_resource('ppu/profile_ppu.py').read_text(),
            'classify_ncu.py': profiling_resource('nvidia/classify_ncu.py').read_text(),
            'ncu_classification.py': profiling_resource('nvidia/ncu_classification.py').read_text(),
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
                max_output_file_bytes=args.max_output_file_bytes or args.max_output_file_mb * 1024 * 1024,
                capture_outputs=True, operation=kind)
        except gateway.SSHTransportError as exc:
            gateway._record_environment_failure(target=args.ssh, stage='transport', detail=str(exc))
            raise ExecutionUnavailable('SSH request transport failed: ' + str(exc)) from exc
        outputs = getattr(process, 'output_files', ())
        for name, content, mode in outputs:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        receipt = root / 'scratch' / 'job.json'
        if not receipt.is_file():
            _check_health(gateway, args)
            raise ExecutionUnavailable('SSH request returned no complete result: ' + process.stderr[-1000:])
        job = json.loads(receipt.read_text())
        job['job_id'] = getattr(process, 'remote_id', job.get('job_id'))
        if job['status'] != 'succeeded':
            _check_health(gateway, args)
        job['transport'] = endpoint_identity(args)
        result = subprocess.CompletedProcess(process.args, 0 if job['status'] == 'succeeded' else 1,
                                             json.dumps(job), process.stderr)
        result.output_files = tuple(item for item in outputs if item[0] != 'scratch/job.json')
        return result


def validate_native(values, kind):
    allowed = {'frameworks', 'limits', 'counters'} if kind == 'env' else {
        'profiler', 'level', 'counters', 'kernel_regex', 'kernel_name', 'source',
        'launch_skip', 'launch_count', 'top_kernels', 'arch', 'sanitize', 'fmt'}
    unsupported = set(values) - allowed
    if values.get('shape_shards') == 1:
        unsupported.discard('shape_shards')
    if unsupported:
        raise ValueError('Unsupported Gateway-only SSH options: ' + ', '.join(sorted(unsupported)))


def _check_health(gateway, args):
    try:
        health = gateway._run_ssh_health(args.ssh, args.ssh_init, args.health_command,
                                         args.ssh_runtime_bind, args.ssh_gpu)
    except gateway.SSHGPUBusy:
        # Another allocation may acquire the GPU before this post-job probe.
        # Preserve the job's evidence without reporting an environment failure.
        return
    except gateway.SSHTransportError as exc:
        gateway._record_environment_failure(target=args.ssh, stage='health', detail=str(exc))
        raise ExecutionUnavailable('SSH health probe unavailable after a failed request') from exc
    if health.returncode:
        gateway._record_environment_failure(target=args.ssh, stage='health',
                                            detail=health.stderr, health_status=health.returncode)
        raise ExecutionUnavailable('SSH GPU environment unavailable after a failed request')
