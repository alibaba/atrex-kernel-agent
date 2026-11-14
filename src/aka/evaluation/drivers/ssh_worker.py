"""Stateless typed GPU worker, shipped by the Supervisor into an SSH namespace."""
from __future__ import annotations

from contextlib import ExitStack
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import uuid


def write(root, name, text):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('source path must be relative')
    target = root.joinpath(*path.parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    return target


def materialize(source, request, root):
    if isinstance(source, str):
        return write(root, 'kernel.py', source)
    if source.get('schema_version'):
        from ssh_native import materialize as native
        return native(source, request, root)
    files = request['__atrex_eval_archives'][source['archive']]
    for name, text in files.items():
        write(root, name, text)
    entry = root / source['entry_point']
    entry.resolve().relative_to(root.resolve())
    if not entry.is_file():
        raise ValueError('candidate entry point is missing')
    return entry


def requirements(request):
    from importlib.metadata import version
    from packaging.requirements import Requirement
    for raw in request.get('requirements', []):
        item = Requirement(raw)
        if item.marker and not item.marker.evaluate():
            continue
        if item.url or version(item.name) not in item.specifier:
            raise RuntimeError(f'SSH runtime dependency is not provisioned: {item.name}{item.specifier}')


def evaluate(request, root):
    import local_eval
    requirements(request)
    if request.get('atrex_bench_version'):
        from atrex_bench import __version__
        if __version__ != request['atrex_bench_version']:
            raise RuntimeError(f'Provisioned Atrex-Bench version {__version__} does not match the requested pin')
    candidate = materialize(request['candidate'], request, root / 'candidate')
    def wrapper(entry, name):
        return write(root, name, 'import runpy, sys\n'
                     f'sys.path.insert(0, {str(entry.parent)!r})\n'
                     f'Model = runpy.run_path({str(entry)!r})["Model"]\n')
    candidate = wrapper(candidate, 'candidate.py')
    reference = root / 'reference'
    for field, name in (('reference_py', 'reference.py'), ('input_py', 'input.py'),
                        ('shapes', 'shapes.json'), ('metadata', 'metadata.json'), ('roofline', 'roofline.json')):
        if field in request['reference']:
            value = request['reference'][field]
            write(reference, name, value if isinstance(value, str) else json.dumps(value))
    if 'metadata' not in request['reference']:
        write(reference, 'metadata.json', '{}')
    options = request.get('options', {})
    config = dict(request.get('runner_overrides', {}), input=str(candidate), reference_dir=str(reference),
                  local_output=str(root / 'eval_output'), validation_mode=request.get('mode', 'full'))
    for name in ('atol', 'rtol', 'correctness_max_rel_l2', 'num_correctness_cases', 'bench_iters'):
        if name in options:
            config[name] = options[name]
    if request.get('abba'):
        baseline = materialize(request['abba']['baseline'], request, root / 'baseline')
        baseline = wrapper(baseline, 'baseline.py')
        config.update(eval_mode='abba', baseline_input=str(baseline),
                      local_abba_repeats=request['abba']['repeats'])
    runner = root / 'atrex-bench/scripts/run_eval.py'
    if not runner.is_file():
        write(root, 'run_eval.py', "from atrex_bench.cli.run_eval import main\nmain()\n")
        runner = root / 'run_eval.py'
    from __atrex_abba import _clock_lock
    import torch
    supported = (not getattr(torch.version, 'hip', None) and os.environ.get('ATREX_SSH_GPU_VENDOR') != 'ppu'
                 and shutil.which('nvidia-smi') is not None)
    lock_clocks = bool(request.get('lock_clocks', options.get('lock_clocks', False)))
    with ExitStack() as stack:
        try:
            clocks = stack.enter_context(_clock_lock(lock_clocks and supported))
        except RuntimeError as error:
            # An ordinary SSH account may run kernels without clock-control
            # privileges. Preserve that fact; evaluator failures and reset
            # failures still propagate instead of being swallowed here.
            if not any(reason in str(error).lower() for reason in (
                    'insufficient permission', 'not supported', 'no supported graphics clocks')):
                raise
            clocks = {'applied': False, 'backend': 'nvidia-smi', 'reason': str(error)}
        config['clock_locked'] = clocks['applied']
        path = write(root, 'eval-config.json', json.dumps(config))
        local_eval.run(path, runner)
    result = json.loads((root / 'eval_output/local-result.json').read_text())
    result['clock_lock'] = dict(clocks, requested=lock_clocks)
    return result


def diagnostic(request, root, kind):
    requirements(request)
    candidate = materialize(request['candidate'], request, root / 'candidate')
    reference = request.get('reference', request.get('diagnostic_reference'))
    if not isinstance(reference, dict) or not reference.get('input_py') or not reference.get('shapes'):
        raise ValueError('SSH diagnostics require frozen Task input.py and Shapes')
    write(root, 'input.py', reference['input_py'])
    parameters = {name: request[name] for name in (
        'profiler', 'level', 'counters', 'kernel_regex', 'kernel_name', 'source',
        'launch_skip', 'launch_count', 'top_kernels', 'arch', 'sanitize', 'fmt') if name in request}
    payload = {'operation': kind, 'package_root': '.',
               'entrypoint': str(candidate.relative_to(root / 'candidate')),
               'runtime_requirements': [], 'requirements': request.get('requirements', []),
               'shapes': reference['shapes'], 'parameters': parameters,
               'lock_clocks': False, 'timeout_s': request.get('options', {}).get('timeout_s', 600)}
    hardware = (request.get('spec', {}).get('target_hardware') or [''])[0]
    payload['vendor'] = os.environ.get('ATREX_SSH_GPU_VENDOR') or (
        'ppu' if hardware.upper().startswith(('PPU', 'ZW')) else None)
    path = write(root, 'request.json', json.dumps(payload))
    process = subprocess.run([sys.executable, str(root / '__atrex_diagnostic.py'), str(path)],
                             capture_output=True, text=True)
    prefix = '__ATREX_RUNTIME_SOURCE_DIAGNOSTIC__='
    for line in reversed(process.stdout.splitlines()):
        if line.startswith(prefix):
            return json.loads(line[len(prefix):])
    raise RuntimeError('Diagnostic produced no result: ' + process.stderr[-2000:])


def main():
    root = Path.cwd()
    envelope = json.loads(Path(sys.argv[1]).read_text())
    kind, request = envelope['kind'], envelope['request']
    job = {'job_id': 'ssh-' + uuid.uuid4().hex, 'kind': 'eval' if kind == 'run' else kind}
    try:
        if request.get('spec', {}).get('num_gpus', 1) != 1:
            raise ValueError('This SSH allocation owns one GPU')
        if kind == 'run':
            result = evaluate(request, root)
        else:
            result = diagnostic(request, root, kind)
        job.update(status='succeeded', result=result)
    except Exception as exc:
        job.update(status='failed', error={'reason': 'ssh_execution_failed', 'message': str(exc)}, result={})
    write(root, 'scratch/job.json', json.dumps(job))


if __name__ == '__main__':
    main()
