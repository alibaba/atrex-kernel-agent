"""Hardware and toolchain observations inside the constrained SSH namespace."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys


def _amd_counter_names(output, index):
    agents = {}
    current = None
    for line in output.splitlines():
        key, separator, value = line.partition(':')
        if separator and key.strip() == 'GPU':
            current = value.strip()
        elif current is not None and key.strip() == 'Counter_Name':
            agents.setdefault(current, []).append(value.strip())
    # Device isolation can renumber the sole visible agent to zero. With more
    # than one agent, require the assigned physical index; never pick arbitrarily.
    names = next(iter(agents.values())) if len(agents) == 1 else agents.get(index)
    if not names:
        raise RuntimeError('rocprofv3 returned no counters for the assigned GPU')
    return '\n'.join(dict.fromkeys(names))


def environment(request):
    import torch
    props = torch.cuda.get_device_properties(0)
    hip = getattr(torch.version, 'hip', None)
    vendor = 'amd' if hip else os.environ.get('ATREX_SSH_GPU_VENDOR') or (
        'ppu' if request['hardware'].upper().startswith(('PPU', 'ZW')) else 'nvidia')
    result = {'gpu': request['hardware'], 'vendor': vendor, 'gpu_model': props.name,
              'arch': props.gcnArchName.split(':')[0] if hip else 'sm_%d%d' % torch.cuda.get_device_capability(0),
              'sm_count': props.multi_processor_count, 'total_memory_mb': props.total_memory // 2**20,
              'toolchain': {'torch': torch.__version__, 'runtime': hip or torch.version.cuda}}
    result['gpu_uuid'] = os.environ.get('ATREX_SSH_GPU_UUID')
    result['device_selector'] = {name: os.environ[name] for name in ('CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES')
                                 if name in os.environ}
    from importlib.metadata import distributions, version, PackageNotFoundError
    result['libraries'] = {item.metadata['Name']: item.version for item in distributions()
                           if item.metadata.get('Name')}
    result['frameworks'] = {}
    for module, package in (('torch', 'torch'), ('triton', 'triton'), ('tilelang', 'tilelang'),
                            ('flydsl', 'flydsl'), ('cutlass', 'nvidia-cutlass-dsl')):
        available = importlib.util.find_spec(module) is not None
        try:
            revision = version(package)
        except PackageNotFoundError:
            revision = 'unknown'
        result['frameworks'][module] = {'available': available, 'supported': available, 'version': revision}
    nvcc = shutil.which('nvcc')
    result['frameworks']['cuda_cpp'] = {'available': bool(nvcc) and vendor == 'nvidia',
                                         'supported': bool(nvcc) and vendor == 'nvidia'}
    for name in ('nvcc', 'ptxas'):
        executable = shutil.which(name)
        info = {'available': bool(executable), 'version': 'unavailable'}
        if executable:
            process = subprocess.run([executable, '--version'], capture_output=True, text=True, timeout=10)
            info['version'] = process.stdout.strip() if process.returncode == 0 else 'unknown'
        result['toolchain'][name] = info
    result.update(source='probe', ready=True, max_gpus=1, requirements_supported=False,
                  features={'source_archives': True},
                  operations=['run', 'same_allocation_abba', 'profile', 'check', 'disassemble', 'dev'])
    if request.get('limits'):
        result['limits'] = {'num_gpus': 1, 'network': False, 'dependencies': 'preinstalled'}
    if request.get('capabilities'):
        candidates = {'ncu': 'ncu', 'rocprofv3': 'rocprofv3',
                      'acu': '/usr/local/PPU_SDK/asight/bin/acu'}
        result['profilers'] = [name for name, executable in candidates.items() if shutil.which(executable)]
    if request.get('counters'):
        tool = request['counters']
        commands = {'ncu': ['ncu', '--query-metrics'], 'rocprofv3': ['rocprofv3', '--list-avail'],
                    'acu': ['/usr/local/PPU_SDK/asight/bin/acu', '--query-metrics']}
        if tool not in commands:
            raise ValueError(f'Unknown profiler: {tool}')
        output = subprocess.run(commands[tool], capture_output=True, text=True,
                                timeout=30, check=True).stdout
        if tool == 'rocprofv3':
            # The SDK lists every physical agent despite ROCR visibility. Keep
            # the assigned GPU's counter names, not repeated full descriptions.
            output = _amd_counter_names(output, os.environ['ATREX_SSH_GPU_INDEX'])
        # Keep Env usable through the Agent's bounded response channel even on
        # toolchains with very large metric catalogs. Never imply completeness.
        result['counters'] = output[:65536]
        result['counters_truncated'] = len(output) > 65536
    return result



if __name__ == "__main__":
    print(json.dumps(environment(json.loads(sys.argv[1]))))
