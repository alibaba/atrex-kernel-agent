"""Build a native manifest into an Atrex-Bench Model inside one SSH job."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess


def materialize(manifest, request, root):
    from ssh_worker import write
    targets = manifest.get('targets', {})
    spec = request.get('spec', {})
    reference = request.get('reference', request.get('diagnostic_reference', {}))
    if ('hardware' in targets and not set(spec.get('target_hardware', [])) <= set(targets['hardware'])
            or 'num_gpus' in targets and targets['num_gpus'] != spec.get('num_gpus', 1)
            or 'shape_ids' in targets and not set(reference.get('shapes', {})) <= set(targets['shape_ids'])):
        raise ValueError('Native manifest targets do not cover the request')
    source = manifest['source']
    files = request['__atrex_eval_archives'][source['archive']]
    for name, digest in source['files'].items():
        text = files[name]
        if hashlib.sha256(text.encode()).hexdigest() != digest:
            raise ValueError('Native source digest mismatch')
        write(root, name, text)
    includes = []
    for dependency in manifest.get('dependencies', []):
        kind = dependency['kind']
        if kind == 'pip':
            from ssh_worker import requirements
            requirements({'requirements': [dependency['requirement']]})
        elif kind == 'git':
            name = dependency['id']
            if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
                raise ValueError('Invalid dependency id')
            directory = Path('/opt/aka-dependencies') / name
            def git(*args):
                return subprocess.check_output(['git', '-C', str(directory), *args], text=True).strip()
            ref = dependency['ref']
            ref = ref if re.fullmatch(r'[a-fA-F0-9]{40}', ref) or ref.startswith('refs/tags/') else 'refs/tags/' + ref
            commit = git('rev-parse', '--verify', ref + '^{commit}')
            if git('rev-parse', 'HEAD') != commit or dependency.get('commit', commit) != commit:
                raise ValueError(f'Provisioned dependency {name} does not match its pinned revision')
            if git('status', '--porcelain'):
                raise ValueError(f'Provisioned dependency {name} is modified')
            for relative in dependency.get('include_dirs', []):
                path = (directory / relative).resolve()
                path.relative_to(directory.resolve())
                if not path.is_dir():
                    raise ValueError(f'Dependency include directory is missing: {name}/{relative}')
                includes.append(str(path))
        elif kind != 'system':
            raise ValueError(f'SSH native dependency kind is not provisionable: {kind}')
    entry = manifest['entry_point']
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', entry['symbol']):
        raise ValueError('Invalid CUDA launcher symbol')
    build = entry['build']
    if build.get('toolchain') != 'torch_extension':
        raise ValueError('Native SSH compilation requires torch_extension')
    for name in build['sources']:
        if name not in source['files']:
            raise ValueError('Native build source was not declared')
    for name in build.get('include_dirs', []):
        (root / name).resolve().relative_to(root.resolve())
    write(root, 'native-entry.json', json.dumps(entry))
    # The build runs once in this allocation; ABBA subprocesses load its library
    # without including build time in their steady-state measurements.
    from torch.utils.cpp_extension import load
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:20]
    directory = root / '.build'
    directory.mkdir()
    library = load(name='aka_' + digest, sources=[str(root / name) for name in build['sources']],
                   extra_cflags=build.get('cflags', []), extra_cuda_cflags=build.get('cuda_cflags', []),
                   extra_ldflags=build.get('ldflags', []),
                   extra_include_paths=[str(root / name) for name in build.get('include_dirs', [])] + includes,
                   build_directory=str(directory), is_python_module=False)
    write(root, 'native-library.json', json.dumps(library))
    return write(root, 'kernel.py', ADAPTER)


ADAPTER = r'''
import ctypes
import json
from pathlib import Path
import torch

root = Path(__file__).parent
entry = json.loads((root / 'native-entry.json').read_text())
library = ctypes.CDLL(json.loads((root / 'native-library.json').read_text()))
launcher = getattr(library, entry['symbol'])
launcher.restype = None
scalar_types = {'int64': ctypes.c_int64, 'float32': ctypes.c_float,
                'float64': ctypes.c_double, 'bool': ctypes.c_bool}

class Model(torch.nn.Module):
    def forward(self, *args, **kwargs):
        if len(args) > len(entry['inputs']):
            raise TypeError('Too many native inputs')
        values = dict(zip((item['name'] for item in entry['inputs']), args))
        if values.keys() & kwargs.keys():
            raise TypeError('Duplicate native input')
        values.update(kwargs)
        if set(values) != {item['name'] for item in entry['inputs']}:
            raise TypeError('Native inputs do not match the manifest')
        device = None
        types = {}
        for item in entry['inputs']:
            value = values[item['name']]
            if item['kind'] == 'tensor':
                if not value.is_cuda or not value.is_contiguous() or value.dtype != getattr(torch, item['dtype']):
                    raise ValueError('Native tensor device, dtype or layout does not match the manifest')
                if device is not None and value.device != device:
                    raise ValueError('Native tensors must share a device')
                device = value.device
                types[item['name']] = ctypes.c_void_p
            else:
                types[item['name']] = scalar_types[item['dtype']]
        if device is None:
            raise ValueError('A native launcher requires a tensor device')
        outputs = []
        for item in entry['outputs']:
            like = values[item['like']]
            shape = item.get('shape', like.shape)
            shape = [values[dim['input']].shape[dim['dim']] if isinstance(dim, dict) else dim for dim in shape]
            value = torch.empty(shape, dtype=getattr(torch, item['dtype']) if item.get('dtype') else like.dtype,
                                device=device)
            outputs.append(value)
            values[item['name']] = value
            types[item['name']] = ctypes.c_void_p
        actual, signature = [], []
        for item in entry['arguments']:
            if 'ref' in item:
                name = item['ref']
                value = values[name]
                kind = types[name]
                actual.append(kind(value.data_ptr() if kind is ctypes.c_void_p else value))
                signature.append(kind)
            else:
                value = values[item.get('numel', item.get('size'))]
                actual.append(ctypes.c_int64(value.numel() if 'numel' in item else value.shape[item['dim']]))
                signature.append(ctypes.c_int64)
        launcher.argtypes = signature + [ctypes.c_void_p]
        with torch.cuda.device(device):
            launcher(*actual, ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream))
        return outputs[0] if len(outputs) == 1 else tuple(outputs)
'''
