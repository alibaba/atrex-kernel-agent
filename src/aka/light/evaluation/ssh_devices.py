"""Remote, standard-library-only physical GPU assignment for OpenSSH."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def resolve(index: int, isolation: str = 'device', device_map: dict | None = None) -> dict:
    if isolation not in {'device', 'runtime'}:
        raise ValueError('SSH GPU isolation must be device or runtime')
    if device_map is not None:
        assignment = dict(device_map[str(index)], index=index)
        for device in assignment['devices']:
            import stat
            if not device.startswith('/dev/') or not stat.S_ISCHR(Path(device).stat().st_mode):
                raise ValueError(f'Not a GPU character device: {device}')
        return assignment
    # KFD topology supplies the actual render minor: renderD128 + index is not
    # valid on partitioned GPUs or hosts with gaps in their device numbering.
    if Path('/dev/kfd').exists():
        devices = []
        for node in sorted(Path('/sys/class/kfd/kfd/topology/nodes').iterdir(),
                           key=lambda path: int(path.name)):
            values = dict(line.split() for line in (node / 'properties').read_text().splitlines())
            minor = int(values.get('drm_render_minor', 0))
            render = Path(f'/dev/dri/renderD{minor}')
            if minor and render.exists():
                devices.append((values, render))
        if index >= len(devices):
            raise ValueError(f'GPU index {index} not found in accessible KFD devices')
        values, render = devices[index]
        uuid = 'GPU-' + format(int(values['unique_id']), '016x')
        return {'vendor': 'amd', 'uuid': uuid, 'index': index,
                'devices': ['/dev/kfd', *([str(render)] if isolation == 'device' else
                                         [str(path) for path in sorted(Path('/dev/dri').glob('renderD*'))])],
                'environment': {'ROCR_VISIBLE_DEVICES': uuid}}
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,uuid,minor_number,mig.mode.current',
         '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    for line in result.stdout.splitlines():
        ordinal, uuid, minor, mig = (part.strip() for part in line.split(','))
        if ordinal != str(index):
            continue
        if mig.lower() == 'enabled':
            raise ValueError('MIG requires explicit capability-node assignment')
        return {'vendor': 'nvidia', 'uuid': uuid, 'index': index,
                'devices': [f'/dev/nvidia{int(minor)}',
                            *[str(path) for name in ('nvidiactl', 'nvidia-uvm',
                              'nvidia-uvm-tools', 'nvidia-modeset')
                              if (path := Path('/dev') / name).exists()]],
                'environment': {'CUDA_VISIBLE_DEVICES': uuid}}
    raise ValueError(f'GPU index {index} was not found')


if __name__ == '__main__':
    print(json.dumps(resolve(int(sys.argv[1]), sys.argv[2], json.loads(sys.argv[3]))))
