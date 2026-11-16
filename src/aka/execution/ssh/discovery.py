"""Observe the assigned SSH namespace without submitting a workload job."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from aka.contracts.execution_environment import environment_document

def route_identity(environment, target):
    from . import transport
    from types import SimpleNamespace
    args = SimpleNamespace(ssh=target, ssh_gpu=environment.get('ATREX_SANDBOX_SSH_GPU'),
                           ssh_init=environment.get('ATREX_SANDBOX_SSH_INIT', ''),
                           ssh_runtime_bind=None, workspace='.')
    identity = transport.endpoint_identity(args, environment)
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def discover(environment, target, query):
    # Run the probe client in its selected environment, avoiding global mutation
    # when separate Runtime allocations query different SSH routes concurrently.
    request = {'target': target, 'hardware': query.hardware,
               'parameters': environment_document(query.parameters)}
    try:
        process = subprocess.run([sys.executable, '-m', __name__],
            input=json.dumps(request), capture_output=True, text=True,
            env=dict(environment), timeout=90)
        envelope = json.loads(process.stdout.splitlines()[-1])
        if process.returncode or not isinstance(envelope, dict):
            return {}, ('ssh-environment-probe-rejected',)
        return envelope.get('result', {}), tuple(envelope.get('diagnostics', ()))
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
        return {}, ('ssh-environment-probe-unavailable',)


def main():
    from aka.execution.gateway import sandbox
    from . import environment as probe
    request = json.loads(sys.stdin.read())
    parameters = request['parameters']
    allowed = {'frameworks', 'limits', 'capabilities', 'counters', 'force'}
    if set(parameters) - allowed:
        raise ValueError('Unsupported Gateway-only SSH environment options')
    raw_gpu = os.environ.get('ATREX_SANDBOX_SSH_GPU', '')
    index = sandbox._ssh_gpu_index(raw_gpu)
    binds = sandbox._environment_ssh_runtime_binds()
    init = os.environ.get('ATREX_SANDBOX_SSH_INIT', '')
    query = {'hardware': request['hardware'], **parameters}
    import shlex
    command = shlex.join(['python3', '-c', Path(probe.__file__).read_text(), json.dumps(query)])
    try:
        process = sandbox._run_ssh_health(request['target'], init, command, binds, index)
    except sandbox.SSHGPUBusy:
        print(json.dumps({'result': {}, 'diagnostics': ['ssh_gpu_busy']}))
        return 0
    except sandbox.SSHDependencyError:
        print(json.dumps({'result': {}, 'diagnostics': ['ssh-flock-unsupported']}))
        return 0
    except sandbox.SSHTransportError:
        print(json.dumps({'result': {}, 'diagnostics': ['ssh-environment-probe-unavailable']}))
        return 0
    if process.returncode == 255:
        print(json.dumps({'result': {}, 'diagnostics': ['ssh-environment-probe-unavailable']}))
        return 0
    if process.returncode:
        print(json.dumps({'result': {}, 'diagnostics': ['ssh-environment-probe-rejected']}))
        return 0
    try:
        result = json.loads(process.stdout.splitlines()[-1])
        if not isinstance(result, dict):
            raise ValueError('SSH environment probe must return an object')
    except (ValueError, IndexError):
        print(json.dumps({'result': {}, 'diagnostics': ['ssh-environment-probe-rejected']}))
        return 0
    identity = {'route': route_identity(os.environ, request['target']),
                'facts': {key: value for key, value in result.items()
                          if key not in {'counters', 'counters_truncated', 'probed_at'}}}
    result['environment_identity'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    print(json.dumps({'result': result, 'diagnostics': []}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
