"""Standalone stdlib runner uploaded to one task-private Remote directory."""
from __future__ import annotations
import base64
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def live(handle):
    try:
        fields = Path('/proc', str(handle['pid']), 'stat').read_text().split()
        return fields[21] == handle['start_ticks'] and fields[2] != 'Z'
    except FileNotFoundError:
        return False


def status(root):
    if (root/'result.json').exists():
        return {'state': 'finished', 'result': json.loads((root/'result.json').read_text())}
    if (root/'handle.json').exists():
        return {'state': 'running' if live(json.loads((root/'handle.json').read_text())) else 'lost'}
    return {'state': 'unknown' if (root/'claimed').exists() else 'prepared'}


def supervise(root):
    spec = json.loads((root/'spec.json').read_text())
    result = {'exit_code': None, 'timed_out': False, 'cancelled': False}
    child = None
    try:
        with (root/'stdout.log').open('wb') as out, (root/'stderr.log').open('wb') as err:
            child = subprocess.Popen(spec['argv'], cwd=root/'workspace',
                env={**os.environ, **spec['environment']}, stdin=subprocess.DEVNULL,
                stdout=out, stderr=err, start_new_session=True)
            deadline = time.monotonic() + spec['timeout']
            while child.poll() is None:
                result['cancelled'] = (root/'cancel').exists()
                result['timed_out'] = time.monotonic() >= deadline
                if result['cancelled'] or result['timed_out']:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    break
                time.sleep(.2)
            result['exit_code'] = child.wait()
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
    finally:
        if child is not None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        atomic(root/'result.json', result)


def dispatch(root, operation, args):
    if operation == 'part':
        number, data = args
        path = root/f'part-{int(number)}'
        blob = base64.b64decode(data, validate=True)
        # Replacement is idempotent even if the transport response is lost.
        temporary = path.with_suffix('.tmp')
        temporary.write_bytes(blob)
        temporary.replace(path)
        return {'bytes': len(blob)}
    if operation == 'status':
        return status(root)
    if operation == 'cancel':
        (root/'cancel').touch()
        return status(root)
    if operation == 'start':
        state = status(root)
        if state['state'] != 'prepared':
            return state
        spec = json.loads((root/'spec.json').read_text())
        archive = b''.join((root/f'part-{i}').read_bytes() for i in range(spec['parts']))
        if hashlib.sha256(archive).hexdigest() != spec['archive_sha256']:
            raise ValueError('Source archive hash mismatch')
        (root/'source.zip').write_bytes(archive)
        workspace = root/'workspace'
        workspace.mkdir(exist_ok=True)
        with zipfile.ZipFile(root/'source.zip') as z:
            for info in z.infolist():
                target = (workspace/info.filename).resolve()
                if not target.is_relative_to(workspace.resolve()):
                    raise ValueError('Archive path escapes workspace')
            z.extractall(workspace)
        try:
            fd = os.open(root/'claimed', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return status(root)
        os.close(fd)
        with (root/'runner.log').open('wb') as log:
            process = subprocess.Popen([sys.executable, str(root/'runner.py'), str(root), 'run'],
                stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        fields = Path('/proc', str(process.pid), 'stat').read_text().split()
        atomic(root/'handle.json', {'pid': process.pid, 'start_ticks': fields[21]})
        return status(root)
    if operation in {'stat', 'read'}:
        name = args[0]
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Artifact path escapes job directory')
        if operation == 'stat':
            if not path.is_file():
                return {'exists': False}
            return {'exists': True, 'bytes': path.stat().st_size}
        offset, size = map(int, args[1:])
        if offset < 0 or not 0 <= size <= 500000:
            raise ValueError('Invalid artifact slice')
        with path.open('rb') as stream:
            stream.seek(offset)
            data = stream.read(size)
        return {'base64': base64.b64encode(data).decode()}
    raise ValueError('Unknown Remote operation')


if __name__ == '__main__':
    root = Path(sys.argv[1])
    if sys.argv[2] == 'run':
        supervise(root)
    else:
        try:
            value = dispatch(root, sys.argv[2], sys.argv[3:])
        except Exception as error:
            value = {'error': f'{type(error).__name__}: {error}'}
        print('AKA_REMOTE=' + json.dumps(value), flush=True)
