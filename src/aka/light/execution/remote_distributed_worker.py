"""Standalone single-node distributed launcher and frozen-harness protocol.

Installed in a content-addressed directory as aka_distributed.py. Harnesses use
stage(), initialize() and report(); the broker alone owns GPU leases/retries.
"""
from contextlib import contextmanager
import json
import hashlib
import math
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import time
import uuid

PHASES = ('startup', 'compile', 'initialize', 'warmup', 'capture', 'correctness', 'measure', 'complete')
DEFAULT_TIMEOUTS = dict(startup=120, compile=900, initialize=120, warmup=120,
                        capture=120, correctness=120, measure=300, complete=60)
PROTECTED = {'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'GROUP_RANK',
             'ROLE_RANK', 'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT',
             'TORCHELASTIC_RUN_ID', 'TORCHELASTIC_RESTART_COUNT', 'TORCHELASTIC_USE_AGENT_STORE',
             'AKA_DISTRIBUTED_CONFIG', 'AKA_DISTRIBUTED_RANK', 'AKA_DISTRIBUTED_NONCE'}
_sequence = 0


def validate_options(value, gpu_count):
    allowed = {'launcher', 'world_size', 'phase_timeouts_s', 'bind_device', 'max_retries'}
    if not isinstance(value, dict) or set(value)-allowed:
        raise ValueError('Invalid distributed execution options')
    if value.get('launcher', 'torchrun') != 'torchrun':
        raise ValueError('Distributed execution requires the managed torchrun launcher')
    if type(value.get('world_size')) is not int or value['world_size'] != gpu_count:
        raise ValueError('distributed.world_size must equal scheduling.gpu_count')
    if type(value.get('bind_device', True)) is not bool:
        raise ValueError('Invalid distributed device binding')
    retries = value.get('max_retries', 2)
    if type(retries) is not int or not 0 <= retries <= 1000:
        raise ValueError('Invalid distributed retry limit')
    deadlines = value.get('phase_timeouts_s', {})
    if not isinstance(deadlines, dict) or set(deadlines)-set(PHASES):
        raise ValueError('Invalid distributed phase deadlines')
    if any(type(x) not in (int,float) or not math.isfinite(x) or not 0 < x <= 86400 for x in deadlines.values()):
        raise ValueError('Distributed phase deadlines must be finite positive seconds')
    return dict(launcher='torchrun',world_size=gpu_count,bind_device=value.get('bind_device',True),
                phase_timeouts_s={**DEFAULT_TIMEOUTS,**deadlines},max_retries=retries)


def atomic(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False);stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def configuration():
    value = read(os.environ['AKA_DISTRIBUTED_CONFIG'])
    if value['nonce'] != os.environ['AKA_DISTRIBUTED_NONCE']:
        raise RuntimeError('Distributed attempt identity mismatch')
    return value


def rank_root(config=None):
    config = config or configuration()
    return Path(config['directory']) / ('rank-' + os.environ['AKA_DISTRIBUTED_RANK'])


def stage(name):
    """Enter a phase together; the host releases only a matching all-rank barrier."""
    global _sequence
    if name not in PHASES:
        raise ValueError('Unknown distributed phase: ' + name)
    config = configuration();_sequence += 1
    identity = dict(nonce=config['nonce'], rank=int(os.environ['AKA_DISTRIBUTED_RANK']),
                    sequence=_sequence, phase=name, observed_at=time.time(), pid=os.getpid())
    atomic(rank_root(config) / 'state.json', identity)
    release = Path(config['directory']) / ('release-' + str(_sequence) + '.json')
    # A local bound also prevents a lost group controller from hanging ranks.
    deadline = time.monotonic() + sum(config['phase_timeouts_s'].values())
    while time.monotonic() < deadline:
        if release.exists():
            value = read(release)
            if value['nonce'] != config['nonce'] or value['phase'] != name:
                raise RuntimeError('Distributed phase identity mismatch')
            return
        if (Path(config['directory']) / 'abort.json').exists():
            raise RuntimeError('Distributed group aborted')
        time.sleep(.02)
    raise TimeoutError('Distributed phase controller disappeared')


def initialize(backend='nccl', **kwargs):
    """Initialize the launcher-provided group after the compile phase is ready."""
    import torch
    import torch.distributed as dist
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group(backend=backend, **kwargs)


def finalize(graphs=()):
    """Release captured communicator references before destroying NCCL groups."""
    import torch
    import torch.distributed as dist
    torch.cuda.synchronize()
    for graph in graphs:graph.reset()
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier();torch.cuda.synchronize();dist.destroy_process_group()


def report(passed, samples_us=(), *, details=None):
    """Publish this rank's correctness and aligned per-iteration GPU samples."""
    config = configuration()
    if type(passed) is not bool:
        raise ValueError('Rank correctness must be boolean')
    samples = list(samples_us)
    if any(type(x) not in (float, int) or not math.isfinite(x) or x < 0 for x in samples):
        raise ValueError('Rank samples must be finite nonnegative microseconds')
    atomic(rank_root(config) / 'result.json', dict(nonce=config['nonce'],
        rank=int(os.environ['AKA_DISTRIBUTED_RANK']), passed=passed,
        samples_us=samples, details=details or {}))


@contextmanager
def iteration(index):
    """Align one timing iteration and separately record its synchronized wall time.

    CUDA events around the operation belong to the frozen harness. This wall
    metric includes the completion barrier and must not replace event samples.
    """
    import torch
    import torch.distributed as dist
    config = configuration()
    torch.cuda.synchronize();dist.barrier();torch.cuda.synchronize()
    started = time.perf_counter()
    yield
    torch.cuda.synchronize();dist.barrier();torch.cuda.synchronize()
    atomic(rank_root(config) / ('iteration-' + str(index) + '.json'),
           dict(nonce=config['nonce'], rank=int(os.environ['AKA_DISTRIBUTED_RANK']),
                index=index, synchronized_wall_us=(time.perf_counter()-started)*1e6))


def launcher_command(config_path, world_size):
    return [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
            '--nproc-per-node=' + str(world_size), '--max-restarts=0',
            str(Path(__file__).resolve()), 'rank', str(config_path)]


def stop_launcher(child):
    # The broker supervisor tracks the full owned tree and verifies CUDA cleanup.
    # Avoid process-name matching and never signal an observed external CUDA PID.
    if child.poll() is None:
        child.terminate()
        try:child.wait(timeout=3)
        except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)


def run_rank(config_path):
    config = read(config_path);rank = int(os.environ['LOCAL_RANK'])
    if not 0 <= rank < config['world_size'] or int(os.environ['WORLD_SIZE']) != config['world_size']:
        raise RuntimeError('Launcher world size does not match the GPU lease')
    os.environ.update(AKA_DISTRIBUTED_CONFIG=str(config_path), AKA_DISTRIBUTED_RANK=str(rank),
                      AKA_DISTRIBUTED_NONCE=config['nonce'])
    root = rank_root(config);root.mkdir(parents=True, exist_ok=True)
    token=None
    if Path('/proc/self/stat').exists():
        raw=Path('/proc/self/stat').read_text();token=raw[raw.rindex(')')+2:].split()[19]
    atomic(root / 'identity.json', dict(nonce=config['nonce'], rank=rank, pid=os.getpid(),start_token=token,
                                      gpu_uuid=config['devices'][rank]['uuid']))
    code = 0
    with (root/'stdout.log').open('w', buffering=1) as stdout, (root/'stderr.log').open('w', buffering=1) as stderr:
        os.dup2(stdout.fileno(), 1);os.dup2(stderr.fileno(), 2)
        sys.stdout, sys.stderr = stdout, stderr
        try:
            launch_directory=Path.cwd()
            os.chdir(root/'workspace')
            # Python expands relative PYTHONPATH entries before this chdir.
            # Rebase them so every rank imports its own frozen source tree.
            relative_imports=[]
            for entry in os.environ.get('PYTHONPATH','').split(os.pathsep):
                if entry and not Path(entry).is_absolute():
                    previous=str((launch_directory/entry).resolve())
                    sys.path[:]=[p for p in sys.path if p!=previous]
                    relative_imports.append(str((root/'workspace'/entry).resolve()))
            sys.path[:0]=relative_imports
            sys.path.insert(0, str(root/'workspace'))
            # Make imports by the harness share this module's phase sequence.
            sys.modules['aka_distributed'] = sys.modules[__name__]
            atomic(root/'state.json', dict(nonce=config['nonce'], rank=rank, sequence=0,
                                            phase='startup', observed_at=time.time(), pid=os.getpid()))
            if config.get('bind_device', True):
                import torch
                torch.cuda.set_device(rank)
            argv = config['argv']
            python = Path(argv[0]).name.startswith('python') or argv[0] == sys.executable
            if python and len(argv) >= 2:
                if argv[1] == '-c':
                    sys.argv = ['-c', *argv[3:]];exec(compile(argv[2], '<rank-command>', 'exec'), {'__name__':'__main__'})
                elif argv[1] == '-m':
                    sys.argv = argv[2:];runpy.run_module(argv[2], run_name='__main__', alter_sys=True)
                elif not argv[1].startswith('-'):
                    sys.argv = argv[1:];runpy.run_path(argv[1], run_name='__main__')
                else:
                    raise ValueError('Python launcher flags must be expressed in a wrapper script')
            else:
                code = subprocess.call(argv)
        except SystemExit as error:
            code = error.code if type(error.code) is int else (0 if error.code is None else 1)
        except BaseException:
            import traceback
            traceback.print_exc();code = 1
        finally:
            atomic(root/'exit.json', dict(nonce=config['nonce'], rank=rank, exit_code=code))
            stdout.flush();stderr.flush()
    return code


def summarize(config):
    rows = []
    count = None
    for rank in range(config['world_size']):
        root = Path(config['directory'])/('rank-'+str(rank))
        result, exit_record = read(root/'result.json'), read(root/'exit.json')
        if (result.get('nonce') != config['nonce'] or exit_record.get('nonce') != config['nonce']
                or result.get('rank') != rank or exit_record.get('rank') != rank
                or result.get('passed') is not True or exit_record.get('exit_code') != 0):
            raise ValueError('Incomplete or failed rank receipt')
        samples = result.get('samples_us')
        if not isinstance(samples, list) or any(type(x) not in (int,float) or not math.isfinite(x) or x < 0 for x in samples):
            raise ValueError('Invalid rank timing samples')
        if count is not None and count != len(samples):
            raise ValueError('Ranks reported different timing iterations')
        count = len(samples)
        walls = []
        for index in range(len(samples)):
            path = root/('iteration-'+str(index)+'.json')
            if path.exists():
                wall = read(path)
                if wall.get('nonce') != config['nonce'] or wall.get('rank') != rank or wall.get('index') != index:
                    raise ValueError('Stale synchronized iteration receipt')
                value = wall.get('synchronized_wall_us')
                if type(value) not in (int,float) or not math.isfinite(value) or value < 0:
                    raise ValueError('Invalid synchronized wall sample')
                walls.append(value)
        if walls and len(walls) != len(samples):
            raise ValueError('Missing synchronized iteration receipt')
        rows.append({**result, 'gpu_uuid': config['devices'][rank]['uuid'], 'exit_code': 0,
                     'synchronized_wall_us': walls})
    if any(row['synchronized_wall_us'] for row in rows) and not all(row['synchronized_wall_us'] for row in rows):
        raise ValueError('Not all ranks measured synchronized wall time')
    return dict(complete=True, ranks=rows,
        max_rank_samples_us=[max(row['samples_us'][i] for row in rows) for i in range(count or 0)],
        synchronized_wall_samples_us=[max(row['synchronized_wall_us'][i] for row in rows)
            for i in range(count or 0)] if rows and rows[0]['synchronized_wall_us'] else [])


def launch(attempt, *, command_factory=None):
    attempt = Path(attempt);config = read(attempt/'distributed/config.json')
    directory = Path(config['directory']);config_path = directory/'config.json'
    command = (command_factory or launcher_command)(config_path, config['world_size'])
    record = dict(nonce=config['nonce'], fence=config['fence'], world_size=config['world_size'],
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  complete=False, retryable=False, phase='startup', started_at=time.time())
    child = None;active_phase='startup';released=0;phase_started=time.monotonic();history=[]
    try:
        with (directory/'launcher.log').open('ab') as output:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output, stderr=output)
        while True:
            states=[];exits=[]
            for rank in range(config['world_size']):
                root=directory/('rank-'+str(rank))
                for name in ['state.json','exit.json','result.json']:
                    path=root/name
                    if not path.exists():continue
                    value=read(path)
                    if value.get('nonce') != config['nonce'] or value.get('rank') != rank:
                        raise ValueError('Rank attempt identity mismatch')
                    if name=='state.json':states.append(value)
                    elif name=='exit.json':
                        exits.append(value)
                        if value['exit_code'] != 0:
                            record.update(failure_kind='rank_error', failed_rank=rank);raise RuntimeError('Rank failed')
                    elif value.get('passed') is not True:
                        record.update(failure_kind='correctness_failure', failed_rank=rank);raise RuntimeError('Rank correctness failed')
            if len(states)==config['world_size']:
                sequences={value['sequence'] for value in states}
                if any(value['sequence'] > released+1 or value['sequence'] < released for value in states):
                    raise ValueError('Invalid distributed phase sequence')
                if sequences=={released+1}:
                    phases={value['phase'] for value in states}
                    if len(phases)!=1 or not phases <= set(PHASES):raise ValueError('Ranks entered different phases')
                    active_phase=phases.pop();released+=1;phase_started=time.monotonic()
                    history.append(dict(sequence=released,phase=active_phase,started_at=time.time()))
                    atomic(directory/('release-'+str(released)+'.json'),dict(nonce=config['nonce'],phase=active_phase))
            atomic(directory/'progress.json', dict(nonce=config['nonce'],phase=active_phase,
                phase_elapsed_s=time.monotonic()-phase_started, ranks=states, exited=exits))
            if child.poll() is not None:
                if child.returncode != 0:
                    record.update(failure_kind='rank_lost' if not any(x['exit_code'] for x in exits) else 'rank_error',
                                  retryable=not any(x['exit_code'] for x in exits))
                    raise RuntimeError('Launcher or rank exited')
                if active_phase!='complete':raise ValueError('Ranks exited before the complete phase')
                record.update(summarize(config));break
            if time.monotonic()-phase_started > config['phase_timeouts_s'][active_phase]:
                record.update(failure_kind='phase_timeout',retryable=True,phase=active_phase)
                raise TimeoutError('Distributed phase deadline exceeded')
            time.sleep(.04)
        rank0=directory/'rank-0/stdout.log'
        if rank0.exists():sys.stdout.write(rank0.read_text(errors='replace'))
    except BaseException as error:
        record.update(error=type(error).__name__,failure_kind=record.get('failure_kind','protocol_error'))
        atomic(directory/'abort.json',dict(nonce=config['nonce'],reason=record['failure_kind']))
    finally:
        if child is not None:stop_launcher(child)
        record.update(finished_at=time.time(),phase=active_phase,phase_history=history,
                      launcher_exit_code=child.returncode if child is not None else None)
        # Preserve partial rank records on failure, including absent ranks.
        if not record['complete']:
            record['ranks']=[]
            for rank in range(config['world_size']):
                root=directory/('rank-'+str(rank));value={'rank':rank,'gpu_uuid':config['devices'][rank]['uuid']}
                for name in ['state','exit','result']:
                    path=root/(name+'.json')
                    if path.exists():value[name]=read(path)
                record['ranks'].append(value)
        atomic(directory/'group.json',record)
    return 0 if record['complete'] else 1


if __name__=='__main__':
    if sys.argv[1]=='rank':raise SystemExit(run_rank(Path(sys.argv[2])))
    if sys.argv[1]=='launch':raise SystemExit(launch(Path(sys.argv[2])))
    raise SystemExit('Unsupported distributed runner operation')
