"""Standalone Linux GPU admission broker. Only this file is installed in the Pod.

The broker owns only its job directories and process groups. NVML observations
never authorize killing a foreign process. A completed CUDA command is not a
valid measurement if foreign compute was observed during its execution.
"""
import base64
from contextlib import contextmanager
import fcntl
import hashlib
import io
import importlib.util
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import uuid
import zipfile

PROTOCOL = 1
MAX_SOURCE = 64 * 1024 * 1024
TERMINAL = {'completed', 'failed', 'cancelled', 'blocked'}


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, sort_keys=True, separators=(',', ':'))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def inside(root, name):
    root = Path(root).resolve()
    target = (root / name).resolve()
    if not target.is_relative_to(root):
        raise ValueError('Path escapes the owned job directory')
    return target


def support_module(root,record,name):
    sha=record['sha256'];path=Path(record['path']).resolve()
    if not re.fullmatch('[a-f0-9]{64}',sha) or path!=(Path(root)/('support-'+sha)/name).resolve():raise ValueError('Invalid content support path')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('Content support checksum mismatch')
    spec=importlib.util.spec_from_file_location('aka_owned_content_'+sha,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def process_table():
    """Use PID start tokens, not numeric PIDs alone; Linux is the deployed path."""
    rows = {}
    if Path('/proc/self/stat').exists():
        for path in Path('/proc').glob('[0-9]*/stat'):
            try:
                raw = path.read_text()
                pid = int(raw[:raw.index(' ')])
                fields = raw[raw.rindex(')') + 2:].split()
                if fields[0] == 'Z':
                    continue
                host_pid = pid
                for line in path.with_name('status').read_text().splitlines():
                    if line.startswith('NSpid:'):
                        host_pid = int(line.split()[1])
                rows[pid] = dict(pid=pid, ppid=int(fields[1]), pgid=int(fields[2]),
                                 token=fields[19], host_pid=host_pid)
            except (OSError, ValueError, IndexError):
                continue
    else:  # Process-lifecycle unit tests on the macOS parent; never used for GPU discovery.
        response = subprocess.run(['ps', '-axo', 'pid=,ppid=,pgid=,stat=,lstart='],
                                  capture_output=True, text=True, check=True)
        for line in response.stdout.splitlines():
            fields = line.split()
            if len(fields) >= 9 and not fields[3].startswith('Z'):
                pid = int(fields[0])
                rows[pid] = dict(pid=pid, ppid=int(fields[1]), pgid=int(fields[2]),
                                 token=' '.join(fields[4:]), host_pid=pid)
    return rows


def live(handle, table=None):
    row = (process_table() if table is None else table).get(handle.get('pid'))
    return bool(row and row['token'] == handle.get('token'))


class ProbeUnavailable(RuntimeError):
    pass


class ResourceViolation(RuntimeError):
    pass


class NvmlProbe:
    def __init__(self):
        try:
            import pynvml
            self.nvml = pynvml
            pynvml.nvmlInit()
        except Exception:
            raise ProbeUnavailable('NVML process monitoring is unavailable') from None

    def snapshot(self):
        try:
            n = self.nvml
            rows = []
            for index in range(n.nvmlDeviceGetCount()):
                handle = n.nvmlDeviceGetHandleByIndex(index)
                device = n.nvmlDeviceGetUUID(handle)
                if isinstance(device, bytes):
                    device = device.decode()
                compute = n.nvmlDeviceGetComputeRunningProcesses(handle)
                graphics = n.nvmlDeviceGetGraphicsRunningProcesses(handle)
                rows.append(dict(index=index, uuid=device,
                                 pids=sorted({int(p.pid) for p in [*compute, *graphics]})))
            return rows
        except Exception:
            raise ProbeUnavailable('GPU process inventory could not be verified') from None

    def compatible(self, devices, topology):
        if topology=='any' or len(devices)<2:return True
        n=self.nvml
        try:
            names=(['NVML_P2P_CAPS_INDEX_NVLINK'] if topology=='nvlink' else
                   ['NVML_P2P_CAPS_INDEX_READ','NVML_P2P_CAPS_INDEX_WRITE'])
            caps=[]
            for name in names:
                value=getattr(n,name)
                # Some shipped pynvml bindings define READ as (0,) rather
                # than 0. Normalize that binding typo without skipping the
                # driver's capability check or assuming NVLink implies P2P.
                if isinstance(value,tuple) and len(value)==1:value=value[0]
                if type(value) is not int:raise ValueError('Invalid NVML capability enum')
                caps.append(value)
            for first,second in itertools.permutations(devices,2):
                a=n.nvmlDeviceGetHandleByIndex(first['index']);b=n.nvmlDeviceGetHandleByIndex(second['index'])
                if any(n.nvmlDeviceGetP2PStatus(a,b,cap)!=n.NVML_P2P_STATUS_OK for cap in caps):return False
            return True
        except Exception:
            raise ProbeUnavailable('Requested GPU topology cannot be verified') from None


def distributed_module(spec, root):
    path=Path(spec['distributed_runner']).resolve()
    sha=spec['distributed_runner_sha256']
    if not re.fullmatch('[a-f0-9]{64}',sha) or path != (Path(root)/('support-'+sha)/'aka_distributed.py').resolve():
        raise ValueError('Distributed runner path is not content-addressed under this pool')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('Distributed runner checksum mismatch')
    loader=importlib.util.spec_from_file_location('aka_owned_distributed',path)
    module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
    return module


class OwnedProcesses:
    def __init__(self, child):
        self.child = child
        self.known = {}
        self.host_pids = set()
        self.refresh()

    def refresh(self):
        table = process_table()
        root = table.get(self.child.pid)
        if root and self.child.poll() is None:
            self.known.setdefault(root['pid'], root['token'])
        owned = {pid for pid, token in self.known.items()
                 if pid in table and table[pid]['token'] == token}
        while True:
            descendants = {pid for pid, row in table.items() if row['ppid'] in owned}
            if descendants <= owned:
                break
            owned.update(descendants)
        for pid in owned:
            self.known[pid] = table[pid]['token']
            self.host_pids.add(table[pid]['host_pid'])
        # A recycled numeric PID must never become an owned process.
        for pid, row in table.items():
            if pid in self.known and row['token'] != self.known[pid]:
                self.host_pids.discard(row['host_pid'])
        return table, owned

    def stop(self):
        table, owned = self.refresh()
        groups = {table[p]['pgid'] for p in owned if table[p]['pgid'] == self.child.pid}
        for group in groups:
            try:
                os.killpg(group, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            self.child.wait(timeout=2)
        except subprocess.TimeoutExpired:
            table, owned = self.refresh()
            if any(table[p]['pgid'] == self.child.pid for p in owned):
                try:
                    os.killpg(self.child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self.child.wait(timeout=5)
        # Remove only positively identified descendants, never observed foreign CUDA PIDs.
        table, owned = self.refresh()
        for pid in owned:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def inspect_foreign(probe, assigned, owner):
    owner.refresh()
    devices = {row['uuid']: row for row in probe.snapshot()}
    assigned_ids = {row['uuid'] for row in assigned}
    if any(set(row['pids']) & owner.host_pids for key, row in devices.items() if key not in assigned_ids):
        raise ResourceViolation('Owned CUDA process used an unassigned GPU')
    foreign = {}
    for device in assigned:
        if device['uuid'] not in devices:
            raise ProbeUnavailable('Assigned GPU identity disappeared')
        others = sorted(set(devices[device['uuid']]['pids']) - owner.host_pids)
        if others:
            foreign[device['uuid']] = others
    return foreign


def inspect_rank_devices(probe, attempt, assigned, owner):
    table,owned=owner.refresh();expected={}
    for rank,device in enumerate(assigned):
        identity=Path(attempt)/'distributed'/('rank-'+str(rank))/'identity.json'
        if not identity.exists():continue
        value=json.loads(identity.read_text());pid=value['pid']
        if pid not in owned:continue
        if value.get('start_token') and table[pid]['token']!=value['start_token']:continue
        members={pid}
        while True:
            added={p for p,row in table.items() if row['ppid'] in members}
            if added<=members:break
            members.update(added)
        expected.update({table[p]['host_pid']:device['uuid'] for p in members if p in owned})
    for device in probe.snapshot():
        if any(pid in expected and expected[pid]!=device['uuid'] for pid in device['pids']):
            raise ResourceViolation('A distributed rank used a GPU assigned to another rank')


def wait_for_owned_cleanup(probe, devices, owner, attempt):
    """A dead launcher or an expired heartbeat is not proof that a GPU is free."""
    while True:
        try:
            table, owned = owner.refresh()
            members = [row for row in table.values() if row['pgid'] == owner.child.pid]
            observed = {row['uuid']: row for row in probe.snapshot()}
            cuda = any(set(row['pids']) & owner.host_pids for row in observed.values())
            if not owned and not members and not cuda:
                return
            owner.stop()
        except Exception:
            pass  # Fail closed: keep the inherited device locks while ownership is uncertain.
        atomic(attempt / 'reclaiming.json', {'time': time.time(), 'reason': 'owned_process_cleanup_unconfirmed'})
        time.sleep(.2)


def supervise(root, job_id, attempt_number, descriptor_list, *, probe=None):
    """Own a complete attempt and its GPU locks until processes are gone."""
    root = Path(root)
    job = root / 'jobs' / job_id
    attempt = job / ('attempt-' + str(attempt_number))
    spec = json.loads((job / 'request.json').read_text())
    placement = json.loads((attempt / 'placement.json').read_text())
    child = None
    owner = None
    result = dict(exit_code=None, timed_out=False, cancelled=False, contaminated=False,
                  attempt=attempt_number, devices=placement['devices'], fence=placement['fence'],
                  worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic(attempt / 'supervisor.json', process_table()[os.getpid()])
    try:
        probe = probe or NvmlProbe()
        inventory = {x['uuid']: x for x in probe.snapshot()}
        if (job / 'cancel').exists():
            result['cancelled'] = True
        elif any(x['uuid'] not in inventory or inventory[x['uuid']]['pids'] for x in placement['devices']):
            result.update(contaminated=True, reason='external_before_start')
        else:
            with Broker(root, probe=probe).connect() as db:
                row = db.execute('SELECT state,attempt,fence FROM jobs WHERE id=?', (job_id,)).fetchone()
            if not row or row['state'] != 'running' or row['attempt'] != attempt_number or row['fence'] != placement['fence']:
                raise RuntimeError('Stale launch fence rejected')
            environment = {**os.environ, **spec['environment'],
                           'CUDA_VISIBLE_DEVICES': ','.join(x['uuid'] for x in placement['devices']),
                           'AKA_ASSIGNED_GPU_UUIDS': ','.join(x['uuid'] for x in placement['devices'])}
            if spec.get('source_manifest_id'):
                support_module(root,spec['content_engine'],'aka_cache.py')
                environment.update(AKA_BUILD_CACHE_DIR=str(root/'cache-v2/builds'))
                environment['AKA_BUILD_CACHE_LIMIT']=str(spec.get('cache_options',{}).get('max_build_bytes',16*1024**3))
                environment['PYTHONPATH']=str(Path(spec['content_engine']['path']).parent)+os.pathsep+environment.get('PYTHONPATH','')
            argv=spec['argv']
            if spec.get('distributed'):
                module=distributed_module(spec,root)
                for key in module.PROTECTED:environment.pop(key,None)
                environment['PYTHONPATH']=str(Path(spec['distributed_runner']).parent)+os.pathsep+environment.get('PYTHONPATH','')
                argv=[spec.get('python',sys.executable),spec['distributed_runner'],'launch',str(attempt)]
            elif spec.get('source_manifest_id') and Path(argv[0]).name.startswith('python'):
                argv=[spec['python'],*argv[1:]]
            with (attempt / 'stdout.log').open('wb') as stdout, (attempt / 'stderr.log').open('wb') as stderr:
                child = subprocess.Popen(argv, cwd=attempt / 'workspace', env=environment,
                    stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, start_new_session=True,
                    pass_fds=tuple(descriptor_list))
                owner = OwnedProcesses(child)
                atomic(attempt / 'child.json', process_table().get(child.pid, {'pid': child.pid}))
                started = time.monotonic()
                atomic(attempt / 'started.json', {'time': time.time(), 'pid': child.pid})
                recorded_owned = None
                while True:
                    foreign = inspect_foreign(probe, placement['devices'], owner)
                    if spec.get('distributed'):inspect_rank_devices(probe,attempt,placement['devices'],owner)
                    if owner.known != recorded_owned:
                        atomic(attempt / 'owned.json', {str(pid): token for pid, token in owner.known.items()})
                        recorded_owned = dict(owner.known)
                    if foreign:
                        result.update(contaminated=True, reason='external_during_execution', foreign=foreign)
                    if (job / 'cancel').exists():
                        result['cancelled'] = True
                    if time.monotonic() - started >= spec['timeout_s']:
                        result['timed_out'] = True
                    if result['contaminated'] or result['cancelled'] or result['timed_out']:
                        owner.stop()
                        break
                    if child.poll() is not None:
                        break
                    time.sleep(spec['monitor_interval_s'])
                result['exit_code'] = child.wait()
                if spec.get('distributed') and (attempt/'distributed/group.json').exists():
                    group=json.loads((attempt/'distributed/group.json').read_text())
                    if group.get('fence')!=placement['fence']:raise ValueError('Stale distributed group receipt')
                    result['distributed']=group
                    result['retryable_group_failure']=not group.get('complete') and group.get('retryable',False)
                    if not group.get('complete') and result['exit_code']==0:result['exit_code']=126
                # Includes the boundary after completion; success cannot bypass the final check.
                foreign = inspect_foreign(probe, placement['devices'], owner)
                if foreign:
                    result.update(contaminated=True, reason='external_at_completion', foreign=foreign)
                owner.stop()
                wait_for_owned_cleanup(probe, placement['devices'], owner, attempt)
    except ResourceViolation:
        if owner is not None:
            owner.stop()
            wait_for_owned_cleanup(probe, placement['devices'], owner, attempt)
        result.update(exit_code=(child.returncode or 126) if child is not None else 126,
                      reason='resource_violation', valid_measurement=False)
    except Exception as error:
        if owner is not None:
            owner.stop()
            wait_for_owned_cleanup(probe, placement['devices'], owner, attempt)
        result.update(contaminated=True, reason='monitor_or_supervisor_unavailable', error=type(error).__name__)
    finally:
        for name in ['stdout.log', 'stderr.log']:
            (attempt / name).touch(exist_ok=True)
        if result['contaminated']:
            result['valid_measurement'] = False
        else:
            result['valid_measurement'] = result['exit_code'] == 0 and not result['cancelled'] and not result['timed_out']
        atomic(attempt / 'attempt-result.json', result)
        for descriptor in descriptor_list:
            try:
                os.close(descriptor)
            except OSError:
                pass
    return result


class Broker:
    def __init__(self, root, *, probe=None, launcher=None, device_lock_root=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.probe = probe
        self.launcher = launcher or self.launch
        self.device_lock_root = Path(device_lock_root or '/tmp/aka-remote-gpu-device-locks-v1')
        self.device_lock_root.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS jobs(
              id TEXT PRIMARY KEY,fingerprint TEXT NOT NULL,task TEXT NOT NULL,
              state TEXT NOT NULL,created REAL NOT NULL,attempt INTEGER NOT NULL DEFAULT 0,
              fence INTEGER NOT NULL DEFAULT 0,served REAL NOT NULL DEFAULT 0,
              spec TEXT NOT NULL,placement TEXT,result TEXT);
            CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY,last_served REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS counters(id INTEGER PRIMARY KEY,value INTEGER NOT NULL);
            INSERT OR IGNORE INTO counters VALUES(1,0);
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            ''')
            db.execute('INSERT OR IGNORE INTO metadata VALUES(?,?)', ('epoch', uuid.uuid4().hex))
            columns={x[1] for x in db.execute('PRAGMA table_info(jobs)')}
            for name in ['contamination_retries','distributed_retries']:
                if name not in columns:db.execute('ALTER TABLE jobs ADD COLUMN '+name+' INTEGER NOT NULL DEFAULT 0')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.root / 'queue.sqlite3', timeout=10)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA journal_mode=WAL')
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def inventory(self):
        if self.probe is None:
            self.probe = NvmlProbe()
        return self.probe.snapshot()

    def cached(self, sha):
        if not re.fullmatch('[a-f0-9]{64}', sha):
            raise ValueError('Invalid source hash')
        path = self.root / 'sources' / (sha + '.zip')
        return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == sha

    def cache_source(self, spec):
        sha = spec['archive_sha256']
        if self.cached(sha):
            return
        source = Path(spec['source_root'])
        upload = json.loads((source / 'spec.json').read_text())
        if upload['archive_sha256'] != sha or not 0 <= upload['parts'] <= 10000:
            raise ValueError('Source request mismatch')
        raw = b''.join((source / ('part-' + str(i))).read_bytes() for i in range(upload['parts']))
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError('Source archive checksum mismatch')
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            if sum(x.file_size for x in archive.infolist()) > MAX_SOURCE:
                raise ValueError('Expanded source archive exceeds limit')
            for info in archive.infolist():
                inside(source / 'workspace', info.filename)
        path = self.root / 'sources' / (sha + '.zip')
        path.parent.mkdir(exist_ok=True)
        temporary = path.with_name(path.name + '.' + uuid.uuid4().hex)
        temporary.write_bytes(raw)
        temporary.replace(path)

    def enqueue(self, spec):
        if not re.fullmatch('remote-[a-f0-9]{32}', spec.get('job_id', '')):
            raise ValueError('Invalid logical execution ID')
        if not isinstance(spec.get('task'), str) or not spec['task']:
            raise ValueError('Missing task identity')
        if not spec.get('argv') or not all(isinstance(v, str) for v in spec['argv']):
            raise ValueError('Invalid command')
        if any(k in spec['environment'] for k in ['CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES', 'AKA_ASSIGNED_GPU_UUIDS']):
            raise ValueError('Device selection is owned by the scheduler')
        if spec.get('distributed'):
            module=distributed_module(spec,self.root)
            if module.validate_options(spec['distributed'],spec['gpu_count'])!=spec['distributed']:
                raise ValueError('Distributed options must be normalized by the host')
            if set(spec['environment'])&module.PROTECTED or any(k.startswith('AKA_DISTRIBUTED_') for k in spec['environment']):
                raise ValueError('Rank identity cannot be overridden')
        for name in spec['outputs']:
            inside(self.root / 'unused-workspace', name)
        fingerprint = digest(spec)
        with self.connect() as db:
            old = db.execute('SELECT fingerprint FROM jobs WHERE id=?', (spec['job_id'],)).fetchone()
        if old:
            if old['fingerprint'] != fingerprint:
                raise ValueError('Logical execution already has different frozen inputs')
            return self.status(spec['job_id'])
        if spec.get('source_manifest_id'):
            module=support_module(self.root,spec['content_engine'],'aka_cache.py')
            if module.validate_manifest(spec['source_manifest'])!=spec['source_manifest_id']:raise ValueError('Source manifest identity mismatch')
            module.Store(self.root/'cache-v2').commit(spec['source_manifest'])
        else:self.cache_source(spec)
        directory = self.root / 'jobs' / spec['job_id']
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / 'request.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            existing = directory / 'request.json'
            if existing.exists() and json.loads(existing.read_text()) != spec:
                raise ValueError('Logical execution input journal changed')
            atomic(existing, spec)
            with self.connect() as db:
                db.execute('BEGIN IMMEDIATE')
                row = db.execute('SELECT fingerprint FROM jobs WHERE id=?', (spec['job_id'],)).fetchone()
                if row and row['fingerprint'] != fingerprint:
                    raise ValueError('Concurrent submission has different frozen inputs')
                db.execute('INSERT OR IGNORE INTO tasks VALUES(?,0)', (spec['task'],))
                db.execute("INSERT OR IGNORE INTO jobs(id,fingerprint,task,state,created,spec) VALUES(?,?,?,'queued',?,?)",
                           (spec['job_id'], fingerprint, spec['task'], time.time(), json.dumps(spec)))
        return self.status(spec['job_id'])

    def status(self, job_id):
        with self.connect() as db:
            row = db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
        if not row:
            return {'state': 'absent'}
        value = {k: row[k] for k in ['id', 'state', 'created', 'attempt', 'fence','contamination_retries','distributed_retries']}
        for name in ['placement', 'result']:
            value[name] = json.loads(row[name]) if row[name] else None
        value['attempt_root'] = str(self.root / 'jobs' / job_id / ('attempt-' + str(row['attempt']))) if row['attempt'] else None
        value['waiting_resource'] = row['state'] == 'queued'
        if value['attempt_root']:
            progress=Path(value['attempt_root'])/'distributed/progress.json'
            if progress.exists():value['distributed_progress']=json.loads(progress.read_text())
        return value

    def cancel(self, job_id, *, queued_only=False):
        directory = self.root / 'jobs' / job_id
        if not directory.is_dir():
            return {'state': 'absent'}
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT state FROM jobs WHERE id=?', (job_id,)).fetchone()
            if not (queued_only and row and row['state'] != 'queued'):
                (directory / 'cancel').touch()
                db.execute("UPDATE jobs SET state='cancelled',result=? WHERE id=? AND state='queued'",
                           (json.dumps({'cancelled': True, 'exit_code': None, 'timed_out': False}), job_id))
        return self.status(job_id)

    def acquire(self, devices):
        descriptors = []
        try:
            for device in sorted(devices, key=lambda x: x['uuid']):
                key = hashlib.sha256(device['uuid'].encode()).hexdigest()
                descriptor = os.open(self.device_lock_root / (key + '.lock'), os.O_CREAT | os.O_RDWR, 0o600)
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    os.close(descriptor)
                    raise
                descriptors.append(descriptor)
            return descriptors
        except BlockingIOError:
            for descriptor in descriptors:
                os.close(descriptor)
            return None

    def choose_and_acquire(self, candidates, count, topology='any'):
        if topology!='any':
            for devices in itertools.islice(itertools.combinations(candidates,count),4096):
                if self.probe.compatible(devices,topology):
                    descriptors=self.acquire(devices)
                    if descriptors is not None:return list(devices),descriptors
            return None,None
        selected, descriptors = [], []
        for device in candidates:
            acquired = self.acquire([device])
            if acquired is not None:
                selected.append(device);descriptors.extend(acquired)
                if len(selected) == count:
                    return selected, descriptors
        for descriptor in descriptors:os.close(descriptor)
        return None, None

    def materialize(self, spec, attempt):
        workspace = attempt / 'workspace'
        if spec.get('source_manifest_id'):
            module=support_module(self.root,spec['content_engine'],'aka_cache.py')
            if spec.get('distributed'):workspace.mkdir(parents=True,exist_ok=True)
            else:module.Store(self.root/'cache-v2').materialize(spec['source_manifest_id'],workspace)
        else:
            workspace.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(self.root / 'sources' / (spec['archive_sha256'] + '.zip')) as archive:
                for info in archive.infolist():inside(workspace, info.filename)
                archive.extractall(workspace)
        # Preserve the original stat/read protocol for resumable large artifacts.
        if spec.get('source_manifest_id'):
            record=spec['artifact_runner'];path=Path(record['path'])
            if path.resolve()!=(self.root/('support-'+record['sha256'])/'runner.py').resolve() or hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
                raise ValueError('Artifact runner checksum mismatch')
            shutil.copyfile(path,attempt/'runner.py')
        else:shutil.copyfile(Path(spec['source_root']) / 'runner.py', attempt / 'runner.py')

    def launch(self, row, spec, devices, descriptors, fence):
        number = row['attempt'] + 1
        directory = self.root / 'jobs' / row['id'] / ('attempt-' + str(number))
        self.materialize(spec, directory)
        atomic(directory / 'placement.json', {'devices': devices, 'fence': fence})
        if spec.get('distributed'):
            distributed_module(spec,self.root)
            group=directory/'distributed';group.mkdir(parents=True,exist_ok=True)
            for rank in range(spec['gpu_count']):
                target=group/('rank-'+str(rank))/'workspace'
                if spec.get('source_manifest_id'):
                    module=support_module(self.root,spec['content_engine'],'aka_cache.py')
                    module.Store(self.root/'cache-v2').materialize(spec['source_manifest_id'],target)
                else:shutil.copytree(directory/'workspace',target)
            atomic(group/'config.json',{**spec['distributed'],'directory':str(group),
                'nonce':uuid.uuid4().hex,'fence':fence,'devices':devices,'argv':spec['argv']})
        with (directory / 'supervisor.log').open('ab') as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(self.root),
                                        'supervise', row['id'], str(number), ','.join(map(str, descriptors))],
                stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True,
                pass_fds=tuple(descriptors))
        return process.pid

    def tick(self):
        inventory = self.inventory()  # No fallback from an unreadable process inventory to 'idle'.
        atomic(self.root / 'inventory.json', {'time': time.time(), 'devices': inventory})
        with self.connect() as db:
            running = [dict(x) for x in db.execute("SELECT * FROM jobs WHERE state='running'")]
        for row in running:
            directory = self.root / 'jobs' / row['id'] / ('attempt-' + str(row['attempt']))
            result_file = directory / 'attempt-result.json'
            if not result_file.exists():
                supervisor = directory / 'supervisor.json'
                if supervisor.exists() and live(json.loads(supervisor.read_text())):
                    continue
                if time.time() - row['served'] < 3:
                    continue
                child = directory / 'child.json'
                if child.exists():
                    handle = json.loads(child.read_text())
                    table = process_table()
                    if live(handle, table) and table[handle['pid']]['pgid'] == handle['pid']:
                        # Only an authenticated owned child whose supervisor died.
                        reaper = directory / 'reaper.json'
                        if not reaper.exists():atomic(reaper, {'started': time.time()})
                        overdue = time.time() - json.loads(reaper.read_text())['started'] > 2
                        try:os.killpg(handle['pid'], signal.SIGKILL if overdue else signal.SIGTERM)
                        except ProcessLookupError:pass
                        continue
                    if any(p['pgid'] == handle['pid'] for p in table.values()):
                        evidence = directory / 'owned.json'
                        known = json.loads(evidence.read_text()) if evidence.exists() else {}
                        for pid, token in known.items():
                            record = table.get(int(pid))
                            if record and record['token'] == token and record['pgid'] == handle['pid']:
                                try:os.kill(int(pid), signal.SIGKILL)
                                except ProcessLookupError:pass
                        continue
                placement = json.loads(row['placement'])
                if any(d['pids'] for d in inventory if d['uuid'] in {v['uuid'] for v in placement['devices']}):
                    continue
                descriptors = self.acquire(placement['devices'])
                if descriptors is None:
                    continue
                for descriptor in descriptors:os.close(descriptor)
                directory.mkdir(parents=True, exist_ok=True)
                atomic(result_file, {'exit_code': None, 'cancelled': False, 'timed_out': False,
                                    'contaminated': True, 'valid_measurement': False,
                                    'reason': 'verified_supervisor_loss', 'fence': row['fence'],
                                    'attempt': row['attempt']})
            result = json.loads(result_file.read_text())
            placement = json.loads(row['placement'])
            if result.get('fence') != row['fence']:
                continue
            # Result publication precedes supervisor exit. Do not dispatch until
            # it released all inherited locks, even for a completed command.
            descriptors = self.acquire(placement['devices'])
            if descriptors is None:
                continue
            for descriptor in descriptors:
                os.close(descriptor)
            spec = json.loads(row['spec'])
            if result.get('cancelled'):
                state = 'cancelled'
            elif result.get('contaminated'):
                state = ('queued' if spec['replay_safe'] and row['contamination_retries'] < spec['max_contamination_retries']
                         else 'blocked')
            elif result.get('retryable_group_failure'):
                state=('queued' if spec['replay_safe'] and row['distributed_retries'] < spec['distributed']['max_retries']
                       else 'blocked')
            else:
                state = 'completed' if result['exit_code'] == 0 and not result['timed_out'] else 'failed'
                if state == 'completed':
                    roots=([directory/'distributed'/('rank-'+str(rank))/'workspace' for rank in range(spec['gpu_count'])]
                           if spec.get('distributed') else [directory/'workspace'])
                    missing = [str(root/name) for root in roots for name in spec['outputs'] if not inside(root, name).is_file()]
                    if missing:
                        state = 'failed'
                        result['missing_outputs'] = missing
                        result['valid_measurement'] = False
                if state != 'completed':
                    result['valid_measurement'] = False
            with self.connect() as db:
                db.execute('UPDATE jobs SET state=?,result=?,contamination_retries=contamination_retries+?,distributed_retries=distributed_retries+? WHERE id=? AND fence=?',
                           (state,json.dumps(result),int(bool(result.get('contaminated'))),
                            int(bool(result.get('retryable_group_failure'))),row['id'],row['fence']))
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            rows = [dict(x) for x in db.execute("SELECT j.* FROM jobs j JOIN tasks t ON t.id=j.task WHERE j.state='queued' "
                                               'ORDER BY t.last_served,j.created')]
            now=time.time()
            aged=[];ordinary=[]
            for item in rows:
                request=json.loads(item['spec'])
                (aged if request['gpu_count']>1 and now-item['created']>=request.get('reservation_after_s',30)
                 else ordinary).append(item)
            rows=sorted(aged,key=lambda x:x['created'])+ordinary
            running_tasks = {x[0] for x in db.execute("SELECT task FROM jobs WHERE state='running'")}
            reserved = {d['uuid'] for x in db.execute("SELECT placement FROM jobs WHERE state='running'")
                        for d in json.loads(x[0])['devices']}
            for row in rows:
                if row['task'] in running_tasks:
                    continue
                spec = json.loads(row['spec'])
                allowed = [d for d in inventory if d['index'] in spec['gpu_ids']]
                idle = [d for d in allowed if not d['pids'] and d['uuid'] not in reserved]
                count = spec['gpu_count']
                if count>len(allowed):
                    db.execute("UPDATE jobs SET state='blocked',result=? WHERE id=?",
                               (json.dumps({'reason':'insufficient_pod_gpu_capacity','valid_measurement':False,'exit_code':None}),row['id']))
                    continue
                if len(idle) < count:
                    # Drain internal single-GPU jobs for an aged gang request,
                    # but do not reserve idle devices behind foreign processes.
                    if count > 1 and time.time()-row['created'] > spec.get('reservation_after_s',30):
                        # Reserve only this request's allowed group; independent
                        # device groups may still backfill without starving it.
                        reserved.update(d['uuid'] for d in allowed)
                    continue
                devices, descriptors = self.choose_and_acquire(idle, count,spec.get('topology','any'))
                if descriptors is None:
                    continue
                fresh = {d['uuid']: d for d in self.inventory()}
                if any(d['uuid'] not in fresh or fresh[d['uuid']]['pids'] for d in devices):
                    for descriptor in descriptors:
                        os.close(descriptor)
                    continue
                db.execute('UPDATE counters SET value=value+1 WHERE id=1')
                counter = db.execute('SELECT value FROM counters WHERE id=1').fetchone()[0]
                epoch = db.execute("SELECT value FROM metadata WHERE key='epoch'").fetchone()[0]
                fence = epoch + ':' + str(counter)
                placement = {'devices': devices, 'fence': fence}
                db.execute("UPDATE jobs SET state='running',attempt=attempt+1,fence=?,served=?,placement=?,result=NULL WHERE id=?",
                           (fence, time.time(), json.dumps(placement), row['id']))
                db.execute('UPDATE tasks SET last_served=? WHERE id=?', (time.time(), row['task']))
                # Commit intent before launching. A late supervisor verifies the
                # fence, and the GPU fd is held continuously through the handoff.
                db.commit()
                try:
                    self.launcher(row, spec, devices, descriptors, fence)
                except Exception:
                    # No blind retry after an ambiguous process launch.
                    atomic(self.root / 'jobs' / row['id'] / 'launch-unknown.json', {'fence': fence, 'time': time.time()})
                finally:
                    for descriptor in descriptors:
                        os.close(descriptor)
                reserved.update(d['uuid'] for d in devices)
                running_tasks.add(row['task'])
                db.execute('BEGIN IMMEDIATE')

    def collect(self, job_id, known_artifacts=()):
        if len(known_artifacts)>512 or any(not re.fullmatch('[a-f0-9]{64}',v) for v in known_artifacts):raise ValueError('Invalid known artifact list')
        status = self.status(job_id)
        if status['state'] not in TERMINAL:
            raise ValueError('Execution has no terminal receipt')
        spec = json.loads((self.root / 'jobs' / job_id / 'request.json').read_text())
        files = {}
        if status['attempt_root']:
            attempt = Path(status['attempt_root'])
            names=['stdout.log','stderr.log',*['workspace/'+x for x in spec['outputs']]]
            if spec.get('distributed'):
                names=['stdout.log','stderr.log','distributed/group.json','distributed/progress.json','distributed/launcher.log']
                for rank in range(spec['gpu_count']):
                    prefix='distributed/rank-'+str(rank)+'/'
                    names.extend(prefix+x for x in ['stdout.log','stderr.log','identity.json','state.json','result.json','exit.json'])
                    names.extend(prefix+'workspace/'+x for x in spec['outputs'])
            for name in names:
                path = inside(attempt, name)
                if path.is_file():
                    data = path.read_bytes()
                    item = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                    if len(data) <= 400000 and item['sha256'] not in known_artifacts:
                        item['base64'] = base64.b64encode(data).decode()
                    files[name] = item
        answer={'status':status,'files':files}
        if spec.get('source_manifest_id') and status['attempt_root']:
            token=digest({'job':job_id,'attempt':status['attempt'],'files':{name:{k:v for k,v in info.items() if k!='base64'} for name,info in files.items()}})
            atomic(self.root/'jobs'/job_id/'collection.json',{'token':token,'attempt':status['attempt']})
            answer['collection_token']=token
        return answer

    def acknowledge(self,entry):
        """Prune only unchanged input copies after verified parent collection."""
        job_id=entry.get('job_id','')
        if not re.fullmatch('remote-[a-f0-9]{32}',job_id):raise ValueError('Invalid collection acknowledgment')
        job=self.root/'jobs'/job_id;proof=job/'collection.json'
        if not proof.exists() or json.loads(proof.read_text())!={'token':entry.get('token'),'attempt':entry.get('attempt')}:raise ValueError('Collection acknowledgment mismatch')
        status=self.status(job_id)
        if status['state'] not in TERMINAL or status['attempt']!=entry['attempt']:raise ValueError('Cannot prune an active attempt')
        done=job/'inputs-pruned.json'
        if done.exists():return json.loads(done.read_text())
        spec=json.loads((job/'request.json').read_text())
        if not spec.get('source_manifest_id'):raise ValueError('Only content snapshots support pruning')
        attempt=Path(status['attempt_root']);roots=[attempt/'workspace']
        if spec.get('distributed'):roots.extend(attempt/'distributed'/('rank-'+str(i))/'workspace' for i in range(spec['gpu_count']))
        removed=0;freed=0
        for root in roots:
            for name,info in spec['source_manifest']['files'].items():
                if name in spec['outputs']:continue
                path=inside(root,name)
                if path.is_file() and not path.is_symlink() and path.stat().st_size==info['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==info['sha256']:
                    freed+=path.stat().st_size;path.unlink();removed+=1
        result={'job_id':job_id,'removed_input_copies':removed,'freed_bytes':freed,'time':time.time()}
        atomic(done,result);return result


def serve(root):
    root = Path(root)
    with (root / 'server.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        broker = Broker(root)
        atomic(root / 'server.json', {**process_table()[os.getpid()], 'agent': str(Path(__file__).resolve())})
        while True:
            try:
                broker.tick()
                atomic(root / 'heartbeat.json', {'time': time.time(), 'pid': os.getpid(), 'ready': True})
            except Exception as error:
                atomic(root / 'heartbeat.json', {'time': time.time(), 'pid': os.getpid(),
                                                'ready': False, 'error': type(error).__name__})
            time.sleep(.1)


def rpc(root, operation, value):
    ensure_server(root)
    broker = Broker(root)
    acknowledged=[];rejected=[]
    for entry in value.get('collection_acks',[])[:1]:
        try:broker.acknowledge(entry);acknowledged.append(entry['job_id'])
        except (OSError,ValueError):rejected.append(entry.get('job_id',''))
    if operation.startswith('content_'):
        module=support_module(root,value['engine'],'aka_cache.py');store=module.Store(Path(root)/'cache-v2')
        def admit(spec):
            if spec['content_engine']!=value['engine']:raise ValueError('Content engine identity mismatch')
            # Recovering an existing ID is a read/validation operation. It must
            # not wait for, or replace, the coordinator running that attempt.
            if broker.status(spec['job_id'])['state']!='absent':return broker.enqueue(spec)
            ensure_server(root,require_version=True)
            status=broker.enqueue(spec)
            (store.root/'staging'/(spec['source_manifest_id']+'.json')).unlink(missing_ok=True)
            with broker.connect() as db:
                keep=[json.loads(row[0]).get('source_manifest_id') for row in db.execute("SELECT spec FROM jobs WHERE state IN ('queued','running')")]
            options=spec.get('cache_options',{})
            store.gc(options.get('max_source_bytes',16*1024**3),[mid for mid in keep if mid],options.get('ttl_s',86400))
            return status
        if operation=='content_prepare':
            spec=module.unwire(value['spec']) if 'spec' in value else None
            manifest=spec['source_manifest'] if spec else module.unwire(value['manifest'])
            result=store.prepare(manifest,value.get('known_packs',[]))
            imported=store.seed_archives(Path(root)/'sources',result['missing']) if result['missing'] else 0
            if imported:result=store.prepare(manifest,value.get('known_packs',[]))
            result['imported_legacy_blobs']=imported
            if spec and not result['missing']:
                result['status']=admit(spec);result['enqueued']=True
        elif operation=='content_put':result=store.put_pack(value['sha256'],value['index'],value['count'],value['bytes'],base64.b64decode(value['data'],validate=True))
        elif operation=='content_commit':result={'manifest_id':store.commit(module.unwire(value['manifest']))}
        elif operation=='content_enqueue':
            spec=module.unwire(value['spec'])
            result=admit(spec)
        elif operation=='content_gc':
            with broker.connect() as db:
                keep=[json.loads(row[0]).get('source_manifest_id') for row in db.execute("SELECT spec FROM jobs WHERE state IN ('queued','running')")]
            result=store.gc(value['max_bytes'],[mid for mid in keep if mid],value.get('ttl_s',86400))
        else:raise ValueError('Unknown content operation')
    elif operation == 'cached':
        result = {'cached': broker.cached(value['sha256'])}
    elif operation == 'enqueue':
        ensure_server(root,require_version=True)
        result = broker.enqueue(value)
    elif operation == 'enqueue_many':
        ensure_server(root,require_version=True)
        if not isinstance(value.get('jobs'), list) or len(value['jobs']) > 64:
            raise ValueError('Invalid bounded submission batch')
        result = {'jobs': [broker.enqueue(spec) for spec in value['jobs']]}
    elif operation == 'cancel':
        result = broker.cancel(value['job_id'], queued_only=value.get('queued_only', False))
    elif operation == 'status':
        result = broker.status(value['job_id'])
    elif operation == 'collect':
        result = broker.collect(value['job_id'],value.get('known_artifacts',()))
    elif operation == 'collect_many':
        if not isinstance(value.get('job_ids'), list) or len(value['job_ids']) > 64:
            raise ValueError('Invalid bounded collection batch')
        result = {'jobs': {key: broker.collect(key) for key in value['job_ids']}}
    elif operation == 'status_many':
        with broker.connect() as db:
            ids = [x[0] for x in db.execute('SELECT id FROM jobs ORDER BY created DESC LIMIT 1000')]
        statuses={key:broker.status(key) for key in ids}
        for status in statuses.values():
            # Completed rank reports belong to collect(), not every poll.
            status.pop('result',None)
        result = {'jobs': statuses,
                  'inventory': json.loads((Path(root) / 'inventory.json').read_text()) if (Path(root) / 'inventory.json').exists() else None,
                  'heartbeat': json.loads((Path(root) / 'heartbeat.json').read_text()) if (Path(root) / 'heartbeat.json').exists() else None}
    else:
        raise ValueError('Unknown pool RPC')
    if acknowledged:result['acknowledged_jobs']=acknowledged
    if rejected:result['rejected_collection_acks']=rejected
    ensure_server(root)
    return result


def ensure_server(root,require_version=False):
    root = Path(root)
    marker = root / 'server.json'
    if marker.exists() and live(json.loads(marker.read_text())):
        server=json.loads(marker.read_text())
        if not require_version or Path(server.get('agent','')).resolve()==Path(__file__).resolve():return
        with Broker(root).connect() as db:
            if db.execute("SELECT count(*) FROM jobs WHERE state IN ('queued','running')").fetchone()[0]:raise ValueError('Pool version upgrade waits for active jobs')
        os.kill(server['pid'],signal.SIGTERM)
        for _ in range(200):
            if not live(server):break
            time.sleep(.01)
    with (root / 'server.log').open('ab') as log:
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(root), 'serve'],
                         stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)


if __name__ == '__main__':
    root, operation = Path(sys.argv[1]), sys.argv[2]
    root.mkdir(parents=True, exist_ok=True)
    if operation == 'serve':
        serve(root)
    elif operation == 'supervise':
        supervise(root, sys.argv[3], int(sys.argv[4]), [int(x) for x in sys.argv[5].split(',')])
    else:
        try:
            result = rpc(root, operation, json.loads(sys.argv[3]))
        except Exception as error:
            result = {'error': type(error).__name__ + ': ' + str(error)}
        print('AKA_REMOTE=' + json.dumps(result), flush=True)
