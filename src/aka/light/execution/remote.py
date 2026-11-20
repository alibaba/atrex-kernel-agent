"""Task Executor for an existing Remote Pod, through a host-selected transport."""
from __future__ import annotations
import base64
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import time
import zipfile
import zlib

from aka.light.execution.contract import BackendCapabilities, ExecutionUnavailable, JobResult, ResourceIdentity
from aka.light.tasks.content import atomic_json, relative, tree
from .remote_resilience import Policy, Failure, call, decode_runner, event
from . import remote_transfer

CHUNK = 90000  # Base64 stays below Linux's 128 KiB single-argument limit.
LEGACY_CHUNK = 36000
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024

# Opt-in transfer optimization for large profiler artifacts. The original job,
# output bytes and runner protocol stay unchanged, including recovery by ID.
COMPRESS_FILE = '''import hashlib,json,pathlib,sys,zlib
r=pathlib.Path(sys.argv[1]).resolve();p=(r/sys.argv[2]).resolve()
if not p.is_relative_to(r):raise ValueError('Artifact path escapes job directory')
b=p.read_bytes()
if len(b)>67108864:raise ValueError('Artifact exceeds transfer limit')
h=hashlib.sha256(b).hexdigest();d=r/'.transfer-cache';d.mkdir(exist_ok=True)
c=zlib.compress(b,6);f=d/(h+'.zlib');t=f.with_suffix('.tmp');t.write_bytes(c);t.replace(f)
print('AKA_REMOTE='+json.dumps({'name':str(f.relative_to(r)),'bytes':len(c),'raw_bytes':len(b),
 'sha256':h,'encoded_sha256':hashlib.sha256(c).hexdigest()}))'''

FILE_INFO = '''import hashlib,json,pathlib,sys
r=pathlib.Path(sys.argv[1]).resolve();p=(r/sys.argv[2]).resolve()
if not p.is_relative_to(r):raise ValueError('Artifact path escapes job directory')
if not p.is_file():v={'exists':False}
else:
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1048576):h.update(b)
 v={'exists':True,'bytes':p.stat().st_size,'sha256':h.hexdigest()}
print('AKA_REMOTE='+json.dumps(v))'''

SAFE_START = '''import fcntl,importlib.util,json,pathlib,sys
r=pathlib.Path(sys.argv[1])
try:
 with (r/'.aka-start.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  s=importlib.util.spec_from_file_location('aka_owned_runner',r/'runner.py')
  m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
  v=m.dispatch(r,'start',[])
except Exception as e:v={'error':type(e).__name__+': '+str(e)}
print('AKA_REMOTE='+json.dumps(v))
'''


def validate_config(config):
    allowed = {'pod', 'container', 'context', 'namespace', 'uid', 'remote_root', 'python',
               'environment', 'poll_interval_s', 'resilience', 'scheduling', 'distributed', 'cache'}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError('Invalid remote execution configuration')
    if not isinstance(config.get('pod'), str) or not config['pod']:
        raise ValueError('remote execution requires config.pod')
    for name in allowed - {'environment', 'poll_interval_s', 'resilience', 'scheduling', 'distributed', 'cache'}:
        if name in config and (not isinstance(config[name], str) or not config[name]):
            raise ValueError(f'remote {name} must be nonempty text')
    remote = PurePosixPath(config.get('remote_root', '/tmp/aka-executions'))
    if not remote.is_absolute() or '..' in remote.parts:
        raise ValueError('remote remote_root must be an absolute path')
    env = config.get('environment', {})
    if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
        raise ValueError('remote environment must contain string values')
    interval = config.get('poll_interval_s', 5)
    if type(interval) not in (int, float) or not 0 < interval <= 60:
        raise ValueError('remote poll_interval_s must be in (0, 60]')
    policy = Policy.parse(config.get('resilience', {}))
    if 'cache' in config:
        from .remote_pool import validate_cache
        validate_cache(config['cache'])
        if 'scheduling' not in config:raise ValueError('Content cache requires pooled execution')
    if 'scheduling' in config:
        from .remote_pool import validate
        validate(config['scheduling'])
        if not config.get('uid'):
            raise ValueError('Pooled execution requires a pinned Pod UID')
    if 'distributed' in config:
        from .remote_distributed_worker import validate_options, PROTECTED
        if 'scheduling' not in config:
            raise ValueError('Distributed execution requires pooled GPU admission')
        validate_options(config['distributed'],config['scheduling'].get('gpu_count',1))
        if set(env)&PROTECTED or any(x.startswith('AKA_DISTRIBUTED_') for x in env):
            raise ValueError('Distributed rank identity is owned by the launcher')
    if policy.replay_on_restart and not config.get('uid'):
        raise ValueError('Remote restart replay requires a pinned Pod UID and caller replay-safety declaration')


class Executor:
    capabilities = BackendCapabilities(frozenset({'run'}), True)

    def __init__(self, *, target, directory, execution_id, previous_job, on_job, control):
        self.config = target.get('config', {})
        self.execution_id = execution_id
        validate_config(self.config)
        self.policy = Policy.parse(self.config.get('resilience', {}))
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.on_job, self.control = on_job, control
        routing = {k: self.config.get(k, '') for k in ('pod', 'container', 'context', 'namespace', 'uid', 'remote_root')}
        self.job = 'remote-' + hashlib.sha256(json.dumps([routing, execution_id], sort_keys=True).encode()).hexdigest()[:32]
        if previous_job and previous_job != self.job:
            raise ValueError('REMOTE recovery target differs from the saved execution')
        self.previous_job = previous_job
        self.logical_job = self.job
        self.attempt = 0
        self.fingerprint = None
        self.declared_input_files = None
        self.chunk_bytes, self.legacy_spec = CHUNK, False
        self.root = str(PurePosixPath(self.config.get('remote_root', '/tmp/aka-executions'))/self.job)
        self.resource_identity = ResourceIdentity('remote', self.job)

    def _transport(self, operation):
        # A host-selected executable owns access credentials and honors the pinned UID.
        command = [os.environ.get("AKA_REMOTE_TRANSPORT", "aka-kubectl-transport"),
                   operation, "--pod", self.config["pod"]]
        for name in ("context", "namespace", "container", "uid"):
            if name in self.config:
                command.extend(["--" + name, self.config[name]])
        return command

    def _exec(self, argv, *, retry_safe=False, operation='raw'):
        if 'scheduling' in self.config:
            from .remote_pool import transport_gate, in_transport_gate
            if not in_transport_gate(self.config):
                with transport_gate(self.config):
                    return self._exec(argv, retry_safe=retry_safe, operation=operation)
        config = self.config
        command = self._transport("exec")
        command.extend(['--', *argv])
        return call(command, config=config, directory=self.directory, policy=self.policy,
                    operation=operation, parser=decode_runner, retry_safe=retry_safe)

    def _rpc(self, operation, *args):
        if operation == 'start':
            # Serialize archive extraction as well as the runner's durable
            # launch claim. An ambiguous response must not allow a second
            # start RPC to rewrite source files while the first starts a build.
            # The wrapper preserves compatibility with existing runner bytes.
            return self._exec([self.config.get('python','python3'),'-c',SAFE_START,self.root],
                              retry_safe=True,operation='start')
        return self._exec([self.config.get('python', 'python3'), self.root+'/runner.py', self.root, operation, *map(str, args)],
                          retry_safe=operation in {'part', 'status', 'start', 'cancel', 'stat', 'read'},
                          operation=operation)

    def _pod_state(self):
        """Read only restart identity; never persist the Pod spec or credentials."""
        if 'scheduling' in self.config:
            from .remote_pool import transport_gate, in_transport_gate
            if not in_transport_gate(self.config):
                with transport_gate(self.config):
                    return self._pod_state()
        if not self.config.get('uid'):
            return {}
        argv = self._transport("identity")
        try:
            response = call(argv, config=self.config, directory=self.directory, policy=self.policy,
                            operation='pod_identity', parser=json.loads, retry_safe=True)
            if not isinstance(response, dict):raise ValueError('Invalid Pod response')
            pod = response
            if not isinstance(pod, dict):raise ValueError('Invalid Pod identity')
            uid = pod['metadata']['uid']
            if uid != self.config['uid']:
                raise ValueError('Pod UID differs from the pinned target')
            statuses = pod['status']['containerStatuses']
            name = self.config.get('container') or (statuses[0]['name'] if len(statuses) == 1 else None)
            container = next(item for item in pod['status']['containerStatuses'] if item['name'] == name)
            started = container['state']['running']['startedAt']
            return dict(pod_uid=uid, container=name, container_id=container.get('containerID', ''),
                        image_id=container.get('imageID', ''),
                        started_at=started, restart_count=container.get('restartCount', 0))
        except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, TypeError, StopIteration):
            raise ExecutionUnavailable('Remote Pod/container restart identity could not be verified') from None

    def _job_metadata(self):
        metadata = dict(job_id=self.job, pod=self.config['pod'], remote_directory=self.root,
                        prepared_at=time.time())
        try:
            state = self._pod_state()
        except ExecutionUnavailable:
            state = {}  # An optional identity read must not replay an admitted command.
        if state:metadata['container_identity'] = state
        return metadata

    def _confirmed_restart_loss(self):
        """A terminal failure is safe only with a pinned UID, later restart and absent job.

        No command is relaunched here. Missing runners without restart evidence,
        authentication failures and other ambiguous outcomes remain recoverable.
        Legacy host markers retain their original mtime as the preparation bound.
        """
        marker = self.directory/'remote-job.json'
        if not self.config.get('uid') or not marker.is_file():return None
        try:
            metadata = json.loads(marker.read_text())
            if (metadata.get('job_id'), metadata.get('pod'), metadata.get('remote_directory')) != (
                    self.job, self.config['pod'], self.root):return None
            prepared = metadata.get('prepared_at', marker.stat().st_mtime)
            current = self._pod_state()
            if current.get('pod_uid') != self.config['uid']:return None
            restarted = datetime.fromisoformat(current['started_at'].replace('Z', '+00:00')).timestamp()
            if not isinstance(prepared, (int, float)) or restarted <= prepared + 5:return None
            previous = metadata.get('container_identity', {})
            if previous.get('pod_uid') and previous['pod_uid'] != current['pod_uid']:return None
            if previous.get('container_id') and previous['container_id'] == current.get('container_id'):return None
            code = "import json,pathlib,sys; print('AKA_REMOTE='+json.dumps({'directory_exists':pathlib.Path(sys.argv[1]).exists()}))"
            probe = self._exec([self.config.get('python', 'python3'), '-c', code, self.root],
                               retry_safe=True, operation='restart_directory_probe')
            if probe.get('directory_exists') is not False:return None
            evidence = dict(job_id=self.job, remote_directory=self.root, prepared_at=prepared,
                            observed_at=time.time(), previous_container=previous,
                            current_container=current, directory_exists=False)
            atomic_json(self.directory/'remote-restart-loss.json', evidence)
            return self._result('failed', dict(exit_code=None, timed_out=False, cancelled=False,
                execution_lost=True, failure_kind='container_restart', stdout='',
                error={'code':'infra/container_restart','message':'Verified container restart removed the old execution directory'},
                stderr='Remote execution directory was lost after a verified container restart; no command was relaunched.',
                restart_evidence=evidence))
        except (ExecutionUnavailable, OSError, ValueError, KeyError, TypeError):
            return None

    def reconcile_unrecorded(self):
        """Explicit recovery of a submission whose on_job commit was interrupted.

        The deterministic remote ID is inspected before any upload/launch. An
        existing directory is recovered by that ID. An absent directory plus no
        committed handle cannot represent a launched command in this protocol:
        on_job must finish before source upload and start. Record an admission
        failure rather than silently launching the ambiguous request again.
        """
        if 'scheduling' in self.config:
            # The node broker atomically deduplicates enqueue by this logical ID.
            # A queued job is not a missing remote command or a failed admission.
            self.on_job(self.logical_job)
            self.previous_job = self.logical_job
            return None
        if self.previous_job:
            raise ValueError('A saved Remote handle does not need unrecorded admission recovery')
        if not self.config.get('uid'):
            raise ExecutionUnavailable('Remote admission recovery requires a pinned Pod UID')
        current = self._pod_state()
        if current.get('pod_uid') != self.config['uid']:
            raise ExecutionUnavailable('Remote admission recovery Pod UID differs from the pinned target')
        code = "import json,pathlib,sys; print('AKA_REMOTE='+json.dumps({'directory_exists':pathlib.Path(sys.argv[1]).exists()}))"
        probe = self._exec([self.config.get('python', 'python3'), '-c', code, self.root],
                           retry_safe=True, operation='admission_directory_probe')
        if type(probe.get('directory_exists')) is not bool:
            raise ExecutionUnavailable('Remote admission recovery returned no directory evidence')
        evidence = dict(job_id=self.job, remote_directory=self.root, observed_at=time.time(),
                        current_container=current, directory_exists=probe['directory_exists'],
                        missing_persisted_handle=True)
        atomic_json(self.directory/'remote-admission-recovery.json', evidence)
        self.on_job(self.logical_job)
        if not probe['directory_exists']:
            return self._result('failed', dict(exit_code=None, timed_out=False, cancelled=False,
                submission_not_admitted=True, failure_kind='submission_interrupted', stdout='',
                error={'code':'infra/submission_interrupted','message':'Unrecorded submission has no remote directory'},
                stderr='Remote submission had no committed handle and its directory is absent; no command was relaunched.',
                admission_evidence=evidence))
        atomic_json(self.directory/'remote-job.json', dict(job_id=self.job,pod=self.config['pod'],
            remote_directory=self.root,prepared_at=time.time(),container_identity=current,
            reconciled_missing_handle=True))
        self.previous_job = self.job
        return None

    def _initialize(self, spec):
        runner = Path(__file__).with_name('remote_worker.py').read_bytes()
        code = '''import base64,hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]);p.mkdir(parents=True,exist_ok=True)
s=json.loads(sys.argv[2]);f=p/'spec.json'
if f.exists() and json.loads(f.read_text())!=s:raise ValueError('Existing execution has different inputs')
t=f.with_suffix('.tmp');t.write_text(json.dumps(s));t.replace(f)
r=p/'runner.py';b=base64.b64decode(sys.argv[3])
if r.exists() and r.read_bytes()!=b:raise ValueError('Existing execution uses a different runner version')
if not r.exists():
 t=r.with_suffix('.tmp');t.write_bytes(b);t.replace(r)
parts={f.name[5:]:hashlib.sha256(f.read_bytes()).hexdigest() for f in p.glob('part-*') if f.name[5:].isdigit() and f.is_file()}
print('AKA_REMOTE='+json.dumps({'prepared':True,'part_sha256':parts}))'''
        return self._exec([self.config.get('python', 'python3'), '-c', code, self.root,
                    json.dumps(spec), base64.b64encode(runner).decode()], retry_safe=True, operation='initialize')

    def _download(self, name, *, required=False):
        info = self._exec([self.config.get('python', 'python3'), '-c', FILE_INFO, self.root, name],
                          retry_safe=True, operation='artifact_manifest')
        if not info['exists']:
            if required:
                raise ExecutionUnavailable('Remote job is missing its execution log')
            return None
        if info['bytes'] > MAX_ARTIFACT_BYTES:
            raise ExecutionUnavailable('Remote artifact exceeds the 64 MiB transfer limit')
        if (os.environ.get('AKA_REMOTE_COMPRESS_ARTIFACTS') == '1' or ('cache' in self.config and self.config['cache'].get('mode','cas')=='cas')) and info['bytes'] >= 500000:
            meta = self._exec([self.config.get('python', 'python3'), '-c', COMPRESS_FILE, self.root, name],
                              retry_safe=True, operation='artifact_compress')
            if meta['raw_bytes'] != info['bytes']:
                raise ExecutionUnavailable('Remote artifact changed during transfer')
            if name.startswith('workspace/'):
                cached = self.directory/'artifacts'/relative(name[len('workspace/'):])
                if cached.is_file() and cached.stat().st_size == info['bytes']:
                    data = cached.read_bytes()
                    if hashlib.sha256(data).hexdigest() == meta['sha256']:
                        return data
            if meta['bytes'] < info['bytes']:
                encoded = self._read_file(meta['name'], meta['bytes'], meta['encoded_sha256'])
                try:
                    decoder = zlib.decompressobj()
                    data = decoder.decompress(encoded, info['bytes'] + 1)
                except zlib.error:
                    raise ExecutionUnavailable('Invalid compressed Remote artifact') from None
                if not decoder.eof or decoder.unused_data or len(data) != info['bytes']:
                    raise ExecutionUnavailable('Incomplete compressed Remote artifact')
            else:
                data = self._read_file(name, info['bytes'], info['sha256'])
            if hashlib.sha256(data).hexdigest() != meta['sha256']:
                raise ExecutionUnavailable('Remote artifact checksum mismatch')
            return data
        return self._read_file(name, info['bytes'], info['sha256'])

    def _artifact_range(self, name, expected, offset, size):
        """Read a bounded range of an original output without downloading the whole report."""
        from .remote_cache import valid_hash
        valid_hash(expected['sha256'])
        if (type(expected['bytes']) is not int or expected['bytes']<0
                or type(offset) is not int or not 0<=offset<=expected['bytes']
                or type(size) is not int or not 0<size<=500000):
            raise ValueError('Invalid artifact byte range')
        wanted=min(size,CHUNK)  # Bound encoded Remote responses as well as raw ranges.
        code='''import base64,hashlib,json,os,pathlib,sys,uuid
try:
 r=pathlib.Path(sys.argv[1]).resolve();p=(r/sys.argv[2]).resolve();sha=sys.argv[3]
 total,offset,size=map(int,sys.argv[4:7])
 if not p.is_relative_to(r):raise ValueError('Artifact path escapes attempt')
 with p.open('rb') as f:
  def identity():
   s=os.fstat(f.fileno());return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
  before=identity()
  if before[2]!=total:raise ValueError('Artifact size changed')
  directory=r/'.transfer-cache';directory.mkdir(exist_ok=True);proof=directory/('range-'+sha+'.json')
  try:cached=json.loads(proof.read_text())
  except (FileNotFoundError,ValueError):cached={}
  if cached!={'sha256':sha,'identity':before}:
   digest=hashlib.sha256()
   while part:=f.read(1048576):digest.update(part)
   if digest.hexdigest()!=sha or identity()!=before:raise ValueError('Artifact checksum changed')
   temporary=proof.with_name(proof.name+'.'+uuid.uuid4().hex+'.tmp')
   temporary.write_text(json.dumps({'sha256':sha,'identity':before}));temporary.replace(proof)
  f.seek(offset);data=f.read(min(size,total-offset))
  if identity()!=before:raise ValueError('Artifact changed during read')
 print('AKA_REMOTE='+json.dumps({'artifact_id':sha,'sha256':sha,'bytes':total,'offset':offset,
  'chunk_sha256':hashlib.sha256(data).hexdigest(),'base64':base64.b64encode(data).decode()}))
except Exception as error:print('AKA_REMOTE='+json.dumps({'artifact_error':type(error).__name__}))
'''
        value=self._exec([self.config.get('python','python3'),'-c',code,self.root,name,
            expected['sha256'],str(expected['bytes']),str(offset),str(wanted)],
            retry_safe=True,operation='artifact_range')
        if value.get('artifact_error'):
            raise Failure('artifact_reference_changed_or_unavailable')
        data=base64.b64decode(value['base64'],validate=True)
        if (value.get('sha256')!=expected['sha256'] or value.get('bytes')!=expected['bytes']
                or value.get('offset')!=offset or len(data)!=min(wanted,expected['bytes']-offset)
                or hashlib.sha256(data).hexdigest()!=value.get('chunk_sha256')):
            raise Failure('artifact_reference_checksum_mismatch')
        return {key:value[key] for key in ['artifact_id','bytes','offset','base64']}

    def _read_file(self, name, size, digest):
        def fetch(offset, wanted):
            value = self._rpc('read', name, offset, wanted)
            try:
                return base64.b64decode(value['base64'], validate=True)
            except (ValueError, KeyError, TypeError):
                raise Failure('invalid_artifact_chunk', transient=True) from None
        return remote_transfer.read(self.directory, self.root + '/' + name, size, digest, fetch)

    def run(self, command):
        if 'scheduling' in self.config:
            from .remote_pool import PoolDriver
            return PoolDriver(self, command).run()
        self._prepare_attempt(command)
        recoveries = 0
        while True:
            try:
                saved = self._local_result(command)
                return saved if saved is not None else self._run(command)
            except ExecutionUnavailable as error:
                if isinstance(error, Failure) and error.code in {
                    'authentication_required','configuration_or_protocol_rejected',
                    'local_storage_full','local_storage_unavailable','local_cli_unavailable'}:
                    raise
                lost = self._confirmed_restart_loss()
                if lost is not None:
                    if self.policy.replay_on_restart and self.attempt < self.policy.max_replays and self.control() == 'run':
                        payload = lost.details['receipt']['payload']
                        proof = payload.get('restart_evidence', {})
                        before = proof.get('previous_container', {}).get('image_id')
                        after = proof.get('current_container', {}).get('image_id')
                        if before and before == after:
                            self._next_attempt(command)
                            continue
                        return self._result('failed', {**payload,
                            'replay_blocked':'container_image_identity_missing_or_changed'})
                    return lost
                if not isinstance(error, Failure) or not error.transient or getattr(error, 'exhausted', False):
                    raise
                recoveries += 1
                event(self.directory, 'same_execution_recovery', error, recoveries)
                if recoveries >= self.policy.attempts:
                    raise
                time.sleep(min(self.policy.base_delay_s * recoveries, self.policy.max_delay_s))
            except OSError as error:
                raise Failure('local_storage_full' if error.errno == 28 else 'local_storage_unavailable') from None

    def _prepare_attempt(self, command):
        files = self.declared_input_files if self.declared_input_files is not None else tree(command.workspace)
        # Freeze once even for direct Executor callers. Restart recovery must
        # never re-read a subsequently edited live directory into the old ID.
        self.declared_input_files = dict(files)
        signature = dict(argv=list(command.command), environment={**self.config.get('environment', {}), **dict(command.environment)},
                         timeout=command.timeout, outputs=list(command.outputs),
                         source={k:hashlib.sha256(v).hexdigest() for k,v in sorted(files.items())})
        self.fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        marker = self.directory/'remote-attempt.json'
        if marker.exists():
            saved = json.loads(marker.read_text())
            if saved.get('logical_job') != self.logical_job or saved.get('fingerprint') != self.fingerprint:
                raise Failure('recovery_input_mismatch')
            self.attempt = saved['attempt']
            if type(self.attempt) is not int or not 0 <= self.attempt <= 1000:
                raise Failure('invalid_attempt_journal')
            self.previous_job = self.logical_job if saved.get('admitted') else ''
        self._select_attempt()

    def _select_attempt(self):
        self.job = self.logical_job if self.attempt == 0 else self.logical_job + '-r' + str(self.attempt)
        self.root = str(PurePosixPath(self.config.get('remote_root', '/tmp/aka-executions'))/self.job)

    def _save_attempt(self, *, admitted):
        atomic_json(self.directory/'remote-attempt.json', dict(logical_job=self.logical_job, attempt=self.attempt,
                    fingerprint=self.fingerprint, admitted=admitted))

    def _next_attempt(self, command):
        evidence = self.directory/'remote-restart-loss.json'
        if evidence.exists():
            atomic_json(self.directory/('restart-loss-attempt-' + str(self.attempt) + '.json'), json.loads(evidence.read_text()))
        self.attempt += 1
        self.previous_job = ''
        self.chunk_bytes, self.legacy_spec = CHUNK, False
        self._select_attempt()
        self._save_attempt(admitted=False)
        event(self.directory, 'restart_replay', Failure('verified_container_restart'), self.attempt)

    def _local_result(self, command):
        path = self.directory/'remote-collected.json'
        if not path.exists():return None
        value = json.loads(path.read_text())
        if not value.get('complete') or value.get('job') != self.job or value.get('fingerprint') != self.fingerprint:
            return None
        expected = {'stdout.log','stderr.log', *['workspace/'+x for x in command.outputs]}
        if set(value.get('files', {})) != expected:
            return None
        if any(value['files'].get(name) is None for name in ['stdout.log','stderr.log']):
            return None
        outputs = {}
        for name, info in value['files'].items():
            if info is None:outputs[name] = None;continue
            file = remote_transfer.cache_file(self.directory, info['sha256'])
            if file.is_symlink() or not file.is_file() or file.stat().st_size != info['bytes'] or remote_transfer.file_hash(file) != info['sha256']:
                return None
            outputs[name] = file.read_bytes()
        return self._collected_result(value['result'], outputs, command.outputs)

    def _collected_result(self, result, files, output_names):
        artifacts = {name:{'base64':base64.b64encode(files['workspace/'+name]).decode()}
                     for name in output_names if files.get('workspace/'+name) is not None}
        state = 'cancelled' if result['cancelled'] else 'failed' if result['timed_out'] or result['exit_code'] != 0 else 'completed'
        missing = [name for name in output_names if files.get('workspace/'+name) is None]
        extra = {}
        if state == 'completed' and missing:
            state = 'failed'
            extra = dict(failure_kind='missing_outputs', missing_outputs=missing)
        return self._result(state, {**result, 'stdout':files['stdout.log'].decode(errors='replace'),
            'stderr':files['stderr.log'].decode(errors='replace'), 'artifacts':artifacts, **extra})

    def _run(self, command):
        self.capabilities.require(command)
        output_names = [relative(name).as_posix() for name in command.outputs]
        if not self.previous_job:
            stream = io.BytesIO()
            files = self.declared_input_files if self.declared_input_files is not None else tree(command.workspace)
            with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in sorted(files.items()):
                    archive.writestr(zipfile.ZipInfo(relative(name).as_posix(), date_time=(1980, 1, 1, 0, 0, 0)), data,
                                     compress_type=zipfile.ZIP_DEFLATED)
            blob = stream.getvalue()
            environment = {'PYTHONPYCACHEPREFIX': '.aka-cache/python', 'TORCH_EXTENSIONS_DIR': '.aka-cache/torch',
                           'TRITON_CACHE_DIR': '.aka-cache/triton', 'CUDA_CACHE_PATH': '.aka-cache/cuda',
                           **self.config.get('environment', {}), **dict(command.environment)}
            spec = {'argv': list(command.command), 'environment': environment, 'timeout': command.timeout,
                    'archive_sha256': hashlib.sha256(blob).hexdigest(),
                    'parts': (len(blob)+self.chunk_bytes-1)//self.chunk_bytes}
            if not self.legacy_spec:
                spec['chunk_bytes'] = self.chunk_bytes
            metadata = self._job_metadata()
            prepared = self._initialize(spec) or {}
            # The acknowledgement bounds when preparation actually happened,
            # including a restart racing with the preceding identity lookup.
            metadata['prepared_at'] = time.time()
            self.on_job(self.logical_job)
            atomic_json(self.directory/'remote-job.json', metadata)
            self.previous_job = self.logical_job
            self._save_attempt(admitted=True)
            for index, offset in enumerate(range(0, len(blob), self.chunk_bytes)):
                if self.control() != 'run':
                    return self._result('cancelled', {'reason': 'Stopped before remote launch'})
                chunk = blob[offset:offset+self.chunk_bytes]
                if prepared.get('part_sha256', {}).get(str(index)) == hashlib.sha256(chunk).hexdigest():
                    continue
                self._rpc('part', index, base64.b64encode(chunk).decode())
            value = self._rpc('start')
        else:
            value = self._rpc('status')
            if value['state'] == 'prepared':
                # Uploaded input may be incomplete after a transport interruption.
                # Re-stage deterministic chunks; start is guarded by a remote claim.
                encoded = self._rpc('read', 'spec.json', 0, 500000)
                previous_spec = json.loads(base64.b64decode(encoded['base64'], validate=True))
                self.legacy_spec = 'chunk_bytes' not in previous_spec
                self.chunk_bytes = previous_spec.get('chunk_bytes', LEGACY_CHUNK)
                if type(self.chunk_bytes) is not int or not 0 < self.chunk_bytes <= CHUNK:
                    raise ExecutionUnavailable('Unsupported saved Remote upload chunk size')
                self.previous_job = ''
                return self._run(command)
        while value['state'] != 'finished':
            if value['state'] in {'unknown', 'lost'}:
                raise ExecutionUnavailable('Remote launch outcome is unknown or its runner was lost; no automatic relaunch')
            if self.control() == 'cancel' or (command.cancel_event is not None and command.cancel_event.is_set()):
                self._rpc('cancel')
            time.sleep(self.config.get('poll_interval_s', 5))
            value = self._rpc('status')
        result = value['result']
        previous = {}
        checkpoint = self.directory/'remote-collected.json'
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            if saved.get('job') == self.job and saved.get('fingerprint') == self.fingerprint and saved.get('result') == result:
                previous = saved.get('files', {})
        journal = dict(job=self.job, fingerprint=self.fingerprint, result=result, files={}, complete=False)
        collected = {}
        for name in ['stdout.log','stderr.log', *['workspace/'+x for x in output_names]]:
            info = previous.get(name)
            cached = remote_transfer.cache_file(self.directory, info['sha256']) if info else None
            if cached and not cached.is_symlink() and cached.is_file() and cached.stat().st_size == info['bytes'] and remote_transfer.file_hash(cached) == info['sha256']:
                data = cached.read_bytes()
            else:
                data = self._download(name, required=name in {'stdout.log','stderr.log'})
            collected[name] = data
            if data is not None:
                digest = remote_transfer.store(self.directory, data)
                journal['files'][name] = dict(sha256=digest, bytes=len(data))
                if name.startswith('workspace/'):
                    path = self.directory/'artifacts'/relative(name[len('workspace/'):])
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_name(path.name+'.tmp')
                    temporary.write_bytes(data)
                    temporary.replace(path)
            else:journal['files'][name] = None
            atomic_json(self.directory/'remote-collected.json', journal)
        journal['complete'] = True
        atomic_json(self.directory/'remote-collected.json', journal)
        return self._collected_result(result, collected, output_names)

    def _result(self, status, payload):
        return JobResult(0 if status == 'completed' else 1, payload.get('stdout', ''), payload.get('stderr', ''),
                         payload.get('exit_code'), self.resource_identity,
                         details={'receipt': {'status': status, 'payload': payload, 'job_id': self.logical_job},
                                  'infra_attempt':self.attempt,'physical_job':self.job})
