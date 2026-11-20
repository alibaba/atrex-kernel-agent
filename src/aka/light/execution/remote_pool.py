"""Opt-in pooled REMOTE execution, preserving the unified execution interface."""
import base64
from contextlib import contextmanager
from contextvars import ContextVar
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import time
import zipfile
import zlib

from aka.light.execution.contract import ExecutionUnavailable
from aka.light.tasks.content import atomic_json, tree
from .remote_resilience import Failure

_GATE = ContextVar('aka_remote_pool_transport', default=frozenset())

# Bound the aggregate reply, including when many individually small reports
# would otherwise all be embedded in one Remote response. Collector semantics and
# its file hashes stay in the original, pinned remote helper.
COLLECT_METADATA='''import base64,hashlib,importlib.util,json,pathlib,sys,zlib
p=pathlib.Path(sys.argv[1])
if hashlib.sha256(p.read_bytes()).hexdigest()!=sys.argv[2]:raise ValueError('Collector source changed')
s=importlib.util.spec_from_file_location('aka_original_collector',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
v=m.rpc(pathlib.Path(sys.argv[3]),'collect',json.loads(sys.argv[4]));budget=40000;original=0
for info in v.get('files',{}).values():
 data=info.get('base64','');original+=len(data)
 if len(data)>budget:info.pop('base64',None)
 else:budget-=len(data)
raw=json.dumps(v,separators=(',',':')).encode()
print('AKA_REMOTE='+json.dumps({'pool_collect_zlib':base64.b64encode(zlib.compress(raw,6)).decode(),
 'original_inline_chars':original,'kept_inline_chars':40000-budget}))
'''


def decode_collect_reply(value):
    if 'pool_collect_zlib' not in value:return value
    try:
        decoder=zlib.decompressobj()
        raw=decoder.decompress(base64.b64decode(value['pool_collect_zlib'],validate=True),16*1024*1024+1)
        if len(raw)>16*1024*1024 or not decoder.eof or decoder.unused_data:raise ValueError('Invalid collection metadata')
        result=json.loads(raw)
        if not isinstance(result,dict):raise ValueError('Invalid collection metadata')
        return result
    except (ValueError,TypeError,zlib.error):
        raise Failure('pool_collection_metadata_invalid') from None


def state_root():
    result = Path(os.environ.get('AKA_REMOTE_POOL_STATE_DIR', str(Path.home() / '.cache/aka/remote-pool')))
    result.mkdir(parents=True, exist_ok=True)
    return result


def gate_name(config):
    if config is None:return 'global'
    route={k:config.get(k,'') for k in ['pod','uid','container','namespace','context']}
    return hashlib.sha256(json.dumps(route,sort_keys=True).encode()).hexdigest()[:24]


@contextmanager
def transport_gate(config=None):
    """One RPC per Pod, at most two pooled RPCs pending source admission."""
    key=gate_name(config)
    if key in _GATE.get():
        yield
        return
    if _GATE.get():raise RuntimeError('Nested cross-Pod transport gates are unsupported')
    with (state_root() / ('transport-'+key+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        slot=None
        try:
            while slot is None:
                for index in range(2):
                    candidate=(state_root()/('transport-inflight-'+str(index)+'.lock')).open('a')
                    try:fcntl.flock(candidate,fcntl.LOCK_EX|fcntl.LOCK_NB);slot=candidate;break
                    except BlockingIOError:candidate.close()
                if slot is None:time.sleep(.02)
            token = _GATE.set(frozenset({key}))
            yield
        finally:
            if slot is not None:
                _GATE.reset(token);slot.close()


def in_transport_gate(config=None):
    return gate_name(config) in _GATE.get()


def validate_cache(value):
    allowed={'mode','research_inputs','artifact_mode','max_source_bytes','max_build_bytes','ttl_s','transport','oss_prefix'}
    if not isinstance(value,dict) or set(value)-allowed:raise ValueError('Invalid REMOTE cache settings')
    for key,choices,default in [('mode',{'archive','cas'},'cas'),('research_inputs',{'all','code'},'all'),
                                ('artifact_mode',{'inline','reference'},'inline'),('transport',{'inline','oss'},'inline')]:
        if value.get(key,default) not in choices:raise ValueError('Invalid cache.'+key)
    for key,default in [('max_source_bytes',16*1024**3),('max_build_bytes',16*1024**3),('ttl_s',86400)]:
        if type(value.get(key,default)) is not int or value.get(key,default)<1:raise ValueError('Invalid cache.'+key)
    if value.get('transport')=='oss' and (not isinstance(value.get('oss_prefix'),str) or not value['oss_prefix'].startswith('oss://')):
        raise ValueError('OSS transport requires a private oss_prefix')


def content_enabled(config):return 'cache' in config and config['cache'].get('mode','cas')=='cas'


def validate(options):
    allowed = {'gpu_ids', 'gpu_count', 'max_wait_s', 'poll_interval_s', 'replay_safe',
               'max_contamination_retries', 'monitor_interval_s', 'task_key', 'topology', 'reservation_after_s'}
    if not isinstance(options, dict) or set(options) - allowed:
        raise ValueError('Invalid REMOTE scheduling options')
    devices = options.get('gpu_ids')
    if not isinstance(devices, list) or not devices or len(devices) != len(set(devices)) or not all(type(x) is int and 0 <= x < 256 for x in devices):
        raise ValueError('scheduling.gpu_ids must explicitly list allowed physical GPUs')
    count = options.get('gpu_count', 1)
    if type(count) is not int or not 1 <= count <= len(devices):
        raise ValueError('Invalid atomic GPU group size')
    for key, default, minimum in [('max_wait_s', 86400, .01), ('poll_interval_s', 10, .01),
                                   ('monitor_interval_s', .1, .02)]:
        value = options.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
            raise ValueError('Invalid scheduling.' + key)
    if options.get('monitor_interval_s', .1) > 1:
        raise ValueError('Interference monitoring interval cannot exceed one second')
    if type(options.get('replay_safe', False)) is not bool:
        raise ValueError('replay_safe must be a caller declaration')
    retries = options.get('max_contamination_retries', 1000)
    if type(retries) is not int or not 0 <= retries <= 1000000:
        raise ValueError('Invalid contamination retry limit')
    if 'task_key' in options and (not isinstance(options['task_key'], str) or not options['task_key']):
        raise ValueError('Invalid task identity')
    if options.get('topology','any') not in {'any','p2p','nvlink'}:
        raise ValueError('Invalid requested GPU topology')
    aging=options.get('reservation_after_s',30)
    if type(aging) not in (int,float) or not math.isfinite(aging) or not 0 <= aging <= 86400:
        raise ValueError('Invalid GPU reservation aging')


BOOTSTRAP = '''import base64,hashlib,json,pathlib,sys,os,signal,sqlite3,time
r=pathlib.Path(sys.argv[1]);r.mkdir(parents=True,exist_ok=True)
b=base64.b64decode(sys.argv[2]);sha=hashlib.sha256(b).hexdigest();p=r/('agent-'+sha+'.py')
if p.exists() and p.read_bytes()!=b:raise ValueError('Pool agent checksum mismatch')
if not p.exists():
 t=p.with_suffix('.tmp');t.write_bytes(b);t.replace(p)
m=r/'server.json'
if m.exists():
 h=json.loads(m.read_text());proc=pathlib.Path('/proc',str(h['pid']))
 try:
  raw=(proc/'stat').read_text();fields=raw[raw.rindex(')')+2:].split();alive=fields[19]==h['token'] and fields[0]!='Z'
  args=(proc/'cmdline').read_bytes().split(bytes([0]))
 except FileNotFoundError:alive=False;args=[]
 old=next((x.decode() for x in args if x.startswith(str(r).encode()+b'/agent-')),None)
 if alive and old and old!=str(p):
  db=sqlite3.connect(r/'queue.sqlite3',timeout=10);db.execute('BEGIN IMMEDIATE')
  active=db.execute("SELECT COUNT(*) FROM jobs WHERE state IN ('queued','running')").fetchone()[0]
  db.close()
  if not active:
   if b'serve' not in args:raise ValueError('Pool agent identity mismatch')
   os.kill(h['pid'],signal.SIGTERM)
   for _ in range(100):
    try:
     raw=(proc/'stat').read_text();f=raw[raw.rindex(')')+2:].split()
     if f[19]!=h['token'] or f[0]=='Z':break
    except FileNotFoundError:break
    time.sleep(.01)
v={'agent':str(p),'sha256':sha}
if len(sys.argv)>3:
 g=base64.b64decode(sys.argv[3]);h=hashlib.sha256(g).hexdigest();d=r/('support-'+h);d.mkdir(exist_ok=True)
 f=d/'aka_distributed.py'
 if f.exists() and f.read_bytes()!=g:raise ValueError('Distributed runner checksum mismatch')
 if not f.exists():
  t=f.with_suffix('.tmp');t.write_bytes(g);t.replace(f)
 v.update(distributed_runner=str(f),distributed_runner_sha256=h)
if len(sys.argv)>4:
 for name,data in zip(['aka_cache.py','runner.py'],sys.argv[4:6]):
  b=base64.b64decode(data);h=hashlib.sha256(b).hexdigest();d=r/('support-'+h);d.mkdir(exist_ok=True);f=d/name
  if f.exists() and f.read_bytes()!=b:raise ValueError('Content support checksum mismatch')
  if not f.exists():
   t=f.with_suffix('.tmp');t.write_bytes(b);t.replace(f)
  v['content_engine' if name=='aka_cache.py' else 'artifact_runner']={'path':str(f),'sha256':h}
print('AKA_REMOTE='+json.dumps(v))
'''


class PoolDriver:
    def __init__(self, executor, command):
        self.e, self.command = executor, command
        self.options = executor.config['scheduling']
        self.host_marker = executor.directory / 'remote-pool.json'
        self.routing = {k: executor.config.get(k, '') for k in ['pod', 'uid', 'container', 'namespace', 'context']}
        key = hashlib.sha256(json.dumps(self.routing, sort_keys=True).encode()).hexdigest()[:24]
        # Shared across every task root and caller on this container incarnation.
        self.pool_root = '/tmp/aka-remote-pool-v1/' + key
        self.cache = state_root() / (key + '.json')
        self.support_cache=state_root()/(key+'-support-v2.json')
        self.ack_queue=state_root()/(key+'-collection-acks.json')
        self.agent = None
        self.distributed_runner = None
        self.content_engine = None
        self.artifact_runner = None
        self._support_repaired=False
        self.requested_support={}
        self.content_admitted=False
        self.source_root = str(PurePosixPath(executor.config.get('remote_root', '/tmp/aka-executions')) / executor.logical_job)

    def rpc(self, operation, value):
        if self.agent is None:
            source = Path(__file__).with_name('remote_pool_worker.py').read_bytes()
            group_source = Path(__file__).with_name('remote_distributed_worker.py').read_bytes()
            extra=[]
            if content_enabled(self.e.config):
                from .remote_cache import Store
                assets=Store(state_root()/'support-v2')
                extra=[]
                for key,name in [('content_engine','remote_cache.py'),('artifact_runner','remote_worker.py')]:
                    record=self.requested_support.get(key)
                    data=assets.get(record['sha256']) if record else Path(__file__).with_name(name).read_bytes()
                    assets.put(data);extra.append(base64.b64encode(data).decode())
                if self.requested_support.get('distributed_runner_sha256'):
                    group_source=assets.get(self.requested_support['distributed_runner_sha256'])
                else:assets.put(group_source)
            expected=[hashlib.sha256(source).hexdigest(),hashlib.sha256(group_source).hexdigest(),
                      *[hashlib.sha256(base64.b64decode(x)).hexdigest() for x in extra]]
            boot=getattr(self,'boot',None)
            cached=json.loads(self.support_cache.read_text()) if self.support_cache.exists() else {}
            if boot and cached.get('boot')==boot and cached.get('hashes')==expected and cached.get('python')==self.e.config.get('python','python3'):
                answer=cached['answer']
            else:
                answer = self.e._exec([self.e.config.get('python', 'python3'), '-c', BOOTSTRAP,
                                      self.pool_root, base64.b64encode(source).decode(),base64.b64encode(group_source).decode(),*extra],
                                     retry_safe=True, operation='pool_bootstrap')
                if boot:atomic_json(self.support_cache,{'boot':boot,'hashes':expected,'answer':answer,'python':self.e.config.get('python','python3')})
            if answer['sha256'] != hashlib.sha256(source).hexdigest():
                raise Failure('pool_agent_identity_mismatch')
            self.agent = answer['agent']
            if answer.get('distributed_runner_sha256') != hashlib.sha256(group_source).hexdigest():
                raise Failure('distributed_runner_identity_mismatch')
            self.distributed_runner = answer['distributed_runner']
            if extra:
                self.content_engine=answer['content_engine'];self.artifact_runner=answer['artifact_runner']
                for record,data in zip([self.content_engine,self.artifact_runner],extra):
                    if record['sha256']!=hashlib.sha256(base64.b64decode(data)).hexdigest():raise Failure('content_support_identity_mismatch')
        if operation.startswith('content_'):value={**value,'engine':self.content_engine}
        pending_acks=json.loads(self.ack_queue.read_text()) if self.ack_queue.exists() else []
        if pending_acks:value={**value,'collection_acks':pending_acks[:1]}
        try:
            if operation=='collect':
                reply=self.e._exec([self.e.config.get('python','python3'),'-c',COLLECT_METADATA,
                    self.agent,hashlib.sha256(Path(__file__).with_name('remote_pool_worker.py').read_bytes()).hexdigest(),
                    self.pool_root,json.dumps(value,sort_keys=True)],retry_safe=True,operation='pool_collect')
                result=decode_collect_reply(reply)
                atomic_json(self.e.directory/'collect-rpc.json',{'original_inline_chars':reply.get('original_inline_chars'),
                    'kept_inline_chars':reply.get('kept_inline_chars'),'encoded_reply_bytes':len(json.dumps(reply))})
            else:
                result=self.e._exec([self.e.config.get('python', 'python3'), self.agent, self.pool_root,
                                     operation, json.dumps(value, sort_keys=True)],retry_safe=True,operation='pool_'+operation)
            if result.get('acknowledged_jobs') or result.get('rejected_collection_acks'):
                from .remote_cache import lock
                with lock(self.ack_queue.with_suffix('.lock')):
                    current=json.loads(self.ack_queue.read_text()) if self.ack_queue.exists() else []
                    handled=set(result.get('acknowledged_jobs',[])+result.get('rejected_collection_acks',[]))
                    atomic_json(self.ack_queue,[e for e in current if e['job_id'] not in handled])
            return result
        except Failure as error:
            if error.code!='remote_state_missing' or self._support_repaired:raise
            # A cache eviction/restart may remove helper files. Reinstall the
            # same bytes and retry the idempotent RPC under the original ID.
            self.support_cache.unlink(missing_ok=True);self.agent=None
            self._support_repaired=True
            return self.rpc(operation,value)

    def refresh(self, *, force=False):
        with transport_gate(self.routing):
            if self.cache.exists() and not force:
                cached = json.loads(self.cache.read_text())
                if time.time() - cached['observed_at'] < self.options.get('poll_interval_s', 10):
                    return cached['value']
            value = self.rpc('status_many', {})
            atomic_json(self.cache, {'observed_at': time.time(), 'value': value})
            return value

    def prepare(self):
        e, command = self.e, self.command
        files = e.declared_input_files if e.declared_input_files is not None else tree(command.workspace)
        files = dict(files)
        use_content=content_enabled(e.config)
        if self.host_marker.exists() and 'archive_sha256' in json.loads(self.host_marker.read_text())['request']:
            use_content=False  # Saved executions keep their original transport and identity.
        environment = {**e.config.get('environment', {}), **dict(command.environment)}
        if any(k in environment for k in ['CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES', 'AKA_ASSIGNED_GPU_UUIDS']):
            raise ValueError('Pooled execution owns device selection; use scheduling.gpu_ids')
        distributed=None
        if 'distributed' in e.config and getattr(e,'launch_mode',None)!='command':
            from .remote_distributed_worker import validate_options, PROTECTED
            distributed=validate_options(e.config['distributed'],self.options.get('gpu_count',1))
            if set(environment)&PROTECTED or any(k.startswith('AKA_DISTRIBUTED_') for k in environment):
                raise ValueError('Distributed rank identity is owned by the launcher')
        if use_content:
            from . import remote_cache as content
            source_manifest=content.manifest(files);blob=b''
        else:
            archive = io.BytesIO()
            with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
                for name, data in sorted(files.items()):
                    bundle.writestr(zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0)), data,
                                    compress_type=zipfile.ZIP_DEFLATED)
            blob = archive.getvalue()
        task = self.options.get('task_key') or next((x for x in e.directory.parts if x.startswith('task-')), e.execution_id)
        spec = {'job_id': e.logical_job, 'task': task, 'source_root': self.source_root,
                'archive_sha256': hashlib.sha256(blob).hexdigest(), 'argv': list(command.command),
                'environment': {'PYTHONPYCACHEPREFIX': '.aka-cache/python', **environment},
                'outputs': list(command.outputs), 'timeout_s': command.timeout,
                'gpu_ids': self.options['gpu_ids'], 'gpu_count': self.options.get('gpu_count', 1),
                'replay_safe': self.options.get('replay_safe', False),
                'monitor_interval_s': self.options.get('monitor_interval_s', .1),
                'max_contamination_retries': self.options.get('max_contamination_retries', 1000)}
        if use_content:
            # Support paths are pure content addresses and need no RPC to derive.
            def support(name,remote_name):
                data=Path(__file__).with_name(name).read_bytes()
                sha=content.Store(state_root()/'support-v2').put(data)
                return {'path':self.pool_root+'/support-'+sha+'/'+remote_name,'sha256':sha}
            spec.pop('archive_sha256')
            spec.update(source_manifest=source_manifest,source_manifest_id=content.validate_manifest(source_manifest),
                content_engine=support('remote_cache.py','aka_cache.py'),artifact_runner=support('remote_worker.py','runner.py'),
                cache_options=e.config.get('cache',{}),python=e.config.get('python','python3'))
        if self.options.get('topology','any')!='any':spec['topology']=self.options['topology']
        if 'reservation_after_s' in self.options:spec['reservation_after_s']=self.options['reservation_after_s']
        if distributed is not None:
            spec['distributed']=distributed
            spec['distributed_runner_sha256']=hashlib.sha256(Path(__file__).with_name('remote_distributed_worker.py').read_bytes()).hexdigest()
            spec['distributed_runner']=self.pool_root+'/support-'+spec['distributed_runner_sha256']+'/aka_distributed.py'
            if use_content:content.Store(state_root()/'support-v2').put(Path(__file__).with_name('remote_distributed_worker.py').read_bytes())
        elif getattr(e,'launch_mode',None)=='command':spec['launch_mode']='command'
        if use_content:
            if self.host_marker.exists():
                frozen=json.loads(self.host_marker.read_text())['request']
                if frozen.get('source_manifest_id'):
                    for name in ['content_engine','artifact_runner','distributed_runner','distributed_runner_sha256']:
                        if name in frozen:spec[name]=frozen[name]
            self.requested_support={k:spec[k] for k in ['content_engine','artifact_runner','distributed_runner_sha256'] if k in spec}
        fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        if self.host_marker.exists():
            previous = json.loads(self.host_marker.read_text())
            if previous['fingerprint'] != fingerprint:
                raise Failure('recovery_input_mismatch')
            self.boot = previous['container_identity']
            current = e._pod_state()
            if current.get('container_id') != self.boot.get('container_id'):
                if (not self.options.get('replay_safe', False) or not self.boot.get('image_id')
                        or current.get('image_id') != self.boot.get('image_id')):
                    raise Failure('pool_container_replay_not_authorized')
                previous.setdefault('container_history', []).append(self.boot)
                self.boot = current
                previous['container_identity'] = current
                atomic_json(self.host_marker, previous)
        else:
            # Verify the pinned UID and image before any dispatch, not just when collecting results.
            self.boot = e._pod_state()
            if not self.boot or not self.boot.get('image_id'):
                raise Failure('pool_container_identity_unverified')
            atomic_json(self.host_marker, {'logical_job': e.logical_job, 'fingerprint': fingerprint,
                        'container_identity': self.boot, 'request': spec, 'created_at': time.time()})
        e.on_job(e.logical_job)  # Commit the logical handle before upload, enqueue or launch.
        if use_content:
            if not self.prepare_content(spec,files):return None
            if not self.content_admitted:self.rpc('content_enqueue',{'spec':content.wire(spec)})
            return spec
        cached = self.rpc('cached', {'sha256': spec['archive_sha256']})['cached']
        # A per-execution legacy runner supplies stat/read for large artifact recovery.
        upload = dict(argv=spec['argv'], environment=spec['environment'], timeout=command.timeout,
                      archive_sha256=spec['archive_sha256'], parts=(len(blob) + e.chunk_bytes - 1) // e.chunk_bytes,
                      chunk_bytes=e.chunk_bytes)
        e.root = self.source_root
        prepared = e._initialize(upload) or {}
        if not cached:
            for index, offset in enumerate(range(0, len(blob), e.chunk_bytes)):
                if e.control() != 'run':
                    return None
                data = blob[offset:offset + e.chunk_bytes]
                if prepared.get('part_sha256', {}).get(str(index)) != hashlib.sha256(data).hexdigest():
                    e._rpc('part', index, base64.b64encode(data).decode())
        self.rpc('enqueue', spec)
        return spec

    def prepare_content(self,spec,files):
        from . import remote_cache as content
        if self.e.control()!='run':return False
        local=content.Store(state_root()/'content-v2')
        local.put_files(files)
        journal=self.e.directory/'content-upload.json'
        previous=json.loads(journal.read_text()) if journal.exists() else {'packs':[]}
        pending=self.rpc('content_prepare',{'spec':content.wire(spec),
            'known_packs':[p['sha256'] for p in previous['packs']]})
        missing=pending['missing'];plan=[];covered=set()
        for entry in previous['packs']:
            if set(entry['blobs'])&set(missing):
                plan.append((entry['sha256'],local.get(entry['sha256']),entry['blobs']));covered.update(entry['blobs'])
        plan.extend(content.packs(files,set(missing)-covered))
        entries=[]
        for sha,raw,blobs in plan:
            local.put(raw);entries.append({'sha256':sha,'blobs':blobs,'bytes':len(raw)})
        atomic_json(journal,{'manifest_id':spec['source_manifest_id'],'packs':entries})
        stats_path=self.e.directory/'content-transfer.json'
        previous_stats=json.loads(stats_path.read_text()) if stats_path.exists() else {}
        if previous_stats and previous_stats['manifest_id']!=spec['source_manifest_id']:raise Failure('source_transfer_checkpoint_mismatch')
        stats={'manifest_id':spec['source_manifest_id'],'total_files':len(files),'total_bytes':sum(map(len,files.values())),
            'missing_blobs':max(len(missing),previous_stats.get('missing_blobs',0)),
            'missing_raw_bytes':max(sum(missing.values()),previous_stats.get('missing_raw_bytes',0)),
            'last_prepare_missing_blobs':len(missing),'preparations':previous_stats.get('preparations',0)+1,
            **{key:previous_stats.get(key,0) for key in ['uploaded_bytes','upload_calls','resumed_parts','object_batches','object_fallbacks']}}
        stats['imported_legacy_blobs']=pending.get('imported_legacy_blobs',0)+previous_stats.get('imported_legacy_blobs',0)
        def checkpoint():atomic_json(stats_path,stats)
        checkpoint()
        if pending.get('enqueued'):
            self.content_admitted=True
            atomic_json(self.e.directory/'content-transfer.json',stats)
            return True
        for sha,raw,blobs in plan:
            if self.e.config.get('cache',{}).get('transport')=='oss' and len(raw)>90000:
                from .remote_oss import Channel
                try:
                    channel=Channel(self.e.config['cache']['oss_prefix'],state_root()/'object-transfers')
                    channel.import_pack(self.e,self.content_engine,self.pool_root+'/cache-v2',sha,raw)
                    stats['uploaded_bytes']+=len(raw);stats['upload_calls']+=1
                    stats['object_batches']=stats.get('object_batches',0)+1
                    checkpoint()
                    continue
                except Failure:
                    stats['object_fallbacks']=stats.get('object_fallbacks',0)+1
            count=(len(raw)+self.e.chunk_bytes-1)//self.e.chunk_bytes
            for index,offset in enumerate(range(0,len(raw),self.e.chunk_bytes)):
                if self.e.control()!='run':return False
                part=raw[offset:offset+self.e.chunk_bytes]
                if pending.get('packs',{}).get(sha,{}).get(str(index))==content.digest(part):stats['resumed_parts']+=1;continue
                self.rpc('content_put',{'sha256':sha,'index':index,'count':count,'bytes':len(raw),'data':base64.b64encode(part).decode()})
                stats['uploaded_bytes']+=len(part);stats['upload_calls']+=1
                checkpoint()
        if self.e.control()!='run':return False
        self.rpc('content_enqueue',{'spec':content.wire(spec)})
        self.content_admitted=True
        atomic_json(self.e.directory/'content-transfer.json',stats)
        return self.e.control()=='run'

    def collect(self, status):
        e = self.e
        from .remote import MAX_ARTIFACT_BYTES
        content_store=None;known=[]
        if content_enabled(e.config):
            from .remote_cache import Store
            content_store=Store(state_root()/'artifact-v2')
            known=[p.name for p in sorted((content_store.root/'blobs').glob('*/*'),key=lambda p:p.stat().st_mtime,reverse=True)[:512] if p.is_file()]
        raw = self.rpc('collect', {'job_id': e.logical_job,**({'known_artifacts':known} if content_store else {})})
        status = raw['status']
        blocked = status['state'] == 'blocked'
        if status.get('attempt_root'):
            e.root = status['attempt_root']
        contents = {}
        artifact_stats={}
        remote_artifacts={}
        for name, info in raw['files'].items():
            if (content_store and e.config.get('cache',{}).get('artifact_mode')=='reference'
                    and info['bytes']>MAX_ARTIFACT_BYTES
                    and name.startswith(('workspace/','distributed/'))):
                # The trusted collector hashes these declared outputs in the
                # original terminal attempt. Keep large reports there; range
                # retrieval verifies the same hash before exposing any bytes.
                from .remote_cache import valid_hash
                valid_hash(info['sha256'])
                remote_artifacts[name]={'artifact_id':info['sha256'],'sha256':info['sha256'],
                    'bytes':info['bytes'],'remote_only':True}
                artifact_stats[name]={'sha256':info['sha256'],'bytes':info['bytes'],
                    'transport':'remote_reference'}
                continue
            if content_store and content_store.has(info['sha256'],info['bytes']):
                data=content_store.get(info['sha256'])
                method='cache'
            elif 'base64' in info:
                data = base64.b64decode(info['base64'], validate=True)
                method='inline'
            else:
                data=None
                if content_store and e.config.get('cache',{}).get('transport')=='oss':
                    from .remote_oss import Channel
                    try:
                        data=Channel(e.config['cache']['oss_prefix'],state_root()/'object-transfers').download(e,name,info)
                        method='oss'
                    except Failure:pass
                if data is None:data = e._download(name, required=True);method='inline_chunks'
            if len(data) != info['bytes'] or hashlib.sha256(data).hexdigest() != info['sha256']:
                raise Failure('pool_artifact_checksum_mismatch')
            contents[name] = data
            if content_store:content_store.put(data)
            artifact_stats[name]={'sha256':info['sha256'],'bytes':len(data),'transport':method}
        if content_store:atomic_json(e.directory/'artifact-transfer.json',artifact_stats)
        def artifact(data):
            if content_store and e.config.get('cache',{}).get('artifact_mode')=='reference' and len(data)>65536:
                return {'artifact_id':hashlib.sha256(data).hexdigest(),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
            return {'base64':base64.b64encode(data).decode()}
        result = status.get('result') or {}
        if result.get('contaminated') and not blocked:
            raise Failure('pool_contaminated_result_rejected')
        if blocked:
            origin='execution' if result.get('retryable_group_failure') and not result.get('contaminated') else 'infrastructure'
            result={**result,'valid_measurement':False,'pool_blocked':True,'failure_origin':origin,
                'error':{'code':'code/distributed_retry_exhausted' if origin=='execution' else 'infra/pool_blocked',
                    'message':result.get('reason') or 'The original attempt exhausted its safe recovery budget'}}
        payload = {**result, 'stdout': contents.pop('stdout.log', b'').decode(errors='replace'),
                   'stderr': contents.pop('stderr.log', b'').decode(errors='replace'),
                   'artifacts': {name[len('workspace/'):]: artifact(data)
                                 for name, data in contents.items() if name.startswith('workspace/')},
                   'pool': {'placement': status.get('placement'), 'attempt': status['attempt'],
                            'logical_job': e.logical_job, 'valid_measurement': result.get('valid_measurement', False)}}
        for name,data in contents.items():
            if name.startswith('distributed/'):
                payload['artifacts']['aka-'+name]=artifact(data)
        for name,entry in remote_artifacts.items():
            key=name[len('workspace/'):] if name.startswith('workspace/') else 'aka-'+name
            payload['artifacts'][key]=entry
        receipt = e._result('failed' if blocked else status['state'], payload)
        atomic_json(e.directory / 'pool-collected.json', receipt.details['receipt'])
        atomic_json(e.directory / 'pool-collected-proof.json', {
            'sha256': hashlib.sha256((e.directory / 'pool-collected.json').read_bytes()).hexdigest(),
            'logical_job': e.logical_job})
        if raw.get('collection_token'):
            from .remote_cache import lock
            entry={'job_id':e.logical_job,'attempt':status['attempt'],'token':raw['collection_token']}
            with lock(self.ack_queue.with_suffix('.lock')):
                current=json.loads(self.ack_queue.read_text()) if self.ack_queue.exists() else []
                atomic_json(self.ack_queue,[v for v in current if v['job_id']!=e.logical_job]+[entry])
        return receipt

    def run(self):
        if (self.e.directory / 'pool-collected.json').exists():
            # prepare still verifies the caller's frozen request before returning an old result.
            data = (self.e.directory / 'pool-collected.json').read_bytes()
            proof = json.loads((self.e.directory / 'pool-collected-proof.json').read_text())
            if proof['sha256'] != hashlib.sha256(data).hexdigest() or proof['logical_job'] != self.e.logical_job:
                raise Failure('pool_local_result_corrupted')
            cached = json.loads(data)
            marker = json.loads(self.host_marker.read_text())
            saved = marker['request']
            if self.e.config.get('distributed') and getattr(self.e,'launch_mode',None)!='command':
                from .remote_distributed_worker import validate_options
                expected=validate_options(self.e.config['distributed'],self.options.get('gpu_count',1))
                if (saved.get('distributed')!=expected or (not saved.get('source_manifest_id') and saved.get('distributed_runner_sha256')!=
                        hashlib.sha256(Path(__file__).with_name('remote_distributed_worker.py').read_bytes()).hexdigest())):
                    raise Failure('recovery_input_mismatch')
            elif saved.get('distributed'):
                raise Failure('recovery_input_mismatch')
            expected_mode='command' if getattr(self.e,'launch_mode',None)=='command' else None
            if saved.get('launch_mode')!=expected_mode:
                raise Failure('recovery_input_mismatch')
            files = self.e.declared_input_files if self.e.declared_input_files is not None else tree(self.command.workspace)
            if saved.get('source_manifest_id'):
                from . import remote_cache as content
                same_source=saved['source_manifest_id']==content.validate_manifest(content.manifest(files))
                if saved.get('cache_options')!=self.e.config.get('cache',{}):raise Failure('recovery_input_mismatch')
            else:
                source = io.BytesIO()
                with zipfile.ZipFile(source, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
                    for name, data in sorted(files.items()):
                        bundle.writestr(zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0)), data,
                                        compress_type=zipfile.ZIP_DEFLATED)
                same_source=saved['archive_sha256']==hashlib.sha256(source.getvalue()).hexdigest()
            if (not same_source
                    or saved['argv'] != list(self.command.command)
                    or saved['outputs'] != list(self.command.outputs)
                    or saved['timeout_s'] != self.command.timeout
                    or {'PYTHONPYCACHEPREFIX': '.aka-cache/python', **self.e.config.get('environment', {}),
                        **dict(self.command.environment)} != saved['environment']
                    or saved['gpu_ids'] != self.options['gpu_ids']
                    or saved['gpu_count'] != self.options.get('gpu_count', 1)
                    or saved['replay_safe'] != self.options.get('replay_safe', False)
                    or saved['monitor_interval_s'] != self.options.get('monitor_interval_s', .1)
                    or saved['max_contamination_retries'] != self.options.get('max_contamination_retries', 1000)):
                raise Failure('recovery_input_mismatch')
            if (saved.get('topology','any')!=self.options.get('topology','any') or
                    saved.get('reservation_after_s',30)!=self.options.get('reservation_after_s',30)):
                raise Failure('recovery_input_mismatch')
            return self.e._result(cached['status'], cached['payload'])
        if self.prepare() is None:
            return self.e._result('cancelled', {'cancelled': True, 'exit_code': None, 'reason': 'Stopped before queue admission'})
        observed = time.monotonic()
        waiting_seconds = 0
        previous_state = None
        cancelled = False
        while True:
            state = self.refresh()['jobs'].get(self.e.logical_job)
            if state is None:
                state = self.rpc('status', {'job_id': self.e.logical_job})
            if state['state'] in {'completed', 'failed', 'cancelled', 'blocked'}:
                return self.collect(state)
            control = self.e.control()
            if not cancelled and (control == 'cancel' or (control == 'pause' and state['state'] == 'queued')):
                self.rpc('cancel', {'job_id': self.e.logical_job, 'queued_only': control == 'pause'})
                cancelled = True
            now = time.monotonic()
            if previous_state == 'queued':
                waiting_seconds += now - observed
            observed, previous_state = now, state['state']
            atomic_json(self.e.directory / 'pool-progress.json', {'observed_at': time.time(),
                        'queue_wait_seconds': waiting_seconds, **state})
            if waiting_seconds > self.options.get('max_wait_s', 86400):
                # Preserve the original ID and remote ownership; never mark the kernel failed.
                raise Failure('pool_wait_budget_exhausted', transient=True)
            time.sleep(min(self.options.get('poll_interval_s', 10), 1))
