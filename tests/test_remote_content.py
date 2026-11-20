"""Incremental transfer, independent snapshots and frozen plan regression tests."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import types
from types import SimpleNamespace
from contextlib import redirect_stdout
import threading
import unittest
from unittest.mock import patch
import zipfile

from aka.light.execution import remote_cache as cache
from aka.light.execution import remote_pool_worker as worker
from aka.light.execution.remote_pool import PoolDriver
from aka.light.execution.remote_resilience import Failure
import test_remote_pool_host as host
import test_task_contracts as task


class ContentTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.store=cache.Store(self.root/'store')

    def upload(self,files):
        value=cache.manifest(files);missing=self.store.prepare(value)['missing']
        for sha,raw,_ in cache.packs(files,missing):
            count=(len(raw)+89999)//90000
            for i,offset in enumerate(range(0,len(raw),90000)):
                self.store.put_pack(sha,i,count,len(raw),raw[offset:offset+90000])
        return self.store.commit(value)

    def test_warm_inputs_and_duplicate_paths_need_no_new_objects(self):
        files={'a.hpp':b'header','copy/a.hpp':b'header','main.py':b'print(1)'}
        mid=self.upload(files)
        self.assertEqual(self.store.prepare(cache.manifest(files))['missing'],{})
        self.assertEqual(len(list((self.store.root/'blobs').glob('*/*'))),2)
        self.assertEqual(self.store.files(mid),files)

    def test_one_changed_file_renames_and_deletions_are_exact(self):
        self.upload({'header.hpp':b'unchanged','old.py':b'old','remove.me':b'gone'})
        files={'header.hpp':b'unchanged','renamed.py':b'old','new.py':b'new'}
        request=self.store.prepare(cache.manifest(files))
        self.assertEqual(request['missing'],{cache.digest(b'new'):3})
        mid=self.upload(files);self.store.materialize(mid,self.root/'attempt')
        self.assertFalse((self.root/'attempt/remove.me').exists());self.assertFalse((self.root/'attempt/old.py').exists())
        self.assertEqual({str(p.relative_to(self.root/'attempt')) for p in (self.root/'attempt').rglob('*') if p.is_file()},set(files))

    def test_attempt_writes_do_not_mutate_cached_sources_or_other_ranks(self):
        mid=self.upload({'x':b'original'})
        for i in range(2):self.store.materialize(mid,self.root/str(i))
        (self.root/'0/x').write_bytes(b'changed')
        self.assertEqual((self.root/'1/x').read_bytes(),b'original')
        self.assertEqual(self.store.get(cache.digest(b'original')),b'original')

    def test_corrupt_or_evicted_blob_is_requested_again(self):
        files={'x':b'original'};self.upload(files);sha=cache.digest(b'original')
        self.store.path(sha).write_bytes(b'corrupted')
        self.assertIn(sha,self.store.prepare(cache.manifest(files))['missing'])
        self.upload(files);self.store.path(sha).unlink()
        self.assertIn(sha,self.store.prepare(cache.manifest(files))['missing'])
        self.upload(files);self.assertEqual(self.store.get(sha),b'original')

    def test_partial_batch_resumes_and_checksum_rejection_never_commits(self):
        files={'random.bin':os.urandom(200000)};value=cache.manifest(files)
        sha,raw,_=cache.packs(files,self.store.prepare(value)['missing'])[0]
        count=(len(raw)+89999)//90000
        self.store.put_pack(sha,0,count,len(raw),raw[:90000])
        response=self.store.prepare(value,[sha])
        self.assertEqual(response['packs'][sha],{'0':cache.digest(raw[:90000])})
        with self.assertRaises(ValueError):self.store.commit(value)
        for i in range(1,count):self.store.put_pack(sha,i,count,len(raw),raw[i*90000:(i+1)*90000])
        self.store.commit(value)
        bad=cache.digest(b'bad')
        with self.assertRaises(ValueError):self.store.put_pack(bad,0,1,3,b'bad')

    def test_invalid_paths_sizes_and_duplicate_batch_entries_are_rejected(self):
        for name in ['../x','/x','a/../x','.git/config','a//x','x\\y']:
            with self.subTest(name=name),self.assertRaises(ValueError):cache.manifest({name:b'x'})
        with self.assertRaises(ValueError):cache.manifest({'a':b'x','a/b':b'y'})
        value=cache.manifest({'x':b'x'});value['files']['x']['bytes']=cache.MAX_BYTES+1
        with self.assertRaises(ValueError):cache.validate_manifest(value)

    def test_concurrent_identical_uploads_and_gc_live_pins(self):
        files={'x':b'same bytes'};errors=[]
        def upload():
            try:self.upload(files)
            except BaseException as e:errors.append(e)
        threads=[threading.Thread(target=upload) for _ in range(4)]
        for t in threads:t.start()
        for t in threads:t.join(5)
        self.assertFalse(errors);self.assertTrue(all(not t.is_alive() for t in threads))
        mid=cache.validate_manifest(cache.manifest(files))
        # Expire the staging lease, then distinguish protected and unprotected snapshots.
        p=self.store.root/'staging'/(mid+'.json');v=json.loads(p.read_text());v['expires']=0;p.write_text(json.dumps(v))
        self.store.gc(0,keep=[mid],ttl_s=0);self.assertEqual(self.store.files(mid),files)
        self.store.gc(0,ttl_s=0);self.assertFalse(self.store.has(cache.digest(b'same bytes')))

    def test_empty_input_manifest_is_valid(self):
        mid=self.store.put_files({});self.store.materialize(mid,self.root/'empty')
        self.assertEqual(list((self.root/'empty').iterdir()),[])

    def test_verified_legacy_archive_migration_and_invalid_archive_are_safe(self):
        source=self.root/'legacy';source.mkdir();raw=io.BytesIO()
        with zipfile.ZipFile(raw,'w') as archive:archive.writestr('old-prefix/header.hpp',b'old header')
        data=raw.getvalue();(source/(cache.digest(data)+'.zip')).write_bytes(data)
        (source/('0'*64+'.zip')).write_bytes(data)
        missing={cache.digest(b'old header'):10}
        self.assertEqual(self.store.seed_archives(source,missing),1)
        self.assertEqual(self.store.get(cache.digest(b'old header')),b'old header')


class ContentDriverTests(host.DriverTests):
    def driver(self):
        driver=super().driver();old=driver.rpc.side_effect
        driver.e.config['cache']={'mode':'cas'};driver.pool_root=str(self.broker.root)
        for filename,remote in [('remote_cache.py','aka_cache.py'),('remote_worker.py','runner.py')]:
            source=Path(worker.__file__).with_name(filename).read_bytes();sha=cache.digest(source)
            p=self.broker.root/('support-'+sha)/remote;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(source)
        def rpc(operation,value):
            if operation.startswith('content_'):
                sha=cache.digest(Path(cache.__file__).read_bytes())
                with patch.object(worker,'ensure_server'):
                    result=worker.rpc(self.broker.root,operation,{**value,'engine':{'path':str(self.broker.root/('support-'+sha)/'aka_cache.py'),'sha256':sha}})
                if operation=='content_enqueue' and self.options.pop('lost_enqueue',False):raise Failure('transport_timeout',transient=True)
                return result
            if operation=='collect':return self.broker.collect(value['job_id'],value.get('known_artifacts',()))
            return old(operation,value)
        driver.rpc.side_effect=rpc
        return driver

    def new_execution(self,driver,number):
        directory=self.root/('host-job-'+str(number));directory.mkdir()
        e=driver.e;e.directory=directory;e.logical_job=e.job='remote-'+format(number,'032x')
        return PoolDriver(e,driver.command)

    def test_transfer_stats_and_compact_manifest_resume(self):
        driver=self.driver();driver.run()
        first=json.loads((driver.e.directory/'content-transfer.json').read_text())
        self.assertEqual(first['missing_blobs'],1);self.assertEqual(first['upload_calls'],1)
        # A new logical execution reuses blob contents, not an old execution result.
        e=driver.e;directory=self.root/'second';directory.mkdir();e.directory=directory;e.logical_job=e.job='remote-'+format(2,'032x')
        again=PoolDriver(e,driver.command);again.pool_root=driver.pool_root;again.rpc=driver.rpc
        result=again.run();stats=json.loads((directory/'content-transfer.json').read_text())
        self.assertEqual(result.exit_code,0);self.assertEqual(stats['missing_blobs'],0);self.assertEqual(stats['upload_calls'],0)
        self.assertEqual(len(self.launches),2)

    def finish(self,*args,**kwargs):
        result=super().finish(*args,**kwargs)
        if hasattr(self,'large_artifact'):
            root=Path(self.broker.status(args[0])['attempt_root'])
            (root/'workspace/report.bin').write_bytes(self.large_artifact)
        return result

    def test_large_artifact_is_a_verified_reference(self):
        driver=self.driver();driver.e.config['cache']['artifact_mode']='reference'
        self.large_artifact=b'report data'*20000
        result=driver.run();entry=result.details['receipt']['payload']['artifacts']['report.bin']
        self.assertNotIn('base64',entry);self.assertEqual(entry['artifact_id'],cache.digest(self.large_artifact))
        from aka.light.execution.remote_pool import state_root
        self.assertEqual(cache.Store(state_root()/'artifact-v2').get(entry['artifact_id']),self.large_artifact)

    def test_failed_object_channel_falls_back_to_verified_inline_chunks(self):
        driver=self.driver();driver.e.config['cache'].update(transport='oss',oss_prefix='oss://unit-test-bucket/private')
        (driver.command.workspace/'input.bin').write_bytes(os.urandom(220000))
        with patch('aka.light.execution.remote_oss.Channel') as channel:
            channel.return_value.import_pack.side_effect=Failure('object_transport_unavailable',transient=True)
            result=driver.run()
        stats=json.loads((driver.e.directory/'content-transfer.json').read_text())
        self.assertEqual(result.exit_code,0);self.assertEqual(stats['object_fallbacks'],1)
        self.assertGreater(stats['upload_calls'],1)

    def test_oversized_reference_stays_remote_and_survives_input_pruning(self):
        driver=self.driver();driver.e.config['cache']['artifact_mode']='reference'
        self.large_artifact=b'remote report'*50000
        with patch('aka.light.execution.remote.MAX_ARTIFACT_BYTES',400000),patch.object(
                driver.e,'_download',side_effect=AssertionError('Oversized output was downloaded')):
            result=driver.run()
        entry=result.details['receipt']['payload']['artifacts']['report.bin']
        self.assertTrue(entry['remote_only'])
        self.assertEqual(entry['artifact_id'],cache.digest(self.large_artifact))
        from aka.light.execution.remote_pool import state_root
        self.assertFalse(cache.Store(state_root()/'artifact-v2').has(entry['artifact_id']))
        raw=self.broker.collect(driver.e.logical_job)
        self.broker.acknowledge({'job_id':driver.e.logical_job,'attempt':raw['status']['attempt'],
            'token':raw['collection_token']})
        report=Path(raw['status']['attempt_root'])/'workspace/report.bin'
        self.assertEqual(report.read_bytes(),self.large_artifact)

    def test_recovery_preserves_the_original_content_engine_after_sdk_change(self):
        driver=self.driver();driver.run();saved=json.loads(driver.host_marker.read_text())['request']
        original=Path.read_bytes
        def read(p):
            value=original(p)
            return value+b'\n# a newer SDK helper\n' if p.name=='remote_cache.py' else value
        with patch.object(Path,'read_bytes',read),patch.object(driver,'prepare_content',return_value=True),patch.object(driver,'rpc'):
            spec=driver.prepare()
        self.assertEqual(spec['content_engine'],saved['content_engine'])

    def test_source_transfer_counters_survive_recovery(self):
        driver=self.driver();driver.run();driver.prepare()
        stats=json.loads((driver.e.directory/'content-transfer.json').read_text())
        self.assertEqual(stats['upload_calls'],1);self.assertGreater(stats['uploaded_bytes'],0)
        self.assertEqual(stats['last_prepare_missing_blobs'],0);self.assertEqual(stats['preparations'],2)

    def test_only_verified_collection_can_prune_unchanged_input_copies(self):
        driver=self.driver();driver.run();raw=self.broker.collect(driver.e.logical_job)
        entry={'job_id':driver.e.logical_job,'attempt':raw['status']['attempt'],'token':raw['collection_token']}
        with self.assertRaises(ValueError):self.broker.acknowledge({**entry,'token':'invalid'})
        result=self.broker.acknowledge(entry)
        root=Path(raw['status']['attempt_root'])/'workspace'
        self.assertFalse((root/'input.bin').exists());self.assertEqual((root/'report.bin').read_bytes(),b'verified output')
        self.assertEqual(result['removed_input_copies'],1)
        self.assertEqual(self.broker.acknowledge(entry),result)


class TransportGateTests(unittest.TestCase):
    def test_two_pods_can_progress_but_inflight_and_per_pod_limits_hold(self):
        from aka.light.execution.remote_pool import transport_gate
        with tempfile.TemporaryDirectory() as temporary,patch.dict(os.environ,{'AKA_REMOTE_POOL_STATE_DIR':temporary}):
            release=threading.Event();entered={};active={};peak=[0];mutex=threading.Lock();errors=[]
            def run(name,key):
                try:
                    with transport_gate({'pod':key}):
                        with mutex:
                            active[key]=active.get(key,0)+1
                            if active[key]>1:errors.append('overlapping Pod RPC')
                            peak[0]=max(peak[0],sum(active.values()));entered[name].set()
                        release.wait(3)
                        with mutex:active[key]-=1
                except BaseException as error:errors.append(error)
            threads=[]
            try:
                for name,key in [('a','pod-a'),('b','pod-b')]:
                    entered[name]=threading.Event();t=threading.Thread(target=run,args=(name,key));threads.append(t);t.start()
                    self.assertTrue(entered[name].wait(2))
                for name,key in [('c','pod-c'),('same','pod-a')]:
                    entered[name]=threading.Event();t=threading.Thread(target=run,args=(name,key));threads.append(t);t.start()
                self.assertFalse(entered['c'].wait(.1));self.assertFalse(entered['same'].is_set())
            finally:
                release.set()
                for t in threads:t.join(4)
            self.assertFalse(errors);self.assertEqual(peak[0],2)


class InputPlanTests(unittest.TestCase):
    setUp=task.TaskContractsTest.setUp
    write_spec=task.TaskContractsTest.write_spec

    def test_explicit_only_and_code_policy_keep_required_scratch_inputs(self):
        from aka.light.tasks.content import materialize
        self.spec['target']={'backend':'remote','config':{'pod':'p','uid':'u',
            'scheduling':{'gpu_ids':[0]},'cache':{'mode':'cas','research_inputs':'code'}}}
        self.write_spec();job=self.service.submit(self.spec_path,idempotency_key='input-policy',start=False)['id']
        workspace=self.service.directory(job)/'workspace'
        materialize(workspace,{'scratch/probe.py':b'probe','scratch/research-log.md':b'notes','scratch/ab_ref/ref.cuh':b'reference','old.ncu-rep':b'large report'})
        captured=[]
        from aka.contracts.task import ExecutionReceipt
        def execute(plan,*args,**kwargs):
            captured.append(plan)
            return ExecutionReceipt('completed',{'stdout':'ok','exit_code':0})
        with patch('aka.light.tasks.execution.execute',side_effect=execute):
            self.service.command(job,argv=['python3','scratch/probe.py'])
            self.service.command(job,argv=['python3','probe.py'],source_mode='explicit',files={'probe.py':'only this'})
        store=cache.Store(self.service.root/'content-v2')
        first=store.files(captured[0].request['files_ref']);second=store.files(captured[1].request['files_ref'])
        self.assertNotIn('files',captured[0].request)
        self.assertIn('scratch/ab_ref/ref.cuh',first);self.assertIn('scratch/probe.py',first)
        self.assertNotIn('scratch/research-log.md',first);self.assertNotIn('old.ncu-rep',first)
        self.assertEqual(second,{'probe.py':b'only this'})

    def test_artifact_endpoint_is_task_scoped_and_recovers_evicted_local_bytes(self):
        from aka.light.execution.remote_pool import state_root
        from aka.contracts.task import ExecutionReceipt
        raw=b'owned artifact';sha=cache.digest(raw);foreign=cache.digest(b'foreign artifact')
        with patch.dict(os.environ,{'AKA_REMOTE_POOL_STATE_DIR':str(self.root/'pool')}):
            store=cache.Store(state_root()/'artifact-v2');store.put(raw);store.put(b'foreign artifact')
            receipt=ExecutionReceipt('completed',{'stdout':'ok','exit_code':0,'attempt':1,
                'artifacts':{'report.bin':{'artifact_id':sha,'sha256':sha,'bytes':len(raw)}}})
            with patch('aka.light.tasks.execution.execute',return_value=receipt):self.service.command(self.task,argv=['probe'])
            self.assertEqual(base64.b64decode(self.service.artifact(self.task,sha)['base64']),raw)
            with self.assertRaises(ValueError):self.service.artifact(self.task,foreign)
            # Use a pinned REMOTE target for the read-only original-attempt recovery path.
            target={'backend':'remote','config':{'pod':'p','uid':'u','scheduling':{'gpu_ids':[0]}}}
            task=self.service.store.task(self.task);task['manifest']['target']=target
            store.path(sha).unlink()
            with patch.object(self.service.store,'task',return_value=task),patch('aka.light.execution.remote.Executor._download',return_value=raw) as fetch:
                self.assertEqual(base64.b64decode(self.service.artifact(self.task,sha)['base64']),raw)
                self.assertEqual(fetch.call_count,1)

    def test_remote_only_artifact_endpoint_reads_only_requested_range(self):
        from aka.contracts.task import ExecutionReceipt
        from aka.light.execution.remote_pool import state_root
        sha=cache.digest(b'large remote output');size=65*1024*1024
        receipt=ExecutionReceipt('completed',{'exit_code':0,'pool':{'attempt':3},
            'artifacts':{'large.ncu-rep':{'artifact_id':sha,'sha256':sha,'bytes':size,'remote_only':True}}})
        with patch.dict(os.environ,{'AKA_REMOTE_POOL_STATE_DIR':str(self.root/'pool')}):
            with patch('aka.light.tasks.execution.execute',return_value=receipt):
                self.service.command(self.task,argv=['probe'])
            task=self.service.store.task(self.task)
            task['manifest']['target']={'backend':'remote','config':{'pod':'p','uid':'u','scheduling':{'gpu_ids':[0]}}}
            answer={'artifact_id':sha,'bytes':size,'offset':17,'base64':base64.b64encode(b'chunk').decode()}
            with patch.object(self.service.store,'task',return_value=task),patch(
                    'aka.light.execution.remote.Executor._artifact_range',return_value=answer) as read,patch(
                    'aka.light.execution.remote.Executor._download',side_effect=AssertionError('Full download')):
                self.assertEqual(self.service.artifact(self.task,sha,offset=17,size=23),answer)
                self.assertEqual(read.call_args.args[0],'workspace/large.ncu-rep')
                self.assertEqual(read.call_args.args[2:],(17,23))
            self.assertFalse(cache.Store(state_root()/'artifact-v2').has(sha))

    def test_evicted_local_input_store_recovers_from_frozen_source(self):
        import shutil
        from aka.light.execution.contract import ExecutionUnavailable
        from aka.contracts.task import ExecutionReceipt
        self.spec['target']={'backend':'remote','config':{'pod':'p','uid':'u','scheduling':{'gpu_ids':[0]},'cache':{'mode':'cas'}}}
        self.write_spec();job=self.service.submit(self.spec_path,idempotency_key='local-cache-recovery',start=False)['id']
        with patch('aka.light.tasks.execution.execute',side_effect=ExecutionUnavailable('test interruption')):
            with self.assertRaises(ExecutionUnavailable):self.service.command(job,argv=['python3','compute.py'])
        with self.service.store.connect() as db:record=db.execute('SELECT id,plan FROM evaluations WHERE task_id=?',(job,)).fetchone()
        mid=json.loads(record['plan'])['request']['files_ref'];shutil.rmtree(self.service.root/'content-v2')
        with patch('aka.light.tasks.execution.execute',return_value=ExecutionReceipt('completed',{'stdout':'ok','exit_code':0})):
            self.service.command(job,execution_id=record['id'])
        self.assertIn('compute.py',cache.Store(self.service.root/'content-v2').files(mid))


class BuildCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.source=self.root/'source';self.source.mkdir()
        for name in ['bindings.cpp','source_build.py']:(self.source/name).write_text('original '+name)
        self.calls=[]
        torch=types.ModuleType('torch');utils=types.ModuleType('torch.utils');cpp=types.ModuleType('torch.utils.cpp_extension')
        torch.__version__='fixture';torch.version=SimpleNamespace(cuda='fixture');torch.compiled_with_cxx11_abi=lambda:True
        torch.cuda=SimpleNamespace(get_device_capability=lambda:(10,0),get_device_properties=lambda _:SimpleNamespace(multi_processor_count=148),current_device=lambda:0)
        def build(**kwargs):
            self.calls.append(kwargs);directory=Path(kwargs['build_directory']);library=directory/(kwargs['name']+'.so')
            # Reproduce Torch's in-place link-option extension.
            kwargs['extra_ldflags'].append('-ladded-by-torch');library.write_bytes(b'compiled fixture')
            return SimpleNamespace(__file__=str(library))
        cpp.load=build;cpp.CUDA_HOME='/fixture/cuda';torch.utils=utils;utils.cpp_extension=cpp
        self.addCleanup(patch.stopall)
        patch.dict('sys.modules',{'torch':torch,'torch.utils':utils,'torch.utils.cpp_extension':cpp}).start()
        patch.dict(os.environ,{'AKA_BUILD_CACHE_DIR':str(self.root/'cache')}).start()
        patch.object(cache.subprocess,'run',return_value=SimpleNamespace(stdout='compiler fixture')).start()
        patch('importlib.util.spec_from_file_location',return_value=SimpleNamespace(loader=SimpleNamespace(exec_module=lambda _:None))).start()
        patch('importlib.util.module_from_spec',return_value=SimpleNamespace()).start()

    def load(self,cflags=None):
        with redirect_stdout(io.StringIO()) as output:
            cache.load_torch_extension(self.source,list(self.source.glob('*')),cflags or ['-O3'],['-lcuda'])
        return json.loads(output.getvalue().split('AKA_BUILD_CACHE=')[1])

    def test_mutated_torch_arguments_do_not_change_saved_recipe(self):
        self.assertFalse(self.load()['hit']);self.assertTrue(self.load()['hit']);self.assertEqual(len(self.calls),1)

    def test_flags_source_and_corrupt_binary_invalidate_cache(self):
        first=self.load();self.assertFalse(first['hit']);self.assertFalse(self.load(['-O2'])['hit'])
        (self.source/'bindings.cpp').write_text('changed source');third=self.load();self.assertFalse(third['hit'])
        Path(third['library']).write_bytes(b'corrupt binary');self.assertFalse(self.load()['hit'])
        self.assertEqual(len(self.calls),4)

    def test_build_gc_does_not_remove_a_loaded_entry(self):
        self.load();result=cache.gc_builds(self.root/'cache',0)
        self.assertTrue(result['over_capacity']);self.assertEqual(result['freed_bytes'],0)

    def test_uncommitted_build_with_orphan_torch_baton_is_rebuilt(self):
        from torch.utils.cpp_extension import load as original_load
        first=self.load();directory=Path(first['library']).parent
        (directory/'complete.json').unlink();(directory/'lock').touch()
        (directory/'bindings.o').write_bytes(b'left by a killed compiler')
        def acquire_baton(**kwargs):
            # Torch otherwise waits indefinitely for this O_EXCL lock file.
            self.assertFalse((Path(kwargs['build_directory'])/'lock').exists())
            return original_load(**kwargs)
        with patch('torch.utils.cpp_extension.load',side_effect=acquire_baton):
            result=self.load()
        self.assertFalse(result['hit']);self.assertEqual(result['recovered'],'incomplete')
        self.assertEqual(len(self.calls),2)

    def test_malformed_completion_proof_is_rebuilt(self):
        first=self.load();proof=Path(first['library']).parent/'complete.json'
        for invalid in ['{partial','[1,2]','{}']:
            with self.subTest(proof=invalid):
                proof.write_text(invalid);result=self.load()
                self.assertFalse(result['hit']);self.assertEqual(result['recovered'],'invalid')
        self.assertEqual(len(self.calls),4)

    def test_verified_hit_does_not_clear_a_leftover_torch_baton(self):
        first=self.load();baton=Path(first['library']).parent/'lock';baton.touch()
        result=self.load()
        self.assertTrue(result['hit']);self.assertEqual(result['recovered'],'')
        self.assertTrue(baton.exists());self.assertEqual(len(self.calls),1)


if __name__=='__main__':unittest.main()
