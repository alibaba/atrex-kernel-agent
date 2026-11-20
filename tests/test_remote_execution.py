"""Remote transport lifecycle, exact package transfer and existing-task routing."""
import base64
import hashlib
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from aka.light.execution.contract import CommandSpec, ExecutionUnavailable
from aka.contracts.task import EvaluationPlan
from aka.light.execution.remote import Executor, validate_config
from aka.light.execution.remote_resilience import Failure
from aka.light.execution import remote_worker
from aka.light.execution.task_transport import execute


class RemoteExecutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.workspace = self.root/'source'
        self.workspace.mkdir()
        (self.workspace/'data.bin').write_bytes(bytes(range(256)))
        self.target = {'backend':'remote','config':{'pod':'pod-a','container':'worker0',
            'resilience':{'attempts':1,'admission_interval_s':0,'base_delay_s':0}}}

    def executor(self, **overrides):
        values = dict(target=self.target, directory=self.root/'execution', execution_id='eval-1',
                      previous_job='', on_job=lambda job: None, control=lambda: 'run')
        values.update(overrides)
        return Executor(**values)

    def test_config_requires_pod_and_validates_options(self):
        for config in ({}, {'pod':'p','remote_root':'relative'}, {'pod':'p','poll_interval_s':0},
                       {'pod':'p','environment':{'X':1}}, {'pod':'p','unknown':True}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_config(config)
        validate_config({'pod':'p','environment':{'CUDA_VISIBLE_DEVICES':'2'}})

    def test_cli_arguments_do_not_expand_shell_and_reject_ambiguous_transport(self):
        executor = self.executor()
        raw = SimpleNamespace(returncode=0, stdout='AKA_REMOTE={"state":"running"}\n', stderr='')
        with patch('aka.light.execution.remote.subprocess.run', return_value=raw) as run:
            self.assertEqual(executor._exec(['python3','-c','print("$(touch bad)")']), {'state':'running'})
            argv = run.call_args.args[0]
            self.assertEqual(argv[-1], 'print("$(touch bad)")')
            self.assertEqual(argv[argv.index('--container')+1], 'worker0')
        with patch('aka.light.execution.remote.subprocess.run', side_effect=subprocess.TimeoutExpired('remote',60)) as run:
            with self.assertRaises(ExecutionUnavailable):executor._exec(['python3','probe.py'])
            self.assertEqual(run.call_count,1)

    def test_source_environment_outputs_and_success_receipt(self):
        executor = self.executor()
        parts = {}; jobs=[]; captured={}
        executor.on_job = jobs.append
        result = {'state':'finished','result':{'exit_code':0,'timed_out':False,'cancelled':False}}
        def rpc(operation,*args):
            if operation=='part':parts[int(args[0])]=base64.b64decode(args[1]);return {}
            if operation=='start':return result
            raise AssertionError(operation)
        def initialize(spec):captured.update(spec)
        def download(name,**kwargs):
            return {'stdout.log':b'OK','stderr.log':b'','workspace/report.bin':b'\x00\xff'}[name]
        with patch.object(executor,'_initialize',side_effect=initialize), \
             patch.object(executor,'_rpc',side_effect=rpc), patch.object(executor,'_download',side_effect=download):
            value = executor.run(CommandSpec(self.workspace, ('python3','probe.py'),
                environment={'X':'a b'},outputs=('report.bin',)))
        archive=b''.join(parts[i] for i in sorted(parts))
        with zipfile.ZipFile(io.BytesIO(archive)) as z:self.assertEqual(z.read('data.bin'),bytes(range(256)))
        self.assertEqual(captured['environment']['X'],'a b')
        self.assertEqual(value.exit_code,0)
        receipt=value.details['receipt']
        self.assertEqual(receipt['status'],'completed')
        self.assertEqual(base64.b64decode(receipt['payload']['artifacts']['report.bin']['base64']),b'\x00\xff')
        self.assertEqual(jobs,[executor.job])

    def test_recovery_does_not_upload_or_launch_finished_or_unknown_jobs(self):
        original=self.executor()
        executor=self.executor(previous_job=original.job)
        result={'state':'finished','result':{'exit_code':0,'timed_out':False,'cancelled':False}}
        with patch.object(executor,'_rpc',return_value=result) as rpc, \
             patch.object(executor,'_initialize') as initialize, \
             patch.object(executor,'_download',return_value=b''):
            executor.run(CommandSpec(self.workspace,('true',)))
            rpc.assert_called_once_with('status');initialize.assert_not_called()
        (executor.directory/'remote-collected.json').unlink()
        for state in ('unknown','lost'):
            with patch.object(executor,'_rpc',return_value={'state':state}) as rpc:
                with self.assertRaises(ExecutionUnavailable):executor.run(CommandSpec(self.workspace,('true',)))
                rpc.assert_called_once_with('status')
        with self.assertRaises(ValueError):self.executor(previous_job='other-job')

    def test_cancel_waits_for_remote_confirmation(self):
        executor=self.executor(previous_job=self.executor().job,control=lambda:'cancel')
        final={'state':'finished','result':{'exit_code':-15,'timed_out':False,'cancelled':True}}
        with patch.object(executor,'_rpc',side_effect=[{'state':'running'},{'state':'running'},final]) as rpc, \
             patch.object(executor,'_download',return_value=b''), \
             patch('aka.light.execution.remote.time.sleep'):
            value=executor.run(CommandSpec(self.workspace,('sleep','60')))
        self.assertEqual([c.args[0] for c in rpc.call_args_list],['status','cancel','status'])
        self.assertEqual(value.details['receipt']['status'],'cancelled')

    def test_partial_upload_recovery_keeps_inputs_and_launches_once(self):
        original=self.executor()
        initialized=[]
        command=CommandSpec(self.workspace,('true',))
        with patch.object(original,'_initialize',side_effect=lambda spec:initialized.append(spec)), \
             patch.object(original,'_rpc',side_effect=ExecutionUnavailable('Upload response lost')):
            with self.assertRaises(ExecutionUnavailable):original.run(command)
        resumed=self.executor(previous_job=original.job)
        final={'state':'finished','result':{'exit_code':0,'timed_out':False,'cancelled':False}}
        def rpc(operation,*args):
            if operation=='status':return {'state':'prepared'}
            if operation=='read':return {'base64':base64.b64encode(json.dumps(initialized[0]).encode()).decode()}
            if operation=='start':return final
            if operation=='part':return {}
            raise AssertionError(operation)
        with patch.object(resumed,'_initialize',side_effect=lambda spec:initialized.append(spec)), \
             patch.object(resumed,'_rpc',side_effect=rpc) as calls, \
             patch.object(resumed,'_download',return_value=b''):
            resumed.run(command)
        self.assertEqual(initialized[0],initialized[1])
        self.assertEqual(sum(c.args[0]=='start' for c in calls.call_args_list),1)

    def test_new_upload_chunks_stay_below_linux_argument_limit(self):
        (self.workspace/'large.bin').write_bytes(os.urandom(400000))
        executor = self.executor()
        chunks=[];specs=[]
        def rpc(operation, *args):
            if operation == 'part':chunks.append(args[1]);return {}
            if operation == 'start':return {'state':'finished','result':{'exit_code':0,'timed_out':False,'cancelled':False}}
            raise AssertionError(operation)
        with patch.object(executor,'_initialize',side_effect=lambda s:specs.append(s)), \
             patch.object(executor,'_rpc',side_effect=rpc),patch.object(executor,'_download',return_value=b''):
            executor.run(CommandSpec(self.workspace,('true',)))
        self.assertEqual(specs[0]['chunk_bytes'],90000)
        self.assertLessEqual(max(map(len,chunks)),120000)
        self.assertEqual(len(chunks),5)

    def test_declared_inputs_include_explicit_cache_paths_without_unlisted_files(self):
        executor=self.executor();parts={}
        executor.declared_input_files={'kernel.cu':b'source','.aka-output/report.bin':b'\x00\xff',
                                       'scratch/__pycache__/probe.pyc':b'explicit-input'}
        (self.workspace/'unlisted.py').write_bytes(b'must-not-transfer')
        def rpc(operation,*args):
            if operation=='part':parts[args[0]]=base64.b64decode(args[1]);return {}
            if operation=='start':return {'state':'finished','result':{'exit_code':0,'timed_out':False,'cancelled':False}}
            raise AssertionError(operation)
        with patch.object(executor,'_initialize',return_value={}),patch.object(executor,'_rpc',side_effect=rpc),patch.object(executor,'_download',return_value=b''):
            executor.run(CommandSpec(self.workspace,('true',)))
        with zipfile.ZipFile(io.BytesIO(b''.join(parts[i] for i in sorted(parts)))) as archive:
            self.assertEqual(set(archive.namelist()),set(executor.declared_input_files))
            for name,data in executor.declared_input_files.items():self.assertEqual(archive.read(name),data)

    def test_legacy_partial_upload_preserves_original_spec(self):
        executor = self.executor(previous_job=self.executor().job)
        captured=[]
        # Reconstruct the legacy spec using the same deterministic source archive.
        legacy=self.executor();legacy.chunk_bytes=36000;legacy.legacy_spec=True
        with patch.object(legacy,'_initialize',side_effect=lambda s:captured.append(s)), \
             patch.object(legacy,'_rpc',side_effect=ExecutionUnavailable('stop after initialization')):
            with self.assertRaises(ExecutionUnavailable):legacy.run(CommandSpec(self.workspace,('true',)))
        original=captured[0]
        def rpc(operation,*args):
            if operation=='status':return {'state':'prepared'}
            if operation=='read':return {'base64':base64.b64encode(json.dumps(original).encode()).decode()}
            if operation=='part':return {}
            if operation=='start':return {'state':'finished','result':{'exit_code':0,'timed_out':False,'cancelled':False}}
            raise AssertionError(operation)
        with patch.object(executor,'_initialize',side_effect=lambda s:captured.append(s)), \
             patch.object(executor,'_rpc',side_effect=rpc),patch.object(executor,'_download',return_value=b''):
            executor.run(CommandSpec(self.workspace,('true',)))
        self.assertEqual(captured,[original,original])
        self.assertNotIn('chunk_bytes',original)

    def test_recovery_skips_only_checksum_verified_source_parts(self):
        executor=self.executor()
        (self.workspace/'large.bin').write_bytes(os.urandom(400000))
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            for name in ['data.bin','large.bin']:
                archive.writestr(zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0)),(self.workspace/name).read_bytes(),compress_type=zipfile.ZIP_DEFLATED)
        first=stream.getvalue()[:90000];uploaded=[]
        def rpc(operation,*args):
            if operation=='part':uploaded.append(args[0]);return {}
            if operation=='start':return {'state':'finished','result':{'exit_code':0,'timed_out':False,'cancelled':False}}
            raise AssertionError(operation)
        prepared={'prepared':True,'part_sha256':{'0':hashlib.sha256(first).hexdigest(),'1':'wrong'}}
        with patch.object(executor,'_initialize',return_value=prepared),patch.object(executor,'_rpc',side_effect=rpc),patch.object(executor,'_download',return_value=b''):
            executor.run(CommandSpec(self.workspace,('true',)))
        self.assertNotIn(0,uploaded)
        self.assertIn(1,uploaded)

    def test_timeout_is_not_a_successful_command(self):
        executor=self.executor(previous_job=self.executor().job)
        result={'state':'finished','result':{'exit_code':-15,'timed_out':True,'cancelled':False}}
        with patch.object(executor,'_rpc',return_value=result),patch.object(executor,'_download',return_value=b''):
            value=executor.run(CommandSpec(self.workspace,('sleep','60')))
        self.assertEqual(value.details['receipt']['status'],'failed')

    def restart_executor(self, *, legacy=False):
        self.target['config']['uid']='uid-a'
        executor=self.executor(previous_job=self.executor().job)
        metadata=dict(job_id=executor.job,pod='pod-a',remote_directory=executor.root)
        if not legacy:
            metadata.update(prepared_at=900000000,container_identity=dict(pod_uid='uid-a',container_id='old'))
        marker=executor.directory/'remote-job.json'
        marker.write_text(json.dumps(metadata))
        os.utime(marker,(900000000,900000000))
        current=dict(pod_uid='uid-a',container='worker0',container_id='new',
                     started_at='2000-01-01T00:00:00Z',restart_count=1)
        return executor,current

    def test_verified_restart_and_missing_directory_are_terminal_failure_without_relaunch(self):
        executor,current=self.restart_executor()
        with patch.object(executor,'_rpc',side_effect=ExecutionUnavailable('missing runner')), \
             patch.object(executor,'_pod_state',return_value=current), \
             patch.object(executor,'_exec',return_value={'directory_exists':False}), \
             patch.object(executor,'_initialize') as initialize:
            value=executor.run(CommandSpec(self.workspace,('true',)))
        receipt=value.details['receipt']
        self.assertEqual(receipt['status'],'failed')
        self.assertTrue(receipt['payload']['execution_lost'])
        self.assertIsNone(receipt['payload']['exit_code'])
        self.assertEqual(receipt['job_id'],executor.job)
        self.assertTrue((executor.directory/'remote-restart-loss.json').is_file())
        initialize.assert_not_called()

    def test_legacy_marker_mtime_supports_verified_restart_loss(self):
        executor,current=self.restart_executor(legacy=True)
        with patch.object(executor,'_rpc',side_effect=ExecutionUnavailable('missing runner')), \
             patch.object(executor,'_pod_state',return_value=current), \
             patch.object(executor,'_exec',return_value={'directory_exists':False}):
            value=executor.run(CommandSpec(self.workspace,('true',)))
        self.assertTrue(value.details['receipt']['payload']['execution_lost'])

    def test_restart_verification_does_not_close_an_existing_job(self):
        executor,current=self.restart_executor()
        with patch.object(executor,'_rpc',side_effect=ExecutionUnavailable('ambiguous')), \
             patch.object(executor,'_pod_state',return_value=current), \
             patch.object(executor,'_exec',return_value={'directory_exists':True}):
            with self.assertRaises(ExecutionUnavailable):executor.run(CommandSpec(self.workspace,('true',)))

    def test_restart_verification_requires_uid_time_and_changed_container(self):
        for change in [dict(pod_uid='other'),dict(container_id='old'),dict(started_at='1990-01-01T00:00:00Z')]:
            with self.subTest(change=change):
                executor,current=self.restart_executor();current.update(change)
                with patch.object(executor,'_rpc',side_effect=ExecutionUnavailable('ambiguous')), \
                     patch.object(executor,'_pod_state',return_value=current), \
                     patch.object(executor,'_exec') as probe:
                    with self.assertRaises(ExecutionUnavailable):executor.run(CommandSpec(self.workspace,('true',)))
                    probe.assert_not_called()

    def test_restart_verification_auth_failure_stays_unknown(self):
        executor,current=self.restart_executor()
        with patch.object(executor,'_rpc',side_effect=ExecutionUnavailable('original')), \
             patch.object(executor,'_pod_state',side_effect=ExecutionUnavailable('auth')), \
             patch.object(executor,'_initialize') as initialize:
            with self.assertRaisesRegex(ExecutionUnavailable,'original'):
                executor.run(CommandSpec(self.workspace,('true',)))
        initialize.assert_not_called()

    def test_restart_verification_requires_saved_host_marker(self):
        executor,current=self.restart_executor();(executor.directory/'remote-job.json').unlink()
        with patch.object(executor,'_rpc',side_effect=ExecutionUnavailable('ambiguous')), \
             patch.object(executor,'_pod_state') as state:
            with self.assertRaises(ExecutionUnavailable):executor.run(CommandSpec(self.workspace,('true',)))
            state.assert_not_called()

    def test_pod_identity_extracts_only_restart_fields_and_rejects_changed_uid(self):
        self.target['config']['uid']='uid-a';executor=self.executor()
        pod={'metadata':{'uid':'uid-a'},'spec':{'env':'not-for-exports'},'status':{'containerStatuses':[
            {'name':'worker0','containerID':'cid','restartCount':2,'state':{'running':{'startedAt':'2000-01-01T00:00:00Z'}}}]}}
        response=SimpleNamespace(returncode=0,stdout=json.dumps(pod),stderr='')
        with patch('aka.light.execution.remote.subprocess.run',return_value=response):
            state=executor._pod_state()
            self.assertEqual(state['container_id'],'cid');self.assertNotIn('spec',state)
            pod['metadata']['uid']='uid-b';response.stdout=json.dumps(pod)
            with self.assertRaises(ExecutionUnavailable):executor._pod_state()

    def test_unrecorded_absent_submission_is_failed_without_start(self):
        self.target['config']['uid']='uid-a';executor=self.executor();saved=[];executor.on_job=saved.append
        with patch.object(executor,'_pod_state',return_value={'pod_uid':'uid-a'}), \
             patch.object(executor,'_exec',return_value={'directory_exists':False}), \
             patch.object(executor,'_initialize') as initialize,patch.object(executor,'_rpc') as rpc:
            value=executor.reconcile_unrecorded()
        self.assertEqual(value.details['receipt']['status'],'failed')
        self.assertTrue(value.details['receipt']['payload']['submission_not_admitted'])
        self.assertIsNone(value.exit_code)
        self.assertEqual(saved,[executor.job]);initialize.assert_not_called();rpc.assert_not_called()

    def test_unrecorded_existing_job_is_recovered_without_relaunch(self):
        self.target['config']['uid']='uid-a';executor=self.executor();saved=[];executor.on_job=saved.append
        with patch.object(executor,'_pod_state',return_value={'pod_uid':'uid-a'}), \
             patch.object(executor,'_exec',return_value={'directory_exists':True}):
            self.assertIsNone(executor.reconcile_unrecorded())
        final={'state':'finished','result':{'exit_code':0,'timed_out':False,'cancelled':False}}
        with patch.object(executor,'_rpc',return_value=final) as rpc, \
             patch.object(executor,'_download',return_value=b''),patch.object(executor,'_initialize') as initialize:
            value=executor.run(CommandSpec(self.workspace,('true',)))
        self.assertEqual(value.details['receipt']['status'],'completed')
        self.assertEqual(saved,[executor.job]);rpc.assert_called_once_with('status');initialize.assert_not_called()

    def test_unrecorded_recovery_requires_verified_uid_and_probe(self):
        executor=self.executor()
        with self.assertRaises(ExecutionUnavailable):executor.reconcile_unrecorded()
        self.target['config']['uid']='uid-a';executor=self.executor()
        for state,probe in [({'pod_uid':'other'},{'directory_exists':False}),({'pod_uid':'uid-a'},{})]:
            with patch.object(executor,'_pod_state',return_value=state),patch.object(executor,'_exec',return_value=probe),patch.object(executor,'on_job') as saved:
                with self.assertRaises(ExecutionUnavailable):executor.reconcile_unrecorded()
                saved.assert_not_called()

    def test_command_transport_uses_inspected_unrecorded_failure_without_run(self):
        receipt={'status':'failed','payload':{'exit_code':None,'submission_not_admitted':True},'job_id':'remote-1'}
        self.target['config']['uid']='uid-a'
        with patch('aka.light.execution.remote.Executor') as factory, \
             patch('aka.light.execution.contract.validate_executor'):
            factory.return_value.reconcile_unrecorded.return_value=SimpleNamespace(details={'receipt':receipt})
            value=execute(EvaluationPlan('command',{'files':{'x':'text'},'argv':['python3','x']}),
                self.target,self.root/'transport','eval-x','',lambda j:None,lambda:'run',recover_unrecorded=True)
            self.assertEqual(asdict(value),receipt)
            factory.return_value.run.assert_not_called()

    def test_unrecorded_recovery_is_not_enabled_for_other_backends(self):
        with self.assertRaises(ValueError):
            execute(EvaluationPlan('command',{'files':{},'argv':['true']}),{'backend':'local'},
                self.root/'bad','e','',lambda j:None,lambda:'run',recover_unrecorded=True)

    def artifact_executor(self, raw):
        executor = self.executor()
        remote = self.root/'remote-files'
        (remote/'workspace').mkdir(parents=True)
        (remote/'workspace/report.bin').write_bytes(raw)
        executor.root = str(remote)
        def execute(argv, **kwargs):
            result = subprocess.run([sys.executable, *argv[1:]], capture_output=True, text=True, check=True)
            return json.loads(result.stdout.removeprefix('AKA_REMOTE='))
        def rpc(operation, *args):
            return remote_worker.dispatch(remote, operation, [str(a) for a in args])
        return executor, execute, rpc

    def test_remote_artifact_ranges_check_identity_and_reject_changed_bytes(self):
        raw=bytes(range(256))*1000
        executor,execute,_=self.artifact_executor(raw)
        expected={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
        with patch.object(executor,'_exec',side_effect=execute):
            first=executor._artifact_range('workspace/report.bin',expected,17,500000)
            self.assertEqual(base64.b64decode(first['base64']),raw[17:90017])
            second=executor._artifact_range('workspace/report.bin',expected,90017,31)
            self.assertEqual(base64.b64decode(second['base64']),raw[90017:90048])
            path=Path(executor.root)/'workspace/report.bin';before=path.stat()
            path.write_bytes(b'x'*len(raw));os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
            with self.assertRaises(Failure):executor._artifact_range('workspace/report.bin',expected,0,32)

    def test_remote_artifact_range_refuses_paths_outside_original_attempt(self):
        raw=b'original output';executor,execute,_=self.artifact_executor(raw)
        expected={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
        (Path(executor.root).parent/'foreign.bin').write_bytes(raw)
        with patch.object(executor,'_exec',side_effect=execute):
            with self.assertRaises(Failure):executor._artifact_range('../foreign.bin',expected,0,8)
        with self.assertRaises(ValueError):executor._artifact_range('workspace/report.bin',expected,-1,8)

    def test_large_compressed_artifact_roundtrip_preserves_exact_bytes(self):
        raw = bytes(range(256))*12000 + os.urandom(600000)
        executor, execute, rpc = self.artifact_executor(raw)
        with patch.dict(os.environ, {'AKA_REMOTE_COMPRESS_ARTIFACTS':'1'}), \
             patch.object(executor, '_exec', side_effect=execute), \
             patch.object(executor, '_rpc', side_effect=rpc) as calls:
            self.assertEqual(executor._download('workspace/report.bin'), raw)
        reads = [c for c in calls.call_args_list if c.args[0] == 'read']
        self.assertLess(len(reads), 3)
        self.assertTrue(all(c.args[1].startswith('.transfer-cache/') for c in reads))

    def test_recovery_reuses_only_hash_verified_local_artifact(self):
        raw = b'profile\x00\xff'*100000
        executor, execute, rpc = self.artifact_executor(raw)
        cached = executor.directory/'artifacts/report.bin'
        cached.parent.mkdir(parents=True)
        cached.write_bytes(raw)
        with patch.dict(os.environ, {'AKA_REMOTE_COMPRESS_ARTIFACTS':'1'}), \
             patch.object(executor, '_exec', side_effect=execute), \
             patch.object(executor, '_rpc', side_effect=rpc) as calls:
            self.assertEqual(executor._download('workspace/report.bin'), raw)
            self.assertEqual([c for c in calls.call_args_list if c.args[0] == 'read'], [])
            cached.write_bytes(b'x'*len(raw))
            self.assertEqual(executor._download('workspace/report.bin'), raw)
            self.assertTrue(any(c.args[0] == 'read' for c in calls.call_args_list))

    def test_corrupt_compressed_artifact_is_not_accepted(self):
        executor, execute, rpc = self.artifact_executor(b'a'*1000000)
        def corrupt(operation, *args):
            value = rpc(operation, *args)
            if operation == 'read':
                data = bytearray(base64.b64decode(value['base64']))
                data[0] ^= 255
                value['base64'] = base64.b64encode(data).decode()
            return value
        with patch.dict(os.environ, {'AKA_REMOTE_COMPRESS_ARTIFACTS':'1'}), \
             patch.object(executor, '_exec', side_effect=execute), \
             patch.object(executor, '_rpc', side_effect=corrupt):
            with self.assertRaises(ExecutionUnavailable):
                executor._download('workspace/report.bin')

    def test_incompressible_artifact_uses_original_bytes(self):
        raw = os.urandom(800000)
        executor, execute, rpc = self.artifact_executor(raw)
        with patch.dict(os.environ, {'AKA_REMOTE_COMPRESS_ARTIFACTS':'1'}), \
             patch.object(executor, '_exec', side_effect=execute), \
             patch.object(executor, '_rpc', side_effect=rpc) as calls:
            self.assertEqual(executor._download('workspace/report.bin'), raw)
        self.assertTrue(all(c.args[1] == 'workspace/report.bin' for c in calls.call_args_list if c.args[0] == 'read'))

    def test_common_command_transport_selects_builtin_remote_without_gateway(self):
        receipt={'status':'completed','payload':{'stdout':'YES','stderr':'','exit_code':0},'job_id':'remote-1'}
        with patch('aka.light.execution.remote.Executor') as factory, \
             patch('aka.light.execution.task_transport.gateway_client') as gateway, \
             patch('aka.light.execution.contract.validate_executor'):
            factory.return_value.run.return_value=SimpleNamespace(details={'receipt':receipt})
            value=execute(EvaluationPlan('command',{'files':{'x':'text'},'argv':['python3','x']}),
                self.target,self.root/'transport','eval-x','',lambda j:None,lambda:'run')
            self.assertEqual(asdict(value),receipt);gateway.assert_not_called()
            command=factory.return_value.run.call_args.args[0]
            self.assertEqual(command.command,('python3','x'))
            self.assertEqual(command.workspace,self.root/'transport/input')

    def test_remote_worker_rejects_output_escape_and_does_not_restart_claimed_job(self):
        root=self.root/'remote';root.mkdir();(root/'claimed').touch()
        self.assertEqual(remote_worker.dispatch(root,'start',[]),{'state':'unknown'})
        with self.assertRaises(ValueError):remote_worker.dispatch(root,'read',['../outside','0','10'])
        remote_worker.atomic(root/'result.json',{'exit_code':7})
        self.assertEqual(remote_worker.dispatch(root,'status',[])['result']['exit_code'],7)
