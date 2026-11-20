"""Fault injection for hidden infra recovery and explicit replay boundaries."""
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from aka.light.execution.contract import CommandSpec, ExecutionUnavailable
from aka.light.execution.remote import Executor, validate_config, SAFE_START
from aka.light.execution.remote_resilience import Failure, Policy, admit, classify
from aka.light.execution import remote_transfer
import test_remote_execution as base_tests


class ResilienceTests(unittest.TestCase):
    setUp = base_tests.RemoteExecutionTests.setUp
    executor = base_tests.RemoteExecutionTests.executor
    restart_executor = base_tests.RemoteExecutionTests.restart_executor

    def response(self, value=None, *, code=0, stderr=''):
        return subprocess.CompletedProcess([], code,
            'AKA_REMOTE='+json.dumps(value or {'state':'finished'})+'\n', stderr)

    def resilient(self, *, attempts=4, **kwargs):
        self.target['config']['resilience'].update(attempts=attempts)
        return self.executor(**kwargs)

    def test_rpc_timeout_is_retried_without_exposing_failure(self):
        executor=self.resilient()
        with patch('aka.light.execution.remote_resilience.subprocess.run',
                   side_effect=[subprocess.TimeoutExpired('remote',60),self.response({'state':'running'})]) as run:
            self.assertEqual(executor._rpc('status'),{'state':'running'})
            self.assertEqual(run.call_count,2)
            self.assertEqual(run.call_args_list[0].args[0],run.call_args_list[1].args[0])

    def test_raw_ambiguous_command_is_never_retried(self):
        executor=self.resilient()
        with patch('aka.light.execution.remote_resilience.subprocess.run',
                   side_effect=subprocess.TimeoutExpired('remote',60)) as run:
            with self.assertRaises(Failure):executor._exec(['python3','unknown-side-effects.py'])
            self.assertEqual(run.call_count,1)

    def test_admission_rejection_is_retried_even_for_raw_command(self):
        executor=self.resilient()
        with patch('aka.light.execution.remote_resilience.subprocess.run',side_effect=[
            self.response(code=1,stderr='server returned 429: quota exhausted'),self.response()]) as run:
            executor._exec(['python3','unknown-side-effects.py'])
            self.assertEqual(run.call_count,2)

    def test_rolling_quota_wait_is_honored(self):
        value=classify('server returned 429: 请至少等待 17 秒后重试')
        self.assertEqual(value.retry_after,17)
        self.assertEqual(classify('server returned 429: Retry-After: 9').retry_after,9)

    def test_auth_failure_is_classified_and_not_retried(self):
        executor=self.resilient()
        secret='server returned 401: https://sso.invalid/login?login_code=secret'
        with patch('aka.light.execution.remote_resilience.subprocess.run',return_value=self.response(code=1,stderr=secret)) as run:
            with self.assertRaises(Failure) as caught:executor._rpc('status')
            self.assertEqual(caught.exception.code,'authentication_required')
            self.assertEqual(run.call_count,1)
        log=(executor.directory/'infra-events.jsonl').read_text()
        self.assertNotIn('login_code',log);self.assertNotIn('https://',log)
        self.assertNotIn('secret',str(caught.exception))

    def test_transient_server_failure_and_bad_response_recover(self):
        executor=self.resilient()
        with patch('aka.light.execution.remote_resilience.subprocess.run',side_effect=[
            self.response(code=1,stderr='server returned 503'),
            subprocess.CompletedProcess([],0,'gateway garbled response',''),
            self.response({'exists':True})]) as run:
            self.assertEqual(executor._rpc('stat','a'),{'exists':True})
            self.assertEqual(run.call_count,3)

    def test_retry_exhaustion_is_not_a_successful_receipt(self):
        executor=self.resilient(attempts=3)
        with patch('aka.light.execution.remote_resilience.subprocess.run',return_value=self.response(code=1,stderr='connection reset')) as run:
            with self.assertRaises(Failure) as caught:executor._rpc('status')
            self.assertTrue(caught.exception.exhausted);self.assertEqual(run.call_count,3)

    def test_protocol_or_input_mismatch_is_not_retried(self):
        executor=self.resilient()
        with patch('aka.light.execution.remote_resilience.subprocess.run',
                   return_value=self.response({'error':'Existing execution has different inputs'})) as run:
            with self.assertRaises(Failure) as caught:executor._rpc('status')
            self.assertEqual(caught.exception.code,'configuration_or_protocol_rejected')
            self.assertEqual(run.call_count,1)

    def test_file_not_found_is_sent_to_restart_reconciliation(self):
        executor=self.resilient()
        with patch('aka.light.execution.remote_resilience.subprocess.run',return_value=
                   self.response(code=1,stderr='python: runner.py: No such file or directory')) as run:
            with self.assertRaises(Failure) as caught:executor._rpc('status')
            self.assertEqual(caught.exception.code,'remote_state_missing');self.assertEqual(run.call_count,1)

    def test_deadline_bounds_the_cli_timeout(self):
        self.target['config']['resilience']['budget_s']=.05
        executor=self.resilient(attempts=1)
        with patch('aka.light.execution.remote_resilience.subprocess.run',return_value=self.response()) as run:
            executor._rpc('status');self.assertLessEqual(run.call_args.kwargs['timeout'],.05)

    def test_configuration_rejects_unbounded_or_ambiguous_policy(self):
        for policy in [{'unknown':1},{'attempts':0},{'attempts':True},{'budget_s':0},
                       {'budget_s':float('inf')},{'replay_on_restart':1},{'admission_interval_s':-1}]:
            with self.subTest(policy=policy),self.assertRaises(ValueError):Policy.parse(policy)
        with self.assertRaises(ValueError):validate_config({'pod':'p','resilience':{'replay_on_restart':True}})

    def test_cross_process_admission_is_shared(self):
        state=self.root/'shared-admission'
        env={**os.environ,'AKA_REMOTE_ADMISSION_DIR':str(state)}
        env.pop('AKA_REMOTE_ADMISSION_LOCK',None)
        code='from aka.light.execution.remote_resilience import admit; print(admit({},.05))'
        processes=[subprocess.Popen([sys.executable,'-c',code],env=env,stdout=subprocess.PIPE,text=True) for _ in range(4)]
        values=sorted(float(p.communicate(timeout=10)[0]) for p in processes)
        self.assertTrue(all(b-a>=.045 for a,b in zip(values,values[1:])),values)

    def test_partial_download_resumes_at_saved_chunk_after_new_executor(self):
        raw=os.urandom(1200000);digest=hashlib.sha256(raw).hexdigest();reads=[]
        def first(offset,size):
            reads.append(offset)
            if offset==500000:raise Failure('transport_timeout',transient=True)
            return raw[offset:offset+size]
        with self.assertRaises(Failure):remote_transfer.read(self.root,'file',len(raw),digest,first)
        self.assertEqual(reads,[0,500000]);reads.clear()
        def resumed(offset,size):reads.append(offset);return raw[offset:offset+size]
        self.assertEqual(remote_transfer.read(self.root,'file',len(raw),digest,resumed),raw)
        self.assertEqual(reads,[500000,1000000])

    def test_corrupt_partial_is_rejected_and_clean_retry_succeeds(self):
        raw=b'x'*900000;digest=hashlib.sha256(raw).hexdigest()
        partial=remote_transfer.cache_file(self.root,digest).with_suffix('.part');partial.write_bytes(b'bad'+b'x'*499997)
        with self.assertRaises(Failure):
            remote_transfer.read(self.root,'file',len(raw),digest,lambda o,n:raw[o:o+n])
        self.assertFalse(partial.exists())
        self.assertEqual(remote_transfer.read(self.root,'file',len(raw),digest,lambda o,n:raw[o:o+n]),raw)

    def test_changed_digest_never_reuses_an_old_prefix(self):
        old=b'a'*700000;new=b'b'*700000
        partial=remote_transfer.cache_file(self.root,hashlib.sha256(old).hexdigest()).with_suffix('.part')
        partial.write_bytes(old[:500000]);offsets=[]
        def fetch(o,n):offsets.append(o);return new[o:o+n]
        self.assertEqual(remote_transfer.read(self.root,'file',len(new),hashlib.sha256(new).hexdigest(),fetch),new)
        self.assertEqual(offsets[0],0)

    def test_short_chunk_is_not_checkpointed(self):
        raw=b'q'*600000;digest=hashlib.sha256(raw).hexdigest()
        with self.assertRaises(Failure):remote_transfer.read(self.root,'f',len(raw),digest,lambda o,n:b'q'*(n-1))
        self.assertEqual(remote_transfer.cache_file(self.root,digest).with_suffix('.part').stat().st_size,0)

    def test_process_death_short_write_is_rolled_back_to_chunk_boundary(self):
        raw=b'm'*1100000;digest=hashlib.sha256(raw).hexdigest();offsets=[]
        remote_transfer.cache_file(self.root,digest).with_suffix('.part').write_bytes(raw[:600000])
        def fetch(o,n):offsets.append(o);return raw[o:o+n]
        self.assertEqual(remote_transfer.read(self.root,'f',len(raw),digest,fetch),raw)
        self.assertEqual(offsets,[500000,1000000])

    def test_complete_download_works_without_network(self):
        raw=b'complete';digest=remote_transfer.store(self.root,raw)
        with patch.object(remote_transfer,'file_hash',wraps=remote_transfer.file_hash):
            self.assertEqual(remote_transfer.read(self.root,'f',len(raw),digest,lambda *a:self.fail('network called')),raw)

    def test_cache_symlink_escape_is_rejected(self):
        raw=b'outside';outside=self.root/'outside';outside.write_bytes(raw);digest=hashlib.sha256(raw).hexdigest()
        remote_transfer.cache_file(self.root,digest).symlink_to(outside)
        with self.assertRaises(Failure):remote_transfer.read(self.root,'f',len(raw),digest,lambda o,n:raw)

    def test_empty_file_roundtrip(self):
        digest=hashlib.sha256(b'').hexdigest()
        self.assertEqual(remote_transfer.read(self.root,'f',0,digest,lambda *a:self.fail()),b'')

    def test_completed_receipt_is_recovered_locally_after_remote_disappears(self):
        executor=self.executor(previous_job=self.executor().logical_job)
        done={'state':'finished','result':{'exit_code':0,'timed_out':False,'cancelled':False}}
        command=CommandSpec(self.workspace,('true',),outputs=('result.bin',))
        def download(name,**kwargs):return {'stdout.log':b'GPU completed','stderr.log':b'',
                                          'workspace/result.bin':b'\x00\xff'}[name]
        with patch.object(executor,'_rpc',return_value=done),patch.object(executor,'_download',side_effect=download):
            result=executor.run(command)
        recovered=self.executor(previous_job=executor.logical_job)
        with patch.object(recovered,'_rpc',side_effect=AssertionError('remote was called')):
            self.assertEqual(recovered.run(command).details['receipt'],result.details['receipt'])

    def test_corrupt_completed_receipt_file_is_not_trusted(self):
        executor=self.executor(previous_job=self.executor().logical_job);command=CommandSpec(self.workspace,('true',))
        done={'state':'finished','result':{'exit_code':0,'timed_out':False,'cancelled':False}}
        with patch.object(executor,'_rpc',return_value=done),patch.object(executor,'_download',return_value=b'abc'):
            executor.run(command)
        for f in (executor.directory/'transfer-cache').glob('*.complete'):f.write_bytes(b'bad')
        recovered=self.executor(previous_job=executor.logical_job)
        with patch.object(recovered,'_rpc',side_effect=ExecutionUnavailable('remote unavailable')):
            with self.assertRaises(ExecutionUnavailable):recovered.run(command)

    def test_lost_start_response_recovers_by_status_without_second_launch(self):
        executor=self.resilient();starts=[];done=False
        def rpc(op,*args):
            nonlocal done
            if op=='part':return {}
            if op=='start':
                starts.append(op);done=True;raise Failure('transport_timeout',transient=True)
            if op=='status' and done:return {'state':'finished','result':{'exit_code':0,'timed_out':False,'cancelled':False}}
            self.fail(op)
        with patch.object(executor,'_initialize',return_value={}),patch.object(executor,'_rpc',side_effect=rpc),patch.object(executor,'_download',return_value=b''):
            value=executor.run(CommandSpec(self.workspace,('true',)))
        self.assertEqual(value.details['receipt']['status'],'completed');self.assertEqual(len(starts),1)

    def test_kernel_failure_is_returned_without_infra_replay(self):
        executor=self.resilient(previous_job=self.executor().logical_job)
        result={'state':'finished','result':{'exit_code':7,'timed_out':False,'cancelled':False}}
        with patch.object(executor,'_rpc',return_value=result) as rpc,patch.object(executor,'_download',return_value=b''):
            value=executor.run(CommandSpec(self.workspace,('false',)))
        self.assertEqual(value.exit_code,7);self.assertEqual(value.details['receipt']['status'],'failed')
        self.assertEqual(rpc.call_count,1);self.assertEqual(executor.attempt,0)

    def test_verified_restart_replay_requires_explicit_permission(self):
        executor,current=self.restart_executor();command=CommandSpec(self.workspace,('true',))
        with patch.object(executor,'_rpc',side_effect=ExecutionUnavailable('missing runner')),patch.object(executor,'_pod_state',return_value=current),patch.object(executor,'_exec',return_value={'directory_exists':False}),patch.object(executor,'_initialize') as launch:
            value=executor.run(command)
        self.assertTrue(value.details['receipt']['payload']['execution_lost']);launch.assert_not_called()

    def test_explicit_replay_keeps_logical_id_and_separates_physical_attempt(self):
        self.target['config']['uid']='uid-a';self.target['config']['resilience'].update(replay_on_restart=True,max_replays=1)
        executor=self.executor(previous_job=self.executor().logical_job);attempts=[]
        def run(command):
            attempts.append(executor.job)
            if executor.attempt==0:raise Failure('remote_state_missing')
            executor._save_attempt(admitted=True)
            return executor._result('completed',{'exit_code':0,'stdout':'new measurement','stderr':''})
        lost=executor._result('failed',{'exit_code':None,'execution_lost':True,
            'restart_evidence':{'previous_container':{'image_id':'sha256:same'},
                                'current_container':{'image_id':'sha256:same'}}})
        with patch.object(executor,'_run',side_effect=run),patch.object(executor,'_confirmed_restart_loss',return_value=lost):
            value=executor.run(CommandSpec(self.workspace,('true',)))
        self.assertEqual(attempts,[executor.logical_job,executor.logical_job+'-r1'])
        self.assertEqual(value.details['receipt']['job_id'],executor.logical_job)
        self.assertEqual(value.details['infra_attempt'],1)
        resumed=self.executor(previous_job=executor.logical_job)
        resumed._prepare_attempt(CommandSpec(self.workspace,('true',)))
        self.assertEqual(resumed.job,executor.logical_job+'-r1')
        self.assertEqual(resumed.previous_job,executor.logical_job)

    def test_retry_never_replays_unknown_launch_without_restart_evidence(self):
        self.target['config']['uid']='uid-a';self.target['config']['resilience'].update(replay_on_restart=True)
        executor=self.executor(previous_job=self.executor().logical_job)
        with patch.object(executor,'_rpc',return_value={'state':'unknown'}),patch.object(executor,'_confirmed_restart_loss',return_value=None),patch.object(executor,'_initialize') as launch:
            with self.assertRaises(ExecutionUnavailable):executor.run(CommandSpec(self.workspace,('true',)))
        launch.assert_not_called();self.assertEqual(executor.attempt,0)

    def test_replay_rejects_changed_source_command_or_environment(self):
        executor=self.executor();executor._prepare_attempt(CommandSpec(self.workspace,('true',)));executor._save_attempt(admitted=True)
        for command in [CommandSpec(self.workspace,('false',)),CommandSpec(self.workspace,('true',),environment={'A':'changed'})]:
            with self.assertRaises(Failure):self.executor(previous_job=executor.logical_job)._prepare_attempt(command)
        (self.workspace/'data.bin').write_bytes(b'changed')
        with self.assertRaises(Failure):self.executor(previous_job=executor.logical_job)._prepare_attempt(CommandSpec(self.workspace,('true',)))

    def test_local_disk_full_never_becomes_gpu_failure_or_success(self):
        executor=self.executor()
        with patch.object(executor,'_run',side_effect=OSError(28,'disk full')):
            with self.assertRaises(Failure) as caught:executor.run(CommandSpec(self.workspace,('true',)))
        self.assertEqual(caught.exception.code,'local_storage_full')

    def test_missing_declared_artifact_is_not_success(self):
        executor=self.executor(previous_job=self.executor().logical_job)
        result={'state':'finished','result':{'exit_code':0,'timed_out':False,'cancelled':False}}
        def download(name,**kwargs):return None if name.startswith('workspace/') else b''
        with patch.object(executor,'_rpc',return_value=result),patch.object(executor,'_download',side_effect=download):
            value=executor.run(CommandSpec(self.workspace,('true',),outputs=('missing.bin',)))
        self.assertEqual(value.details['receipt']['status'],'failed')
        self.assertEqual(value.details['receipt']['payload']['missing_outputs'],['missing.bin'])

    def test_direct_source_is_frozen_before_first_remote_action(self):
        executor=self.executor()
        executor._prepare_attempt(CommandSpec(self.workspace,('true',)))
        old=executor.declared_input_files['data.bin']
        (self.workspace/'data.bin').write_bytes(b'edited while transport waits')
        self.assertEqual(executor.declared_input_files['data.bin'],old)

    def test_auth_or_local_storage_failure_does_not_trigger_restart_probe(self):
        executor=self.executor()
        with patch.object(executor,'_run',side_effect=Failure('authentication_required')),patch.object(executor,'_confirmed_restart_loss') as probe:
            with self.assertRaises(Failure):executor.run(CommandSpec(self.workspace,('true',)))
        probe.assert_not_called()

    def test_changed_or_missing_image_blocks_automatic_replay(self):
        self.target['config']['uid']='uid-a';self.target['config']['resilience']['replay_on_restart']=True
        for previous,current in [('sha256:old','sha256:new'),('', 'sha256:new')]:
            executor=self.executor(previous_job=self.executor().logical_job)
            lost=executor._result('failed',{'exit_code':None,'execution_lost':True,
                'restart_evidence':{'previous_container':{'image_id':previous},
                                    'current_container':{'image_id':current}}})
            with patch.object(executor,'_run',side_effect=Failure('remote_state_missing')) as run,patch.object(executor,'_confirmed_restart_loss',return_value=lost):
                value=executor.run(CommandSpec(self.workspace,('true',)))
            self.assertEqual(run.call_count,1);self.assertEqual(executor.attempt,0)
            self.assertIn('image_identity',value.details['receipt']['payload']['replay_blocked'])

    def test_start_rpc_uses_serialized_materialization_wrapper(self):
        executor=self.executor()
        with patch.object(executor,'_exec',return_value={'state':'running'}) as execute:
            executor._rpc('start')
        self.assertEqual(execute.call_args.args[0][1:3],['-c',SAFE_START])
        self.assertTrue(execute.call_args.kwargs['retry_safe'])

    def test_concurrent_start_attempts_do_not_materialize_or_launch_twice(self):
        remote=self.root/'start-race';remote.mkdir()
        (remote/'runner.py').write_text('''import json,time
def dispatch(root,operation,args):
 result=root/'result.json'
 if result.exists():return json.loads(result.read_text())
 count=root/'count';n=int(count.read_text())+1 if count.exists() else 1;count.write_text(str(n))
 time.sleep(.1)
 value={'state':'finished','result':{'exit_code':0}};result.write_text(json.dumps(value));return value
''')
        processes=[subprocess.Popen([sys.executable,'-c',SAFE_START,str(remote)],stdout=subprocess.PIPE,text=True) for _ in range(4)]
        values=[json.loads(p.communicate(timeout=10)[0].removeprefix('AKA_REMOTE=')) for p in processes]
        self.assertEqual((remote/'count').read_text(),'1')
        self.assertTrue(all(x['state']=='finished' for x in values))
