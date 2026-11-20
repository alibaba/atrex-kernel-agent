"""Distributed lifecycle tests with real processes and a CPU launcher surrogate."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from aka.light.execution import remote_distributed_worker as group
from aka.light.execution import remote_pool_worker as worker
from aka.light.execution.remote import validate_config
from aka.light.execution.remote_pool import validate
import test_remote_pool as pool_fixture
import test_task_contracts as task_fixture

LAUNCHER = '''import json,os,subprocess,sys,signal,time
c=json.load(open(sys.argv[2]));children=[]
def stop(*args):
 for p in children:
  if p.poll() is None:p.terminate()
 for p in children:
  try:p.wait(timeout=2)
  except subprocess.TimeoutExpired:p.kill();p.wait()
 if args:raise SystemExit(143)
signal.signal(signal.SIGTERM,stop)
try:
 for rank in range(c['world_size']):
  env={**os.environ,'LOCAL_RANK':str(rank),'RANK':str(rank),'WORLD_SIZE':str(c['world_size'])}
  children.append(subprocess.Popen([sys.executable,sys.argv[1],'rank',sys.argv[2]],env=env))
 while any(p.poll() is None for p in children):
  if any(p.poll() not in [None,0] for p in children):raise SystemExit(1)
  time.sleep(.01)
 raise SystemExit(0 if all(p.returncode==0 for p in children) else 1)
finally:stop()
'''


class DistributedTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.directory=self.root/'distributed';self.directory.mkdir()
        self.module=Path(group.__file__).resolve()
        self.config={**group.validate_options({'world_size':2,'bind_device':False,
            'phase_timeouts_s':{x:3 for x in group.PHASES}},2),
            'directory':str(self.directory),'nonce':'unique-attempt','fence':'epoch:1',
            'devices':[{'index':i,'uuid':'GPU-'+str(i)} for i in range(2)],
            'argv':[sys.executable,'case.py']}
        self.launcher=self.root/'launcher.py';self.launcher.write_text(LAUNCHER)

    def run_group(self, code, *, timeout=None, files=None):
        if timeout:self.config['phase_timeouts_s'].update(timeout)
        for rank in range(2):
            root=self.directory/('rank-'+str(rank))/'workspace';root.mkdir(parents=True)
            (root/'case.py').write_text(code)
            for name,data in (files or {}).items():
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(data)
        group.atomic(self.directory/'config.json',self.config)
        started=time.monotonic()
        status=group.launch(self.root,command_factory=lambda p,n:[sys.executable,str(self.launcher),str(self.module),str(p)])
        result=group.read(self.directory/'group.json');result['duration']=time.monotonic()-started
        return status,result

    def test_all_ranks_pass_and_iteration_max_is_not_sum_or_mean(self):
        status,result=self.run_group('''import os
from pathlib import Path
from aka_distributed import stage,report
rank=int(os.environ['RANK']);stage('compile');stage('initialize');stage('correctness');stage('measure')
Path('own-output').write_text(str(rank))
report(True,[10+rank,20+rank]);stage('complete')
''')
        self.assertEqual(status,0);self.assertTrue(result['complete'])
        self.assertEqual(result['max_rank_samples_us'],[11,21])
        self.assertEqual(len({x['gpu_uuid'] for x in result['ranks']}),2)
        for rank in range(2):self.assertEqual((self.directory/f'rank-{rank}/workspace/own-output').read_text(),str(rank))

    def test_relative_pythonpath_is_resolved_in_each_rank_workspace(self):
        with patch.dict(os.environ,{'PYTHONPATH':'candidate'}):
            status,result=self.run_group('''import os
from pathlib import Path
from aka_distributed import stage,report
import own_module
assert own_module.VALUE==int(os.environ['RANK'])
assert Path(own_module.__file__).resolve().parent==(Path.cwd()/'candidate').resolve()
stage('measure');report(True,[1]);stage('complete')
''',files={'candidate/own_module.py':'import os\nVALUE=int(os.environ["RANK"])\n'})
        self.assertEqual(status,0);self.assertTrue(result['complete'])

    def test_rank_error_aborts_peers_and_is_not_infra_retryable(self):
        status,result=self.run_group('''import os,time
from aka_distributed import stage
stage('compile')
if os.environ['RANK']=='1':raise RuntimeError('candidate compilation failed')
time.sleep(20)
''')
        self.assertEqual(status,1);self.assertEqual(result['failure_kind'],'rank_error')
        self.assertFalse(result['retryable']);self.assertLess(result['duration'],8)
        self.assertIn('candidate compilation failed',(self.directory/'rank-1/stderr.log').read_text())

    def test_lost_rank_without_exit_receipt_is_bounded_infra_retryable(self):
        status,result=self.run_group('''import os,signal,time
from aka_distributed import stage
stage('initialize')
if os.environ['RANK']=='1':os.kill(os.getpid(),signal.SIGKILL)
time.sleep(20)
''')
        self.assertEqual(status,1);self.assertEqual(result['failure_kind'],'rank_lost')
        self.assertTrue(result['retryable']);self.assertFalse(result['complete'])

    def test_collective_phase_hang_is_timed_out_for_the_group(self):
        status,result=self.run_group('''import time
from aka_distributed import stage
stage('compile');stage('initialize');time.sleep(20)
''',timeout={'initialize':.2})
        self.assertEqual(status,1);self.assertEqual(result['phase'],'initialize')
        self.assertEqual(result['failure_kind'],'phase_timeout');self.assertTrue(result['retryable'])
        self.assertLess(result['duration'],8)

    def test_one_rank_correctness_failure_cannot_publish_rank_zero_success(self):
        status,result=self.run_group('''import os,time
from aka_distributed import stage,report
stage('correctness');report(os.environ['RANK']=='0',[1]);time.sleep(20)
''')
        self.assertEqual(status,1);self.assertEqual(result['failure_kind'],'correctness_failure')
        self.assertFalse(result['retryable'])

    def test_mismatched_phases_are_rejected(self):
        status,result=self.run_group('''import os
from aka_distributed import stage
stage('compile' if os.environ['RANK']=='0' else 'initialize')
''')
        self.assertEqual(status,1);self.assertEqual(result['failure_kind'],'protocol_error')

    def test_missing_rank_report_is_not_success(self):
        status,result=self.run_group('''from aka_distributed import stage
stage('complete')
''')
        self.assertEqual(status,1);self.assertFalse(result['complete'])

    def test_rank_sample_counts_must_match(self):
        status,result=self.run_group('''import os
from aka_distributed import stage,report
stage('measure');report(True,[1]*(1+int(os.environ['RANK'])));stage('complete')
''')
        self.assertEqual(status,1);self.assertFalse(result['complete'])

    def test_stale_attempt_receipt_is_rejected(self):
        for rank in range(2):
            root=self.directory/f'rank-{rank}';root.mkdir()
            group.atomic(root/'exit.json',dict(rank=rank,nonce='unique-attempt',exit_code=0))
            group.atomic(root/'result.json',dict(rank=rank,nonce='stale',passed=True,samples_us=[]))
        with self.assertRaises(ValueError):group.summarize(self.config)

    def test_world_size_and_deadlines_are_validated(self):
        for options in [{'world_size':3},{'world_size':True},{'world_size':2,'launcher':'external'},
                        {'world_size':2,'max_retries':-1},{'world_size':2,'phase_timeouts_s':{'initialize':float('inf')}}]:
            with self.subTest(options=options),self.assertRaises(ValueError):group.validate_options(options,2)
        with self.assertRaises(ValueError):validate_config({'pod':'p','distributed':{'world_size':2}})
        with self.assertRaises(ValueError):validate_config({'pod':'p','uid':'u','scheduling':{'gpu_ids':[0,1],'gpu_count':2},
            'distributed':{'world_size':2},'environment':{'WORLD_SIZE':'3'}})

    def test_graphs_are_released_before_communicator_destroy(self):
        events=[]
        fake=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:events.append('sync')))
        distributed=SimpleNamespace(is_initialized=lambda:True,barrier=lambda:events.append('barrier'),
                                    destroy_process_group=lambda:events.append('destroy'))
        fake.distributed=distributed
        with patch.dict(sys.modules,{'torch':fake,'torch.distributed':distributed}):
            group.finalize([SimpleNamespace(reset=lambda:events.append('graph-reset'))])
        self.assertLess(events.index('graph-reset'),events.index('destroy'))


class TopologyProbeTests(unittest.TestCase):
    def setUp(self):
        self.calls=[]
        self.probe=object.__new__(worker.NvmlProbe)
        self.probe.nvml=SimpleNamespace(NVML_P2P_CAPS_INDEX_READ=(0,),
            NVML_P2P_CAPS_INDEX_WRITE=1,NVML_P2P_CAPS_INDEX_NVLINK=2,NVML_P2P_STATUS_OK=0,
            nvmlDeviceGetHandleByIndex=lambda i:i,
            nvmlDeviceGetP2PStatus=lambda a,b,cap:self.calls.append((a,b,cap)) or 0)
        self.devices=[dict(index=i,uuid='GPU-'+str(i)) for i in range(2)]

    def test_shipped_pynvml_tuple_enum_still_checks_both_p2p_directions(self):
        self.assertTrue(self.probe.compatible(self.devices,'p2p'))
        self.assertEqual(self.calls,[(0,1,0),(0,1,1),(1,0,0),(1,0,1)])

    def test_driver_rejection_is_not_replaced_by_a_topology_assumption(self):
        self.probe.nvml.nvmlDeviceGetP2PStatus=lambda a,b,cap:5 if cap==1 else 0
        self.assertFalse(self.probe.compatible(self.devices,'p2p'))
        self.assertTrue(self.probe.compatible(self.devices,'nvlink'))

    def test_unrecognized_binding_enum_fails_closed(self):
        self.probe.nvml.NVML_P2P_CAPS_INDEX_READ=(0,1)
        with self.assertRaises(worker.ProbeUnavailable):self.probe.compatible(self.devices,'p2p')


class DistributedBrokerTests(unittest.TestCase):
    setUp=pool_fixture.PoolTests.setUp
    close_all=pool_fixture.PoolTests.close_all
    request=pool_fixture.PoolTests.request
    launch=pool_fixture.PoolTests.launch
    finish=pool_fixture.PoolTests.finish
    def distributed_spec(self, number=1, **changes):
        spec=self.request(number,**changes)
        source=Path(group.__file__).read_bytes();sha=hashlib.sha256(source).hexdigest()
        runner=self.broker.root/('support-'+sha)/'aka_distributed.py';runner.parent.mkdir(exist_ok=True);runner.write_bytes(source)
        spec.update(distributed=group.validate_options({'world_size':spec['gpu_count']},spec['gpu_count']),
                    distributed_runner=str(runner),distributed_runner_sha256=sha)
        return spec

    def test_distributed_retry_budget_is_separate_and_bounded(self):
        spec=self.distributed_spec(gpu_count=2);spec['distributed']['max_retries']=1
        self.broker.enqueue(spec);self.broker.tick()
        for expected in ['running','blocked']:
            result=self.finish(spec['job_id']);root=Path(self.broker.status(spec['job_id'])['attempt_root'])
            result.update(exit_code=1,valid_measurement=False,retryable_group_failure=True)
            worker.atomic(root/'attempt-result.json',result);self.broker.tick()
            self.assertEqual(self.broker.status(spec['job_id'])['state'],expected)
        self.assertEqual(self.broker.status(spec['job_id'])['distributed_retries'],2)
        self.assertEqual(self.broker.status(spec['job_id'])['contamination_retries'],0)

    def test_request_larger_than_physical_pod_is_blocked_before_launch(self):
        self.probe.rows=self.probe.rows[:1]
        spec=self.request(gpu_count=2);self.broker.enqueue(spec);self.broker.tick()
        self.assertFalse(self.launches)
        self.assertEqual(self.broker.status(spec['job_id'])['result']['reason'],'insufficient_pod_gpu_capacity')

    def test_aged_gang_reserves_devices_even_when_an_external_task_is_present(self):
        gang=self.request(1,gpu_count=2,reservation_after_s=0)
        single=self.request(2)
        self.probe.rows[1]['pids']=[777]
        self.broker.enqueue(gang);self.broker.enqueue(single);self.broker.tick()
        self.assertFalse(self.launches)
        self.probe.rows[1]['pids']=[];self.broker.tick()
        self.assertEqual(self.launches[0][0],gang['job_id'])


    def test_topology_selection_tries_a_compatible_group(self):
        self.probe.rows=[dict(index=i,uuid='GPU-'+str(i),pids=[]) for i in range(4)]
        self.probe.compatible=lambda ds,t:{x['index'] for x in ds}=={2,3}
        spec=self.request(gpu_ids=[0,1,2,3],gpu_count=2,topology='nvlink')
        self.broker.enqueue(spec);self.broker.tick()
        self.assertEqual(self.launches[0][2],[2,3])

    def test_aged_group_from_served_task_precedes_new_single_gpu_tasks(self):
        gang=self.request(1,task='previously-served',gpu_count=2,reservation_after_s=0)
        single=self.request(2,task='brand-new')
        self.broker.enqueue(gang);self.broker.enqueue(single)
        with self.broker.connect() as db:db.execute('UPDATE tasks SET last_served=? WHERE id=?',(time.time(),'previously-served'))
        self.broker.tick()
        self.assertEqual(self.launches[0][0],gang['job_id'])


class DistributedTaskRoutingTests(unittest.TestCase):
    setUp=task_fixture.TaskContractsTest.setUp
    write_spec=task_fixture.TaskContractsTest.write_spec

    def distributed_task(self):
        self.spec['target']={'backend':'remote','config':{'pod':'p','uid':'u',
            'scheduling':{'gpu_ids':[0,1],'gpu_count':2},'distributed':{'world_size':2}}}
        self.write_spec()
        return self.service.submit(self.spec_path,idempotency_key='distributed',start=False)['id']

    def test_research_runs_once_by_default_and_rank_launch_is_explicit(self):
        task=self.distributed_task()
        receipt={'id':'eval','job_id':'job','candidate_id':'source','receipt':{'status':'completed'}}
        with patch.object(self.service,'measure',return_value=receipt) as measure:
            self.service.command(task,argv=['compiler','source.cu'])
            self.assertEqual(measure.call_args.kwargs['_research']['launch_mode'],'command')
            self.service.command(task,argv=['python3','distributed_case.py'],launch_mode='ranks')
            self.assertEqual(measure.call_args.kwargs['_research']['launch_mode'],'ranks')

    def test_rank_launch_cannot_change_a_non_distributed_task(self):
        with self.assertRaises(ValueError):self.service.command(self.task,argv=['command'],launch_mode='ranks')

    def test_cli_explicit_mode_is_sent_to_the_bound_task(self):
        from aka.light.execution.command_cli import main
        from contextlib import redirect_stdout
        import io
        with patch('aka.light.execution.command_cli.invoke',return_value={'receipt':{'status':'completed'}}) as invoke, redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--launch-mode','ranks','--','python3','case.py']),0)
        self.assertEqual(invoke.call_args.args[1]['launch_mode'],'ranks')

if __name__=='__main__':unittest.main()
