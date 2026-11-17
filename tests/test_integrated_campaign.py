"""Exercise the real composition, Campaign and Runtime with offline Agent/GPU replies."""
import json
import os
import sys
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from aka.bootstrap.host import run_profile
from aka.bootstrap.workflow_cli import profile_path
from aka.bootstrap.launch import create_launch
from aka.contracts.workflow import WorkflowSettings, CampaignRuntimeSettings
from aka.contracts.workflow_launch import encode_launch
from aka.contracts.startup import Invocation
from aka.contracts.execution import JobResult, ExecutionFacts
from aka.agent.tools.client import call_tool
from aka.experiment.measurement_store import MeasurementStore

ROOT=Path(__file__).resolve().parents[1]
REVIEW={'verdict':'supported','primary_issue':'Evidence matches the recorded observation.',
    'findings':[{'claim':'The probe returns its printed result.','judgment':'supported','decision_critical':True,
                 'reason':'The supplied receipt contains that output.','evidence':[]}],
    'scope_safe_interpretation':'Only this probe is established.','next_check':'Measure a candidate.','confidence':'high'}

class ComposedCampaignTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()
        bench=self.root/'bench'
        (bench/'src/atrex_bench').mkdir(parents=True)
        (bench/'src/atrex_bench/__init__.py').touch()
        (bench/'scripts').mkdir()
        (bench/'scripts/run_eval.py').write_text('raise AssertionError("implicit V0 evaluation")\n')
        self.problem=bench/'data/vector-add'
        shutil.copytree(ROOT/'skills/atrex-aka-light/examples/native-atrex-bench-vector-add',self.problem)

    def run_campaign(self,mode,modules,search=None, evaluate=False, abba=False, agate_tools=("env", "evaluate", "command")):
        launch=create_launch(self.problem,workspace_base=self.root/('runs-'+mode+'-'+'-'.join(modules)+'-'+str(search)),
            runtime=CampaignRuntimeSettings(agate_tools=agate_tools),
            settings=WorkflowSettings(platform='L20N', framework='cuda' if mode=='ultra' else '',optimization_mode=mode,max_iters=2))
        calls=[];tool_replies=[];gpu_calls=[]
        def fake_gpu(owner,command):
            gpu_calls.append(command)
            stdout='probe-result\n'
            if command.operation == 'run':
                request=json.loads(command.submission)['request']
                self.assertNotIn('ANTHROPIC_API_KEY',request.get('env_vars',{}))
                self.assertNotIn('HOME',request.get('env_vars',{}))
                self.assertNotIn('PATH',request.get('env_vars',{}))
                self.assertNotIn('AKA_HOST_CANARY',request.get('env_vars',{}))
                self.assertIs(request['lock_clocks'],True)
                self.assertNotIn('lock_clocks',request['options'])
                for values in (request,request['options'],request.get('runner_overrides',{})):
                    self.assertNotIn('gpu_clock_mhz',values)
                    self.assertNotIn('graphics_mhz',values)
                shapes=request['options'].get('shape_ids') or list(request['reference']['shapes'])
                stdout=json.dumps({'job_id':'ev-offline-test','status':'succeeded','result':{
                    'environment': {'gpu_name': 'NVIDIA RTX PRO 5000 72GB Blackwell',
                        'gpu_arch': 'sm_120', 'driver_version': '580.126.09',
                        'runtime_version': '13.0', 'torch_version': '2.9.0+cu130',
                        'clock_locked': False},
                    'passed':{'compile':{'status':'passed'},'correctness':{shape:{'status':'passed'} for shape in shapes}},
                    'correctness':{'shapes':{shape:{} for shape in shapes}},
                    'performance':{'shapes':{shape:{'samples':[{'end_to_end_time_ms':0.005}]} for shape in shapes}}}})
            from aka.execution.gateway.environment import with_job_environment
            return with_job_environment(JobResult(0,stdout=stdout,stderr='',exit_code=0,
                transport_status='succeeded',facts=ExecutionFacts(disposition='terminal')))
        def fake_agent(owner,command,cwd,timeout,env):
            calls.append((cwd,command,timeout))
            self.assertIsNone(timeout)
            self.assertIn('AKA_AGENT_TOOL_ENDPOINT',env)
            cwd=Path(cwd)
            self.assertTrue((cwd/'PROGRESS.md').exists())
            self.assertFalse((cwd/'test_kernel.py').exists())
            self.assertFalse((cwd/'profile_driver.py').exists())
            (cwd/'scratch/probe.py').write_text('print("probe-result")\n')
            def invoke(name,body):
                value=json.loads(call_tool(name,json.dumps(body).encode(),workspace=cwd,environment=env))
                self.assertEqual(value['status'],200,value)
                result=value['result']
                self.assertNotEqual(result.get('status'),'running',result)
                self.assertNotIn('gateway_task_running',json.dumps(result))
                return result
            help_result=subprocess.run([sys.executable,'tools/sandbox.py','--kind','profile','--help'],
                cwd=cwd,env={**os.environ,**env,'PYTHONPATH':str(ROOT/'src')},capture_output=True,text=True)
            available=json.loads(env['AKA_AGATE_TOOLS'])
            discovery=subprocess.run([sys.executable,'-m','aka.resources.cli','list'],cwd=cwd,
                env={**os.environ,**env,'PYTHONPATH':str(ROOT/'src')},capture_output=True,text=True)
            self.assertEqual(discovery.returncode,0,discovery.stdout+discovery.stderr)
            self.assertNotIn('gpu-wiki',discovery.stdout)
            self.assertNotIn('kernel-wiki',discovery.stdout)
            self.assertNotIn('aka-execution',discovery.stdout)
            self.assertNotIn('atrex-gpu-gateway',discovery.stdout)
            import re
            measurement_skill=cwd/'.claude/skills/gpu-measurement'
            guidance='\n'.join(p.read_text() for p in measurement_skill.rglob('*.md'))
            self.assertEqual(set(re.findall(r'--kind ([a-z][a-z-]*)',guidance)),set(available))
            self.assertEqual(available,list(launch.request.runtime.agate_tools))
            if 'profile' in available:
                self.assertEqual(help_result.returncode,0,help_result.stderr)
                self.assertIn('--profiler',help_result.stdout)
            else:
                self.assertNotEqual(help_result.returncode,0)
                self.assertIn('runtime_tool_disabled',help_result.stderr)
                self.assertNotIn('invalid_runtime_response',help_result.stderr)
            overview=subprocess.run([sys.executable,'tools/sandbox.py','--help'],cwd=cwd,
                env={**os.environ,**env},capture_output=True,text=True)
            self.assertEqual(overview.returncode,0)
            self.assertIn('Agate Runtime tools: ' + ', '.join(available),overview.stdout)
            process=subprocess.run([sys.executable,'tools/sandbox.py','--kind','command','--','python3','-m','scratch.probe'],
                cwd=cwd,env={**os.environ,**env,'PYTHONPATH':str(ROOT/'src')},capture_output=True,text=True)
            self.assertEqual(process.returncode,0,process.stderr)
            receipt=json.loads(process.stdout)
            self.assertEqual(receipt['result']['stdout'],'probe-result\n',receipt)
            self.assertTrue((cwd/receipt['record_path']).exists())
            tool_replies.append(receipt)
            if evaluate:
                measured=invoke('workflow.gateway',{'argv':['--kind','evaluate']})
                self.assertIn('result',measured,measured)
                self.assertTrue(measured['result'].get('correct'),measured)
                self.assertEqual(measured['status'],'completed',measured)
                self.assertTrue((cwd/measured['record_path']).exists(),measured)
                commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=cwd,text=True).strip()
                submission=invoke('workflow.journal',{'operation':'kernel_submit','request':{'kernel_commit_id':commit}})
                accepted=submission['result']
                self.assertTrue(accepted['accepted'],accepted)
                self.assertIn(commit,(cwd/'PROGRESS.md').read_text())
                if abba:
                    comparison=invoke('workflow.gateway',{'argv':['--kind','evaluate','--baseline-commit',commit]})
                    record=json.loads((cwd/comparison['record_path']).read_text())
                    self.assertIn('gateway_record_id',comparison,record)
                    from aka.task.source.formats.artifacts import read
                    artifact=read(cwd)
                    stored=MeasurementStore(launch.request.workspace/'measurements').read(comparison['gateway_record_id'])
                    self.assertTrue(artifact.verify_measurement_inputs(stored['request'],'candidate'))
                    self.assertTrue(artifact.verify_measurement_inputs(stored['request'],'baseline'))
            direction=None
            if 'directions' in modules:
                result=invoke('workflow.journal',{'operation':'direction_update','request':{
                    'action':'propose','name':'Memory traffic and launch architecture','change_scope':'major_redesign',
                    'objective':'Reduce memory and launch overhead','applicability':'All supported vector sizes',
                    'expected_benefit':'Reduce full-workload latency','plan':['Compare launch families','Measure occupancy regimes'],
                    'stop_conditions':['All promising families measured']}})
                proposed=result['result']
                direction=proposed['direction_id']
                self.assertEqual(proposed['review']['claim_review_quality'],'supported')
                self.assertEqual(proposed['revision'],1)
                self.assertTrue(proposed['archive_path'].endswith('/v1/direction.json'))
                proposal_path=cwd/proposed['archive_path']
                proposal_bytes=proposal_path.read_bytes()
                started=invoke('workflow.journal',{'operation':'direction_update','request':{
                    'action':'start','direction_id':direction,'analysis':'Explore the planned strategy families'}})['result']
                self.assertNotIn('review',started)
                self.assertEqual(started['revision'],2)
                self.assertTrue(started['archive_path'].endswith('/v2/direction.json'))
                self.assertEqual(proposal_path.read_bytes(),proposal_bytes)
                self.assertEqual(json.loads((cwd/started['archive_path']).read_text())['status'],'in_progress')
            if 'experiments' in modules:
                result=invoke('workflow.journal',{'operation':'experiment_record','request':{
                    'name':'Probe output','intent':'Check the command transport','interpretation':'The script printed probe-result.',
                    'gateway_record_ids':[receipt['gateway_record_id']], 'references':['scratch/probe.py'],
                    **({'direction_id':direction} if direction else {})}})
                self.assertIn('supported',json.dumps(result['result']))
            # Errors must traverse the same scoped ToolHost, not collapse into transport errors.
            bad=json.loads(call_tool('workflow.gateway',b'{"argv":["--kind","invalid"]}',workspace=cwd,environment=env))
            self.assertEqual(bad['status'],400,bad)
            invalid=json.loads(call_tool('workflow.gateway',b'{"argv":["--kind","evaluate","--warmup","-1"]}',workspace=cwd,environment=env))
            self.assertEqual(invalid['status'],400,invalid)
            self.assertTrue((cwd/invalid['result']['record_path']).exists())
            environment=invoke('workflow.gateway',{'argv':['--kind','env','--env-capabilities']})
            self.assertEqual(environment['operation'],'env')
            self.assertIn('counters',json.dumps(environment['result']))
            self.assertTrue((cwd/environment['record_path']).exists())
            return json.dumps({'type':'result','subtype':'success','usage':{'input_tokens':12,'output_tokens':3}})+'\n','',0,False
        config={'tool_modules':list(modules),**({'search':search} if search else {})}
        with patch.dict(os.environ, {'AKA_HOST_CANARY':'must-stay-on-host','ANTHROPIC_API_KEY':'host-only-test-key'}), \
             patch('aka.workflow.light.runtime_preparation.prepare_runtime',return_value=('sm_89','fake')), \
             patch('aka.agent.provider.Agent._execute',fake_agent), \
             patch('aka.execution.binding.SandboxExecutor.run',fake_gpu), \
             patch('aka.workflow.light.orchestration.supervisor_runtime.SupervisorRuntime._measurement_environment'), \
             patch('aka.execution.binding.SandboxExecutor.query_environment', return_value=__import__('aka.contracts.execution_environment',fromlist=['EnvironmentSnapshot']).EnvironmentSnapshot('agate','test','L20N','test',observed={'counters':['c']*2000})), \
             patch('aka.workflow.light.orchestration.campaign.Campaign.review_research_claim',return_value=REVIEW):
            result=run_profile(profile_path('light'),Invocation(encode_launch(launch)),
                variables={'repo_root':str(ROOT)},patches=[{'id':'workflow','config':config}])
        self.assertEqual(result,0)
        root=launch.request.workspace
        self.assertFalse((root/'.git').exists())
        self.assertFalse(list(root.rglob('.atrex_long_horizon')))
        self.assertFalse(list(root.rglob('journal.json')))
        records=MeasurementStore(root/'measurements').record_ids()
        self.assertTrue(records)
        if 'directions' in modules:
            for claim in (root/'measurements/reviews').glob('direction_*'):
                self.assertTrue((claim/'v1/result.json').exists())
                self.assertFalse((claim/'v2/result.json').exists())  # Start creates no Review.
        self.assertGreaterEqual(len(gpu_calls),len(tool_replies))
        for cwd in {Path(call[0]) for call in calls}:
            own=[call for call in calls if call[0]==cwd]
            self.assertEqual(len(own),2)
            self.assertNotIn('--resume',own[0][1]);self.assertIn('--resume',own[1][1])
            self.assertEqual(set(subprocess.check_output(['git','ls-files'],cwd=cwd,text=True).splitlines()),{'.gitignore','kernel.py'})
        return root

    def test_full_evaluate_and_submission(self):
        self.run_campaign("fast",(),evaluate=True)

    def test_abba_retains_candidate_and_baseline_evidence(self):
        self.run_campaign("fast",(),evaluate=True,abba=True)

    def test_modes_and_research_modules(self):
        for mode,modules in [('fast',()),('performance',('directions',)),('ultra',('experiments',)),('fast',('directions','experiments'))]:
            with self.subTest(mode=mode,modules=modules):self.run_campaign(mode,modules)

    def test_synchronous_and_asynchronous_persistent_lineages(self):
        for synchronization in ('synchronous','asynchronous'):
            with self.subTest(synchronization=synchronization):
                root=self.run_campaign('fast',('directions','experiments'),{
                    'synchronization':synchronization,'epochs':2,'trajectories':2,'max_parallel_episodes':2,'episodes_per_trajectory':1})
                self.assertTrue((root/'coordinator.json').exists())
                self.assertTrue(list((root/'publications').glob('p*/publication.json')))
