import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from aka.contracts.execution import CommandSpec
from aka.experiment.factory import ExperimentFactory
from aka.experiment.measurement_store import MeasurementStore
from aka.task.candidate_workspace.provider import GitCandidateWorkspace
from aka.workflow.light.measurements import SharedMeasurements
from aka.workflow.light.orchestration.episode_workspace import EpisodeWorkspace
from test_campaign_measurement_storage import TestExecutor, TestEvaluator


class Resources:
    def install(self, workspace): pass
    def refresh_custom_skills(self, workspace): pass
    def installed_paths(self, workspace): return ()


class RuntimeIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.store = MeasurementStore(self.root / 'measurements')
        self.task = GitCandidateWorkspace()
        self.evaluator = TestEvaluator()
        self.executor = TestExecutor()
        self.executor.run = Mock(wraps=self.executor.run)
        self.factory = ExperimentFactory(command_evaluator=self.evaluator).for_campaign(self.store.root)

    def bridge(self, lineage):
        path = self.root / 'lineages' / lineage / 'agent-workspace'
        path.mkdir(parents=True, exist_ok=True)
        return SharedMeasurements(factory=self.factory, executor=self.executor, evaluator=self.evaluator,
            command_evaluator=self.evaluator, namespace_workspace=path, evaluation_input=None,
            workspace_provider=self.task, store=self.store), path

    def test_cross_lineage_reuse_returns_normal_receipt_and_one_gateway_record(self):
        identity = {'operation': 'dev', 'options': {}, 'inputs': {}, 'command': ['measure']}
        first, w1 = self.bridge('t0001')
        second, w2 = self.bridge('t0002')
        a = first.execute(CommandSpec(w1, ('measure',), operation='dev'), request_id='first', identity=identity, kernels={})
        b = second.execute(CommandSpec(w2, ('measure',), operation='dev'), request_id='second', identity=identity, kernels={})
        self.assertEqual(self.executor.run.call_count, 1)
        self.assertEqual(a['agent_result'], b['agent_result'])
        self.assertEqual(a['agent_result']['status'], 'completed')
        self.assertNotIn('duplicate', json.dumps(b))
        self.assertEqual(len(self.store.record_ids()), 1)
        self.assertEqual(len(list(self.store.execution_records().iter_records())), 1)
        record = self.store.read(a['agent_result']['gateway_record_id'])
        self.assertEqual(record['runtime_request_id'], 'first')
        self.assertEqual(a['agent_result']['record_path'], f"records/{record['gateway_record_id']}.json")
        self.assertEqual(record['response']['stdout'].splitlines()[0], 'raw-dev')

    def test_failed_execution_and_preparation_remain_in_both_audit_layers(self):
        bridge, path = self.bridge('t0001')
        for command in ('compile-error', 'timeout', 'invalid-input'):
            result = bridge.execute(CommandSpec(path, (command,), operation='dev'), request_id=command,
                identity={'operation': 'dev', 'options': {}, 'inputs': {}, 'command': command}, kernels={})
            self.assertNotEqual(result['exit_code'], 0)
            self.store.read(result['agent_result']['gateway_record_id'])
        self.assertEqual(len(self.store.record_ids()), 3)
        self.assertEqual(len(list(self.store.execution_records().iter_records())), 3)

    def test_workspace_resume_keeps_head_index_wip_and_private_branches(self):
        root = self.root / 'agent-workspace'
        root.mkdir()
        def git(*args):
            return subprocess.check_output(['git', *args], cwd=root, stderr=subprocess.PIPE)
        git('init', '-q')
        git('config', 'user.email', 'test@example.invalid')
        git('config', 'user.name', 'Test')
        (root / '.gitignore').write_text('/*\n!/.gitignore\n!/kernel.py\n!/impl/\n!/impl/**\n')
        (root / 'kernel.py').write_text('initial = 1\n')
        git('add', '.gitignore', 'kernel.py');git('commit', '-qm', 'initial')
        git('branch', 'unmerged-experiment')
        (root / 'kernel.py').write_text('staged = 2\n');git('add', 'kernel.py')
        (root / 'kernel.py').write_text('dirty = 3\n')
        before=(git('rev-parse', 'HEAD'),git('show', ':kernel.py'),git('show-ref'),git('status', '--porcelain'))
        view=EpisodeWorkspace(root, resources_registry=Resources(), research_modules=('directions','experiments'))
        view.prepare();view.publish();view.prepare()
        after=(git('rev-parse', 'HEAD'),git('show', ':kernel.py'),git('show-ref'),git('status', '--porcelain'))
        self.assertEqual(before,after)
        self.assertEqual((root/'kernel.py').read_text(),'dirty = 3\n')
        self.assertFalse(list(self.root.rglob('journals')))
        self.assertIn('records',view.prepare_session())
        self.assertIn('PROGRESS.md',view.prepare_session())


    def test_claim_review_follows_selected_sandbox_and_declares_evidence_access(self):
        from aka.agent.provider import Agent, AgentOptions
        from aka.contracts.agent import AgentRunResult, TokenUsage
        from aka.workflow.light.campaign import ManagedCampaign
        evidence=self.root/'review-inputs';evidence.mkdir()
        for name in ['submissions','gateway-records','references','kernels']:(evidence/name).mkdir()
        (evidence/'evidence-manifest.json').write_text('{}')
        for sandbox in ('none', 'bwrap'):
            with self.subTest(sandbox=sandbox), patch('aka.agent.execution.bubblewrap.BubblewrapExecution') as bwrap:
                agent=Agent(AgentOptions(sandbox=sandbox,proc_mode='host'))
                self.addCleanup(agent.close)
                runtime=agent.runtime('claude')
                runtime.run=Mock(return_value=AgentRunResult('claude',0,False,TokenUsage.zero(),(),None,(),'',''))
                campaign=ManagedCampaign.__new__(ManagedCampaign)
                campaign.agent=agent
                campaign._run_session(evidence,'Review this evidence',None,
                    extra_environment={'ATREX_AGENT_WORKSPACE_ROLE':'research-claim-review'})
                request=runtime.run.call_args.args[0]
                self.assertEqual(request.workspace,evidence)
                self.assertIsNone(request.timeout_s)
                self.assertEqual({p.name for p in request.access.read_only_paths},
                    {'submissions','gateway-records','references','kernels','evidence-manifest.json'})
                self.assertNotIn(evidence/'review.json',request.access.read_only_paths)
                self.assertEqual(agent.capabilities.filesystem_isolation,sandbox=='bwrap')
                if sandbox=='bwrap':bwrap.assert_called_once_with('bwrap',proc_mode='host')
                else:bwrap.assert_not_called()


if __name__=='__main__': unittest.main()
