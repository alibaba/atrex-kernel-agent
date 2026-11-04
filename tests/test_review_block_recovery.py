import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from aka.contracts.acceptance import ValidationBlocked
from aka.contracts.workflow import WorkflowSettings
from aka.legacy.acceptance.source_policy import production_kernel_violations
from aka.light.execution.retry import ReviewerExecutionTimeout, retry_review
from aka.light.workflow.episode import LongHorizonCampaign


class ReviewRecoveryTests(unittest.TestCase):
    def test_timeout_budget_survives_restart_and_new_policy_can_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch('aka.light.execution.retry._state_path',
                       side_effect=lambda workspace, category, key: root / category / (key + '.json')):
                operation = Mock(side_effect=ReviewerExecutionTimeout('deadline'))
                for _ in range(2):
                    with self.assertRaises(ValidationBlocked):
                        retry_review(root, 'candidate:600', operation)
                self.assertEqual(operation.call_count, 2)
                verdict = Mock(return_value=[])
                self.assertEqual(retry_review(root, 'candidate:0', verdict), [])
                verdict.assert_called_once()

    def test_corrupt_retry_state_blocks_without_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / 'state.json'
            path.write_text('{broken')
            operation = Mock()
            with patch('aka.light.execution.retry._state_path', return_value=path):
                with self.assertRaises(ValidationBlocked):
                    retry_review(path.parent, 'candidate:600', operation)
            operation.assert_not_called()
            self.assertEqual(path.read_text(), '{broken')

    def test_policy_keeps_block_separate_from_rejection(self):
        with self.assertRaises(ValidationBlocked):
            production_kernel_violations(Path('.'), 'cuda',
                structure_validator=lambda *a, **k: [],
                production_reviewer=Mock(side_effect=ValidationBlocked('deadline')))
        self.assertEqual(production_kernel_violations(Path('.'), 'cuda',
            structure_validator=lambda *a, **k: [],
            production_reviewer=lambda *a: ['forbidden dependency']), ['forbidden dependency'])

    def test_handoff_preserved_and_same_candidate_resumes(self):
        base = SimpleNamespace(workspace=Path('/campaign'),
            candidate_workspace=SimpleNamespace(validate_candidate=Mock(return_value=('', ['kernel.py']))))
        campaign = LongHorizonCampaign(base)
        campaign._completion_check = Mock(return_value='')
        store = SimpleNamespace(save_active=Mock())
        active = {'episode': 21, 'phase': 'exploring'}
        candidate = SimpleNamespace(path=Path('/candidate'), base_commit='base')
        handoff = SimpleNamespace(status='candidate_ready', candidate_commit='selected')
        verification = SimpleNamespace(passed=True)
        verifier = SimpleNamespace(repeats=2, per_run_timeout=300, shape_batch_size=64,
                                   verify=Mock(return_value=verification))
        with patch('aka.light.workflow.episode.main_adapter.candidate_is_gluon', return_value=False), \
             patch('aka.light.workflow.episode.main_adapter.candidate_policy_violations',
                   side_effect=[ValidationBlocked('deadline'), []]), \
             patch('aka.light.execution.retry.retry_infrastructure', side_effect=lambda w,s,op,**k: op()):
            with self.assertRaises(ValidationBlocked):
                campaign._assess_terminal_handoff(store, active, candidate, handoff,
                    memory_version=51, conversion_pending=False, verifier=verifier)
            verifier.verify.assert_not_called()
            self.assertEqual(active['validation_blocker'], 'deadline')
            self.assertEqual(active['phase'], 'verifying')
            result = campaign._assess_terminal_handoff(store, active, candidate, handoff,
                memory_version=51, conversion_pending=False, verifier=verifier)
            self.assertTrue(result[-1])
            self.assertNotIn('validation_blocker', active)
            self.assertEqual(verifier.verify.call_args.kwargs['candidate_commit'], 'selected')

    def test_review_deadline_zero_is_explicitly_supported(self):
        from aka.light.agent.execution_clock import validate_timeout, ActiveBudget
        self.assertEqual(validate_timeout(0), math.inf)
        self.assertEqual(ActiveBudget(0).remaining, math.inf)
        self.assertEqual(WorkflowSettings('ZW-M890P', 'Cuda', production_review_timeout=0).production_review_timeout, 0)
        for invalid in [-1, True, 0.5]:
            with self.assertRaises(ValueError):
                WorkflowSettings('ZW-M890P', 'Cuda', production_review_timeout=invalid)


if __name__ == '__main__':
    unittest.main()
