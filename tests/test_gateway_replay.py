"""Unknown Dev outcomes remain inspectable and cannot trigger another POST."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from aka.light.execution.gateway_jobs import execute_job, retry_kind
from aka.light.experiment.store import JOB_ROOT_ENV, EvidenceUnavailable


def response(**fields):
    return subprocess.CompletedProcess([], 1 if fields.get('status') == 'failed' else 0,
                                       json.dumps(fields), '')


class GatewayReplayTest(unittest.TestCase):
    def test_ambiguous_terminal_is_not_replayed_even_by_custom_classifier_or_resume(self):
        uncertain = response(job_id='dv-original', status='failed', execution_state='unknown',
                             replay_policy='never', error={'error_class': 'infra',
                                                          'reason': 'dev_submission_ambiguous'})
        self.assertIsNone(retry_kind(json.loads(uncertain.stdout)))
        with tempfile.TemporaryDirectory() as raw, patch.dict(os.environ, {JOB_ROOT_ENV: str(Path(raw).resolve())}):
            submit = Mock(return_value=uncertain)
            poll = Mock(side_effect=lambda process, *args, **kwargs: process)
            with patch('aka.light.execution.gateway_jobs.time.sleep', side_effect=AssertionError('replayed')):
                for _ in range(2):
                    result = execute_job({'kind': 'dev', 'command': 'probe'}, submit, poll,
                                         wait_budget=10, retry_classifier=lambda job: 'infra')
                    self.assertEqual(result.stdout, uncertain.stdout)
            submit.assert_called_once()
            saved = list(Path(raw).glob('jobs/*/state.json'))
            self.assertEqual(json.loads(saved[0].read_text())['phase'], 'terminal')

    def test_known_stopped_infra_job_with_allow_policy_can_retry(self):
        failed = response(job_id='ev-old', status='failed', execution_state='stopped',
                          replay_policy='allow', error={'error_class': 'infra'})
        passed = response(job_id='ev-new', status='succeeded', execution_state='stopped',
                          replay_policy='allow', result={'ok': True})
        with tempfile.TemporaryDirectory() as raw, patch.dict(os.environ, {JOB_ROOT_ENV: str(Path(raw).resolve())}):
            submit = Mock(side_effect=[failed, passed])
            with patch('aka.light.execution.gateway_jobs.time.sleep'):
                result = execute_job({'kind': 'eval'}, submit,
                                     lambda process, *args, **kwargs: process,
                                     wait_budget=10, max_retries=1)
            self.assertEqual(json.loads(result.stdout)['job_id'], 'ev-new')
            self.assertEqual(submit.call_count, 2)

    def test_wait_failure_cannot_replace_a_job_with_never_replay_policy(self):
        accepted = response(job_id='dv-original', status='queued', replay_policy='never')
        with tempfile.TemporaryDirectory() as raw, patch.dict(os.environ, {JOB_ROOT_ENV: str(Path(raw).resolve())}):
            submit = Mock(return_value=accepted)
            poll = Mock(side_effect=EvidenceUnavailable('wait ended'))
            with self.assertRaisesRegex(EvidenceUnavailable, 'wait ended'):
                execute_job({'kind': 'dev'}, submit, poll, wait_budget=10)
            submit.assert_called_once()

    def test_cli_transport_loss_without_id_requires_reconciliation_on_resume(self):
        lost = subprocess.CompletedProcess([], 1, '', 'transport lost')
        with tempfile.TemporaryDirectory() as raw, patch.dict(os.environ, {JOB_ROOT_ENV: str(Path(raw).resolve())}):
            submit = Mock(return_value=lost)
            poll = Mock(side_effect=AssertionError('no accepted ID'))
            execute_job({'kind': 'dev'}, submit, poll, wait_budget=10)
            with self.assertRaisesRegex(EvidenceUnavailable, 'reconcile its outcome'):
                execute_job({'kind': 'dev'}, submit, poll, wait_budget=10)
            submit.assert_called_once()

    def test_new_no_replay_evidence_is_retained_when_collection_is_incomplete(self):
        accepted = response(job_id='dv-original', status='queued')
        running = response(job_id='dv-original', status='running',
                           execution_state='unknown', replay_policy='never')
        with tempfile.TemporaryDirectory() as raw, patch.dict(os.environ, {JOB_ROOT_ENV: str(Path(raw).resolve())}):
            submit = Mock(return_value=accepted)
            poll = Mock(side_effect=[running, EvidenceUnavailable('collection interrupted')])
            self.assertEqual(execute_job({'kind': 'dev'}, submit, poll, wait_budget=10).stdout, running.stdout)
            with self.assertRaisesRegex(EvidenceUnavailable, 'collection interrupted'):
                execute_job({'kind': 'dev'}, submit, poll, wait_budget=10)
            submit.assert_called_once()
            saved = json.loads(next(Path(raw).glob('jobs/*/state.json')).read_text())
            self.assertEqual(json.loads(saved['process']['stdout'])['replay_policy'], 'never')


if __name__ == '__main__':
    unittest.main()
