"""Host/remote protocol recovery and active-time accounting, without real GPUs."""
import base64
import copy
import io
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.request

from aka.light.execution.contract import CommandSpec
from aka.light.execution.remote import Executor
from aka.light.execution.remote_pool import PoolDriver
from aka.light.execution.remote_resilience import Failure
from aka.agent.execution.control import SessionControl
from aka.light.tasks.runtime import TaskRuntime, attach_task_control, invoke
import test_remote_pool as fixture


class DriverTests(unittest.TestCase):
    setUp = fixture.PoolTests.setUp
    close_all = fixture.PoolTests.close_all
    launch = fixture.PoolTests.launch
    finish = fixture.PoolTests.finish

    def driver(self):
        source = self.root / 'source';source.mkdir(exist_ok=True)
        (source / 'input.bin').write_bytes(b'original source')
        target = {'backend': 'remote', 'config': {'pod': 'pod', 'uid': 'uid',
            'remote_root': str(self.root / 'uploaded'),
            'scheduling': {'gpu_ids': [0, 1], 'replay_safe': True, 'poll_interval_s': .01}}}
        executor = Executor(target=target, directory=self.root / 'host-job', execution_id='eval-1',
                            previous_job='', on_job=lambda _:None, control=lambda:'run')
        command = CommandSpec(source, ('command',), outputs=('report.bin',))
        driver = PoolDriver(executor, command)
        boot = {'pod_uid': 'uid', 'container_id': 'container-1', 'image_id': 'image-sha'}
        self.addCleanup(patch.stopall)
        patch.object(executor, '_pod_state', return_value=boot).start()
        def initialize(value):
            path = Path(executor.root);path.mkdir(parents=True, exist_ok=True)
            (path / 'spec.json').write_text(json.dumps(value));(path / 'runner.py').write_text('# protocol')
            return {}
        patch.object(executor, '_initialize', side_effect=initialize).start()
        def part(operation, index, data):
            self.assertEqual(operation, 'part')
            (Path(executor.root) / ('part-' + str(index))).write_bytes(base64.b64decode(data))
        patch.object(executor, '_rpc', side_effect=part).start()
        self.options = dict(lost_enqueue=False, complete=True, exit_code=0)
        def rpc(operation, value):
            if operation == 'cached':return {'cached': self.broker.cached(value['sha256'])}
            if operation == 'enqueue':
                result = self.broker.enqueue(value)
                if self.options.pop('lost_enqueue', False):raise Failure('transport_timeout', transient=True)
                return result
            if operation == 'status':return self.broker.status(value['job_id'])
            if operation == 'cancel':return self.broker.cancel(value['job_id'], queued_only=value.get('queued_only', False))
            if operation == 'collect':return self.broker.collect(value['job_id'])
            if operation == 'status_many':
                self.broker.tick()
                if self.options['complete']:
                    for job in list(self.held):
                        status = self.broker.status(job)
                        (Path(status['attempt_root']) / 'workspace/report.bin').write_bytes(b'verified output')
                        self.finish(job, exit_code=self.options['exit_code'])
                    self.broker.tick()
                with self.broker.connect() as db:ids=[x[0] for x in db.execute('SELECT id FROM jobs')]
                return {'jobs': {key: self.broker.status(key) for key in ids}}
            raise AssertionError(operation)
        patch.object(driver, 'rpc', side_effect=rpc).start()
        patch.dict(os.environ, {'AKA_REMOTE_POOL_STATE_DIR': str(self.root / 'host-state')}).start()
        return driver

    def test_real_host_protocol_collects_verified_output(self):
        driver = self.driver();result = driver.run()
        self.assertEqual(result.exit_code, 0)
        payload = result.details['receipt']['payload']
        self.assertEqual(base64.b64decode(payload['artifacts']['report.bin']['base64']), b'verified output')
        self.assertEqual(payload['pool']['attempt'], 1)
        self.assertEqual(len(self.launches), 1)

    def test_completed_receipt_recovers_offline_with_same_input(self):
        driver = self.driver();driver.run()
        with patch.object(driver, 'rpc', side_effect=AssertionError('network accessed')):
            self.assertEqual(driver.run().exit_code, 0)
        self.assertEqual(len(self.launches), 1)

    def test_terminal_blocked_attempt_returns_nonpassing_receipt_without_retry(self):
        driver=self.driver()
        failed={'state':'blocked','attempt':3,'result':{'exit_code':None,
            'contaminated':True,'valid_measurement':False,'reason':'verified_supervisor_loss'}}
        driver.rpc.side_effect=None
        driver.rpc.return_value={'status':failed,'files':{}}
        receipt=driver.collect(failed).details['receipt']
        self.assertEqual(receipt['status'],'failed')
        self.assertTrue(receipt['payload']['contaminated'])
        self.assertFalse(receipt['payload']['valid_measurement'])
        self.assertEqual(receipt['payload']['failure_origin'],'infrastructure')
        self.assertEqual(receipt['payload']['reason'],'verified_supervisor_loss')
        self.assertEqual(len(self.launches),0)

    def test_distributed_retry_exhaustion_retains_execution_failure_origin(self):
        driver=self.driver()
        failed={'state':'blocked','attempt':3,'result':{'exit_code':1,
            'retryable_group_failure':True,'valid_measurement':False,'reason':'rank initialization failed'}}
        driver.rpc.side_effect=None;driver.rpc.return_value={'status':failed,'files':{}}
        receipt=driver.collect(failed).details['receipt']
        self.assertEqual(receipt['status'],'failed')
        self.assertEqual(receipt['payload']['failure_origin'],'execution')
        self.assertFalse(receipt['payload']['valid_measurement'])

    def test_changed_inputs_or_gpu_request_cannot_reuse_a_complete_receipt(self):
        driver = self.driver();driver.run()
        for key, value in [('monitor_interval_s', .02), ('max_contamination_retries', 0)]:
            with self.subTest(key=key):
                driver.options[key] = value
                with self.assertRaises(Failure):driver.run()
                del driver.options[key]
        driver.options['gpu_ids'] = [1]
        with self.assertRaises(Failure):driver.run()
        driver.options['gpu_ids'] = [0, 1]
        (driver.command.workspace / 'input.bin').write_bytes(b'changed')
        with self.assertRaises(Failure):driver.run()

    def test_corrupted_local_receipt_is_not_returned_as_success(self):
        driver = self.driver();driver.run()
        (driver.e.directory / 'pool-collected.json').write_text('{}')
        with self.assertRaises(Failure):driver.run()

    def test_lost_enqueue_ack_recovers_original_id_without_an_extra_launch(self):
        driver = self.driver();self.options['lost_enqueue'] = True
        with self.assertRaises(Failure):driver.run()
        self.assertEqual(driver.run().exit_code, 0)
        self.assertEqual(len(self.launches), 1)

    def test_waiting_client_can_cancel_without_taking_a_gpu(self):
        driver = self.driver()
        for device in self.probe.rows:device['pids'] = [777]
        driver.e.control = lambda:'cancel'
        # First control read is before upload; no admission is needed at all.
        result = driver.run()
        self.assertEqual(result.details['receipt']['status'], 'cancelled')
        self.assertFalse(self.launches)

    def test_kernel_failure_reaches_caller_without_retry(self):
        driver = self.driver();self.options['exit_code'] = 7
        result = driver.run()
        self.assertEqual(result.exit_code, 7)
        self.assertEqual(result.details['receipt']['status'], 'failed')
        self.assertEqual(len(self.launches), 1)

    def test_changed_container_image_blocks_replay(self):
        driver = self.driver();self.options['lost_enqueue'] = True
        with self.assertRaises(Failure):driver.run()
        driver.e._pod_state = lambda:{'pod_uid': 'uid', 'container_id': 'container-2', 'image_id': 'different-image'}
        with self.assertRaises(Failure):driver.run()
        self.assertFalse(self.launches)


class BudgetTests(unittest.TestCase):
    def test_only_pooled_execution_calls_delegate_wait_deadlines_to_host(self):
        environment = {'AKA_TASK_ENDPOINT': 'http://127.0.0.1:1234', 'AKA_TASK_TOKEN': 'test'}
        for pooled, operation, expected in [(True, 'measure', None), (True, 'execute', None),
                                             (True, 'results', 10860), (False, 'measure', 10860)]:
            with self.subTest(pooled=pooled, operation=operation):
                with patch.dict(os.environ, {**environment, 'AKA_TASK_POOLED_WAIT': '1' if pooled else ''}), \
                     patch('urllib.request.urlopen', return_value=io.BytesIO(b'{}')) as request:
                    self.assertEqual(invoke(operation, {}), {})
                    self.assertEqual(request.call_args.kwargs['timeout'], expected)

    def test_pooled_runtime_wait_pauses_provider_and_its_active_clock(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'], start_new_session=True)
        control = SessionControl();control.attach(child.pid)
        entered, release = threading.Event(), threading.Event()
        class Service:
            store = SimpleNamespace(task=lambda _: {'manifest': {'target': {'config': {'scheduling': {'gpu_ids': [0]}}}}})
            def command(self, task, **args):
                entered.set();release.wait(3)
                return {'execution_id': 'logical-id'}
        try:
            with TaskRuntime(Service(), 'task') as runtime:
                detach = attach_task_control(runtime.environment, control)
                result = []
                def request():
                    req = urllib.request.Request(runtime.environment['AKA_TASK_ENDPOINT'] + '/execute',
                        data=b'{"argv":["command"]}', headers={'Authorization': 'Bearer ' + runtime.token,
                                                               'Content-Type': 'application/json'})
                    with urllib.request.urlopen(req, timeout=5) as response:result.append(json.load(response))
                thread = threading.Thread(target=request);thread.start()
                self.assertTrue(entered.wait(3));before = control.clock.active_seconds()
                self.assertEqual(control.pending, 1)
                time.sleep(.2)
                self.assertLess(control.clock.active_seconds() - before, .03)
                release.set();thread.join(5)
                self.assertEqual(result, [{'execution_id': 'logical-id'}])
                self.assertEqual(control.pending, 0)
                self.assertEqual(control.pending, 0)
                detach();self.assertIsNone(runtime.session_control)
        finally:
            release.set();control.kill();child.wait(timeout=5)

    def test_legacy_task_runtime_does_not_suspend_an_existing_session(self):
        class Service:
            store = SimpleNamespace(task=lambda _: {'manifest': {'target': {'backend': 'local'}}})
        with TaskRuntime(Service(), 'task') as runtime:
            control = SessionControl()
            detach = attach_task_control(runtime.environment, control)
            self.assertIsNone(runtime.session_control)
            detach()


if __name__ == '__main__':unittest.main()
