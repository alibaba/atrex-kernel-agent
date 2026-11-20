"""GPU admission safety tests; simulated GPU inventory, real OS leases/processes."""
import base64
import copy
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import zipfile

from aka.light.execution.contract import CommandSpec
from aka.light.execution.remote import Executor, validate_config
from aka.light.execution.remote_pool import PoolDriver, validate
from aka.light.execution import remote_pool_worker as worker
from aka.light.execution.remote_resilience import Failure


class Probe:
    def __init__(self, count=2):
        self.rows = [{'index': i, 'uuid': 'GPU-test-' + str(i), 'pids': []} for i in range(count)]

    def snapshot(self):
        return copy.deepcopy(self.rows)


class PoolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.probe = Probe()
        self.launches = []
        self.held = {}
        self.broker = worker.Broker(self.root / 'broker', probe=self.probe,
                                    launcher=self.launch, device_lock_root=self.root / 'gpu-locks')
        self.addCleanup(self.close_all)

    def close_all(self):
        for handles in self.held.values():
            for handle in handles:
                try:os.close(handle)
                except OSError:pass
        self.held.clear()

    def request(self, number=1, task=None, **changes):
        source = self.root / ('input-' + str(number))
        source.mkdir(exist_ok=True)
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as archive:
            archive.writestr('input.txt', 'frozen original input')
        blob = stream.getvalue()
        sha = hashlib.sha256(blob).hexdigest()
        (source / 'part-0').write_bytes(blob)
        (source / 'spec.json').write_text(json.dumps({'archive_sha256': sha, 'parts': 1}))
        (source / 'runner.py').write_text('# stat/read fixture\n')
        spec = dict(job_id='remote-' + format(number, '032x'), task=task or 'task-' + str(number),
                    source_root=str(source), archive_sha256=sha,
                    argv=[sys.executable, '-c', 'import time;time.sleep(.2)'], environment={}, outputs=[],
                    timeout_s=5, gpu_ids=[0, 1], gpu_count=1, replay_safe=True,
                    monitor_interval_s=.02, max_contamination_retries=5)
        spec.update(changes)
        return spec

    def launch(self, row, spec, devices, descriptors, fence):
        attempt = self.broker.root / 'jobs' / row['id'] / ('attempt-' + str(row['attempt'] + 1))
        self.broker.materialize(spec, attempt)
        worker.atomic(attempt / 'placement.json', {'devices': devices, 'fence': fence})
        self.launches.append((row['id'], row['attempt'] + 1, [x['index'] for x in devices], fence))
        self.held[row['id']] = [os.dup(fd) for fd in descriptors]

    def finish(self, job, *, contaminated=False, exit_code=0, release=True):
        status = self.broker.status(job)
        result = dict(exit_code=exit_code, timed_out=False, cancelled=False,
                      contaminated=contaminated, valid_measurement=not contaminated,
                      fence=status['fence'], attempt=status['attempt'])
        root = Path(status['attempt_root'])
        for name in ['stdout.log', 'stderr.log']:(root / name).write_bytes(b'')
        worker.atomic(root / 'attempt-result.json', result)
        if release:
            for fd in self.held.pop(job):os.close(fd)
        return result

    def test_selects_idle_gpu_even_when_preferred_first_card_is_busy(self):
        self.probe.rows[0]['pids'] = [777]
        spec = self.request()
        self.broker.enqueue(spec);self.broker.tick()
        self.assertEqual(self.launches[0][2], [1])

    def test_all_busy_waits_and_dispatches_when_a_card_becomes_idle(self):
        for gpu in self.probe.rows:gpu['pids'] = [888]
        spec = self.request();self.broker.enqueue(spec)
        self.broker.tick();self.broker.tick()
        self.assertFalse(self.launches)
        self.assertEqual(self.broker.status(spec['job_id'])['state'], 'queued')
        self.probe.rows[1]['pids'] = []
        self.broker.tick()
        self.assertEqual(self.launches[0][2], [1])

    def test_skip_a_locked_card_before_its_cuda_process_appears(self):
        held = self.broker.acquire([self.probe.rows[0]])
        try:
            spec = self.request();self.broker.enqueue(spec);self.broker.tick()
            self.assertEqual(self.launches[0][2], [1])
        finally:
            for fd in held:os.close(fd)

    def test_eight_tasks_share_two_cards_without_overlap(self):
        specs = [self.request(i) for i in range(1, 9)]
        for spec in specs:self.broker.enqueue(spec)
        completed = []
        for _ in range(4):
            self.broker.tick()
            active = [s for s in specs if self.broker.status(s['job_id'])['state'] == 'running']
            self.assertEqual(len(active), 2)
            assigned = [self.broker.status(s['job_id'])['placement']['devices'][0]['uuid'] for s in active]
            self.assertEqual(len(set(assigned)), 2)
            for spec in active:self.finish(spec['job_id']);completed.append(spec['job_id'])
        self.broker.tick()
        self.assertEqual(len(set(completed)), 8)
        self.assertTrue(all(self.broker.status(s['job_id'])['state'] == 'completed' for s in specs))

    def test_task_round_robin_prevents_one_task_monopolizing_the_card(self):
        self.probe.rows = self.probe.rows[:1]
        specs = [self.request(1, task='A'), self.request(2, task='A'), self.request(3, task='B')]
        for spec in specs:self.broker.enqueue(spec)
        self.broker.tick();self.assertEqual(self.launches[-1][0], specs[0]['job_id'])
        self.finish(specs[0]['job_id']);self.broker.tick()
        self.assertEqual(self.launches[-1][0], specs[2]['job_id'])

    def test_same_task_has_at_most_one_running_experiment(self):
        for spec in [self.request(1, task='A'), self.request(2, task='A'), self.request(3, task='B')]:self.broker.enqueue(spec)
        self.broker.tick()
        self.assertEqual({x[0] for x in self.launches}, {'remote-' + format(1, '032x'), 'remote-' + format(3, '032x')})

    def test_duplicate_enqueue_or_lost_ack_does_not_launch_twice(self):
        spec = self.request()
        for _ in range(4):self.broker.enqueue(spec)
        self.broker.tick();self.broker.enqueue(spec);self.broker.tick()
        self.assertEqual(len(self.launches), 1)
        other = {**spec, 'argv': ['changed']}
        with self.assertRaises(ValueError):self.broker.enqueue(other)

    def test_restart_broker_keeps_existing_lease_and_job(self):
        spec = self.request();self.broker.enqueue(spec);self.broker.tick()
        replacement = worker.Broker(self.broker.root, probe=self.probe,
                                   launcher=self.launch, device_lock_root=self.root / 'gpu-locks')
        replacement.tick()
        self.assertEqual(replacement.status(spec['job_id'])['attempt'], 1)
        self.assertEqual(len(self.launches), 1)

    def test_different_broker_roots_still_share_physical_device_locks(self):
        spec = self.request();self.broker.enqueue(spec);self.broker.tick()
        self.probe.rows = self.probe.rows[:1]
        calls = []
        second = worker.Broker(self.root / 'second-broker', probe=self.probe,
                              launcher=lambda *args:calls.append(args), device_lock_root=self.root / 'gpu-locks')
        second.enqueue(self.request(2));second.tick()
        self.assertFalse(calls)

    def test_result_is_not_released_before_supervisor_gpu_lock(self):
        self.probe.rows = self.probe.rows[:1]
        first, second = self.request(), self.request(2)
        self.broker.enqueue(first);self.broker.enqueue(second);self.broker.tick()
        self.finish(first['job_id'], release=False);self.broker.tick()
        self.assertEqual(len(self.launches), 1)
        for fd in self.held.pop(first['job_id']):os.close(fd)
        self.broker.tick();self.assertEqual(len(self.launches), 2)

    def test_contaminated_attempt_is_discarded_and_requeued_with_fresh_inputs(self):
        spec = self.request(gpu_ids=[0]);self.broker.enqueue(spec);self.broker.tick()
        first = Path(self.broker.status(spec['job_id'])['attempt_root'])
        (first / 'workspace/input.txt').write_text('changed by invalid attempt')
        self.probe.rows[0]['pids'] = [777]
        self.finish(spec['job_id'], contaminated=True);self.broker.tick()
        self.assertEqual(self.broker.status(spec['job_id'])['state'], 'queued')
        with self.assertRaises(ValueError):self.broker.collect(spec['job_id'])
        self.probe.rows[0]['pids'] = [];self.broker.tick()
        second = Path(self.broker.status(spec['job_id'])['attempt_root'])
        self.assertNotEqual(first, second)
        self.assertEqual((second / 'workspace/input.txt').read_text(), 'frozen original input')
        self.finish(spec['job_id']);self.broker.tick()
        self.assertEqual(self.broker.status(spec['job_id'])['state'], 'completed')
        self.assertEqual(self.broker.status(spec['job_id'])['attempt'], 2)

    def test_contamination_requires_explicit_replay_safety(self):
        spec = self.request(replay_safe=False);self.broker.enqueue(spec);self.broker.tick()
        self.finish(spec['job_id'], contaminated=True);self.broker.tick()
        self.assertEqual(self.broker.status(spec['job_id'])['state'], 'blocked')
        self.assertEqual(len(self.launches), 1)

    def test_real_kernel_failure_is_not_infrastructure_retried(self):
        spec = self.request();self.broker.enqueue(spec);self.broker.tick()
        self.finish(spec['job_id'], exit_code=7);self.broker.tick()
        self.assertEqual(self.broker.status(spec['job_id'])['state'], 'failed')
        self.assertEqual(len(self.launches), 1)

    def test_missing_outputs_cannot_become_a_successful_measurement(self):
        spec = self.request(outputs=['result.json']);self.broker.enqueue(spec);self.broker.tick()
        self.finish(spec['job_id']);self.broker.tick()
        self.assertEqual(self.broker.status(spec['job_id'])['state'], 'failed')

    def test_multi_gpu_request_acquires_all_devices_atomically(self):
        spec = self.request(gpu_count=2);self.broker.enqueue(spec)
        held = self.broker.acquire([self.probe.rows[1]])
        try:
            self.broker.tick();self.assertFalse(self.launches)
            free = self.broker.acquire([self.probe.rows[0]])
            self.assertIsNotNone(free)
            for fd in free:os.close(fd)
        finally:
            for fd in held:os.close(fd)
        self.broker.tick();self.assertEqual(self.launches[0][2], [0, 1])

    def test_queued_cancel_never_launches_a_cuda_program(self):
        spec = self.request();self.broker.enqueue(spec)
        self.broker.cancel(spec['job_id']);self.broker.tick()
        self.assertFalse(self.launches)
        self.assertEqual(self.broker.collect(spec['job_id'])['status']['state'], 'cancelled')

    def test_pause_using_stale_queued_status_does_not_cancel_a_running_job(self):
        spec = self.request();self.broker.enqueue(spec);self.broker.tick()
        status = self.broker.cancel(spec['job_id'], queued_only=True)
        self.assertEqual(status['state'], 'running')
        self.assertFalse((self.broker.root / 'jobs' / spec['job_id'] / 'cancel').exists())

    def test_cancel_between_grant_and_launch_creates_no_child(self):
        spec, attempt = self.supervisor_fixture('raise SystemExit("must not run")')
        self.broker.cancel(spec['job_id'])
        result = worker.supervise(self.broker.root, spec['job_id'], 1,
                                  self.held.pop(spec['job_id']), probe=self.probe)
        self.assertTrue(result['cancelled'])
        self.assertFalse((attempt / 'child.json').exists())

    def test_new_database_epoch_never_reuses_an_old_launch_fence(self):
        spec = self.request();self.broker.enqueue(spec);self.broker.tick()
        old = self.broker.status(spec['job_id'])['fence']
        second = worker.Broker(self.root / 'other-epoch', probe=self.probe,
                              launcher=lambda *args:None, device_lock_root=self.root / 'other-locks')
        second.enqueue(self.request(2));second.tick()
        self.assertNotEqual(old, second.status(self.request(2)['job_id'])['fence'])

    def test_failed_inventory_never_means_idle(self):
        spec = self.request();self.broker.enqueue(spec)
        with patch.object(self.probe, 'snapshot', side_effect=worker.ProbeUnavailable('cannot read')):
            with self.assertRaises(worker.ProbeUnavailable):self.broker.tick()
        self.assertFalse(self.launches)

    def supervisor_fixture(self, program, *, timeout=3):
        spec = self.request(argv=[sys.executable, '-c', program], timeout_s=timeout)
        self.broker.enqueue(spec);self.broker.tick()
        status = self.broker.status(spec['job_id'])
        attempt = Path(status['attempt_root'])
        return spec, attempt

    def test_supervisor_allows_its_own_cuda_pid(self):
        spec, attempt = self.supervisor_fixture('import time;time.sleep(.15);print("correct")')
        outer = self
        class OwnProbe:
            def snapshot(self):
                rows = copy.deepcopy(outer.probe.rows)
                file = attempt / 'child.json'
                if file.exists():
                    pid = json.loads(file.read_text())['pid']
                    if pid in worker.process_table():rows[0]['pids'] = [pid]
                return rows
        result = worker.supervise(self.broker.root, spec['job_id'], 1,
                                  self.held.pop(spec['job_id']), probe=OwnProbe())
        self.assertEqual(result['exit_code'], 0)
        self.assertFalse(result['contaminated'])
        self.assertTrue(result['valid_measurement'])

    def test_foreign_process_invalidates_attempt_but_survives_cleanup(self):
        spec, attempt = self.supervisor_fixture('import time;time.sleep(30)')
        foreign = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'])
        self.addCleanup(lambda: foreign.poll() is None and foreign.terminate())
        outer = self
        class ForeignProbe:
            calls = 0
            def snapshot(self):
                self.calls += 1
                rows = copy.deepcopy(outer.probe.rows)
                if self.calls >= 3:rows[0]['pids'] = [foreign.pid]
                return rows
        result = worker.supervise(self.broker.root, spec['job_id'], 1,
                                  self.held.pop(spec['job_id']), probe=ForeignProbe())
        self.assertTrue(result['contaminated'])
        self.assertFalse(result['valid_measurement'])
        self.assertIsNone(foreign.poll(), 'An observed external task was killed')
        self.assertIn(foreign.pid, result['foreign']['GPU-test-0'])
        foreign.terminate();foreign.wait(timeout=5)

    def test_owned_cuda_process_on_unassigned_gpu_is_a_visible_error(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'], start_new_session=True)
        owner = worker.OwnedProcesses(child)
        try:
            self.probe.rows[1]['pids'] = [child.pid]
            with self.assertRaises(worker.ResourceViolation):
                worker.inspect_foreign(self.probe, [self.probe.rows[0]], owner)
            self.assertIsNone(child.poll())
        finally:
            owner.stop()

    def test_supervisor_does_not_infra_retry_its_own_wrong_gpu_usage(self):
        spec, attempt = self.supervisor_fixture('import time;time.sleep(30)')
        outer = self
        class WrongDeviceProbe:
            def snapshot(self):
                rows = copy.deepcopy(outer.probe.rows)
                file = attempt / 'child.json'
                if file.exists():
                    pid = json.loads(file.read_text())['pid']
                    if pid in worker.process_table():rows[1]['pids'] = [pid]
                return rows
        result = worker.supervise(self.broker.root, spec['job_id'], 1,
                                  self.held.pop(spec['job_id']), probe=WrongDeviceProbe())
        self.assertEqual(result['reason'], 'resource_violation')
        self.assertNotEqual(result['exit_code'], 0)
        self.assertFalse(result['contaminated'])
        self.assertFalse(result['valid_measurement'])
        self.broker.tick()
        self.assertEqual(self.broker.status(spec['job_id'])['state'], 'failed')
        self.assertEqual(len(self.launches), 1)

    def test_execution_timeout_starts_at_launch_not_queue_creation(self):
        spec, attempt = self.supervisor_fixture('import time;time.sleep(30)', timeout=.1)
        with self.broker.connect() as db:db.execute('UPDATE jobs SET created=? WHERE id=?', (time.time()-1000, spec['job_id']))
        result = worker.supervise(self.broker.root, spec['job_id'], 1,
                                  self.held.pop(spec['job_id']), probe=self.probe)
        self.assertTrue(result['timed_out'])
        self.assertFalse(result['contaminated'])

    def test_runtime_cancel_terminates_only_owned_processes(self):
        spec, attempt = self.supervisor_fixture('import time;time.sleep(30)')
        timer = threading.Timer(.15, lambda:self.broker.cancel(spec['job_id']))
        timer.start();self.addCleanup(timer.cancel)
        result = worker.supervise(self.broker.root, spec['job_id'], 1,
                                  self.held.pop(spec['job_id']), probe=self.probe)
        self.assertTrue(result['cancelled'])
        self.assertFalse(result['valid_measurement'])

    def test_validation_rejects_ambiguous_or_unbounded_selection(self):
        for options in [{}, {'gpu_ids': [True]}, {'gpu_ids': [0, 0]}, {'gpu_ids': [0], 'gpu_count': 2},
                        {'gpu_ids': [0], 'max_wait_s': float('inf')},
                        {'gpu_ids': [0], 'monitor_interval_s': 2}]:
            with self.subTest(options=options), self.assertRaises(ValueError):validate(options)
        with self.assertRaises(ValueError):validate_config({'pod': 'p', 'scheduling': {'gpu_ids': [0]}})


if __name__ == '__main__':unittest.main()
