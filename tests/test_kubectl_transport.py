import json
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

from aka.light.execution.kubectl import VERIFY_UID, main


class KubernetesTransportTests(unittest.TestCase):
    def test_identity_returns_only_the_requested_status(self):
        pod = {'metadata': {'uid': 'u'}, 'spec': {'ignored': 'private'},
               'status': {'containerStatuses': [{'name': 'worker'}]}}
        result = subprocess.CompletedProcess([], 0, json.dumps(pod).encode(), b'')
        with patch('subprocess.run', return_value=result) as run, patch('builtins.print') as emit:
            self.assertEqual(main(['identity', '--pod', 'example', '--namespace', 'research']), 0)
        self.assertEqual(run.call_args.args[0], ['kubectl', '--namespace', 'research', 'get', 'pod', 'example', '-o', 'json'])
        self.assertNotIn('spec', json.loads(emit.call_args.args[0]))

    def test_pinned_launch_checks_identity_inside_the_destination_container(self):
        with patch('os.execvp') as execute:
            main(['exec', '--pod', 'example', '--container', 'worker', '--uid', 'u',
                  '--', 'command', 'a b', '$(false)'])
        args = execute.call_args.args[1]
        self.assertEqual(args, ['kubectl', 'exec', 'example', '--container', 'worker', '--',
                                'python3', '-c', VERIFY_UID, 'u', 'command', 'a b', '$(false)'])
        command = [sys.executable, '-c', VERIFY_UID, 'u', sys.executable, '-c', 'print("started")']
        for actual in ('u', '', 'replacement'):
            proc = subprocess.run(command, capture_output=True, text=True, env={**os.environ, 'AKA_POD_UID': actual})
            self.assertEqual(proc.returncode == 0, actual == 'u')
            self.assertEqual(proc.stdout.strip(), 'started' if actual == 'u' else '')

    def test_target_cannot_be_an_option(self):
        with self.assertRaises(SystemExit):
            main(['exec', '--pod=--help', '--', 'true'])
