"""The installed Agate CLI's device-list response is successful discovery."""
import contextlib
import io
import json
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from aka.light.evaluation import gateway


class EnvironmentCliTest(unittest.TestCase):
    def test_device_list_is_returned_once_without_infrastructure_retry(self):
        devices = [{'gpu': 'gpu-target-a', 'arch': 'sm_90'}, {'gpu': 'gpu-target-b', 'arch': 'sm_100'}]
        args = SimpleNamespace(ssh='', url='', gateway_profile='', env_gpu='',
                               env_capabilities=False, frameworks=False, counters='',
                               limits=False, env_force=False, timeout=10)
        result = subprocess.CompletedProcess([], 0, json.dumps(devices), '')
        output = io.StringIO()
        with patch.object(gateway, 'find_agate', return_value='/fixture/agate'), patch.object(
                gateway, '_resolved_gateway_url', return_value=''), patch(
                'aka.light.evaluation.agate_native.cli_arguments', return_value=[]), patch.object(
                gateway.subprocess, 'run', return_value=result) as execute, patch.object(
                gateway.time, 'sleep', side_effect=AssertionError('successful discovery retried')), contextlib.redirect_stdout(output):
            self.assertEqual(gateway._run_environment_query(args), 0)
        execute.assert_called_once()
        self.assertEqual(json.loads(output.getvalue().removeprefix(gateway.ENV_RESULT_PREFIX)), devices)


if __name__ == '__main__':
    unittest.main()
