"""Exercise worker Python selection with real, distinct virtual environments."""
import json
import os
import shlex
import subprocess
import tempfile
import unittest
import venv
from pathlib import Path
from unittest import mock

import local_gateway


class WorkerVirtualEnvironmentTests(unittest.TestCase):
    def test_symlinked_gateway_python_keeps_venv_packages_and_explicit_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            base = root / "base"
            active = root / "active"
            venv.EnvBuilder(with_pip=False, symlinks=False).create(base)
            venv.EnvBuilder(with_pip=False, symlinks=False).create(active)
            python = active / "bin/python"
            python.unlink()
            python.symlink_to(base / "bin/python")
            site = Path(subprocess.check_output(
                [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True).strip())
            (site / "gateway_venv_only.py").write_text("VALUE = 'active-only'\n")
            # Prove that following the symlink selects a different environment.
            self.assertEqual(Path(subprocess.check_output(
                [str(python), "-c", "import sys; print(sys.prefix)"], text=True).strip()), active)
            self.assertEqual(Path(subprocess.check_output(
                [str(python.resolve()), "-c", "import sys; print(sys.prefix)"], text=True).strip()), base)
            gateway = local_gateway.LocalGateway(root / "state")
            try:
                inherited_path = str(base / "bin") + os.pathsep + os.defpath
                for explicit in (False, True):
                    with self.subTest(explicit_path=explicit):
                        script = "import json,sys; print(json.dumps({'prefix':sys.prefix}))"
                        if not explicit:
                            script = "import gateway_venv_only; " + script
                        request = {"command": shlex.join(["python", "-c", script]), "files": {},
                                   "timeout_s": 15, "env_vars": {"PATH": inherited_path} if explicit else {}}
                        job, _ = gateway.store.create("dev", request, "venv-regression")
                        job_id, kind, claimed = gateway.store.claim_next()
                        with mock.patch.object(local_gateway.sys, "executable", str(python)), \
                                mock.patch.dict(os.environ, {"PATH": inherited_path}):
                            gateway.scheduler._execute(job_id, kind, claimed)
                        result = gateway.store.get(job["job_id"])
                        self.assertEqual(result["status"], "succeeded", result)
                        self.assertEqual(Path(json.loads(result["result"]["stdout"])["prefix"]), base if explicit else active)
            finally:
                gateway.close()


if __name__ == "__main__":
    unittest.main()
