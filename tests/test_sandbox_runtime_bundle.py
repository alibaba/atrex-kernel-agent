from __future__ import annotations

import base64
import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from tools.sandbox import _make_atrex_bench_runtime_bundle


class SandboxRuntimeBundleTest(unittest.TestCase):
    def _runtime_workspace(self, root: Path, *, with_sdk: bool) -> Path:
        runtime = root / "runtime"
        package = runtime / "src" / "atrex_bench"
        (package / "eval").mkdir(parents=True)
        (runtime / "scripts").mkdir()
        (runtime / "scripts" / "run_eval.py").write_text("# evaluator\n")
        (package / "__init__.py").write_text("# package\n")
        (package / "utils.py").write_text("# utilities\n")
        (package / "eval" / "_runtime.py").write_text("# runtime\n")
        if with_sdk:
            (package / "sdk.py").write_text("SDK_MARKER = 'included'\n")

        workspace = root / "workspace"
        workspace.mkdir()
        (workspace / "atrex-bench").symlink_to(runtime, target_is_directory=True)
        return workspace

    def _members(self, bundle: str) -> dict[str, bytes]:
        payload = base64.b64decode(bundle, validate=True)
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
            return {
                member.name: archive.extractfile(member).read()
                for member in archive.getmembers()
                if member.isfile()
            }

    def test_evaluator_bundle_includes_sdk_when_available(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sandbox-runtime-sdk-") as temp:
            workspace = self._runtime_workspace(Path(temp), with_sdk=True)
            bundle = _make_atrex_bench_runtime_bundle(workspace, evaluator_only=True)

        self.assertIsNotNone(bundle)
        members = self._members(bundle)
        sdk_path = "atrex-bench/src/atrex_bench/sdk.py"
        self.assertEqual(members[sdk_path], b"SDK_MARKER = 'included'\n")

    def test_evaluator_bundle_remains_compatible_without_sdk(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sandbox-runtime-legacy-") as temp:
            workspace = self._runtime_workspace(Path(temp), with_sdk=False)
            bundle = _make_atrex_bench_runtime_bundle(workspace, evaluator_only=True)

        self.assertIsNotNone(bundle)
        members = self._members(bundle)
        self.assertNotIn("atrex-bench/src/atrex_bench/sdk.py", members)
        self.assertIn("atrex-bench/src/atrex_bench/__init__.py", members)
        self.assertIn("atrex-bench/src/atrex_bench/utils.py", members)
        self.assertIn("atrex-bench/src/atrex_bench/eval/_runtime.py", members)


if __name__ == "__main__":
    unittest.main()
