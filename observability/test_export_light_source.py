from __future__ import annotations

from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from observability.export_light_source import (
    LocatedSource,
    export_source,
    locate_from_status,
    locate_git_commit,
)


class ExportLightSourceTest(unittest.TestCase):
    def test_export_single_file_snapshot_as_directory(self):
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            output = root / "exported"
            result = export_source(
                LocatedSource(
                    b"def run(x):\n    return x\n",
                    {"kernel_id": "kernel-" + "a" * 32},
                    ("/campaign",),
                ),
                output,
            )
            self.assertEqual((output / "kernel.py").read_text(), "def run(x):\n    return x\n")
            self.assertEqual(result["source_kind"], "single_file")
            self.assertIn("kernel.py", result["files"])

    def test_export_refuses_existing_destination(self):
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            output = Path(temporary) / "existing"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "already exists"):
                export_source(LocatedSource(b"pass\n", {}, ()), output)

    def test_locate_git_commit_in_campaign_repository(self):
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            campaign = root / "campaign"
            campaign.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=campaign, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=campaign, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=campaign, check=True)
            (campaign / "kernel.py").write_text("def run(x):\n    return x\n")
            subprocess.run(["git", "add", "kernel.py"], cwd=campaign, check=True)
            subprocess.run(["git", "commit", "-q", "-m", "candidate"], cwd=campaign, check=True)
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=campaign, check=True,
                capture_output=True, text=True,
            ).stdout.strip()
            located = locate_git_commit(campaign, root / "runtime", commit)
        self.assertEqual(located.snapshot, b"def run(x):\n    return x\n")
        self.assertEqual(located.identity, {"git_commit": commit})

    def test_status_accepted_uses_git_commit(self):
        status = {"accepted_best": {"git_commit": "a" * 40}}
        with mock.patch("observability.export_light_source.locate_git_commit") as locate:
            locate.return_value = LocatedSource(b"pass\n", {}, ("/campaign",))
            result = locate_from_status(Path("/campaign"), Path("/runtime"), status, "accepted_best")
        self.assertEqual(result.snapshot, b"pass\n")
        locate.assert_called_once_with(Path("/campaign"), Path("/runtime"), "a" * 40)

    def test_status_eval_prefers_gateway_record(self):
        status = {"best_passing_full_eval": {
            "gateway_record_id": "gateway-" + "1" * 32,
            "kernel_id": "kernel-" + "2" * 32,
        }}
        with mock.patch("observability.export_light_source.locate_gateway_record") as locate:
            locate.return_value = LocatedSource(b"pass\n", {}, ("/campaign",))
            result = locate_from_status(
                Path("/campaign"), Path("/runtime"), status, "best_passing_full_eval"
            )
        self.assertEqual(result.snapshot, b"pass\n")
        locate.assert_called_once_with(
            Path("/campaign"), Path("/runtime"), "gateway-" + "1" * 32
        )


if __name__ == "__main__":
    unittest.main()
