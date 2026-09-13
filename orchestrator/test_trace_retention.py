from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from orchestrator.supervisor_runtime import supervisor_campaign_root
from orchestrator.trace_retention import (
    MANIFEST_NAME,
    collect_evidence_files,
    collect_private_evidence_files,
    write_trace_retention_manifest,
)


class TraceRetentionTest(unittest.TestCase):
    def test_promotion_audit_is_only_in_private_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "campaign"
            (workspace / "memory").mkdir(parents=True)
            (workspace / "memory/v1.json").write_text('{"version":"v1"}')
            (workspace / "memory/long_horizon_e0001.json").write_text('{"legacy":true}')
            root = supervisor_campaign_root(workspace)
            audits = root / "promotions"
            audits.mkdir(parents=True)
            (audits / "long_horizon_e0001.json").write_text('{"private":true}')
            (audits / "unrelated.json").write_text('{}')
            (audits / "long_horizon_e0002.json").symlink_to(workspace / "memory/v1.json")
            output = write_trace_retention_manifest(workspace, "completed")
            self.assertEqual(json.loads(output.read_text())["files"], [
                {"path": "memory/v1.json", "role": "canonical-memory"},
            ])
            self.assertEqual(json.loads((root / MANIFEST_NAME).read_text())["files"], [
                {"path": "promotions/long_horizon_e0001.json", "role": "promotion-audit"},
            ])

    def test_private_baseline_is_retained_without_wiki_queries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "campaign"
            workspace.mkdir()
            root = supervisor_campaign_root(workspace)
            root.mkdir(parents=True)
            (root / "framework_baseline.json").write_text('{"version":"v1"}')
            output = write_trace_retention_manifest(workspace, "completed")
            self.assertEqual(json.loads(output.read_text())["files"], [])
            private = json.loads((root / MANIFEST_NAME).read_text())
            self.assertEqual(private["files"], [{
                "path": "framework_baseline.json", "role": "framework-baseline-pin",
            }])

    def test_private_wiki_manifest_is_separate_and_survives_runtime_lifetime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve() / "campaign"
            workspace.mkdir()
            (workspace / "kernel.py").write_text("# candidate\n")
            # Old Agent-visible records are not treated as trusted private evidence.
            old_profile = workspace / ".gpu_wiki_profile"
            old_profile.mkdir()
            (old_profile / "run.json").write_text('{"untrusted":true}')
            root = supervisor_campaign_root(workspace)
            profile = root / "wiki-profile"
            events = profile / "raw" / "query_events" / "2026-09-12"
            events.mkdir(parents=True)
            (profile / "run.json").write_text('{"run_id":"test"}')
            event = events / "event.json"
            event.write_text('{"query_id":"query-1","request":"original query"}')
            (profile / "unrelated.txt").write_text("do not archive")
            expected_paths = [
                "wiki-profile/run.json",
                "wiki-profile/raw/query_events/2026-09-12/event.json",
            ]
            hardware = {"platform": "B200", "arch": "sm_100", "sandbox_hardware": "test"}
            for status in ("completed", "interrupted", "failed"):
                with self.subTest(status=status):
                    # No active Supervisor Runtime is needed by the completion hook.
                    output = write_trace_retention_manifest(workspace, status, hardware=hardware)
                    self.assertEqual(output, workspace / MANIFEST_NAME)
                    public = json.loads(output.read_text())
                    private = json.loads((root / MANIFEST_NAME).read_text())
                    self.assertEqual(
                        public["files"], [{"path": "kernel.py", "role": "extraction-core"}]
                    )
                    self.assertNotIn(str(root), output.read_text())
                    self.assertNotIn("wiki-profile", output.read_text())
                    self.assertEqual([row["path"] for row in private["files"]], expected_paths)
                    self.assertEqual([row["role"] for row in private["files"]],
                                     ["wiki-run-identity", "wiki-query-event"])
                    self.assertEqual(private["status"], status)
                    self.assertEqual(private["hardware"], hardware)
                    self.assertTrue(all((root / row["path"]).is_file() for row in private["files"]))
            self.assertEqual(json.loads(event.read_text())["request"], "original query")
            self.assertEqual(list(old_profile.iterdir()), [old_profile / "run.json"])
            self.assertFalse((workspace / "wiki-profile").exists())

    def test_private_collection_rejects_symlinks_outside_private_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "private"
            profile = root / "wiki-profile"
            events = profile / "raw" / "query_events"
            events.mkdir(parents=True)
            outside = Path(directory) / "outside"
            outside.mkdir()
            (outside / "event.json").write_text("{}")
            (profile / "run.json").symlink_to(outside / "event.json")
            (events / "2026-09-12").symlink_to(outside, target_is_directory=True)
            self.assertEqual(collect_private_evidence_files(root), [])
            other_root = Path(directory) / "other-private"
            other_root.mkdir()
            (other_root / "wiki-profile").symlink_to(profile, target_is_directory=True)
            self.assertEqual(collect_private_evidence_files(other_root), [])

    def test_private_manifest_failure_does_not_block_workspace_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            with (
                patch("orchestrator.trace_retention.supervisor_campaign_root",
                      side_effect=OSError("private storage unavailable")),
                patch("sys.stderr", io.StringIO()) as stderr,
            ):
                output = write_trace_retention_manifest(workspace, "failed")
            self.assertIsNotNone(output)
            self.assertIn("private trace retention manifest", stderr.getvalue())
            self.assertEqual(collect_evidence_files(workspace), [])

    def test_workspace_collection_failure_does_not_lose_private_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            workspace.mkdir()
            with (
                patch("orchestrator.trace_retention.collect_evidence_files",
                      side_effect=OSError("workspace unavailable")),
                patch("sys.stderr", io.StringIO()) as stderr,
            ):
                self.assertIsNone(write_trace_retention_manifest(workspace, "failed"))
            self.assertIn("workspace evidence could not be collected", stderr.getvalue())
            self.assertTrue((supervisor_campaign_root(workspace) / MANIFEST_NAME).is_file())


if __name__ == "__main__":
    unittest.main()
