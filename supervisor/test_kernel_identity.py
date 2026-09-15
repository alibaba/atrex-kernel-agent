"""Global Kernel IDs without Episode-local identities or migration aliases."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from supervisor import gateway, journal
from supervisor.identifiers import kernel_id_for_digest


class GlobalKernelIdentityTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.source = b"class Model: pass\n"
        self.digest = gateway._kernel_artifact_digest(self.source)
        self.kernel_id = kernel_id_for_digest(self.digest)
        (self.workspace / "kernel.py").write_bytes(self.source)
        self.evidence = self.root / "current"
        self.history = self.root / "episodes"
        self.enterContext(
            patch.dict(
                os.environ,
                {
                    gateway.SUPERVISOR_EVIDENCE_ROOT_ENV: str(self.evidence),
                    gateway.SUPERVISOR_HISTORY_ROOT_ENV: str(self.history),
                },
            )
        )

    def record(self, evidence, *, source=None, kind="run", subjects=None):
        with patch.dict(os.environ, {gateway.SUPERVISOR_EVIDENCE_ROOT_ENV: str(evidence)}):
            return gateway._record_episode_evaluation(
                self.workspace,
                {"all_pass": True, "latency_us_by_shape": {"0": 5.0}},
                gateway_kind=kind,
                kernel_bytes=self.source if source is None else source,
                kernel_subjects=subjects,
            )

    def legacy_record(self, evidence, alias, *, source=None, kind="run", subjects=None):
        source = self.source if source is None else source
        record = self.record(evidence, source=source, kind=kind, subjects=subjects)
        digest = gateway._kernel_artifact_digest(source)
        identity = evidence / "kernel-artifacts" / "sha256" / digest[7:] / "identity.json"
        identity.write_text(json.dumps({"kernel_id": alias, "kernel_artifact_digest": digest}))
        result = evidence / "gateway-records" / record["record_id"] / "result.json"
        data = json.loads(result.read_text())
        data["kernel_id"] = alias
        if subjects:
            data["kernel_subject_ids"]["candidate"] = alias
        result.write_text(json.dumps(data))
        return record, {path: path.read_bytes() for path in (identity, result)}

    def test_namespace_and_name_are_stable_across_releases(self):
        self.assertEqual(self.kernel_id, "kernel-202dc5fc40695c08b3b65439d1dae71b")
        self.assertRegex(self.kernel_id, r"^kernel-[0-9a-f]{32}$")
        self.assertNotIn(self.digest[7:], self.kernel_id)
        with self.assertRaises(ValueError):
            kernel_id_for_digest("invalid")

    def test_identity_depends_on_exact_source_not_semantic_equivalence(self):
        identities = {
            gateway._store_kernel_artifact_at_root(self.evidence, source)["kernel_id"]
            for source in (self.source, self.source + b"\n", self.source + b"# comment\n")
        }
        self.assertEqual(len(identities), 3)

    def test_independent_processes_and_stores_generate_one_global_id(self):
        code = (
            "import json,sys; from pathlib import Path; from supervisor import gateway; "
            "print(json.dumps(gateway._store_kernel_artifact_at_root("
            "Path(sys.argv[1]), b'class Model: pass\\n')))"
        )

        def store(ordinal):
            # Two writers share a store; the others have unrelated Campaign directories.
            directory = self.root / f"campaign-{max(0, ordinal - 1)}"
            result = subprocess.run(
                [sys.executable, "-c", code, str(directory)],
                cwd=gateway.REPO_ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
            return json.loads(result.stdout)

        with ThreadPoolExecutor(max_workers=4) as executor:
            identities = list(executor.map(store, range(4)))
        self.assertEqual({item["kernel_id"] for item in identities}, {self.kernel_id})

    def test_all_operations_across_episodes_share_id_and_are_indexed(self):
        records = []
        for ordinal, kind in enumerate(("run", "profile", "dev", "check", "disassemble"), 1):
            evidence = self.history / f"e{ordinal:04d}" / "supervisor_runtime"
            record = self.record(evidence, kind=kind)
            self.assertEqual(record["kernel_id"], self.kernel_id)
            records.append(record)
        records.append(self.record(self.evidence))
        index = gateway._kernel_gateway_records(self.workspace, self.kernel_id)
        self.assertEqual(
            {item["gateway_record_id"] for item in index},
            {item["record_id"] for item in records},
        )
        self.assertEqual(
            gateway._kernel_artifact_bytes(self.workspace, self.kernel_id), self.source
        )

    def test_old_ids_and_bindings_are_rejected_without_conversion(self):
        old_id = "kernel-100-aaaaaaaaaaaa"
        record, original = self.legacy_record(self.evidence, old_id)
        with self.assertRaisesRegex(ValueError, "invalid format"):
            gateway._load_kernel_identity(self.workspace, old_id)
        with self.assertRaisesRegex(ValueError, "invalid Kernel ID"):
            gateway._load_gateway_record(self.workspace, record["record_id"])
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            gateway._store_kernel_artifact_at_root(self.evidence, self.source)
        for path, contents in original.items():
            self.assertEqual(path.read_bytes(), contents)

    def test_source_read_returns_global_id(self):
        self.record(self.evidence)
        with redirect_stdout(io.StringIO()) as output:
            gateway._read_kernel_source(self.workspace, self.kernel_id, "scratch/restored.py")
        self.assertEqual((self.workspace / "scratch/restored.py").read_bytes(), self.source)
        self.assertEqual(
            json.loads(output.getvalue().removeprefix(gateway.RECORD_RESULT_PREFIX))["kernel_id"],
            self.kernel_id,
        )

    def test_abba_binds_both_global_subjects_across_episodes(self):
        incumbent_source = b"class Model: baseline = True\n"
        incumbent_digest = gateway._kernel_artifact_digest(incumbent_source)
        incumbent_id = kernel_id_for_digest(incumbent_digest)
        archived = self.history / "e0001" / "supervisor_runtime"
        self.record(archived, source=incumbent_source)
        record = self.record(
            self.evidence,
            kind="same_allocation_abba",
            subjects={"incumbent": incumbent_digest[7:], "candidate": self.digest[7:]},
        )
        loaded = gateway._load_gateway_record(self.workspace, record["record_id"])
        self.assertEqual(
            loaded["kernel_subject_ids"],
            {
                "incumbent": incumbent_id,
                "candidate": self.kernel_id,
            },
        )
        self.assertEqual(
            gateway._kernel_gateway_records(self.workspace, self.kernel_id)[0]["role"], "candidate"
        )
        self.assertIn(
            {
                "gateway_record_id": record["record_id"],
                "operation": "same_allocation_abba",
                "status": "completed",
                "role": "incumbent",
            },
            gateway._kernel_gateway_records(self.workspace, incumbent_id),
        )

    def test_global_identity_does_not_grant_cross_campaign_access(self):
        foreign = self.record(self.root / "foreign-campaign")
        self.assertEqual(foreign["kernel_id"], self.kernel_id)
        with self.assertRaisesRegex(ValueError, "visible history"):
            gateway._load_kernel_identity(self.workspace, self.kernel_id)
        with self.assertRaisesRegex(ValueError, "visible history"):
            gateway._kernel_gateway_records(self.workspace, self.kernel_id)
        local = self.record(self.evidence)
        self.assertEqual(
            [
                item["gateway_record_id"]
                for item in gateway._kernel_gateway_records(self.workspace, self.kernel_id)
            ],
            [local["record_id"]],
        )

    def test_corrupt_identity_cannot_rebind_a_global_id(self):
        self.record(self.evidence)
        identity = self.evidence / "kernel-artifacts" / "sha256" / self.digest[7:] / "identity.json"
        identity.write_text(
            json.dumps(
                {
                    "kernel_id": kernel_id_for_digest(
                        gateway._kernel_artifact_digest(b"different")
                    ),
                    "kernel_artifact_digest": self.digest,
                }
            )
        )
        with self.assertRaisesRegex(ValueError, "visible history"):
            gateway._load_kernel_identity(self.workspace, self.kernel_id)
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            gateway._store_kernel_artifact_at_root(self.evidence, self.source)

    def test_source_digest_is_verified_on_read(self):
        self.record(self.evidence)
        source = self.evidence / "kernel-artifacts" / "sha256" / self.digest[7:] / "kernel.py"
        source.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "digest verification"):
            gateway._kernel_artifact_bytes(self.workspace, self.kernel_id)

    def test_journal_accepts_global_but_rejects_legacy_kernel_bindings(self):
        current = self.record(self.evidence)
        self.assertEqual(
            journal._validate_record_ids(self.evidence, self.root, [current["record_id"]]),
            [current["record_id"]],
        )
        old, _ = self.legacy_record(self.evidence, "kernel-100-aaaaaaaaaaaa")
        with self.assertRaisesRegex(ValueError, "real Kernel"):
            journal._validate_record_ids(self.evidence, self.root, [old["record_id"]])
