from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from long_horizon.journal import (
    normalize_accepted_ppu_diagnostics,
    validate_accepted_ppu_evidence,
)


class JournalProfileEvidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name)
        self.current = self.workspace / "current.json"
        self.calibration = self.workspace / "calibration.json"
        self.current.write_text("current", encoding="utf-8")
        self.calibration.write_text("calibration", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def descriptor(path: Path) -> dict:
        return {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }

    def document(self, *, include_kernel: bool) -> dict:
        kernel_sha256 = "a" * 64 if include_kernel else None
        binding_payload = {
            "kernel_sha256": kernel_sha256,
            "inputs": {
                "current": {
                    "sha256": hashlib.sha256(self.current.read_bytes()).hexdigest(),
                    "size_bytes": self.current.stat().st_size,
                },
                "calibration": {
                    "sha256": hashlib.sha256(self.calibration.read_bytes()).hexdigest(),
                    "size_bytes": self.calibration.stat().st_size,
                },
            },
        }
        return {
            "schema": "ppu-envelope-measurement/v1",
            "validation": "accepted",
            "evidence_grade": "decision",
            "evidence_id": hashlib.sha256(
                json.dumps(
                    binding_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest(),
            "kernel_sha256": kernel_sha256,
            "binding_payload": binding_payload,
            "inputs": {
                "current": self.descriptor(self.current),
                "calibration": self.descriptor(self.calibration),
            },
        }

    def row(self, artifact: Path, document: dict) -> dict:
        artifact.write_text(json.dumps(document), encoding="utf-8")
        return {
            "route": "envelope",
            "question": "How much measured compute headroom remains?",
            "kernel_specialization": "block_n=32",
            "workload_identity": "rows=16384,dtype=bf16",
            "device_identity": "ZW-M890P physical device 0",
            "launch_topology": "grid=16384x24, block=128",
            "control_pipeline_identity": "split-k partial plus merge",
            "finding": "measured compute headroom is below uncertainty",
            "decision_impact": "supports an independently reviewed ceiling claim",
            "evidence": {
                "artifact": artifact.relative_to(self.workspace).as_posix(),
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "schema": document["schema"],
                "evidence_id": document["evidence_id"],
            },
            "invalidation_conditions": ["kernel or workload changes"],
        }

    @mock.patch("long_horizon.journal.subprocess.run")
    def test_accepts_hash_bound_envelope(self, run: mock.Mock) -> None:
        run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
        artifact = self.workspace / "compute.envelope.json"
        document = self.document(include_kernel=True)
        row = self.row(artifact, document)
        normalized, errors = normalize_accepted_ppu_diagnostics([row])
        self.assertEqual(errors, [])
        self.assertEqual(validate_accepted_ppu_evidence(normalized, self.workspace), [])

    @mock.patch("long_horizon.journal.subprocess.run")
    def test_accepts_absolute_path_inside_workspace(self, run: mock.Mock) -> None:
        run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
        artifact = self.workspace / "compute.envelope.json"
        document = self.document(include_kernel=True)
        document["inputs"]["current"]["path"] = str(self.current.resolve())
        row = self.row(artifact, document)
        normalized, errors = normalize_accepted_ppu_diagnostics([row])
        self.assertEqual(errors, [])
        self.assertEqual(
            validate_accepted_ppu_evidence(normalized, self.workspace), []
        )

    @mock.patch("long_horizon.journal.subprocess.run")
    def test_rejects_new_evidence_without_kernel_hash(self, run: mock.Mock) -> None:
        run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
        artifact = self.workspace / "compute.envelope.json"
        document = self.document(include_kernel=False)
        row = self.row(artifact, document)
        normalized, errors = normalize_accepted_ppu_diagnostics([row])
        self.assertEqual(errors, [])
        self.assertIn(
            "accepted_ppu_diagnostics[0].evidence has no authoritative kernel_sha256",
            validate_accepted_ppu_evidence(normalized, self.workspace),
        )

    @mock.patch("long_horizon.journal.subprocess.run")
    def test_rejects_tampered_transitive_artifact(self, run: mock.Mock) -> None:
        run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
        artifact = self.workspace / "compute.envelope.json"
        row = self.row(artifact, self.document(include_kernel=True))
        self.current.write_text("tampered", encoding="utf-8")
        normalized, errors = normalize_accepted_ppu_diagnostics([row])
        self.assertEqual(errors, [])
        self.assertIn(
            "accepted_ppu_diagnostics[0].evidence.inputs.current content hash mismatch",
            validate_accepted_ppu_evidence(normalized, self.workspace),
        )

    @mock.patch("long_horizon.journal.subprocess.run")
    def test_rejects_fabricated_binding_id(self, run: mock.Mock) -> None:
        run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
        artifact = self.workspace / "compute.envelope.json"
        document = self.document(include_kernel=True)
        document["evidence_id"] = "fabricated"
        row = self.row(artifact, document)
        normalized, errors = normalize_accepted_ppu_diagnostics([row])
        self.assertEqual(errors, [])
        self.assertIn(
            "accepted_ppu_diagnostics[0].evidence binding_payload hash mismatch",
            validate_accepted_ppu_evidence(normalized, self.workspace),
        )

    @mock.patch("long_horizon.journal.subprocess.run")
    def test_external_descriptor_is_rejected_before_subprocess(
        self, run: mock.Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as external_directory:
            external = Path(external_directory) / "external.json"
            external.write_text("external", encoding="utf-8")
            artifact = self.workspace / "compute.envelope.json"
            document = self.document(include_kernel=True)
            external_descriptor = self.descriptor(external)
            external_descriptor["path"] = str(external.resolve())
            document["inputs"]["current"] = external_descriptor
            document["binding_payload"]["inputs"]["current"] = {
                "sha256": hashlib.sha256(external.read_bytes()).hexdigest(),
                "size_bytes": external.stat().st_size,
            }
            document["evidence_id"] = hashlib.sha256(
                json.dumps(
                    document["binding_payload"],
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            row = self.row(artifact, document)
            normalized, errors = normalize_accepted_ppu_diagnostics([row])
            self.assertEqual(errors, [])
            validation_errors = validate_accepted_ppu_evidence(
                normalized, self.workspace
            )
            self.assertTrue(
                any("outside the workspace" in error for error in validation_errors)
            )
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
