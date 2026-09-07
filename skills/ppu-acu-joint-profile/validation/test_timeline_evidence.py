from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

import sys

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from critical_path import CriticalPathError, _plan, analyze, validate_report
from evidence import sha256_file, validate_timeline_receipt
from timeline import ABI_MAJOR, ABI_MINOR, COMMITTED, HEADER, MAGIC, RECORD, decode


class TimelineEvidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.kernel = self.root / "kernel.py"
        self.instrumented = self.root / "kernel.instrumented.cu"
        self.binary = self.root / "kernel.so"
        self.workload = self.root / "workload.json"
        for path, content in (
            (self.kernel, "probe-free"),
            (self.instrumented, "instrumented"),
            (self.binary, "binary"),
            (self.workload, "workload"),
        ):
            path.write_text(content, encoding="utf-8")
        self.device = {"physical_device": 0, "serial": "ppu-0"}
        self.workload_identity = "rows=16,dtype=bf16"
        self.correctness = self.root / "correctness.json"
        self.correctness.write_text(
            json.dumps(
                {
                    "schema": "ppu-timeline-correctness/v2",
                    "validation": "accepted",
                    "kernel_name": "qsa_kernel",
                    "kernel_sha256": sha256_file(self.kernel),
                    "workload_identity": self.workload_identity,
                    "device_identity": self.device,
                    "checks": [{"name": "allclose", "status": "passed"}],
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_clean_reference_samples_come_from_artifact(self) -> None:
        identity = {"kernel_name": "qsa_kernel", "workload_identity": "rows=16"}
        measurement = self.root / "clean-measurement.json"
        measurement.write_text(
            json.dumps(
                {
                    "schema": "ppu-clean-measurement/v1",
                    "validation": "accepted",
                    "identity": identity,
                    "duration_ns_samples": [100.0, 110.0, 120.0],
                }
            ),
            encoding="utf-8",
        )
        plan = self.root / "clean-plan.json"
        plan.write_text(
            json.dumps(
                {
                    "schema": "ppu-critical-path-plan/v2",
                    "parent": {"site_id": 1, "name": "mainloop"},
                    "components": [],
                    "clean_reference": {
                        "source": "probe-free harness",
                        "identity": identity,
                        "artifact": {
                            "path": measurement.name,
                            "sha256": sha256_file(measurement),
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        self.assertEqual(_plan(plan)["clean_samples"], [100.0, 110.0, 120.0])

    def test_critical_path_output_cannot_overwrite_input(self) -> None:
        plan = self.root / "plan.json"
        plan.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(CriticalPathError, "must not overwrite"):
            analyze(plan, [(self.root / "capture.json", self.root / "receipt.json")], plan)

    def test_decode_binds_complete_artifact_graph(self) -> None:
        raw = self.root / "timeline.bin"
        raw.write_bytes(
            HEADER.pack(
                MAGIC,
                ABI_MAJOR,
                ABI_MINOR,
                HEADER.size,
                RECORD.size,
                2,
                1,
                2,
                0,
                1,
                1,
                1,
                32,
                1,
                1,
                17,
            )
            + RECORD.pack(100, 0, COMMITTED | 1)
            + RECORD.pack(200, 0, COMMITTED | (1 << 16) | 1)
            + struct.pack("<Q", (1 << 32) | 1)
        )
        manifest = self.root / "timeline.manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema": "ppu-fixed-slot-timeline-manifest/v5",
                    "backend": "ppu_fixed_slot",
                    "capture_mode": "coarse",
                    "sampling_rationale": "measure one complete phase",
                    "kernel_name": "qsa_kernel",
                    "kernel_duration_ns": 300,
                    "grid": [1, 1, 1],
                    "block": [32, 1, 1],
                    "launch_id": 17,
                    "records_per_owner": 2,
                    "timer": {"source": "globaltimer", "unit": "ns"},
                    "correctness_artifact": "correctness.json",
                    "runtime_identity": {"compiler": "hggc", "runtime": "sdk"},
                    "provenance": {
                        "evidence_grade": "decision",
                        "kernel_specialization": "block=32",
                        "cache_policy": "warm",
                        "clock_configuration": "locked",
                        "authoritative_kernel": {"path": "kernel.py", "identity": "probe-free kernel"},
                        "instrumented_sources": [{"path": "kernel.instrumented.cu", "identity": "instrumented source"}],
                        "compiled_binaries": [{"path": "kernel.so", "identity": "loaded binary"}],
                        "workload_inputs": [{"path": "workload.json", "identity": "input descriptor"}],
                    },
                    "clock_scope": "owner_local",
                    "owner_layout": {
                        "kind": "explicit_writers",
                        "owners": [{"owner": 0, "block": 0, "thread": 0, "label": "owner", "purpose": "phase timing"}],
                    },
                    "coverage": {"all_blocks": False},
                    "workload_identity": self.workload_identity,
                    "device_identity": self.device,
                }
            ),
            encoding="utf-8",
        )
        dictionary = self.root / "timeline.events.json"
        dictionary.write_text(
            json.dumps(
                {
                    "schema": "ppu-fixed-slot-events/v2",
                    "sites": [
                        {
                            "site_id": 1,
                            "name": "mainloop",
                            "kind": "range",
                            "role": "observation",
                            "boundary_semantics": "mainloop entry to exit",
                            "async_domain": "control",
                            "source_anchor": "kernel mainloop",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        prefix = self.root / "decoded"
        summary = decode(raw, manifest, dictionary, prefix)
        receipt_path = self.root / "decoded.receipt.json"
        receipt = validate_timeline_receipt(receipt_path, require_decision=True)
        self.assertEqual(summary["kernel_sha256"], sha256_file(self.kernel))
        self.assertEqual(receipt["kernel_sha256"], sha256_file(self.kernel))

        plan = self.root / "critical-path.plan.json"
        plan.write_text(
            json.dumps(
                {
                    "schema": "ppu-critical-path-plan/v2",
                    "owner_topology": "same",
                    "parent": {"site_id": 1, "name": "mainloop"},
                    "components": [],
                }
            ),
            encoding="utf-8",
        )
        critical_path = self.root / "critical-path.report.json"
        analyze(
            plan,
            [(self.root / "decoded.canonical.json", receipt_path)],
            critical_path,
        )
        self.assertEqual(
            validate_report(critical_path)["schema"],
            "ppu-critical-path-report/v3",
        )

        (self.root / "decoded.summary.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "content hash mismatch"):
            validate_timeline_receipt(receipt_path, require_decision=True)


if __name__ == "__main__":
    unittest.main()
