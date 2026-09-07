from __future__ import annotations

import json
import os
import struct
import tempfile
import unittest
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS))

from acu_report import export
from evidence import (
    acu_binding_payload,
    digest_json,
    portable_descriptors,
    sha256_file,
    validate_acu_metadata,
)
from long_horizon.journal import (
    normalize_accepted_ppu_diagnostics,
    validate_accepted_ppu_evidence,
)
from profile_report import (
    build_envelope,
    compare_acu,
    comparison_payload,
    seal_calibration,
    validate_calibration,
    validate_comparison,
    validate_profile_artifact,
)


def _varint(value: int) -> bytes:
    result = bytearray()
    while value >= 0x80:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value)
    return bytes(result)


def _field(number: int, wire_type: int, value: bytes | int) -> bytes:
    tag = _varint((number << 3) | wire_type)
    if wire_type == 0:
        return tag + _varint(int(value))
    if wire_type == 1:
        return tag + bytes(value)
    payload = bytes(value)
    return tag + _varint(len(payload)) + payload


def _acu_report(metric: float, duration_ns: float) -> bytes:
    metric_name = b"cu__inst_executed.avg.pct_of_peak_sustained_elapsed"
    interval = int(duration_ns) // 10
    samples = b"".join(
        _field(
            3,
            2,
            _field(1, 0, index * interval)
            + _field(2, 1, struct.pack("<d", metric))
            + _field(3, 0, (index + 1) * interval),
        )
        for index in range(10)
    )
    metric_message = _field(1, 2, metric_name) + samples
    payload = _field(3, 2, metric_message)
    packet_value = _field(15, 2, payload)
    trace_packet = _field(88, 2, packet_value)
    return _field(1, 2, trace_packet)


class ProfileReportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.identity = {
            "kernel_name": "qsa_kernel",
            "kernel_specialization": "block_n=32",
            "workload_identity": "rows=16384,dtype=bf16",
            "device_identity": {"physical_device": 0, "serial": "ppu-0"},
            "runtime_identity": {"compiler": "hggc", "runtime": "sdk"},
            "cache_policy": "warm",
            "clock_configuration": "locked",
        }

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def descriptor(path: Path, identity: str | None = None) -> dict:
        value = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        if identity is not None:
            value["identity"] = identity
        return value

    def acu_receipt(
        self,
        name: str,
        *,
        duration_ns: float,
        metric: float,
        kernel_text: str,
        identity: dict | None = None,
    ) -> Path:
        directory = self.root / name
        directory.mkdir()
        effective_identity = dict(identity or self.identity)
        files = {}
        for filename, content in {
            "acu-version.txt": "acu 2.2",
            "kernel.py": kernel_text,
            "kernel.so": "binary",
            "workload.json": "workload",
        }.items():
            path = directory / filename
            path.write_text(content, encoding="utf-8")
            files[filename] = path
        files["profile.acurep"] = directory / "profile.acurep"
        files["profile.acurep"].write_bytes(_acu_report(metric, duration_ns))
        files["profile.collection.json"] = directory / "profile.collection.json"
        interval_ns = int(duration_ns) // 10
        files["profile.raw.csv"] = directory / "profile.raw.csv"
        files["profile.raw.csv"].write_text(
            "Kernel Name,Grid Size,Block Size,Device,ppu__time_duration.sum,"
            "pmsampler__interval_time.max,pmsampler__dropped_samples.max,"
            "pmsampler__buffer_size_bytes.max,device__attribute_cu_count,"
            "launch__occupancy_blocks_per_cu,launch__registers_per_thread,"
            "launch__shared_mem_per_block\n"
            f'qsa_kernel,"(1,1,1)","(32,1,1)",0,{duration_ns},'
            f"{interval_ns},0,1024,64,2,248,0\n",
            encoding="utf-8",
        )
        files["profile.samples.csv"] = directory / "profile.samples.csv"
        pm_rows = "".join(
            "0,cu__inst_executed.avg.pct_of_peak_sustained_elapsed,compute,"
            "percent,device_global_aggregate,"
            f"{index},valid,{index * interval_ns},{(index + 1) * interval_ns},"
            f"{interval_ns},{metric}\n"
            for index in range(10)
        )
        files["profile.samples.csv"].write_text(
            "packet_index,metric_name,logical_metric_group,metric_unit,scope,"
            "sample_index,validity,window_start_ns,window_end_ns,interval_ns,"
            "metric_value\n"
            + pm_rows,
            encoding="utf-8",
        )
        kernel_sha256 = sha256_file(files["kernel.py"])
        correctness_path = directory / "correctness.json"
        correctness_path.write_text(
            json.dumps(
                {
                    "schema": "ppu-profile-correctness/v1",
                    "validation": "accepted",
                    "kernel_sha256": kernel_sha256,
                    "workload_identity": effective_identity["workload_identity"],
                    "device_identity": effective_identity["device_identity"],
                    "checks": [{"name": "allclose", "status": "passed"}],
                }
            ),
            encoding="utf-8",
        )
        files["profile.collection.json"].write_text(
            json.dumps(
                {
                    "schema": "ppu-acu-collection/v2",
                    "producer": {"name": "acu", "version": "2.2"},
                    "producer_artifact": {
                        "path": "acu-version.txt",
                        "identity": "ACU version output",
                    },
                    "evidence_grade": "decision",
                    "report": "profile.acurep",
                    "authoritative_kernel": {
                        "path": "kernel.py",
                        "identity": "probe-free kernel",
                    },
                    "correctness_artifact": {
                        "path": "correctness.json",
                        "identity": "full correctness",
                    },
                    **effective_identity,
                    "requested_metrics": [
                        "cu__inst_executed.avg.pct_of_peak_sustained_elapsed"
                    ],
                    "source_artifacts": [
                        {"path": "kernel.py", "identity": "clean source"}
                    ],
                    "binary_artifacts": [
                        {"path": "kernel.so", "identity": "loaded binary"}
                    ],
                    "workload_inputs": [
                        {"path": "workload.json", "identity": "input descriptor"}
                    ],
                }
            ),
            encoding="utf-8",
        )
        receipt_path = directory / "profile.extract.json"
        receipt = {
            "schema": "ppu-acu-extraction/v4",
            "validation": {"status": "accepted", "errors": [], "warnings": [], "notes": []},
            "evidence_grade": "decision",
            "producer": {"name": "acu", "version": "2.2"},
            "kernel_sha256": kernel_sha256,
            "identity": effective_identity,
            "inputs": {
                "report": self.descriptor(files["profile.acurep"]),
                "raw_csv": self.descriptor(files["profile.raw.csv"]),
                "collection": self.descriptor(files["profile.collection.json"]),
            },
            "bound_artifacts": {
                "source_artifacts": [self.descriptor(files["kernel.py"], "clean source")],
                "binary_artifacts": [self.descriptor(files["kernel.so"], "loaded binary")],
                "workload_inputs": [
                    self.descriptor(files["workload.json"], "input descriptor")
                ],
                "producer": self.descriptor(
                    files["acu-version.txt"], "ACU version output"
                ),
                "authoritative_kernel": self.descriptor(files["kernel.py"], "probe-free kernel"),
                "correctness": self.descriptor(correctness_path, "full correctness"),
            },
            "outputs": {"pm_csv": self.descriptor(files["profile.samples.csv"])},
            "launch": {
                "kernel_name": "qsa_kernel",
                "grid": [1, 1, 1],
                "block": [32, 1, 1],
                "device": 0,
                "duration_ns": duration_ns,
                "pm_interval_ns": float(interval_ns),
                "dropped_samples": 0.0,
                "buffer_size_bytes": 1024.0,
                "cu_count": 64,
                "occupancy_blocks_per_cu": 2.0,
                "registers_per_thread": 248,
                "shared_mem_per_block": 0,
            },
            "metric_summaries": {
                "packet_0:cu__inst_executed.avg.pct_of_peak_sustained_elapsed": {
                    "packet_index": 0,
                    "metric_name": "cu__inst_executed.avg.pct_of_peak_sustained_elapsed",
                    "logical_metric_group": "compute",
                    "scope": "device_global_aggregate",
                    "unit": "percent",
                    "sample_count": 10,
                    "valid_sample_count": 10,
                    "excluded_sample_count": 0,
                    "validity_counts": {"valid": 10},
                    "coverage_end_ns": duration_ns,
                    "coverage_ratio": 1.0,
                    "interval_min_ns": float(interval_ns),
                    "interval_max_ns": float(interval_ns),
                    "time_weighted_mean": metric,
                    "min": metric,
                    "max": metric,
                }
            },
        }
        receipt["inputs"] = portable_descriptors(receipt["inputs"], receipt_path)
        receipt["bound_artifacts"] = portable_descriptors(
            receipt["bound_artifacts"], receipt_path
        )
        receipt["outputs"] = portable_descriptors(receipt["outputs"], receipt_path)
        receipt["binding_payload"] = acu_binding_payload(receipt)
        receipt["evidence_id"] = digest_json(receipt["binding_payload"])
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        return receipt_path

    def test_public_acu_exporter_emits_valid_receipt(self) -> None:
        self.acu_receipt(
            "export", duration_ns=200.0, metric=60.0, kernel_text="candidate"
        )
        directory = self.root / "export"
        metadata_path = directory / "exported.extract.json"
        metadata = export(
            directory / "profile.acurep",
            directory / "profile.raw.csv",
            directory / "profile.collection.json",
            directory / "exported.samples.csv",
            metadata_path,
        )
        self.assertEqual(metadata["schema"], "ppu-acu-extraction/v4")
        self.assertEqual(metadata["validation"]["status"], "accepted")
        self.assertEqual(
            validate_profile_artifact(metadata_path)["kernel_sha256"],
            sha256_file(directory / "kernel.py"),
        )
        with self.assertRaisesRegex(RuntimeError, "must not overwrite"):
            export(
                directory / "profile.acurep",
                directory / "profile.raw.csv",
                directory / "profile.collection.json",
                directory / "kernel.py",
                directory / "other.extract.json",
            )

    def test_compare_and_build_envelope(self) -> None:
        incumbent = self.acu_receipt(
            "incumbent", duration_ns=200.0, metric=60.0, kernel_text="baseline"
        )
        candidate = self.acu_receipt(
            "candidate", duration_ns=160.0, metric=80.0, kernel_text="candidate"
        )
        comparison_path = self.root / "comparison.json"
        comparison = compare_acu(incumbent, candidate, comparison_path)
        self.assertEqual(comparison["launch"]["duration_ns"]["speedup"], 1.25)
        self.assertEqual(
            validate_comparison(comparison_path)["candidate"]["kernel_sha256"],
            sha256_file(self.root / "candidate" / "kernel.py"),
        )

        calibration_raw = self.root / "calibration.raw.json"
        calibration_raw.write_text('{"compute_peak": 100.0}', encoding="utf-8")
        calibration_spec = self.root / "calibration.spec.json"
        calibration_spec.write_text(
            json.dumps(
                {
                    "schema": "ppu-calibration-spec/v1",
                    "identity": {
                        field: self.identity[field]
                        for field in (
                            "device_identity",
                            "runtime_identity",
                            "cache_policy",
                            "clock_configuration",
                        )
                    },
                    "measurements": {
                        "compute_peak": {
                            "kind": "compute",
                            "unit": "percent",
                            "direction": "higher_is_better",
                            "source_artifact": 0,
                            "json_pointer": "/compute_peak",
                        }
                    },
                    "source_artifacts": [
                        {"path": "calibration.raw.json", "identity": "measured compute calibration"}
                    ],
                }
            ),
            encoding="utf-8",
        )
        calibration_path = self.root / "calibration.json"
        seal_calibration(calibration_spec, calibration_path)
        envelope_path = self.root / "envelope.json"
        envelope = build_envelope(
            candidate,
            calibration_path,
            "compute",
            "/metric_summaries/packet_0:cu__inst_executed.avg.pct_of_peak_sustained_elapsed/time_weighted_mean",
            "compute_peak",
            envelope_path,
        )
        self.assertEqual(
            envelope["kernel_sha256"],
            sha256_file(self.root / "candidate" / "kernel.py"),
        )
        self.assertEqual(envelope["headroom_pct"], 20.0)
        self.assertEqual(
            validate_profile_artifact(envelope_path)["schema"],
            "ppu-envelope-measurement/v1",
        )
        diagnostic = {
            "route": "envelope",
            "question": "How much measured compute headroom remains?",
            "kernel_specialization": "block_n=32",
            "workload_identity": self.identity["workload_identity"],
            "device_identity": "ZW-M890P physical device 0",
            "launch_topology": "grid=1, block=32",
            "control_pipeline_identity": "single kernel",
            "finding": "20 percent measured headroom remains",
            "decision_impact": "continue optimization",
            "evidence": {
                "artifact": envelope_path.relative_to(self.root).as_posix(),
                "sha256": sha256_file(envelope_path),
                "schema": envelope["schema"],
                "evidence_id": envelope["evidence_id"],
            },
            "invalidation_conditions": ["kernel or workload changes"],
        }
        normalized, errors = normalize_accepted_ppu_diagnostics([diagnostic])
        self.assertEqual(errors, [])
        self.assertEqual(
            validate_accepted_ppu_evidence(normalized, self.root), []
        )

    def test_comparison_rejects_identity_drift(self) -> None:
        incumbent = self.acu_receipt(
            "incumbent", duration_ns=200.0, metric=60.0, kernel_text="baseline"
        )
        changed_identity = dict(self.identity)
        changed_identity["cache_policy"] = "cold"
        candidate = self.acu_receipt(
            "candidate",
            duration_ns=160.0,
            metric=80.0,
            kernel_text="candidate",
            identity=changed_identity,
        )
        with self.assertRaisesRegex(RuntimeError, "cache_policy"):
            compare_acu(incumbent, candidate, self.root / "comparison.json")

    def test_receipt_rejects_tampered_kernel(self) -> None:
        receipt = self.acu_receipt(
            "candidate", duration_ns=160.0, metric=80.0, kernel_text="candidate"
        )
        (self.root / "candidate" / "kernel.py").write_text("tampered", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "content hash mismatch"):
            validate_acu_metadata(receipt)

    def test_calibration_rejects_tampered_source(self) -> None:
        raw = self.root / "calibration.raw.json"
        raw.write_text('{"launch_floor": 10.0}', encoding="utf-8")
        spec = self.root / "calibration.spec.json"
        spec.write_text(
            json.dumps(
                {
                    "schema": "ppu-calibration-spec/v1",
                    "identity": {
                        field: self.identity[field]
                        for field in (
                            "device_identity",
                            "runtime_identity",
                            "cache_policy",
                            "clock_configuration",
                        )
                    },
                    "measurements": {
                        "launch_floor": {
                            "kind": "launch_merge",
                            "unit": "us",
                            "direction": "lower_is_better",
                            "source_artifact": 0,
                            "json_pointer": "/launch_floor",
                        }
                    },
                    "source_artifacts": [
                        {"path": "calibration.raw.json", "identity": "launch calibration"}
                    ],
                }
            ),
            encoding="utf-8",
        )
        receipt = self.root / "calibration.json"
        seal_calibration(spec, receipt)
        raw.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "content hash mismatch"):
            validate_calibration(receipt)

    def test_missing_descriptor_path_cannot_bypass_validation(self) -> None:
        receipt_path = self.acu_receipt(
            "candidate", duration_ns=160.0, metric=80.0, kernel_text="candidate"
        )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        del receipt["bound_artifacts"]["authoritative_kernel"]["path"]
        receipt["evidence_id"] = digest_json(acu_binding_payload(receipt))
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "incomplete artifact descriptor"):
            validate_acu_metadata(receipt_path)

    def test_acu_receipt_recomputes_derived_values(self) -> None:
        receipt_path = self.acu_receipt(
            "candidate", duration_ns=160.0, metric=80.0, kernel_text="candidate"
        )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["launch"]["duration_ns"] = 1.0
        receipt["evidence_id"] = digest_json(acu_binding_payload(receipt))
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "does not match raw CSV"):
            validate_acu_metadata(receipt_path)

    def test_comparison_recomputes_derived_values(self) -> None:
        incumbent = self.acu_receipt(
            "incumbent", duration_ns=200.0, metric=60.0, kernel_text="baseline"
        )
        candidate = self.acu_receipt(
            "candidate", duration_ns=160.0, metric=80.0, kernel_text="candidate"
        )
        output = self.root / "comparison.json"
        compare_acu(incumbent, candidate, output)
        report = json.loads(output.read_text(encoding="utf-8"))
        report["launch"]["duration_ns"]["speedup"] = 99.0
        report["evidence_id"] = digest_json(comparison_payload(report))
        output.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "launch deltas"):
            validate_comparison(output)

    def test_zero_metric_is_comparable(self) -> None:
        incumbent = self.acu_receipt(
            "incumbent", duration_ns=200.0, metric=0.0, kernel_text="baseline"
        )
        candidate = self.acu_receipt(
            "candidate", duration_ns=160.0, metric=10.0, kernel_text="candidate"
        )
        report = compare_acu(incumbent, candidate, self.root / "comparison.json")
        metric = report["metrics"]["packet_0:cu__inst_executed.avg.pct_of_peak_sustained_elapsed"]
        self.assertTrue(metric["comparable"])
        self.assertIsNone(metric["relative_delta"])

    def test_envelope_rejects_unit_mismatch(self) -> None:
        candidate = self.acu_receipt(
            "candidate", duration_ns=160.0, metric=80.0, kernel_text="candidate"
        )
        raw = self.root / "calibration.raw.json"
        raw.write_text('{"compute_peak": 100.0}', encoding="utf-8")
        spec = self.root / "calibration.spec.json"
        spec.write_text(
            json.dumps(
                {
                    "schema": "ppu-calibration-spec/v1",
                    "identity": {
                        field: self.identity[field]
                        for field in (
                            "device_identity",
                            "runtime_identity",
                            "cache_policy",
                            "clock_configuration",
                        )
                    },
                    "measurements": {
                        "compute_peak": {
                            "kind": "compute",
                            "unit": "GB/s",
                            "direction": "higher_is_better",
                            "source_artifact": 0,
                            "json_pointer": "/compute_peak",
                        }
                    },
                    "source_artifacts": [
                        {"path": "calibration.raw.json", "identity": "compute calibration"}
                    ],
                }
            ),
            encoding="utf-8",
        )
        calibration = self.root / "calibration.json"
        seal_calibration(spec, calibration)
        with self.assertRaisesRegex(RuntimeError, "units do not match"):
            build_envelope(
                candidate,
                calibration,
                "compute",
                "/metric_summaries/packet_0:cu__inst_executed.avg.pct_of_peak_sustained_elapsed/time_weighted_mean",
                "compute_peak",
                self.root / "envelope.json",
            )

    def test_output_cannot_overwrite_transitive_input(self) -> None:
        incumbent = self.acu_receipt(
            "incumbent", duration_ns=200.0, metric=60.0, kernel_text="baseline"
        )
        candidate = self.acu_receipt(
            "candidate", duration_ns=160.0, metric=80.0, kernel_text="candidate"
        )
        candidate_kernel = self.root / "candidate" / "kernel.py"
        with self.assertRaisesRegex(RuntimeError, "must not overwrite"):
            compare_acu(incumbent, candidate, candidate_kernel)
        hard_link = self.root / "candidate-kernel-hardlink"
        os.link(candidate_kernel, hard_link)
        with self.assertRaisesRegex(RuntimeError, "must not overwrite"):
            compare_acu(incumbent, candidate, hard_link)
        validate_acu_metadata(candidate)


if __name__ == "__main__":
    unittest.main()
