#!/usr/bin/env python3
from __future__ import annotations

import gzip
import importlib.util
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SKILL = Path(__file__).resolve().parents[1]
BACKEND = SKILL / "backends" / "cuda_backend"
REPOSITORY = SKILL.parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(REPOSITORY))
import adapter  # noqa: E402
from long_horizon.git_episode import EpisodeWorktree  # noqa: E402
from long_horizon.store import CampaignStore  # noqa: E402

SPEC = importlib.util.spec_from_file_location("timeline_tool", Path(__file__).with_name("timeline.py"))
assert SPEC and SPEC.loader
timeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(timeline)

IKET_SPEC = importlib.util.spec_from_file_location(
    "iket_adapter_test", BACKEND.parent / "cutedsl_backend" / "adapter.py"
)
assert IKET_SPEC and IKET_SPEC.loader
iket_adapter = importlib.util.module_from_spec(IKET_SPEC)
IKET_SPEC.loader.exec_module(iket_adapter)


def tag(site: int, kind: int, sm: int, flags: int = 0) -> int:
    return site | (kind << 16) | (sm << 18) | (((flags & 7) | 8) << 28)


class TimelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="atrex-timeline-")
        self.root = Path(self.temporary.name)
        self.grid = (2, 1, 1)
        self.block = (128, 1, 1)
        self.writers = [
            {"ordinal": 0, "label": "producer", "warp": 0, "group": 0},
            {"ordinal": 1, "label": "consumer", "warp": 2, "group": 1},
        ]
        self.owners = 4
        self.records_per_owner = 4
        header = adapter.make_header(
            owner_count=self.owners,
            records_per_owner=self.records_per_owner,
            grid=self.grid,
            block=self.block,
            launch_id=71,
        )
        self.raw = bytearray(
            header
            + bytes(self.owners * self.records_per_owner * adapter.RECORD_BYTES)
            + bytes(self.owners * adapter.CLAIM_BYTES)
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def put(self, owner: int, sequence: int, timestamp: int, site: int, kind: int, sm: int, payload=0):
        slot = owner * self.records_per_owner + sequence
        offset = adapter.HEADER_STRUCT.size + slot * adapter.RECORD_BYTES
        struct.pack_into("<QII", self.raw, offset, timestamp, payload, tag(site, kind, sm))
        cta = owner // len(self.writers)
        writer = self.writers[owner % len(self.writers)]
        thread = int(writer.get("warp", 0)) * 32 + int(writer.get("lane", 0))
        claims = adapter.HEADER_STRUCT.size + self.owners * self.records_per_owner * 16
        struct.pack_into("<Q", self.raw, claims + owner * 8, ((cta + 1) << 32) | (thread + 1))

    def files(self) -> tuple[Path, Path, Path, Path, Path]:
        raw_path = self.root / "raw.bin"
        manifest_path = self.root / "manifest.json"
        dictionary_path = self.root / "events.json"
        clean = self.root / "clean.py"
        instrumented = self.root / "instrumented.py"
        raw_path.write_bytes(self.raw)
        manifest_path.write_text(
            json.dumps(
                {
                    "schema": timeline.MANIFEST_SCHEMA,
                    "backend": "cuda",
                    "launch_id": 71,
                    "grid": list(self.grid),
                    "block": list(self.block),
                    "records_per_owner": self.records_per_owner,
                    "owner_layout": {"kind": "cta_writers", "writers": self.writers},
                }
            ),
            encoding="utf-8",
        )
        dictionary_path.write_text(
            json.dumps(
                {
                    "schema": timeline.DICTIONARY_SCHEMA,
                    "sites": [
                        {
                            "site_id": 5,
                            "name": "load",
                            "kind": "range",
                            "boundary_semantics": "software_region",
                            "source_anchor": "kernel.py:10",
                        },
                        {
                            "site_id": 8,
                            "name": "iteration",
                            "kind": "instant",
                            "boundary_semantics": "mark",
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        clean.write_text("kernel = 'clean'\n", encoding="utf-8")
        instrumented.write_text("kernel = 'instrumented'\n", encoding="utf-8")
        return raw_path, manifest_path, dictionary_path, clean, instrumented

    def evaluator_evidence(self, source: Path, *, all_pass: bool = True) -> Path:
        path = self.root / "evaluations.jsonl"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "gateway_kind": "run",
                    "kernel_sha256": timeline.digest(source),
                    "result": {
                        "all_pass": all_pass,
                        "failures": [] if all_pass else ["fixture correctness failure"],
                        "max_abs_err": 0.0,
                        "max_rel_err": 0.0,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def test_header_abi_is_exactly_64_plus_16_byte_records(self) -> None:
        self.assertEqual(adapter.HEADER_STRUCT.size, 64)
        self.assertEqual(adapter.RECORD_BYTES, 16)
        self.assertEqual(len(self.raw), 64 + 4 * 4 * 16 + 4 * 8)
        parsed = timeline.parse_header(bytes(self.raw))
        self.assertEqual(parsed["capacity"], 16)

    def test_evaluator_correctness_must_match_the_instrumented_source(self) -> None:
        source = self.root / "kernel.py"
        source.write_text("kernel = 'instrumented'\n", encoding="utf-8")
        evidence = self.evaluator_evidence(source)
        selected = timeline.load_evaluator_correctness(evidence, source)
        self.assertTrue(selected["result"]["all_pass"])
        source.write_text("kernel = 'changed_after_evaluation'\n", encoding="utf-8")
        with self.assertRaisesRegex(timeline.TimelineError, "no result for the instrumented source"):
            timeline.load_evaluator_correctness(evidence, source)

    def test_embedded_header_compiles_with_nvrtc_enabled_and_disabled(self) -> None:
        try:
            from cuda.bindings import nvrtc
        except ImportError:
            self.skipTest("cuda.bindings.nvrtc is unavailable")
        body = r'''
extern "C" __global__ void probe(void* storage) {
  bool selected = threadIdx.x == 0;
  atrex::timeline::Recorder recorder(storage, 0, selected);
  recorder.mark(3, 17);
}
'''
        source = adapter.embed_header(body, source_name="embedded_probe.cu").encode()
        for enabled in (False, True):
            error, program = nvrtc.nvrtcCreateProgram(source, b"embedded_probe.cu", 0, [], [])
            self.assertEqual(int(error), 0)
            options = [b"--gpu-architecture=compute_89", b"--std=c++17"]
            if enabled:
                options.append(b"--define-macro=ATREX_TIMELINE_ENABLED")
            (result,) = nvrtc.nvrtcCompileProgram(program, len(options), options)
            _, log_size = nvrtc.nvrtcGetProgramLogSize(program)
            log = bytearray(log_size)
            nvrtc.nvrtcGetProgramLog(program, log)
            self.assertEqual(int(result), 0, bytes(log).rstrip(b"\0").decode(errors="replace"))
            _, ptx_size = nvrtc.nvrtcGetPTXSize(program)
            self.assertGreater(ptx_size, 1)
            nvrtc.nvrtcDestroyProgram(program)

    def test_multi_writer_identity_is_recovered_from_slots(self) -> None:
        # Each owner writes one unique instant. Owners 0/1 belong to CTA 0; 2/3 to CTA 1.
        for owner in range(4):
            self.put(owner, 0, 100 + owner, 8, 2, 20 + owner, payload=owner)
        _, manifest_path, dictionary_path, _, _ = self.files()
        header = timeline.parse_header(bytes(self.raw))
        writers = timeline.validate_manifest(
            timeline.load_object(manifest_path, "manifest"), header
        )
        sites = timeline.load_sites(timeline.load_object(dictionary_path, "dictionary"))
        records, canonical = timeline.decode_records(bytes(self.raw), header, writers, sites)
        self.assertEqual(
            [(r["cta_linear"], r["writer_ordinal"], r["payload"]) for r in records],
            [(0, 0, 0), (0, 1, 1), (1, 0, 2), (1, 1, 3)],
        )
        self.assertEqual(len(canonical), 4)

    def test_manifest_writer_must_identify_a_real_block_thread(self) -> None:
        _, manifest_path, _, _, _ = self.files()
        manifest = timeline.load_object(manifest_path, "manifest")
        header = timeline.parse_header(bytes(self.raw))
        manifest["owner_layout"]["writers"][0].pop("warp")
        with self.assertRaisesRegex(timeline.TimelineError, "must declare warp or thread"):
            timeline.validate_manifest(manifest, header)

        manifest["owner_layout"]["writers"][0]["thread"] = 128
        with self.assertRaisesRegex(timeline.TimelineError, "outside the block"):
            timeline.validate_manifest(manifest, header)
        manifest["owner_layout"]["writers"][0]["thread"] = 1
        writers = timeline.validate_manifest(manifest, header)
        self.assertEqual(timeline.owner_identity(0, header, writers)["warp"], 0)

    def test_range_pairing_and_perfetto_preserve_original_nanoseconds(self) -> None:
        self.put(0, 0, 10_000_000_001, 5, 0, 7, payload=11)
        self.put(0, 1, 10_000_000_065, 5, 1, 7, payload=12)
        _, manifest_path, dictionary_path, _, _ = self.files()
        header = timeline.parse_header(bytes(self.raw))
        writers = timeline.validate_manifest(
            timeline.load_object(manifest_path, "manifest"), header
        )
        sites = timeline.load_sites(timeline.load_object(dictionary_path, "dictionary"))
        records, canonical = timeline.decode_records(bytes(self.raw), header, writers, sites)
        self.assertEqual(len(records), 2)
        self.assertEqual(canonical[0]["duration_ns"], 64)
        trace = timeline.make_perfetto(canonical)
        complete = next(event for event in trace["traceEvents"] if event.get("ph") == "X")
        self.assertEqual(complete["dur"], 0.064)
        self.assertEqual(complete["args"]["timestamp_ns"], 10_000_000_001)
        self.assertEqual(complete["args"]["end_timestamp_ns"], 10_000_000_065)

    def test_overflow_status_fails_closed(self) -> None:
        status_offset = 28
        struct.pack_into("<I", self.raw, status_offset, 1)
        with self.assertRaisesRegex(timeline.TimelineError, "overflow"):
            timeline.parse_header(bytes(self.raw))

    def test_duplicate_owner_status_fails_closed(self) -> None:
        status_offset = 28
        struct.pack_into("<I", self.raw, status_offset, 1 << 4)
        with self.assertRaisesRegex(timeline.TimelineError, "duplicate_owner"):
            timeline.parse_header(bytes(self.raw))

    def test_unknown_abi_minor_fails_closed(self) -> None:
        abi_minor_offset = 10
        struct.pack_into("<H", self.raw, abi_minor_offset, 1)
        with self.assertRaisesRegex(timeline.TimelineError, "ABI minor"):
            timeline.parse_header(bytes(self.raw))

    def test_unmatched_range_boundaries_fail_closed(self) -> None:
        _, manifest_path, dictionary_path, _, _ = self.files()
        writers = timeline.validate_manifest(
            timeline.load_object(manifest_path, "manifest"), timeline.parse_header(bytes(self.raw))
        )
        sites = timeline.load_sites(timeline.load_object(dictionary_path, "dictionary"))

        self.put(0, 0, 100, 5, 0, 1)
        header = timeline.parse_header(bytes(self.raw))
        with self.assertRaisesRegex(timeline.TimelineError, "unmatched begin"):
            timeline.decode_records(bytes(self.raw), header, writers, sites)

        self.raw[adapter.HEADER_STRUCT.size :] = bytes(len(self.raw) - adapter.HEADER_STRUCT.size)
        self.put(0, 0, 100, 5, 1, 1)
        with self.assertRaisesRegex(timeline.TimelineError, "ends without a begin"):
            timeline.decode_records(bytes(self.raw), header, writers, sites)

    def test_hole_and_unknown_site_fail_closed(self) -> None:
        self.put(0, 1, 100, 5, 2, 1)
        _, manifest_path, dictionary_path, _, _ = self.files()
        header = timeline.parse_header(bytes(self.raw))
        writers = timeline.validate_manifest(
            timeline.load_object(manifest_path, "manifest"), header
        )
        sites = timeline.load_sites(timeline.load_object(dictionary_path, "dictionary"))
        with self.assertRaisesRegex(timeline.TimelineError, "after an empty"):
            timeline.decode_records(bytes(self.raw), header, writers, sites)

        self.raw[adapter.HEADER_STRUCT.size :] = bytes(len(self.raw) - adapter.HEADER_STRUCT.size)
        self.put(0, 0, 100, 99, 2, 1)
        with self.assertRaisesRegex(timeline.TimelineError, "unknown site_id"):
            timeline.decode_records(bytes(self.raw), header, writers, sites)

        self.raw[adapter.HEADER_STRUCT.size :] = bytes(len(self.raw) - adapter.HEADER_STRUCT.size)
        self.put(0, 0, 100, 5, 2, 1)
        with self.assertRaisesRegex(timeline.TimelineError, "does not match range site"):
            timeline.decode_records(bytes(self.raw), header, writers, sites)

    def test_end_to_end_receipt_recomputes_measurement_and_hashes(self) -> None:
        self.put(0, 0, 100, 5, 0, 3)
        self.put(0, 1, 180, 5, 1, 3)
        raw, manifest, dictionary, clean, instrumented = self.files()
        binary = self.root / "kernel.cubin"
        binary.write_bytes(b"test-binary")
        measurement = self.root / "measurement.json"
        measurement.write_text(
            json.dumps(
                {
                    "schema": timeline.MEASUREMENT_SCHEMA,
                    "schedule": [
                        "baseline", "instrumented", "instrumented", "baseline",
                        "instrumented", "baseline", "baseline", "instrumented",
                    ],
                    "runs": [
                        {"ordinal": 0, "variant": "baseline", "latency_ms": 1.0},
                        {"ordinal": 1, "variant": "instrumented", "latency_ms": 1.05},
                        {"ordinal": 2, "variant": "instrumented", "latency_ms": 1.07},
                        {"ordinal": 3, "variant": "baseline", "latency_ms": 1.02},
                        {"ordinal": 4, "variant": "instrumented", "latency_ms": 1.05},
                        {"ordinal": 5, "variant": "baseline", "latency_ms": 1.0},
                        {"ordinal": 6, "variant": "baseline", "latency_ms": 1.02},
                        {"ordinal": 7, "variant": "instrumented", "latency_ms": 1.07},
                    ],
                    "warmup": 3,
                    "iterations": 20,
                    "synchronized": True,
                    "workload_identity": "shape=m128n128k64,dtype=fp16",
                    "device_identity": {"uuid": "test-gpu"},
                    "variants": {
                        "baseline": {
                            "source": clean.name,
                            "source_sha256": timeline.digest(clean),
                            "binary": binary.name,
                            "binary_sha256": timeline.digest(binary),
                        },
                        "instrumented": {
                            "source": instrumented.name,
                            "source_sha256": timeline.digest(instrumented),
                            "binary": binary.name,
                            "binary_sha256": timeline.digest(binary),
                        },
                    },
                    "aggregation": {
                        "center": "median",
                        "overhead_percent": "(instrumented_ms / baseline_ms - 1) * 100",
                    },
                }
            ),
            encoding="utf-8",
        )
        correctness_evidence = self.evaluator_evidence(instrumented)
        output = self.root / "output"
        result = timeline.main(
            [
                "decode",
                "--raw",
                str(raw),
                "--manifest",
                str(manifest),
                "--dictionary",
                str(dictionary),
                "--clean-source",
                str(clean),
                "--instrumented-source",
                str(instrumented),
                "--binary",
                str(binary),
                "--workload-identity",
                "shape=m128n128k64,dtype=fp16",
                "--correctness",
                "passed",
                "--correctness-evidence",
                str(correctness_evidence),
                "--measurement",
                str(measurement),
                "--stage",
                "final",
                "--output-dir",
                str(output),
            ]
        )
        self.assertEqual(result, 0)
        receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
        self.assertEqual(receipt["evidence_class"], "decision_grade")
        self.assertTrue(receipt["correctness_evidence"]["result"]["all_pass"])
        self.assertIn("measurement_recomputed", receipt["classification_reasons"])
        self.assertEqual(receipt["event_dictionary_sha256"], timeline.digest(dictionary))
        self.assertAlmostEqual(receipt["measurement"]["overhead_percent"], 4.9504950495)
        self.assertEqual(timeline.main(["validate", str(output / "receipt.json")]), 0)
        with gzip.open(output / "trace.perfetto.json.gz", "rt", encoding="utf-8") as stream:
            trace = json.load(stream)
        self.assertEqual(trace["otherData"]["clock"], "%globaltimer")

        unverified_output = self.root / "unverified-output"
        self.assertEqual(
            timeline.main(
                [
                    "decode",
                    "--raw",
                    str(raw),
                    "--manifest",
                    str(manifest),
                    "--dictionary",
                    str(dictionary),
                    "--clean-source",
                    str(clean),
                    "--instrumented-source",
                    str(instrumented),
                    "--binary",
                    str(binary),
                    "--workload-identity",
                    "shape=m128n128k64,dtype=fp16",
                    "--correctness",
                    "passed",
                    "--measurement",
                    str(measurement),
                    "--stage",
                    "final",
                    "--output-dir",
                    str(unverified_output),
                ]
            ),
            0,
        )
        unverified = json.loads((unverified_output / "receipt.json").read_text(encoding="utf-8"))
        self.assertEqual(unverified["evidence_class"], "diagnostic")
        self.assertIn("correctness_not_verified", unverified["classification_reasons"])

        receipt_path = output / "receipt.json"
        trace_path = output / "trace.perfetto.json.gz"
        range_event = next(item for item in trace["traceEvents"] if item.get("ph") == "X")
        range_event["args"]["duration_ns"] += 1
        timeline.write_gzip_json(trace_path, trace)
        receipt["trace_sha256"] = timeline.digest(trace_path)
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(timeline.TimelineError, "canonical trace is not reproducible"):
            timeline.main(["validate", str(receipt_path)])
        range_event["args"]["duration_ns"] -= 1
        timeline.write_gzip_json(trace_path, trace)
        receipt["trace_sha256"] = timeline.digest(trace_path)

        receipt["trace_integrity"] = "failed"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(timeline.TimelineError, "trace integrity"):
            timeline.main(["validate", str(receipt_path)])
        receipt["trace_integrity"] = "passed"
        receipt["stage"] = "invented"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(timeline.TimelineError, "stage is invalid"):
            timeline.main(["validate", str(receipt_path)])
        receipt["stage"] = "final"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

        measurement.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(timeline.TimelineError, "measurement hash mismatch"):
            timeline.main(["validate", str(receipt_path)])

    def test_measurement_and_receipt_tampering_fail_closed(self) -> None:
        clean = self.root / "clean.py"
        instrumented = self.root / "instrumented.py"
        binary = self.root / "kernel.bin"
        clean.write_text("clean\n", encoding="utf-8")
        instrumented.write_text("instrumented\n", encoding="utf-8")
        binary.write_bytes(b"binary")
        base = {
            "schema": timeline.MEASUREMENT_SCHEMA,
            "schedule": [
                "baseline", "instrumented", "instrumented", "baseline",
                "instrumented", "baseline", "baseline", "instrumented",
            ],
            "runs": [
                {"ordinal": 0, "variant": "baseline", "latency_ms": 1.0},
                {"ordinal": 1, "variant": "instrumented", "latency_ms": 1.1},
                {"ordinal": 2, "variant": "instrumented", "latency_ms": 1.1},
                {"ordinal": 3, "variant": "baseline", "latency_ms": 1.0},
                {"ordinal": 4, "variant": "instrumented", "latency_ms": 1.1},
                {"ordinal": 5, "variant": "baseline", "latency_ms": 1.0},
                {"ordinal": 6, "variant": "baseline", "latency_ms": 1.0},
                {"ordinal": 7, "variant": "instrumented", "latency_ms": 1.1},
            ],
            "warmup": 1,
            "iterations": 10,
            "synchronized": True,
            "workload_identity": "same-workload",
            "device_identity": {"uuid": "test-gpu"},
            "variants": {
                name: {
                    "source": source.name,
                    "source_sha256": timeline.digest(source),
                    "binary": binary.name,
                    "binary_sha256": timeline.digest(binary),
                }
                for name, source in (("baseline", clean), ("instrumented", instrumented))
            },
            "aggregation": {
                "center": "median",
                "overhead_percent": "(instrumented_ms / baseline_ms - 1) * 100",
            },
        }
        measurement = self.root / "measurement.json"
        invalid = dict(base)
        invalid["schedule"] = ["baseline", "instrumented", "baseline", "instrumented"]
        measurement.write_text(json.dumps(invalid), encoding="utf-8")
        with self.assertRaisesRegex(timeline.TimelineError, "ABBA or BAAB"):
            timeline.measurement_summary(measurement)

        measurement.write_text(json.dumps(base), encoding="utf-8")
        summary = timeline.measurement_summary(measurement)
        self.assertAlmostEqual(summary["overhead_percent"], 10.0)

    def test_measure_command_runs_fresh_counterbalanced_processes(self) -> None:
        sample_program = self.root / "sample.py"
        sample_program.write_text(
            """import json, os, sys
payload = {
    "latency_ms": float(sys.argv[1]),
    "correctness": "passed",
    "synchronized": True,
    "workload_identity": "fixture-workload",
    "device_identity": {"uuid": "fixture-gpu"},
    "warmup": int(os.environ["ATREX_TIMELINE_WARMUP"]),
    "iterations": int(os.environ["ATREX_TIMELINE_ITERATIONS"]),
}
print("__ATREX_TIMELINE_SAMPLE__=" + json.dumps(payload))
""",
            encoding="utf-8",
        )
        clean = self.root / "clean.cu"
        instrumented = self.root / "instrumented.cu"
        baseline_binary = self.root / "clean.bin"
        instrumented_binary = self.root / "instrumented.bin"
        clean.write_text("clean\n", encoding="utf-8")
        instrumented.write_text("instrumented\n", encoding="utf-8")
        baseline_binary.write_bytes(b"clean-binary")
        instrumented_binary.write_bytes(b"instrumented-binary")
        output = self.root / "formal-measurement.json"
        self.assertEqual(
            timeline.main(
                [
                    "measure",
                    "--baseline-command",
                    json.dumps([sys.executable, str(sample_program), "1.0"]),
                    "--instrumented-command",
                    json.dumps([sys.executable, str(sample_program), "1.05"]),
                    "--baseline-source",
                    str(clean),
                    "--instrumented-source",
                    str(instrumented),
                    "--baseline-binary",
                    str(baseline_binary),
                    "--instrumented-binary",
                    str(instrumented_binary),
                    "--workload-identity",
                    "fixture-workload",
                    "--warmup",
                    "2",
                    "--iterations",
                    "10",
                    "--order",
                    "ABBA",
                    "--order",
                    "BAAB",
                    "--output",
                    str(output),
                ]
            ),
            0,
        )
        artifact = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(
            artifact["schedule"],
            [
                "baseline", "instrumented", "instrumented", "baseline",
                "instrumented", "baseline", "baseline", "instrumented",
            ],
        )
        self.assertEqual([run["ordinal"] for run in artifact["runs"]], list(range(8)))
        self.assertAlmostEqual(timeline.measurement_summary(output)["overhead_percent"], 5.0)

        jit_output = self.root / "jit-measurement.json"
        jit_args = [
            "measure",
            "--baseline-command",
            json.dumps([sys.executable, str(sample_program), "1.0"]),
            "--instrumented-command",
            json.dumps([sys.executable, str(sample_program), "1.05"]),
            "--baseline-source",
            str(clean),
            "--instrumented-source",
            str(instrumented),
            "--workload-identity",
            "fixture-workload",
            "--warmup",
            "2",
            "--iterations",
            "10",
            "--order",
            "ABBA",
            "--order",
            "BAAB",
            "--output",
            str(jit_output),
        ]
        self.assertEqual(timeline.main(jit_args), 0)
        jit_artifact = json.loads(jit_output.read_text(encoding="utf-8"))
        self.assertIsNone(jit_artifact["variants"]["baseline"]["binary"])
        self.assertAlmostEqual(timeline.measurement_summary(jit_output)["overhead_percent"], 5.0)

    def test_iket_json_is_strictly_normalized(self) -> None:
        run_dir = self.root / "iket-run"
        run_dir.mkdir()
        (run_dir / "capture.pftrace").write_bytes(b"perfetto")
        (run_dir / "capture.trace.json").write_text(
            json.dumps(
                {
                    "graphLaunches": {},
                    "stringTable": ["phase", "point"],
                    "locationTable": [
                        {
                            "clusterId": [0, -1, -1],
                            "ctaId": [1, 2, 3],
                            "gpcId": 4,
                            "smId": 9,
                            "tpcId": 2,
                            "warpId": 5,
                        }
                    ],
                    "launches": [
                        {
                            "kernelName": "target_kernel",
                            "gridId": 7,
                            "contextId": 1,
                            "gridDimX": 2,
                            "gridDimY": 3,
                            "gridDimZ": 4,
                            "blockDimX": 192,
                            "blockDimY": 1,
                            "blockDimZ": 1,
                            "markers": [
                                {
                                    "markerNameIdx": 1,
                                    "locIdx": 0,
                                    "timestamp": 1032,
                                    "payloadType": 5,
                                    "payloadVal": 4,
                                }
                            ],
                            "ranges": [
                                {
                                    "rangeNameIdx": 0,
                                    "warpLocIdxs": [0, 0],
                                    "startTs": 1000,
                                    "endTs": 1064,
                                    "rangeId": 11,
                                    "rangeScope": 0,
                                    "rangeType": 1,
                                    "internalEvents": [],
                                }
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        dictionary = self.root / "iket-events.json"
        dictionary.write_text(
            json.dumps(
                {
                    "schema": iket_adapter.DICTIONARY_SCHEMA,
                    "events": [
                        {"name": "phase", "kind": "range"},
                        {"name": "point", "kind": "instant"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        canonical, metadata = iket_adapter.normalize_run(
            run_dir, kernel_pattern="target_kernel", dictionary_path=dictionary
        )
        self.assertEqual(len(canonical), 2)
        self.assertEqual(metadata["launches"][0]["grid"], [2, 3, 4])
        self.assertEqual(next(item for item in canonical if item["kind"] == "range")["duration_ns"], 64)
        perfetto = iket_adapter.make_perfetto(canonical)
        self.assertEqual(
            {item.get("ph") for item in perfetto["traceEvents"] if item.get("ph") != "M"},
            {"X", "i"},
        )

        trace_path = run_dir / "capture.trace.json"
        invalid_trace = json.loads(trace_path.read_text(encoding="utf-8"))
        invalid_trace["locationTable"].append(
            {**invalid_trace["locationTable"][0], "warpId": 4}
        )
        invalid_trace["launches"][0]["ranges"][0]["warpLocIdxs"] = [0, 1]
        trace_path.write_text(json.dumps(invalid_trace), encoding="utf-8")
        with self.assertRaisesRegex(iket_adapter.IketError, "changes CTA, warp, or physical SM"):
            iket_adapter.normalize_run(
                run_dir, kernel_pattern="target_kernel", dictionary_path=dictionary
            )
        invalid_trace["launches"][0]["ranges"][0]["warpLocIdxs"] = [0, 0]
        invalid_trace["locationTable"][0]["ctaId"] = [2, 2, 3]
        trace_path.write_text(json.dumps(invalid_trace), encoding="utf-8")
        with self.assertRaisesRegex(iket_adapter.IketError, "outside the launch grid"):
            iket_adapter.normalize_run(
                run_dir, kernel_pattern="target_kernel", dictionary_path=dictionary
            )
        invalid_trace["locationTable"][0]["ctaId"] = [1, 2, 3]
        invalid_trace["locationTable"][0]["warpId"] = 6
        trace_path.write_text(json.dumps(invalid_trace), encoding="utf-8")
        with self.assertRaisesRegex(iket_adapter.IketError, "outside the block"):
            iket_adapter.normalize_run(
                run_dir, kernel_pattern="target_kernel", dictionary_path=dictionary
            )
        invalid_trace["locationTable"][0]["warpId"] = 5
        trace_path.write_text(json.dumps(invalid_trace), encoding="utf-8")

        dictionary.write_text(
            json.dumps(
                {
                    "schema": iket_adapter.DICTIONARY_SCHEMA,
                    "events": [{"name": "missing", "kind": "instant"}],
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(iket_adapter.IketError, "was not observed"):
            iket_adapter.normalize_run(
                run_dir, kernel_pattern="target_kernel", dictionary_path=dictionary
            )

        dictionary.write_text(
            json.dumps(
                {
                    "schema": iket_adapter.DICTIONARY_SCHEMA,
                    "events": [
                        {"name": "phase", "kind": "range"},
                        {"name": "point", "kind": "instant"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        invalid_trace["graphLaunches"] = {"graph_exec_1:0": invalid_trace["launches"]}
        invalid_trace["launches"] = []
        trace_path.write_text(json.dumps(invalid_trace), encoding="utf-8")
        canonical, metadata = iket_adapter.normalize_run(
            run_dir, kernel_pattern="target_kernel", dictionary_path=dictionary
        )
        self.assertEqual(len(canonical), 2)
        self.assertIn(":graph:graph_exec_1:0:0", metadata["launches"][0]["launch_key"])

        tracker_dir = run_dir / "tracker" / "pid_1"
        tracker_dir.mkdir(parents=True)
        cubin = tracker_dir / "module.cubin"
        cubin.write_bytes(b"cubin")
        (tracker_dir / "tracker.json").write_text(
            json.dumps(
                {
                    "launches": [],
                    "graphLaunches": {
                        "graph_exec_1:0": [{"kernelName": "target_kernel", "moduleId": 7}]
                    },
                    "modules": [{"moduleId": 7, "image": str(cubin)}],
                }
            ),
            encoding="utf-8",
        )
        self.assertEqual(iket_adapter.target_binaries(run_dir, "target_kernel"), [cubin.resolve()])


class EpisodeIntegrationTests(unittest.TestCase):
    def test_temporary_profile_snapshot_is_archived_but_not_a_candidate(self) -> None:
        if shutil.which("git") is None:
            self.skipTest("git is unavailable")
        with tempfile.TemporaryDirectory(prefix="atrex-timeline-episode-") as temporary:
            workspace = Path(temporary) / "episode"
            workspace.mkdir()
            subprocess.run(["git", "init", "-b", "episode"], cwd=workspace, check=True,
                           capture_output=True)
            subprocess.run(["git", "config", "user.email", "timeline-test@example.invalid"],
                           cwd=workspace, check=True)
            subprocess.run(["git", "config", "user.name", "Timeline Test"], cwd=workspace,
                           check=True)
            kernel = workspace / "kernel.py"
            kernel.write_text("value = 'clean'\n", encoding="utf-8")
            subprocess.run(["git", "add", "kernel.py"], cwd=workspace, check=True)
            subprocess.run(["git", "commit", "-m", "baseline"], cwd=workspace, check=True,
                           capture_output=True)
            base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=workspace, check=True,
                                  capture_output=True, text=True).stdout.strip()
            CampaignStore.ensure_excluded(workspace)

            attempt = workspace / "profiles" / "episode_1" / "timeline" / "attempt-1"
            attempt.mkdir(parents=True)
            kernel.write_text("value = 'instrumented'\n", encoding="utf-8")
            (attempt / "instrumented_kernel.py").write_text(kernel.read_text(encoding="utf-8"),
                                                              encoding="utf-8")
            (attempt / "receipt.json").write_text('{"trace_integrity":"passed"}\n',
                                                   encoding="utf-8")

            kernel.write_text("cute.experimental.iket.mark('probe')\n", encoding="utf-8")
            subprocess.run(["git", "add", "kernel.py"], cwd=workspace, check=True)
            subprocess.run(["git", "commit", "-m", "private profiling checkpoint"],
                           cwd=workspace, check=True, capture_output=True)
            profiled = subprocess.run(["git", "rev-parse", "HEAD"], cwd=workspace, check=True,
                                      capture_output=True, text=True).stdout.strip()
            episode = EpisodeWorktree(episode=1, base_commit=base, branch="episode", path=workspace)
            violation, changed = episode.validate_candidate(profiled)
            self.assertEqual(violation, "candidate kernel.py still contains timeline profiling probes")
            self.assertEqual(changed, ["kernel.py"])

            kernel.write_text("value = 'optimized_without_probes'\n", encoding="utf-8")
            subprocess.run(["git", "add", "kernel.py"], cwd=workspace, check=True)
            subprocess.run(["git", "commit", "-m", "candidate"], cwd=workspace, check=True,
                           capture_output=True)
            candidate = subprocess.run(["git", "rev-parse", "HEAD"], cwd=workspace, check=True,
                                       capture_output=True, text=True).stdout.strip()
            violation, changed = episode.validate_candidate(candidate)
            self.assertEqual(violation, "")
            self.assertEqual(changed, ["kernel.py"])

            archive = episode.archive(Path(temporary) / "archive", candidate)
            archived_attempt = archive / "worktree_files" / attempt.relative_to(workspace)
            self.assertTrue((archived_attempt / "instrumented_kernel.py").is_file())
            self.assertTrue((archived_attempt / "receipt.json").is_file())


@unittest.skipUnless(os.environ.get("ATREX_RUN_GPU_TESTS") == "1", "GPU integration is opt-in")
class CudaRuntimeIntegrationTests(unittest.TestCase):
    def test_device_records_decode_to_complete_evidence(self) -> None:
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            self.skipTest("nvcc is unavailable")
        with tempfile.TemporaryDirectory(prefix="atrex-timeline-gpu-") as temporary:
            root = Path(temporary)
            executable = root / "probe"
            raw = root / "raw.bin"
            source = BACKEND / "test_backend.cu"
            arch = os.environ.get("ATREX_TEST_CUDA_ARCH", "sm_89")
            subprocess.run(
                [
                    nvcc,
                    "-std=c++17",
                    f"-arch={arch}",
                    "-DATREX_TIMELINE_ENABLED",
                    str(source),
                    "-o",
                    str(executable),
                ],
                check=True,
            )
            subprocess.run([str(executable), "--duplicate-owner"], check=True)
            subprocess.run([str(executable), "--overflow"], check=True)
            subprocess.run([str(executable), "--zero-probe"], check=True)
            subprocess.run([str(executable), "--one-range"], check=True)
            subprocess.run([str(executable), str(raw)], check=True)
            disabled = root / "probe-disabled"
            subprocess.run(
                [
                    nvcc,
                    "-std=c++17",
                    f"-arch={arch}",
                    str(source),
                    "-o",
                    str(disabled),
                ],
                check=True,
            )
            subprocess.run([str(disabled)], check=True)
            manifest = root / "manifest.json"
            dictionary = root / "events.json"
            clean = root / "clean.cu"
            instrumented = root / "instrumented.cu"
            manifest.write_text(
                json.dumps(
                    {
                        "schema": timeline.MANIFEST_SCHEMA,
                        "backend": "cuda",
                        "launch_id": 0x1234,
                        "grid": [2, 1, 1],
                        "block": [128, 1, 1],
                        "records_per_owner": 3,
                        "owner_layout": {
                            "kind": "cta_writers",
                            "writers": [
                                {"ordinal": 0, "label": "warp0", "warp": 0},
                                {"ordinal": 1, "label": "warp2", "warp": 2},
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
            dictionary.write_text(
                json.dumps(
                    {
                        "schema": timeline.DICTIONARY_SCHEMA,
                        "sites": [
                            {"site_id": 7, "name": "work", "boundary_semantics": "software_region"},
                            {"site_id": 9, "name": "done", "boundary_semantics": "mark"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            clean.write_text("// clean fixture\n", encoding="utf-8")
            instrumented.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            output = root / "evidence"
            self.assertEqual(
                timeline.main(
                    [
                        "decode",
                        "--raw",
                        str(raw),
                        "--manifest",
                        str(manifest),
                        "--dictionary",
                        str(dictionary),
                        "--clean-source",
                        str(clean),
                        "--instrumented-source",
                        str(instrumented),
                        "--binary",
                        str(executable),
                        "--workload-identity",
                        "cuda-runtime-fixture",
                        "--correctness",
                        "passed",
                        "--output-dir",
                        str(output),
                    ]
                ),
                0,
            )
            summary = json.loads((output / "trace.summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["committed_records"], 12)
            self.assertEqual(summary["ranges"], 4)
            self.assertEqual(summary["instants"], 4)
            self.assertEqual(summary["owners_observed"], 4)
            receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["evidence_class"], "exploration")
            self.assertEqual(receipt["classification_reasons"], ["stage_is_exploration", "overhead_not_measured"])
            self.assertEqual(timeline.main(["validate", str(output / "receipt.json")]), 0)

            matrix_raw = root / "matrix.bin"
            subprocess.run([str(executable), "--matrix", str(matrix_raw)], check=True)
            matrix_manifest = root / "matrix-manifest.json"
            matrix_dictionary = root / "matrix-events.json"
            matrix_manifest.write_text(
                json.dumps(
                    {
                        "schema": timeline.MANIFEST_SCHEMA,
                        "backend": "cuda",
                        "launch_id": 0x1234,
                        "grid": [2, 2, 2],
                        "block": [32, 1, 1],
                        "records_per_owner": 6,
                        "owner_layout": {
                            "kind": "cta_writers",
                            "writers": [{"ordinal": 0, "label": "warp0", "warp": 0}],
                        },
                    }
                ),
                encoding="utf-8",
            )
            matrix_dictionary.write_text(
                json.dumps(
                    {
                        "schema": timeline.DICTIONARY_SCHEMA,
                        "sites": [
                            {"site_id": 11, "name": "loop", "boundary_semantics": "software_region"},
                            {"site_id": 12, "name": "odd", "boundary_semantics": "mark"},
                            {"site_id": 13, "name": "even", "boundary_semantics": "counter"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            matrix_output = root / "matrix-evidence"
            self.assertEqual(
                timeline.main(
                    [
                        "decode",
                        "--raw",
                        str(matrix_raw),
                        "--manifest",
                        str(matrix_manifest),
                        "--dictionary",
                        str(matrix_dictionary),
                        "--clean-source",
                        str(clean),
                        "--instrumented-source",
                        str(instrumented),
                        "--binary",
                        str(executable),
                        "--workload-identity",
                        "cuda-runtime-3d-loop-branch-fixture",
                        "--correctness",
                        "passed",
                        "--output-dir",
                        str(matrix_output),
                    ]
                ),
                0,
            )
            matrix_summary = json.loads(
                (matrix_output / "trace.summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(matrix_summary["committed_records"], 48)
            self.assertEqual(matrix_summary["ranges"], 16)
            self.assertEqual(matrix_summary["instants"], 8)
            self.assertEqual(matrix_summary["counters"], 8)
            self.assertEqual(matrix_summary["owners_observed"], 8)
            with gzip.open(matrix_output / "trace.perfetto.json.gz", "rt", encoding="utf-8") as stream:
                matrix_trace = json.load(stream)
            observed_ctas = {
                tuple(event["args"]["cta"])
                for event in matrix_trace["traceEvents"]
                if event.get("ph") in {"X", "i", "C"}
            }
            self.assertEqual(
                observed_ctas,
                {(x, y, z) for z in range(2) for y in range(2) for x in range(2)},
            )
            self.assertEqual(timeline.main(["validate", str(matrix_output / "receipt.json")]), 0)

    def test_real_iket_capture_and_export(self) -> None:
        if shutil.which("run-iket") is None:
            self.skipTest("run-iket is unavailable")
        with tempfile.TemporaryDirectory(prefix="atrex-iket-gpu-") as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            fixture = BACKEND.parent / "cutedsl_backend" / "test_iket.py"
            dictionary = root / "events.json"
            dictionary.write_text(
                json.dumps(
                    {
                        "schema": iket_adapter.DICTIONARY_SCHEMA,
                        "events": [
                            {"name": "probe_start", "kind": "instant"},
                            {"name": "probe_body", "kind": "range"},
                            {"name": "probe_end", "kind": "instant"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            clean = root / "clean.py"
            clean.write_text("# clean fixture\n", encoding="utf-8")
            output = root / "evidence"
            self.assertEqual(
                timeline.main(
                    [
                        "profile-iket",
                        "--run-dir",
                        str(run_dir),
                        "--evidence-dir",
                        str(output),
                        "--kernel-regex",
                        "kernel_cutlass_timeline_probe_0",
                        "--dictionary",
                        str(dictionary),
                        "--clean-source",
                        str(clean),
                        "--instrumented-source",
                        str(fixture),
                        "--workload-identity",
                        "cutedsl-iket-fixture",
                        "--correctness",
                        "passed",
                        "--",
                        sys.executable,
                        str(fixture),
                    ]
                ),
                0,
            )
            summary = json.loads((output / "trace.summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["backend"], "iket")
            self.assertEqual(summary["matched_launches"], 1)
            self.assertEqual(summary["ranges"], 1)
            self.assertEqual(summary["instants"], 2)
            receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["backend"], "iket")
            self.assertIsNotNone(receipt["binary"])
            self.assertEqual(timeline.main(["validate", str(output / "receipt.json")]), 0)


if __name__ == "__main__":
    unittest.main()
