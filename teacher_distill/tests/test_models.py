from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from teacher_distill.models import (
    AbbaStatus,
    CampaignLock,
    CampaignTerminalStatus,
    KnowledgeDeny,
    OperatorIdentity,
    SourceProvenance,
    TargetIdentity,
    TeacherCampaignResult,
    TeacherProgress,
    TeacherProvenance,
    TeacherTarget,
    canonical_json,
)


ROOT = Path(__file__).resolve().parents[2]


class TeacherProvenanceTest(unittest.TestCase):
    def _valid(self) -> dict:
        return {
            "schema_version": 1,
            "operator": {
                "canonical_id": "gdn_decode",
                "aliases": [
                    "gdn",
                    "gated_delta_rule",
                    "fused_recurrent_gated_delta_rule_fwd",
                ],
            },
            "source": {
                "project": "flashinfer",
                "revision": "abc123",
                "license": "Apache-2.0",
            },
            "target": {"framework": "CuteDSL", "architecture": "sm90"},
            "knowledge_deny": {
                "sources": ["flashinfer"],
                "paths": [],
                "tags": ["gdn", "gdn_decode", "gated_delta_rule"],
            },
        }

    def test_valid_provenance_round_trips_to_canonical_mapping(self) -> None:
        provenance = TeacherProvenance.from_mapping(self._valid())

        self.assertEqual(provenance.operator.canonical_id, "gdn_decode")
        self.assertEqual(provenance.target.framework, "CuteDSL")
        self.assertEqual(provenance.to_mapping(), self._valid())
        self.assertEqual(
            canonical_json(provenance.to_mapping()),
            canonical_json(json.loads(json.dumps(self._valid()))),
        )

    def test_required_identity_fields_fail_closed(self) -> None:
        cases = (
            ("source", "revision"),
            ("source", "license"),
            ("target", "framework"),
            ("target", "architecture"),
            ("operator", "aliases"),
            ("knowledge_deny", "tags"),
        )
        for section, field in cases:
            with self.subTest(section=section, field=field):
                raw = self._valid()
                del raw[section][field]
                with self.assertRaises(ValueError):
                    TeacherProvenance.from_mapping(raw)

    def test_unknown_schema_and_empty_deny_selectors_are_rejected(self) -> None:
        raw = self._valid()
        raw["schema_version"] = 2
        with self.assertRaisesRegex(ValueError, "schema_version"):
            TeacherProvenance.from_mapping(raw)

        raw = self._valid()
        raw["knowledge_deny"] = {"sources": [], "paths": [], "tags": []}
        with self.assertRaisesRegex(ValueError, "deny selector"):
            TeacherProvenance.from_mapping(raw)

    def test_unknown_contract_fields_are_rejected(self) -> None:
        raw = self._valid()
        raw["unexpected"] = "value"
        with self.assertRaisesRegex(ValueError, "unexpected fields"):
            TeacherProvenance.from_mapping(raw)

        raw = self._valid()
        raw["source"]["url"] = "https://example.invalid"
        with self.assertRaisesRegex(ValueError, "unexpected fields"):
            TeacherProvenance.from_mapping(raw)

    def test_direct_models_are_immutable(self) -> None:
        operator = OperatorIdentity("gdn_decode", ("gdn",))
        source = SourceProvenance("flashinfer", "abc", "Apache-2.0")
        target = TargetIdentity("CuteDSL", "sm90")
        deny = KnowledgeDeny(("flashinfer",), (), ("gdn",))
        provenance = TeacherProvenance(1, operator, source, target, deny)

        with self.assertRaises((AttributeError, TypeError)):
            provenance.operator = OperatorIdentity("other", ("other",))  # type: ignore[misc]


class TeacherTargetTest(unittest.TestCase):
    def _target(self, **overrides: object) -> TeacherTarget:
        values = {
            "schema_version": 1,
            "teacher_id": "teacher_0123456789abcdef",
            "geomean_latency_us": 100.0,
            "latency_us_by_shape": {"0": 80.0, "1": 125.0},
            "geomean_ratio": 1.05,
            "shape_ratio": 1.10,
            "measurement_config_hash": "a" * 64,
            "knowledge_view_hash": "b" * 64,
        }
        values.update(overrides)
        return TeacherTarget(**values)

    def test_public_target_contains_metrics_but_no_private_provenance(self) -> None:
        mapping = self._target().to_mapping()
        rendered = canonical_json(mapping)

        self.assertEqual(mapping["teacher_id"], "teacher_0123456789abcdef")
        self.assertNotIn("flashinfer", rendered)
        self.assertNotIn("revision", rendered)
        self.assertNotIn("license", rendered)
        self.assertNotIn("path", rendered)

    def test_target_rejects_non_finite_or_non_positive_measurements(self) -> None:
        for value in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self._target(geomean_latency_us=value)

        with self.assertRaises(ValueError):
            self._target(latency_us_by_shape={"0": 80.0, "1": 0.0})

    def test_target_ratios_must_be_finite_and_at_least_one(self) -> None:
        for value in (0.99, math.nan, math.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self._target(geomean_ratio=value)
                with self.assertRaises(ValueError):
                    self._target(shape_ratio=value)

    def test_per_shape_target_mapping_is_immutable(self) -> None:
        target = self._target()
        with self.assertRaises(TypeError):
            target.latency_us_by_shape["0"] = 1.0  # type: ignore[index]


class CampaignLockTest(unittest.TestCase):
    def _lock(self, **overrides: object) -> CampaignLock:
        values = {
            "schema_version": 1,
            "campaign_id": "campaign_01234567",
            "teacher_id": "teacher_01234567",
            "platform": "H20",
            "architecture": "sm90",
            "framework": "CuteDSL",
            "workload_hash": "1" * 64,
            "evaluator_hash": "2" * 64,
            "measurement_config_hash": "3" * 64,
            "knowledge_view_hash": "4" * 64,
            "geomean_ratio": 1.05,
            "shape_ratio": 1.10,
        }
        values.update(overrides)
        return CampaignLock(**values)

    def test_fingerprint_is_deterministic_and_changes_with_semantics(self) -> None:
        first = self._lock()
        second = CampaignLock.from_mapping(dict(reversed(list(first.to_mapping().items()))))

        self.assertEqual(first.fingerprint(), second.fingerprint())
        self.assertNotEqual(first.fingerprint(), self._lock(shape_ratio=1.20).fingerprint())

    def test_lock_rejects_empty_hash_or_unsupported_schema(self) -> None:
        with self.assertRaises(ValueError):
            self._lock(workload_hash="")
        with self.assertRaisesRegex(ValueError, "schema_version"):
            self._lock(schema_version=2)


class TeacherProgressAndResultTest(unittest.TestCase):
    def test_progress_serializes_typed_abba_status(self) -> None:
        progress = TeacherProgress(
            target_id="teacher_01234567",
            candidate_to_teacher_geomean_ratio=1.18,
            worst_shape_ratio=1.27,
            worst_shape_key="shape_4",
            geomean_gate_met=False,
            shape_gate_met=False,
            provisional_target_met=False,
            abba_status=AbbaStatus.NOT_RUN,
        )

        self.assertEqual(progress.to_mapping()["abba_status"], "NOT_RUN")
        with self.assertRaises(ValueError):
            TeacherProgress.from_mapping({**progress.to_mapping(), "abba_status": "MAYBE"})

    def test_terminal_result_accepts_only_declared_statuses(self) -> None:
        result = TeacherCampaignResult(
            schema_version=1,
            campaign_id="campaign_01234567",
            status=CampaignTerminalStatus.SUCCESS,
            reason="teacher target reached",
            final_version="v9",
            final_candidate_to_teacher_ratio=1.03,
        )
        self.assertEqual(result.to_mapping()["status"], "SUCCESS")

        with self.assertRaises(ValueError):
            TeacherCampaignResult.from_mapping({**result.to_mapping(), "status": "DONE"})

    def test_schema_files_are_valid_json_and_teacher_progress_is_optional(self) -> None:
        for relative in (
            "teacher_distill/schemas/provenance.schema.json",
            "teacher_distill/schemas/campaign-lock.schema.json",
        ):
            parsed = json.loads((ROOT / relative).read_text(encoding="utf-8"))
            self.assertEqual(parsed["$schema"], "https://json-schema.org/draft/2020-12/schema")

        iteration = json.loads((ROOT / "reference/v_iteration.schema.json").read_text(encoding="utf-8"))
        self.assertIn("teacher_progress", iteration)
        self.assertIsNone(iteration["teacher_progress"])


if __name__ == "__main__":
    unittest.main()
