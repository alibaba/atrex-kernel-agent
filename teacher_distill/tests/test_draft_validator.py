from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from teacher_distill.draft_validator import (
    DraftValidationError,
    validate_distillation_drafts,
)
from teacher_distill.state import PRIVATE_STATE_FILE


class DraftValidatorTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        drafts = root / "drafts"
        private = root / "private"
        teacher = root / "teacher"
        (drafts / "evidence").mkdir(parents=True)
        (drafts / "optimization_cards").mkdir()
        teacher.mkdir()
        (teacher / "kernel.py").write_text(
            "# Teacher source\nUNIQUE_PIPELINE_SEQUENCE = 12345\n", encoding="utf-8"
        )
        (private).mkdir()
        (private / PRIVATE_STATE_FILE).write_text(
            json.dumps({"bundle_path": str(teacher)}), encoding="utf-8"
        )
        evidence = {
            "schema_version": 1,
            "workspace_head": "abc",
            "evidence": [
                {
                    "evidence_id": "E-V1-MEMORY",
                    "kind": "memory",
                    "scope": "candidate",
                    "path": "memory/v1.json",
                    "sha256": "1" * 64,
                    "classification": "accepted",
                    "citable_as_verified": True,
                },
                {
                    "evidence_id": "E-V2-MEMORY",
                    "kind": "memory",
                    "scope": "candidate",
                    "path": "memory/v2.json",
                    "sha256": "2" * 64,
                    "classification": "reverted",
                    "citable_as_verified": False,
                },
            ],
        }
        (drafts / "evidence/evidence_manifest.json").write_text(
            json.dumps(evidence), encoding="utf-8"
        )
        (drafts / "evidence/performance_trajectory.json").write_text(
            json.dumps({"schema_version": 1, "versions": []}), encoding="utf-8"
        )
        (drafts / "teacher_gap_analysis.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "hypothesis",
                    "promotion_eligible": False,
                    "findings": [{"claim": "possible", "status": "hypothesis"}],
                }
            ),
            encoding="utf-8",
        )
        (drafts / "teacher_gap_analysis.md").write_text(
            "# Hypotheses\n\nPossible scheduling difference.\n", encoding="utf-8"
        )
        (drafts / "journey.md").write_text(
            "# Journey\n\nLatency improved to 100 us. Evidence: [E-V1-MEMORY]\n",
            encoding="utf-8",
        )
        (drafts / "pitfalls.md").write_text(
            "# Pitfalls\n\nThe experiment regressed and was reverted. [E-V2-MEMORY]\n",
            encoding="utf-8",
        )
        (drafts / "optimization_cards/vectorized.md").write_text(
            "# Vectorized load\n\n"
            "Architecture: sm90\n\nFramework: CuteDSL\n\nScope: tested shapes\n\n"
            "Verified improvement. [E-V1-MEMORY]\n",
            encoding="utf-8",
        )
        (drafts / "promotion_checklist.md").write_text(
            "# Promotion checklist\n", encoding="utf-8"
        )
        (drafts / "draft_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "evidence_level": "single-campaign",
                    "documents": [
                        "journey.md",
                        "pitfalls.md",
                        "optimization_cards/vectorized.md",
                        "promotion_checklist.md",
                    ],
                }
            ),
            encoding="utf-8",
        )
        return drafts, private

    def test_valid_drafts_produce_a_pass_report(self) -> None:
        with tempfile.TemporaryDirectory(prefix="draft-validator-") as temp_dir:
            drafts, private = self._fixture(Path(temp_dir))
            result = validate_distillation_drafts(drafts, private)
            report = json.loads(result.report_path.read_text(encoding="utf-8"))

        self.assertEqual(report["result"], "PASS")
        self.assertEqual(report["evidence_level"], "single-campaign")
        self.assertIn("optimization_cards/vectorized.md", report["documents"])

    def test_unknown_or_missing_citation_rejects_performance_claim(self) -> None:
        cases = (
            "# Journey\n\nLatency improved to 100 us.\n",
            "# Journey\n\nLatency improved to 100 us. [E-UNKNOWN]\n",
        )
        for content in cases:
            with self.subTest(content=content):
                with tempfile.TemporaryDirectory(prefix="draft-citation-") as temp_dir:
                    drafts, private = self._fixture(Path(temp_dir))
                    (drafts / "journey.md").write_text(content, encoding="utf-8")
                    with self.assertRaisesRegex(DraftValidationError, "evidence"):
                        validate_distillation_drafts(drafts, private)

    def test_optimization_card_cannot_use_reverted_evidence_as_proof(self) -> None:
        with tempfile.TemporaryDirectory(prefix="draft-reverted-") as temp_dir:
            drafts, private = self._fixture(Path(temp_dir))
            card = drafts / "optimization_cards/vectorized.md"
            card.write_text(
                card.read_text(encoding="utf-8").replace("E-V1-MEMORY", "E-V2-MEMORY"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DraftValidationError, "verified evidence"):
                validate_distillation_drafts(drafts, private)

    def test_hardware_and_causal_claims_require_verified_evidence(self) -> None:
        cases = (
            "The target has 132 SMs.\n",
            "Latency fell because occupancy increased. [E-V2-MEMORY]\n",
        )
        for text in cases:
            with self.subTest(text=text):
                with tempfile.TemporaryDirectory(prefix="draft-claim-evidence-") as temp_dir:
                    drafts, private = self._fixture(Path(temp_dir))
                    (drafts / "journey.md").write_text(
                        "# Journey\n\n" + text,
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(
                        DraftValidationError, "evidence|verified|citation"
                    ):
                        validate_distillation_drafts(drafts, private)

    def test_unlisted_generated_document_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="draft-unlisted-") as temp_dir:
            drafts, private = self._fixture(Path(temp_dir))
            (drafts / "unlisted.md").write_text("# Undeclared\n", encoding="utf-8")
            with self.assertRaisesRegex(DraftValidationError, "omits generated documents"):
                validate_distillation_drafts(drafts, private)

    def test_gap_hypothesis_cannot_be_promoted_or_marked_verified(self) -> None:
        with tempfile.TemporaryDirectory(prefix="draft-gap-") as temp_dir:
            drafts, private = self._fixture(Path(temp_dir))
            gap = json.loads((drafts / "teacher_gap_analysis.json").read_text())
            gap["promotion_eligible"] = True
            gap["findings"][0]["status"] = "verified"
            (drafts / "teacher_gap_analysis.json").write_text(
                json.dumps(gap), encoding="utf-8"
            )
            with self.assertRaisesRegex(DraftValidationError, "hypothesis"):
                validate_distillation_drafts(drafts, private)

    def test_gap_findings_cannot_embed_verified_or_promotable_claims(self) -> None:
        with tempfile.TemporaryDirectory(prefix="draft-gap-finding-") as temp_dir:
            drafts, private = self._fixture(Path(temp_dir))
            gap = json.loads((drafts / "teacher_gap_analysis.json").read_text())
            gap["findings"][0].update(
                {
                    "promotion_eligible": True,
                    "claim": "Verified causal speedup from the Teacher structure",
                }
            )
            (drafts / "teacher_gap_analysis.json").write_text(
                json.dumps(gap), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                DraftValidationError, "hypothesis|promotion|verified|causal"
            ):
                validate_distillation_drafts(drafts, private)

    def test_gap_markdown_cannot_claim_causality(self) -> None:
        with tempfile.TemporaryDirectory(prefix="draft-gap-markdown-") as temp_dir:
            drafts, private = self._fixture(Path(temp_dir))
            (drafts / "teacher_gap_analysis.md").write_text(
                "# Gap\n\nThis difference is proven causal.\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DraftValidationError, "hypothesis|causal"):
                validate_distillation_drafts(drafts, private)

    def test_teacher_source_fragment_in_gap_json_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="draft-gap-json-source-") as temp_dir:
            drafts, private = self._fixture(Path(temp_dir))
            gap = json.loads((drafts / "teacher_gap_analysis.json").read_text())
            gap["findings"][0]["claim"] = "UNIQUE_PIPELINE_SEQUENCE = 12345"
            (drafts / "teacher_gap_analysis.json").write_text(
                json.dumps(gap), encoding="utf-8"
            )
            with self.assertRaisesRegex(DraftValidationError, "Teacher source"):
                validate_distillation_drafts(drafts, private)

    def test_teacher_source_fragment_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="draft-source-leak-") as temp_dir:
            drafts, private = self._fixture(Path(temp_dir))
            (drafts / "journey.md").write_text(
                "# Journey\n\nUNIQUE_PIPELINE_SEQUENCE = 12345\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DraftValidationError, "Teacher source"):
                validate_distillation_drafts(drafts, private)

    def test_output_under_canonical_gpu_wiki_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="draft-wiki-root-") as temp_dir:
            fake_repo = Path(temp_dir) / "repo"
            drafts, private = self._fixture(fake_repo / "gpu-wiki" / "generated")
            with mock.patch("teacher_distill.draft_validator.optimize.REPO_ROOT", fake_repo):
                with self.assertRaisesRegex(DraftValidationError, "canonical gpu-wiki"):
                    validate_distillation_drafts(drafts, private)

    def test_schema_is_valid_json(self) -> None:
        root = Path(__file__).resolve().parents[2]
        schema = json.loads(
            (root / "teacher_distill/schemas/draft-manifest.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")


if __name__ == "__main__":
    unittest.main()
