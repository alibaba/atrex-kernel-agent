from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from teacher_distill.knowledge_view import build_knowledge_view
from teacher_distill.models import TeacherProvenance


REPO_ROOT = Path(__file__).resolve().parents[2]


class KnowledgeViewTest(unittest.TestCase):
    def _provenance(self) -> TeacherProvenance:
        return TeacherProvenance.from_mapping(
            {
                "schema_version": 1,
                "operator": {
                    "canonical_id": "gdn_decode",
                    "aliases": ["gdn", "gated_delta_rule"],
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
        )

    def _write_wiki(self, root: Path) -> None:
        pages = {
            "docs/nvidia/hopper/hardware-specs/hardware.md":
                "# Hopper Hardware\n\nTarget hardware facts.\n",
            "docs/nvidia/common/ref-docs/cutedsl/api.md":
                "# CuteDSL API\n\nGeneric framework programming model.\n",
            "docs/nvidia/hopper/kernel-opt/techniques/vectorized.md":
                "# Vectorized Loads\n\nGeneric aligned vector access. "
                "[Operator example](../../ref-docs/cutedsl/gdn.md). "
                "[External reference](https://example.com/secret-source).\n",
            "docs/nvidia/hopper/kernel-opt/techniques/operator-note.md":
                "# Generic Operator Note\n\nA generic note for recurrent kernels.\n\n"
                "For GDN, copy the Teacher's TMA structure.\n",
            "docs/nvidia/hopper/ref-docs/cutedsl/gdn.md":
                "# GDN Decode\n\nOperator-specific implementation using cp.async.\n",
            "docs/nvidia/hopper/pitfalls/cutedsl/gdn-pitfalls.md":
                "# Gated Delta Rule Pitfalls\n\nOperator-specific answer.\n",
            "docs/nvidia/hopper/ref-docs/cutedsl/attention.md":
                "# Attention CuTeDSL\n\nCross-operator implementation pattern.\n",
            "docs/amd/cdna3/hardware-specs/hardware.md":
                "# CDNA3 Hardware\n\nWrong vendor.\n",
            "reference-kernels/nvidia/hopper/cutedsl/flashinfer/gdn.py":
                "def hidden_teacher_pattern(): pass\n",
            "reference-kernels/nvidia/hopper/cutedsl/example/attention.py":
                "def attention_pattern(): pass\n",
            "reference-kernels/nvidia/blackwell/cutedsl/example/attention.py":
                "def wrong_arch_pattern(): pass\n",
        }
        for relative, content in pages.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        manifest = {
            "version": 1,
            "docs": {
                "defaults": [
                    {
                        "prefix": "nvidia/hopper/",
                        "architectures": ["hopper"],
                        "vendors": ["nvidia"],
                    },
                    {
                        "prefix": "nvidia/common/",
                        "architectures": [],
                        "vendors": ["nvidia"],
                    },
                    {
                        "prefix": "amd/cdna3/",
                        "architectures": ["cdna3"],
                        "vendors": ["amd"],
                    },
                ],
                "entries": {
                    "nvidia/hopper/ref-docs/cutedsl/gdn.md": {
                        "dsls": ["cutedsl"],
                        "operators": ["gdn"],
                    },
                    "nvidia/hopper/pitfalls/cutedsl/gdn-pitfalls.md": {
                        "dsls": ["cutedsl"],
                        "operators": ["gdn"],
                    },
                    "nvidia/hopper/ref-docs/cutedsl/attention.md": {
                        "dsls": ["cutedsl"],
                        "operators": ["flash-attention"],
                    },
                    "nvidia/common/ref-docs/cutedsl/api.md": {
                        "dsls": ["cutedsl"]
                    },
                },
            },
            "reference-kernels": {
                "defaults": [
                    {
                        "prefix": "nvidia/hopper/",
                        "architectures": ["hopper"],
                        "vendors": ["nvidia"],
                        "status": "unclassified",
                    },
                    {"prefix": "nvidia/hopper/cutedsl/", "dsls": ["cutedsl"]},
                    {
                        "prefix": "nvidia/hopper/cutedsl/flashinfer/",
                        "source": "flashinfer",
                    },
                    {
                        "prefix": "nvidia/blackwell/",
                        "architectures": ["blackwell"],
                        "vendors": ["nvidia"],
                    },
                ],
                "entries": {
                    "nvidia/hopper/cutedsl/flashinfer/gdn.py": {
                        "operators": ["gdn"],
                        "kind": "kernel",
                    },
                    "nvidia/hopper/cutedsl/example/attention.py": {
                        "operators": ["flash-attention"],
                        "kind": "kernel",
                    },
                    "nvidia/blackwell/cutedsl/example/attention.py": {
                        "operators": ["flash-attention"],
                        "kind": "kernel",
                    },
                },
            },
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        (root / "README.md").write_text("# Source Wiki\n", encoding="utf-8")
        (root / "scripts").mkdir()
        for script in ("query.py", "check-self-contained.py"):
            shutil.copy2(REPO_ROOT / "gpu-wiki" / "scripts" / script, root / "scripts" / script)

    def test_view_keeps_scoped_generic_knowledge_and_removes_teacher_leaks(self) -> None:
        with tempfile.TemporaryDirectory(prefix="knowledge-view-") as temp_dir:
            base = Path(temp_dir)
            source = base / "source"
            output = base / "views"
            self._write_wiki(source)

            view = build_knowledge_view(
                source,
                output,
                architecture="sm90",
                framework="CuteDSL",
                provenance=self._provenance(),
            )

            self.assertTrue((view.root / "docs/nvidia/hopper/hardware-specs/hardware.md").is_file())
            self.assertTrue((view.root / "docs/nvidia/common/ref-docs/cutedsl/api.md").is_file())
            self.assertTrue((view.root / "docs/nvidia/hopper/ref-docs/cutedsl/attention.md").is_file())
            self.assertTrue(
                (view.root / "reference-kernels/nvidia/hopper/cutedsl/example/attention.py").is_file()
            )
            self.assertFalse((view.root / "docs/nvidia/hopper/ref-docs/cutedsl/gdn.md").exists())
            self.assertFalse(
                (view.root / "docs/nvidia/hopper/kernel-opt/techniques/operator-note.md").exists()
            )
            self.assertFalse((view.root / "docs/nvidia/hopper/pitfalls/cutedsl/gdn-pitfalls.md").exists())
            self.assertFalse(
                (view.root / "reference-kernels/nvidia/hopper/cutedsl/flashinfer/gdn.py").exists()
            )
            self.assertFalse((view.root / "docs/amd").exists())
            self.assertFalse((view.root / "reference-kernels/nvidia/blackwell").exists())
            self.assertFalse(any(path.is_symlink() for path in view.root.rglob("*")))

            vectorized = (
                view.root / "docs/nvidia/hopper/kernel-opt/techniques/vectorized.md"
            ).read_text(encoding="utf-8")
            self.assertNotIn("gdn.md", vectorized)
            self.assertIn("Operator example", vectorized)
            self.assertIn("External reference", vectorized)
            self.assertNotIn("https://", vectorized)
            self.assertNotIn("example.com", vectorized)

            report = json.loads((view.root / "knowledge-view.json").read_text(encoding="utf-8"))
            sensitive_reasons = {
                "explicit-path",
                "teacher-source",
                "operator-metadata",
                "operator-identity",
            }
            sensitive = [
                row for row in report["excluded"] if row["reason"] in sensitive_reasons
            ]
            self.assertTrue(sensitive)
            self.assertTrue(
                all(row["path"].startswith("redacted:") for row in sensitive)
            )
            serialized_report = json.dumps(report).casefold()
            self.assertNotIn("flashinfer", serialized_report)
            self.assertNotIn("gdn.md", serialized_report)
            self.assertNotIn("operator-note.md", serialized_report)

    def test_view_is_content_addressed_reused_and_changes_with_allowed_content(self) -> None:
        with tempfile.TemporaryDirectory(prefix="knowledge-view-hash-") as temp_dir:
            base = Path(temp_dir)
            source = base / "source"
            output = base / "views"
            self._write_wiki(source)

            first = build_knowledge_view(source, output, "sm90", "CuteDSL", self._provenance())
            second = build_knowledge_view(source, output, "sm90", "CuteDSL", self._provenance())
            self.assertEqual(first.view_hash, second.view_hash)
            self.assertEqual(first.root, second.root)

            allowed = source / "docs/nvidia/hopper/kernel-opt/techniques/vectorized.md"
            allowed.write_text(allowed.read_text(encoding="utf-8") + "More detail.\n", encoding="utf-8")
            third = build_knowledge_view(source, output, "sm90", "CuteDSL", self._provenance())
            self.assertNotEqual(first.view_hash, third.view_hash)

    def test_generated_view_is_queryable_self_contained_and_read_only(self) -> None:
        with tempfile.TemporaryDirectory(prefix="knowledge-view-tools-") as temp_dir:
            base = Path(temp_dir)
            source = base / "source"
            output = base / "views"
            self._write_wiki(source)
            view = build_knowledge_view(source, output, "h20", "cutedsl", self._provenance())

            query = subprocess.run(
                [
                    "python3",
                    str(view.root / "scripts/query.py"),
                    "attention",
                    "--root",
                    str(view.root),
                    "--arch",
                    "h20",
                    "--dsl",
                    "cutedsl",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(query.returncode, 0, query.stdout + query.stderr)
            self.assertIn("attention", query.stdout.lower())
            self.assertNotIn("gdn.py", query.stdout)

            check = subprocess.run(
                [
                    "python3",
                    str(view.root / "scripts/check-self-contained.py"),
                    "--root",
                    str(view.root),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(check.returncode, 0, check.stdout + check.stderr)
            self.assertEqual((view.root / "manifest.json").stat().st_mode & 0o222, 0)
            self.assertEqual((view.root / "README.md").stat().st_mode & 0o222, 0)
            self.assertEqual(view.root.stat().st_mode & 0o222, 0)
            self.assertEqual((view.root / "docs").stat().st_mode & 0o222, 0)

    def test_missing_wiki_contract_files_and_unknown_architecture_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="knowledge-view-invalid-") as temp_dir:
            base = Path(temp_dir)
            source = base / "source"
            output = base / "views"
            self._write_wiki(source)
            (source / "manifest.json").unlink()
            with self.assertRaisesRegex(ValueError, "manifest"):
                build_knowledge_view(source, output, "sm90", "CuteDSL", self._provenance())

        with tempfile.TemporaryDirectory(prefix="knowledge-view-arch-") as temp_dir:
            base = Path(temp_dir)
            source = base / "source"
            self._write_wiki(source)
            with self.assertRaisesRegex(ValueError, "architecture"):
                build_knowledge_view(
                    source,
                    base / "views",
                    "unknown9000",
                    "CuteDSL",
                    self._provenance(),
                )

    def test_knowledge_view_schema_is_valid_json(self) -> None:
        schema = json.loads(
            (REPO_ROOT / "teacher_distill/schemas/knowledge-view.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")


if __name__ == "__main__":
    unittest.main()
