from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from teacher_distill.bundle import validate_teacher_bundle


class TeacherBundleValidationTest(unittest.TestCase):
    def _write_bundle(
        self,
        root: Path,
        *,
        source_paths: list[str] | None = None,
        entry_point: str = "kernel.py::run",
        languages: list[str] | None = None,
        provenance_framework: str = "CuteDSL",
        provenance_architecture: str = "sm90",
        creation_order: tuple[str, ...] = ("kernel", "helper", "solution", "provenance"),
    ) -> None:
        payloads = {
            "kernel": (
                root / "kernel.py",
                "import cutlass.cute as cute\n\ndef run(x, out):\n    out[:] = x\n",
            ),
            "helper": (root / "helpers" / "layout.py", "TILE = 128\n"),
            "solution": (
                root / "solution.json",
                json.dumps(
                    {
                        "name": "teacher",
                        "spec": {
                            "languages": languages or ["pytorch", "cutedsl"],
                            "target_hardware": ["H20"],
                            "entry_point": entry_point,
                            "dependencies": ["torch", "nvidia-cutlass-dsl"],
                            "destination_passing_style": True,
                        },
                        "sources": [
                            {"path": path}
                            for path in (source_paths or ["kernel.py", "helpers/layout.py"])
                        ],
                    },
                    indent=2,
                )
                + "\n",
            ),
            "provenance": (
                root / "provenance.json",
                json.dumps(
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
                        "target": {
                            "framework": provenance_framework,
                            "architecture": provenance_architecture,
                        },
                        "knowledge_deny": {
                            "sources": ["flashinfer"],
                            "paths": [],
                            "tags": ["gdn", "gdn_decode"],
                        },
                    },
                    indent=2,
                )
                + "\n",
            ),
        }
        for key in creation_order:
            path, content = payloads[key]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

    def test_valid_bundle_returns_stable_private_metadata(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-bundle-") as temp_dir:
            root = Path(temp_dir) / "bundle"
            self._write_bundle(root)

            bundle = validate_teacher_bundle(
                root,
                expected_framework="CuteDSL",
                expected_architecture="sm90",
            )

        self.assertRegex(bundle.bundle_hash, r"^[0-9a-f]{64}$")
        self.assertEqual(bundle.teacher_id, "teacher_" + bundle.bundle_hash[:24])
        self.assertEqual(bundle.entry_point, "kernel.py::run")
        self.assertEqual(bundle.provenance.target.framework, "CuteDSL")
        self.assertEqual(
            bundle.source_paths,
            ("helpers/layout.py", "kernel.py"),
        )
        self.assertNotIn(str(root), bundle.public_identity())
        self.assertNotIn("flashinfer", bundle.public_identity())

    def test_hash_is_independent_of_file_creation_order(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-bundle-order-") as temp_dir:
            first = Path(temp_dir) / "first"
            second = Path(temp_dir) / "second"
            self._write_bundle(first)
            self._write_bundle(
                second,
                creation_order=("provenance", "solution", "helper", "kernel"),
            )

            first_hash = validate_teacher_bundle(first, "CuteDSL", "sm90").bundle_hash
            second_hash = validate_teacher_bundle(second, "CuteDSL", "sm90").bundle_hash

        self.assertEqual(first_hash, second_hash)

    def test_any_source_mutation_changes_the_bundle_hash(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-bundle-mutation-") as temp_dir:
            root = Path(temp_dir) / "bundle"
            self._write_bundle(root)
            before = validate_teacher_bundle(root, "CuteDSL", "sm90").bundle_hash
            (root / "helpers" / "layout.py").write_text("TILE = 256\n", encoding="utf-8")
            after = validate_teacher_bundle(root, "CuteDSL", "sm90").bundle_hash

        self.assertNotEqual(before, after)

    def test_missing_required_files_fail_closed(self) -> None:
        for filename in ("kernel.py", "solution.json", "provenance.json"):
            with self.subTest(filename=filename):
                with tempfile.TemporaryDirectory(prefix="teacher-bundle-missing-") as temp_dir:
                    root = Path(temp_dir) / "bundle"
                    self._write_bundle(root)
                    (root / filename).unlink()
                    with self.assertRaisesRegex(ValueError, filename.replace(".", r"\.")):
                        validate_teacher_bundle(root, "CuteDSL", "sm90")

    def test_solution_paths_must_be_relative_files_inside_the_bundle(self) -> None:
        cases = ("../secret.py", "/tmp/secret.py", "helpers/missing.py", "helpers\\layout.py")
        for source_path in cases:
            with self.subTest(source_path=source_path):
                with tempfile.TemporaryDirectory(prefix="teacher-bundle-path-") as temp_dir:
                    root = Path(temp_dir) / "bundle"
                    self._write_bundle(root, source_paths=["kernel.py", source_path])
                    with self.assertRaisesRegex(ValueError, "source path"):
                        validate_teacher_bundle(root, "CuteDSL", "sm90")

    def test_entry_point_must_reference_supported_kernel_symbol(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-bundle-model-entry-") as temp_dir:
            root = Path(temp_dir) / "bundle"
            self._write_bundle(root, entry_point="kernel.py::Model")
            bundle = validate_teacher_bundle(root, "CuteDSL", "sm90")
            self.assertEqual(bundle.entry_point, "kernel.py::Model")

        for entry_point in ("helpers/layout.py::run", "kernel.py::main", "kernel.py", "/kernel.py::run"):
            with self.subTest(entry_point=entry_point):
                with tempfile.TemporaryDirectory(prefix="teacher-bundle-entry-") as temp_dir:
                    root = Path(temp_dir) / "bundle"
                    self._write_bundle(root, entry_point=entry_point)
                    with self.assertRaisesRegex(ValueError, "entry_point"):
                        validate_teacher_bundle(root, "CuteDSL", "sm90")

    def test_framework_and_architecture_must_match_the_campaign(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-bundle-target-") as temp_dir:
            root = Path(temp_dir) / "bundle"
            self._write_bundle(root)
            with self.assertRaisesRegex(ValueError, "framework"):
                validate_teacher_bundle(root, "Triton", "sm90")
            with self.assertRaisesRegex(ValueError, "architecture"):
                validate_teacher_bundle(root, "CuteDSL", "sm120")

        with tempfile.TemporaryDirectory(prefix="teacher-bundle-language-") as temp_dir:
            root = Path(temp_dir) / "bundle"
            self._write_bundle(root, languages=["pytorch", "triton"])
            with self.assertRaisesRegex(ValueError, "languages"):
                validate_teacher_bundle(root, "CuteDSL", "sm90")

    def test_symlinks_and_unlisted_files_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-bundle-symlink-") as temp_dir:
            root = Path(temp_dir) / "bundle"
            self._write_bundle(root)
            outside = Path(temp_dir) / "outside.py"
            outside.write_text("OUTSIDE = True\n", encoding="utf-8")
            try:
                os.symlink(outside, root / "helpers" / "outside.py")
            except (OSError, NotImplementedError):
                self.skipTest("symlinks are not available on this platform")
            with self.assertRaisesRegex(ValueError, "symlink"):
                validate_teacher_bundle(root, "CuteDSL", "sm90")

        with tempfile.TemporaryDirectory(prefix="teacher-bundle-unlisted-") as temp_dir:
            root = Path(temp_dir) / "bundle"
            self._write_bundle(root)
            (root / "helpers" / "unused.py").write_text("UNUSED = True\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not declared"):
                validate_teacher_bundle(root, "CuteDSL", "sm90")

    def test_duplicate_json_keys_and_extra_top_level_files_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-bundle-json-") as temp_dir:
            root = Path(temp_dir) / "bundle"
            self._write_bundle(root)
            (root / "solution.json").write_text(
                '{"spec": {}, "spec": {}, "sources": []}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                validate_teacher_bundle(root, "CuteDSL", "sm90")

        with tempfile.TemporaryDirectory(prefix="teacher-bundle-extra-") as temp_dir:
            root = Path(temp_dir) / "bundle"
            self._write_bundle(root)
            (root / "README.md").write_text("not part of the bundle contract\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "top-level"):
                validate_teacher_bundle(root, "CuteDSL", "sm90")


if __name__ == "__main__":
    unittest.main()
