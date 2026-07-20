import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import query  # noqa: E402


class QueryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        pages = {
            "kernel-opt/nvidia/common/blackwell/patterns/pipeline-stalls.md":
                "# Pipeline Stalls\n\nTMA and tcgen05 pipeline diagnosis.\n",
            "kernel-opt/nvidia/common/sm90/hands-on/wgmma.md":
                "# Hopper WGMMA GEMM\n\nA Hopper implementation.\n",
            "kernel-opt/nvidia/common/hands-on/tcgen05.md":
                "# TCGEN05 and TMEM\n\nThis directory is scoped to SM100 by its README.\n",
            "ref-docs/nvidia/cutedsl/sm120/gdn.md":
                "# SM120 Blackwell GDN CuTeDSL\n\nA gated delta net kernel.\n",
            "ref-docs/amd/flydsl/gfx942/flash-attention.md":
                "# CDNA3 Flash Attention FlyDSL\n\nAn AMD attention kernel.\n",
            "kernel-opt/amd/common/gfx942/flash-attention-tilelang.md":
                "# CDNA3 Flash Attention TileLang\n\nA different DSL.\n",
            "ref-docs/generic/gemm-optimization.md":
                "# GEMM Optimization\n\nArchitecture-neutral tiling.\n",
        }
        for relative, content in pages.items():
            path = self.root / "docs" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def run_query(self, *args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            code = query.main([*args, "--root", str(self.root)])
        return code, output.getvalue()

    def test_architecture_scope_excludes_other_architectures_but_keeps_generic(self):
        code, output = self.run_query("gemm", "--arch", "b200", "--vendor", "nvidia")
        self.assertEqual(code, 0)
        self.assertIn("generic/gemm-optimization.md", output)
        self.assertNotIn("sm90/hands-on/wgmma.md", output)
        self.assertNotIn("sm120/gdn.md", output)
        self.assertNotIn("gfx942/flash-attention.md", output)

    def test_symptom_selects_stable_diagnosis_card(self):
        code, output = self.run_query("--arch", "sm100", "--symptom", "pipeline-stalls")
        self.assertEqual(code, 0)
        self.assertIn("patterns/pipeline-stalls.md", output)
        self.assertNotIn("gdn.md", output)

    def test_directory_level_architecture_scope_is_enforced(self):
        _, blackwell = self.run_query("--arch", "b200", "--vendor", "nvidia")
        _, hopper = self.run_query("--arch", "sm90", "--vendor", "nvidia")
        self.assertIn("common/hands-on/tcgen05.md", blackwell)
        self.assertNotIn("common/hands-on/tcgen05.md", hopper)

    def test_operator_dsl_and_section_filters_compose(self):
        code, output = self.run_query(
            "--arch", "sm120", "--vendor", "nvidia", "--dsl", "cutedsl",
            "--section", "ref-docs", "--operator", "gdn",
        )
        self.assertEqual(code, 0)
        self.assertIn("cutedsl/sm120/gdn.md", output)
        self.assertNotIn("generic/gemm-optimization.md", output)

    def test_dsl_scope_excludes_competing_dsl_filename(self):
        code, output = self.run_query(
            "attention", "--arch", "gfx942", "--vendor", "amd", "--dsl", "flydsl"
        )
        self.assertEqual(code, 0)
        self.assertIn("flydsl/gfx942/flash-attention.md", output)
        self.assertNotIn("flash-attention-tilelang.md", output)

    def test_unknown_arch_fails_closed(self):
        code, output = self.run_query("--arch", "sm999")
        self.assertEqual(code, 1)
        self.assertIn("unknown-arch", output)


if __name__ == "__main__":
    unittest.main()
