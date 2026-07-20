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

REPO_ROOT = SCRIPTS_DIR.parents[1]


class QueryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        pages = {
            "kernel-opt/nvidia/common/blackwell/patterns/pipeline-stalls.md":
                "# Pipeline Stalls\n\nTMA and tcgen05 pipeline diagnosis.\n",
            "kernel-opt/nvidia/common/sm90/hands-on/wgmma.md":
                "# Hopper WGMMA GEMM\n\nA Hopper implementation.\n",
            "ref-docs/nvidia/common/sm80/a100-gemm.md":
                "# A100 GEMM\n\nAn Ampere implementation.\n",
            "kernel-opt/nvidia/common/hands-on/tcgen05.md":
                "# TCGEN05 and TMEM\n\nThis directory is scoped to SM100 by its README.\n",
            "ref-docs/nvidia/cutedsl/sm120/gdn.md":
                "# SM120 Blackwell GDN CuTeDSL\n\nA gated delta net kernel.\n",
            "ref-docs/amd/flydsl/gfx942/flash-attention.md":
                "# CDNA3 Flash Attention FlyDSL\n\nAn AMD attention kernel.\n",
            "ref-docs/amd/gluon/gfx950/matmul.md":
                "# CDNA4 Gluon Matmul\n\nAn MI355X implementation.\n",
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

    def test_h20_a100_and_amd_aliases_route_to_their_families(self):
        _, h20 = self.run_query("--arch", "h20", "--vendor", "nvidia")
        self.assertIn("sm90/hands-on/wgmma.md", h20)
        self.assertNotIn("sm80/a100-gemm.md", h20)
        self.assertNotIn("sm120/gdn.md", h20)

        _, a100 = self.run_query("--arch", "a100", "--vendor", "nvidia")
        self.assertIn("sm80/a100-gemm.md", a100)
        self.assertNotIn("sm90/hands-on/wgmma.md", a100)
        self.assertNotIn("common/hands-on/tcgen05.md", a100)

        _, mi300x = self.run_query("--arch", "mi300x", "--vendor", "amd")
        self.assertIn("gfx942/flash-attention.md", mi300x)
        self.assertNotIn("gfx950/matmul.md", mi300x)

        _, mi355x = self.run_query("--arch", "mi355x", "--vendor", "amd")
        self.assertIn("gfx950/matmul.md", mi355x)
        self.assertNotIn("gfx942/flash-attention.md", mi355x)

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

    def test_pro5000_and_sm120_aliases_resolve_to_same_scope(self):
        aliases = ("pro5000", "pro-5000", "rtx pro 5000", "sm120", "sm_120")
        outputs = []
        for alias in aliases:
            code, output = self.run_query("--arch", alias, "--vendor", "nvidia")
            self.assertEqual(code, 0, alias)
            self.assertIn("arch=blackwell-geforce", output, alias)
            self.assertIn("sm120/gdn.md", output, alias)
            outputs.append(output)
        self.assertTrue(all(output == outputs[0] for output in outputs[1:]))


class Pro5000KnowledgeTests(unittest.TestCase):
    def test_official_pro5000_facts_and_sources_are_consistent(self):
        hardware = REPO_ROOT / "gpu-wiki/docs/hardware-specs/hardware_specs_sm120.md"
        text = hardware.read_text(encoding="utf-8")
        self.assertIn("professional-desktop-gpus/rtx-pro-5000/", text)
        self.assertIn("workstation-datasheet-blackwell-rtx-pro-5000", text)
        self.assertIn("| Memory Interface | 512-bit |", text)
        self.assertIn("1.344 TB/s (datasheet: 1,344 GB/s)", text)

        docs = REPO_ROOT / "gpu-wiki/docs"
        conflicts = []
        for path in docs.rglob("*.md"):
            page = path.read_text(encoding="utf-8", errors="ignore")
            if "PRO 5000" in page and "GDDR7 384-bit" in page:
                conflicts.append(path.relative_to(REPO_ROOT).as_posix())
        self.assertEqual(conflicts, [], "conflicting Pro5000 memory-interface claims")

    def test_pro5000_scope_excludes_sm100_gdn_documents(self):
        pages = query.load_pages(REPO_ROOT / "gpu-wiki/docs")
        scoped = {
            page.rel_path
            for page in pages
            if query.matches_dimension(page, query.ARCH_ALIASES, {"blackwell-geforce"})
            and query.matches_dimension(page, query.VENDOR_ALIASES, {"nvidia"})
        }
        self.assertFalse(any("/sm100/" in path for path in scoped))
        self.assertNotIn(
            "ref-docs/nvidia/common/sm100/qwen3.5-gdn-prefill-kernel-optimization.md",
            scoped,
        )
        self.assertNotIn(
            "ref-docs/nvidia/common/sm100/gdn-decode-kernel-no-tensor-core.md",
            scoped,
        )


if __name__ == "__main__":
    unittest.main()
