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
            "nvidia/blackwell/kernel-opt/patterns/pipeline-stalls.md":
                "# Pipeline Stalls\n\nTMA and tcgen05 pipeline diagnosis.\n",
            "nvidia/hopper/kernel-opt/hands-on/wgmma.md":
                "# Hopper WGMMA GEMM\n\nA Hopper implementation.\n",
            "nvidia/hopper/kernel-opt/hands-on/software-pipeline.md":
                "# Hopper Software Pipeline\n\nPipeline staging on SM90.\n",
            "nvidia/ampere/ref-docs/a100-gemm.md":
                "# A100 GEMM\n\nAn Ampere implementation.\n",
            "nvidia/blackwell/kernel-opt/hands-on/tcgen05.md":
                "# TCGEN05 and TMEM\n\nThis directory is physically scoped to SM100.\n",
            "nvidia/blackwell-geforce/ref-docs/cutedsl/gdn.md":
                "# SM120 Blackwell GDN CuTeDSL\n\nA gated delta net kernel.\n",
            "amd/cdna3/ref-docs/flydsl/flash-attention.md":
                "# CDNA3 Flash Attention FlyDSL\n\nAn AMD attention kernel.\n",
            "amd/cdna4/ref-docs/gluon/matmul.md":
                "# CDNA4 Gluon Matmul\n\nAn MI355X implementation.\n",
            "amd/cdna3/mi300x/hardware-specs/hardware_specs_mi300x.md":
                "# MI300X Hardware Specifications\n",
            "amd/cdna3/mi308x/hardware-specs/hardware_specs_mi308x.md":
                "# MI308X Hardware Specifications\n",
            "nvidia/blackwell/b200/hardware-specs/hardware_specs_b200.md":
                "# B200 Hardware Specifications\n",
            "nvidia/blackwell-ultra/hardware-specs/hardware_specs_b300.md":
                "# B300 Hardware Specifications\n",
            "amd/cdna3/kernel-opt/flash-attention-tilelang.md":
                "# CDNA3 Flash Attention TileLang\n\nA different DSL.\n",
            "generic/ref-docs/gemm-optimization.md":
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
        self.assertIn("generic/ref-docs/gemm-optimization.md", output)
        self.assertNotIn("sm90/hands-on/wgmma.md", output)
        self.assertNotIn("sm120/gdn.md", output)
        self.assertNotIn("gfx942/flash-attention.md", output)

    def test_symptom_selects_stable_diagnosis_card(self):
        code, output = self.run_query("--arch", "sm100", "--symptom", "pipeline-stalls")
        self.assertEqual(code, 0)
        self.assertIn("patterns/pipeline-stalls.md", output)
        self.assertNotIn("gdn.md", output)

    def test_symptom_uses_stable_keywords_outside_blackwell_cards(self):
        code, output = self.run_query("--arch", "h20", "--symptom", "pipeline-stalls")
        self.assertEqual(code, 0)
        self.assertIn("hopper/kernel-opt/hands-on/software-pipeline.md", output)
        self.assertNotIn("blackwell/kernel-opt/patterns/pipeline-stalls.md", output)

    def test_directory_level_architecture_scope_is_enforced(self):
        _, blackwell = self.run_query("--arch", "b200", "--vendor", "nvidia")
        _, hopper = self.run_query("--arch", "sm90", "--vendor", "nvidia")
        self.assertIn("blackwell/kernel-opt/hands-on/tcgen05.md", blackwell)
        self.assertNotIn("blackwell/kernel-opt/hands-on/tcgen05.md", hopper)

    def test_h20_a100_and_amd_aliases_route_to_their_families(self):
        _, h20 = self.run_query("--arch", "h20", "--vendor", "nvidia")
        self.assertIn("hopper/kernel-opt/hands-on/wgmma.md", h20)
        self.assertNotIn("ampere/ref-docs/a100-gemm.md", h20)
        self.assertNotIn("blackwell-geforce/ref-docs/cutedsl/gdn.md", h20)

        _, a100 = self.run_query("--arch", "a100", "--vendor", "nvidia")
        self.assertIn("ampere/ref-docs/a100-gemm.md", a100)
        self.assertNotIn("hopper/kernel-opt/hands-on/wgmma.md", a100)
        self.assertNotIn("blackwell/kernel-opt/hands-on/tcgen05.md", a100)

        _, mi300x = self.run_query("--arch", "mi300x", "--vendor", "amd")
        self.assertIn("cdna3/ref-docs/flydsl/flash-attention.md", mi300x)
        self.assertNotIn("cdna4/ref-docs/gluon/matmul.md", mi300x)

        _, mi355x = self.run_query("--arch", "mi355x", "--vendor", "amd")
        self.assertIn("cdna4/ref-docs/gluon/matmul.md", mi355x)
        self.assertNotIn("cdna3/ref-docs/flydsl/flash-attention.md", mi355x)

    def test_architecture_implies_vendor_when_vendor_is_omitted(self):
        _, a100 = self.run_query("--arch", "a100")
        self.assertIn("vendor=nvidia", a100)
        self.assertNotIn("docs/amd/", a100)

        _, mi355x = self.run_query("--arch", "mi355x")
        self.assertIn("vendor=amd", mi355x)
        self.assertNotIn("docs/nvidia/", mi355x)

    def test_sm120_only_legacy_pitfalls_do_not_leak_to_hopper(self):
        pages = {
            "nvidia/blackwell-geforce/pitfalls/cutedsl/gdn-decode-pitfalls.md":
                "# CuTeDSL GDN Decode on sm_120 — Pitfalls\n",
        }
        for relative, content in pages.items():
            path = self.root / "docs" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        _, hopper = self.run_query("gdn", "--arch", "h20", "--section", "pitfalls")
        self.assertNotIn("gdn-decode-pitfalls.md", hopper)

        _, sm120 = self.run_query("gdn", "--arch", "sm120", "--section", "pitfalls")
        self.assertIn("gdn-decode-pitfalls.md", sm120)

    def test_product_specific_hardware_pages_exclude_sibling_products(self):
        _, mi300x = self.run_query("--arch", "mi300x", "--section", "hardware-specs")
        self.assertIn("hardware_specs_mi300x.md", mi300x)
        self.assertNotIn("hardware_specs_mi308x.md", mi300x)

        _, mi308x = self.run_query("--arch", "mi308x", "--section", "hardware-specs")
        self.assertIn("hardware_specs_mi308x.md", mi308x)
        self.assertNotIn("hardware_specs_mi300x.md", mi308x)

        _, b200 = self.run_query("--arch", "b200", "--section", "hardware-specs")
        self.assertIn("hardware_specs_b200.md", b200)
        self.assertNotIn("hardware_specs_b300.md", b200)

        _, b300 = self.run_query("--arch", "b300", "--section", "hardware-specs")
        self.assertIn("hardware_specs_b300.md", b300)
        self.assertNotIn("hardware_specs_b200.md", b300)

    def test_architecture_family_query_keeps_cdna3_products(self):
        _, output = self.run_query("--arch", "gfx942", "--section", "hardware-specs")
        self.assertIn("hardware_specs_mi300x.md", output)
        self.assertIn("hardware_specs_mi308x.md", output)

    def test_amd_product_specific_pitfalls_do_not_leak_to_sibling_products(self):
        pages = {
            "amd/cdna3/mi308x/pitfalls/flydsl/flash-attn-pitfalls.md":
                "# FlyDSL Flash Attention on MI308X\n",
            "amd/cdna4/pitfalls/flydsl/chunk-gdn-pitfalls.md":
                "# FlyDSL Chunk GDN on MI355X\n",
        }
        for relative, content in pages.items():
            path = self.root / "docs" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        _, mi300x = self.run_query("--arch", "mi300x", "--section", "pitfalls")
        self.assertNotIn("flash-attn-pitfalls.md", mi300x)
        self.assertNotIn("chunk-gdn-pitfalls.md", mi300x)

        _, mi308x = self.run_query("--arch", "mi308x", "--section", "pitfalls")
        self.assertIn("flash-attn-pitfalls.md", mi308x)
        self.assertNotIn("chunk-gdn-pitfalls.md", mi308x)

        _, mi355x = self.run_query("--arch", "mi355x", "--section", "pitfalls")
        self.assertIn("chunk-gdn-pitfalls.md", mi355x)
        self.assertNotIn("flash-attn-pitfalls.md", mi355x)

    def test_hopper_only_cluster_card_does_not_enter_a100_scope(self):
        path = self.root / "docs/nvidia/common/kernel-opt/thread-block-cluster.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Thread Block Cluster\n", encoding="utf-8")

        _, a100 = self.run_query("cluster", "--arch", "a100")
        self.assertNotIn("thread-block-cluster.md", a100)
        _, h20 = self.run_query("cluster", "--arch", "h20")
        self.assertIn("thread-block-cluster.md", h20)

    def test_b200_experiment_pages_do_not_enter_b300_scope(self):
        path = self.root / "docs/nvidia/blackwell/b200/pitfalls/triton/sm100-sparse-decode-split-k-pitfalls.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# B200 sparse decode pitfalls\n", encoding="utf-8")

        _, b200 = self.run_query("sparse", "--arch", "b200")
        self.assertIn("sm100-sparse-decode-split-k-pitfalls.md", b200)
        _, b300 = self.run_query("sparse", "--arch", "b300")
        self.assertNotIn("sm100-sparse-decode-split-k-pitfalls.md", b300)

    def test_operator_dsl_and_section_filters_compose(self):
        code, output = self.run_query(
            "--arch", "sm120", "--vendor", "nvidia", "--dsl", "cutedsl",
            "--section", "ref-docs", "--operator", "gdn",
        )
        self.assertEqual(code, 0)
        self.assertIn("blackwell-geforce/ref-docs/cutedsl/gdn.md", output)
        self.assertNotIn("generic/ref-docs/gemm-optimization.md", output)

    def test_dsl_scope_excludes_competing_dsl_filename(self):
        code, output = self.run_query(
            "attention", "--arch", "gfx942", "--vendor", "amd", "--dsl", "flydsl"
        )
        self.assertEqual(code, 0)
        self.assertIn("cdna3/ref-docs/flydsl/flash-attention.md", output)
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
            self.assertIn("blackwell-geforce/ref-docs/cutedsl/gdn.md", output, alias)
            outputs.append(output)
        self.assertTrue(all(output == outputs[0] for output in outputs[1:]))


class ArchitectureFirstLayoutTests(unittest.TestCase):
    @property
    def docs(self):
        return REPO_ROOT / "gpu-wiki/docs"

    def scoped_paths(self, architecture, vendor):
        return {
            page.rel_path
            for page in query.load_pages(self.docs)
            if query.matches_dimension(page, query.ARCH_ALIASES, {architecture})
            and query.matches_dimension(page, query.VENDOR_ALIASES, {vendor})
        }

    def test_every_searchable_document_has_architecture_first_scope_and_role(self):
        self.assertEqual(
            {"amd", "generic", "nvidia"},
            {path.name for path in self.docs.iterdir() if path.is_dir()},
        )
        content_files = {
            path.relative_to(self.docs).as_posix()
            for path in self.docs.rglob("*.md")
            if path.name != "README.md" and path.parent != self.docs
        }
        pages = query.load_pages(self.docs)
        self.assertEqual(content_files, {page.rel_path for page in pages})
        self.assertEqual(344, len(pages))
        for page in pages:
            self.assertIn(page.segments[0], {"amd", "generic", "nvidia"})
            self.assertIsNotNone(query.section_value(page), page.rel_path)

    def test_live_hardware_aliases_route_to_exact_physical_scope(self):
        expected = {
            "a100": "nvidia/ampere/hardware-specs/hardware_specs_ampere.md",
            "h20": "nvidia/hopper/hardware-specs/hardware_specs_hopper.md",
            "b200": "nvidia/blackwell/b200/hardware-specs/hardware_specs_b200.md",
            "b300": "nvidia/blackwell-ultra/hardware-specs/hardware_specs_b300.md",
            "pro5000": "nvidia/blackwell-geforce/hardware-specs/hardware_specs_sm120.md",
            "sm120": "nvidia/blackwell-geforce/hardware-specs/hardware_specs_sm120.md",
            "mi300x": "amd/cdna3/mi300x/hardware-specs/hardware_specs_mi300x.md",
            "mi308x": "amd/cdna3/mi308x/hardware-specs/hardware_specs_mi308x.md",
            "mi355x": "amd/cdna4/hardware-specs/hardware_specs_mi355x.md",
        }
        for alias, path in expected.items():
            output = io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                code = query.main([
                    "--root", str(REPO_ROOT / "gpu-wiki"),
                    "--arch", alias,
                    "--section", "hardware-specs",
                ])
            self.assertEqual(0, code, alias)
            self.assertIn(f"docs/{path}", output.getvalue(), alias)

    def test_live_product_scopes_inherit_parent_but_exclude_siblings(self):
        b200 = self.scoped_paths("b200", "nvidia")
        b300 = self.scoped_paths("blackwell-ultra", "nvidia")
        pro5000 = self.scoped_paths("blackwell-geforce", "nvidia")
        mi300x = self.scoped_paths("mi300x", "amd")
        mi308x = self.scoped_paths("mi308x", "amd")

        self.assertTrue(any(path.startswith("nvidia/blackwell/kernel-opt/") for path in b200))
        self.assertTrue(any(path.startswith("nvidia/blackwell/b200/") for path in b200))
        self.assertTrue(any(path.startswith("nvidia/blackwell/kernel-opt/") for path in b300))
        self.assertFalse(any(path.startswith("nvidia/blackwell/b200/") for path in b300))
        self.assertFalse(any(path.startswith("nvidia/blackwell/") for path in pro5000))
        self.assertFalse(any("/mi308x/" in path for path in mi300x))
        self.assertFalse(any("/mi300x/" in path for path in mi308x))


class Pro5000KnowledgeTests(unittest.TestCase):
    def test_official_pro5000_facts_and_sources_are_consistent(self):
        hardware = REPO_ROOT / "gpu-wiki/docs/nvidia/blackwell-geforce/hardware-specs/hardware_specs_sm120.md"
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
        self.assertFalse(any(path.startswith("nvidia/blackwell/") for path in scoped))
        self.assertNotIn(
            "nvidia/blackwell/ref-docs/qwen3.5-gdn-prefill-kernel-optimization.md",
            scoped,
        )
        self.assertNotIn(
            "nvidia/blackwell/b200/ref-docs/gdn-decode-kernel-no-tensor-core.md",
            scoped,
        )


class HardwareKnowledgeTests(unittest.TestCase):
    def test_b200_shared_memory_matches_official_tuning_guide(self):
        path = REPO_ROOT / "gpu-wiki/docs/nvidia/blackwell/b200/hardware-specs/hardware_specs_b200.md"
        text = path.read_text(encoding="utf-8")
        self.assertIn("256 KB per SM", text)
        self.assertIn("Up to 228 KB per SM", text)
        self.assertIn("227 KB addressable per block", text)
        self.assertNotIn("128 KB physical pool", text)

    def test_b200_examples_use_one_explicit_target_configuration(self):
        relative_paths = (
            "gpu-wiki/docs/nvidia/blackwell/kernel-opt/hardware/clc.md",
            "gpu-wiki/docs/nvidia/blackwell/kernel-opt/patterns/tail-effect.md",
            "gpu-wiki/docs/nvidia/blackwell/kernel-opt/techniques/tile-scheduling.md",
        )
        text = "\n".join(
            (REPO_ROOT / relative).read_text(encoding="utf-8")
            for relative in relative_paths
        )
        self.assertIn("148-SM B200", text)
        self.assertNotRegex(text, r"B200[^\n]*(?:132|142) SM")


if __name__ == "__main__":
    unittest.main()
