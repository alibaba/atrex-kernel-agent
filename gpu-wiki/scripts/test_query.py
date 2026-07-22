import contextlib
import io
import json
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

    def write_manifest(self, defaults, entries=None):
        path = self.root / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({
                "version": 1,
                "reference-kernels": {"defaults": defaults, "entries": entries or {}},
            }),
            encoding="utf-8",
        )

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

    def test_blackwell_family_and_sm100_exact_queries_are_distinct(self):
        _, family = self.run_query("--arch", "blackwell", "--section", "hardware-specs")
        self.assertIn("hardware_specs_b200.md", family)
        self.assertIn("hardware_specs_b300.md", family)
        self.assertIn("arch=blackwell-family", family)

        _, sm100 = self.run_query("--arch", "sm100", "--section", "hardware-specs")
        self.assertIn("hardware_specs_b200.md", sm100)
        self.assertNotIn("hardware_specs_b300.md", sm100)
        self.assertIn("arch=blackwell", sm100)

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

    def test_fuzzy_operator_typo_matches_after_architecture_filtering(self):
        pages = {
            "nvidia/hopper/ref-docs/cutedsl/rmsnorm-optimization.md":
                "# Hopper RMSNorm Optimization\n\nMemory-bound reduction guidance.\n",
            "amd/cdna4/ref-docs/gluon/rmsnorm-optimization.md":
                "# MI355X RMSNorm Optimization\n\nAMD-specific reduction guidance.\n",
        }
        for relative, content in pages.items():
            path = self.root / "docs" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        _, strict = self.run_query("rms_nrom", "--arch", "h20")
        self.assertNotIn("rmsnorm-optimization.md", strict)

        code, fuzzy = self.run_query("rms_nrom", "--arch", "h20", "--fuzzy")
        self.assertEqual(code, 0)
        self.assertIn("nvidia/hopper/ref-docs/cutedsl/rmsnorm-optimization.md", fuzzy)
        self.assertNotIn("amd/cdna4/ref-docs/gluon/rmsnorm-optimization.md", fuzzy)
        self.assertRegex(fuzzy, r"fuzzy=0\.\d+")

    def test_fuzzy_threshold_can_reject_a_weak_match(self):
        path = self.root / "docs/nvidia/hopper/ref-docs/cutedsl/rmsnorm-optimization.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Hopper RMSNorm Optimization\n", encoding="utf-8")

        _, output = self.run_query(
            "rms_nrom", "--arch", "h20", "--fuzzy", "--fuzzy-threshold", "0.95"
        )
        self.assertNotIn("rmsnorm-optimization.md", output)

    def test_reference_kernels_are_searched_with_architecture_and_dsl_isolation(self):
        pages = {
            "nvidia/hopper/cutedsl/quack/rmsnorm.py":
                "def rmsnorm_kernel(x):\n    return x\n",
            "amd/cdna4/gluon/rmsnorm.py":
                "def rmsnorm_kernel(x):\n    return x\n",
        }
        for relative, content in pages.items():
            path = self.root / "reference-kernels" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        code, output = self.run_query("rmsnorm", "--arch", "h20", "--dsl", "cutedsl")
        self.assertEqual(code, 0)
        self.assertIn("reference-kernels/nvidia/hopper/cutedsl/quack/rmsnorm.py", output)
        self.assertNotIn("reference-kernels/amd/cdna4/gluon/rmsnorm.py", output)

    def test_area_filter_can_select_docs_or_reference_kernels(self):
        path = self.root / "reference-kernels/nvidia/hopper/cutedsl/quack/gemm.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def gemm_kernel():\n    pass\n", encoding="utf-8")

        _, docs = self.run_query("gemm", "--arch", "h20", "--area", "docs")
        self.assertIn("docs/generic/ref-docs/gemm-optimization.md", docs)
        self.assertNotIn("reference-kernels/", docs)

        _, references = self.run_query(
            "gemm", "--arch", "h20", "--area", "reference-kernels"
        )
        self.assertIn("reference-kernels/nvidia/hopper/cutedsl/quack/gemm.py", references)
        self.assertNotIn("docs/generic/ref-docs/gemm-optimization.md", references)

    def test_reference_product_marker_prevents_mi308x_kernel_leaking_to_mi300x(self):
        path = self.root / (
            "reference-kernels/amd/cdna3/flydsl/FlyDSL/"
            "flash_attn_func_mi308x.py"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def flash_attention():\n    pass\n", encoding="utf-8")

        _, mi300x = self.run_query("flash", "--arch", "mi300x")
        self.assertNotIn("flash_attn_func_mi308x.py", mi300x)

        _, mi308x = self.run_query("flash", "--arch", "mi308x")
        self.assertIn("flash_attn_func_mi308x.py", mi308x)

    def test_manifest_entry_overrides_prefix_metadata(self):
        relative = "nvidia/hopper/cutedsl/example.py"
        path = self.root / "reference-kernels" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def gemm():\n    pass\n", encoding="utf-8")
        self.write_manifest(
            [
                {
                    "prefix": "nvidia/hopper/",
                    "architectures": ["hopper"],
                    "vendors": ["nvidia"],
                },
                {
                    "prefix": "nvidia/hopper/cutedsl/",
                    "dsls": ["cutedsl"],
                    "status": "diagnostic-archive",
                },
            ],
            {
                relative: {
                    "operators": ["gemm"],
                    "status": "runnable",
                }
            },
        )

        _, runnable = self.run_query(
            "gemm", "--arch", "h20", "--dsl", "cutedsl", "--status", "runnable"
        )
        self.assertIn("nvidia/hopper/cutedsl/example.py", runnable)
        _, archived = self.run_query(
            "gemm", "--arch", "h20", "--status", "diagnostic-archive"
        )
        self.assertNotIn("nvidia/hopper/cutedsl/example.py", archived)

    def test_manifest_status_is_displayed_and_filterable(self):
        archived = self.root / (
            "reference-kernels/nvidia/blackwell-geforce/cuda/archive/gemm.py"
        )
        unclassified = self.root / (
            "reference-kernels/nvidia/blackwell-geforce/cuda/current/gemm.py"
        )
        for path in (archived, unclassified):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("def gemm():\n    pass\n", encoding="utf-8")
        self.write_manifest([
            {
                "prefix": "nvidia/blackwell-geforce/",
                "architectures": ["blackwell-geforce"],
                "vendors": ["nvidia"],
            },
            {
                "prefix": "nvidia/blackwell-geforce/cuda/",
                "dsls": ["cuda"],
            },
            {
                "prefix": "nvidia/blackwell-geforce/cuda/archive/",
                "status": "diagnostic-archive",
                "operators": ["gemm"],
            },
        ])

        code, output = self.run_query(
            "gemm", "--arch", "sm120", "--status", "diagnostic-archive"
        )
        self.assertEqual(0, code)
        self.assertIn("status=diagnostic-archive", output)
        self.assertIn("cuda/archive/gemm.py", output)
        self.assertNotIn("cuda/current/gemm.py", output)

    def test_auxiliary_reference_files_are_excluded_unless_requested(self):
        kernel = self.root / "reference-kernels/nvidia/hopper/cutedsl/gemm.py"
        test = self.root / "reference-kernels/nvidia/hopper/cutedsl/test_gemm.py"
        kernel.parent.mkdir(parents=True, exist_ok=True)
        kernel.write_text("def gemm():\n    pass\n", encoding="utf-8")
        test.write_text("def test_gemm():\n    pass\n", encoding="utf-8")

        _, normal = self.run_query("gemm", "--arch", "h20", "--area", "reference-kernels")
        self.assertIn("nvidia/hopper/cutedsl/gemm.py", normal)
        self.assertNotIn("test_gemm.py", normal)
        _, auxiliary = self.run_query(
            "gemm", "--arch", "h20", "--area", "reference-kernels",
            "--include-auxiliary",
        )
        self.assertIn("test_gemm.py", auxiliary)

    def test_docs_manifest_metadata_overrides_path_inference(self):
        path = self.root / "manifest.json"
        path.write_text(json.dumps({
            "version": 1,
            "docs": {
                "defaults": [{
                    "prefix": "nvidia/hopper/",
                    "architectures": ["ampere"],
                    "vendors": ["nvidia"],
                }],
                "entries": {},
            },
        }), encoding="utf-8")

        _, ampere = self.run_query("wgmma", "--arch", "a100", "--area", "docs")
        self.assertIn("nvidia/hopper/kernel-opt/hands-on/wgmma.md", ampere)
        _, hopper = self.run_query("wgmma", "--arch", "h20", "--area", "docs")
        self.assertNotIn("nvidia/hopper/kernel-opt/hands-on/wgmma.md", hopper)

    def test_invalid_manifest_path_and_value_fail_closed(self):
        reference = self.root / "reference-kernels/nvidia/hopper/cutedsl"
        reference.mkdir(parents=True, exist_ok=True)
        cases = (
            (
                [],
                {"nvidia/hopper/cutedsl/missing.py": {"architectures": ["hopper"]}},
                "unknown-entry",
            ),
            (
                [{"prefix": "nvidia/hopper/", "architectures": ["sm999"]}],
                {},
                "unknown-architectures",
            ),
            (
                [{"prefix": "nvidia/hopper/", "vendors": ["amd"]}],
                {},
                "vendor-path-conflict",
            ),
            (
                [{"prefix": "nvidia/hopper/", "architectures": ["cdna4"]}],
                {},
                "architecture-path-conflict",
            ),
            (
                [{"prefix": "nvidia/hopper/cutedsl/", "dsls": ["cuda"]}],
                {},
                "dsl-path-conflict",
            ),
        )
        for defaults, entries, expected in cases:
            with self.subTest(expected=expected):
                self.write_manifest(defaults, entries)
                code, output = self.run_query("--area", "reference-kernels")
                self.assertEqual(1, code)
                self.assertIn("invalid-reference-manifest", output)
                self.assertIn(expected, output)

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


class ReferenceManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pages = query.load_reference_pages(REPO_ROOT / "gpu-wiki/reference-kernels")
        cls.by_path = {page.rel_path: page for page in cls.pages}

    def test_live_manifest_loads_and_covers_the_reference_index(self):
        self.assertEqual(522, len(self.pages))
        self.assertEqual(
            499,
            sum(
                Path(page.rel_path).suffix.lower() in query.REFERENCE_SOURCE_SUFFIXES
                for page in self.pages
            ),
        )
        self.assertEqual(23, sum(page.kind == "guide" for page in self.pages))
        page = self.by_path[
            "nvidia/blackwell-geforce/cuda/nvfp4_prefill_gemm/"
            "prefill_mma_padded_sm120_experimental.cu"
        ]
        self.assertEqual("diagnostic-archive", page.status)
        self.assertEqual(frozenset({"cuda"}), page.dsls)
        self.assertIn("gemm", page.operators)

    def test_live_reference_metadata_has_no_missing_classification(self):
        for page in self.pages:
            self.assertIsNotNone(page.status, page.rel_path)
            self.assertIsNotNone(page.source, page.rel_path)
            self.assertIn(page.kind, query.REFERENCE_KINDS, page.rel_path)
            self.assertTrue(page.operators, page.rel_path)

    def test_target_architecture_ranks_before_parent_and_generic_sources(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            code = query.main([
                "gemm", "--root", str(REPO_ROOT / "gpu-wiki"),
                "--arch", "b300", "--area", "reference-kernels", "--limit", "5",
            ])
        self.assertEqual(0, code)
        result_lines = [line for line in output.getvalue().splitlines() if line.startswith("  [")]
        self.assertIn("nvidia/blackwell-ultra/", result_lines[0])
        self.assertIn("nvidia/blackwell-ultra/", result_lines[1])

    def test_source_directory_name_does_not_create_operator_false_positive(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            code = query.main([
                "attention", "--root", str(REPO_ROOT / "gpu-wiki"),
                "--arch", "h20", "--area", "reference-kernels", "--limit", "50",
            ])
        self.assertEqual(0, code)
        result = output.getvalue()
        self.assertNotIn("flash-attention/cross_entropy.py", result)
        self.assertNotIn("flash-attention/mlp.py", result)

    def test_manifest_selected_guides_are_indexed_but_navigation_readmes_are_not(self):
        self.assertIn(
            "nvidia/blackwell-geforce/cuda/nvfp4_splitk_gemv/"
            "omoexplore-kernel-notes.md",
            self.by_path,
        )
        self.assertIn(
            "nvidia/blackwell-geforce/cutedsl/gdn_chunk_fwd/README.md",
            self.by_path,
        )
        self.assertNotIn("nvidia/hopper/README.md", self.by_path)
        self.assertNotIn("generic/README.md", self.by_path)

        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            code = query.main([
                "reuse", "--root", str(REPO_ROOT / "gpu-wiki"),
                "--arch", "sm120", "--kind", "guide", "--limit", "50",
            ])
        self.assertEqual(0, code)
        self.assertIn("omoexplore-kernel-notes.md", output.getvalue())

    def test_live_blackwell_ultra_files_use_their_physical_scope(self):
        paths = (
            "nvidia/blackwell-ultra/cutedsl/flashinfer/"
            "dense_blockscaled_gemm_sm103.py",
            "nvidia/blackwell-ultra/cutedsl/cutlass/"
            "sm103_dense_blockscaled_gemm_persistent.py",
        )
        for path in paths:
            with self.subTest(path=path):
                page = self.by_path[path]
                self.assertEqual(frozenset({"blackwell-ultra"}), page.architectures)
                self.assertTrue(
                    query.matches_dimension(
                        page, query.ARCH_ALIASES, {"blackwell-ultra"}
                    )
                )
                self.assertFalse(
                    query.matches_dimension(
                        page, query.ARCH_ALIASES, {"blackwell-geforce"}
                    )
                )

        self.assertNotIn(
            "nvidia/blackwell-geforce/cutedsl/flashinfer/"
            "dense_blockscaled_gemm_sm103.py",
            self.by_path,
        )
        self.assertNotIn(
            "nvidia/blackwell/cutedsl/cutlass/"
            "sm103_dense_blockscaled_gemm_persistent.py",
            self.by_path,
        )

    def test_copied_filename_matches_without_fuzzy(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            code = query.main([
                "dense_blockscaled_gemm_sm103.py",
                "--root", str(REPO_ROOT / "gpu-wiki"),
                "--arch", "sm103",
                "--area", "reference-kernels",
                "--limit", "20",
            ])
        self.assertEqual(0, code)
        result = output.getvalue()
        self.assertIn(
            "nvidia/blackwell-ultra/cutedsl/flashinfer/"
            "dense_blockscaled_gemm_sm103.py",
            result,
        )
        self.assertNotIn("fuzzy=", result)

    def test_live_mi308x_entries_do_not_enter_mi300x_scope(self):
        paths = (
            "amd/cdna3/flydsl/FlyDSL/flash_attn_func_mi308x.py",
            "amd/cdna3/flydsl/FlyDSL/chunk_gdn_flydsl_operator.py",
            "amd/cdna/triton/chunk_gdn/chunk_gdn_triton_baseline.py",
        )
        for path in paths:
            with self.subTest(path=path):
                page = self.by_path[path]
                self.assertTrue(
                    query.matches_dimension(page, query.ARCH_ALIASES, {"mi308x"})
                )
                self.assertFalse(
                    query.matches_dimension(page, query.ARCH_ALIASES, {"mi300x"})
                )


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

    def test_top_level_manifest_supplies_live_docs_scope(self):
        defaults, entries = query.load_docs_manifest(self.docs)
        self.assertGreaterEqual(len(defaults), 14)
        self.assertGreaterEqual(len(entries), 10)
        pages = query.load_pages(self.docs)
        for page in pages:
            if page.rel_path.startswith("nvidia/"):
                self.assertEqual(frozenset({"nvidia"}), page.vendors, page.rel_path)
            elif page.rel_path.startswith("amd/"):
                self.assertEqual(frozenset({"amd"}), page.vendors, page.rel_path)

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

    def test_blackwell_fp32_peaks_match_official_hgx_totals(self):
        b200 = (REPO_ROOT / "gpu-wiki/docs/nvidia/blackwell/b200/hardware-specs/hardware_specs_b200.md").read_text(
            encoding="utf-8"
        )
        b300 = (REPO_ROOT / "gpu-wiki/docs/nvidia/blackwell-ultra/hardware-specs/hardware_specs_b300.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("| **FP32 (CUDA Core)** | 75.0 |", b200)
        self.assertIn("8× B200 provides 600 FP32 and 296 FP64 TFLOPS", b200)
        self.assertIn("| **FP32 (CUDA Core)** | 75 |", b300)
        self.assertIn("8× B300 provides 600 FP32 and 10 FP64 TFLOPS", b300)
        self.assertNotIn("| **FP32 (CUDA Core)** | 37.5 |", b200)
        self.assertNotIn("| **FP32 (CUDA Core)** | 37 |", b300)

    def test_b300_int8_peak_matches_official_hgx_total(self):
        b300 = (
            REPO_ROOT
            / "gpu-wiki/docs/nvidia/blackwell-ultra/hardware-specs/hardware_specs_b300.md"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "| **INT8 (Tensor Core)** | 187.5 TOPS | 375 TOPS |",
            b300,
        )
        self.assertIn("3 sparse INT8 POPS for 8 GPUs", b300)
        self.assertNotIn(
            "| **INT8 (Tensor Core)** | 4,500 TOPS | 9,000 TOPS |",
            b300,
        )

    def test_blackwell_compute_capability_limits_are_not_conflated(self):
        path = REPO_ROOT / "gpu-wiki/docs/nvidia/common/kernel-opt/nvidia-compute-capabilities.md"
        text = path.read_text(encoding="utf-8")
        self.assertIn(
            "| Max Resident Threads/SM | 1024 | 2048 | 1536 | 1536 | 2048 | 2048 | 1536 |",
            text,
        )
        self.assertIn(
            "| Max Resident Warps/SM | 32 | 64 | 48 | 48 | 64 | 64 | 48 |",
            text,
        )
        self.assertIn("| 10.3 | Blackwell Ultra | B300, GB300 |", text)
        self.assertIn("| 12.0 | Blackwell GeForce/workstation |", text)
        self.assertNotIn("Rubin (expected)", text)

        quack = (
            REPO_ROOT / "gpu-wiki/docs/nvidia/common/ref-docs/cutedsl/"
            "quack-architecture-overview.md"
        ).read_text(encoding="utf-8")
        sm120 = (
            REPO_ROOT / "gpu-wiki/docs/nvidia/blackwell-geforce/hardware-specs/"
            "hardware_specs_sm120.md"
        ).read_text(encoding="utf-8")
        self.assertIn("SM103 (B300)", quack)
        self.assertNotIn("SM100 (B200/B300)", quack)
        self.assertIn("SM100 / SM103 (B200 / B300)", sm120)
        self.assertNotIn("SM100 / SM100a (B200/B300)", sm120)

    def test_reference_readme_statistics_match_indexed_sources(self):
        text = (REPO_ROOT / "gpu-wiki/reference-kernels/README.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("| **Total** | **499** |", text)
        self.assertIn("| amd/cdna3 | 28 |", text)
        self.assertIn("## Indexed Source Statistics", text)

    def test_mi355x_dense_and_sparse_peaks_are_distinct(self):
        hardware = REPO_ROOT / "gpu-wiki/docs/amd/cdna4/hardware-specs/hardware_specs_mi355x.md"
        text = hardware.read_text(encoding="utf-8")
        self.assertIn("amd.com/en/products/accelerators/instinct/mi350/mi355x.html", text)
        self.assertIn("2.5 PFLOPS dense; 5 PFLOPS with structured sparsity", text)
        self.assertIn("~312.5 FLOPs/Byte dense; ~625 with structured sparsity", text)
        self.assertNotIn("5,033.2 TFLOPS (Matrix)", text)

        affected = "\n".join(
            path.read_text(encoding="utf-8", errors="ignore")
            for path in (REPO_ROOT / "gpu-wiki/docs/amd").rglob("*.md")
            if "mi355" in path.as_posix().lower() or "cdna4" in path.as_posix().lower()
        )
        self.assertNotRegex(affected, r"MI355X[^\n]*(?:Ridge|ridge)[^\n]*(?:245|629)")


if __name__ == "__main__":
    unittest.main()
