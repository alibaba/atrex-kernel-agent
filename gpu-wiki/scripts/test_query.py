import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import query  # noqa: E402


def make_page(rel, title="", summary="", body=""):
    return query.Page(
        rel_path=rel,
        title=title,
        summary=summary,
        segments=query.path_segments(rel),
        filename=rel.rsplit("/", 1)[-1].lower(),
        keyword_blob=query.keyword_blob(rel),
        body=body.lower(),
    )


class DimensionTests(unittest.TestCase):
    def test_arch_from_path_segment(self):
        page = make_page("nvidia/hopper/cutedsl/flash_fwd.md")
        self.assertEqual({"hopper"}, query.dimension_values(page, query.ARCH_ALIASES))

    def test_arch_from_filename(self):
        page = make_page("amd/hardware-specs/hardware_specs_mi300x.md")
        self.assertEqual({"cdna3"}, query.dimension_values(page, query.ARCH_ALIASES))

    def test_blackwell_and_geforce_are_distinct(self):
        # the whole point: blackwell must NOT match blackwell-geforce (sm120)
        geforce = make_page("nvidia/blackwell-geforce/cutedsl/sm120-gemm.md")
        self.assertEqual({"blackwell-geforce"}, query.dimension_values(geforce, query.ARCH_ALIASES))
        self.assertFalse(query.matches_dimension(geforce, query.ARCH_ALIASES, {"blackwell"}))
        self.assertTrue(query.matches_dimension(geforce, query.ARCH_ALIASES, {"blackwell-geforce"}))

        datacenter = make_page("nvidia/blackwell/techniques/swizzling.md")
        self.assertEqual({"blackwell"}, query.dimension_values(datacenter, query.ARCH_ALIASES))
        self.assertFalse(query.matches_dimension(datacenter, query.ARCH_ALIASES, {"blackwell-geforce"}))

    def test_neutral_page_has_no_arch(self):
        page = make_page("nvidia/common/ptx/ptx-instruction-set.md")
        self.assertEqual(set(), query.dimension_values(page, query.ARCH_ALIASES))

    def test_blackwell_filter_excludes_hopper(self):
        hopper = make_page("nvidia/hopper/hands-on/warp-specialization.md")
        self.assertFalse(query.matches_dimension(hopper, query.ARCH_ALIASES, {"blackwell"}))

    def test_neutral_survives_any_arch_filter(self):
        neutral = make_page("generic/gpu-execution-model.md")
        self.assertTrue(query.matches_dimension(neutral, query.ARCH_ALIASES, {"blackwell"}))
        self.assertTrue(query.matches_dimension(neutral, query.ARCH_ALIASES, {"hopper"}))

    def test_generic_is_vendor_neutral(self):
        generic = make_page("generic/gpu-execution-model.md")
        self.assertTrue(query.matches_dimension(generic, query.VENDOR_ALIASES, {"nvidia"}))
        amd = make_page("amd/common/amd-mfma-matrix-cores.md")
        self.assertFalse(query.matches_dimension(amd, query.VENDOR_ALIASES, {"nvidia"}))

    def test_resolve_arch_aliases(self):
        self.assertEqual("hopper", query.resolve_arch("sm90"))
        self.assertEqual("cdna3", query.resolve_arch("gfx942"))
        self.assertEqual("blackwell-geforce", query.resolve_arch("sm120"))
        self.assertEqual("blackwell", query.resolve_arch("blackwell"))
        self.assertIsNone(query.resolve_arch("nonsense"))


class ScoringTests(unittest.TestCase):
    def test_title_match_scores_higher_than_body(self):
        page = make_page("a.md", title="Bank conflict swizzle", summary="s", body="body text")
        self.assertEqual(query.TITLE_WEIGHT, query.score_page(page, ["bank"], match_any=False))

    def test_and_semantics_requires_all_terms(self):
        page = make_page("a.md", title="Bank conflict", summary="s", body="nothing else")
        self.assertEqual(0, query.score_page(page, ["bank", "missingterm"], match_any=False))

    def test_any_semantics_allows_partial(self):
        page = make_page("a.md", title="Bank conflict", summary="s", body="nothing else")
        self.assertGreater(query.score_page(page, ["bank", "missingterm"], match_any=True), 0)


class EndToEndTests(unittest.TestCase):
    def make_wiki(self):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name) / "gpu-wiki"
        (root / "docs").mkdir(parents=True)
        self.addCleanup(temp.cleanup)
        return root

    def write(self, root, rel, text):
        path = root / "docs" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def test_scoped_search_excludes_other_arch(self):
        root = self.make_wiki()
        self.write(root, "nvidia/blackwell/techniques/x.md", "# X\n\nUses tcgen05 pipeline.\n")
        self.write(root, "nvidia/hopper/hands-on/y.md", "# Y\n\nUses wgmma pipeline.\n")
        self.write(root, "nvidia/blackwell-geforce/cutedsl/z.md", "# Z\n\nsm120 nvfp4 gemm.\n")
        pages = query.load_pages(root / "docs")
        scoped = {p.rel_path for p in pages if query.matches_dimension(p, query.ARCH_ALIASES, {"blackwell"})}
        self.assertIn("nvidia/blackwell/techniques/x.md", scoped)
        self.assertNotIn("nvidia/hopper/hands-on/y.md", scoped)
        self.assertNotIn("nvidia/blackwell-geforce/cutedsl/z.md", scoped)

    def test_quoted_multiword_query_splits_into_terms(self):
        # a single quoted arg "moe gemm" must behave as two AND-ed keywords, not a
        # literal-phrase substring search
        root = self.make_wiki()
        self.write(root, "nvidia/blackwell/kernels/fused-moe.md", "# Fused MoE Dual GEMM\n\nRouting plus dual GEMM.\n")
        self.assertEqual(0, query.main(["moe gemm", "--root", str(root), "--vendor", "nvidia"]))
        page = query.load_pages(root / "docs")[0]
        self.assertGreater(query.score_page(page, ["moe", "gemm"], match_any=False), 0)


if __name__ == "__main__":
    unittest.main()
