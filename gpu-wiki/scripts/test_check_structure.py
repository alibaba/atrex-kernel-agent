import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_index  # noqa: E402
import check_structure  # noqa: E402


class HelperTests(unittest.TestCase):
    def test_has_h1_true(self):
        self.assertTrue(check_structure.has_h1(["# Title", "", "text"]))

    def test_has_h1_false_when_only_h2(self):
        self.assertFalse(check_structure.has_h1(["## Section", "", "text"]))

    def test_h1_inside_code_fence_is_ignored(self):
        lines = ["## Section", "", "```", "# not a title", "```", "text"]
        self.assertFalse(check_structure.has_h1(lines))

    def test_has_summary_true(self):
        self.assertTrue(check_structure.has_summary(["# Title", "", "A one line summary.", "", "## Body"]))

    def test_has_summary_false_when_section_follows_title(self):
        self.assertFalse(check_structure.has_summary(["# Title", "", "## Body", "", "| a | b |"]))

    def test_related_heading_variants_accepted(self):
        for heading in ("## Related", "## Related Docs", "## Related Documents"):
            self.assertTrue(check_structure.has_related([heading]), heading)

    def test_related_missing(self):
        self.assertFalse(check_structure.has_related(["# Title", "", "## Body"]))


class ScanTests(unittest.TestCase):
    def make_wiki(self):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name) / "gpu-wiki"
        (root / "docs").mkdir(parents=True)
        self.addCleanup(temp.cleanup)
        return root

    def write(self, root, rel, text):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def codes_for(self, findings, rel):
        return {f.code for f in findings if f.path.name == Path(rel).name}

    def test_clean_page_has_no_findings(self):
        root = self.make_wiki()
        self.write(root, "docs/README.md", "[page](topic.md)\n")
        self.write(
            root,
            "docs/topic.md",
            "# Topic\n\nOne line summary.\n\n## Body\n\n## Related\n- [README](README.md)\n",
        )
        findings = check_structure.scan(root)
        self.assertEqual(set(), self.codes_for(findings, "docs/topic.md"))

    def test_missing_h1_and_summary_reported(self):
        root = self.make_wiki()
        self.write(root, "docs/README.md", "[page](topic.md)\n")
        self.write(root, "docs/topic.md", "## Body\n\n| a | b |\n")
        codes = self.codes_for(check_structure.scan(root), "docs/topic.md")
        self.assertIn("missing-h1", codes)
        self.assertIn("missing-summary", codes)

    def test_orphan_page_reported(self):
        root = self.make_wiki()
        self.write(root, "docs/README.md", "no links here\n")
        self.write(root, "docs/lonely.md", "# Lonely\n\nSummary.\n\n## Related\n- [README](README.md)\n")
        self.assertIn("orphan-page", self.codes_for(check_structure.scan(root), "docs/lonely.md"))

    def test_index_does_not_rescue_orphan(self):
        root = self.make_wiki()
        self.write(root, "docs/README.md", "no links here\n")
        self.write(root, "docs/index.md", "- [lonely](lonely.md)\n")
        self.write(root, "docs/lonely.md", "# Lonely\n\nSummary.\n\n## Related\n- [README](README.md)\n")
        self.assertIn("orphan-page", self.codes_for(check_structure.scan(root), "docs/lonely.md"))

    def test_gating_codes_membership(self):
        self.assertIn("missing-h1", check_structure.GATING_CODES)
        self.assertIn("missing-summary", check_structure.GATING_CODES)
        self.assertNotIn("missing-related", check_structure.GATING_CODES)
        self.assertNotIn("orphan-page", check_structure.GATING_CODES)


if __name__ == "__main__":
    unittest.main()
