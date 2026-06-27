import importlib.util
import tempfile
import textwrap
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("check-self-contained.py")
spec = importlib.util.spec_from_file_location("check_self_contained", SCRIPT_PATH)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class SelfContainmentCheckerTests(unittest.TestCase):
    def make_wiki(self):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name) / "gpu-wiki"
        root.mkdir()
        self.addCleanup(temp.cleanup)
        return root

    def test_relative_markdown_link_inside_wiki_passes(self):
        root = self.make_wiki()
        (root / "README.md").write_text("[Docs](docs/README.md)\n", encoding="utf-8")
        (root / "docs").mkdir()
        (root / "docs" / "README.md").write_text("# Docs\n", encoding="utf-8")

        findings = checker.scan(root)

        self.assertEqual([], [f for f in findings if f.code == "markdown-link-missing"])
        self.assertEqual([], [f for f in findings if f.code == "markdown-link-escapes"])

    def test_relative_markdown_link_escaping_wiki_is_reported(self):
        root = self.make_wiki()
        (root / "README.md").write_text("[Outside](../outside.md)\n", encoding="utf-8")

        findings = checker.scan(root)

        self.assertTrue(any(f.code == "markdown-link-escapes" for f in findings))

    def test_missing_markdown_link_is_reported(self):
        root = self.make_wiki()
        (root / "README.md").write_text("[Missing](docs/missing.md)\n", encoding="utf-8")
        (root / "docs").mkdir()

        findings = checker.scan(root)

        self.assertTrue(any(f.code == "markdown-link-missing" for f in findings))

    def test_existing_directory_markdown_link_passes(self):
        root = self.make_wiki()
        (root / "README.md").write_text("[Docs](docs/)\n", encoding="utf-8")
        (root / "docs").mkdir()

        findings = checker.scan(root)

        self.assertEqual([], [f for f in findings if f.code.startswith("markdown-link")])

    def test_markdown_links_inside_fenced_code_blocks_are_ignored(self):
        root = self.make_wiki()
        (root / "README.md").write_text(
            textwrap.dedent(
                """
                ```python
                kernel[(grid,)](x_ptr, BLOCK_SIZE=block_size,
                               num_warps=num_warps, num_stages=1,
                               NUM_WARPS=num_warps)
                ```
                """
            ),
            encoding="utf-8",
        )

        findings = checker.scan(root)

        self.assertEqual([], [f for f in findings if f.code.startswith("markdown-link")])

    def test_markdown_link_targets_containing_newlines_are_ignored(self):
        root = self.make_wiki()
        (root / "README.md").write_text(
            "[Launch](x_ptr,\n num_warps=num_warps)\n",
            encoding="utf-8",
        )

        findings = checker.scan(root)

        self.assertEqual([], [f for f in findings if f.code.startswith("markdown-link")])

    def test_markdown_image_links_are_ignored(self):
        root = self.make_wiki()
        (root / "README.md").write_text("![Diagram](docs/missing.png)\n", encoding="utf-8")
        (root / "docs").mkdir()

        findings = checker.scan(root)

        self.assertEqual([], [f for f in findings if f.code.startswith("markdown-link")])

    def test_symbol_only_markdown_labels_are_ignored(self):
        root = self.make_wiki()
        (root / "README.md").write_text(
            "static_for<0, 4, 1>{}([&](auto i) { return i; })\n",
            encoding="utf-8",
        )

        findings = checker.scan(root)

        self.assertEqual([], [f for f in findings if f.code.startswith("markdown-link")])

    def test_markdown_link_with_optional_title_does_not_report_missing(self):
        root = self.make_wiki()
        (root / "README.md").write_text("[Docs](docs/README.md \"Docs\")\n", encoding="utf-8")
        (root / "docs").mkdir()
        (root / "docs" / "README.md").write_text("# Docs\n", encoding="utf-8")

        findings = checker.scan(root)

        self.assertEqual([], [f for f in findings if f.code == "markdown-link-missing"])

    def test_unicode_markdown_link_label_is_checked(self):
        root = self.make_wiki()
        (root / "README.md").write_text("[文档](docs/missing.md)\n", encoding="utf-8")
        (root / "docs").mkdir()

        findings = checker.scan(root)

        self.assertTrue(any(f.code == "markdown-link-missing" for f in findings))

    def test_absolute_wiki_clone_path_is_reported(self):
        root = self.make_wiki()
        (root / "README.md").write_text("See `/tmp/gpu-wiki/reference-kernels/README.md`.\n", encoding="utf-8")

        findings = checker.scan(root)

        self.assertTrue(any(f.code == "absolute-wiki-path" for f in findings))

    def test_personal_workspace_path_is_reported(self):
        root = self.make_wiki()
        (root / "README.md").write_text("Source: `/Users/liangyan/Program/ref-gpu-kernel`.\n", encoding="utf-8")

        findings = checker.scan(root)

        self.assertTrue(any(f.code == "personal-absolute-path" for f in findings))

    def test_labeled_historical_provenance_path_is_non_blocking(self):
        root = self.make_wiki()
        (root / "README.md").write_text(
            "Historical provenance path: `/Users/liangyan/Program/ref-gpu-kernel`.\n",
            encoding="utf-8",
        )

        findings = checker.scan(root)

        provenance_findings = [f for f in findings if f.code == "personal-absolute-path"]
        self.assertTrue(provenance_findings)
        self.assertFalse(any(f.blocking for f in provenance_findings))

    def test_reference_source_absolute_path_is_non_blocking(self):
        root = self.make_wiki()
        (root / "README.md").write_text(
            "Reference source: `/opt/custom/reference-project`.\n",
            encoding="utf-8",
        )

        findings = checker.scan(root)

        path_findings = [f for f in findings if f.code == "absolute-path"]
        self.assertTrue(path_findings)
        self.assertFalse(any(f.blocking for f in path_findings))

    def test_hyphenated_source_material_absolute_path_is_non_blocking(self):
        root = self.make_wiki()
        (root / "README.md").write_text(
            "Source-material: `/opt/custom/source-material`.\n",
            encoding="utf-8",
        )

        findings = checker.scan(root)

        path_findings = [f for f in findings if f.code == "absolute-path"]
        self.assertTrue(path_findings)
        self.assertFalse(any(f.blocking for f in path_findings))

    def test_unlabeled_absolute_path_is_still_blocking(self):
        root = self.make_wiki()
        (root / "README.md").write_text("Use `/opt/custom/source-material`.\n", encoding="utf-8")

        findings = checker.scan(root)

        path_findings = [f for f in findings if f.code == "absolute-path"]
        self.assertTrue(path_findings)
        self.assertTrue(any(f.blocking for f in path_findings))

    def test_allowed_tool_paths_are_not_blocking(self):
        root = self.make_wiki()
        (root / "README.md").write_text(
            textwrap.dedent(
                """
                ROCm usually installs under `/opt/rocm`.
                ROCm versioned installs may use `/opt/rocm-5.7`.
                Temporary benchmark output may use `/tmp/flydsl_bench`.
                Conda install path: `/opt/conda`.
                Temporary TTGIR dumps may use `/tmp/fwd.ttgir` or `/tmp/bwd.ttgir`.
                Generic cache examples may mention `/tmp/${USER}/cutlass_python_cache`.
                """
            ),
            encoding="utf-8",
        )

        findings = checker.scan(root)

        self.assertFalse(any(f.blocking for f in findings), findings)

    def test_rocm_like_non_versioned_opt_path_is_reported(self):
        root = self.make_wiki()
        (root / "README.md").write_text("Unexpected ROCm-like path: `/opt/rocmXYZ/bin`.\n", encoding="utf-8")

        findings = checker.scan(root)

        self.assertTrue(any(f.code == "absolute-path" for f in findings))

    def test_disallowed_opt_paths_are_reported(self):
        root = self.make_wiki()
        (root / "README.md").write_text(
            "Custom opt path should be rejected: `/opt/custom/tool`.\n",
            encoding="utf-8",
        )

        findings = checker.scan(root)

        self.assertTrue(any(f.code == "absolute-path" for f in findings))

    def test_conda_versioned_interpreter_path_is_reported(self):
        root = self.make_wiki()
        (root / "README.md").write_text(
            "Use the configured Python, not `/opt/conda310/bin/python3.10`.\n",
            encoding="utf-8",
        )

        findings = checker.scan(root)

        self.assertTrue(any(f.code == "absolute-path" for f in findings))

    def test_atrex_runtime_protocol_names_are_reported(self):
        root = self.make_wiki()
        (root / "README.md").write_text(
            textwrap.dedent(
                """
                The final artifact must be output/optimized.py.
                Start from case/original.py and submit generated_kernel.py.
                Use test_kernel.py as the task validation harness.
                """
            ),
            encoding="utf-8",
        )

        findings = checker.scan(root)

        self.assertTrue(any(f.code == "runtime-protocol" for f in findings))

    def test_neutral_runtime_words_are_allowed(self):
        root = self.make_wiki()
        (root / "README.md").write_text(
            textwrap.dedent(
                """
                CUDA runtime behavior and HIP runtime traces are valid GPU topics.
                JIT runtime cache keys and runtime data types are backend concepts.
                """
            ),
            encoding="utf-8",
        )

        findings = checker.scan(root)

        self.assertEqual([], [f for f in findings if f.code == "runtime-protocol"])

    def test_self_check_files_may_contain_runtime_protocol_patterns(self):
        root = self.make_wiki()
        (root / "scripts").mkdir()
        (root / "scripts" / "check-self-contained.py").write_text(
            'RUNTIME_PROTOCOL_PATTERNS = ("output/optimized.py", "Normandy task path")\n',
            encoding="utf-8",
        )
        (root / "scripts" / "test_check_self_contained.py").write_text(
            'fixture = "case/original.py generated_kernel.py test_kernel.py"\n',
            encoding="utf-8",
        )

        findings = checker.scan(root)

        self.assertEqual([], [f for f in findings if f.code == "runtime-protocol"])

    def test_download_commands_outside_provenance_are_reported(self):
        root = self.make_wiki()
        (root / "README.md").write_text(
            "Run `git clone --depth 1 https://github.com/NVIDIA/cutlass.git /tmp/reference-projects/cutlass`.\n",
            encoding="utf-8",
        )

        findings = checker.scan(root)

        self.assertTrue(any(f.code == "download-step" for f in findings))

    def test_download_commands_in_provenance_are_allowed(self):
        root = self.make_wiki()
        (root / "README.md").write_text(
            "Provenance: upstream source can be identified by `git clone https://github.com/NVIDIA/cutlass.git`.\n",
            encoding="utf-8",
        )

        findings = checker.scan(root)

        self.assertEqual([], [f for f in findings if f.code == "download-step"])

    def test_download_commands_in_reference_source_are_allowed(self):
        root = self.make_wiki()
        (root / "README.md").write_text(
            "Reference source: `git clone https://github.com/NVIDIA/cutlass.git`.\n",
            encoding="utf-8",
        )

        findings = checker.scan(root)

        self.assertEqual([], [f for f in findings if f.code == "download-step"])

    def test_download_commands_in_source_material_are_allowed(self):
        root = self.make_wiki()
        (root / "README.md").write_text(
            "Source material: `curl https://example.com/kernel-notes.md`.\n",
            encoding="utf-8",
        )

        findings = checker.scan(root)

        self.assertEqual([], [f for f in findings if f.code == "download-step"])

    def test_download_commands_in_hyphenated_reference_source_are_allowed(self):
        root = self.make_wiki()
        (root / "README.md").write_text(
            "Reference-source: `git clone https://github.com/NVIDIA/cutlass.git`.\n",
            encoding="utf-8",
        )

        findings = checker.scan(root)

        self.assertEqual([], [f for f in findings if f.code == "download-step"])

    def test_download_commands_in_hyphenated_source_material_are_allowed(self):
        root = self.make_wiki()
        (root / "README.md").write_text(
            "Source-material: `wget https://example.com/kernel-notes.md`.\n",
            encoding="utf-8",
        )

        findings = checker.scan(root)

        self.assertEqual([], [f for f in findings if f.code == "download-step"])

    def test_curl_and_wget_task_steps_are_reported(self):
        root = self.make_wiki()
        (root / "README.md").write_text(
            textwrap.dedent(
                """
                Run `curl https://example.com/kernel-notes.md`.
                Run `wget https://example.com/kernel-notes.md`.
                """
            ),
            encoding="utf-8",
        )

        findings = checker.scan(root)

        download_findings = [f for f in findings if f.code == "download-step"]
        self.assertEqual(2, len(download_findings))

    def test_download_commands_inside_maintenance_are_allowed(self):
        root = self.make_wiki()
        skill_dir = root / ".skill" / "gpu-wiki-clean"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "Run `git clone https://github.com/NVIDIA/cutlass.git`.\n",
            encoding="utf-8",
        )

        findings = checker.scan(root)

        self.assertEqual([], [f for f in findings if f.code == "download-step"])

    def test_skill_package_docs_outside_maintenance_are_reported(self):
        root = self.make_wiki()
        docs = root / "docs" / "converter"
        docs.mkdir(parents=True)
        (docs / "SKILL.md").write_text("---\nname: old-skill\n---\n# Old Skill\n", encoding="utf-8")
        (docs / "helper-skill.md").write_text("# Helper Skill\n", encoding="utf-8")

        findings = checker.scan(root)

        self.assertTrue(any(f.code == "skill-package-doc" for f in findings))

    def test_skill_package_entrypoint_with_frontmatter_reports_once(self):
        root = self.make_wiki()
        docs = root / "docs" / "converter"
        docs.mkdir(parents=True)
        (docs / "SKILL.md").write_text("---\nname: old-skill\n---\n# Old Skill\n", encoding="utf-8")

        findings = checker.scan(root)

        skill_findings = [f for f in findings if f.code == "skill-package-doc"]
        self.assertEqual(1, len(skill_findings))
        self.assertIn("entrypoints", skill_findings[0].message)

    def test_thematic_separator_with_later_name_field_is_not_skill_frontmatter(self):
        root = self.make_wiki()
        (root / "README.md").write_text(
            textwrap.dedent(
                """
                # Notes

                ---

                The config field can be written as:

                name: block_m
                """
            ),
            encoding="utf-8",
        )

        findings = checker.scan(root)

        self.assertEqual([], [f for f in findings if f.code == "skill-package-doc"])

    def test_leading_yaml_name_without_description_is_not_skill_frontmatter(self):
        root = self.make_wiki()
        (root / "README.md").write_text(
            "---\nname: glossary\nowner: docs\n---\n# Glossary\n",
            encoding="utf-8",
        )

        findings = checker.scan(root)

        self.assertEqual([], [f for f in findings if f.code == "skill-package-doc"])

    def test_skill_frontmatter_on_ordinary_markdown_filename_is_reported(self):
        root = self.make_wiki()
        docs = root / "docs"
        docs.mkdir()
        (docs / "notes.md").write_text(
            "---\nname: old-skill\ndescription: helper skill docs\n---\n# Notes\n",
            encoding="utf-8",
        )

        findings = checker.scan(root)

        skill_findings = [f for f in findings if f.code == "skill-package-doc"]
        self.assertEqual(1, len(skill_findings))
        self.assertEqual((docs / "notes.md").resolve(), skill_findings[0].path)

    def test_skill_package_docs_inside_maintenance_are_allowed(self):
        root = self.make_wiki()
        skill_dir = root / ".skill" / "gpu-wiki-clean"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: gpu-wiki-clean\n---\n# Skill\n", encoding="utf-8")

        findings = checker.scan(root)

        self.assertEqual([], [f for f in findings if f.code == "skill-package-doc"])

    def test_only_top_level_maintenance_dirs_allow_skill_package_docs(self):
        root = self.make_wiki()
        top_level_tools = root / "tools"
        nested_tools = root / "docs" / "tools"
        top_level_tools.mkdir()
        nested_tools.mkdir(parents=True)
        (top_level_tools / "SKILL.md").write_text("---\nname: allowed-tool\n---\n# Tool\n", encoding="utf-8")
        (nested_tools / "SKILL.md").write_text("---\nname: docs-tool\n---\n# Docs Tool\n", encoding="utf-8")

        findings = checker.scan(root)

        skill_findings = [f for f in findings if f.code == "skill-package-doc"]
        self.assertEqual([(nested_tools / "SKILL.md").resolve()], [f.path for f in skill_findings])

    def test_normandy_style_worker_wording_is_allowed(self):
        root = self.make_wiki()
        (root / "README.md").write_text(
            "Use a Normandy-style ROCm 6.4.3 worker for reproducing backend behavior.\n",
            encoding="utf-8",
        )

        findings = checker.scan(root)

        self.assertEqual([], [f for f in findings if f.code == "runtime-protocol"])

    def test_unsafe_filename_and_ds_store_are_reported(self):
        root = self.make_wiki()
        (root / "bad name").mkdir()
        (root / "bad name" / "README.md").write_text("# Bad\n", encoding="utf-8")
        (root / "bad name.md").write_text("# Bad\n", encoding="utf-8")
        (root / "bad name .md").write_text("# Bad\n", encoding="utf-8")
        (root / ".DS_Store").write_bytes(b"junk")

        findings = checker.scan(root)

        self.assertTrue(any(f.code == "unsafe-filename" for f in findings))
        self.assertTrue(any(f.code == "os-metadata" for f in findings))

    def test_personal_email_is_reported(self):
        root = self.make_wiki()
        (root / "README.md").write_text("Contact owner@example.internal for access.\n", encoding="utf-8")

        findings = checker.scan(root)

        self.assertTrue(any(f.code == "personal-info" for f in findings))

    def test_example_email_is_allowed(self):
        root = self.make_wiki()
        (root / "README.md").write_text("Use user@example.com as placeholder text.\n", encoding="utf-8")

        findings = checker.scan(root)

        self.assertEqual([], [f for f in findings if f.code == "personal-info"])

    def test_phone_number_is_reported(self):
        root = self.make_wiki()
        (root / "README.md").write_text("Fallback contact: 13812345678.\n", encoding="utf-8")

        findings = checker.scan(root)

        self.assertTrue(any(f.code == "personal-info" for f in findings))

    def test_long_numeric_url_is_not_phone_number(self):
        root = self.make_wiki()
        (root / "README.md").write_text(
            "[Kernel article](https://example.com/p/2011045579652895890)\n",
            encoding="utf-8",
        )

        findings = checker.scan(root)

        self.assertEqual([], [f for f in findings if f.code == "personal-info"])

    def test_public_ip_with_host_context_is_reported(self):
        root = self.make_wiki()
        (root / "README.md").write_text("Connect to host 8.8.8.8 for the trace server.\n", encoding="utf-8")

        findings = checker.scan(root)

        self.assertTrue(any(f.code == "personal-info" for f in findings))

    def test_private_and_documentation_ips_are_allowed(self):
        root = self.make_wiki()
        (root / "README.md").write_text(
            "Example host 192.168.1.2, 10.0.0.1, 172.16.0.1, and 203.0.113.7.\n",
            encoding="utf-8",
        )

        findings = checker.scan(root)

        self.assertEqual([], [f for f in findings if f.code == "personal-info"])

    def test_ptx_section_numbers_are_not_ip_addresses(self):
        root = self.make_wiki()
        (root / "README.md").write_text(
            "See [9.1.5.1. Range Checking](9.1.5.1_range_checking.md) for address rules.\n",
            encoding="utf-8",
        )

        findings = checker.scan(root)

        self.assertEqual([], [f for f in findings if f.code == "personal-info"])


if __name__ == "__main__":
    unittest.main()
