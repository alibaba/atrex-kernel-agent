"""V1 assets and prompt integration, with acceptance-boundary regression coverage."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from aka.light.workflow.bootstrap import Campaign
from aka.light.workspace.assets import link_runtime
from aka.light.workspace.resource_catalog import resource


class V1WorkflowTest(unittest.TestCase):
    def test_agent_receives_standalone_cuda_source(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            registry = SimpleNamespace(plugins=[], install_agent_skills=lambda workspace: [])
            link_runtime(workspace, plugin_registry=registry)
            for name in ("templates/cuda_nvrtc.py", "references/cuda-starting-point.md"):
                path = "skills/gpu-measurement/" + name
                self.assertEqual((workspace / path).read_bytes(), resource(path).read_bytes())
            compile((workspace / "skills/gpu-measurement/templates/cuda_nvrtc.py").read_text(), "template", "exec")

    def prompt(self, framework, platform, arch, wiki=False):
        campaign = SimpleNamespace(
            workspace=Path("/workspace/operator"), platform=platform, framework=framework,
            arch=arch, notes="", agent_cli="pi", tool_runtime=SimpleNamespace(wiki=wiki),
            plugin_registry=SimpleNamespace(knowledge_skills=lambda: ("KernelWiki",)),
            plugin_directive=lambda phase: "", wiki_prompt=lambda phase: "Resolve one missing fact.",
            _framework_baseline_smoke_command=lambda n: ("python3 tools/sandbox.py --kind run", "Bounded smoke."),
            _framework_baseline_sandbox_directive=lambda: "", _evaluator_directive=lambda: "",
            _framework_baseline_correctness_guidance_text=lambda: "", _mode_directive=lambda: "",
        )
        return Campaign._framework_baseline_prompt(campaign, 1)

    def test_template_is_selected_only_for_nvidia_cuda(self):
        for framework, platform, arch, expected in (
            ("cuda", "L20N", "sm_120", True),
            ("triton", "L20N", "sm_120", False),
            ("cuda", "PPU-ZW810E", "sm_80", False),
        ):
            for wiki in (False, True):
                with self.subTest(framework=framework, platform=platform, wiki=wiki):
                    prompt = self.prompt(framework, platform, arch, wiki)
                    self.assertEqual("templates/cuda_nvrtc.py" in prompt, expected)
                    self.assertNotIn("{{", prompt)
                    self.assertIn("Bounded smoke.", prompt)
                    self.assertIn("full-domain multi-seed correctness", prompt)
                    self.assertIn("without changing references, probes, thresholds", prompt)

    def test_independent_gates_still_reject_policy_and_correctness_failures(self):
        for violations, validation_error in ((["forbidden compute"], ""), ([], "incorrect output")):
            with self.subTest(violations=violations, validation_error=validation_error):
                numerical = Mock()
                campaign = SimpleNamespace(
                    source_checks_enabled=True, framework="cuda", workspace=Path("/workspace/operator"),
                    _production_kernel_violations=lambda: violations,
                    _validate_framework_baseline=lambda n: ({"latency_us_geomean": 1}, validation_error),
                    supplemental_numerical_feedback=numerical,
                )
                _, problem = Campaign._framework_baseline_external_gates(campaign, 1)
                self.assertTrue(problem)
                numerical.assert_not_called()

    def test_passing_primary_gates_still_require_numerical_acceptance(self):
        numerical = Mock(return_value="Supplemental numerical probes: measured mismatch")
        campaign = SimpleNamespace(
            source_checks_enabled=True, framework="cuda", workspace=Path("/workspace/operator"),
            _production_kernel_violations=lambda: [],
            _validate_framework_baseline=lambda n: ({"latency_us_geomean": 1}, ""),
            supplemental_numerical_feedback=numerical,
        )
        _, problem = Campaign._framework_baseline_external_gates(campaign, 1)
        self.assertIn("measured mismatch", problem)
        numerical.assert_called_once_with(campaign.workspace, flow="framework_baseline")


if __name__ == "__main__":
    unittest.main()
