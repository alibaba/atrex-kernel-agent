from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orchestrator import optimize
from orchestrator.agent_runtime.process import ProcessAccessPolicy


class CampaignRuntimeBindingTest(unittest.TestCase):
    def test_campaign_workspace_policy_records_the_selected_runtime(self) -> None:
        with tempfile.TemporaryDirectory(prefix="campaign-runtime-binding-") as temp_dir:
            campaign = optimize.Campaign(
                name="demo",
                kernel_demo="/tmp/reference.py",
                platform="H20",
                framework="Triton",
                work_dir=temp_dir,
                agent_cli="codex",
            )
            campaign.workspace.mkdir(parents=True)
            with mock.patch.object(optimize, "link_runtime"):
                campaign._link_runtime()
            state = json.loads(
                (campaign.workspace / ".orchestrator_mode.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(state["agent_runtime"], "codex")

    def test_campaign_can_override_runtime_hydration_without_changing_policy_binding(self) -> None:
        with tempfile.TemporaryDirectory(prefix="campaign-runtime-override-") as temp_dir:
            calls: list[Path] = []
            campaign = optimize.Campaign(
                name="demo",
                kernel_demo="/tmp/reference.py",
                platform="H20",
                framework="CuteDSL",
                work_dir=temp_dir,
                optimization_mode="production",
                runtime_linker=lambda current: calls.append(current.workspace),
            )
            campaign.workspace.mkdir(parents=True)
            campaign._link_runtime()

            self.assertEqual(calls, [campaign.workspace])
            state = json.loads(
                (campaign.workspace / ".orchestrator_mode.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state["mode"], "production")
            self.assertEqual(state["framework"], "CuteDSL")

    def test_campaign_session_hook_prepends_directive_and_passes_opaque_policy(self) -> None:
        with tempfile.TemporaryDirectory(prefix="campaign-session-hook-") as temp_dir:
            access = ProcessAccessPolicy(
                forbidden_roots=(Path(temp_dir) / "private",),
                network_disabled=True,
            )
            campaign = optimize.Campaign(
                name="demo",
                kernel_demo="/tmp/reference.py",
                platform="H20",
                framework="CuteDSL",
                work_dir=temp_dir,
                session_directive="## restricted knowledge",
                session_access_policy=access,
            )
            campaign.workspace.mkdir(parents=True)
            with mock.patch.object(
                optimize,
                "run_session",
                return_value=optimize.SessionResult(0, False, 1, "", "", "sid"),
            ) as run:
                campaign._run_agent_session(
                    "# task",
                    timeout=10,
                    extra_environment={"TRACE": "1"},
                )

            self.assertTrue(run.call_args.args[1].startswith("## restricted knowledge\n\n# task"))
            self.assertIs(run.call_args.kwargs["access_policy"], access)
            self.assertEqual(run.call_args.kwargs["extra_environment"], {"TRACE": "1"})

    def test_campaign_session_hook_filters_before_reasserting_directive(self) -> None:
        with tempfile.TemporaryDirectory(prefix="campaign-prompt-filter-") as temp_dir:
            campaign = optimize.Campaign(
                name="demo",
                kernel_demo="/tmp/reference.py",
                platform="H20",
                framework="CuteDSL",
                work_dir=temp_dir,
                session_directive="## final policy",
                session_prompt_filter=lambda prompt: prompt.replace("forbidden", "sanitized"),
            )
            campaign.workspace.mkdir(parents=True)
            with mock.patch.object(
                optimize,
                "run_session",
                return_value=optimize.SessionResult(0, False, 1, "", "", "sid"),
            ) as run:
                campaign._run_agent_session("# forbidden task", timeout=10)

            rendered = run.call_args.args[1]
            self.assertNotIn("forbidden", rendered)
            self.assertEqual(
                rendered,
                "## final policy\n\n# sanitized task\n\n## final policy",
            )

    def test_layer_policy_helper_records_the_selected_runtime(self) -> None:
        with tempfile.TemporaryDirectory(prefix="layer-runtime-binding-") as temp_dir:
            layer = optimize.LayerCampaign(
                name="decoder",
                layer_demo="/tmp/reference.py",
                platform="H20",
                framework="CuteDSL",
                work_dir=temp_dir,
                agent_cli="qodercli",
            )
            workspace = Path(temp_dir) / "policy-target"
            workspace.mkdir()
            layer._install_workspace_policy(workspace)
            state = json.loads(
                (workspace / ".orchestrator_mode.json").read_text(encoding="utf-8")
            )
        self.assertEqual(state["agent_runtime"], "qodercli")


if __name__ == "__main__":
    unittest.main()
