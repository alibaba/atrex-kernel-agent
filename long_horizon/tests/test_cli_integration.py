from __future__ import annotations

import re
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from long_horizon import cli


class MainCliIntegrationTests(unittest.TestCase):
    def test_help_inherits_every_current_main_option(self) -> None:
        root = Path(__file__).resolve().parents[2]
        main_help = subprocess.run(
            [sys.executable, "orchestrator/optimize.py", "--help"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        long_help = subprocess.run(
            [sys.executable, "-m", "long_horizon", "--help"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        option = re.compile(r"(?<![\w-])--[a-z][a-z0-9-]*")
        self.assertEqual(set(option.findall(main_help)) - set(option.findall(long_help)), set())

    def test_main_cli_is_authoritative_and_long_options_are_stripped(self) -> None:
        argv = [
            "--op-dir", "/tmp/op",
            "--platform", "B200",
            "--max-workload-buckets", "6",
            "--aggregate-min-improvement-pct", "1.5",
            "--agent-cli", "codex",
            "--handoff-resumes", "4",
            "--verify-repeats", "3",
        ]
        with mock.patch.object(cli.base, "main", return_value=17) as main:
            self.assertEqual(cli.main(argv), 17)
        delegated = main.call_args.args[0]
        self.assertIn("--max-workload-buckets", delegated)
        self.assertIn("--aggregate-min-improvement-pct", delegated)
        self.assertIn("--agent-cli", delegated)
        self.assertNotIn("--handoff-resumes", delegated)
        self.assertNotIn("--verify-repeats", delegated)

    def test_framework_dispatch_reenters_long_horizon_and_forwards_options(self) -> None:
        observed: dict = {}

        def dispatch(argv, frameworks, workspace, arch, platform, mode):
            observed["argv"] = argv
            observed["entry"] = cli.base.__file__
            return 23

        options = cli.LongHorizonOptions(handoff_resumes=5, verify_repeats=4)
        with mock.patch.object(cli.base, "dispatch_framework_campaigns", side_effect=dispatch):
            with cli._install_main_integration(options):
                result = cli.base.dispatch_framework_campaigns(
                    ["--op-dir", "/tmp/op"],
                    ("Triton", "Cuda"),
                    Path("/tmp/runs"),
                    "sm_100",
                    "B200",
                    "leaderboard",
                )
        self.assertEqual(result, 23)
        self.assertTrue(observed["entry"].endswith("long_horizon/__main__.py"))
        self.assertIn("--handoff-resumes", observed["argv"])
        self.assertEqual(
            observed["argv"][observed["argv"].index("--verify-repeats") + 1], "4"
        )

    def test_campaign_run_replaces_only_the_iteration_loop(self) -> None:
        campaign = cli.base.Campaign(
            name="op",
            kernel_demo="/tmp/reference.py",
            platform="B200",
            framework="CuteDSL",
            max_iters=12,
            token_budget=900,
            iter_timeout=777,
            max_stall=3,
            sandbox_hardware="REMOTE_GPU",
        )
        campaign._finish = mock.Mock(return_value="finished")
        long_campaign = mock.Mock()
        long_campaign.run.return_value = "budget: max-iters"
        with mock.patch.object(cli, "LongHorizonCampaign", return_value=long_campaign) as factory:
            with mock.patch.object(cli, "GatewayABBAValidator"):
                with cli._install_main_integration(cli.LongHorizonOptions()):
                    self.assertEqual(campaign.run(), "finished")
        kwargs = factory.call_args.kwargs
        self.assertIs(kwargs["base_campaign"], campaign)
        self.assertEqual(kwargs["max_version"], 12)
        self.assertEqual(kwargs["token_budget"], 900)
        self.assertEqual(kwargs["session_timeout"], 777)
        self.assertEqual(kwargs["max_stall"], 3)
        campaign._finish.assert_called_once_with("budget: max-iters")

    def test_layer_boundary_keeps_main_layer_policy_without_conversion_latch(self) -> None:
        layer = SimpleNamespace(
            name="layer",
            platform="B200",
            framework="Triton",
            notes="test",
            arch="sm_100",
            work_dir="/tmp/runs",
            workspace_suffix="triton_b200",
            max_iters=20,
            iter_timeout=500,
            setup_timeout=600,
            sandbox_hardware="REMOTE_GPU",
            sandbox_profile="",
            sandbox_url="",
            sandbox_timeout=600,
            agent_cli="claude",
            optimization_mode="leaderboard",
            _boundary_ws=lambda name: Path("/tmp/runs") / f"kernel_opt_layer__{name}_triton_b200",
        )
        campaign = cli._boundary_campaign(layer, {"name": "attention"})
        self.assertEqual(campaign.convert_after, 0)
        self.assertEqual(campaign.workspace, layer._boundary_ws("attention"))


if __name__ == "__main__":
    unittest.main()
