from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orchestrator import optimize


class OptimizeFrameworkDispatchTest(unittest.TestCase):
    def test_mode_policy_is_prepended_without_template_placeholder(self) -> None:
        with tempfile.TemporaryDirectory(prefix="optimize-render-test-") as temp_dir:
            template = Path(temp_dir) / "prompt.md"
            template.write_text("# Base prompt\n", encoding="utf-8")
            rendered = optimize._render(template, MODE_POLICY="## Production policy")
            self.assertEqual(rendered, "## Production policy\n\n# Base prompt\n")

    def test_runtime_arch_selects_supported_frameworks(self) -> None:
        self.assertEqual(
            optimize.supported_frameworks("desensitized", "sm_90"),
            ("Triton", "CuteDSL", "Cuda"),
        )
        self.assertEqual(
            optimize.supported_frameworks("desensitized", "gfx942"),
            ("Triton", "FlyDSL"),
        )
        self.assertEqual(
            optimize.supported_frameworks("unknown-accelerator", ""),
            ("Triton",),
        )

    def test_platform_name_is_only_the_vendor_fallback(self) -> None:
        for platform in ("H20", "L40S", "B200", "Pro5000"):
            with self.subTest(platform=platform):
                self.assertEqual(optimize.hardware_vendor(platform), "nvidia")
        for platform in ("MI308X", "AMD Instinct MI300X"):
            with self.subTest(platform=platform):
                self.assertEqual(optimize.hardware_vendor(platform), "amd")

        # Runtime architecture wins over a contradictory/desensitized name.
        self.assertEqual(optimize.hardware_vendor("MI308X", "sm_90"), "nvidia")
        self.assertEqual(optimize.hardware_vendor("H20", "gfx942"), "amd")

    def test_dispatch_spawns_every_framework_before_waiting(self) -> None:
        events: list[str] = []
        commands: list[list[str]] = []

        class FakeProcess:
            def __init__(self, framework: str, index: int) -> None:
                self.framework = framework
                self.pid = 1000 + index
                self.returncode: int | None = None

            def wait(self, timeout: int | None = None) -> int:
                events.append(f"wait:{self.framework}")
                self.returncode = 0
                return 0

            def poll(self) -> int | None:
                return self.returncode

        def fake_popen(cmd: list[str], **kwargs: object) -> FakeProcess:
            self.assertTrue(kwargs["start_new_session"])
            self.assertTrue(kwargs["text"])
            framework = cmd[cmd.index("--framework") + 1]
            events.append(f"spawn:{framework}")
            commands.append(cmd)
            return FakeProcess(framework, len(commands))

        raw_argv = [
            "--op-dir", "/tmp/op",
            "--platform", "H20",
            "--workspace", "/old-root",
            "--workspace-suffix", "old_suffix",
            "--arch=sm_80",
            "--optimization-mode", "production",
            "--max-iters", "2",
        ]
        frameworks = ("Triton", "CuteDSL", "Cuda")
        with tempfile.TemporaryDirectory(prefix="optimize-dispatch-test-") as temp_dir:
            base = Path(temp_dir)
            with mock.patch.object(optimize.subprocess, "Popen", side_effect=fake_popen):
                result = optimize.dispatch_framework_campaigns(
                    raw_argv, frameworks, base, "sm_90", "H20"
                )

            self.assertEqual(result, 0)
            self.assertEqual(events[:3], [
                "spawn:Triton", "spawn:CuteDSL", "spawn:Cuda",
            ])
            self.assertEqual(events[3:], [
                "wait:Triton", "wait:CuteDSL", "wait:Cuda",
            ])
            self.assertEqual(list(base.iterdir()), [])

        for framework, cmd in zip(frameworks, commands):
            self.assertEqual(cmd.count("--framework"), 1)
            self.assertEqual(cmd[cmd.index("--framework") + 1], framework)
            self.assertEqual(cmd.count("--workspace"), 1)
            self.assertNotIn("/old-root", cmd)
            self.assertEqual(cmd[cmd.index("--workspace") + 1], str(base))
            self.assertEqual(cmd.count("--workspace-suffix"), 1)
            expected_suffix = optimize.framework_workspace_suffix(framework, "H20")
            self.assertEqual(
                cmd[cmd.index("--workspace-suffix") + 1], expected_suffix
            )
            self.assertNotIn("old_suffix", cmd)
            self.assertNotIn("--arch=sm_80", cmd)
            self.assertEqual(cmd[cmd.index("--arch") + 1], "sm_90")
            self.assertEqual(
                cmd[cmd.index("--optimization-mode") + 1], "production"
            )

    def test_suffix_produces_flat_workspace_names(self) -> None:
        campaign = optimize.Campaign(
            name="attention_forward",
            kernel_demo="/tmp/reference.py",
            platform="H20",
            framework="Triton",
            work_dir="/tmp/runs",
            workspace_suffix="triton_h20",
        )
        self.assertEqual(
            campaign.workspace,
            Path("/tmp/runs/kernel_opt_attention_forward_triton_h20"),
        )

        layer = optimize.LayerCampaign(
            name="decoder",
            layer_demo="/tmp/reference.py",
            platform="H20",
            framework="CuteDSL",
            work_dir="/tmp/runs",
            workspace_suffix="cutedsl_h20",
        )
        self.assertEqual(layer.layer_dir, Path("/tmp/runs/layer_decoder_cutedsl_h20"))
        self.assertEqual(
            layer._boundary_ws("attention"),
            Path("/tmp/runs/kernel_opt_decoder__attention_cutedsl_h20"),
        )

    def test_main_uses_auto_dispatch_only_when_framework_is_omitted(self) -> None:
        op = {
            "name": "demo",
            "reference": "/tmp/op/reference.py",
            "roofline_py": "",
            "op_dir": "/tmp/op",
        }
        with tempfile.TemporaryDirectory(prefix="optimize-main-test-") as temp_dir:
            argv = [
                "--op-dir", "/tmp/op",
                "--platform", "H20",
                "--arch", "sm_90",
                "--workspace", temp_dir,
            ]
            with (
                mock.patch.object(optimize.shutil, "which", return_value="/bin/claude"),
                mock.patch.object(optimize, "_resolve_op", return_value=op),
                mock.patch.object(optimize, "ensure_submodules"),
                mock.patch.object(
                    optimize, "dispatch_framework_campaigns", return_value=0
                ) as dispatch,
            ):
                self.assertEqual(optimize.main(argv), 0)

            dispatch.assert_called_once_with(
                argv,
                ("Triton", "CuteDSL", "Cuda"),
                Path(temp_dir).resolve(),
                "sm_90",
                "H20",
            )

    def test_main_explicit_framework_uses_the_same_flat_suffix(self) -> None:
        op = {
            "name": "demo",
            "reference": "/tmp/op/reference.py",
            "roofline_py": "",
            "op_dir": "/tmp/op",
        }
        campaign_instance = mock.Mock()
        with tempfile.TemporaryDirectory(prefix="optimize-main-explicit-test-") as temp_dir:
            argv = [
                "--op-dir", "/tmp/op",
                "--platform", "H20",
                "--sandbox-hardware", "local",
                "--arch", "sm_90",
                "--framework", "CuteDSL",
                "--workspace", temp_dir,
            ]
            with (
                mock.patch.object(optimize.shutil, "which", return_value="/bin/claude"),
                mock.patch.object(optimize, "_resolve_op", return_value=op),
                mock.patch.object(optimize, "ensure_submodules"),
                mock.patch.object(
                    optimize, "Campaign", return_value=campaign_instance
                ) as campaign,
            ):
                self.assertEqual(optimize.main(argv), 0)

            self.assertEqual(campaign.call_args.kwargs["workspace_suffix"], "cutedsl_h20")
            self.assertEqual(campaign.call_args.kwargs["sandbox_hardware"], "local")
            self.assertEqual(campaign.call_args.kwargs["platform"], "H20")
            self.assertEqual(campaign.call_args.kwargs["optimization_mode"], "leaderboard")
            campaign_instance.run.assert_called_once_with()

    def test_production_explicit_framework_disables_conversion(self) -> None:
        op = {
            "name": "demo",
            "reference": "/tmp/op/reference.py",
            "roofline_py": "",
            "op_dir": "/tmp/op",
        }
        campaign_instance = mock.Mock()
        with tempfile.TemporaryDirectory(prefix="optimize-production-test-") as temp_dir:
            argv = [
                "--op-dir", "/tmp/op",
                "--platform", "H20",
                "--arch", "sm_90",
                "--framework", "Triton",
                "--optimization-mode", "production",
                "--workspace", temp_dir,
            ]
            with (
                mock.patch.object(optimize.shutil, "which", return_value="/bin/claude"),
                mock.patch.object(optimize, "_resolve_op", return_value=op),
                mock.patch.object(optimize, "ensure_submodules"),
                mock.patch.object(
                    optimize, "Campaign", return_value=campaign_instance
                ) as campaign,
            ):
                self.assertEqual(optimize.main(argv), 0)

            self.assertEqual(campaign.call_args.kwargs["optimization_mode"], "production")
            self.assertEqual(campaign.call_args.kwargs["convert_after"], 0)
            campaign_instance.run.assert_called_once_with()

    def test_production_without_framework_auto_dispatches_constrained_children(self) -> None:
        op = {
            "name": "demo",
            "reference": "/tmp/op/reference.py",
            "roofline_py": "",
            "op_dir": "/tmp/op",
        }
        with tempfile.TemporaryDirectory(prefix="optimize-production-dispatch-") as temp_dir:
            argv = [
                "--op-dir", "/tmp/op",
                "--platform", "H20",
                "--arch", "sm_90",
                "--optimization-mode", "production",
                "--workspace", temp_dir,
            ]
            with (
                mock.patch.object(optimize.shutil, "which", return_value="/bin/claude"),
                mock.patch.object(optimize, "_resolve_op", return_value=op),
                mock.patch.object(optimize, "ensure_submodules"),
                mock.patch.object(
                    optimize, "dispatch_framework_campaigns", return_value=0
                ) as dispatch,
            ):
                self.assertEqual(optimize.main(argv), 0)

            dispatch.assert_called_once_with(
                argv,
                ("Triton", "CuteDSL", "Cuda"),
                Path(temp_dir).resolve(),
                "sm_90",
                "H20",
            )

if __name__ == "__main__":
    unittest.main()
