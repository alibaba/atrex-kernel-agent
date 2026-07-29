from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orchestrator import optimize


class AgentCliTest(unittest.TestCase):
    def test_codex_is_a_supported_backend(self) -> None:
        self.assertIn("codex", optimize.AGENT_CLI_CHOICES)

    def test_codex_command_is_fresh_json_and_noninteractive(self) -> None:
        with mock.patch.dict(
            optimize.os.environ,
            {},
            clear=True,
        ):
            cmd = optimize._session_command("codex", "do one iteration", "ignored-session-id")

        self.assertEqual(cmd[:2], ["codex", "exec"])
        self.assertIn("--json", cmd)
        self.assertIn("--ephemeral", cmd)
        self.assertIn("--dangerously-bypass-approvals-and-sandbox", cmd)
        self.assertEqual(cmd[-1], "do one iteration")
        self.assertNotIn("ignored-session-id", cmd)
        self.assertIn('model_reasoning_effort="max"', cmd)
        self.assertNotIn("--plugin-dir", cmd)

    def test_codex_provider_settings_become_repeatable_config_flags(self) -> None:
        settings = json.dumps(
            {
                "model": "gpt-5.6-sol",
                "model_reasoning_effort": "xhigh",
                "features.codex_hooks": False,
            }
        )
        with mock.patch.dict(
            optimize.os.environ,
            {"ATREX_CODEX_SESSION_SETTINGS": settings},
            clear=False,
        ):
            cmd = optimize._session_command("codex", "prompt", "unused")

        pairs = [cmd[index + 1] for index, arg in enumerate(cmd[:-1]) if arg == "-c"]
        self.assertIn('model="gpt-5.6-sol"', pairs)
        self.assertIn('model_reasoning_effort="xhigh"', pairs)
        self.assertIn("features.codex_hooks=false", pairs)
        # Provider settings occur after defaults, so a user override wins in Codex config order.
        self.assertGreater(
            pairs.index('model_reasoning_effort="xhigh"'),
            pairs.index('model_reasoning_effort="max"'),
        )

    def test_codex_settings_accept_literal_toml_pairs(self) -> None:
        args = optimize._codex_settings_args(
            json.dumps(['model="custom"', 'model_reasoning_effort="high"'])
        )
        self.assertEqual(
            args,
            ["-c", 'model="custom"', "-c", 'model_reasoning_effort="high"'],
        )

    def test_codex_settings_fail_closed_on_ambiguous_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be a JSON object"):
            optimize._codex_settings_args("model=gpt-5")
        with self.assertRaisesRegex(ValueError, "key=value strings"):
            optimize._codex_settings_args('["missing-equals"]')
        with self.assertRaisesRegex(ValueError, "must be strings"):
            optimize._codex_settings_args('{"nested": {"unsupported": true}}')

    def test_codex_turn_completed_usage_drives_token_budget(self) -> None:
        stdout = "\n".join(
            [
                '{"type":"thread.started","thread_id":"abc"}',
                '{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}',
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {
                            "input_tokens": 11709,
                            "cached_input_tokens": 2000,
                            "cache_write_input_tokens": 9709,
                            "output_tokens": 5,
                            "reasoning_output_tokens": 3,
                        },
                    }
                ),
            ]
        )
        # Cache/reasoning counters are subsets; counting them again would overcharge the budget.
        self.assertEqual(optimize._tokens_from_stream(stdout), 11714)

    def test_run_session_injects_the_existing_gpu_sandbox_contract_for_codex(self) -> None:
        captured: dict[str, object] = {}

        def fake_run(
            cmd: list[str], cwd: Path, timeout: int, env: dict | None = None
        ) -> tuple[str, str, int, bool]:
            captured.update(cmd=cmd, cwd=cwd, timeout=timeout, env=env)
            return (
                '{"type":"turn.completed","usage":{"input_tokens":7,"output_tokens":2}}',
                "",
                0,
                False,
            )

        with tempfile.TemporaryDirectory(prefix="codex-session-") as temp_dir:
            with (
                mock.patch.object(optimize, "_run_bounded", side_effect=fake_run),
                mock.patch.dict(
                    optimize.os.environ,
                    {},
                    clear=True,
                ),
            ):
                result = optimize.run_session(
                    Path(temp_dir),
                    "prompt",
                    timeout=123,
                    agent_cli="codex",
                    sandbox_hardware="REMOTE_GPU",
                    sandbox_url="https://gateway.example.test",
                    sandbox_timeout=456,
                )

        env = captured["env"]
        self.assertIsInstance(env, dict)
        self.assertEqual(env["IS_SANDBOX"], "1")
        self.assertEqual(env["ATREX_SANDBOX_GPU"], "REMOTE_GPU")
        self.assertEqual(env["ATREX_SANDBOX_URL"], "https://gateway.example.test")
        self.assertEqual(env["ATREX_SANDBOX_TIMEOUT"], "456")
        self.assertEqual(result.tokens, 9)

    def test_link_runtime_installs_repository_scoped_codex_skills(self) -> None:
        if not (optimize.HUMANIZE_DIR / "skills" / "humanize-gen-plan" / "SKILL.md").is_file():
            self.skipTest("Humanize submodule is not initialized")
        with tempfile.TemporaryDirectory(prefix="codex-runtime-") as temp_dir:
            workspace = Path(temp_dir)
            optimize.link_runtime(workspace)

            skills = workspace / ".agents" / "skills"
            self.assertTrue((skills / "gpu-kernel-baseline" / "SKILL.md").is_file())
            self.assertTrue((skills / "humanize-gen-plan" / "SKILL.md").is_file())
            humanize_text = (skills / "humanize-gen-plan" / "SKILL.md").read_text(
                encoding="utf-8"
            )
            self.assertNotIn("{{HUMANIZE_RUNTIME_ROOT}}", humanize_text)
            self.assertIn(str(skills / "humanize"), humanize_text)
            self.assertTrue((skills / "humanize" / "scripts").is_dir())
            self.assertIn("/.agents", (workspace / ".gitignore").read_text(encoding="utf-8"))

    def test_main_accepts_codex_and_passes_it_to_campaign(self) -> None:
        op = {
            "name": "demo",
            "reference": "/tmp/op/reference.py",
            "roofline_py": "",
            "op_dir": "/tmp/op",
        }
        campaign_instance = mock.Mock()
        with tempfile.TemporaryDirectory(prefix="optimize-codex-main-") as temp_dir:
            argv = [
                "--op-dir", "/tmp/op",
                "--platform", "H20",
                "--arch", "sm_90",
                "--framework", "Triton",
                "--agent-cli", "codex",
                "--workspace", temp_dir,
            ]
            with (
                mock.patch.object(optimize.shutil, "which", return_value="/bin/codex"),
                mock.patch.object(optimize, "_resolve_op", return_value=op),
                mock.patch.object(optimize, "ensure_submodules"),
                mock.patch.object(
                    optimize, "Campaign", return_value=campaign_instance
                ) as campaign,
            ):
                self.assertEqual(optimize.main(argv), 0)

        self.assertEqual(campaign.call_args.kwargs["agent_cli"], "codex")
        campaign_instance.run.assert_called_once_with()

    def test_codex_prompt_directions_use_native_skill_mentions(self) -> None:
        self.assertIn("$gpu-kernel-baseline", optimize._baseline_driver_directive("codex"))
        plan = optimize._plan_generator_directive("codex", 3)
        self.assertIn("$humanize-gen-plan", plan)
        self.assertIn("plans/v3_draft.md", plan)
        self.assertIn("plans/v3_plan.md", plan)
        self.assertIn("/humanize:gen-plan", optimize._plan_generator_directive("claude", 3))

    def test_codex_rendered_prompts_have_no_runtime_placeholders(self) -> None:
        setup = optimize._render(
            optimize.PROMPTS_DIR / "setup.md",
            WORKSPACE="/tmp/ws",
            PLATFORM="H20",
            FRAMEWORK="Triton",
            KERNEL_DEMO="/tmp/reference.py",
            NOTES="none",
            AGENT_RUNTIME=optimize._agent_runtime_directive("codex"),
            BASELINE_DRIVER=optimize._baseline_driver_directive("codex"),
            HARDWARE="hardware",
            SANDBOX="sandbox",
        )
        iteration = optimize._render(
            optimize.PROMPTS_DIR / "iteration.md",
            WORKSPACE="/tmp/ws",
            N=1,
            PREV=0,
            PLATFORM="H20",
            NOTES="none",
            AGENT_RUNTIME=optimize._agent_runtime_directive("codex"),
            PLAN_GENERATOR=optimize._plan_generator_directive("codex", 1),
            HARDWARE="hardware",
            SANDBOX="sandbox",
        )
        self.assertNotIn("{{AGENT_RUNTIME}}", setup)
        self.assertNotIn("{{BASELINE_DRIVER}}", setup)
        self.assertNotIn("{{PLAN_GENERATOR}}", iteration)
        self.assertIn("$gpu-kernel-baseline", setup)
        self.assertIn("$humanize-gen-plan", iteration)


if __name__ == "__main__":
    unittest.main()
