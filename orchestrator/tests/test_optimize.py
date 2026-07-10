import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import optimize


class TokenParsingTests(unittest.TestCase):
    def test_claude_uses_terminal_result(self):
        stream = "\n".join([
            json.dumps({"message": {"usage": {"input_tokens": 10, "output_tokens": 2}}}),
            json.dumps({"type": "result", "usage": {
                "input_tokens": 20,
                "output_tokens": 3,
                "cache_read_input_tokens": 5,
            }}),
        ])
        self.assertEqual(optimize._tokens_from_stream(stream, "claude"), 28)

    def test_codex_sums_completed_turns_only(self):
        stream = "\n".join([
            json.dumps({"type": "turn.completed", "usage": {
                "input_tokens": 20,
                "cached_input_tokens": 7,
                "output_tokens": 3,
                "reasoning_output_tokens": 2,
            }}),
            json.dumps({"type": "item.completed", "usage": {"input_tokens": 999}}),
            json.dumps({"type": "turn.completed", "usage": {
                "input_tokens": 4,
                "output_tokens": 1,
            }}),
        ])
        self.assertEqual(optimize._tokens_from_stream(stream, "codex"), 28)


class SessionCommandTests(unittest.TestCase):
    def test_codex_command_is_ephemeral_json_and_writable(self):
        workspace = Path("/tmp/workspace")
        command = optimize._session_command("codex", workspace, "prompt")
        self.assertEqual(command[:4], ["codex", "exec", "--json", "--ephemeral"])
        self.assertIn("danger-full-access", command)
        self.assertEqual(command[-3:], ["-C", str(workspace), "prompt"])

    def test_claude_command_keeps_existing_stream_mode(self):
        with patch.object(optimize.uuid, "uuid4", return_value="session-id"):
            command = optimize._session_command("claude", Path("/tmp/workspace"), "prompt")
        self.assertEqual(command, [
            "claude", "--print", "--verbose", "--output-format", "stream-json",
            "--session-id", "session-id", "prompt",
        ])


class RuntimeLinkTests(unittest.TestCase):
    def test_links_codex_agent_playbooks(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            optimize.link_runtime(workspace)
            self.assertTrue((workspace / "agents").is_symlink())
            self.assertIn("/agents", (workspace / ".gitignore").read_text())

    def test_adds_agents_ignore_to_existing_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / ".gitignore").write_text("/tools\n/reference\n/skills\n")
            optimize.link_runtime(workspace)
            lines = (workspace / ".gitignore").read_text().splitlines()
            self.assertEqual(lines.count("/agents"), 1)


class InstallerContractTests(unittest.TestCase):
    def test_installer_copies_self_contained_optimizer(self):
        install_script = (Path(__file__).parents[2] / "install.sh").read_text()
        self.assertIn(
            "SKILL_WHITELIST=(orchestrator agents reference skills tools SKILL.md)",
            install_script,
        )
        self.assertIn("link_skill_gpu_wiki", install_script)


if __name__ == "__main__":
    unittest.main()
