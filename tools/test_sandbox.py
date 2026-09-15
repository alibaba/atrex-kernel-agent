from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

from tools import sandbox


class SandboxCliTest(unittest.TestCase):
    def test_single_file_cli_runs_without_repository_imports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "sandbox.py"
            shutil.copy2(sandbox.__file__, script)
            environment = dict(os.environ)
            for key in (sandbox.RUNTIME_URL_ENV, sandbox.RUNTIME_TOKEN_ENV):
                environment.pop(key, None)
            for arguments, expected in ((["--help"], 0), (["--kind", "env"], 75)):
                result = subprocess.run(
                    [sys.executable, "-I", str(script), *arguments], cwd=directory,
                    env=environment, capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, expected, result.stderr)
                self.assertNotIn("ModuleNotFoundError", result.stderr)

    def test_http_transport_preserves_authority_payload_and_cli_output(self) -> None:
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({
            "stdout": "record saved", "stderr": "diagnostic", "exit_code": 4,
        }).encode()
        with (
            patch.dict(os.environ, {
                sandbox.RUNTIME_URL_ENV: "http://127.0.0.1:12345/",
                sandbox.RUNTIME_TOKEN_ENV: "test-capability",
            }),
            patch.object(sandbox.urllib.request, "urlopen", return_value=response) as opened,
            patch("sys.stdout", io.StringIO()) as stdout,
            patch("sys.stderr", io.StringIO()) as stderr,
        ):
            self.assertEqual(sandbox.proxy_journal("direction_load", direction_id="direction_test"), 4)
            request = opened.call_args.args[0]
            self.assertEqual(request.full_url, "http://127.0.0.1:12345/v1/journal/execute")
            self.assertEqual(request.get_header("Authorization"), "Bearer test-capability")
            self.assertEqual(json.loads(request.data), {
                "operation": "direction_load", "direction_id": "direction_test",
            })
            self.assertEqual(stdout.getvalue(), "record saved\n")
            self.assertEqual(stderr.getvalue(), "diagnostic\n")

    def test_http_transport_errors_are_reported_without_fallback(self) -> None:
        response = MagicMock()
        for raw, expected in (
            (b"not json", "invalid JSON"),
            (b"\xff", "invalid JSON"),
            (b"[]", "non-object response"),
            (b'{"exit_code": true}', "valid exit_code"),
            (b"x" * (sandbox.MAX_RUNTIME_RESPONSE_BYTES + 1), "client limit"),
        ):
            response.__enter__.return_value.read.return_value = raw
            with (
                self.subTest(expected=expected),
                patch.dict(os.environ, {
                    sandbox.RUNTIME_URL_ENV: "http://127.0.0.1:12345",
                    sandbox.RUNTIME_TOKEN_ENV: "test-capability",
                }),
                patch.object(sandbox.urllib.request, "urlopen", return_value=response),
                patch("sys.stderr", io.StringIO()) as stderr,
            ):
                self.assertEqual(sandbox.proxy_gateway(["--kind", "env"]), 2)
                self.assertIn(expected, stderr.getvalue())

    def test_http_error_preserves_structured_guidance_without_an_automatic_retry(self) -> None:
        value = {
            "ok": False, "repairable": False,
            "error": {"code": "runtime_failure", "message": "Outcome unknown",
                      "next_action": "Inspect existing records; do not retry blindly."},
        }
        failure = urllib.error.HTTPError(
            "http://127.0.0.1:12345", 500, "Internal error", {},
            io.BytesIO(json.dumps(value).encode()),
        )
        with (
            patch.dict(os.environ, {
                sandbox.RUNTIME_URL_ENV: "http://127.0.0.1:12345",
                sandbox.RUNTIME_TOKEN_ENV: "test-capability",
            }),
            patch.object(sandbox.urllib.request, "urlopen", side_effect=failure) as opened,
            patch("sys.stderr", io.StringIO()) as stderr,
        ):
            self.assertEqual(sandbox.proxy_gateway(["--kind", "env"]), 75)
            self.assertEqual(json.loads(stderr.getvalue()), value)
            opened.assert_called_once()

    def test_wiki_operations_strip_only_selector_and_preserve_arguments(self) -> None:
        cases = [
            (["--kind", "wiki-query", "BF16 norm on sm_100", "--brief"],
             "query_nl", ["BF16 norm on sm_100", "--brief"]),
            (["--kind=wiki-query", "--file", "scratch/query.txt", "--max-records", "6"],
             "query_nl", ["--file", "scratch/query.txt", "--max-records", "6"]),
            (["--arch", "sm_100", "--kind", "wiki-search", "--coverage"],
             "query_wiki", ["--arch", "sm_100", "--coverage"]),
            (["--kind", "wiki-hardware", "--product", "b200", "--field", "peak_compute.bf16.dense"],
             "query_hardware", ["--product", "b200", "--field", "peak_compute.bf16.dense"]),
            (["--kind", "wiki-query", "--", "--kind", "dev"],
             "query_nl", ["--", "--kind", "dev"]),
        ]
        for kind, tool in sandbox._WIKI_KINDS.items():
            cases.append((["--kind", kind, "--help"], tool, ["--help"]))
        for arguments, tool, expected in cases:
            with (
                self.subTest(arguments=arguments),
                patch.object(sandbox, "proxy_wiki", return_value=4) as wiki,
                patch.object(sandbox, "proxy_gateway") as gateway,
                patch.object(sandbox, "proxy_journal") as journal,
            ):
                self.assertEqual(sandbox.main(arguments), 4)
                wiki.assert_called_once_with(tool, expected)
                gateway.assert_not_called()
                journal.assert_not_called()

    def test_wiki_requires_capability_without_direct_fallback(self) -> None:
        for kind in sandbox._WIKI_KINDS:
            with (
                self.subTest(kind=kind),
                patch.object(sandbox, "proxy_wiki", return_value=None),
                patch.object(sandbox, "proxy_gateway") as gateway,
                patch("sys.stderr", io.StringIO()) as stderr,
            ):
                self.assertEqual(sandbox.main(["--kind", kind, "--help"]), 75)
                self.assertIn("Supervisor Runtime capability is required", stderr.getvalue())
                gateway.assert_not_called()

    def test_all_journal_operations_use_the_unified_flag_style(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request_file = Path(directory) / "request.json"
            request = {"name": "test"}
            request_file.write_text(json.dumps(request))
            cases = [
                (
                    "update-direction",
                    "--request-file",
                    str(request_file),
                    "direction_update",
                    {"request": request},
                ),
                (
                    "record-experiment",
                    "--request-file",
                    str(request_file),
                    "experiment_record",
                    {"request": request},
                ),
                (
                    "episode-report",
                    "--request-file",
                    str(request_file),
                    "episode_report",
                    {"request": request},
                ),
                (
                    "list-directions",
                    "--output-path",
                    "scratch/directions.json",
                    "directions_list",
                    {"file": "scratch/directions.json"},
                ),
                (
                    "list-experiments",
                    "--output-path",
                    "scratch/experiments.json",
                    "experiments_list",
                    {"file": "scratch/experiments.json"},
                ),
                (
                    "load-direction",
                    "--record-id",
                    "direction_" + "1" * 32,
                    "direction_load",
                    {"direction_id": "direction_" + "1" * 32},
                ),
                (
                    "load-experiment",
                    "--record-id",
                    "experiment_" + "1" * 32,
                    "experiment_load",
                    {"experiment_id": "experiment_" + "1" * 32},
                ),
            ]
            for kind, flag, value, operation, payload in cases:
                with (
                    self.subTest(kind=kind),
                    patch.object(sandbox, "proxy_journal", return_value=0) as journal,
                    patch.object(sandbox, "proxy_gateway") as gateway,
                ):
                    self.assertEqual(sandbox.main(["--kind", kind, flag, value]), 0)
                    journal.assert_called_once_with(operation, **payload)
                    gateway.assert_not_called()

    def test_gateway_flags_and_dev_command_are_forwarded_unchanged(self) -> None:
        cases = [
            ["--kind", "run", "--no-sync"],
            ["--kind", "run", "--help"],
            ["--kind", "record-read", "--record-id", "gateway-0123456789abcdef0123456789abcdef"],
            ["--kind", "dev", "--", "python3", "probe.py", "--kind", "episode-report"],
            ["--", "python3", "probe.py", "--kind", "load-direction", "--help"],
            ["--kind", "dev", "--", "python3", "probe.py", "--kind", "wiki-query"],
        ]
        for arguments in cases:
            with (
                self.subTest(arguments=arguments),
                patch.object(sandbox, "proxy_gateway", return_value=7) as gateway,
                patch.object(sandbox, "proxy_journal") as journal,
                patch.object(sandbox, "proxy_wiki") as wiki,
            ):
                self.assertEqual(sandbox.main(arguments), 7)
                gateway.assert_called_once_with(arguments)
                journal.assert_not_called()
                wiki.assert_not_called()

    def test_invalid_requests_are_rejected_before_http(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            for content in ("{", "[]", "null"):
                path.write_text(content)
                with (
                    self.subTest(content=content),
                    patch.object(sandbox, "proxy_journal") as journal,
                    patch("sys.stderr", io.StringIO()),
                ):
                    with self.assertRaises(SystemExit) as error:
                        sandbox.main(["--kind", "episode-report", "--request-file", str(path)])
                    self.assertEqual(error.exception.code, 2)
                    journal.assert_not_called()

    def test_journal_rejects_missing_and_obsolete_flags(self) -> None:
        for arguments in (
            ["--kind", "load-direction", "--direction-id", "direction_" + "1" * 32],
            ["--kind", "list-experiments"],
            ["--kind", "episode-report", "--request", "request.json"],
        ):
            with self.subTest(arguments=arguments), patch("sys.stderr", io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    sandbox.main(arguments)
                self.assertEqual(error.exception.code, 2)

    def test_journal_requires_a_scoped_runtime_capability(self) -> None:
        with (
            patch.object(sandbox, "proxy_journal", return_value=None),
            patch("sys.stderr", io.StringIO()) as stderr,
        ):
            self.assertEqual(
                sandbox.main(
                    [
                        "--kind=list-directions",
                        "--output-path",
                        "scratch/directions.json",
                    ]
                ),
                75,
            )
            self.assertIn("Supervisor Runtime capability is required", stderr.getvalue())

    def test_general_help_lists_all_operations_without_http(self) -> None:
        with (
            patch.object(sandbox, "proxy_gateway") as gateway,
            patch.object(sandbox, "proxy_wiki") as wiki,
            patch("sys.stdout", io.StringIO()) as stdout,
        ):
            self.assertEqual(sandbox.main(["--help"]), 0)
            for kind in (*sandbox._JOURNAL_KINDS, *sandbox._WIKI_KINDS):
                self.assertIn(kind, stdout.getvalue())
            gateway.assert_not_called()
            wiki.assert_not_called()


if __name__ == "__main__":
    unittest.main()
