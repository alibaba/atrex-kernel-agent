"""Untrusted argv must not redirect privileged Gateway or Wiki operations."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from orchestrator.supervisor_runtime import (
    SupervisorRuntime,
    SupervisorRuntimeConfig,
    _CANONICAL_GATEWAY_VALUE_OPTIONS,
)
from supervisor.gateway import build_parser

ROOT = Path(__file__).resolve().parents[1]


class RuntimeArgumentAuthorityTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        workspace = self.root / "agent"
        workspace.mkdir()
        self.runtime = SupervisorRuntime(
            SupervisorRuntimeConfig(
                repository_root=ROOT,
                hardware="L20N",
                sandbox_timeout=600,
                sandbox_url="https://trusted.invalid",
                agent_sandbox="none",
            ),
            {},
        )
        self.addCleanup(self.runtime.close)
        self.capability = SimpleNamespace(
            workspace=workspace,
            request_environment={},
            campaign_root=self.root,
            evidence_root=self.root / "private" / "session" / "evidence",
            wiki_profile_root=self.root / "private" / "wiki-profile",
            worktree=None,
        )

    def test_gateway_rejects_authority_abbreviations_before_dispatch(self) -> None:
        for option in (
            "--ur",
            "--workspac",
            "--hardwar",
            "--gateway-p",
            "--ssh-i",
            "--ssh-g",
            "--ssh-runtime-b",
            "--health-c",
            "--time",
        ):
            for supplied in ([option, "untrusted"], [option + "=untrusted"]):
                with (
                    self.subTest(argv=supplied),
                    patch(
                        "orchestrator.supervisor_runtime.subprocess.run",
                        return_value=subprocess.CompletedProcess([], 0, "{}", ""),
                    ) as run,
                ):
                    with self.assertRaisesRegex(ValueError, "[Aa]bbreviat"):
                        self.runtime.execute_gateway(
                            self.capability,
                            {
                                "argv": ["--kind", "run", *supplied],
                            },
                        )
                    run.assert_not_called()
        self.assertFalse(self.capability.evidence_root.exists())

    def test_full_overrides_cannot_change_authority_and_dev_argv_is_preserved(self) -> None:
        supplied = [
            "--workspace=/another-session",
            "--hardware",
            "OTHER",
            "--url=https://evil.invalid",
            "--gateway-profile",
            "pre",
            "--ssh=other-host",
            "--ssh-init",
            "evil-init",
            "--ssh-gpu",
            "3",
            "--ssh-runtime-bind=/private",
            "--health-command=evil-health",
            "--timeout",
            "999",
            "--kind",
            "dev",
        ]
        remote = ["--", "python3", "probe.py", "--ur=remote-data", "--workspac", "literal"]
        with patch(
            "orchestrator.supervisor_runtime.subprocess.run",
            return_value=(subprocess.CompletedProcess([], 0, '{"ok":true}\n', "")),
        ) as run:
            self.runtime.execute_gateway(self.capability, {"argv": supplied + remote})
        argv = run.call_args.args[0][2:]
        parsed = build_parser().parse_args(argv)
        self.assertEqual(parsed.workspace, str(self.capability.workspace))
        self.assertEqual(parsed.hardware, "L20N")
        self.assertEqual(parsed.url, "https://trusted.invalid")
        self.assertEqual(parsed.timeout, 600)
        self.assertIsNone(parsed.gateway_profile)
        self.assertIsNone(parsed.ssh)
        self.assertIsNone(parsed.ssh_runtime_bind)
        self.assertEqual(parsed.command, remote)
        self.assertEqual(argv.count("--workspace"), 1)
        self.assertNotIn("https://evil.invalid", " ".join(argv))

    def test_gateway_parser_rejects_every_nonexact_authority_prefix(self) -> None:
        parser = build_parser()
        known = parser._option_string_actions
        for full in sorted(_CANONICAL_GATEWAY_VALUE_OPTIONS):
            for length in range(3, len(full)):
                prefix = full[:length]
                if prefix in known:
                    continue
                for supplied in ([prefix, "untrusted"], [prefix + "=untrusted"]):
                    with self.subTest(argv=supplied), redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as error:
                            parser.parse_args(["--kind", "run", *supplied])
                        self.assertEqual(error.exception.code, 2)

    def test_incomplete_override_cannot_consume_another_option_or_separator(self) -> None:
        for supplied in (
            ["--url"],
            ["--url", "--workspace", "/other"],
            ["--url", "--", "python3", "probe.py"],
        ):
            with (
                self.subTest(argv=supplied),
                patch("orchestrator.supervisor_runtime.subprocess.run") as run,
            ):
                with self.assertRaisesRegex(ValueError, "requires a value"):
                    self.runtime.execute_gateway(self.capability, {"argv": supplied})
                run.assert_not_called()

    def test_wiki_rejects_checked_option_abbreviations_before_dispatch(self) -> None:
        for tool, option in (
            ("query_nl", "--st"),
            ("query_nl", "--fi"),
            ("query_nl", "--keep-w"),
            ("query_nl", "--max-b"),
            ("query_wiki", "--json-s"),
            ("query_hardware", "--sto"),
        ):
            for supplied in ([option, "/private"], [option + "=/private"]):
                with (
                    self.subTest(tool=tool, argv=supplied),
                    patch(
                        "orchestrator.supervisor_runtime.subprocess.run",
                        return_value=subprocess.CompletedProcess([], 0, "{}", ""),
                    ) as run,
                ):
                    with self.assertRaisesRegex(ValueError, "[Aa]bbreviat"):
                        self.runtime.execute_wiki(self.capability, {"tool": tool, "argv": supplied})
                    run.assert_not_called()
        self.assertFalse(self.capability.wiki_profile_root.exists())

    def test_wiki_cli_parsers_reject_abbreviations_without_effects(self) -> None:
        environment = dict(os.environ)
        environment.pop("ATREX_AKA_RUNTIME_URL", None)
        environment.pop("ATREX_AKA_RUNTIME_TOKEN", None)
        environment["ATREX_WIKI_PROFILE_ROOT"] = str(self.root / "unexpected-telemetry")
        for tool, option in (
            ("query_nl", "--fi"),
            ("query_nl", "--store-r"),
            ("query_nl", "--keep-w"),
            ("query_wiki", "--json-s"),
            ("query_hardware", "--sto"),
        ):
            variants = (
                [[option]]
                if option == "--keep-w"
                else [
                    [option, str(self.root / "private-input")],
                    [option + "=" + str(self.root / "private-input")],
                ]
            )
            for supplied in variants:
                with self.subTest(tool=tool, argv=supplied):
                    process = subprocess.run(
                        [
                            sys.executable,
                            str(ROOT / "gpu-wiki" / "tools" / f"{tool}.py"),
                            *supplied,
                        ],
                        cwd=self.root,
                        env=environment,
                        text=True,
                        capture_output=True,
                        timeout=15,
                    )
                    self.assertEqual(process.returncode, 2, process.stderr)
                    self.assertIn("unrecognized arguments", process.stderr)
        self.assertFalse((self.root / "unexpected-telemetry").exists())

    def test_http_reports_repairable_errors_without_launching_privileged_process(self) -> None:
        self.runtime.start()
        self.runtime._capabilities["test-capability"] = self.capability
        for endpoint, payload in (
            ("/v1/gateway/execute", {"argv": ["--kind", "run", "--ur=https://evil.invalid"]}),
            ("/v1/wiki/query", {"tool": "query_nl", "argv": ["--fi=/private/input.py"]}),
        ):
            request = urllib.request.Request(
                self.runtime.url + endpoint,
                data=json.dumps(payload).encode(),
                headers={
                    "Authorization": "Bearer test-capability",
                    "Content-Type": "application/json",
                },
            )
            with (
                self.subTest(endpoint=endpoint),
                patch("orchestrator.supervisor_runtime.subprocess.run") as run,
                self.assertRaises(urllib.error.HTTPError) as rejected,
            ):
                urllib.request.urlopen(request, timeout=5)
            with rejected.exception as response:
                self.assertEqual(response.code, 400)
                error = json.load(response)
            self.assertTrue(error["repairable"])
            self.assertEqual(error["error"]["code"], "invalid_arguments")
            self.assertIn("complete", error["error"]["next_action"])
            run.assert_not_called()
