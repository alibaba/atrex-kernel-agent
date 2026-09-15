from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from orchestrator.agent_sandbox import VISIBLE_WORKSPACE, wrap_agent_command
from orchestrator.sandbox_launch import SandboxLaunch
from orchestrator.constants import ATREX_BENCH_RUNTIME_ENV
from orchestrator.session_io import _sandbox_command
from orchestrator.supervisor_runtime import (
    MAX_DEV_STDOUT_BYTES,
    MAX_GATEWAY_RESULT_BYTES,
    MAX_WIKI_PAYLOAD_BYTES,
    RUNTIME_URL_ENV,
    SUPERVISOR_EVIDENCE_ROOT_ENV,
    WIKI_PROFILE_ROOT_ENV,
    SupervisorRuntime,
    SupervisorRuntimeConfig,
    _gateway_output_limit,
    activate_supervisor_runtime,
    supervisor_campaign_root,
)
from orchestrator.workspace_runtime import link_runtime
from orchestrator.optimization_policy import install_workspace_policy, MODE_STATE_ENV

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


class SupervisorRuntimeTest(unittest.TestCase):
    @patch("orchestrator.agent_sandbox.platform.system", return_value="Linux")
    def test_projected_candidate_is_measured_then_reported_to_supervisor_git(self, _system: object) -> None:
        from long_horizon.store import CampaignStore
        from supervisor.journal import initialize_journal
        from supervisor.test_journal import _record_kernel

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve() / "worktree"
            workspace.mkdir()
            subprocess.run(["git", "init", "-qb", "episode-1"], cwd=workspace, check=True)
            CampaignStore.ensure_excluded(workspace)
            (workspace / "kernel.py").write_text("# seed\n")
            subprocess.run(["git", "add", "kernel.py"], cwd=workspace, check=True)
            subprocess.run([
                "git", "-c", "user.name=Test", "-c", "user.email=test@example.test",
                "commit", "-qm", "seed",
            ], cwd=workspace, check=True)
            base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=workspace, text=True).strip()
            runtime = SupervisorRuntime(self._config(agent_sandbox="bwrap", bwrap_executable="/usr/bin/true"))
            runtime._server = SimpleNamespace(server_address=("127.0.0.1", 12345))
            try:
                lease = runtime.prepare_session(["true"], workspace, dict(os.environ))
                capability = runtime.authorize(lease.token)
                assert capability is not None
                source = b"def run(x): return x\n"
                (capability.workspace / "kernel.tmp").write_bytes(source)
                (capability.workspace / "kernel.tmp").replace(capability.workspace / "kernel.py")
                self.assertEqual((workspace / "kernel.py").read_text(), "# seed\n")
                with patch("orchestrator.supervisor_runtime.subprocess.run", return_value=subprocess.CompletedProcess(
                    [], 0, '{"result":{"all_pass":true}}\n', "",
                )) as submit:
                    runtime.execute_gateway(capability, {"argv": ["--kind", "run"]})
                self.assertEqual(submit.call_args.kwargs["cwd"], capability.workspace)
                self.assertEqual(submit.call_args.kwargs["env"][MODE_STATE_ENV], str(
                    capability.evidence_root.parents[2] / "optimization-policy.json",
                ))
                _, record_id = _record_kernel(capability.evidence_root, source)
                path = capability.evidence_root / "journal.json"
                initialize_journal(path, episode=1, base_commit=base, branch="episode-1")

                def journal_request(operation: str, body: dict) -> dict:
                    response = runtime.execute_journal(capability, {"operation": operation, "request": body})
                    self.assertEqual(response["exit_code"], 0, response)
                    return json.loads(response["stdout"])

                direction_id = journal_request("direction_update", {
                    "action": "propose", "name": "identity", "hypothesis": "avoid a copy",
                    "rationale": "memory traffic", "plan": ["measure candidate"],
                    "success_criteria": ["correct and faster"], "stop_conditions": ["no gain"],
                })["direction_id"]
                journal_request("direction_update", {
                    "action": "start", "direction_id": direction_id, "analysis": "explore",
                })
                experiment_id = journal_request("experiment_record", {
                    "direction_id": direction_id, "name": "identity measurement",
                    "hypothesis": "avoid a copy", "change": "return input", "action": "keep_after",
                    "gateway_record_ids": [record_id],
                    "evidence": "Evaluate passed", "analysis": "candidate is correct",
                })["experiment_id"]
                journal_request("direction_update", {
                    "action": "complete", "direction_id": direction_id,
                    "analysis": "measured candidate", "hypothesis_status": "supported",
                    "supporting_experiment_ids": [experiment_id],
                })
                request = {"operation": "episode_report", "request": {
                    "status": "candidate_ready", "summary": "measured candidate",
                    "selected_experiment_id": experiment_id,
                }}
                (capability.workspace / "kernel.py").write_text("# not measured\n")
                result = runtime.execute_journal(capability, request)
                self.assertEqual(result["exit_code"], 2)
                self.assertIn("Kernel exactly matches current kernel.py", result["stdout"])
                (capability.workspace / "kernel.py").write_bytes(source)
                result = runtime.execute_journal(capability, request)
                self.assertEqual(result["exit_code"], 0, result)
                self.assertEqual(subprocess.check_output(["git", "show", "HEAD:kernel.py"], cwd=workspace), source)
                self.assertTrue((workspace / ".atrex_long_horizon/handoff.json").exists())
                self.assertFalse((capability.workspace / ".atrex_long_horizon").exists())
                self.assertFalse((capability.workspace / ".git").exists())
                lease.close()
            finally:
                runtime._server = None
                runtime.close()

    def test_wiki_cannot_read_hidden_legacy_promotion_audits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            (workspace / "memory").mkdir()
            (workspace / "scratch").mkdir()
            audit = workspace / "memory/long_horizon_e0001.json"
            audit.write_text('{"private":true}')
            (workspace / "scratch/query.txt").symlink_to(audit)
            runtime = SupervisorRuntime(self._config())
            try:
                for path in ("memory/long_horizon_e0001.json", "scratch/query.txt"):
                    with self.subTest(path=path), self.assertRaisesRegex(ValueError, "Supervisor-only"):
                        runtime.execute_wiki(SimpleNamespace(workspace=workspace), {
                            "tool": "query_nl", "argv": ["--file", path],
                        })
            finally:
                runtime.close()

    def _config(self, **overrides: object) -> SupervisorRuntimeConfig:
        values: dict[str, object] = {
            "repository_root": REPOSITORY_ROOT,
            "hardware": "L20N",
            "sandbox_timeout": 120,
            "sandbox_url": "https://gateway.example.test",
            "agent_sandbox": "none",
        }
        values.update(overrides)
        return SupervisorRuntimeConfig(**values)  # type: ignore[arg-type]

    def test_agent_sandbox_facade_fails_closed_without_runtime(self) -> None:
        environment = os.environ.copy()
        environment.pop("ATREX_AKA_RUNTIME_URL", None)
        environment.pop("ATREX_AKA_RUNTIME_TOKEN", None)
        process = subprocess.run(
            [sys.executable, str(REPOSITORY_ROOT / "tools" / "sandbox.py")],
            cwd=REPOSITORY_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(process.returncode, 75)
        self.assertIn("Supervisor Runtime capability is required", process.stderr)

    def test_only_installed_skill_provides_wiki_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = SupervisorRuntime(self._config())
            try:
                guide = (REPOSITORY_ROOT / "skills" / "KernelWiki" / "SKILL.md").read_text()
                self.assertFalse((runtime.state_root / "wiki-facade").exists())
                for kind in ("wiki-query", "wiki-search", "wiki-hardware"):
                    self.assertIn("tools/sandbox.py --kind " + kind, guide)
                workspace = Path(directory)
                # A resumed workspace can still contain the formerly installed
                # external Skill, whose scripts are not available in the sandbox.
                for name in (".claude", ".qoder", ".agents"):
                    destination = workspace / name / "skills" / "KernelWiki"
                    destination.parent.mkdir(parents=True)
                    destination.symlink_to(
                        REPOSITORY_ROOT / "gpu-wiki" / "3rdparty" / "KernelWiki"
                    )
                link_runtime(workspace)
                self.assertFalse((workspace / "gpu-wiki").exists())
                for name in (".claude", ".qoder", ".agents"):
                    text = (workspace / name / "skills" / "KernelWiki" / "SKILL.md").read_text()
                    self.assertIn(guide[guide.index("# GPU Wiki"):], text)
                    self.assertIn("Runtime execution contract", text)
            finally:
                runtime._temporary.cleanup()

    def test_unified_cli_wiki_round_trip_over_http(self) -> None:
        run_client = subprocess.run
        cases = (
            ("wiki-query", "query_nl", ["--file", "scratch/query.txt"], 0),
            ("wiki-search", "query_wiki", ["--arch", "sm_100", "--coverage"], 0),
            ("wiki-hardware", "query_hardware", ["--product", "h20"], 4),
        )
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "scratch").mkdir()
            (workspace / "scratch" / "query.txt").write_text("memory-bound normalization")
            with SupervisorRuntime(self._config()) as runtime:
                lease = runtime.prepare_session(["claude"], workspace, dict(os.environ))
                capability = runtime.authorize(lease.token)
                assert capability is not None
                for kind, tool, options, code in cases:
                    payload = {"records": {"example": {"payload": {"summary": tool}}}}
                    completed = subprocess.CompletedProcess(
                        args=[tool], returncode=code, stdout=json.dumps(payload) + "\n",
                        stderr="not recorded\n" if code else "",
                    )
                    with self.subTest(kind=kind), patch(
                        "orchestrator.supervisor_runtime.subprocess.run", return_value=completed
                    ) as query:
                        process = run_client(
                            [sys.executable, str(REPOSITORY_ROOT / "tools" / "sandbox.py"),
                             "--kind", kind, *options],
                            cwd=workspace, env=lease.environment, text=True,
                            capture_output=True, check=False, timeout=15,
                        )
                    self.assertEqual(process.returncode, code, process.stderr + process.stdout)
                    self.assertEqual(json.loads(process.stdout), payload)
                    self.assertEqual(process.stderr, completed.stderr)
                    command = query.call_args.args[0]
                    self.assertEqual(Path(command[1]).name, tool + ".py")
                    self.assertEqual(command[2:2 + len(options)], options)
                    self.assertNotIn("--kind", command)
                    self.assertNotIn(RUNTIME_URL_ENV, query.call_args.kwargs["env"])
                audit = [
                    json.loads(line)
                    for line in (capability.evidence_root / "runtime-requests.jsonl")
                    .read_text().splitlines()
                ]
                self.assertEqual(
                    [row["operation"] for row in audit],
                    ["wiki:" + tool for _, tool, _, _ in cases],
                )
                self.assertEqual([row["exit_code"] for row in audit], [0, 0, 4])
                self.assertTrue(all(row["response_record_id"] for row in audit))
                lease.close()

    def test_unified_cli_journal_round_trip_and_repair_over_http(self) -> None:
        from supervisor.journal import initialize_journal, load_journal

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            scratch = workspace / "scratch"
            scratch.mkdir()
            with SupervisorRuntime(self._config()) as runtime:
                lease = runtime.prepare_session(["claude"], workspace, dict(os.environ))
                capability = runtime.authorize(lease.token)
                assert capability is not None
                initialize_journal(
                    capability.evidence_root / "journal.json",
                    episode=1,
                    base_commit="1" * 40,
                    branch="episode-1",
                )

                def call(
                    kind: str, *options: str, request: object = None, expected_code: int = 0
                ) -> dict:
                    if request is not None:
                        (scratch / "request.json").write_text(json.dumps(request))
                        options = ("--request-file", "scratch/request.json")
                    process = subprocess.run(
                        [
                            sys.executable,
                            str(REPOSITORY_ROOT / "tools" / "sandbox.py"),
                            "--kind",
                            kind,
                            *options,
                        ],
                        cwd=workspace,
                        env=lease.environment,
                        text=True,
                        capture_output=True,
                        timeout=15,
                        check=False,
                    )
                    self.assertEqual(
                        process.returncode, expected_code, process.stderr + process.stdout
                    )
                    return json.loads(process.stdout)

                direction = call(
                    "update-direction",
                    request={
                        "action": "propose",
                        "name": "fusion",
                        "hypothesis": "reduce writes",
                        "rationale": "intermediate tensor",
                        "plan": ["fuse", "measure"],
                        "success_criteria": ["faster"],
                        "stop_conditions": ["resource pressure"],
                    },
                )["direction_id"]
                call(
                    "update-direction",
                    request={
                        "action": "start",
                        "direction_id": direction,
                        "analysis": "start",
                    },
                )
                rejected = call(
                    "record-experiment",
                    request={
                        "direction_id": direction,
                    },
                    expected_code=2,
                )
                self.assertTrue(rejected["repairable"])
                from supervisor import gateway

                (workspace / "kernel.py").write_text("def run(x): return x\n")
                with patch.dict(os.environ, {gateway.SUPERVISOR_EVIDENCE_ROOT_ENV: str(capability.evidence_root)}):
                    diagnostic = gateway._record_episode_evaluation(
                        workspace, {"passed": False, "status": "completed"}, gateway_kind="check",
                    )
                experiment = call(
                    "record-experiment",
                    request={
                        "direction_id": direction,
                        "name": "structural dead end",
                        "hypothesis": "fusion",
                        "change": "investigated fusion",
                        "gateway_record_ids": [diagnostic["record_id"]],
                        "evidence": "incompatible layouts",
                        "analysis": "abandon this path",
                        "action": "abandon_direction",
                    },
                )["experiment_id"]
                call(
                    "update-direction",
                    request={
                        "action": "abandon",
                        "direction_id": direction,
                        "analysis": "dead end",
                        "hypothesis_status": "unresolved",
                        "supporting_experiment_ids": [experiment],
                    },
                )
                self.assertEqual(
                    call("load-direction", "--record-id", direction)["status"], "abandoned"
                )
                self.assertEqual(
                    call("load-experiment", "--record-id", experiment)["name"],
                    "structural dead end",
                )
                for collection in ("directions", "experiments"):
                    result = call(
                        "list-" + collection, "--output-path", f"scratch/{collection}.json"
                    )
                    self.assertEqual(result["count"], 1)
                    self.assertEqual(
                        len(json.loads((scratch / f"{collection}.json").read_text())[collection]), 1
                    )
                rejected = call(
                    "episode-report",
                    request={
                        "status": "candidate_ready",
                        "summary": "missing candidate",
                    },
                    expected_code=2,
                )
                self.assertTrue(rejected["repairable"])
                journal_path = capability.evidence_root / "journal.json"
                before_report = journal_path.read_bytes()
                rejected = call(
                    "episode-report",
                    request={
                        "status": "candidate_ready", "summary": "old report format",
                        "selected_experiment_index": 1,
                    },
                    expected_code=2,
                )
                self.assertTrue(rejected["repairable"])
                self.assertEqual(rejected["error"]["unexpected_fields"], ["selected_experiment_index"])
                self.assertEqual(journal_path.read_bytes(), before_report)
                self.assertFalse((workspace / ".atrex_long_horizon" / "handoff.json").exists())
                self.assertEqual(
                    call(
                        "episode-report",
                        request={
                            "status": "pivot",
                            "summary": "explore a different layout",
                        },
                    )["status"],
                    "accepted",
                )
                self.assertEqual(
                    load_journal(capability.evidence_root / "journal.json")["state"], "pivot"
                )
                damaged = load_journal(journal_path)
                damaged["direction_events"] = [{
                    "direction_id": direction, "action": "complete",
                }]
                journal_path.write_text(json.dumps(damaged))
                unavailable = call(
                    "episode-report", request={"status": "pivot", "summary": "cannot repair private history"},
                    expected_code=75,
                )
                self.assertFalse(unavailable["repairable"])
                self.assertIn("Invalid Direction history", unavailable["error"]["message"])
                self.assertIn("operator", unavailable["error"]["next_action"])
                (capability.evidence_root / "journal.json").write_text("corrupt fixture")
                unavailable = call(
                    "list-directions", "--output-path", "scratch/directions.json",
                    expected_code=75,
                )
                self.assertFalse(unavailable["repairable"])
                self.assertEqual(unavailable["error"]["code"], "runtime_state_unavailable")
                self.assertIn("operator", unavailable["error"]["next_action"])
                lease.close()

    def test_http_internal_failure_and_revocation_do_not_recommend_blind_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            with SupervisorRuntime(self._config()) as runtime:
                lease = runtime.prepare_session(["claude"], workspace, dict(os.environ))

                def call() -> dict:
                    process = subprocess.run(
                        [sys.executable, str(REPOSITORY_ROOT / "tools/sandbox.py"),
                         "--kind", "env"],
                        cwd=workspace, env=lease.environment, text=True, capture_output=True,
                        timeout=10, check=False,
                    )
                    self.assertEqual(process.returncode, 75)
                    self.assertNotIn("private failure detail", process.stderr)
                    return json.loads(process.stderr)

                with patch.object(runtime, "execute_gateway", side_effect=RuntimeError(
                    "private failure detail"
                )), self.assertLogs("orchestrator.supervisor_runtime", level="ERROR") as logs:
                    failed = call()
                self.assertFalse(failed["repairable"])
                self.assertIn("do not retry blindly", failed["error"]["next_action"])
                self.assertIn(failed["error"]["error_id"], "\n".join(logs.output))
                lease.close()
                revoked = call()
                self.assertFalse(revoked["repairable"])
                self.assertEqual(revoked["error"]["code"], "invalid_capability")
                self.assertIn("operator", revoked["error"]["next_action"])

    def test_capability_hides_credentials_and_binds_gateway_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            environment = {
                "HOME": str(Path.home()),
                "PATH": os.environ.get("PATH", ""),
                "ATREX_AGENT_CLI": "claude",
                "AGATE_AK": "must-not-enter-agent",
                "AGATE_SK": "must-not-enter-agent",
                "ATREX_PRIVATE_REFERENCE_DIR": "/private/reference",
                "ATREX_TELEMETRY_ATTEMPT_ID": "attempt-7",
                "ATREX_WIKI_TASK_ID": "campaign-1",
                "ATREX_AKA_REUSE_GATEWAY_RESULTS": "1",
                "ATREX_AKA_INTERNAL_MEASUREMENT": "1",
                "ATREX_AKA_COMPARISON_RUN_TIMEOUT": "999",
            }
            completed = subprocess.CompletedProcess(
                args=["sandbox"], returncode=0, stdout='{"ok":true}\n', stderr=""
            )
            runtime = SupervisorRuntime(self._config(), environment)
            runtime._server = SimpleNamespace(  # type: ignore[assignment]
                server_address=("127.0.0.1", 12345)
            )
            try:
                lease = runtime.prepare_session(["claude", "-p", "hello"], workspace, environment)
                self.assertNotIn("AGATE_AK", lease.environment)
                self.assertNotIn("AGATE_SK", lease.environment)
                self.assertNotIn("ATREX_PRIVATE_REFERENCE_DIR", lease.environment)
                self.assertNotIn("ATREX_AKA_REUSE_GATEWAY_RESULTS", lease.environment)
                self.assertNotIn("ATREX_AKA_INTERNAL_MEASUREMENT", lease.environment)
                self.assertEqual(lease.environment[RUNTIME_URL_ENV], runtime.url)

                capability = runtime.authorize(lease.token)
                assert capability is not None
                with patch(
                    "orchestrator.supervisor_runtime.subprocess.run",
                    return_value=completed,
                ) as run:
                    result = runtime.execute_gateway(
                        capability,
                        {
                            "argv": [
                                "--workspace",
                                "/tmp/escape",
                                "--hardware",
                                "OTHER",
                                "--url",
                                "https://evil.invalid",
                                "--kind",
                                "run",
                                "--no-sync",
                                "--",
                                "echo",
                                "ok",
                            ]
                        },
                    )
                self.assertEqual(result["exit_code"], 0)
                self.assertEqual(result["stdout"], '{"ok":true}\n')
                command = run.call_args.args[0]
                self.assertEqual(Path(command[1]), REPOSITORY_ROOT / "supervisor" / "gateway.py")
                self.assertEqual(command.count("--workspace"), 1)
                self.assertEqual(
                    command[command.index("--workspace") + 1], str(capability.workspace)
                )
                self.assertEqual(command.count("--hardware"), 1)
                self.assertEqual(command[command.index("--hardware") + 1], "L20N")
                self.assertNotIn("https://evil.invalid", command)
                self.assertIn("https://gateway.example.test", command)
                request_environment = run.call_args.kwargs["env"]
                self.assertNotIn("ATREX_AKA_REUSE_GATEWAY_RESULTS", request_environment)
                self.assertNotIn("ATREX_AKA_INTERNAL_MEASUREMENT", request_environment)
                self.assertNotIn("ATREX_AKA_COMPARISON_RUN_TIMEOUT", request_environment)
                self.assertEqual(request_environment["ATREX_TELEMETRY_ATTEMPT_ID"], "attempt-7")
                self.assertEqual(request_environment["ATREX_WIKI_TASK_ID"], "campaign-1")
                self.assertEqual(
                    request_environment[SUPERVISOR_EVIDENCE_ROOT_ENV],
                    str(capability.evidence_root),
                )
                self.assertTrue((capability.evidence_root / "runtime-requests.jsonl").is_file())
                self.assertTrue((capability.evidence_root / "gateway-records").is_dir())
                self.assertFalse((workspace / ".atrex_long_horizon").exists())

                lease.close()
                self.assertIsNone(runtime.authorize(lease.token))
            finally:
                runtime._server = None
                runtime._temporary.cleanup()

    def test_wiki_rejects_store_and_workspace_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "link").symlink_to("/tmp", target_is_directory=True)
            runtime = SupervisorRuntime(self._config())
            runtime._server = SimpleNamespace(  # type: ignore[assignment]
                server_address=("127.0.0.1", 12345)
            )
            try:
                lease = runtime.prepare_session(
                    ["claude", "-p", "hello"],
                    workspace,
                    {"HOME": str(Path.home()), "PATH": os.environ.get("PATH", "")},
                )
                capability = runtime.authorize(lease.token)
                assert capability is not None
                with self.assertRaisesRegex(ValueError, "override"):
                    runtime.execute_wiki(
                        capability,
                        {"tool": "query_nl", "argv": ["query", "--store-root", "/tmp"]},
                    )
                with self.assertRaisesRegex(ValueError, "inside"):
                    runtime.execute_wiki(
                        capability,
                        {"tool": "query_nl", "argv": ["--file=link/request.txt"]},
                    )
                with self.assertRaisesRegex(ValueError, "retain"):
                    runtime.execute_wiki(
                        capability,
                        {"tool": "query_nl", "argv": ["query", "--keep-workspace"]},
                    )
                lease.close()
            finally:
                runtime._server = None
                runtime._temporary.cleanup()

    def test_dev_output_is_bounded_but_exact_response_is_private(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            runtime = SupervisorRuntime(self._config())
            runtime._server = SimpleNamespace(  # type: ignore[assignment]
                server_address=("127.0.0.1", 12345)
            )
            try:
                lease = runtime.prepare_session(
                    ["claude", "-p", "hello"],
                    workspace,
                    {"HOME": str(Path.home()), "PATH": os.environ.get("PATH", "")},
                )
                capability = runtime.authorize(lease.token)
                assert capability is not None
                raw_stdout = "begin\n" + "x" * (MAX_DEV_STDOUT_BYTES * 2) + "\nend\n"
                completed = subprocess.CompletedProcess(
                    args=["sandbox"], returncode=0, stdout=raw_stdout, stderr=""
                )
                with patch(
                    "orchestrator.supervisor_runtime.subprocess.run",
                    return_value=completed,
                ):
                    result = runtime.execute_gateway(
                        capability,
                        {"argv": ["--kind", "dev", "--", "python3", "probe.py"]},
                    )

                self.assertIn("truncated", result)
                self.assertLessEqual(
                    len(str(result["stdout"]).encode("utf-8")),
                    MAX_DEV_STDOUT_BYTES + 4,
                )
                self.assertIn("begin", str(result["stdout"]))
                self.assertIn("end", str(result["stdout"]))
                records = list((capability.evidence_root / "runtime-responses").glob("*.json"))
                self.assertEqual(len(records), 1)
                private = json.loads(records[0].read_text(encoding="utf-8"))
                self.assertEqual(private["stdout"], raw_stdout)
                audit = json.loads(
                    (capability.evidence_root / "runtime-requests.jsonl")
                    .read_text(encoding="utf-8")
                    .strip()
                )
                self.assertEqual(audit["response_record_id"], private["response_record_id"])
                lease.close()
            finally:
                runtime._server = None
                runtime._temporary.cleanup()

    def test_typed_diagnostic_output_uses_gateway_result_limit(self) -> None:
        self.assertEqual(
            _gateway_output_limit(["--kind", "check", "--no-sync"]),
            MAX_GATEWAY_RESULT_BYTES,
        )
        self.assertEqual(
            _gateway_output_limit(["--kind", "disassemble", "--format", "sass", "--no-sync"]),
            MAX_GATEWAY_RESULT_BYTES,
        )
        self.assertEqual(
            _gateway_output_limit(["--kind", "record-read", "--record-id", "gateway-1"]),
            MAX_GATEWAY_RESULT_BYTES,
        )

    def test_dry_run_hides_supervisor_authority_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            runtime = SupervisorRuntime(self._config())
            runtime._server = SimpleNamespace(  # type: ignore[assignment]
                server_address=("127.0.0.1", 12345)
            )
            try:
                lease = runtime.prepare_session(
                    ["claude", "-p", "hello"],
                    workspace,
                    {"HOME": str(Path.home()), "PATH": os.environ.get("PATH", "")},
                )
                capability = runtime.authorize(lease.token)
                assert capability is not None
                raw = json.dumps(
                    {
                        "hardware": "L20N",
                        "kind": "run",
                        "workspace": str(workspace),
                        "url": "https://gateway.example.test",
                        "candidate_bytes": 100,
                        "shape_count": "private",
                    }
                )
                completed = subprocess.CompletedProcess(
                    args=["sandbox"], returncode=0, stdout=raw, stderr=""
                )
                with patch(
                    "orchestrator.supervisor_runtime.subprocess.run",
                    return_value=completed,
                ):
                    result = runtime.execute_gateway(
                        capability,
                        {"argv": ["--kind", "run", "--dry-run", "--", "echo"]},
                    )
                visible = json.loads(str(result["stdout"]))
                self.assertEqual(visible["hardware"], "L20N")
                self.assertNotIn("workspace", visible)
                self.assertNotIn("url", visible)
                private = next(
                    (capability.evidence_root / "runtime-responses").glob("*.json")
                ).read_text(encoding="utf-8")
                self.assertIn(str(workspace), private)
                self.assertIn("gateway.example.test", private)
                lease.close()
            finally:
                runtime._server = None
                runtime._temporary.cleanup()

    def test_wiki_enforces_brief_context_projection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            runtime = SupervisorRuntime(self._config())
            runtime._server = SimpleNamespace(  # type: ignore[assignment]
                server_address=("127.0.0.1", 12345)
            )
            try:
                lease = runtime.prepare_session(
                    ["claude", "-p", "hello"],
                    workspace,
                    {"HOME": str(Path.home()), "PATH": os.environ.get("PATH", "")},
                )
                capability = runtime.authorize(lease.token)
                assert capability is not None
                completed = subprocess.CompletedProcess(
                    args=["query_nl"], returncode=0, stdout='{"records":{}}\n', stderr=""
                )
                with patch(
                    "orchestrator.supervisor_runtime.subprocess.run",
                    return_value=completed,
                ) as run:
                    runtime.execute_wiki(
                        capability,
                        {
                            "tool": "query_nl",
                            "argv": ["memory-bound attention", "--max-bytes", "999999"],
                        },
                    )
                command = run.call_args.args[0]
                self.assertIn("--brief", command)
                index = command.index("--max-bytes")
                self.assertEqual(command[index + 1], str(MAX_WIKI_PAYLOAD_BYTES))
                profile_root = capability.wiki_profile_root
                self.assertEqual(
                    run.call_args.kwargs["env"][WIKI_PROFILE_ROOT_ENV], str(profile_root)
                )
                self.assertFalse(profile_root.is_relative_to(workspace))
                self.assertNotIn(WIKI_PROFILE_ROOT_ENV, lease.environment)
                lease.close()
            finally:
                runtime._server = None
                runtime._temporary.cleanup()

    def test_wiki_events_share_private_campaign_root_across_worktrees_and_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            campaign = root / "campaign"
            campaign.mkdir()

            def git(*args: str) -> None:
                subprocess.run(
                    ["git", "-C", str(campaign), *args],
                    check=True, capture_output=True, text=True,
                )

            git("init", "-q")
            git("-c", "user.name=Test", "-c", "user.email=test@example.test",
                "commit", "--allow-empty", "-qm", "seed")
            episodes = [root / "episode-1", root / "episode-2"]
            for episode in episodes:
                git("worktree", "add", "--detach", str(episode), "HEAD")
            private_root = supervisor_campaign_root(campaign)
            profile_root = private_root / "wiki-profile"
            override = campaign / ".gpu_wiki_profile"
            environment = {
                "HOME": str(Path.home()), "PATH": os.environ.get("PATH", ""),
                WIKI_PROFILE_ROOT_ENV: str(override), "ATREX_WIKI_TASK_ID": "test-campaign",
            }
            run_ids = []
            for index, episode in enumerate([*episodes, episodes[0]], start=1):
                # Use the real Wiki producer in dry-run mode: no model, GPU,
                # or remote service is called. Recreate the Runtime each time.
                runtime = SupervisorRuntime(self._config(), environment)
                runtime._server = SimpleNamespace(server_address=("127.0.0.1", 12345))
                try:
                    # Exercise HTTP scoping only; no Agent process is launched.
                    with patch(
                        "orchestrator.agent_sandbox.wrap_agent_command",
                        side_effect=lambda command, **kwargs: SandboxLaunch(command, kwargs["environment"]),
                    ):
                        lease = runtime.prepare_session(["claude"], episode, environment)
                    capability = runtime.authorize(lease.token)
                    assert capability is not None
                    self.assertEqual(capability.wiki_profile_root, profile_root)
                    self.assertEqual(supervisor_campaign_root(episode), private_root)
                    self.assertNotIn(WIKI_PROFILE_ROOT_ENV, lease.environment)
                    self.assertNotIn(WIKI_PROFILE_ROOT_ENV, dict(capability.request_environment))
                    result = runtime.execute_wiki(
                        capability, {"tool": "query_nl", "argv": [f"query {index}", "--dry-run"]}
                    )
                    self.assertEqual(result["exit_code"], 0, result["stderr"])
                    self.assertNotIn(str(profile_root), str(result))
                    lease.close()
                finally:
                    runtime._server = None
                    runtime.close()
                events = [json.loads(path.read_text()) for path in
                          profile_root.glob("raw/query_events/*/*.json")]
                self.assertEqual(len(events), index)
                self.assertEqual({event["request"] for event in events},
                                 {f"query {n}" for n in range(1, index + 1)})
                self.assertTrue(all(event["status"] == "dry_run" for event in events))
                self.assertTrue(all(event["task_id"] == "test-campaign" for event in events))
                run_ids.append(json.loads((profile_root / "run.json").read_text())["run_id"])
            self.assertEqual(len(set(run_ids)), 1)
            self.assertFalse(override.exists())
            self.assertTrue(
                all(not (episode / ".gpu_wiki_profile").exists() for episode in episodes)
            )

    def test_wiki_rejects_symlinked_private_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            workspace.mkdir()
            runtime = SupervisorRuntime(self._config())
            runtime._server = SimpleNamespace(server_address=("127.0.0.1", 12345))
            try:
                lease = runtime.prepare_session(["claude"], workspace, {})
                capability = runtime.authorize(lease.token)
                assert capability is not None
                capability.wiki_profile_root.symlink_to(workspace, target_is_directory=True)
                with patch("orchestrator.supervisor_runtime.subprocess.run") as run:
                    with self.assertRaisesRegex(ValueError, "cannot be a symlink"):
                        runtime.execute_wiki(capability, {"tool": "query_nl", "argv": ["query"]})
                    run.assert_not_called()
                lease.close()
            finally:
                runtime._server = None
                runtime.close()

    def test_atrex_bench_runtime_is_private_and_not_in_agent_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            (source / "scripts").mkdir(parents=True)
            (source / "scripts" / "run_eval.py").write_text("print('eval')\n")
            package = source / "src" / "atrex_bench"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("")
            (package / "utils.py").write_text("VALUE = 1\n")
            workspace = root / "workspace"
            workspace.mkdir()
            runtime = SupervisorRuntime(self._config(atrex_bench_root=source))
            runtime._server = SimpleNamespace(  # type: ignore[assignment]
                server_address=("127.0.0.1", 12345)
            )
            try:
                private_runtime = runtime.atrex_bench_runtime
                assert private_runtime is not None
                self.assertTrue((private_runtime / "scripts" / "run_eval.py").is_file())
                self.assertTrue((private_runtime / "src" / "atrex_bench" / "utils.py").is_file())
                self.assertTrue(private_runtime.is_relative_to(runtime.state_root))
                self.assertFalse((workspace / "atrex-bench").exists())

                lease = runtime.prepare_session(
                    ["claude", "-p", "hello"],
                    workspace,
                    {
                        "HOME": str(Path.home()),
                        "PATH": os.environ.get("PATH", ""),
                        ATREX_BENCH_RUNTIME_ENV: "/agent-must-not-see-this",
                    },
                )
                self.assertNotIn(ATREX_BENCH_RUNTIME_ENV, lease.environment)
                capability = runtime.authorize(lease.token)
                assert capability is not None
                completed = subprocess.CompletedProcess(
                    args=["sandbox"], returncode=0, stdout="{}\n", stderr=""
                )
                with patch(
                    "orchestrator.supervisor_runtime.subprocess.run",
                    return_value=completed,
                ) as run:
                    runtime.execute_gateway(
                        capability,
                        {"argv": ["--kind", "dev", "--", "echo", "ok"]},
                    )
                self.assertEqual(
                    run.call_args.kwargs["env"][ATREX_BENCH_RUNTIME_ENV],
                    str(private_runtime),
                )
                lease.close()
            finally:
                runtime._server = None
                runtime._temporary.cleanup()

    def test_workspace_linking_rejects_unmanaged_legacy_atrex_bench_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            legacy = workspace / "atrex-bench"
            legacy.mkdir()
            (legacy / "agent-visible.py").write_text("SHOULD_NOT_EXIST = True\n")

            with self.assertRaisesRegex(RuntimeError, "explicit cleanup"):
                link_runtime(workspace)
            self.assertTrue((legacy / "agent-visible.py").exists())

    def test_supervisor_evaluation_uses_private_runtime_and_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            (source / "scripts").mkdir(parents=True)
            (source / "scripts" / "run_eval.py").write_text("print('eval')\n")
            package = source / "src" / "atrex_bench"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("")
            (package / "utils.py").write_text("VALUE = 1\n")
            workspace = root / "workspace"
            workspace.mkdir()
            runtime = SupervisorRuntime(self._config(atrex_bench_root=source))
            process = SimpleNamespace(
                pid=123,
                returncode=0,
                communicate=lambda timeout=None: ("ok\n", ""),
            )
            try:
                with (
                    activate_supervisor_runtime(runtime),
                    patch(
                        "orchestrator.session_io.spawn_owned_session",
                        return_value=process,
                    ) as spawn,
                ):
                    result = _sandbox_command(
                        workspace,
                        "L20N",
                        "",
                        "https://gateway.example.test",
                        120,
                        ["python3", "test_kernel.py", "--no-memory"],
                    )
                self.assertEqual(result.returncode, 0)
                environment = spawn.call_args.kwargs["environment"]
                self.assertEqual(
                    environment[ATREX_BENCH_RUNTIME_ENV],
                    str(runtime.atrex_bench_runtime),
                )
                self.assertEqual(
                    environment[SUPERVISOR_EVIDENCE_ROOT_ENV],
                    str(runtime.evidence_root(workspace)),
                )
            finally:
                runtime.close()

    @patch("orchestrator.agent_sandbox.platform.system", return_value="Linux")
    def test_bwrap_projects_only_one_writable_workspace(self, _system: object) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "kernel.py").write_text("def run(x): return x\n")
            provider_homes = root / "provider-homes"
            provider_homes.mkdir()
            launch = wrap_agent_command(
                ["bash", "-lc", f"pwd; touch {workspace.resolve()}/candidate"],
                workspace=workspace,
                environment={
                    "HOME": str(root),
                    "PATH": os.environ.get("PATH", ""),
                    "ATREX_AKA_RUNTIME_URL": "http://127.0.0.1:1234",
                    "ATREX_AKA_RUNTIME_TOKEN": "capability",
                },
                repository_root=REPOSITORY_ROOT,
                provider_homes=provider_homes,
                hidden_host_paths=(root,),
                mode="bwrap",
                bwrap_executable="/usr/bin/true",
            )
            self.addCleanup(launch.close)
            command, mapped = launch.command, launch.environment
            self.assertEqual(command[0], "/usr/bin/true")
            self.assertIn("--ro-bind", command)
            self.assertIn("--bind", command)
            self.assertIn(str(VISIBLE_WORKSPACE), command)
            self.assertNotIn("--unshare-net", command)
            self.assertEqual(mapped["ATREX_WORKSPACE"], str(VISIBLE_WORKSPACE))
            self.assertEqual(mapped["HOME"], str(VISIBLE_WORKSPACE))
            self.assertNotIn(str(workspace.resolve()) + "/candidate", "\0".join(command))
            self.assertNotIn("gateway-records", "\0".join(command))

    @patch("orchestrator.agent_sandbox.platform.system", return_value="Linux")
    def test_provider_session_home_persists_in_episode_workspace(self, _system: object) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            runtime = SupervisorRuntime(
                self._config(agent_sandbox="bwrap", bwrap_executable="/usr/bin/true")
            )
            runtime._server = SimpleNamespace(  # type: ignore[assignment]
                server_address=("127.0.0.1", 12345)
            )
            try:
                lease = runtime.prepare_session(
                    ["claude", "--session-id", "session-1", "-p", "hello"],
                    workspace,
                    {"HOME": str(Path.home()), "PATH": os.environ.get("PATH", "")},
                )
                lease.close()
                private_root = root / ".atrex-supervisor-runtime"
                roots = list(private_root.glob("*/workspaces/*/provider-sessions/session-1"))
                self.assertEqual(len(roots), 1)
                self.assertFalse((workspace / ".atrex_long_horizon").exists())
            finally:
                runtime._server = None
                runtime._temporary.cleanup()

    @unittest.skipUnless(
        os.environ.get("ATREX_BWRAP_INTEGRATION") == "1"
        and platform.system() == "Linux"
        and shutil.which("bwrap") is not None,
        "requires explicit Linux bwrap integration opt-in",
    )
    def test_real_bwrap_auxiliary_session_inputs_and_outputs(self) -> None:
        from orchestrator.agent_workspace import WORKSPACE_LAYOUTS, WORKSPACE_ROLE_ENV

        for role in WORKSPACE_LAYOUTS.keys() - {"optimizer"}:
            with self.subTest(role=role), tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory).resolve() / "stage"
                workspace.mkdir()
                files, trees, outputs = WORKSPACE_LAYOUTS[role]
                inputs = [*files, *(f"{tree}/kernel.py" for tree in trees)]
                for name in inputs:
                    path = workspace / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("trusted input")
                # Existing drafts must remain writable on a repair/resume.
                for name in outputs:
                    (workspace / name).write_text("old draft")
                (workspace / "private.txt").write_text("must stay hidden")
                code = (
                    "from pathlib import Path\n"
                    "assert not Path('private.txt').exists()\n"
                    f"for name in {inputs!r}:\n"
                    "    assert Path(name).read_text() == 'trusted input'\n"
                    "    try: Path(name).write_text('forged')\n"
                    "    except OSError: pass\n"
                    "    else: raise AssertionError('input was writable')\n"
                    f"for name in {list(outputs)!r}:\n"
                    "    Path(name).write_text('completed output')\n"
                )
                with SupervisorRuntime(self._config(agent_sandbox="bwrap")) as runtime:
                    lease = runtime.prepare_session(
                        ["python3", "-c", code], workspace,
                        {**os.environ, "ATREX_AGENT_CLI": "claude", WORKSPACE_ROLE_ENV: role},
                    )
                    try:
                        result = subprocess.run(
                            lease.command, env=lease.environment, cwd=workspace,
                            pass_fds=lease.pass_fds,
                            capture_output=True, text=True, timeout=20,
                        )
                    finally:
                        lease.close()
                    self.assertEqual(result.returncode, 0, result.stderr)
                for name in outputs:
                    self.assertEqual((workspace / name).read_text(), "completed output")
                for name in inputs:
                    self.assertEqual((workspace / name).read_text(), "trusted input")

    @unittest.skipUnless(
        os.environ.get("ATREX_BWRAP_INTEGRATION") == "1"
        and platform.system() == "Linux"
        and shutil.which("bwrap") is not None,
        "requires explicit Linux bwrap integration opt-in",
    )
    def test_real_bwrap_process_capture_home_usage_and_subagents(self) -> None:
        from orchestrator.agent_runtime.process import run_bounded
        from orchestrator.test_session_capture import assistant, lines, usage
        from long_horizon.main_adapter import normalize_stream

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve() / "worktree"
            workspace.mkdir()
            (workspace / "kernel.py").write_text("def run(x): return x\n")
            main = lines(assistant())
            child = lines(assistant("msg-child", agent="child"))
            terminal = lines({"type": "result", "usage": usage()})
            code = (
                "import os,pathlib; "
                "assert os.environ['HOME']==os.getcwd(); "
                "assert os.environ['CLAUDE_CONFIG_DIR']==os.getcwd()+'/.claude'; "
                "root=pathlib.Path.home()/'.claude/projects/p'; "
                "(root/'main/subagents').mkdir(parents=True,exist_ok=True); "
                f"(root/'main.jsonl').write_text({main!r}); "
                f"(root/'main/subagents/agent-child.jsonl').write_text({child!r}); "
                f"print({terminal!r},end='')"
            )
            with SupervisorRuntime(self._config(agent_sandbox="bwrap")) as runtime:
                with activate_supervisor_runtime(runtime):
                    stdout, stderr, status, timed_out = run_bounded(
                        ["python3", "-c", code], workspace, 20,
                        env={**os.environ, "ATREX_AGENT_CLI": "claude",
                             "ATREX_TELEMETRY_ATTEMPT_ID": "episode-1-invocation-1"},
                    )
                self.assertEqual(status, 0, stderr)
                self.assertFalse(timed_out)
                root = runtime._private_scope_for(workspace)[1]
                captures = list((root / "sessions").glob("run-*"))
                self.assertEqual(len(captures), 1)
                report = json.loads((captures[0] / "token-usage.json").read_text())
                self.assertEqual(report["total"]["total_tokens"], 38)
                self.assertEqual(report["total"]["measurement"], "exact")
                self.assertEqual(report["state"], "completed")
                conversation = (captures[0] / "conversation.jsonl").read_text()
                self.assertIn("msg-child", conversation)
                self.assertIn("msg-main", conversation)
                self.assertFalse((root / "agent-workspace/sessions").exists())
                _, total, _, _ = normalize_stream("claude", stdout)
                self.assertEqual(total.total_tokens, 38)

    @unittest.skipUnless(
        os.environ.get("ATREX_BWRAP_INTEGRATION") == "1"
        and platform.system() == "Linux"
        and shutil.which("bwrap") is not None,
        "requires explicit Linux bwrap integration opt-in",
    )
    def test_real_bwrap_hides_runtime_ledgers_while_proxy_persists_them(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            incumbent = root / "incumbent"
            incumbent.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=incumbent, check=True)
            (incumbent / "kernel.py").write_text("def run(x): return x\n")
            (incumbent / "memory").mkdir()
            (incumbent / "memory/v0.json").write_text('{"version":"v0"}')
            (incumbent / "memory/long_horizon_e0001.json").write_text('{"private":"audit"}')
            subprocess.run(["git", "add", "kernel.py", "memory"], cwd=incumbent, check=True)
            subprocess.run([
                "git", "-c", "user.name=Test", "-c", "user.email=test@example.test",
                "commit", "-qm", "seed",
            ], cwd=incumbent, check=True)
            workspace = root / "workspace"
            subprocess.run([
                "git", "worktree", "add", "-qb", "episode", str(workspace), "HEAD",
            ], cwd=incumbent, check=True)
            link_runtime(workspace, agent_skills=())
            shutil.copyfile(REPOSITORY_ROOT / "reference/CLAUDE.md", workspace / "CLAUDE.md")
            install_workspace_policy(workspace, "production", "Triton")
            source = root / "atrex-bench-source"
            (source / "scripts").mkdir(parents=True)
            (source / "scripts" / "run_eval.py").write_text("print('eval')\n")
            package = source / "src" / "atrex_bench"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("")
            (package / "utils.py").write_text("VALUE = 1\n")
            config = self._config(
                agent_sandbox="bwrap",
                agent_skills=(),
                bwrap_executable=str(shutil.which("bwrap")),
                atrex_bench_root=source,
            )
            private_pin = supervisor_campaign_root(workspace) / "framework_baseline.json"
            private_pin.parent.mkdir(parents=True, exist_ok=True)
            private_pin.write_text('{"version":"v1"}')
            with SupervisorRuntime(config) as runtime:
                lease = runtime.prepare_session(
                    [
                        "bash",
                        "-lc",
                        "set -e; python3 tools/sandbox.py --kind dev --dry-run --no-sync -- "
                        "python3 -c 'print(1)' >/dev/null; "
                        "python3 tools/sandbox.py --kind wiki-query --dry-run 'test query' "
                        ">/dev/null; "
                        "test ! -e .atrex_long_horizon/evaluations.jsonl; "
                        "test ! -e .atrex_long_horizon/runtime-requests.jsonl; "
                        "test ! -e .atrex_long_horizon/gateway-records; "
                        "test -r memory/v0.json; test ! -e memory/long_horizon_e0001.json; "
                        "if touch memory/new.json 2>/dev/null; then exit 1; fi; "
                        "if echo overwrite >>memory/v0.json 2>/dev/null; then exit 1; fi; "
                        f"test ! -e {supervisor_campaign_root(workspace) / 'promotions'}; "
                        "test ! -e atrex-bench; "
                        "test ! -e test_kernel.py; test ! -e profile_driver.py; "
                        "test ! -e reference/test_kernel.py; "
                        "test ! -e reference/atrex_bench_test_kernel.py; "
                        "test ! -e reference/profile_driver.py; "
                        "test ! -e reference; test ! -e reference-projects; "
                        "test -f skills/KernelWiki/SKILL.md; "
                        "test -f .claude/skills/KernelWiki/SKILL.md; "
                        "test ! -e skills/gen-plan; test ! -e skills/ncu-report-skill; "
                        "test ! -e tools/memory_manager.py; test ! -e tools/iteration_trace.py; "
                        "test ! -e tools/profile_nvidia.sh; "
                        "test ! -L tools; touch tools/write-test; rm tools/write-test; "
                        "printf 'print(42)\n' >tools/probe.py; python3 tools/probe.py; "
                        "cp tools/sandbox.py scratch/client.py; "
                        "printf '\n# Episode-local edit\n' >>scratch/client.py; "
                        "mv scratch/client.py tools/sandbox.py; "
                        "if touch skills/KernelWiki/write-test 2>/dev/null; then exit 1; fi; "
                        "test -r skills/gpu-measurement/SKILL.md; "
                        "test -r .claude/skills/gpu-measurement/references/requests.md; "
                        "grep -q 'skills/runtime-records/SKILL.md' CLAUDE.md; "
                        "grep -q -- '--kind record-read' skills/runtime-records/references/records.md; "
                        "test -r skills/runtime-records/SKILL.md; "
                        "test -r .claude/skills/runtime-records/references/journal.md; "
                        "test -r .agents/skills/runtime-records/references/records.md; "
                        "if touch skills/runtime-records/write-test 2>/dev/null; then exit 1; fi; "
                        "test ! -e skills/autonomous-gpu-kernel-timeline; "
                        "if touch skills/gpu-measurement/write-test 2>/dev/null; then exit 1; fi; "
                        'test "$HOME" = "$PWD"; test "$CLAUDE_CONFIG_DIR" = "$PWD/.claude"; '
                        'printf "home-write" > "$HOME/home-test"; '
                        "test ! -e .gpu_wiki_profile; "
                        "test ! -e .git; test ! -L .git; "
                        "test ! -e .orchestrator_mode.json; test ! -e gpu-wiki; "
                        f'test -z "${{{MODE_STATE_ENV}+x}}"; '
                        "if cat .git >/dev/null 2>&1; then exit 1; fi; "
                        "if git status >/dev/null 2>&1; then exit 1; fi; "
                        f"test ! -r {incumbent / '.git/config'}; "
                        "cp kernel.py kernel.tmp; printf '# edit\n' >>kernel.tmp; "
                        "mv kernel.tmp kernel.py; "
                        "printf 'query after atomic edit' >scratch/wiki.txt; "
                        "python3 tools/sandbox.py --kind wiki-query --dry-run --file scratch/wiki.txt >/dev/null; "
                        f"test ! -e {supervisor_campaign_root(workspace) / 'wiki-profile'}; "
                        f"test ! -e {private_pin}; "
                        "test ! -e framework_baseline.json; test ! -e baseline_report.md; "
                        "test ! -e .gitignore; "
                        f"test ! -e {REPOSITORY_ROOT / 'supervisor'}; "
                        f"test ! -e {REPOSITORY_ROOT / 'orchestrator'}; "
                        f"test ! -e {REPOSITORY_ROOT / 'reference'}; "
                        f"test ! -e {REPOSITORY_ROOT / 'reference-projects'}; "
                        f"test ! -e {REPOSITORY_ROOT / '3rdparty'}; "
                        f"test ! -e {REPOSITORY_ROOT / 'tools' / 'local_gateway.py'}; "
                        f'test -z "${{{ATREX_BENCH_RUNTIME_ENV}+x}}"; '
                        f'test -z "${{{WIKI_PROFILE_ROOT_ENV}+x}}"; '
                        "echo PRIVATE_RUNTIME_OK",
                    ],
                    workspace,
                    {
                        "HOME": str(Path.home()),
                        "PATH": os.environ.get("PATH", ""),
                        "ATREX_AGENT_CLI": "claude",
                        "AGATE_AK": "must-not-enter-agent",
                    },
                )
                try:
                    process = subprocess.run(
                        lease.command,
                        cwd=workspace,
                        env=lease.environment,
                        pass_fds=lease.pass_fds,
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                finally:
                    lease.close()
                capability_view = runtime._private_scope_for(workspace)[1] / "agent-workspace"
                self.assertTrue((capability_view / "tools/probe.py").is_file())
                self.assertTrue((capability_view / "tools/sandbox.py").read_text().endswith("# Episode-local edit\n"))
                self.assertFalse((workspace / "tools/sandbox.py").read_text().endswith("# Episode-local edit\n"))
                evidence_root = runtime.evidence_root(workspace)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn("PRIVATE_RUNTIME_OK", process.stdout)
            self.assertTrue(private_pin.is_file())
            self.assertTrue((supervisor_campaign_root(workspace) / "promotions/long_horizon_e0001.json").is_file())
            self.assertEqual((workspace / "memory/v0.json").read_text(), '{"version":"v0"}')
            self.assertTrue((workspace / "memory/long_horizon_e0001.json").is_file())
            self.assertTrue((workspace / "kernel.py").read_text().endswith("# edit\n"))
            self.assertEqual(subprocess.run(
                ["git", "rev-list", "--count", "HEAD"], cwd=workspace, check=True,
                text=True, capture_output=True,
            ).stdout.strip(), "1")
            ledger = evidence_root / "runtime-requests.jsonl"
            self.assertEqual(len(ledger.read_text(encoding="utf-8").splitlines()), 3)
            profile = supervisor_campaign_root(workspace) / "wiki-profile"
            self.assertEqual(len(list(profile.glob("raw/query_events/*/*.json"))), 2)


if __name__ == "__main__":
    unittest.main()
