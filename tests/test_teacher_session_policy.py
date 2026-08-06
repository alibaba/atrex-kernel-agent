from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orchestrator import agent_runtime
from orchestrator import optimize
from orchestrator.agent_runtime.model import AgentRunRequest
from orchestrator.agent_runtime.process import (
    ACCESS_POLICY_ENV,
    ProcessAccessPolicy,
    dependency_process_violation,
    register_access_policy,
    resolve_access_policy,
    run_bounded,
    unregister_access_policy,
)
from teacher_distill.session_policy import TeacherSessionPolicy


class TeacherRuntimeHydrationTest(unittest.TestCase):
    def test_hydration_links_only_sanitized_knowledge_and_omits_reference_projects(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-runtime-") as temp_dir:
            root = Path(temp_dir) / "repo"
            workspace = Path(temp_dir) / "workspace"
            sanitized = Path(temp_dir) / "sanitized"
            private = Path(temp_dir) / "private"
            teacher = private / "teacher"
            workspace.mkdir()
            sanitized.mkdir()
            teacher.mkdir(parents=True)
            private.mkdir(exist_ok=True)
            (workspace / ".gitignore").write_text("", encoding="utf-8")
            for common in ("tools", "reference", "reference-projects", "gpu-wiki", "agents"):
                (root / common).mkdir(parents=True, exist_ok=True)
            project_skill = root / "skills" / "gpu-kernel-profile-optimizer"
            project_skill.mkdir(parents=True)
            (project_skill / "SKILL.md").write_text("# optimizer\n", encoding="utf-8")
            ncu = root / "3rdparty" / "ncu-report-skill"
            ncu.mkdir(parents=True)
            (ncu / "SKILL.md").write_text("# ncu\n", encoding="utf-8")
            leaked_kernel_wiki = root / "gpu-wiki" / "3rdparty" / "KernelWiki"
            leaked_kernel_wiki.mkdir(parents=True)
            (leaked_kernel_wiki / "SKILL.md").write_text("# must not link\n", encoding="utf-8")

            policy = TeacherSessionPolicy(
                knowledge_view=sanitized,
                teacher_solution=teacher,
                private_root=private,
                source_wiki=root / "gpu-wiki",
                reference_projects=root / "reference-projects",
            )
            with (
                mock.patch.object(optimize, "REPO_ROOT", root),
                mock.patch.object(optimize, "HUMANIZE_DIR", root / "missing-humanize"),
            ):
                policy.link_runtime(workspace)

            self.assertTrue((workspace / "gpu-wiki").is_symlink())
            self.assertEqual((workspace / "gpu-wiki").resolve(), sanitized.resolve())
            self.assertFalse((workspace / "reference-projects").exists())
            for runtime_skills in (
                workspace / ".claude" / "skills",
                workspace / ".qoder" / "skills",
                workspace / ".agents" / "skills",
            ):
                self.assertFalse((runtime_skills / "KernelWiki").exists())
                self.assertTrue((runtime_skills / "ncu-report-skill").exists())

    def test_search_directive_exposes_policy_but_no_private_identity(self) -> None:
        policy = TeacherSessionPolicy(
            knowledge_view=Path("/sanitized/view"),
            teacher_solution=Path("/private/flashinfer/teacher"),
            private_root=Path("/private/supervisor"),
            source_wiki=Path("/repo/gpu-wiki"),
            reference_projects=Path("/repo/reference-projects"),
        )
        directive = policy.knowledge_directive()

        self.assertIn("sanitized", directive.lower())
        self.assertIn("public web", directive.lower())
        self.assertIn("reference-projects", directive)
        self.assertNotIn("flashinfer", directive.lower())
        self.assertNotIn("/private/", directive)
        self.assertNotIn("/repo/", directive)

    def test_prompt_filter_removes_inherited_external_search_instructions(self) -> None:
        policy = TeacherSessionPolicy(
            knowledge_view=Path("/sanitized/view"),
            teacher_solution=Path("/private/teacher"),
            private_root=Path("/private/supervisor"),
            source_wiki=Path("/repo/gpu-wiki"),
            reference_projects=Path("/repo/reference-projects"),
        )
        prompt = (
            "Use gpu-wiki first.\n"
            "- L2: inspect reference-projects/ when needed.\n"
            "- L3: use public web and web search.\n"
            "- Search KernelWiki under gpu-wiki/3rdparty/.\n"
            "Then implement one change.\n"
        )

        filtered = policy.filter_prompt(prompt)

        self.assertIn("Use gpu-wiki first", filtered)
        self.assertIn("Then implement one change", filtered)
        self.assertNotIn("reference-projects", filtered.casefold())
        self.assertNotIn("public web", filtered.casefold())
        self.assertNotIn("web search", filtered.casefold())
        self.assertNotIn("kernelwiki", filtered.casefold())
        self.assertNotIn("gpu-wiki/3rdparty", filtered.casefold())


class ProcessAccessPolicyTest(unittest.TestCase):
    def _policy(self, audit_log: Path | None = None) -> ProcessAccessPolicy:
        return ProcessAccessPolicy(
            forbidden_roots=(Path("/private/teacher"), Path("/repo/gpu-wiki")),
            network_disabled=True,
            audit_log=audit_log,
            label="teacher-hidden-audited",
        )

    def test_forbidden_paths_and_network_commands_are_classified(self) -> None:
        policy = self._policy()
        cases = (
            (["cat", "/private/teacher/kernel.py"], "forbidden path", None),
            (["find", "/private"], "forbidden path", None),
            (["cat", "../teacher/kernel.py"], "forbidden path", Path("/private/workspace")),
            (["bash", "-c", "rg gdn /repo/gpu-wiki"], "forbidden path", None),
            (["curl", "https://example.com/source.py"], "network access", None),
            (["git", "clone", "https://example.com/repo"], "network access", None),
            (["git", "push", "origin", "HEAD"], "network access", None),
            (["python", "-c", "import requests; requests.get('https://example.com')"], "network access", None),
            (["python", "-c", "import socket; socket.create_connection(('example.com', 80))"], "network access", None),
        )
        for command, expected, cwd in cases:
            with self.subTest(command=command):
                reason = dependency_process_violation(command, access_policy=policy, cwd=cwd)
                self.assertIsNotNone(reason)
                self.assertIn("teacher knowledge access policy violation", reason)
                self.assertIn(expected, reason)
                self.assertNotIn("/private/", reason)
                self.assertNotIn("/repo/", reason)

    def test_local_git_and_sanitized_workspace_commands_remain_allowed(self) -> None:
        policy = self._policy()
        for command in (
            ["git", "status", "--short"],
            ["git", "commit", "-m", "v2"],
            ["rg", "cp.async", "gpu-wiki"],
            ["python", "tools/memory_manager.py", "summary", "--workspace", "."],
        ):
            with self.subTest(command=command):
                self.assertIsNone(
                    dependency_process_violation(command, access_policy=policy)
                )

    def test_policy_paths_are_hidden_behind_an_opaque_runtime_id_for_every_backend(self) -> None:
        private_path = "/private/teacher-do-not-expose"
        for backend in agent_runtime.SUPPORTED_RUNTIME_IDS:
            captured: dict[str, object] = {}

            def fake_run(command, cwd, timeout, env=None):
                captured.update(command=command, cwd=cwd, timeout=timeout, env=dict(env or {}))
                return '{"type":"result","usage":{"input_tokens":1,"output_tokens":1}}\n', "", 0, False

            runtime = agent_runtime.build_agent_runtime(backend, process_runner=fake_run)
            with tempfile.TemporaryDirectory(prefix="teacher-policy-runtime-") as temp_dir:
                result = runtime.run(
                    AgentRunRequest(
                        workspace=Path(temp_dir),
                        prompt="test",
                        timeout_s=10,
                        access_policy=ProcessAccessPolicy(
                            forbidden_roots=(Path(private_path),),
                            network_disabled=True,
                            label="teacher-hidden-audited",
                        ),
                    )
                )
            environment = captured["env"]
            command = captured["command"]
            self.assertIsInstance(environment, dict)
            self.assertIsInstance(command, list)
            if backend == "claude":
                self.assertIn("--disallowedTools", command)
                self.assertIn("WebSearch,WebFetch", command)
            if backend == "pi":
                self.assertIn("--tools", command)
                self.assertIn("read,bash,edit,write", command)
            rendered = json.dumps(environment, sort_keys=True)
            self.assertNotIn(private_path, rendered)
            policy_id = environment.get("ATREX_ACCESS_POLICY_ID")
            self.assertRegex(str(policy_id), r"^[0-9a-f]{32}$")
            self.assertIsNone(resolve_access_policy(str(policy_id)))
            self.assertEqual(result.exit_status, 0)

    def test_bounded_runner_terminates_and_audits_forbidden_descendant(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-policy-audit-") as temp_dir:
            root = Path(temp_dir)
            forbidden = root / "private" / "teacher.py"
            forbidden.parent.mkdir()
            forbidden.write_text("teacher source\n", encoding="utf-8")
            audit = root / "audit" / "access-violations.jsonl"
            policy = ProcessAccessPolicy(
                forbidden_roots=(forbidden.parent,),
                network_disabled=True,
                audit_log=audit,
                label="teacher-hidden-audited",
            )
            policy_id = register_access_policy(policy)
            environment = os.environ.copy()
            environment[ACCESS_POLICY_ENV] = policy_id
            try:
                _stdout, stderr, returncode, _timed_out = run_bounded(
                    [
                        sys.executable,
                        "-c",
                        "import subprocess; subprocess.run(['bash','-c',"
                        + repr("cat private/teacher.py; sleep 5")
                        + "])",
                    ],
                    cwd=root,
                    timeout=10,
                    env=environment,
                )
            finally:
                unregister_access_policy(policy_id)

            self.assertEqual(returncode, 126)
            self.assertIn("teacher knowledge access policy violation", stderr)
            records = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(records[-1]["policy"], "teacher-hidden-audited")
            self.assertIn("forbidden path", records[-1]["reason"])

    def test_shell_guard_violation_marks_parent_session_failed_and_audited(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-shell-marker-") as temp_dir:
            root = Path(temp_dir)
            audit = root / "audit.jsonl"
            policy = ProcessAccessPolicy(
                network_disabled=True,
                audit_log=audit,
                label="teacher-hidden-audited",
            )
            policy_id = register_access_policy(policy)
            environment = os.environ.copy()
            environment[ACCESS_POLICY_ENV] = policy_id
            environment["BASH_ENV"] = str(
                Path(__file__).resolve().parents[1]
                / "tools"
                / "session_shell_guard.sh"
            )
            try:
                _stdout, stderr, returncode, _timed_out = run_bounded(
                    [
                        sys.executable,
                        "-c",
                        "import subprocess; "
                        "subprocess.run(['bash','-c','curl --version']); "
                        "print('agent continued')",
                    ],
                    cwd=root,
                    timeout=10,
                    env=environment,
                )
            finally:
                unregister_access_policy(policy_id)
            records = [json.loads(line) for line in audit.read_text().splitlines()]

        self.assertEqual(returncode, 126)
        self.assertIn("network access", stderr)
        self.assertIn("network access", records[-1]["reason"])

    def test_shell_guard_blocks_network_tools_but_keeps_local_git(self) -> None:
        environment = os.environ.copy()
        environment[ACCESS_POLICY_ENV] = "opaque-policy-id"
        environment["ATREX_SESSION_WORKSPACE"] = str(Path.cwd())
        environment["BASH_ENV"] = str(
            Path(__file__).resolve().parents[1] / "tools" / "session_shell_guard.sh"
        )
        blocked = subprocess.run(
            ["bash", "-c", "curl --version"],
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        )
        blocked_path = subprocess.run(
            ["bash", "-c", "cat ../outside-workspace"],
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        )
        blocked_git = subprocess.run(
            ["bash", "-c", "git push origin HEAD"],
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        )
        allowed = subprocess.run(
            ["bash", "-c", "git --version"],
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        )

        self.assertEqual(blocked.returncode, 126)
        self.assertIn("network access", blocked.stderr)
        self.assertEqual(blocked_path.returncode, 126)
        self.assertIn("outside the workspace", blocked_path.stderr)
        self.assertEqual(blocked_git.returncode, 126)
        self.assertIn("network access", blocked_git.stderr)
        self.assertEqual(allowed.returncode, 0, allowed.stderr)

    def test_session_policy_builds_private_parent_side_access_policy(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-policy-private-") as temp_dir:
            root = Path(temp_dir)
            session = TeacherSessionPolicy(
                knowledge_view=root / "view",
                teacher_solution=root / "teacher",
                private_root=root / "private",
                source_wiki=root / "source-wiki",
                reference_projects=root / "reference-projects",
            )
            access = session.process_access_policy()

        self.assertTrue(access.network_disabled)
        self.assertEqual(access.label, "teacher-hidden-audited")
        self.assertEqual(len(access.forbidden_roots), 4)
        self.assertEqual(access.audit_log.name, "access-violations.jsonl")


if __name__ == "__main__":
    unittest.main()
