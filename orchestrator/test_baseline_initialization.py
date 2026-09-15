"""V0 initialization must not fall back to a Setup Agent."""

from __future__ import annotations

import io
import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from long_horizon.campaign import LongHorizonCampaign
from long_horizon.git_episode import EpisodeWorktree, ignored_evidence_files
from long_horizon.main_adapter import prepare_campaign

from .campaign import Campaign
from .constants import REPO_ROOT
from .operator_layout import AGENT_PROBLEM_FILENAME, validate_operator_layout
from .optimize import _resolve_op, _run_main
from .workspace_runtime import _agent_runtime_directive, link_runtime
from .workspace_state import (
    git_head, latest_version, read_memory, resolve_framework_baseline_commit, v0_baseline_commit,
)


class BaselineInitializationTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.bench = self.root / "bench"
        self.operator = self.bench / "data" / "identity"
        self.operator.mkdir(parents=True)
        (self.bench / "scripts").mkdir()
        (self.bench / "scripts" / "run_eval.py").write_text("# canonical evaluator\n")
        (self.bench / "src" / "atrex_bench").mkdir(parents=True)
        (self.operator / "reference.py").write_text("class Model: pass\n")
        (self.operator / "input.py").write_text("def _make_inputs(**kwargs): return {}\n")
        (self.operator / "shapes.json").write_text(
            json.dumps({"0": {"init_kwargs": {}, "input_kwargs": {"n": 4}}})
        )

    def campaign(self, **kwargs) -> Campaign:
        return Campaign(
            name="identity",
            kernel_demo=str(self.operator / "reference.py"),
            platform="L20N",
            framework="Triton",
            work_dir=str(self.root),
            **kwargs,
        )

    def make_sol(self) -> None:
        (self.operator / "reference.py").write_text("def run(x): return x\n")
        (self.operator / "definition.json").write_text(
            json.dumps({"name": "identity", "inputs": {"x": {}}, "outputs": {"out": {}}})
        )
        (self.operator / "workload.jsonl").write_text("{}\n")
        (self.operator / "input.py").unlink()
        (self.operator / "shapes.json").unlink()

    def test_native_resolution(self) -> None:
        self.assertEqual(validate_operator_layout(self.operator), self.bench)
        self.assertEqual(_resolve_op(str(self.operator))["atrex_bench_root"], str(self.bench))

    def test_sol_resolution_needs_no_native_evaluator(self) -> None:
        self.make_sol()
        (self.bench / "scripts" / "run_eval.py").unlink()
        self.assertIsNone(validate_operator_layout(self.operator))
        self.assertEqual(_resolve_op(str(self.operator))["atrex_bench_root"], "")

    def test_incomplete_native_operator_is_rejected(self) -> None:
        for name in ("reference.py", "input.py", "shapes.json"):
            with self.subTest(missing=name):
                path = self.operator / name
                content = path.read_bytes()
                path.unlink()
                try:
                    with self.assertRaisesRegex(SystemExit, name.replace(".", r"\.")):
                        _resolve_op(str(self.operator))
                finally:
                    path.write_bytes(content)

    def test_malformed_shapes_are_rejected(self) -> None:
        (self.operator / "shapes.json").write_text('{"0": {"input_kwargs": []}}')
        with self.assertRaisesRegex(SystemExit, "init_kwargs/input_kwargs"):
            _resolve_op(str(self.operator))

    def test_native_operator_requires_canonical_evaluator(self) -> None:
        (self.bench / "scripts" / "run_eval.py").unlink()
        with self.assertRaisesRegex(SystemExit, "canonical scripts/run_eval.py"):
            _resolve_op(str(self.operator))

    def test_invalid_inputs_fail_before_workspace_or_session_creation(self) -> None:
        (self.operator / "input.py").unlink()
        campaign = self.campaign()
        with (
            patch("orchestrator.campaign.run_session") as session,
            patch("orchestrator.campaign.subprocess.run") as run,
            self.assertRaisesRegex(ValueError, "missing input.py"),
        ):
            campaign.setup_baseline()
        run.assert_not_called()
        session.assert_not_called()
        self.assertFalse(campaign.workspace.exists())

    def _initialize_v0(self, campaign: Campaign) -> None:
        result = {
            "all_pass": True,
            "latency_us_by_shape": {"0": 12.0},
            "latency_us_geomean": 12.0,
            "latency_us_arith_mean": 12.0,
        }
        with (
            patch("orchestrator.campaign.run_session") as session,
            patch.object(campaign, "_framework_baseline_correctness_reviewers", return_value=()),
            patch(
                "orchestrator.campaign._sandbox_command",
                return_value=subprocess.CompletedProcess(
                    [], 0, "[test_kernel] RESULT_JSON=" + json.dumps(result), ""
                ),
            ) as evaluate,
            redirect_stdout(io.StringIO()),
        ):
            campaign.setup_baseline()
        session.assert_not_called()
        evaluate.assert_called_once()
        self.assertEqual(latest_version(campaign.workspace), 0)
        memory = read_memory(campaign.workspace, 0)
        self.assertEqual(memory["correctness"]["status"], "PASS")
        self.assertNotIn("git_commit_hash", memory)
        self.assertNotEqual(v0_baseline_commit(campaign.workspace), git_head(campaign.workspace))
        for name in ("README.md", "memory/v0.json"):
            self.assertTrue((campaign.workspace / name).is_file(), name)
        self.assertTrue((campaign.workspace / "scratch").is_dir())
        for name in (
            "plans", "profiles", "baseline_report.md", "framework_baseline.json", ".gitignore",
        ):
            self.assertFalse((campaign.workspace / name).exists(), name)

    def test_episode_prompts_use_journals_instead_of_plan_or_profile_files(self) -> None:
        for backend in ("claude", "codex", "qodercli", "pi"):
            campaign = self.campaign(agent_cli=backend, atrex_bench_root=str(self.bench))
            runner = LongHorizonCampaign(campaign)
            worktree = EpisodeWorktree(1, "a" * 40, "test-episode", self.root / "episode")
            with self.subTest(backend=backend):
                prompt = runner._prompt(
                    episode=1, version=2, worktree=worktree,
                    conversion_pending=False, resumed=True,
                )
                for obsolete in (
                    "plans/", "profiles/", "PLAN_GENERATOR", "{{", "candidate_commit",
                    "git add", "git commit", "best_commit", "policy_review_request",
                    "gpu-kernel-episode-loop", "iteration_trace.py",
                    "phase-start", "phase-end", "source-read",
                ):
                    self.assertNotIn(obsolete, prompt)
                self.assertIn("Git is Supervisor-only", prompt)
                self.assertIn("scratch/", prompt)
                self.assertIn("record-experiment", prompt)
                self.assertIn("update-direction", prompt)
                self.assertIn("--kind episode-report", prompt)
                compact = " ".join(prompt.lower().split())
                self.assertTrue("not required" in compact or "no plan-generator" in compact)

    def episode_prompt(self, *, platform="L20N", arch="sm_120", skills=(), **options) -> str:
        campaign = self.campaign(atrex_bench_root=str(self.bench), agent_skills=skills)
        campaign.platform, campaign.arch = platform, arch
        parameters = {
            "episode": 1, "version": 2,
            "worktree": EpisodeWorktree(1, "a" * 40, "test-episode", self.root / "episode"),
            "conversion_pending": False,
        }
        parameters.update(options)
        return LongHorizonCampaign(campaign)._prompt(**parameters)

    def test_normal_prompt_omits_irrelevant_conversion_and_ppu_guidance(self) -> None:
        prompt = self.episode_prompt()
        self.assertNotIn("Gluon", prompt)
        self.assertNotIn("accepted_ppu_diagnostics", prompt)
        self.assertNotIn("{{", prompt)
        self.assertEqual(prompt.count("## Execution boundary"), 1)
        self.assertEqual(prompt.count("Git is Supervisor-only"), 1)
        self.assertEqual(prompt.count('--kind wiki-query "'), 1)
        self.assertIn("Target product L20N, runtime architecture sm_120", prompt)
        self.assertIn("operator identity", prompt)
        self.assertIn("take no command after `--`", prompt)
        self.assertIn("--kind dev --input <path> ... -- <command>", prompt)
        self.assertNotIn("tools/sandbox.py ... --`", prompt)
        for text in (
            "at most three Directions", "only one `in_progress`",
            "Close every in-progress Direction", "Correct rejected reports",
            "stop after acceptance", "passing Evaluate for the exact source",
            "skills/runtime-records/SKILL.md", "skills/gpu-measurement/SKILL.md",
            "Do not install packages", "canonical", "private cases",
        ):
            self.assertIn(text, prompt)

    def test_full_conversion_replaces_general_wiki_query_and_uses_policy_tolerance(self) -> None:
        with patch("long_horizon.main_adapter.CONVERT_PERF_TOL", 0.07):
            prompt = self.episode_prompt(conversion_pending=True, arch="sm_90")
        self.assertIn("## Required Triton-to-Gluon conversion", prompt)
        self.assertIn("Extract TTGIR before writing Gluon", prompt)
        self.assertIn("within 7%", prompt)
        self.assertIn("runtime architecture sm_90", prompt)
        self.assertEqual(prompt.count('--kind wiki-query "'), 1)
        self.assertNotIn("techniques and pitfalls for operator", prompt)
        self.assertNotIn("{{", prompt)

    def test_full_workflow_is_guidance_without_extra_plan_files_or_phase_leakage(self) -> None:
        headings = (
            "Understand the starting point", "Choose a Direction", "Research the question",
            "Implement and validate", "Record and decide",
        )
        for conversion in (False, True):
            with self.subTest(conversion=conversion):
                prompt = self.episode_prompt(conversion_pending=conversion)
                loop = prompt.split("## Engineering loop\n", 1)[1].split(
                    "## Terminal contract\n", 1,
                )[0]
                for ordinal, heading in enumerate(headings, start=1):
                    self.assertIn(f"{ordinal}. **{heading}.**", loop)
                for expected in (
                    "not additional mandatory stages", "Register it before exploration",
                    "Direction Journal rather than creating separate draft or plan files",
                    "specific\n   uncertainty", "one coherent candidate at a time",
                    "reusing a matching saved result", "separating measured facts\n   from analysis",
                    "Prepare the report as work proceeds",
                ):
                    self.assertIn(expected, loop)
                for path in ("scratch/draft.md", "scratch/plan.md"):
                    self.assertNotIn(path, prompt)
        campaign = self.campaign(atrex_bench_root=str(self.bench))
        with patch.object(campaign, "_framework_baseline_correctness_reviewers", return_value=()):
            self.assertNotIn("## Engineering loop", campaign._framework_baseline_prompt(1))

    def test_ppu_diagnostics_require_matching_hardware_and_selected_skill(self) -> None:
        for platform in ("L20N", "PPU-ZW810E"):
            for selected in (False, True):
                with self.subTest(platform=platform, selected=selected):
                    prompt = self.episode_prompt(
                        platform=platform, arch="sm_89",
                        skills=("ppu-acu-joint-profile",) if selected else (),
                    )
                    self.assertEqual(
                        "accepted_ppu_diagnostics" in prompt,
                        platform == "PPU-ZW810E" and selected,
                    )
                    self.assertNotIn("{{", prompt)

    def test_episode_prompt_preserves_fresh_and_recovery_workspace_semantics(self) -> None:
        fresh = self.episode_prompt()
        resumed = self.episode_prompt(resumed=True)
        self.assertIn("`scratch/` starts empty", fresh)
        self.assertNotIn("`scratch/` starts empty", resumed)
        self.assertIn("Keep and reuse the existing workspace, scratch files", resumed)
        self.assertIn("repair and resubmit its terminal report", resumed)

    def test_baseline_prompt_uses_shared_boundary_and_a_distinct_finish(self) -> None:
        for mode in ("leaderboard", "production"):
            with self.subTest(mode=mode):
                campaign = self.campaign(
                    atrex_bench_root=str(self.bench), optimization_mode=mode,
                )
                with (
                    patch.object(
                        campaign, "_framework_baseline_correctness_reviewers", return_value=(),
                    ),
                    patch.object(
                        campaign, "_framework_baseline_smoke_shape_ids", return_value=["0", "2"],
                    ),
                ):
                    prompt = campaign._framework_baseline_prompt(1)
                    command, _ = campaign._framework_baseline_smoke_command(1)
                self.assertIn(command, prompt)
                self.assertEqual(prompt.count(command), 1)
                self.assertIn("--shape-id 0 --shape-id 2 --timed-runs 1", prompt)
                self.assertEqual(prompt.count("## Execution boundary"), 1)
                self.assertEqual(prompt.count("Git is Supervisor-only"), 1)
                self.assertEqual(prompt.count('--kind wiki-query "'), 1)
                for text in (
                    "take no command after `--`", "Do not install packages",
                    "no PyTorch compute fallback", "prebuilt operators", "alternate-DSL compute",
                ):
                    self.assertIn(text, prompt)
                self.assertIn("Do not submit `episode-report`", prompt)
                self.assertIn("v1: framework candidate smoke-passed (Triton)", prompt)
                self.assertIn("Smoke success is not baseline acceptance", prompt)
                self.assertIn("reusing matching\nRuntime measurements", prompt)
                self.assertIn("if they are mounted and\nreadable", prompt)
                for obsolete in (
                    "{{", "--no-memory", "tools/sandbox.py ... --`", "policy agent",
                    "Codex and\nQoder", "mixed/alternate implementations",
                    "Mandatory external", "plan file", "profile-analysis files",
                ):
                    self.assertNotIn(obsolete, prompt)

    def test_sol_baseline_prescribed_smoke_may_cover_the_whole_workload(self) -> None:
        self.make_sol()
        campaign = self.campaign()
        with patch.object(campaign, "_framework_baseline_correctness_reviewers", return_value=()):
            prompt = campaign._framework_baseline_prompt(1)
        self.assertIn("Evaluation route: SOL-ExecBench", prompt)
        self.assertIn("may already use the whole workload", prompt)
        self.assertNotIn("Do not run a full-workload evaluator", prompt)
        self.assertNotIn("--shape-id", campaign._framework_baseline_smoke_command(1)[0])
        self.assertNotIn("{{", prompt)

    def test_baseline_guidance_is_advisory_and_references_require_mounts(self) -> None:
        campaign = self.campaign(atrex_bench_root=str(self.bench))
        guidance = {
            "reviews": {
                "codex": {"status": "ok", "guidance": "Preserve the output dtype."},
                "qodercli": {"status": "failed", "reason": "CLI unavailable"},
            },
            "selected_references": [
                {
                    "path": "reference-projects/example/launch.py",
                    "purpose": "launch ABI", "votes": 1,
                },
            ],
        }
        with (
            patch.object(
                campaign, "_framework_baseline_correctness_reviewers",
                return_value=("codex", "qodercli"),
            ),
            patch.object(
                campaign, "_load_framework_baseline_correctness_guidance", return_value=guidance,
            ),
        ):
            prompt = campaign._framework_baseline_prompt(1)
        for expected in (
            "Preserve the output dtype", "Reviewer unavailable: CLI unavailable",
            "reference-projects/example/launch.py", "only if mounted and readable",
            "suggestions never override the public contract",
        ):
            self.assertIn(expected, prompt)
        self.assertNotIn("Shared requirements are mandatory", prompt)
        self.assertNotIn("paged addressing, causal", prompt)
        self.assertNotIn("{{", prompt)

    def test_episode_finish_distinguishes_evaluate_from_supervisor_abba(self) -> None:
        full = self.episode_prompt()
        self.assertIn("reuses a matching Runtime", full)
        self.assertIn("ordinary Evaluate does not\nreplace ABBA", full)
        self.assertIn("Report acceptance is not Kernel promotion", full)
        prompt = self.episode_prompt()
        self.assertIn("passing Evaluate for the exact source", prompt)
        self.assertIn("Profile or ABBA alone is insufficient", prompt)

    def test_general_constraints_do_not_override_phase_specific_policy(self) -> None:
        text = (REPO_ROOT / "reference/CLAUDE.md").read_text()
        for rule in (
            "current arguments", "no answer caching", "pointer-identity caching",
            "default stream", "Tolerances are safety margins", "Opaque Shape IDs",
            "Runtime tools", "error.next_action", "retry until accepted",
        ):
            self.assertIn(rule, text)
        for obsolete in (
            "recommended optimization direction", "both are acceptable", "mandatory from V1",
            "--multi-seed 5", "600 seconds", "every `run()` invocation receives freshly",
        ):
            self.assertNotIn(obsolete, text)

    def test_scratch_evidence_is_ignored_and_archived(self) -> None:
        campaign = self.campaign()
        self._initialize_v0(campaign)
        worktree = EpisodeWorktree.create(
            campaign.workspace, 1, git_head(campaign.workspace), root=self.root / "episodes"
        )
        diagnostic = worktree.path / "scratch/probe.py"
        diagnostic.write_text("# optional diagnostic\n")
        self.assertEqual(ignored_evidence_files(worktree.path), ["scratch/probe.py"])
        archive = worktree.archive(self.root / "archive")
        self.assertEqual(
            (archive / "worktree_files/scratch/probe.py").read_text(), diagnostic.read_text()
        )

    def test_native_v0_is_materialized_measured_and_committed_without_setup_session(self) -> None:
        campaign = self.campaign()
        self._initialize_v0(campaign)
        self.assertEqual(campaign.atrex_bench_root, str(self.bench))
        self.assertEqual(
            (campaign.workspace / "kernel.py").read_bytes(),
            (self.operator / "reference.py").read_bytes(),
        )

    def test_sol_v0_still_uses_supervisor_measurement(self) -> None:
        self.make_sol()
        self._initialize_v0(self.campaign())

    def test_framework_baseline_keeps_commit_in_private_pin_not_report(self) -> None:
        campaign = self.campaign()
        self._initialize_v0(campaign)
        (campaign.workspace / "kernel.py").write_text("# framework baseline\nclass Model: pass\n")
        with redirect_stdout(io.StringIO()):
            commit = campaign._commit_framework_baseline(1, {
                "all_pass": True, "latency_us_geomean": 8.0,
                "latency_us_by_shape": {"0": 8.0},
            })
        memory = read_memory(campaign.workspace, 1)
        self.assertNotIn("git_commit_hash", memory)
        self.assertEqual(memory["correctness"]["status"], "PASS")
        self.assertEqual(resolve_framework_baseline_commit(campaign.workspace), (commit, 1))

    def test_failed_framework_baseline_report_omits_commit(self) -> None:
        campaign = self.campaign()
        memory = campaign.workspace / "memory/v1.json"
        memory.parent.mkdir(parents=True)
        memory.write_text(json.dumps({"git_commit_hash": "a" * 40}))
        campaign._record_framework_baseline_failure("incorrect output")
        report = json.loads(memory.read_text())
        self.assertNotIn("git_commit_hash", report)
        self.assertEqual(report["quality_gate"]["result"], "FAIL")

    def test_contract_authoring_retains_its_own_timeout(self) -> None:
        campaign = self.campaign(
            optimization_mode="production",
            atrex_bench_root=str(self.bench),
            problem_generation_timeout=123,
        )
        campaign.workspace.mkdir()

        def author(staging, prompt, **kwargs):
            self.assertEqual(kwargs["timeout"], 123)
            self.assertIn("agent_problem.json", prompt)
            (staging / AGENT_PROBLEM_FILENAME).write_text('{"objective": "identity"}')
            return SimpleNamespace(tokens=1, exit_status=0, timed_out=False)

        with (
            patch("orchestrator.campaign.run_session", side_effect=author) as session,
            patch("orchestrator.campaign.validate_generated_agent_problem"),
            redirect_stdout(io.StringIO()),
        ):
            campaign._ensure_agent_problem()
        session.assert_called_once()
        self.assertTrue((campaign.workspace / AGENT_PROBLEM_FILENAME).is_file())
        self.assertTrue(campaign._generated_agent_problem_digest)

    def test_framework_baseline_still_follows_v0(self) -> None:
        campaign = self.campaign()
        calls = Mock()
        with (
            patch("long_horizon.main_adapter.latest_version", return_value=-1),
            patch.object(campaign, "setup_baseline", calls.seed),
            patch.object(campaign, "ensure_framework_baseline", calls.framework),
            patch("long_horizon.main_adapter.reconstruct_stall", return_value=0),
            patch("long_horizon.main_adapter.write_stall"),
        ):
            prepare_campaign(campaign)
        self.assertEqual([call[0] for call in calls.mock_calls], ["seed", "framework"])

    def test_setup_assets_and_skill_discovery_are_removed(self) -> None:
        for name in (
            "orchestrator/prompts/setup.md",
            "skills/gpu-kernel-baseline/SKILL.md",
            "agents/gpu-kernel-baseline.md",
        ):
            self.assertFalse((REPO_ROOT / name).exists(), name)
        self.assertTrue((REPO_ROOT / "orchestrator/prompts/framework_baseline.md").is_file())
        workspace = self.root / "workspace"
        workspace.mkdir()
        link_runtime(workspace)
        for backend in ("claude", "qodercli", "codex", "pi"):
            self.assertNotIn("gpu-kernel-baseline", _agent_runtime_directive(backend))
        self.assertFalse((workspace / ".agents/skills/gpu-kernel-baseline").exists())
        self.assertFalse((workspace / ".claude/agents/gpu-kernel-baseline.md").exists())

    def test_cli_documents_contract_timeout_not_setup_timeout(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as stopped:
            _run_main(["--help"])
        self.assertEqual(stopped.exception.code, 0)
        self.assertIn("--problem-generation-timeout", output.getvalue())
        self.assertNotIn("--setup-timeout", output.getvalue())
        self.assertIn("--framework-baseline-timeout", output.getvalue())


if __name__ == "__main__":
    unittest.main()
