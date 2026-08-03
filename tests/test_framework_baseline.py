from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orchestrator import optimize

PYTORCH_V0 = "import torch\n\n\nclass Model(torch.nn.Module):\n    def forward(self, x):\n        return torch.softmax(x, -1)\n"
TRITON_KERNEL = (
    "import torch\nimport triton\nimport triton.language as tl\n\n\n"
    "@triton.jit\ndef _k(x_ptr, o_ptr, n, BLOCK: tl.constexpr):\n"
    "    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\n"
    "    tl.store(o_ptr + i, tl.load(x_ptr + i, mask=i < n), mask=i < n)\n\n\n"
    "class Model(torch.nn.Module):\n    def forward(self, x):\n"
    "        out = torch.empty_like(x)\n        _k[(1,)](x, out, x.numel(), BLOCK=128)\n        return out\n"
)
GLUON_KERNEL = TRITON_KERNEL.replace(
    "import triton.language as tl", "import triton.language as tl\nfrom triton.experimental import gluon"
)
V0_MEMORY = {
    "version": "v0",
    "masked": False,
    "performance": {
        "latency_us": 300.0,
        "latency_us_by_shape": {"0": 200.0, "1": 400.0},
    },
    "correctness": {"status": "PASS"},
    "quality_gate": {"result": "PASS"},
}


def _git(workspace: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=str(workspace), check=True, capture_output=True, text=True
    ).stdout.strip()


def _result(all_pass: bool = True, by_shape: dict | None = None) -> dict:
    return {
        "all_pass": all_pass,
        "latency_us_geomean": 250.0,
        "latency_us_arith_mean": 300.0,
        "latency_us_by_shape": {"0": 180.0, "1": 347.0} if by_shape is None else by_shape,
        "max_abs_err": 0.001,
        "max_rel_err": 0.01,
    }


def _sandbox_ok(result: dict | None = None) -> subprocess.CompletedProcess[str]:
    payload = json.dumps(result if result is not None else _result())
    return subprocess.CompletedProcess(
        args=["sandbox"], returncode=0,
        stdout=f"{optimize.TEST_RESULT_PREFIX}{payload}\n", stderr="",
    )


def _campaign(root: Path, *, framework: str = "Triton", mode: str = "production", **kwargs):
    campaign = optimize.Campaign(
        name="demo", kernel_demo=str(root / "reference.py"), platform="pro5000",
        framework=framework, optimization_mode=mode, work_dir=str(root), **kwargs,
    )
    workspace = campaign.workspace
    (workspace / "memory").mkdir(parents=True, exist_ok=True)
    (workspace / "kernel.py").write_text(PYTORCH_V0, encoding="utf-8")
    (workspace / "test_kernel.py").write_text("# immutable harness\n", encoding="utf-8")
    (workspace / "shapes.json").write_text('{"0": {}, "1": {}}\n', encoding="utf-8")
    (workspace / "solution.json").write_text('{"spec": {"dependencies": ["torch"]}}\n', encoding="utf-8")
    (workspace / "memory" / "v0.json").write_text(json.dumps(V0_MEMORY), encoding="utf-8")
    _git(workspace, "init", "-q")
    _git(workspace, "config", "user.email", "test@local")
    _git(workspace, "config", "user.name", "test")
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-q", "-m", "V0: baseline kernel")
    return campaign


class FrameworkBaselineDecisionTest(unittest.TestCase):
    def test_never_skips_without_touching_the_marker(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fb-never-") as temp_dir:
            campaign = _campaign(Path(temp_dir), framework_baseline="never")
            self.assertEqual(campaign._framework_baseline_decision(), ("skip", ""))

    def test_pytorch_v0_in_production_runs_the_stage(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fb-run-") as temp_dir:
            campaign = _campaign(Path(temp_dir))
            action, _reason = campaign._framework_baseline_decision()
            self.assertEqual(action, "run")

    def test_leaderboard_keeps_the_permissive_v0_unless_forced(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fb-leaderboard-") as temp_dir:
            root = Path(temp_dir)
            self.assertEqual(_campaign(root, mode="leaderboard")._framework_baseline_decision()[0], "skip")
        with tempfile.TemporaryDirectory(prefix="fb-forced-") as temp_dir:
            campaign = _campaign(Path(temp_dir), mode="leaderboard", framework_baseline="always")
            self.assertEqual(campaign._framework_baseline_decision()[0], "run")

    def test_compliant_v0_is_pinned_without_a_session(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fb-pin-") as temp_dir:
            campaign = _campaign(Path(temp_dir))
            (campaign.workspace / "kernel.py").write_text(TRITON_KERNEL, encoding="utf-8")
            _git(campaign.workspace, "commit", "-q", "--amend", "--no-edit", "-a")
            self.assertEqual(campaign._framework_baseline_decision()[0], "pin")

            with mock.patch.object(optimize, "run_session") as run:
                campaign.ensure_framework_baseline()
            run.assert_not_called()
            commit, version = optimize.resolve_framework_baseline_commit(campaign.workspace)
            self.assertTrue(commit)
            self.assertEqual(version, 0)

    def test_progressed_workspace_without_a_pin_is_left_alone(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fb-legacy-") as temp_dir:
            campaign = _campaign(Path(temp_dir))
            (campaign.workspace / "kernel.py").write_text(TRITON_KERNEL + "# optimized\n", encoding="utf-8")
            (campaign.workspace / "memory" / "v5.json").write_text('{"version": "v5"}', encoding="utf-8")
            _git(campaign.workspace, "add", "-A")
            _git(campaign.workspace, "commit", "-q", "-m", "v5")
            head = optimize.git_head(campaign.workspace)

            action, reason = campaign._framework_baseline_decision()
            self.assertEqual(action, "skip")
            self.assertIn("progressed beyond V0", reason)
            with mock.patch.object(optimize, "run_session") as run:
                campaign.ensure_framework_baseline()
            run.assert_not_called()
            self.assertEqual(optimize.git_head(campaign.workspace), head)


class FrameworkBaselineRunTest(unittest.TestCase):
    def test_accepted_baseline_is_committed_validated_and_pinned(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fb-accept-") as temp_dir:
            campaign = _campaign(Path(temp_dir))
            workspace = campaign.workspace

            def session(ws, prompt, **kwargs):
                (ws / "kernel.py").write_text(TRITON_KERNEL, encoding="utf-8")
                _git(ws, "commit", "-q", "-am", "v1: framework baseline")
                return optimize.SessionResult(0, False, 900, "", "", "sid")

            with (
                mock.patch.object(optimize, "run_session", side_effect=session),
                mock.patch.object(optimize, "_sandbox_command", return_value=_sandbox_ok()) as sandbox,
            ):
                campaign.ensure_framework_baseline()

            self.assertEqual(sandbox.call_count, 2)
            record = optimize.read_memory(workspace, 1)
            self.assertEqual(record["quality_gate"]["result"], "PASS")
            self.assertEqual(record["optimization"]["action_category"], "framework_baseline")
            commit, version = optimize.resolve_framework_baseline_commit(workspace)
            self.assertEqual(version, 1)
            self.assertEqual(record["git_commit_hash"], commit)
            self.assertEqual(
                _git(workspace, "rev-parse", f"{commit}:kernel.py"),
                optimize.git_kernel_blob(workspace),
            )
            self.assertFalse(optimize.head_kernel_is_initial_baseline(workspace))
            self.assertTrue(optimize.head_kernel_is_framework_baseline(workspace))
            self.assertEqual(optimize.read_stall(workspace), 0)

            # Idempotent: a second call neither runs a session nor moves HEAD.
            head = optimize.git_head(workspace)
            with mock.patch.object(optimize, "run_session") as run:
                campaign.ensure_framework_baseline()
            run.assert_not_called()
            self.assertEqual(optimize.git_head(workspace), head)

    def test_uncommitted_but_compliant_kernel_is_committed_by_the_orchestrator(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fb-uncommitted-") as temp_dir:
            campaign = _campaign(Path(temp_dir))

            def session(ws, prompt, **kwargs):
                (ws / "kernel.py").write_text(TRITON_KERNEL, encoding="utf-8")
                return optimize.SessionResult(0, False, 900, "", "", "sid")

            with (
                mock.patch.object(optimize, "run_session", side_effect=session),
                mock.patch.object(optimize, "_sandbox_command", return_value=_sandbox_ok()),
            ):
                campaign.ensure_framework_baseline()

            commit, version = optimize.resolve_framework_baseline_commit(campaign.workspace)
            self.assertEqual(version, 1)
            self.assertEqual(
                _git(campaign.workspace, "rev-parse", f"{commit}:kernel.py"),
                optimize.git_worktree_blob(campaign.workspace, "kernel.py"),
            )

    def test_interrupted_run_is_adopted_without_another_session(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fb-adopt-") as temp_dir:
            campaign = _campaign(Path(temp_dir))
            workspace = campaign.workspace
            (workspace / "kernel.py").write_text(TRITON_KERNEL, encoding="utf-8")
            (workspace / "memory" / "v1.json").write_text('{"version": "v1"}', encoding="utf-8")
            _git(workspace, "add", "-A")
            _git(workspace, "commit", "-q", "-m", "v1: framework baseline")

            self.assertEqual(campaign._framework_baseline_decision()[0], "adopt")
            with (
                mock.patch.object(optimize, "run_session") as run,
                mock.patch.object(optimize, "_sandbox_command", return_value=_sandbox_ok()) as sandbox,
            ):
                campaign.ensure_framework_baseline()
            run.assert_not_called()
            self.assertEqual(sandbox.call_count, 2)
            self.assertEqual(optimize.resolve_framework_baseline_commit(workspace)[1], 1)


class FrameworkBaselineRejectionTest(unittest.TestCase):
    def _run_expecting_failure(self, campaign, session, sandbox_result=None):
        head = optimize.git_head(campaign.workspace)
        sandbox = mock.patch.object(
            optimize, "_sandbox_command",
            return_value=_sandbox_ok(sandbox_result) if sandbox_result is not None else _sandbox_ok(),
        )
        with (
            mock.patch.object(optimize, "run_session", side_effect=session) as run,
            sandbox as sandbox_mock,
        ):
            with self.assertRaises(RuntimeError) as caught:
                campaign.ensure_framework_baseline()
        return head, run, sandbox_mock, str(caught.exception)

    def test_still_violating_candidate_gets_one_recovery_then_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fb-violating-") as temp_dir:
            campaign = _campaign(Path(temp_dir))

            def session(ws, prompt, **kwargs):
                (ws / "kernel.py").write_text(PYTORCH_V0 + "# tweaked\n", encoding="utf-8")
                return optimize.SessionResult(0, False, 900, "", "", "sid")

            head, run, sandbox, message = self._run_expecting_failure(campaign, session)
            self.assertEqual(run.call_count, 2)  # one attempt + one recovery
            sandbox.assert_not_called()
            self.assertIn("not a self-contained Triton implementation", message)
            self.assertEqual(optimize.git_head(campaign.workspace), head)
            record = optimize.read_memory(campaign.workspace, 1)
            self.assertEqual(record["quality_gate"]["result"], "FAIL")
            self.assertEqual(record["pitfalls_and_fixes"][-1]["error_type"], "production_policy")

    def test_mutated_ground_truth_is_restored_instead_of_failing(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fb-mutated-") as temp_dir:
            campaign = _campaign(Path(temp_dir))
            harness = campaign.workspace / "test_kernel.py"
            original = harness.read_text(encoding="utf-8")

            def session(ws, prompt, **kwargs):
                (ws / "kernel.py").write_text(TRITON_KERNEL, encoding="utf-8")
                harness.write_text("# tampered harness\n", encoding="utf-8")
                return optimize.SessionResult(0, False, 900, "", "", "sid")

            with (
                mock.patch.object(optimize, "run_session", side_effect=session) as run,
                mock.patch.object(optimize, "_sandbox_command", return_value=_sandbox_ok()),
            ):
                campaign.ensure_framework_baseline()

            self.assertEqual(run.call_count, 1)  # no recovery session was needed
            self.assertEqual(harness.read_text(encoding="utf-8"), original)
            self.assertEqual(optimize.resolve_framework_baseline_commit(campaign.workspace)[1], 1)

    def test_gluon_baseline_is_rejected_for_a_triton_campaign(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fb-gluon-") as temp_dir:
            campaign = _campaign(Path(temp_dir))

            def session(ws, prompt, **kwargs):
                (ws / "kernel.py").write_text(GLUON_KERNEL, encoding="utf-8")
                return optimize.SessionResult(0, False, 900, "", "", "sid")

            _head, _run, _sandbox, message = self._run_expecting_failure(campaign, session)
            self.assertIn("Gluon", message)

    def test_shape_coverage_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fb-shapes-") as temp_dir:
            campaign = _campaign(Path(temp_dir))

            def session(ws, prompt, **kwargs):
                (ws / "kernel.py").write_text(TRITON_KERNEL, encoding="utf-8")
                return optimize.SessionResult(0, False, 900, "", "", "sid")

            _head, _run, sandbox, message = self._run_expecting_failure(
                campaign, session, sandbox_result=_result(by_shape={"0": 180.0})
            )
            self.assertEqual(sandbox.call_count, 2)
            self.assertIn("latency_us_by_shape", message)


class FrameworkBaselineMarkerTest(unittest.TestCase):
    def test_absent_marker_resolves_to_nothing(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fb-marker-none-") as temp_dir:
            campaign = _campaign(Path(temp_dir))
            self.assertEqual(optimize.resolve_framework_baseline_commit(campaign.workspace), ("", 0))

    def test_blob_mismatch_and_missing_commit_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fb-marker-bad-") as temp_dir:
            campaign = _campaign(Path(temp_dir))
            workspace = campaign.workspace
            head = optimize.git_head(workspace)

            (workspace / optimize.FRAMEWORK_BASELINE_FILE).write_text(json.dumps({
                "version": "v1", "commit": head, "kernel_blob": "0" * 40,
            }), encoding="utf-8")
            _git(workspace, "add", "-A")
            _git(workspace, "commit", "-q", "-m", "bad blob")
            with self.assertRaisesRegex(RuntimeError, "kernel blob does not match"):
                optimize.resolve_framework_baseline_commit(workspace)

            (workspace / optimize.FRAMEWORK_BASELINE_FILE).write_text(json.dumps({
                "version": "v1", "commit": "a" * 40, "kernel_blob": "0" * 40,
            }), encoding="utf-8")
            _git(workspace, "commit", "-q", "-am", "dangling commit")
            with self.assertRaisesRegex(RuntimeError, "missing commit"):
                optimize.resolve_framework_baseline_commit(workspace)

    def test_incumbent_latency_ignores_versions_before_the_pin(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fb-incumbent-") as temp_dir:
            campaign = _campaign(Path(temp_dir))
            workspace = campaign.workspace
            (workspace / "memory" / "v1.json").write_text(json.dumps({
                "version": "v1",
                "performance": {"latency_us": 900.0},
                "quality_gate": {"result": "PASS"},
            }), encoding="utf-8")
            self.assertEqual(optimize.best_validated_latency_us(workspace), 300.0)

            (workspace / "kernel.py").write_text(TRITON_KERNEL, encoding="utf-8")
            _git(workspace, "commit", "-q", "-am", "v1 kernel")
            campaign._pin_framework_baseline(optimize.git_head(workspace), version=1)
            self.assertEqual(optimize.best_validated_latency_us(workspace), 900.0)


class BucketSeedingInheritanceTest(unittest.TestCase):
    def _aggregate_with_pin(self, root: Path):
        aggregate = _campaign(root)
        workspace = aggregate.workspace
        (workspace / "kernel.py").write_text(TRITON_KERNEL, encoding="utf-8")
        (workspace / "memory" / "v1.json").write_text(json.dumps({
            "version": "v1",
            "performance": {"latency_us": 250.0, "latency_us_by_shape": {"0": 180.0, "1": 347.0}},
            "correctness": {"status": "PASS"},
            "quality_gate": {"result": "PASS"},
            "optimization": {"action_category": "framework_baseline"},
            "framework_baseline": {"framework": "Triton"},
        }), encoding="utf-8")
        _git(workspace, "add", "-A")
        _git(workspace, "commit", "-q", "-m", "v1: framework baseline")
        kernel_commit = optimize.git_head(workspace)
        aggregate._pin_framework_baseline(kernel_commit, version=1)
        # A later aggregated dispatcher must never be what buckets inherit.
        (workspace / "kernel.py").write_text("# aggregated dispatcher\n", encoding="utf-8")
        _git(workspace, "commit", "-q", "-am", "v2: aggregate dispatcher")
        return aggregate, kernel_commit

    def _bucket(self, root: Path, aggregate):
        op_dir = root / "bucket_op"
        op_dir.mkdir(exist_ok=True)
        (op_dir / "reference.py").write_text("# reference\n", encoding="utf-8")
        (op_dir / "shapes.json").write_text('{"1": {}}\n', encoding="utf-8")
        campaign = optimize.Campaign(
            name="bucket_b", kernel_demo=str(op_dir / "reference.py"), platform="pro5000",
            framework="Triton", optimization_mode=aggregate.optimization_mode,
            work_dir=str(root / "runs"), framework_baseline="never",
        )
        coordinator = optimize.WorkloadBucketCoordinator(aggregate_campaign=aggregate, op_dir=op_dir)
        bucket = optimize.WorkloadBucket(name="b", workload_indices=(1,))
        source = optimize.WorkloadSource(
            kind="shapes", filename="shapes.json", ids=("0", "1"), entries=({}, {}),
        )
        return coordinator, bucket, campaign, source

    def test_bucket_inherits_the_pinned_kernel_not_the_dispatcher_head(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fb-seed-") as temp_dir:
            root = Path(temp_dir)
            aggregate, kernel_commit = self._aggregate_with_pin(root)
            coordinator, bucket, campaign, source = self._bucket(root, aggregate)
            coordinator._seed_bucket_baseline_from_aggregate(bucket, campaign, source)

            self.assertEqual((campaign.workspace / "kernel.py").read_text(encoding="utf-8"), TRITON_KERNEL)
            record = optimize.read_memory(campaign.workspace, 0)
            self.assertEqual(record["baseline_derivation"]["aggregate_baseline_commit"], kernel_commit)
            self.assertEqual(record["baseline_derivation"]["aggregate_baseline_version"], "v1")
            self.assertEqual(record["baseline_derivation"]["source"], "aggregate_framework_baseline")
            self.assertEqual(set(record["performance"]["latency_us_by_shape"]), {"1"})
            self.assertEqual(record["performance"]["latency_us"], 347.0)
            self.assertNotIn("framework_baseline", record)

    def test_campaign_without_a_pin_keeps_deriving_from_the_root_commit(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fb-seed-legacy-") as temp_dir:
            root = Path(temp_dir)
            aggregate = _campaign(root)
            coordinator, bucket, campaign, source = self._bucket(root, aggregate)
            coordinator._seed_bucket_baseline_from_aggregate(bucket, campaign, source)

            self.assertEqual((campaign.workspace / "kernel.py").read_text(encoding="utf-8"), PYTORCH_V0)
            record = optimize.read_memory(campaign.workspace, 0)
            self.assertEqual(record["baseline_derivation"]["source"], "aggregate_v0")
            self.assertEqual(record["baseline_derivation"]["aggregate_baseline_version"], "v0")

    def test_bucket_seeded_from_another_baseline_is_refused_in_production(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fb-seed-mixed-") as temp_dir:
            root = Path(temp_dir)
            aggregate = _campaign(root)
            coordinator, bucket, campaign, source = self._bucket(root, aggregate)
            coordinator._seed_bucket_baseline_from_aggregate(bucket, campaign, source)

            workspace = aggregate.workspace
            (workspace / "kernel.py").write_text(TRITON_KERNEL, encoding="utf-8")
            _git(workspace, "commit", "-q", "-am", "v1: framework baseline")
            aggregate._pin_framework_baseline(optimize.git_head(workspace), version=1)

            with self.assertRaisesRegex(RuntimeError, "--framework-baseline never"):
                coordinator._seed_bucket_baseline_from_aggregate(bucket, campaign, source)


class FrameworkBaselineWiringTest(unittest.TestCase):
    def test_coordinator_runs_the_stage_between_setup_and_bucketing(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fb-wiring-") as temp_dir:
            root = Path(temp_dir)
            aggregate = _campaign(root)
            op_dir = root / "op"
            op_dir.mkdir()
            (op_dir / "shapes.json").write_text('{"0": {}, "1": {}}\n', encoding="utf-8")
            coordinator = optimize.WorkloadBucketCoordinator(
                aggregate_campaign=aggregate, op_dir=op_dir
            )
            calls: list[str] = []
            with (
                mock.patch.object(
                    optimize.Campaign, "ensure_framework_baseline",
                    side_effect=lambda *_a, **_k: calls.append("framework-baseline"),
                ),
                mock.patch.object(
                    coordinator, "inspect_workloads",
                    side_effect=lambda *_a, **_k: calls.append("inspect") or [],
                ),
            ):
                coordinator._ensure_main_workspace()
                coordinator.inspect_workloads()

            self.assertEqual(calls, ["framework-baseline", "inspect"])

    def test_main_threads_the_new_options_and_rejects_a_bad_timeout(self) -> None:
        op = {"name": "demo", "reference": "/tmp/op/reference.py", "roofline_py": "", "op_dir": "/tmp/op"}
        with tempfile.TemporaryDirectory(prefix="fb-cli-") as temp_dir:
            argv = [
                "--op-dir", "/tmp/op", "--platform", "H20", "--arch", "sm_90",
                "--framework", "Triton", "--optimization-mode", "production",
                "--framework-baseline", "always", "--framework-baseline-timeout", "600",
                "--workspace", temp_dir,
            ]
            with (
                mock.patch.object(optimize.shutil, "which", return_value="/bin/claude"),
                mock.patch.object(optimize, "_resolve_op", return_value=op),
                mock.patch.object(optimize, "ensure_submodules"),
                mock.patch.object(optimize, "Campaign", return_value=mock.Mock()) as campaign,
            ):
                self.assertEqual(optimize.main(argv), 0)
            self.assertEqual(campaign.call_args.kwargs["framework_baseline"], "always")
            self.assertEqual(campaign.call_args.kwargs["framework_baseline_timeout"], 600)

            with self.assertRaises(SystemExit):
                optimize.main([*argv[:-2], "--framework-baseline-timeout", "0",
                               "--workspace", temp_dir])

    def test_bucket_campaigns_never_run_the_stage(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fb-bucket-flag-") as temp_dir:
            root = Path(temp_dir)
            aggregate = _campaign(root)
            _coordinator, _bucket, campaign, _source = BucketSeedingInheritanceTest()._bucket(
                root, aggregate
            )
            self.assertEqual(campaign.framework_baseline, "never")


class FrameworkBaselinePromptTest(unittest.TestCase):
    def test_prompt_renders_without_placeholders_and_forbids_plan_and_profile(self) -> None:
        rendered = optimize._render(
            optimize.PROMPTS_DIR / "framework_baseline.md",
            WORKSPACE="/tmp/ws", N=1, PREV=0, PLATFORM="pro5000", FRAMEWORK="CuteDSL",
            ARCH="sm_120", NOTES="none", AGENT_RUNTIME="- runtime", HARDWARE="## hw",
            SANDBOX="## sandbox", EVALUATOR="## evaluator",
            MODE_POLICY="## Optimization mode: production",
        )
        self.assertNotIn("{{", rendered)
        self.assertTrue(rendered.startswith("## Optimization mode: production"))
        self.assertIn("--multi-seed 5", rendered)
        self.assertIn("memory/v1.json", rendered)
        self.assertNotIn("gen-plan", rendered)
        self.assertNotIn("PLAN_GENERATOR", rendered)
        self.assertIn("Do NOT profile", rendered)


if __name__ == "__main__":
    unittest.main()
