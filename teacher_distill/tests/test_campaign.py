from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orchestrator import optimize
from teacher_distill.benchmark import (
    MaterializedTeacherWorkspace,
    TeacherBenchmarkResult,
)
from teacher_distill.campaign import TeacherDistillCampaign
from teacher_distill.cli import TeacherDistillRequest
from teacher_distill.knowledge_view import KnowledgeView
from teacher_distill.models import CampaignTerminalStatus


class FakeCandidate:
    run_reason = "success: teacher ABBA passed (candidate/teacher 1.030)"
    emit_violation = False
    instances: list["FakeCandidate"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.workspace = Path(kwargs["work_dir"]) / (
            "kernel_opt_%s_%s" % (kwargs["name"], kwargs["workspace_suffix"])
        )
        self.stop_policy = kwargs.get("stop_policy")
        self.setup_calls = 0
        self.baseline_calls = 0
        self.run_calls = 0
        FakeCandidate.instances.append(self)

    @staticmethod
    def _git(workspace: Path, *args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=workspace, check=True, capture_output=True, text=True
        ).stdout.strip()

    def setup_baseline(self) -> None:
        self.setup_calls += 1
        self.workspace.mkdir(parents=True, exist_ok=True)
        (self.workspace / "memory").mkdir(exist_ok=True)
        (self.workspace / "kernel.py").write_text(
            "import torch\ndef run(x, out): out[:] = torch.softmax(x, -1)\n",
            encoding="utf-8",
        )
        (self.workspace / "README.md").write_text("# V0\n", encoding="utf-8")
        (self.workspace / "memory/v0.json").write_text(
            json.dumps(
                {
                    "version": "v0",
                    "performance": {"latency_us": 300.0},
                    "correctness": {"status": "PASS"},
                    "quality_gate": {"result": "PASS"},
                }
            ),
            encoding="utf-8",
        )
        self._git(self.workspace, "init", "-q")
        self._git(self.workspace, "config", "user.email", "test@local")
        self._git(self.workspace, "config", "user.name", "test")
        self._git(self.workspace, "add", "-A")
        self._git(self.workspace, "commit", "-q", "-m", "v0")

    def ensure_framework_baseline(self) -> None:
        self.baseline_calls += 1
        marker = self.workspace / optimize.FRAMEWORK_BASELINE_FILE
        if marker.is_file():
            return
        (self.workspace / "kernel.py").write_text(
            "import torch\nimport cutlass\nimport cutlass.cute as cute\n"
            "@cute.kernel\ndef _kernel(x, out): return\n"
            "def run(x, out): _kernel(x, out)\n",
            encoding="utf-8",
        )
        (self.workspace / "memory/v1.json").write_text(
            json.dumps(
                {
                    "version": "v1",
                    "performance": {
                        "latency_us": 180.0,
                        "latency_us_by_shape": {"shape-a": 150.0, "shape-b": 216.0},
                    },
                    "correctness": {"status": "PASS"},
                    "quality_gate": {"result": "PASS"},
                }
            ),
            encoding="utf-8",
        )
        self._git(self.workspace, "add", "kernel.py", "memory/v1.json")
        self._git(self.workspace, "commit", "-q", "-m", "v1")
        kernel_commit = self._git(self.workspace, "rev-parse", "HEAD")
        kernel_blob = self._git(self.workspace, "rev-parse", "%s:kernel.py" % kernel_commit)
        marker.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "version": "v1",
                    "framework": "CuteDSL",
                    "platform": "H20",
                    "arch": "sm90",
                    "commit": kernel_commit,
                    "kernel_blob": kernel_blob,
                }
            ),
            encoding="utf-8",
        )
        self._git(self.workspace, "add", marker.name)
        self._git(self.workspace, "commit", "-q", "-m", "pin v1")

    def run(self) -> str:
        self.run_calls += 1
        if self.emit_violation:
            audit = self.kwargs["session_access_policy"].audit_log
            audit.parent.mkdir(parents=True, exist_ok=True)
            audit.write_text('{"reason":"forbidden"}\n', encoding="utf-8")
        return self.run_reason


class TeacherCampaignTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeCandidate.instances.clear()
        FakeCandidate.run_reason = "success: teacher ABBA passed (candidate/teacher 1.030)"
        FakeCandidate.emit_violation = False

    def _bundle(self, root: Path) -> Path:
        bundle = root / "teacher"
        (bundle / "helpers").mkdir(parents=True)
        (bundle / "kernel.py").write_text(
            "import torch\nimport cutlass\nimport cutlass.cute as cute\n"
            "@cute.kernel\ndef _kernel(x, out): return\n"
            "def run(x, out): _kernel(x, out)\n",
            encoding="utf-8",
        )
        (bundle / "helpers/layout.py").write_text("TILE=128\n", encoding="utf-8")
        (bundle / "solution.json").write_text(
            json.dumps(
                {
                    "spec": {
                        "languages": ["pytorch", "cutedsl"],
                        "entry_point": "kernel.py::run",
                        "dependencies": ["torch", "nvidia-cutlass-dsl"],
                    },
                    "sources": [
                        {"path": "kernel.py"},
                        {"path": "helpers/layout.py"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        (bundle / "provenance.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "operator": {
                        "canonical_id": "gdn_decode",
                        "aliases": ["gdn", "gated_delta_rule"],
                    },
                    "source": {
                        "project": "flashinfer",
                        "revision": "abc123",
                        "license": "Apache-2.0",
                    },
                    "target": {"framework": "CuteDSL", "architecture": "sm90"},
                    "knowledge_deny": {
                        "sources": ["flashinfer"],
                        "paths": [],
                        "tags": ["gdn", "gdn_decode"],
                    },
                }
            ),
            encoding="utf-8",
        )
        return bundle

    def _op(self, root: Path) -> Path:
        op = root / "op"
        op.mkdir()
        (op / "definition.json").write_text(
            json.dumps({"name": "demo", "inputs": {"x": {}}, "outputs": {"out": {}}}),
            encoding="utf-8",
        )
        (op / "reference.py").write_text("def run(x): return x\n", encoding="utf-8")
        (op / "workload.jsonl").write_text(
            json.dumps({"uuid": "shape-a"}) + "\n" + json.dumps({"uuid": "shape-b"}) + "\n",
            encoding="utf-8",
        )
        return op

    def _request(self, root: Path, *, shape_ratio: float = 1.10) -> TeacherDistillRequest:
        op = self._op(root)
        teacher = self._bundle(root)
        return TeacherDistillRequest(
            campaign_mode="teacher-distill",
            name="gdn",
            op_dir=op,
            kernel_demo=op / "reference.py",
            atrex_bench_root=None,
            platform="H20",
            architecture="sm90",
            framework="CuteDSL",
            teacher_solution=teacher,
            private_root=root / "private",
            workspace_root=root / "runs",
            geomean_ratio=1.05,
            shape_ratio=shape_ratio,
            stall_before_episode=3,
            partial_restarts=1,
            optimization_mode="production",
            framework_baseline="always",
            convert_after=0,
            workload_bucketing=False,
            layer=False,
            notes="none",
            max_iters=10,
            token_budget=0,
            max_stall=5,
            iter_timeout=10,
            setup_timeout=10,
            salvage_timeout=0,
            framework_baseline_timeout=10,
            sandbox_hardware="local",
            sandbox_profile="",
            sandbox_url="http://127.0.0.1:8000",
            sandbox_timeout=60,
            agent_cli="pi",
        )

    def _patch_dependencies(self, root: Path):
        view_root = root / "sanitized-view"
        view_root.mkdir(exist_ok=True)

        def view(*_args, **_kwargs):
            return KnowledgeView(view_root, "a" * 64, 10, 2)

        def materialize(_bundle, _op, destination, **_kwargs):
            workspace = Path(destination)
            workspace.mkdir(parents=True, exist_ok=True)
            (workspace / "kernel.py").write_text("# teacher\n", encoding="utf-8")
            (workspace / "solution.json").write_text('{"sources":[{"path":"kernel.py"}]}', encoding="utf-8")
            return MaterializedTeacherWorkspace(
                workspace=workspace,
                kind="sol",
                expected_shape_keys=("shape-a", "shape-b"),
                workload_hash="b" * 64,
                evaluator_hash="c" * 64,
                measurement_config_hash="d" * 64,
            )

        benchmark = TeacherBenchmarkResult(
            geomean_latency_us=100.0,
            latency_us_by_shape={"shape-a": 80.0, "shape-b": 125.0},
            workload_hash="b" * 64,
            evaluator_hash="c" * 64,
            measurement_config_hash="d" * 64,
        )
        return (
            mock.patch("teacher_distill.campaign.build_knowledge_view", side_effect=view),
            mock.patch("teacher_distill.campaign.materialize_teacher_workspace", side_effect=materialize),
            mock.patch("teacher_distill.campaign.benchmark_teacher", return_value=benchmark),
            mock.patch("teacher_distill.campaign.optimize.Campaign", side_effect=FakeCandidate),
            mock.patch("teacher_distill.campaign.generate_distillation"),
        )

    def test_fresh_campaign_locks_public_target_builds_v1_and_returns_success(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-campaign-") as temp_dir:
            root = Path(temp_dir)
            request = self._request(root)
            patches = self._patch_dependencies(root)
            with patches[0], patches[1], patches[2] as benchmark, patches[3], patches[4]:
                result = TeacherDistillCampaign(request).run()

            candidate = FakeCandidate.instances[-1]
            public_target = (candidate.workspace / "teacher_target.json").read_text(encoding="utf-8")
            public_lock = (candidate.workspace / "campaign_lock.json").read_text(encoding="utf-8")
            private_state_path = next(
                request.private_root.glob("campaign_*/private_config.json")
            )
            private_state = private_state_path.read_text(encoding="utf-8")
            incumbent = json.loads(
                (private_state_path.parent / "candidate_incumbent.json").read_text(
                    encoding="utf-8"
                )
            )
            rejection = candidate.kwargs["iteration_acceptance"](
                candidate,
                2,
                {
                    "performance": {"latency_us": incumbent["latency_us"] + 1.0},
                    "correctness": {"status": "PASS"},
                    "quality_gate": {"result": "PASS"},
                },
                None,
            )

        self.assertEqual(result.status, CampaignTerminalStatus.SUCCESS)
        self.assertEqual(candidate.setup_calls, 1)
        self.assertGreaterEqual(candidate.baseline_calls, 1)
        self.assertEqual(candidate.run_calls, 1)
        benchmark.assert_called_once()
        self.assertNotIn("flashinfer", public_target + public_lock)
        self.assertNotIn(str(request.teacher_solution), public_target + public_lock)
        self.assertIn("flashinfer", private_state)
        self.assertIsNotNone(candidate.kwargs["stop_policy"])
        self.assertIsNotNone(candidate.kwargs["session_access_policy"])
        self.assertIsNotNone(candidate.kwargs["session_prompt_filter"])
        self.assertTrue(candidate.kwargs["evaluate_initial_stop"])
        self.assertEqual(candidate.kwargs["stop_policy_infra_retries"], 1)
        self.assertTrue(candidate.kwargs["abort_on_stop_policy_infra"])
        self.assertIsNotNone(candidate.kwargs["iteration_acceptance"])
        self.assertIsNotNone(candidate.kwargs["iteration_change_detector"])
        self.assertIn("did not improve", rejection)
        self.assertIsNotNone(candidate.kwargs["on_iteration"])
        self.assertIn("hidden-audited", candidate.kwargs["session_directive"])

    def test_successful_resume_revalidates_hashes_without_rerunning_campaign(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-campaign-resume-") as temp_dir:
            root = Path(temp_dir)
            request = self._request(root)
            patches = self._patch_dependencies(root)
            with patches[0], patches[1], patches[2] as benchmark, patches[3], patches[4]:
                first = TeacherDistillCampaign(request).run()
                second = TeacherDistillCampaign(request).run()

        self.assertEqual(first.status, CampaignTerminalStatus.SUCCESS)
        self.assertEqual(second.status, CampaignTerminalStatus.SUCCESS)
        benchmark.assert_called_once()
        self.assertEqual(len(FakeCandidate.instances), 2)
        self.assertEqual(FakeCandidate.instances[0].setup_calls, 1)
        self.assertEqual(FakeCandidate.instances[0].run_calls, 1)
        self.assertEqual(FakeCandidate.instances[1].setup_calls, 0)
        self.assertEqual(FakeCandidate.instances[1].run_calls, 0)

    def test_teacher_provenance_must_match_selected_operator(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-operator-mismatch-") as temp_dir:
            root = Path(temp_dir)
            request = self._request(root)
            provenance_path = request.teacher_solution / "provenance.json"
            provenance = json.loads(provenance_path.read_text())
            provenance["operator"] = {
                "canonical_id": "different_operator",
                "aliases": ["other"],
            }
            provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

            result = TeacherDistillCampaign(request).run()

        self.assertEqual(result.status, CampaignTerminalStatus.INFRA_ERROR)
        self.assertIn("operator does not match", result.reason)

    def test_setup_failure_returns_and_persists_infra_terminal_result(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-campaign-setup-failure-") as temp_dir:
            root = Path(temp_dir)
            request = self._request(root)
            patches = self._patch_dependencies(root)
            with (
                patches[0],
                patches[1],
                mock.patch(
                    "teacher_distill.campaign.benchmark_teacher",
                    side_effect=RuntimeError("gateway unavailable"),
                ),
                patches[3],
                patches[4],
            ):
                result = TeacherDistillCampaign(request).run()
            result_files = list(
                request.private_root.glob("setup_failures/*/result.json")
            )

        self.assertEqual(result.status, CampaignTerminalStatus.INFRA_ERROR)
        self.assertIn("gateway unavailable", result.reason)
        self.assertEqual(len(result_files), 1)

    def test_candidate_setup_failure_persists_campaign_infra_result(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-candidate-setup-failure-") as temp_dir:
            root = Path(temp_dir)
            request = self._request(root)
            patches = self._patch_dependencies(root)
            with (
                patches[0],
                patches[1],
                patches[2],
                mock.patch(
                    "teacher_distill.campaign.optimize.Campaign",
                    side_effect=RuntimeError("verifier setup failed"),
                ),
                patches[4],
            ):
                result = TeacherDistillCampaign(request).run()
            result_files = list(request.private_root.glob("campaign_*/result.json"))

        self.assertEqual(result.status, CampaignTerminalStatus.INFRA_ERROR)
        self.assertIn("verifier setup failed", result.reason)
        self.assertEqual(len(result_files), 1)

    def test_plateau_or_budget_terminal_result_does_not_restart_agents_on_resume(self) -> None:
        for terminal_reason, expected in (
            ("stall: 5 iterations with no commit", CampaignTerminalStatus.PLATEAU),
            ("budget: max-iters", CampaignTerminalStatus.BUDGET_EXHAUSTED),
        ):
            with self.subTest(terminal_reason=terminal_reason):
                with tempfile.TemporaryDirectory(prefix="teacher-terminal-resume-") as temp_dir:
                    root = Path(temp_dir)
                    request = self._request(root)
                    patches = self._patch_dependencies(root)
                    FakeCandidate.run_reason = terminal_reason
                    first_index = len(FakeCandidate.instances)
                    with (
                        patches[0],
                        patches[1],
                        patches[2],
                        patches[3],
                        patches[4],
                        mock.patch(
                            "teacher_distill.campaign.TeacherEscalationManager.continue_after_stall",
                            return_value=terminal_reason,
                        ),
                    ):
                        first = TeacherDistillCampaign(request).run()
                        second = TeacherDistillCampaign(request).run()

                self.assertEqual(first.status, expected)
                self.assertEqual(second.status, expected)
                self.assertEqual(FakeCandidate.instances[first_index].run_calls, 1)
                self.assertEqual(FakeCandidate.instances[first_index + 1].run_calls, 0)

    def test_incomplete_private_preparation_is_rebuilt_without_touching_candidate(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-campaign-rebuild-") as temp_dir:
            root = Path(temp_dir)
            request = self._request(root)
            patches = self._patch_dependencies(root)
            with (
                patches[0],
                patches[1],
                mock.patch(
                    "teacher_distill.campaign.benchmark_teacher",
                    side_effect=[RuntimeError("gateway unavailable"), TeacherBenchmarkResult(
                        geomean_latency_us=100.0,
                        latency_us_by_shape={"shape-a": 80.0, "shape-b": 125.0},
                        workload_hash="b" * 64,
                        evaluator_hash="c" * 64,
                        measurement_config_hash="d" * 64,
                    )],
                ),
                patches[3],
                patches[4],
            ):
                with self.assertRaisesRegex(RuntimeError, "gateway unavailable"):
                    TeacherDistillCampaign(request)._load_or_prepare()
                prepared = TeacherDistillCampaign(request)._load_or_prepare()
                private_state_exists = (prepared.private_dir / "private_config.json").is_file()
                candidate_exists = (
                    request.workspace_root / "kernel_opt_gdn_cutedsl_h20_production"
                ).exists()

        self.assertTrue(private_state_exists)
        self.assertFalse(candidate_exists)

    def test_private_preparation_can_resume_before_public_lock_exists(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-campaign-private-resume-") as temp_dir:
            root = Path(temp_dir)
            request = self._request(root)
            patches = self._patch_dependencies(root)
            with patches[0], patches[1], patches[2] as benchmark, patches[3], patches[4]:
                prepared = TeacherDistillCampaign(request)._load_or_prepare()
                self.assertFalse((request.workspace_root / "kernel_opt_gdn_cutedsl_h20_production/campaign_lock.json").exists())
                result = TeacherDistillCampaign(request).run()

        self.assertEqual(result.status, CampaignTerminalStatus.SUCCESS)
        self.assertTrue(prepared.private_dir.is_absolute())
        benchmark.assert_called_once()

    def test_resume_rejects_materialized_teacher_workspace_tampering(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-workspace-tamper-") as temp_dir:
            root = Path(temp_dir)
            request = self._request(root)
            patches = self._patch_dependencies(root)
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                TeacherDistillCampaign(request).run()
                teacher_workspace = next(
                    request.private_root.glob("campaign_*/teacher_workspace")
                )
                (teacher_workspace / "kernel.py").write_text(
                    "# tampered teacher\n", encoding="utf-8"
                )
                result = TeacherDistillCampaign(request).run()

        self.assertEqual(result.status, CampaignTerminalStatus.INFRA_ERROR)
        self.assertIn("materialized Teacher workspace", result.reason)

    def test_changed_threshold_is_rejected_as_resume_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-campaign-mismatch-") as temp_dir:
            root = Path(temp_dir)
            request = self._request(root)
            patches = self._patch_dependencies(root)
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                TeacherDistillCampaign(request).run()
                changed = TeacherDistillRequest(
                    **{**request.__dict__, "shape_ratio": 1.20}
                )
                result = TeacherDistillCampaign(changed).run()
                self.assertEqual(result.status, CampaignTerminalStatus.INFRA_ERROR)
                self.assertIn("RESUME_CONFIG_MISMATCH", result.reason)

    def test_resume_rejects_hardware_or_escalation_drift(self) -> None:
        for override in (
            {"sandbox_hardware": "other-hardware"},
            {"partial_restarts": 0},
            {"stall_before_episode": 4},
            {"agent_cli": "claude"},
        ):
            with self.subTest(override=override):
                with tempfile.TemporaryDirectory(prefix="teacher-campaign-lock-drift-") as temp_dir:
                    root = Path(temp_dir)
                    request = self._request(root)
                    patches = self._patch_dependencies(root)
                    with patches[0], patches[1], patches[2], patches[3], patches[4]:
                        TeacherDistillCampaign(request).run()
                        changed = TeacherDistillRequest(
                            **{**request.__dict__, **override}
                        )
                        result = TeacherDistillCampaign(changed).run()
                        self.assertEqual(
                            result.status, CampaignTerminalStatus.INFRA_ERROR
                        )
                        self.assertIn("RESUME_CONFIG_MISMATCH", result.reason)

    def test_stall_enters_one_escalation_manager_before_terminal_mapping(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-campaign-escalation-") as temp_dir:
            root = Path(temp_dir)
            request = self._request(root)
            patches = self._patch_dependencies(root)
            FakeCandidate.run_reason = "stall: 3 iterations with no commit"
            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
                mock.patch(
                    "teacher_distill.campaign.TeacherEscalationManager.continue_after_stall",
                    return_value="budget: max-iters",
                ) as escalate,
            ):
                result = TeacherDistillCampaign(request).run()

        self.assertEqual(result.status, CampaignTerminalStatus.BUDGET_EXHAUSTED)
        escalate.assert_called_once()
        self.assertEqual(FakeCandidate.instances[-1].kwargs["max_stall"], 3)

    def test_audited_forbidden_access_overrides_an_apparent_success(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-campaign-leakage-") as temp_dir:
            root = Path(temp_dir)
            request = self._request(root)
            patches = self._patch_dependencies(root)
            FakeCandidate.emit_violation = True
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                result = TeacherDistillCampaign(request).run()

        self.assertEqual(result.status, CampaignTerminalStatus.TEACHER_LEAKAGE_VIOLATION)

    def test_native_candidate_v0_is_materialized_without_an_agent_rewrite(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-campaign-native-v0-") as temp_dir:
            root = Path(temp_dir)
            request = self._request(root)
            native = root / "native"
            native.mkdir()
            reference = (
                "import torch\nclass Model(torch.nn.Module):\n"
                "    def forward(self, x): return torch.softmax(x, -1)\n"
            )
            (native / "reference.py").write_text(reference, encoding="utf-8")
            (native / "input.py").write_text("def make_inputs(): return ()\n", encoding="utf-8")
            (native / "shapes.json").write_text('{"0": {}, "1": {}}\n', encoding="utf-8")
            (native / "metadata.json").write_text('{}\n', encoding="utf-8")
            atrex = root / "atrex"
            (atrex / "scripts").mkdir(parents=True)
            (atrex / "scripts/run_eval.py").write_text("# eval\n", encoding="utf-8")
            (atrex / "src/atrex_bench").mkdir(parents=True)
            native_request = TeacherDistillRequest(
                **{
                    **request.__dict__,
                    "op_dir": native,
                    "kernel_demo": native / "reference.py",
                    "atrex_bench_root": atrex,
                }
            )
            supervisor = TeacherDistillCampaign(native_request)
            candidate = optimize.Campaign(
                name=native_request.name,
                kernel_demo=str(native_request.kernel_demo),
                platform="H20",
                framework="CuteDSL",
                arch="sm90",
                work_dir=str(native_request.workspace_root),
                workspace_suffix=optimize.framework_workspace_suffix(
                    "CuteDSL", "H20", "production"
                ),
                optimization_mode="production",
                runtime_linker=lambda _campaign: None,
                atrex_bench_root=str(atrex),
            )
            result = {
                "all_pass": True,
                "latency_us_geomean": 200.0,
                "latency_us_arith_mean": 210.0,
                "latency_us_by_shape": {"0": 190.0, "1": 221.0},
                "max_abs_err": 0.0,
                "max_rel_err": 0.0,
            }
            completed = subprocess.CompletedProcess(
                args=["sandbox"],
                returncode=0,
                stdout=optimize.TEST_RESULT_PREFIX + json.dumps(result) + "\n",
                stderr="",
            )
            with (
                mock.patch.object(optimize, "_sandbox_command", return_value=completed) as sandbox,
                mock.patch.object(optimize, "run_session") as agent,
            ):
                supervisor._setup_native_reference_v0(candidate)

            self.assertEqual(sandbox.call_count, 2)
            agent.assert_not_called()
            self.assertEqual(optimize.latest_version(candidate.workspace), 0)
            self.assertEqual((candidate.workspace / "kernel.py").read_text(encoding="utf-8"), reference)
            self.assertEqual(
                json.loads((candidate.workspace / "memory/v0.json").read_text(encoding="utf-8"))[
                    "optimization"
                ]["action_category"],
                "baseline",
            )

    def test_private_state_root_cannot_contain_candidate_workspace(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-campaign-private-boundary-") as temp_dir:
            root = Path(temp_dir)
            request = self._request(root)
            invalid = TeacherDistillRequest(
                **{**request.__dict__, "private_root": request.workspace_root}
            )
            with self.assertRaisesRegex(ValueError, "disjoint"):
                TeacherDistillCampaign(invalid)

    def test_terminal_reason_mapping_is_explicit(self) -> None:
        cases = {
            "success: teacher ABBA passed (candidate/teacher 1.0)": CampaignTerminalStatus.SUCCESS,
            "stall: 5 iterations with no commit": CampaignTerminalStatus.PLATEAU,
            "budget: max-iters": CampaignTerminalStatus.BUDGET_EXHAUSTED,
            "infra: endpoint unavailable": CampaignTerminalStatus.INFRA_ERROR,
            "TEACHER_LEAKAGE_VIOLATION: blocked": CampaignTerminalStatus.TEACHER_LEAKAGE_VIOLATION,
        }
        for reason, expected in cases.items():
            with self.subTest(reason=reason):
                self.assertEqual(TeacherDistillCampaign._status_from_reason(reason), expected)


if __name__ == "__main__":
    unittest.main()
