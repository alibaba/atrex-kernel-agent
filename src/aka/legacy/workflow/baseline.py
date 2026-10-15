"""Coordinate baseline setup, measurement, policy, and retained Workflow state."""

from __future__ import annotations
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from aka.contracts.evaluation import EvaluationRequest
from aka.contracts.experiment import ExperimentExecutionError
from aka.contracts.acceptance import FrameworkCandidateRequest
from aka.legacy.evaluation.local_results import _record_local_test_result
from aka.legacy.task.candidate_workspace.state import git_path_blob, git_worktree_blob, read_memory, v0_baseline_commit


class WorkflowBaseline:
    def __init__(
        self, campaign, *, preparation, experiment, acceptance, command, sol_seed
    ):
        self.campaign = campaign
        self.preparation = preparation
        self.experiment = experiment
        self.acceptance = acceptance
        self.command = command
        self.sol_seed = sol_seed

    def evaluator_directive(self):
        return self.preparation(True).evaluator_directive()

    def install_native_evaluator(self):
        return self.preparation().install_native_evaluator()

    def install_profile_driver(self):
        return self.preparation().install_profile_driver()

    def native_v0_readme(self, *, generalized):
        return self.preparation().native_v0_readme(generalized=generalized)

    def write_v0_baseline_report(self, result, source_commit):
        return self.preparation().write_v0_baseline_report(result, source_commit)

    def framework_baseline_smoke_shape_ids(self, *, limit=3):
        return self.preparation().framework_baseline_smoke_shape_ids(limit=limit)

    def ensure_framework_baseline_solution_manifest(self):
        return self.preparation().ensure_framework_baseline_solution_manifest()

    def framework_baseline_smoke_command(self, n):
        shapes = (
            self.framework_baseline_smoke_shape_ids()
            if self.campaign.atrex_bench_root
            else []
        )
        command = ["python", "test_kernel.py", "--version", f"v{n}"]
        for shape in shapes:
            command += ["--shape-id", shape]
        if self.campaign.atrex_bench_root:
            command += ["--timed-runs", "1"]
        command.append("--no-memory")
        scope = (
            f"The supervisor selected {len(shapes)} opaque V0 ids spanning the baseline "
            "latency distribution. This is smoke coverage only; do not infer their private inputs."
            if shapes
            else "This evaluator route has no safe targeted-case selector, so run this base-seed "
            "smoke at most once after the final implementation edit."
        )
        return self.command(command), scope

    def run_v0_evaluator(self):
        workspace = self.campaign.workspace
        failed = workspace / "memory/v0.failed.json"
        failed.unlink(missing_ok=True)
        try:
            verification = self.experiment.run(EvaluationRequest(workspace, mode="v0"))
        except ExperimentExecutionError as exc:
            cause = exc.__cause__
            if isinstance(cause, (RuntimeError, json.JSONDecodeError)):
                raise RuntimeError(
                    f"sandbox V0 baseline produced no usable result: {cause}"
                ) from cause
            raise cause if cause is not None else exc
        result = verification.runs[0].result
        memory = _record_local_test_result(workspace, "v0", result)
        if not verification.passed:
            memory.replace(failed)
            raise RuntimeError(
                "sandbox V0 baseline failed correctness/performance validation"
            )
        return result

    def validate_framework_baseline(self, n):
        try:
            verification = self.experiment.run(
                EvaluationRequest(self.campaign.workspace, mode="framework", version=n)
            )
        except ExperimentExecutionError as exc:
            cause = exc.__cause__
            if isinstance(cause, (OSError, subprocess.SubprocessError)):
                return None, f"combined validation failed to run: {cause}"
            if isinstance(cause, (RuntimeError, json.JSONDecodeError)):
                return None, f"combined validation produced no usable result: {cause}"
            raise cause if cause is not None else exc
        except (OSError, subprocess.SubprocessError) as exc:
            return None, f"combined validation failed to run: {exc}"
        if not verification.passed:
            return None, verification.error
        return verification.runs[0].result, ""

    def framework_baseline_problem(
        self, v0_blob, baseline_commit, *, include_policy_review=True
    ):
        return self.acceptance.check_framework_candidate(
            FrameworkCandidateRequest(
                self.campaign.workspace,
                self.campaign.framework,
                v0_blob,
                baseline_commit,
                include_policy_review,
                source_checks_enabled=self.campaign.source_checks_enabled,
            )
        )

    def supplemental_numerical_feedback(
        self, workspace=None, *, standard_correctness_passed=False
    ):
        return self.campaign.numerical_feedback(
            workspace or self.campaign.workspace,
            standard_correctness_passed=standard_correctness_passed,
        )

    def finalize_v0_measurement(
        self,
        result: dict,
        source_commit: str,
        *,
        extra_paths: tuple[str, ...] = (),
    ) -> None:
        """Commit V0 measurement metadata without rewriting its source commit SHA."""
        memory_path = self.campaign.workspace / "memory" / "v0.json"
        memory = json.loads(memory_path.read_text(encoding="utf-8"))
        memory["git_commit_hash"] = source_commit
        optimization = memory.setdefault("optimization", {})
        optimization["action_category"] = "baseline"
        optimization["action_description"] = (
            "selected initial implementation measured by the official full-workload evaluator"
        )
        optimization["expected_impact"] = (
            "correctness and latency reference for later versions"
        )
        optimization["risks_and_rollback"] = "none; immutable V0 reference baseline"
        memory_path.write_text(
            json.dumps(memory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        coverage_problem = self.campaign._generalized_memory_coverage_problem(memory)
        if coverage_problem:
            raise RuntimeError(f"invalid native V0 measurement: {coverage_problem}")
        self.write_v0_baseline_report(result, source_commit)

        staged = ["memory/v0.json", "baseline_report.md"]
        staged.extend(
            path for path in extra_paths if (self.campaign.workspace / path).is_file()
        )
        subprocess.run(
            ["git", "add", *staged], cwd=str(self.campaign.workspace), check=True
        )
        if (
            subprocess.run(
                ["git", "diff", "--cached", "--quiet"],
                cwd=str(self.campaign.workspace),
                check=False,
            ).returncode
            == 0
        ):
            raise RuntimeError(
                "V0 evaluator produced no measurement metadata to commit"
            )
        subprocess.run(
            ["git", "commit", "-m", "V0: record baseline measurement"],
            cwd=str(self.campaign.workspace),
            check=True,
            stdout=subprocess.DEVNULL,
        )
        recorded = read_memory(self.campaign.workspace, 0) or {}
        if recorded.get("git_commit_hash") != source_commit:
            raise RuntimeError("memory/v0.json does not point to the V0 source commit")
        if git_path_blob(
            self.campaign.workspace, source_commit, "kernel.py"
        ) != git_worktree_blob(self.campaign.workspace, "kernel.py"):
            raise RuntimeError("V0 kernel.py differs from its recorded source commit")

    def setup_baseline_native(self, op_dir: Path, *, generalized: bool) -> None:
        """Seed native Atrex-Bench V0 without launching a coding Agent."""
        kernel_path = self.campaign.workspace / "kernel.py"
        reference_path = self.campaign.workspace / "reference.py"
        if not kernel_path.is_file() or not reference_path.is_file():
            raise RuntimeError(
                "native V0 requires workspace kernel.py and reference.py"
            )
        if (
            kernel_path.read_bytes()
            != self.campaign._loaded_problem().initial_source.entry.read_bytes()
        ):
            raise RuntimeError(
                "native V0 kernel.py differs from the selected initial implementation; refusing "
                "mechanical baseline generation"
            )
        (self.campaign.workspace / "README.md").write_text(
            self.native_v0_readme(generalized=generalized), encoding="utf-8"
        )
        source_paths = [
            ".gitignore",
            "CLAUDE.md",
            "README.md",
            "kernel.py",
            "test_kernel.py",
            "profile_driver.py",
            *(item.path for item in self.campaign._loaded_problem().agent.files),
        ]
        source_paths = list(
            dict.fromkeys(
                path
                for path in source_paths
                if (self.campaign.workspace / path).is_file()
            )
        )
        subprocess.run(
            ["git", "add", *source_paths], cwd=str(self.campaign.workspace), check=True
        )
        if (
            subprocess.run(
                ["git", "diff", "--cached", "--quiet"],
                cwd=str(self.campaign.workspace),
                check=False,
            ).returncode
            != 0
        ):
            subprocess.run(
                ["git", "commit", "-m", "V0: baseline kernel"],
                cwd=str(self.campaign.workspace),
                check=True,
                stdout=subprocess.DEVNULL,
            )
        source_commit = v0_baseline_commit(self.campaign.workspace)
        if not source_commit:
            raise RuntimeError("native V0 has no committed kernel.py")
        if git_path_blob(
            self.campaign.workspace, source_commit, "kernel.py"
        ) != git_worktree_blob(self.campaign.workspace, "kernel.py"):
            raise RuntimeError(
                "native V0 source history does not match the selected initial implementation"
            )

        result = self.campaign._run_v0_evaluator_with_correctness_prefetch()
        self.finalize_v0_measurement(result, source_commit)
        self.campaign._assert_generalized_inputs_are_private()
        contract_problem = self.campaign._generalized_contract_commit_problem()
        if contract_problem:
            raise RuntimeError(f"invalid native V0 source commit: {contract_problem}")
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=str(self.campaign.workspace),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if status:
            print(
                "[orchestrator] WARNING: native V0 is continuing with a dirty "
                f"workspace:\n{status}",
                file=sys.stderr,
                flush=True,
            )

    def setup_baseline_sol(self, op_dir: Path) -> None:
        if not self.sol_seed.exists():
            raise FileNotFoundError(f"missing {self.sol_seed}")
        cmd = [
            sys.executable,
            str(self.sol_seed),
            "--op-dir",
            str(op_dir),
            "--name",
            self.campaign.campaign_name,
            "--workspace",
            str(self.campaign.workspace),
            "--framework",
            self.campaign.framework,
            "--platform",
            self.campaign.platform,
            # The local step only materializes sources and git state.  GPU
            # correctness/performance is run below in the remote sandbox.
            "--no-bench",
        ]
        from reference import sol_seed

        sol_seed.main(
            cmd[2:],
            dependencies=dict(
                problem=self.campaign.problem_provider,
                source=self.campaign.source_provider,
                workspace=self.campaign.candidate_workspace,
            ),
            problem=self.campaign._loaded_problem(),
        )
        self.campaign._link_runtime()
        source_commit = v0_baseline_commit(self.campaign.workspace)
        if not source_commit:
            raise RuntimeError("SOL V0 has no committed kernel.py")
        result = self.campaign._run_v0_evaluator_with_correctness_prefetch()
        self.finalize_v0_measurement(
            result,
            source_commit,
            extra_paths=("CLAUDE.md", ".gitignore"),
        )

    def restore_immutable_baseline_paths(self, baseline_commit: str) -> list[str]:
        """Put back any ground-truth file the session edited, and report what was restored.

        A session that "fixes" the harness or memory/v0.json is a compliance problem, but a
        mechanically repairable one — discarding its kernel over it would throw away hours of
        work for nothing. Acceptance is decided by the kernel itself.
        """
        return self.campaign.candidate_workspace.restore_baseline(
            self.campaign.workspace, baseline_commit
        )

    def framework_baseline_external_gates(self, n: int) -> tuple[Optional[dict], str]:
        """Run policy review and the sole authoritative V1 evaluator concurrently."""
        if self.campaign.source_checks_enabled:
            message = "running policy review and combined correctness/performance validation in parallel"
        else:
            message = "multi-GPU Agate source review skipped; running correctness/performance validation"
        print(f"[orchestrator] framework baseline: {message}", flush=True)
        with ThreadPoolExecutor(max_workers=2) as executor:
            policy_future = executor.submit(
                copy_context().run, self.campaign._production_kernel_violations
            )
            validation_future = executor.submit(
                copy_context().run, self.validate_framework_baseline, n
            )
            try:
                violations = policy_future.result()
            except Exception as exc:
                violations = [
                    "independent production policy review failed: "
                    f"{type(exc).__name__}: {exc}"
                ]
            try:
                result, validation_problem = validation_future.result()
            except Exception as exc:
                result, validation_problem = (
                    None,
                    "combined validation failed: " f"{type(exc).__name__}: {exc}",
                )
        if violations:
            return None, (
                f"the candidate is not a self-contained {self.campaign.framework} implementation: "
                + "; ".join(violations)
            )
        if not validation_problem:
            validation_problem = self.supplemental_numerical_feedback(
                standard_correctness_passed=True
            )
        return result, validation_problem
