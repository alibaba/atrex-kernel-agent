"""Production baseline evaluation compatibility for the independent PR5 extraction."""

from __future__ import annotations

import json
import math
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
from typing import Optional



from aka.legacy.resources.paths import ATREX_BENCH_HARNESS, PROFILE_DRIVER, SOL_SEED
from aka.legacy.task.source.inspection import kernel_is_gluon
from aka.legacy.task.problem import AGENT_PROBLEM_FILENAME
from aka.legacy.acceptance.source_policy import production_structure_violations
from aka.legacy.evaluation.local_results import _record_local_test_result, _test_result_from_stdout
from aka.legacy.compat.experiment import baseline_command

from aka.legacy.task.candidate_workspace.state import git_path_blob, git_worktree_blob, read_memory, v0_baseline_commit


def _baseline_preparation(context, *, include_problem=False):
    # Historical Campaign lookup is confined to this external adapter.
    from aka.legacy.evaluation.baseline import BaselineInputs, BaselineResources

    inputs = BaselineInputs(
        context.workspace,
        native=bool(getattr(context, "atrex_bench_root", "")),
        problem=(
            context._loaded_problem()
            if include_problem and not getattr(context, "atrex_bench_root", "")
            else None
        ),
        private_reference_dir=getattr(context, "private_reference_dir", None),
        arch=getattr(context, "arch", ""),
        campaign_name=getattr(context, "campaign_name", ""),
        framework=getattr(context, "framework", ""),
        notes=getattr(context, "notes", ""),
        optimization_mode=getattr(context, "optimization_mode", "leaderboard"),
        platform=getattr(context, "platform", ""),
        target_util=getattr(context, "target_util", 0.0),
    )
    return BaselinePreparation(
        inputs, BaselineResources(ATREX_BENCH_HARNESS, PROFILE_DRIVER)
    )


from aka.legacy.evaluation.baseline import BaselinePreparation
from aka.legacy.evaluation import LegacyEvaluator, EvaluationConfiguration
from aka.contracts.evaluation import EvaluationRequest
from aka.contracts.execution import JobResult


class LegacyBaselineEvaluation:
    @staticmethod
    def evaluator_directive(self) -> str:
        return _baseline_preparation(self, include_problem=True).evaluator_directive()

    @staticmethod
    def install_native_evaluator(self) -> None:
        return _baseline_preparation(self).install_native_evaluator()

    @staticmethod
    def install_profile_driver(self) -> None:
        return _baseline_preparation(self).install_profile_driver()

    @staticmethod
    def native_v0_readme(self, *, generalized: bool) -> str:
        return _baseline_preparation(self).native_v0_readme(generalized=generalized)

    @staticmethod
    def print_v0_evaluator_output(test: subprocess.CompletedProcess[str]) -> None:
        return BaselinePreparation.print_v0_evaluator_output(test)

    @staticmethod
    def run_v0_evaluator(self) -> dict:
        """Run the sole V0 evaluator and persist its local transport record."""
        failed_memory_path = self.workspace / "memory" / "v0.failed.json"
        failed_memory_path.unlink(missing_ok=True)
        test = baseline_command(
            self.workspace,
            self.sandbox_hardware,
            self.sandbox_profile,
            self.sandbox_url,
            self.sandbox_timeout,
            list(
                LegacyEvaluator()
                .prepare(EvaluationRequest(self.workspace, mode="v0"))
                .steps[0]
                .command
            ),
            ssh=self.sandbox_ssh,
            ssh_init=self.sandbox_ssh_init,
            health_command=self.sandbox_health_command,
            gateway_kind="run",
            private_reference_dir=self.private_reference_dir,
        )
        self._print_v0_evaluator_output(test)
        try:
            result = _test_result_from_stdout(test.stdout)
        except (RuntimeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"sandbox V0 baseline produced no usable result: {exc}"
            ) from exc
        memory_path = _record_local_test_result(self.workspace, "v0", result)
        if test.returncode != 0 or not result.get("all_pass"):
            # Keep diagnostics without making latest_version() treat a failed or
            # interrupted baseline as canonical. A rerun will retry V0 directly.
            memory_path.replace(failed_memory_path)
            raise RuntimeError(
                "sandbox V0 baseline failed correctness/performance validation"
            )
        return result

    @staticmethod
    def write_v0_baseline_report(self, result: dict, source_commit: str) -> Path:
        return _baseline_preparation(self).write_v0_baseline_report(
            result, source_commit
        )

    @staticmethod
    def finalize_v0_measurement(
        self,
        result: dict,
        source_commit: str,
        *,
        extra_paths: tuple[str, ...] = (),
    ) -> None:
        """Commit V0 measurement metadata without rewriting its source commit SHA."""
        memory_path = self.workspace / "memory" / "v0.json"
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
        coverage_problem = self._generalized_memory_coverage_problem(memory)
        if coverage_problem:
            raise RuntimeError(f"invalid native V0 measurement: {coverage_problem}")
        self._write_v0_baseline_report(result, source_commit)

        staged = ["memory/v0.json", "baseline_report.md"]
        staged.extend(path for path in extra_paths if (self.workspace / path).is_file())
        subprocess.run(["git", "add", *staged], cwd=str(self.workspace), check=True)
        if (
            subprocess.run(
                ["git", "diff", "--cached", "--quiet"],
                cwd=str(self.workspace),
                check=False,
            ).returncode
            == 0
        ):
            raise RuntimeError(
                "V0 evaluator produced no measurement metadata to commit"
            )
        subprocess.run(
            ["git", "commit", "-m", "V0: record baseline measurement"],
            cwd=str(self.workspace),
            check=True,
            stdout=subprocess.DEVNULL,
        )
        recorded = read_memory(self.workspace, 0) or {}
        if recorded.get("git_commit_hash") != source_commit:
            raise RuntimeError("memory/v0.json does not point to the V0 source commit")
        if git_path_blob(
            self.workspace, source_commit, "kernel.py"
        ) != git_worktree_blob(self.workspace, "kernel.py"):
            raise RuntimeError("V0 kernel.py differs from its recorded source commit")

    @staticmethod
    def setup_baseline_native(self, op_dir: Path, *, generalized: bool) -> None:
        """Seed native Atrex-Bench V0 without launching a coding Agent."""
        kernel_path = self.workspace / "kernel.py"
        reference_path = self.workspace / "reference.py"
        if not kernel_path.is_file() or not reference_path.is_file():
            raise RuntimeError(
                "native V0 requires workspace kernel.py and reference.py"
            )
        if (
            kernel_path.read_bytes()
            != self._loaded_problem().initial_source.entry.read_bytes()
        ):
            raise RuntimeError(
                "native V0 kernel.py differs from the selected initial implementation; refusing "
                "mechanical baseline generation"
            )
        (self.workspace / "README.md").write_text(
            self._native_v0_readme(generalized=generalized), encoding="utf-8"
        )
        source_paths = [
            ".gitignore",
            "CLAUDE.md",
            "README.md",
            "kernel.py",
            "test_kernel.py",
            "profile_driver.py",
            *(item.path for item in self._loaded_problem().agent.files),
        ]
        source_paths = list(
            dict.fromkeys(
                path for path in source_paths if (self.workspace / path).is_file()
            )
        )
        subprocess.run(
            ["git", "add", *source_paths], cwd=str(self.workspace), check=True
        )
        if (
            subprocess.run(
                ["git", "diff", "--cached", "--quiet"],
                cwd=str(self.workspace),
                check=False,
            ).returncode
            != 0
        ):
            subprocess.run(
                ["git", "commit", "-m", "V0: baseline kernel"],
                cwd=str(self.workspace),
                check=True,
                stdout=subprocess.DEVNULL,
            )
        source_commit = v0_baseline_commit(self.workspace)
        if not source_commit:
            raise RuntimeError("native V0 has no committed kernel.py")
        if git_path_blob(
            self.workspace, source_commit, "kernel.py"
        ) != git_worktree_blob(self.workspace, "kernel.py"):
            raise RuntimeError(
                "native V0 source history does not match the selected initial implementation"
            )

        result = self._run_v0_evaluator_with_correctness_prefetch()
        self._finalize_v0_measurement(result, source_commit)
        self._assert_generalized_inputs_are_private()
        contract_problem = self._generalized_contract_commit_problem()
        if contract_problem:
            raise RuntimeError(f"invalid native V0 source commit: {contract_problem}")
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=str(self.workspace),
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

    @staticmethod
    def setup_baseline_sol(self, op_dir: Path) -> None:
        if not SOL_SEED.exists():
            raise FileNotFoundError(f"missing {SOL_SEED}")
        cmd = [
            sys.executable,
            str(SOL_SEED),
            "--op-dir",
            str(op_dir),
            "--name",
            self.campaign_name,
            "--workspace",
            str(self.workspace),
            "--framework",
            self.framework,
            "--platform",
            self.platform,
            # The local step only materializes sources and git state.  GPU
            # correctness/performance is run below in the remote sandbox.
            "--no-bench",
        ]
        from reference import sol_seed

        sol_seed.main(
            cmd[2:], dependencies=self.task_dependencies, problem=self._loaded_problem()
        )
        self._link_runtime()
        source_commit = v0_baseline_commit(self.workspace)
        if not source_commit:
            raise RuntimeError("SOL V0 has no committed kernel.py")
        result = self._run_v0_evaluator_with_correctness_prefetch()
        self._finalize_v0_measurement(
            result,
            source_commit,
            extra_paths=("CLAUDE.md", ".gitignore"),
        )

    @staticmethod
    def framework_baseline_smoke_shape_ids(self, *, limit: int = 3) -> list[str]:
        return _baseline_preparation(self).framework_baseline_smoke_shape_ids(
            limit=limit
        )

    @staticmethod
    def ensure_framework_baseline_solution_manifest(self) -> None:
        return _baseline_preparation(self).ensure_framework_baseline_solution_manifest()

    @staticmethod
    def framework_baseline_smoke_command(self, n: int) -> tuple[str, str]:
        """Return the only ordinary evaluator command the V1 implementation Agent should run."""
        command = ["python", "tools/sandbox.py", "--kind", "run"]
        if self.sandbox_hardware:
            command += ["--hardware", self.sandbox_hardware]
        if self.sandbox_ssh:
            command += ["--ssh", self.sandbox_ssh]
        elif self.sandbox_url:
            command += ["--url", self.sandbox_url]
        elif self.sandbox_profile:
            command += ["--gateway-profile", self.sandbox_profile]
        command += [
            "--no-sync",
            "--",
            "python",
            "test_kernel.py",
            "--version",
            f"v{n}",
        ]
        shape_ids = (
            self._framework_baseline_smoke_shape_ids() if self.atrex_bench_root else []
        )
        for shape_id in shape_ids:
            command += ["--shape-id", shape_id]
        if self.atrex_bench_root:
            command += ["--timed-runs", "1"]
        command.append("--no-memory")
        if shape_ids:
            scope = (
                f"The supervisor selected {len(shape_ids)} opaque V0 ids spanning the baseline "
                "latency distribution. This is smoke coverage only; do not infer their private inputs."
            )
        else:
            scope = (
                "This evaluator route has no safe targeted-case selector, so run this base-seed "
                "smoke at most once after the final implementation edit."
            )
        return shlex.join(command), scope

    @staticmethod
    def restore_immutable_baseline_paths(self, baseline_commit: str) -> list[str]:
        """Put back any ground-truth file the session edited, and report what was restored.

        A session that "fixes" the harness or memory/v0.json is a compliance problem, but a
        mechanically repairable one — discarding its kernel over it would throw away hours of
        work for nothing. Acceptance is decided by the kernel itself.
        """
        return self.candidate_workspace.restore_baseline(
            self.workspace, baseline_commit
        )

    @staticmethod
    def framework_baseline_problem(
        self, v0_blob, baseline_commit, *, include_policy_review=True
    ) -> str:
        from aka.contracts.acceptance import FrameworkCandidateRequest
        from aka.legacy.acceptance import LegacyAcceptancePolicy

        return LegacyAcceptancePolicy(
            candidate_workspace=self.candidate_workspace,
            production_policy=self._production_kernel_violations,
        ).check_framework_candidate(
            FrameworkCandidateRequest(
                self.workspace,
                self.framework,
                v0_blob,
                baseline_commit,
                include_policy_review,
            )
        )

    @staticmethod
    def framework_baseline_external_gates(self, n: int) -> tuple[Optional[dict], str]:
        """Run policy review and the sole authoritative V1 evaluator concurrently."""
        print(
            "[orchestrator] framework baseline: running policy review and combined "
            "correctness/performance validation in parallel",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=2) as executor:
            policy_future = executor.submit(
                copy_context().run, self._production_kernel_violations
            )
            validation_future = executor.submit(
                copy_context().run, self._validate_framework_baseline, n
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
                f"the candidate is not a self-contained {self.framework} implementation: "
                + "; ".join(violations)
            )
        if not validation_problem:
            validation_problem = self._supplemental_numerical_feedback(
                standard_correctness_passed=True
            )
        return result, validation_problem

    @staticmethod
    def supplemental_numerical_feedback(
        self,
        workspace: Path | None = None,
        *,
        standard_correctness_passed: bool = False,
    ) -> str:
        from aka.legacy.workflow.numerical_review import supplemental_feedback

        return supplemental_feedback(
            self,
            workspace or self.workspace,
            standard_correctness_passed=standard_correctness_passed,
        )

    @staticmethod
    def validate_framework_baseline(self, n: int) -> tuple[Optional[dict], str]:
        """Validate V1 once: base-seed performance plus five extra correctness cases."""
        # V1 is a correctness/framework bring-up gate, not a performance gate. Keep a
        # small timing sample so a slow but valid first implementation can enter the
        # optimization loop without exhausting the evaluator's benchmark budget.
        evaluator = LegacyEvaluator(
            EvaluationConfiguration(native=bool(self.atrex_bench_root))
        )
        prepared = evaluator.prepare(
            EvaluationRequest(self.workspace, mode="framework", version=n)
        )
        command = list(prepared.steps[0].command)
        try:
            test = baseline_command(
                self.workspace,
                self.sandbox_hardware,
                self.sandbox_profile,
                self.sandbox_url,
                self.sandbox_timeout,
                command,
                ssh=self.sandbox_ssh,
                ssh_init=self.sandbox_ssh_init,
                health_command=self.sandbox_health_command,
                gateway_kind="run",
                private_reference_dir=self.private_reference_dir,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return None, f"combined validation failed to run: {exc}"
        self._print_v0_evaluator_output(test)
        if test.returncode != 0:
            return None, f"combined validation command failed (exit={test.returncode})"
        try:
            result = _test_result_from_stdout(test.stdout)
        except (RuntimeError, json.JSONDecodeError) as exc:
            return None, f"combined validation produced no usable result: {exc}"
        from aka.legacy.evaluation.evaluator import framework_result_problem

        problem = framework_result_problem(result, self.workspace)
        return (None, problem) if problem else (result, "")
