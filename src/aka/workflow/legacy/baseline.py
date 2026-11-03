"""Coordinate baseline setup, measurement, policy, and retained Workflow state."""

from __future__ import annotations
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from aka.contracts.content import WorkspaceFile
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from aka.contracts.evaluation import EvaluationRequest, EvaluationUnavailable
from aka.contracts.experiment import ExperimentExecutionError
from aka.contracts.acceptance import FrameworkCandidateRequest
from .records import _record_local_test_result
from .records import read_memory
from .revision_state import WorkspaceState


class WorkflowBaseline:
    def __init__(
        self, campaign, *, preparation, experiment, acceptance, command
    ):
        self.campaign = campaign
        self.preparation = preparation
        self.experiment = experiment
        self.acceptance = acceptance
        self.command = command
        self._last_framework_verification = None

    @property
    def _workspace_state(self):
        return WorkspaceState(self.campaign.workspace, self.campaign.candidate_workspace)

    def evaluator_directive(self):
        return self.preparation(True).evaluator_directive()

    def install_native_evaluator(self):
        harness = self.campaign.problem.evaluation.harness
        if harness is None:
            if self.campaign.atrex_bench_root:
                raise ValueError("Task has no native evaluator harness")
            return
        self.campaign.candidate_workspace.install_files(
            self.campaign.workspace, (harness,), immutable=True)

    def install_profile_driver(self):
        from .resources import profile_driver_files
        self.campaign.candidate_workspace.install_files(
            self.campaign.workspace, profile_driver_files(self.campaign.plugin_registry),
            immutable=True)

    def native_v0_readme(self, *, generalized):
        return self.preparation().native_v0_readme(generalized=generalized)

    def write_v0_baseline_report(self, result, source_commit):
        from .baseline_report import render_baseline_report
        campaign = self.campaign
        report = render_baseline_report(result, source_commit,
            private_shapes=(campaign.workspace / "agent_problem.json").is_file())
        campaign.candidate_workspace.install_files(campaign.workspace,
            (WorkspaceFile(path="baseline_report.md", text=report),))
        return campaign.workspace / "baseline_report.md"

    def framework_baseline_smoke_shape_ids(self, *, limit=3):
        return self.preparation().framework_baseline_smoke_shape_ids(limit=limit)

    def ensure_framework_baseline_solution_manifest(self):
        campaign = self.campaign
        if (campaign.workspace / "solution.json").is_file() or not campaign.atrex_bench_root:
            return
        prepare = getattr(campaign.source_provider, "prepare_candidate_manifest", None)
        if not callable(prepare):
            raise TypeError("selected SourceProvider does not implement prepare_candidate_manifest")
        from aka.contracts.workspace import CandidateManifestRequest
        reference = None
        if (campaign.workspace / "reference.py").is_file():
            reference = campaign.candidate_workspace.read_working(
                campaign.workspace, ("reference.py",)
            ).files["reference.py"]
        contribution = prepare(CandidateManifestRequest(
            campaign.services.evaluation_input(), campaign.campaign_name, campaign.framework, reference
        ))
        if not isinstance(contribution, WorkspaceFile) or contribution.path != "solution.json":
            raise TypeError("SourceProvider manifest requires a solution.json WorkspaceFile")
        campaign.candidate_workspace.install_files(campaign.workspace, (contribution,))

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
        if verification.failure is not None:
            raise EvaluationUnavailable(verification.failure)
        if not verification.runs:
            raise RuntimeError("sandbox V0 baseline produced no usable result: " + verification.error)
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
        self._last_framework_verification = verification
        if not verification.passed:
            if verification.failure is not None:
                repair_enabled = (
                    self.campaign.gateway_candidate_recovery == "typed-once"
                    and verification.failure.candidate_repairable
                )
                if not repair_enabled:
                    raise EvaluationUnavailable(verification.failure)
            return None, verification.error
        return verification.runs[0].result, ""

    def framework_baseline_rejection_diagnostic(self, problem: str) -> dict:
        """Return bounded public facts for the latest authoritative V1 gate."""
        workspace = str(self.campaign.workspace.resolve())

        def public_text(value) -> str:
            text = str(value or "").replace(workspace, "<workspace>")
            text = re.sub(r'File "[^"]+"', 'File "<path>"', text)
            return text[:4_000]

        diagnostic = {"message": public_text(problem)}
        verification = self._last_framework_verification
        if verification is None:
            return diagnostic
        diagnostic["gate"] = verification.gate
        if verification.failure is not None:
            diagnostic["failure"] = {
                "category": verification.failure.category,
                "code": verification.failure.code,
                "message": public_text(verification.failure.message),
                "candidate_repairable": verification.failure.candidate_repairable,
            }
        runs = []
        for run in verification.runs[:4]:
            result = run.result if isinstance(run.result, dict) else {}
            public_run = {"exit_code": run.exit_code}
            if type(result.get("all_pass")) is bool:
                public_run["all_pass"] = result["all_pass"]
            if isinstance(result.get("eval_id"), str):
                public_run["evaluation_id"] = result["eval_id"][:128]
            actionable = []
            actionable_rows = result.get("actionable_diagnostics")
            if not isinstance(actionable_rows, list):
                actionable_rows = []
            for row in actionable_rows[:8]:
                if not isinstance(row, dict):
                    continue
                item = {}
                if isinstance(row.get("stage"), str):
                    item["stage"] = row["stage"][:128]
                if isinstance(row.get("message"), str):
                    item["message"] = public_text(row["message"])
                if item:
                    actionable.append(item)
            if actionable:
                public_run["actionable_diagnostics"] = actionable
            runs.append(public_run)
        if runs:
            diagnostic["runs"] = runs
        return diagnostic

    def framework_baseline_problem(
        self, v0_blob, baseline_commit, *, include_policy_review=True
    ):
        from aka.contracts.acceptance import FrameworkCandidateFacts
        from aka.task.source.inspection import kernel_is_gluon
        from aka.workflow.legacy.review_rules.source_policy import production_structure_violations

        campaign = self.campaign
        blob = self._workspace_state.working_identity("kernel.py")
        # Preserve the original ordering: unchanged candidates do not invoke review.
        violations = ()
        gluon = False
        mutated = ()
        if blob and blob != v0_blob:
            if campaign.source_checks_enabled:
                violations = tuple(campaign._production_kernel_violations() if include_policy_review
                    else production_structure_violations(campaign.workspace, campaign.framework))
            if not violations:
                gluon = campaign.source_checks_enabled and campaign.framework.lower() in {"triton", "gluon"} and kernel_is_gluon(campaign.workspace)
                if not gluon:
                    mutated = tuple(campaign.candidate_workspace.baseline_changes(campaign.workspace, baseline_commit))
        return self.acceptance.check_framework_candidate(
            FrameworkCandidateRequest(
                self.campaign.workspace,
                self.campaign.framework,
                v0_blob,
                baseline_commit,
                include_policy_review,
                source_checks_enabled=self.campaign.source_checks_enabled,
                facts=FrameworkCandidateFacts(blob, mutated, violations, gluon),
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
        expected = self.campaign.candidate_workspace.current_revision(self.campaign.workspace)
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
        coverage_problem = self.campaign._generalized_memory_coverage_problem(memory)
        if coverage_problem:
            raise RuntimeError(f"invalid native V0 measurement: {coverage_problem}")
        from .baseline_report import render_baseline_report
        report = render_baseline_report(result, source_commit,
            private_shapes=(self.campaign.workspace / "agent_problem.json").is_file())
        files = [WorkspaceFile(path="memory/v0.json",
                               text=json.dumps(memory, indent=2, ensure_ascii=False) + "\n"),
                 WorkspaceFile(path="baseline_report.md", text=report)]
        files.extend(WorkspaceFile(path=name, source=self.campaign.workspace / name)
                     for name in dict.fromkeys(extra_paths)
                     if name not in {"memory/v0.json", "baseline_report.md"}
                     and (self.campaign.workspace / name).is_file())
        revision = self.campaign.candidate_workspace.record(
            self.campaign.workspace, expected_revision=expected,
            files=tuple(files), message="V0: record baseline measurement",
        )
        if revision == expected:
            raise RuntimeError("V0 evaluator produced no measurement metadata to commit")
        recorded = read_memory(self.campaign.workspace, 0) or {}
        if recorded.get("git_commit_hash") != source_commit:
            raise RuntimeError("memory/v0.json does not point to the V0 source commit")
        if self._workspace_state.content_identity(source_commit, "kernel.py") != self._workspace_state.working_identity("kernel.py"):
            raise RuntimeError("V0 kernel.py differs from its recorded source commit")

    def setup_baseline_native(self, op_dir: Path, *, generalized: bool) -> None:
        """Seed native Atrex-Bench V0 without launching a coding Agent."""
        expected = self.campaign.candidate_workspace.current_revision(self.campaign.workspace)
        kernel_path = self.campaign.workspace / "kernel.py"
        reference_path = self.campaign.workspace / "reference.py"
        if not kernel_path.is_file() or not reference_path.is_file():
            raise RuntimeError(
                "native V0 requires workspace kernel.py and reference.py"
            )
        readme = self.native_v0_readme(generalized=generalized)
        self.campaign.candidate_workspace.install_files(
            self.campaign.workspace, (WorkspaceFile(path="README.md", text=readme),),
            immutable=True)
        source_paths = [
            ".gitignore",
            "kernel.py",
        ]
        source_paths = list(
            dict.fromkeys(
                path
                for path in source_paths
                if (self.campaign.workspace / path).is_file()
            )
        )
        self.campaign.candidate_workspace.record(
            self.campaign.workspace,
            expected_revision=expected,
            files=tuple(WorkspaceFile(path=name, source=self.campaign.workspace / name)
                        for name in source_paths),
            message="V0: baseline kernel",
        )
        source_commit = self._workspace_state.initial_kernel_revision()
        if not source_commit:
            raise RuntimeError("native V0 has no committed kernel.py")
        if self._workspace_state.content_identity(source_commit, "kernel.py") != self._workspace_state.working_identity("kernel.py"):
            raise RuntimeError(
                "native V0 source history does not match the selected initial implementation"
            )

        result = self.campaign._run_v0_evaluator_with_correctness_prefetch()
        self.finalize_v0_measurement(result, source_commit)
        self.campaign._assert_generalized_inputs_are_private()
        contract_problem = self.campaign._generalized_contract_commit_problem()
        if contract_problem:
            raise RuntimeError(f"invalid native V0 source commit: {contract_problem}")
        status = "\n".join(self._workspace_state.working_changes())
        if status:
            print(
                "[orchestrator] WARNING: native V0 is continuing with a dirty "
                f"workspace:\n{status}",
                file=sys.stderr,
                flush=True,
            )

    def setup_baseline_sol(self, op_dir: Path) -> None:
        from aka.contracts.workspace import SourceRequest
        from .resources import campaign_initial_files, campaign_context_directories

        problem = self.campaign._loaded_problem()
        content = self.campaign.source_provider.prepare(SourceRequest(
            kind="sol", initial=problem.initial_source, name=self.campaign.campaign_name,
            framework=self.campaign.framework, platform=self.campaign.platform,
        ))
        source_commit = self.campaign.candidate_workspace.prepare_sol(
            self.campaign.workspace, content, problem_files=problem.agent.files,
            harness=problem.evaluation.harness,
            runtime_files=(*campaign_initial_files(self.campaign, sol=True),
                           *self.campaign.problem_provider.runtime_files(problem)),
            context_directories=campaign_context_directories(self.campaign),
        )
        self.campaign._link_runtime()
        result = self.campaign._run_v0_evaluator_with_correctness_prefetch()
        self.finalize_v0_measurement(
            result,
            source_commit,
            extra_paths=(".gitignore",),
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
            except EvaluationUnavailable:
                raise
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
