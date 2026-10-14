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

from orchestrator._bootstrap import task_modules

task_modules()

from orchestrator.constants import ATREX_BENCH_HARNESS, PROFILE_DRIVER, SOL_SEED
from orchestrator.hardware import kernel_is_gluon
from orchestrator.operator_layout import AGENT_PROBLEM_FILENAME
from orchestrator.optimization_policy import production_structure_violations
from orchestrator.session_io import (
    _record_local_test_result,
    _sandbox_command,
    _test_result_from_stdout,
)
from orchestrator.workspace_state import (
    git_path_blob,
    git_worktree_blob,
    read_memory,
    v0_baseline_commit,
)


class LegacyBaselineEvaluation:
    @staticmethod
    def evaluator_directive(self) -> str:
        if self.atrex_bench_root:
            if self.private_reference_dir is not None:
                return (
                    "## Evaluation route: Atrex-Bench generalized private-case native\n\n"
                    "Treat workspace `agent_problem.json` as the authoritative public optimization "
                    "contract. Exact `shapes.json`, `metadata.json`, and `roofline.json` cases are "
                    "evaluator-only and intentionally absent from the workspace; never search for, "
                    "reconstruct, or read the private reference directory. The immutable "
                    "`test_kernel.py` adapter and sandbox inject those cases only into the remote "
                    "official evaluator. Optimize for the complete declared `shape_domain`, using "
                    "aggregate `distribution_profile` shares only for prioritization. Correctness "
                    "must pass every hidden case. The optimization score is the arithmetic mean "
                    "of each opaque shape's measured speedup against its authoritative Atrex-Bench "
                    "metadata production latency; maximize `performance_score`. After "
                    "evaluation, use the real "
                    "`latency_us_by_shape` map keyed by opaque ids without attempting to infer their "
                    "private inputs. For profiling, choose a real opaque id from canonical "
                    "`memory/vN.json.performance.latency_us_by_shape` with PROFILE_SHAPE_ID; the "
                    "sandbox injects only that real case into the remote profile job. Do not edit "
                    "or replace the adapter or implement a custom correctness/timing harness."
                )
            return (
                "## Evaluation route: Atrex-Bench native\n\n"
                "This workspace's `test_kernel.py` is an orchestrator-installed immutable adapter. "
                "It invokes the canonical `atrex-bench/scripts/run_eval.py` against `kernel.py` and "
                "the workspace `reference.py`/`input.py`/`shapes.json`/`metadata.json`, then emits "
                "the optimizer's `RESULT_JSON` transport line from the official `eval_result.json`. "
                "The optimization score is `performance_score`: for each shape, divide "
                "metadata `production_performance.performance_us` by measured latency, then take "
                "the arithmetic mean across shapes. Maximize this score. "
                "Do not edit or replace this adapter and do not implement a custom correctness or "
                "timing harness. `--multi-seed N` maps to N additional Atrex-Bench correctness "
                "cases while performance remains one official run per shape."
            )
        if self._loaded_problem().initial_source.kind == "sol":
            return (
                "## Evaluation route: SOL-ExecBench\n\n"
                "Keep using the immutable SOL `test_kernel.py`, which invokes `sol-execbench` over "
                "the complete `workload.jsonl`. Each workload's SOL reference is its performance "
                "baseline; maximize `performance_score`, the arithmetic mean of per-workload "
                "speedups. Do not substitute the Atrex-Bench native evaluator."
            )
        return (
            "## Evaluation route: derived legacy boundary\n\n"
            "This derived boundary is not a complete Atrex-Bench operator directory. Preserve its "
            "committed full-shape `test_kernel.py` methodology and do not replace it after V0."
        )

    @staticmethod
    def install_native_evaluator(self) -> None:
        """Seed the immutable adapter used only by native Atrex-Bench shape campaigns."""
        if not self.atrex_bench_root:
            return
        if not ATREX_BENCH_HARNESS.is_file():
            raise FileNotFoundError(f"missing {ATREX_BENCH_HARNESS}")
        shutil.copy2(ATREX_BENCH_HARNESS, self.workspace / "test_kernel.py")

    @staticmethod
    def install_profile_driver(self) -> None:
        """Seed the immutable external profiling entry for every campaign layout.

        Both profiler wrappers run ``python <file>``, so profiling needs a runnable script.
        Keeping it out of ``kernel.py`` means a session that rewrites ``run()``/``Model``
        cannot silently destroy profiling: an in-kernel ``__main__`` block would vanish with
        the rewrite and the profiler would still exit 0 having captured nothing.
        """
        if not PROFILE_DRIVER.is_file():
            raise FileNotFoundError(f"missing {PROFILE_DRIVER}")
        shutil.copy2(PROFILE_DRIVER, self.workspace / "profile_driver.py")
        # Stage it when a repository already exists so the baseline commit tracks it without
        # depending on how the setup session stages files; restoring an immutable path needs
        # a blob in the root commit.
        if (self.workspace / ".git").exists():
            subprocess.run(
                ["git", "add", "profile_driver.py"],
                cwd=str(self.workspace),
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    @staticmethod
    def native_v0_readme(self, *, generalized: bool) -> str:
        contract = (
            "`agent_problem.json` is the authoritative public contract. Exact evaluator "
            "cases remain private; latency-map keys are opaque ids."
            if generalized
            else "`shapes.json` and the other copied operator files define the public workload."
        )
        runtime_arch = (
            self.arch or "unknown (use the runtime GPU API before choosing codegen)"
        )
        notes = self.notes.strip() or "none"
        return (
            f"# kernel_opt_{self.campaign_name}\n\n"
            "Profile-driven optimization of one native Atrex-Bench operator.\n\n"
            "## Goal\n\n"
            "Maximize the arithmetic mean of per-shape speedups against Atrex-Bench metadata "
            "production performance while every evaluator case remains "
            "correct. V0 uses the Problem's selected initial implementation; optimized versions must "
            f"migrate the computation to `{self.framework}`.\n\n"
            "## Campaign\n\n"
            f"- Target platform: `{self.platform}`\n"
            f"- Runtime architecture: `{runtime_arch}`\n"
            f"- Optimization mode: `{self.optimization_mode}`\n"
            f"- Target framework: `{self.framework}`\n"
            f"- Target utilization stop condition: `{self.target_util:g}%`\n"
            f"- Additional notes: {notes}\n\n"
            "## Contract and evaluator\n\n"
            f"- {contract}\n"
            "- `test_kernel.py` is the immutable supervisor-installed adapter to the official "
            "`atrex-bench/scripts/run_eval.py`.\n"
            "- V0 uses exactly one full-workload base-seed evaluator run. It does not profile, "
            "run multi-seed correctness, or perform ABBA.\n"
            "- Ground-truth operator files and `profile_driver.py` are immutable after V0.\n\n"
            "## Hardware evidence policy\n\n"
            "V0 records identity only and does not speculate about peak specifications. Before an "
            "optimization plan uses a hardware limit, source it from enabled knowledge tools or primary specifications "
            "and cite the exact path. The runtime architecture API is authoritative when a device "
            "name or vendor SMI is desensitized.\n\n"
            "## Stop conditions\n\n"
            "The supervisor stops at the configured iteration/token/stall limits, when the target "
            "utilization is reached, or on a terminal repeated blocker. Correctness is mandatory "
            "for every promoted version.\n"
        )

    @staticmethod
    def print_v0_evaluator_output(test: subprocess.CompletedProcess[str]) -> None:
        if test.stdout:
            print(
                test.stdout, end="" if test.stdout.endswith("\n") else "\n", flush=True
            )
        if test.stderr:
            print(
                test.stderr,
                end="" if test.stderr.endswith("\n") else "\n",
                file=sys.stderr,
                flush=True,
            )

    @staticmethod
    def run_v0_evaluator(self) -> dict:
        """Run the sole V0 evaluator and persist its local transport record."""
        failed_memory_path = self.workspace / "memory" / "v0.failed.json"
        failed_memory_path.unlink(missing_ok=True)
        test = _sandbox_command(
            self.workspace,
            self.sandbox_hardware,
            self.sandbox_profile,
            self.sandbox_url,
            self.sandbox_timeout,
            ["python", "test_kernel.py", "--version", "v0", "--no-memory"],
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
        def metric(name: str) -> str:
            value = result.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return "unknown"
            return f"{float(value):.6g}"

        by_shape = result.get("latency_us_by_shape")
        by_shape = by_shape if isinstance(by_shape, dict) else {}
        evaluator = str(result.get("evaluator") or "workspace test_kernel.py")
        eval_id = result.get("eval_id")
        shape_note = (
            "Shape ids are opaque; exact production inputs remain evaluator-private."
            if (self.workspace / AGENT_PROBLEM_FILENAME).is_file()
            else "The complete public workload was evaluated."
        )
        report = (
            "# V0 baseline report\n\n"
            "This report was generated mechanically by the campaign supervisor.\n\n"
            "## Provenance\n\n"
            f"- Source commit: `{source_commit}`\n"
            "- Implementation: the Problem's selected initial source, prepared by Source\n"
            f"- Evaluator: `{evaluator}`\n"
            "- Route: remote sandbox, one base-seed full-workload run\n"
            f"- Evaluator id: `{eval_id if eval_id is not None else 'unknown'}`\n\n"
            "## Result\n\n"
            "- Correctness: `PASS`\n"
            f"- Measured shapes: `{len(by_shape)}`\n"
            f"- Geomean latency: `{metric('latency_us_geomean')} us`\n"
            f"- Arithmetic mean latency: `{metric('latency_us_arith_mean')} us`\n"
            f"- Mean speedup vs metadata: `{metric('speedup_vs_ref_mean')}x`\n"
            f"- Performance score: `{metric('performance_score')}`\n"
            f"- Maximum absolute error: `{metric('max_abs_err')}`\n"
            f"- Maximum relative error: `{metric('max_rel_err')}`\n\n"
            f"{shape_note} Per-shape values are stored once in `memory/v0.json`; they are "
            "not duplicated here.\n"
        )
        path = self.workspace / "baseline_report.md"
        path.write_text(report, encoding="utf-8")
        return path

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
        """Pick fast/median/slow V0 latency ids for bounded native V1 smoke coverage."""
        memory = read_memory(self.workspace, 0) or {}
        performance = memory.get("performance")
        performance = performance if isinstance(performance, dict) else {}
        by_shape = performance.get("latency_us_by_shape", {})
        if not isinstance(by_shape, dict):
            return []
        measured: list[tuple[str, float]] = []
        for shape_id, raw_latency in by_shape.items():
            if isinstance(raw_latency, bool) or not isinstance(
                raw_latency, (int, float)
            ):
                continue
            latency = float(raw_latency)
            if latency > 0.0 and math.isfinite(latency):
                measured.append((str(shape_id), latency))
        if not measured or limit <= 0:
            return []
        measured.sort(key=lambda item: item[1])
        if len(measured) <= limit:
            return [shape_id for shape_id, _latency in measured]
        if limit == 1:
            return [measured[len(measured) // 2][0]]
        indexes = [
            round(index * (len(measured) - 1) / (limit - 1)) for index in range(limit)
        ]
        return [measured[index][0] for index in dict.fromkeys(indexes)]

    @staticmethod
    def ensure_framework_baseline_solution_manifest(self) -> None:
        """Preseed a native V1 manifest so the implementation Agent need not infer its schema."""
        path = self.workspace / "solution.json"
        if path.is_file() or not self.atrex_bench_root:
            return
        framework = self.framework.strip().lower()
        language = {
            "cutedsl": "cutedsl",
            "cuda": "cuda",
            "flydsl": "flydsl",
            "gluon": "gluon",
            "triton": "triton",
        }.get(framework, framework)
        dependencies = {
            "cuda": ["torch", "cuda-python"],
            "triton": ["torch", "triton"],
            "gluon": ["torch", "triton", "gluon"],
            "cutedsl": ["torch", "nvidia-cutlass-dsl"],
            "flydsl": ["torch", "flydsl"],
        }.get(framework, ["torch"])
        reference = self.workspace / "reference.py"
        reference_source = (
            reference.read_text(encoding="utf-8", errors="replace")
            if reference.is_file()
            else ""
        )
        entry_symbol = "Model" if "class Model" in reference_source else "run"
        manifest = {
            "name": f"{self.campaign_name}_v1_{language}",
            "spec": {
                "languages": ["python", language],
                "dependencies": dependencies,
                "entry_point": f"kernel.py::{entry_symbol}",
            },
            "sources": [
                {
                    "path": "kernel.py",
                    "role": (
                        "Single self-contained candidate file. Update this description and "
                        "dependency roles to match the implemented framework loader exactly."
                    ),
                }
            ],
            "dependency_roles": {
                dependency: (
                    "candidate framework/runtime plumbing; update with the exact non-compute "
                    "role used by kernel.py"
                )
                for dependency in dependencies
            },
        }
        path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

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
        self,
        v0_blob: str,
        baseline_commit: str,
        *,
        include_policy_review: bool = True,
    ) -> str:
        """Static acceptance checks on the candidate about to be validated and committed.

        Everything is judged from the worktree: that is what the gateway uploads, and it lets a
        session that wrote the kernel but never committed it still be accepted.
        """
        candidate_blob = git_worktree_blob(self.workspace, "kernel.py")
        if not candidate_blob or candidate_blob == v0_blob:
            return "the session left the V0 kernel unchanged; no framework implementation was produced"
        violations = (
            self._production_kernel_violations()
            if include_policy_review
            else production_structure_violations(self.workspace, self.framework)
        )
        if violations:
            return (
                f"the candidate is not a self-contained {self.framework} implementation: "
                + "; ".join(violations)
            )
        if self.framework.lower() in {"triton", "gluon"} and kernel_is_gluon(
            self.workspace
        ):
            # A Gluon v1 would permanently disarm the orchestrator's mandatory Triton->Gluon latch.
            return "the framework baseline must be plain Triton; Gluon is a later orchestrator escalation"
        mutated = self.candidate_workspace.baseline_changes(
            self.workspace, baseline_commit
        )
        if mutated:
            return "the session modified immutable ground truth: " + ", ".join(mutated)
        return ""

    @staticmethod
    def framework_baseline_external_gates(self, n: int) -> tuple[Optional[dict], str]:
        """Run policy review and the sole authoritative V1 evaluator concurrently."""
        print(
            "[orchestrator] framework baseline: running policy review and combined "
            "correctness/performance validation in parallel",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=2) as executor:
            policy_future = executor.submit(copy_context().run, self._production_kernel_violations)
            validation_future = executor.submit(copy_context().run, self._validate_framework_baseline, n)
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
        from orchestrator.numerical_policy import supplemental_feedback

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
        command = [
            "python",
            "test_kernel.py",
            "--version",
            f"v{n}",
            "--multi-seed",
            "5",
        ]
        # --timed-runs belongs to the native Atrex-Bench adapter. SOL and
        # derived legacy harnesses retain their own pinned benchmark settings.
        if self.atrex_bench_root:
            command += ["--timed-runs", "5"]
        command.append("--no-memory")
        try:
            test = _sandbox_command(
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
        if not result.get("all_pass"):
            return None, "combined multi-seed correctness validation failed"

        latency = result.get("latency_us_geomean")
        if not isinstance(latency, (int, float)) or latency <= 0:
            return None, "validation reported no usable latency_us_geomean"
        performance_score = result.get("performance_score")
        if (
            isinstance(performance_score, bool)
            or not isinstance(performance_score, (int, float))
            or performance_score <= 0
            or not math.isfinite(float(performance_score))
        ):
            return None, "validation reported no usable performance_score"
        # Require the framework baseline to preserve full-workload measurement coverage.
        baseline_shapes = set(
            ((read_memory(self.workspace, 0) or {}).get("performance") or {}).get(
                "latency_us_by_shape", {}
            )
        )
        measured_shapes = set(result.get("latency_us_by_shape") or {})
        if baseline_shapes and measured_shapes != baseline_shapes:
            return None, (
                "latency_us_by_shape does not cover the same workloads as v0 "
                f"(missing {sorted(baseline_shapes - measured_shapes)}, "
                f"unexpected {sorted(measured_shapes - baseline_shapes)})"
            )
        return result, ""
