"""Baseline preparation using explicitly supplied problem/configuration/resources."""

from __future__ import annotations
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from aka.contracts.problem import LoadedProblem
from aka.contracts.content import WorkspaceFile

AGENT_PROBLEM_FILENAME = "agent_problem.json"
from aka.contracts.workflow_records import CanonicalMeasurementReader, measurement_document


@dataclass(frozen=True)
class BaselineInputs:
    workspace: Path
    problem: LoadedProblem | None = None
    native: bool = False
    private_reference_dir: Path | None = None
    arch: str = ""
    campaign_name: str = ""
    framework: str = ""
    notes: str = ""
    optimization_mode: str = "leaderboard"
    platform: str = ""
    target_util: float = 0.0
    measurement_reader: CanonicalMeasurementReader | None = None


@dataclass(frozen=True)
class BaselineResources:
    native_harness: WorkspaceFile | None
    profile_driver: Path


class BaselinePreparation:
    def __init__(self, inputs: BaselineInputs, resources: BaselineResources):
        self.inputs = inputs
        self.resources = resources

    def evaluator_directive(self) -> str:
        if self.inputs.native:
            if self.inputs.private_reference_dir is not None:
                return (
                    "## Evaluation route: Atrex-Bench generalized private-case native\n\n"
                    "Treat workspace `agent_problem.json` as the authoritative public optimization "
                    "contract. Exact `shapes.json`, `metadata.json`, and `roofline.json` cases are "
                    "evaluator-only and intentionally absent from the workspace; never search for, "
                    "reconstruct, or read the private reference directory. The immutable "
                    "`test_kernel.py` adapter and sandbox inject those cases only into the remote "
                    "official evaluator. Optimize for the complete declared `shape_domain`, using "
                    "aggregate `distribution_profile` shares only for prioritization. Correctness "
                    "must pass every hidden case. The reported `performance_score` is the arithmetic mean "
                    "of each opaque shape's measured speedup against its authoritative Atrex-Bench "
                    "metadata production latency. After "
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
                "The reported `performance_score` is computed by dividing each shape's "
                "metadata `production_performance.performance_us` by measured latency, then taking "
                "the arithmetic mean across shapes. "
                "Do not edit or replace this adapter and do not implement a custom correctness or "
                "timing harness. `--multi-seed N` maps to N additional Atrex-Bench correctness "
                "cases while performance remains one official run per shape."
            )
        if self.inputs.problem.initial_source.kind == "sol":
            return (
                "## Evaluation route: SOL-ExecBench\n\n"
                "Keep using the immutable SOL `test_kernel.py`, which invokes `sol-execbench` over "
                "the complete `workload.jsonl`. Each workload's SOL reference is its performance "
                "baseline; `performance_score` reports the arithmetic mean of per-workload "
                "speedups. Do not substitute the Atrex-Bench native evaluator."
            )
        return (
            "## Evaluation route: derived legacy boundary\n\n"
            "This derived boundary is not a complete Atrex-Bench operator directory. Preserve its "
            "committed full-shape `test_kernel.py` methodology and do not replace it after V0."
        )

    def install_native_evaluator(self) -> None:
        """Seed the immutable adapter used only by native Atrex-Bench shape campaigns."""
        harness = self.resources.native_harness
        if harness is None and self.inputs.native:
            from .resources import native_harness_script
            harness = WorkspaceFile("test_kernel.py", source=native_harness_script())
        if harness is None:
            return
        from aka.task.candidate_workspace.content import install_files
        install_files(self.inputs.workspace, (harness,))

    def install_profile_driver(self) -> None:
        """Seed the immutable external profiling entry for every campaign layout.

        Both profiler wrappers run ``python <file>``, so profiling needs a runnable script.
        Keeping it out of ``kernel.py`` means a session that rewrites ``run()``/``Model``
        cannot silently destroy profiling: an in-kernel ``__main__`` block would vanish with
        the rewrite and the profiler would still exit 0 having captured nothing.
        """
        if not self.resources.profile_driver.is_file():
            raise FileNotFoundError(f"missing {self.resources.profile_driver}")
        shutil.copy2(
            self.resources.profile_driver, self.inputs.workspace / "profile_driver.py"
        )
        # Stage it when a repository already exists so the baseline commit tracks it without
        # depending on how the setup session stages files; restoring an immutable path needs
        # a blob in the root commit.
        if (self.inputs.workspace / ".git").exists():
            subprocess.run(
                ["git", "add", "profile_driver.py"],
                cwd=str(self.inputs.workspace),
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def native_v0_readme(self, *, generalized: bool) -> str:
        contract = (
            "`agent_problem.json` is the authoritative public contract. Exact evaluator "
            "cases remain private; latency-map keys are opaque ids."
            if generalized
            else "`shapes.json` and the other copied operator files define the public workload."
        )
        runtime_arch = (
            self.inputs.arch
            or "unknown (use the runtime GPU API before choosing codegen)"
        )
        notes = self.inputs.notes.strip() or "none"
        return (
            f"# kernel_opt_{self.inputs.campaign_name}\n\n"
            "Profile-driven optimization of one native Atrex-Bench operator.\n\n"
            "## Goal\n\n"
            "Record the arithmetic mean of per-shape speedups against Atrex-Bench metadata "
            "production performance and the full-workload GeoMean latency while every evaluator case remains "
            "correct. V0 uses the Problem's selected initial implementation; optimized versions must "
            f"migrate the computation to `{self.inputs.framework}`.\n\n"
            "## Campaign\n\n"
            f"- Target platform: `{self.inputs.platform}`\n"
            f"- Runtime architecture: `{runtime_arch}`\n"
            f"- Optimization mode: `{self.inputs.optimization_mode}`\n"
            f"- Target framework: `{self.inputs.framework}`\n"
            f"- Target utilization stop condition: `{self.inputs.target_util:g}%`\n"
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

    def framework_baseline_smoke_shape_ids(self, *, limit: int = 3) -> list[str]:
        """Pick fast/median/slow V0 latency ids for bounded native V1 smoke coverage."""
        reader = self.inputs.measurement_reader
        fact = reader.baseline(self.inputs.workspace) if reader is not None else None
        performance = measurement_document(fact.result) if fact is not None else {}
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
