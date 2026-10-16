"""Load existing kernel/native/SOL operators with their original validation rules."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aka.contracts.problem import LoadedProblem
from aka.contracts.problem import AgentProblem, InitialSource, EvaluationInput
from aka.contracts.content import WorkspaceFile
from aka.contracts.problem_provider import ProblemContent
from .layout import (
    AGENT_PROBLEM_FILENAME, find_atrex_bench_root, has_agent_problem, is_sol_op,
    should_use_generalized_problem, validate_agent_problem, validate_private_shapes,
)


@dataclass(frozen=True)
class LegacyProblemProvider:
    optimization_mode: str = "leaderboard"
    native_harness: Path | None = None
    worker_requirements: Path | None = None

    def load(self, directory: Path, *, optimization_mode=None, snapshot_directory: Path | None = None) -> LoadedProblem:
        """Preserve the original operator resolver's validation order and errors."""
        optimization_mode = optimization_mode or self.optimization_mode
        d = Path(directory).resolve()
        if snapshot_directory is not None and snapshot_directory.exists():
            from .snapshot import restore
            return restore(snapshot_directory.resolve(), d, optimization_mode)
        if not d.is_dir():
            raise SystemExit(f"--op-dir not found: {d}")
        ref = d / "reference.py"
        if not ref.is_file():
            raise SystemExit(f"--op-dir has no reference.py: {d}")
        atrex_bench_root = ""
        provided_problem = has_agent_problem(d)
        shapes_path = d / "shapes.json"
        generalized = should_use_generalized_problem(d, optimization_mode)
        if generalized and provided_problem:
            if not shapes_path.is_file():
                raise SystemExit(
                    "generalized Atrex-Bench operator requires private evaluator shapes.json: "
                    f"{d}"
                )
            try:
                validate_agent_problem(
                    d / AGENT_PROBLEM_FILENAME,
                    private_shapes_path=shapes_path,
                )
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
        if not is_sol_op(d) and shapes_path.is_file():
            try:
                validate_private_shapes(shapes_path)
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            native_root = find_atrex_bench_root(d)
            if native_root is None:
                raise SystemExit(
                    "native Atrex-Bench operator requires its canonical scripts/run_eval.py and "
                    f"src/atrex_bench runtime in an ancestor directory: {d}"
                )
            atrex_bench_root = str(native_root)
        sol = is_sol_op(d)
        from .layout import agent_visible_operator_files, LEGACY_ATREX_VISIBLE_FILES
        names = ("definition.json", "reference.py", "workload.jsonl") if sol else LEGACY_ATREX_VISIBLE_FILES + (AGENT_PROBLEM_FILENAME,)
        files = tuple(WorkspaceFile(name, source=d / name) for name in names if (d / name).is_file())
        public_names = names if sol else agent_visible_operator_files(d, generalized=generalized)
        public = tuple(item for item in files if item.path in public_names)
        initial = files if sol else (WorkspaceFile("kernel.py", source=ref, copy="cp"),)
        runtime = Path(atrex_bench_root) if atrex_bench_root else None
        from .resources import selected_resources
        harness, requirements = selected_resources(d, sol=sol, native=bool(runtime),
                                                   native_harness=self.native_harness,
                                                   worker_requirements=self.worker_requirements)
        problem = LoadedProblem(
            name=d.name, directory=d, reference=ref,
            bench_root=Path(atrex_bench_root) if atrex_bench_root else None,
            public_contract=d / AGENT_PROBLEM_FILENAME if generalized and provided_problem else None,
            public_contract_source="provided" if generalized and provided_problem else ("auto" if generalized else "none"),
            agent=AgentProblem(public), initial_source=InitialSource("sol" if sol else "kernel", ref, initial),
            evaluation=EvaluationInput("legacy-sol-v1" if sol else ("legacy-native-v1" if runtime else "legacy-kernel-v1"),
                                       d, files, runtime, d if generalized else None,
                                       tuple(str(x) for x in validate_private_shapes(shapes_path)) if generalized else (),
                                       harness, requirements),
        )
        if snapshot_directory is not None:
            from .snapshot import capture
            return capture(problem, snapshot_directory, d, optimization_mode)
        return problem

    def content(self, problem: LoadedProblem, workspace: Path, *, generated_digest: str = "") -> ProblemContent:
        from .evaluation import LegacyEvaluationAdapter
        from .resources import validate_resources
        if problem.snapshot_directory is not None:
            from .snapshot import inspect
            problem, _ = inspect(problem.snapshot_directory)
        validate_resources(problem.evaluation)
        if problem.evaluation is None or problem.agent is None:
            raise TypeError("Problem content requires explicit public and evaluation roles")
        return LegacyEvaluationAdapter(problem.evaluation.directory, workspace,
                                       "production" if problem.evaluation.private_reference_dir else self.optimization_mode,
                                       str(problem.evaluation.runtime_root or ""), generated_digest, problem)

    def snapshot(self, problem):
        from .snapshot import inspect
        if problem.snapshot_directory is None:
            raise ValueError("Problem must be frozen before obtaining its snapshot")
        return inspect(problem.snapshot_directory)[1]

    def export_snapshot(self, problem, destination):
        from .snapshot import capture, copy_snapshot, inspect
        if problem.snapshot_directory is None:
            problem = capture(problem, destination, problem.directory, self.optimization_mode)
            return inspect(problem.snapshot_directory)[1]
        return copy_snapshot(self.snapshot(problem), destination)[1]

    def validate_snapshot(self, snapshot, *, destination=None):
        from .snapshot import inspect
        problem = inspect(snapshot.directory, identity=snapshot.identity)[0]
        if destination is not None:
            target = Path(destination).absolute()
            if target.resolve() != target:
                raise ValueError("unsafe Problem snapshot destination")
            if target.exists():
                inspect(target, identity=snapshot.identity)
        return problem

    def restore_snapshot(self, snapshot, destination):
        from .snapshot import copy_snapshot
        return copy_snapshot(snapshot, destination)[0]
