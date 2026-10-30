"""Load existing kernel/native/SOL operators with their original validation rules."""
from __future__ import annotations

from dataclasses import dataclass, replace
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
    source_tree: dict | None = None

    def __post_init__(self):
        if self.source_tree is not None:
            from aka.task.source.tree import validate_import_options
            options = validate_import_options(self.source_tree)
            object.__setattr__(self, "source_tree", {
                key: str(Path(value).expanduser().absolute()) for key, value in options.items()
            })

    def load(self, directory: Path, *, optimization_mode=None, snapshot_directory: Path | None = None) -> LoadedProblem:
        """Preserve the original operator resolver's validation order and errors."""
        optimization_mode = optimization_mode or self.optimization_mode
        d = Path(directory).resolve()
        if snapshot_directory is not None and snapshot_directory.exists():
            from .snapshot import restore
            return restore(snapshot_directory.resolve(), d, optimization_mode,
                           source_configuration=self.source_tree)
        if self.source_tree is not None and snapshot_directory is None:
            raise ValueError("Source-tree inputs require a retained Task snapshot")
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
            if self.source_tree is not None:
                if sol or runtime is None:
                    raise ValueError("Source trees require a native Atrex-Bench Task")
                from aka.task.source.tree_provider import TreeSourceProvider
                import tempfile
                content = TreeSourceProvider(self.source_tree).prepare()
                from aka.task.source.tree import POLICY, validate_policy
                import json
                declaration = next(item for item in content.files if item.path == POLICY)
                raw = declaration.source.read_bytes() if declaration.source is not None else declaration.text.encode()
                policy = validate_policy(json.loads(raw))
                additional = tuple(item["distribution"] + item.get("version", "")
                                   for item in policy["runtime_requirements"])
                if additional:
                    selected_requirements = problem.evaluation.worker_requirements
                    existing = (selected_requirements.source.read_text() if selected_requirements.source is not None
                                else selected_requirements.text)
                    lines = existing.splitlines()
                    added = tuple(value for value in additional if value not in lines)
                    if added:
                        problem = replace(problem, evaluation=replace(problem.evaluation,
                            worker_requirements=WorkspaceFile(selected_requirements.path,
                                text=existing.rstrip() + "\n" + "\n".join(added) + "\n")))
                # Retain pure source content through the existing path-based
                # snapshot roles, without changing the semantic reference.
                with tempfile.TemporaryDirectory(prefix="aka-task-source-") as raw:
                    root = Path(raw)
                    selected = []
                    for item in content.files:
                        path = root / item.path
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(item.source.read_bytes() if item.source is not None else item.text.encode("utf-8"))
                        selected.append(WorkspaceFile(item.path, source=path, copy=item.copy))
                    problem = replace(problem, initial_source=InitialSource("kernel", root / "kernel.py", tuple(selected)))
                    return capture(problem, snapshot_directory, d, optimization_mode,
                                   source_configuration=self.source_tree)
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

    def runtime_files(self, problem):
        if problem.snapshot_directory is None:
            raise ValueError("runtime files require a frozen Problem")
        from .snapshot import inspect
        frozen, _ = inspect(problem.snapshot_directory)
        runtime = frozen.evaluation.runtime_root
        if runtime is None:
            return ()
        return tuple(WorkspaceFile("atrex-bench/" + path.relative_to(runtime).as_posix(), source=path)
                     for path in sorted(runtime.rglob("*")) if path.is_file())

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
