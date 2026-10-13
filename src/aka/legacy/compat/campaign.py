"""Historical Campaign constructor and private evaluation forwarding."""

from aka.legacy.workflow.campaign import *
from aka.legacy.workflow.campaign import Campaign as WorkflowCampaign
from .workflow import campaign_request, evaluation_configuration
from .configuration import evaluation_settings
from long_horizon.legacy_evaluation.baseline import LegacyBaselineEvaluation


class Campaign(WorkflowCampaign):
    problem: LoadedProblem | None

    def __init__(self, name, kernel_demo, platform, framework, **options):
        from .assembly import default_workflow

        workflow = options.pop("workflow", None)
        task = options.pop("task_dependencies", None)
        self.kernel_demo = kernel_demo
        self.work_dir = options.get("work_dir", "")
        self.workspace_suffix = options.get("workspace_suffix", "")
        selected = (
            workflow
            if workflow is not None
            else default_workflow(
                task=task,
                optimization_mode=options.get("optimization_mode", "leaderboard"),
            )
        )
        configuration = dict(
            name=name,
            kernel_demo=kernel_demo,
            platform=platform,
            framework=framework,
            **options,
        )
        request = campaign_request(
            configuration,
        )
        problem = configuration.get("problem")
        if not isinstance(problem, LoadedProblem):
            problem = selected._load_problem(request)
        request = selected._dependencies.prepare_request(request, problem)
        with evaluation_settings(evaluation_configuration(options)):
            super().__init__(request, selected._dependencies, problem)

    def __getattr__(self, name):
        for prefix, policy in (
            ("v1", "framework_baseline_reviewers"),
            ("fast_episode", "fast_episode_reviewers"),
            ("full_episode", "full_episode_reviewers"),
        ):
            if name in (prefix + "_ask_codex", prefix + "_ask_qoder"):
                return name.rsplit("_", 1)[1] in getattr(self, policy, ())
        evaluation = self.__dict__.get("evaluation")
        if evaluation is not None and hasattr(evaluation, "context"):
            key = name.removeprefix("sandbox_") if name.startswith("sandbox_") else name
            if hasattr(evaluation.context.settings, key):
                return getattr(evaluation.context.settings, key)
        raise AttributeError(name)

    @property
    def campaign_name(self):
        if "request" in self.__dict__:
            return self.request.name
        suffix = getattr(self, "workspace_suffix", "")
        return self.name + ("_" + suffix if suffix else "")

    @property
    def workspace(self):
        if hasattr(self, "request"):
            return self.request.workspace
        from pathlib import Path

        return (Path(self.work_dir) if self.work_dir else Path.cwd()) / (
            "kernel_opt_" + self.name
        )

    def _bound_evaluation(self):
        from .workflow import BoundEvaluation, bind_evaluation

        evaluation = self.evaluation
        return (
            evaluation
            if isinstance(evaluation, BoundEvaluation)
            else bind_evaluation(evaluation, self)
        )

    @property
    def task_dependencies(self):
        if "_task_dependencies" in self.__dict__:
            return self._task_dependencies

        return dict(
            problem=self.problem_provider,
            source=self.source_provider,
            workspace=self.candidate_workspace,
        )

    @task_dependencies.setter
    def task_dependencies(self, value):
        self._task_dependencies = value

    def _evaluator_directive(self) -> str:
        return self._bound_evaluation().evaluator_directive()

    def _install_native_evaluator(self) -> None:
        return self._bound_evaluation().install_native_evaluator()

    def _install_profile_driver(self) -> None:
        return self._bound_evaluation().install_profile_driver()

    def _native_v0_readme(self, *, generalized: bool) -> str:
        return self._bound_evaluation().native_v0_readme(generalized=generalized)

    @staticmethod
    def _print_v0_evaluator_output(test):
        return LegacyBaselineEvaluation.print_v0_evaluator_output(test)

    def _run_v0_evaluator(self) -> dict:
        return self._bound_evaluation().run_v0_evaluator()

    def _write_v0_baseline_report(self, result: dict, source_commit: str) -> Path:
        return self._bound_evaluation().write_v0_baseline_report(result, source_commit)

    def _finalize_v0_measurement(
        self, result: dict, source_commit: str, *, extra_paths: tuple[str, ...] = ()
    ) -> None:
        return self._bound_evaluation().finalize_v0_measurement(
            result, source_commit, extra_paths=extra_paths
        )

    def _setup_baseline_native(self, op_dir: Path, *, generalized: bool) -> None:
        return self._bound_evaluation().setup_baseline_native(
            op_dir, generalized=generalized
        )

    def _setup_baseline_sol(self, op_dir: Path) -> None:
        return self._bound_evaluation().setup_baseline_sol(op_dir)

    def _framework_baseline_smoke_shape_ids(self, *, limit: int = 3) -> list[str]:
        return self._bound_evaluation().framework_baseline_smoke_shape_ids(limit=limit)

    def _ensure_framework_baseline_solution_manifest(self) -> None:
        return self._bound_evaluation().ensure_framework_baseline_solution_manifest()

    def _framework_baseline_smoke_command(self, n: int) -> tuple[str, str]:
        return self._bound_evaluation().framework_baseline_smoke_command(n)

    def _restore_immutable_baseline_paths(self, baseline_commit: str) -> list[str]:
        return self._bound_evaluation().restore_immutable_baseline_paths(
            baseline_commit
        )

    def _framework_baseline_problem(
        self, v0_blob: str, baseline_commit: str, *, include_policy_review: bool = True
    ) -> str:
        return self._bound_evaluation().framework_baseline_problem(
            v0_blob, baseline_commit, include_policy_review=include_policy_review
        )

    def _framework_baseline_external_gates(self, n: int) -> tuple[Optional[dict], str]:
        return self._bound_evaluation().framework_baseline_external_gates(n)

    def _supplemental_numerical_feedback(
        self,
        workspace: Path | None = None,
        *,
        standard_correctness_passed: bool = False,
    ) -> str:
        return self._bound_evaluation().supplemental_numerical_feedback(
            workspace, standard_correctness_passed=standard_correctness_passed
        )

    def _validate_framework_baseline(self, n: int) -> tuple[Optional[dict], str]:
        return self._bound_evaluation().validate_framework_baseline(n)
