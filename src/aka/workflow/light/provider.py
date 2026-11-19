"""Light orchestrates shared owners around persistent Agent workspaces."""
from dataclasses import dataclass
from types import SimpleNamespace
from weakref import WeakSet
import math

from aka.contracts.problem import LoadedProblem
from aka.contracts.workflow import CampaignRequest, ProblemSource
from .search import SearchOptions


@dataclass(frozen=True)
class WorkflowOptions:
    file_system: str | None = None
    source_tree: dict | None = None
    provided_bootstrap: dict | None = None
    goal_episodes: bool = False
    output_limit_recovery_retries: int = 2
    dependency_policy_recovery_retries: int = 2
    search: SearchOptions | dict | None = None
    claim_review: bool = False
    wiki: bool = True
    wiki_options: dict | None = None
    tool_modules: tuple[str, ...] | list[str] = ()
    inherit_state: bool = True
    request_timeout: float | None = None
    queue_timeout: float = 60

    @property
    def research_records(self):
        return bool(self.tool_modules)

    def __post_init__(self):
        from .provided_bootstrap import validate_permission
        validate_permission(self.provided_bootstrap)
        from .journal import JournalFactory
        object.__setattr__(self, "tool_modules", JournalFactory(self.tool_modules).tool_modules)
        if self.file_system is not None:
            from aka.task.source.formats import select
            select(self.file_system)
        for name in ("wiki", "inherit_state", "goal_episodes", "claim_review"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name in ("request_timeout", "queue_timeout"):
            value = getattr(self, name)
            if value is None and name == "request_timeout":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive finite seconds")
        for name in ("output_limit_recovery_retries", "dependency_policy_recovery_retries"):
            if type(getattr(self, name)) is not int or not 0 <= getattr(self, name) <= 10:
                raise ValueError(f"{name} must be an integer from 0 to 10")
        if self.source_tree is not None:
            from aka.task.source.formats.source_tree import validate_import_options
            validate_import_options(self.source_tree)
        if isinstance(self.search, dict):
            object.__setattr__(self, "search", SearchOptions(**self.search) if self.search else None)
        if self.search is not None and not isinstance(self.search, SearchOptions):
            raise ValueError("search must be a pooled-search configuration or empty")


class LightWorkflow:
    def __init__(self, agent, problem, source, candidate, *, executor, execution_context,
                 experiment_factory, evaluator_factory, evaluation_defaults,
                 measurement_parallelism=4, workflow_observer=None, options=WorkflowOptions()):
        if not agent.capabilities.tool_runtime:
            raise ValueError("Light requires an Agent with scoped tool sessions")
        for method in ("read_source", "read_revision", "initialize_agent", "bind_source_contract"):
            if not callable(getattr(candidate, method, None)):
                raise TypeError(f"Light requires Task capability: {method}")
        if not callable(getattr(evaluator_factory, "create_command", None)):
            raise TypeError("Selected Evaluation does not provide command preparation")
        self.agent, self.problem, self.source, self.candidate = agent, problem, source, candidate
        self.options = options
        self.executor, self.execution_context = executor, execution_context
        self.experiment_factory, self.evaluator_factory = experiment_factory, evaluator_factory
        self.evaluation_defaults = evaluation_defaults
        self.measurement_parallelism = measurement_parallelism
        self.workflow_observer = workflow_observer
        self._closed = False
        self._campaigns = WeakSet()

    def _check_open(self):
        if self._closed:
            raise RuntimeError("Workflow selection has been closed")

    def _load_problem(self, request: CampaignRequest) -> LoadedProblem:
        self._check_open()
        snapshot = request.problem.snapshot_directory
        if snapshot is None:
            import hashlib
            identity = hashlib.sha256((str(request.workspace.absolute()) + "\0" +
                str(request.problem.directory.absolute())).encode()).hexdigest()
            snapshot = request.workspace.parent / ".aka-problems" / identity
        if not snapshot.exists() and self.candidate.current_revision(request.workspace):
            raise ValueError("An existing campaign requires its retained Task inputs; use a new workspace")
        problem = self.problem.load(
            request.problem.directory,
            optimization_mode=request.settings.optimization_mode,
            snapshot_directory=snapshot,
            expose_evaluator_shapes=request.settings.expose_evaluator_shapes,
        )
        if not isinstance(problem, LoadedProblem):
            raise TypeError("ProblemProvider.load must return LoadedProblem")
        if problem.initial_source is None or problem.evaluation is None:
            raise TypeError(
                "light Workflow requires explicit Problem initial-source and evaluation roles"
            )
        return problem

    def campaign(self, request: CampaignRequest):
        self._check_open()
        if not isinstance(request.problem, ProblemSource):
            raise TypeError("CampaignRequest.problem must be a ProblemSource")
        return self._campaign(request, self._load_problem(request))

    @staticmethod
    def _evaluation_settings(request):
        from aka.execution.configuration import connection_settings
        from aka.execution.selection import selected_execution

        connection = (connection_settings()
                      if selected_execution().get("profile", "remote") == "remote"
                      else SimpleNamespace(profile="", url="", ssh="", ssh_init="",
                                           ssh_gpu=None, health_command=""))
        runtime = request.runtime
        hardware = runtime.hardware or request.settings.platform
        return SimpleNamespace(
            hardware=hardware,
            profile=connection.profile,
            url=connection.url,
            timeout=runtime.timeout,
            verify_repeats=runtime.verify_repeats,
            evaluation_step_timeout=runtime.evaluation_step_timeout,
            shape_batch_size=runtime.shape_batch_size,
            gpu_clock_mhz=None,
            min_improvement_pct=runtime.min_improvement_pct,
            ssh=connection.ssh,
            ssh_init=connection.ssh_init,
            ssh_gpu=getattr(connection, "ssh_gpu", None),
            health_command=connection.health_command,
        )

    def _campaign(self, request, problem):
        from .campaign import ManagedCampaign
        self._check_open()
        if not isinstance(problem, LoadedProblem) or problem.agent is None or problem.initial_source is None or problem.evaluation is None:
            raise TypeError("light Workflow requires explicit public, initial-source and evaluator Problem roles")
        if problem.initial_source.kind != "sol" and problem.evaluation.runtime_root is None:
            raise ValueError("Light requires a Native Atrex-Bench or SOL evaluation contract")
        evaluation = self._evaluation_settings(request)
        campaign = ManagedCampaign.create(request, problem, self, evaluation)
        if self.workflow_observer is not None:
            try:
                self.workflow_observer.campaign_started(request)
            except Exception as error:
                import sys
                print(f"[workflow] auxiliary observer failed: {type(error).__name__}",
                      file=sys.stderr)
        self._campaigns.add(campaign)
        return campaign

    def close(self):
        self._closed = True
        for campaign in self._campaigns:
            campaign.close_runtime()
        self._campaigns.clear()
