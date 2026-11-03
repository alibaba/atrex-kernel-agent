"""Bridge public Task/Workflow requests to the retained managed implementation.

No global adapter is replaced. The selected Agent is borrowed and remains owned
by its composition; this Workflow owns only Campaigns and their private servers.
"""
from dataclasses import dataclass
from types import SimpleNamespace
from weakref import WeakSet

from aka.contracts.problem import LoadedProblem
from aka.contracts.workflow import CampaignRequest, ProblemSource
from .search import SearchOptions


@dataclass(frozen=True)
class WorkflowOptions:
    numerical_validation: bool = False
    goal_episodes: bool = False
    generate_memory: bool = False
    output_limit_recovery_retries: int = 2
    episode_timeout_seconds: float = 14400
    search: SearchOptions | dict | None = None
    wiki: bool = False
    wiki_options: dict | None = None
    research_records: bool = False
    inherit_state: bool = True
    request_timeout: float | None = None
    queue_timeout: float = 60

    def __post_init__(self):
        import math
        for name in ("wiki", "research_records", "inherit_state"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name in ("request_timeout", "queue_timeout"):
            value = getattr(self, name)
            if value is None and name == "request_timeout":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive finite seconds")
        from aka.resources.wiki.config import WikiConfig
        WikiConfig(**(self.wiki_options or {}))
        from .deadline import episode_timeout
        episode_timeout(self.episode_timeout_seconds)
        if (type(self.output_limit_recovery_retries) is not int
                or not 0 <= self.output_limit_recovery_retries <= 10):
            raise ValueError("output_limit_recovery_retries must be an integer from 0 to 10")
        if isinstance(self.search, dict):
            object.__setattr__(self, "search", SearchOptions(**self.search) if self.search else None)
        if self.search is not None and not isinstance(self.search, SearchOptions):
            raise ValueError("search must be an Epoch search configuration or empty")


class LightWorkflow:
    def __init__(self, agent, problem, source, candidate, *, executor, execution_context,
                 experiment_factory, evaluator_factory, evaluation_defaults,
                 measurement_parallelism=4, workflow_observer=None, options=WorkflowOptions()):
        if not agent.capabilities.tool_runtime:
            raise ValueError("light Workflow requires an Agent with scoped tool sessions")
        if any(not callable(getattr(candidate, method, None)) for method in (
                "read_source", "record", "restore_sealed_candidate",
                "snapshot_files", "restore_files", "clear_paths")):
            raise ValueError("light Workflow requires public source, sealed-candidate and file transactions")
        self.agent, self.problem, self.source, self.candidate = agent, problem, source, candidate
        self.options = options
        if not callable(getattr(evaluator_factory, "create_command", None)):
            raise TypeError("Selected Evaluation does not provide command preparation")
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
        from aka.execution.configuration import legacy_settings
        from aka.execution.selection import selected_execution

        connection = (legacy_settings()
                      if selected_execution().get("profile", "legacy") == "legacy"
                      else SimpleNamespace(profile="", url="", ssh="", ssh_init="",
                                           ssh_gpu=None, health_command=""))
        runtime = request.runtime
        return SimpleNamespace(
            hardware=runtime.hardware or request.settings.platform,
            profile=connection.profile,
            url=connection.url,
            timeout=runtime.timeout,
            verify_repeats=runtime.verify_repeats,
            verify_run_timeout=runtime.verify_run_timeout,
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
            raise ValueError("light Workflow deterministic V0 requires native Atrex-Bench or SOL; use legacy for custom setup")
        evaluation = self._evaluation_settings(request)
        campaign = ManagedCampaign.create(request, problem, self, evaluation)
        from aka.workflow.legacy.campaign_execution import CampaignExecution
        campaign.execution = CampaignExecution(evaluation)
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
