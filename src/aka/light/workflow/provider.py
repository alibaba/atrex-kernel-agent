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
    numerical_validation: bool = True
    goal_episodes: bool = False
    generate_memory: bool = False
    output_limit_recovery_retries: int = 2
    episode_timeout_seconds: float = 28800
    search: SearchOptions | dict | None = None
    source_tree: dict | None = None

    def __post_init__(self):
        from aka.light.agent.execution_clock import validate_timeout
        validate_timeout(self.episode_timeout_seconds)
        if (type(self.output_limit_recovery_retries) is not int
                or not 0 <= self.output_limit_recovery_retries <= 10):
            raise ValueError("output_limit_recovery_retries must be an integer from 0 to 10")
        if self.source_tree is not None:
            from aka.light.workspace.sources import validate_import_options
            validate_import_options(self.source_tree)
        if isinstance(self.search, dict):
            object.__setattr__(self, "search", SearchOptions(**self.search) if self.search else None)
        if self.search is not None and not isinstance(self.search, SearchOptions):
            raise ValueError("search must be an Epoch search configuration or empty")


class LightWorkflow:
    def __init__(self, agent, problem, source, candidate, tool_runtime,
                 *, workflow_observer=None, agent_resources=None, options=WorkflowOptions()):
        if not getattr(agent, "supports_tool_runtime", False):
            raise ValueError("light Workflow requires aka.light.agent.plugin (sandbox may remain none)")
        if not getattr(candidate, "managed_drafts", False):
            raise ValueError("light Workflow requires aka.light.workspace.plugin for sealed candidates")
        self.agent, self.problem, self.source, self.candidate = agent, problem, source, candidate
        self.options = options
        self.tool_runtime = tool_runtime
        self.workflow_observer = workflow_observer
        self._agent_resources = agent_resources
        self._closed = False
        self._campaigns = WeakSet()

    def _check_open(self):
        if self._closed:
            raise RuntimeError("Workflow selection has been closed")

    def _load_problem(self, request: CampaignRequest) -> LoadedProblem:
        self._check_open()
        problem = self.problem.load(
            request.problem.directory,
            optimization_mode=request.settings.optimization_mode,
            snapshot_directory=request.problem.snapshot_directory,
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

    def _resources_for(self, request):
        from aka.resources.binding import bind_campaign_resources
        return bind_campaign_resources(
            self._agent_resources, request.workspace, request.name
        )

    @staticmethod
    def _evaluation_settings(request):
        from aka.legacy.execution.configuration import legacy_settings
        from aka.legacy.execution.selection import selected_execution

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
        resources = self._resources_for(request)
        campaign = ManagedCampaign.create(request, problem, self, evaluation)
        campaign._agent_resources = resources
        from aka.legacy.execution.campaign_runtime import CampaignExecution
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
