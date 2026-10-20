"""Bridge public Task/Workflow requests to the retained managed implementation.

No global adapter is replaced. The selected Agent is borrowed and remains owned
by its composition; this Workflow owns only Campaigns and their private servers.
"""
from dataclasses import dataclass
from types import SimpleNamespace
from weakref import WeakSet

from aka.contracts.problem import LoadedProblem
from aka.legacy.compat.startup import prepare_request
from aka.legacy.compat.configuration import current_settings
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
    def __init__(self, agent, problem, source, candidate, tool_runtime, options=WorkflowOptions()):
        if not getattr(agent, "supports_tool_runtime", False):
            raise ValueError("light Workflow requires aka.light.agent.plugin (sandbox may remain none)")
        if not getattr(candidate, "managed_drafts", False):
            raise ValueError("light Workflow requires aka.light.workspace.plugin for sealed candidates")
        self.agent, self.problem, self.source, self.candidate = agent, problem, source, candidate
        self.options = options
        self.tool_runtime = tool_runtime
        self._closed = False
        self._campaigns = WeakSet()
        # Historical Application construction has one prepared-Problem entry.
        self._dependencies = SimpleNamespace(prepare_request=prepare_request)

    def campaign(self, request):
        if self._closed:
            raise RuntimeError("Workflow selection has been closed")
        problem = self.problem.load(request.problem.directory,
            optimization_mode=request.settings.optimization_mode,
            snapshot_directory=request.problem.snapshot_directory)
        return self._campaign(prepare_request(request, problem), problem)

    def _campaign(self, request, problem):
        from .campaign import ManagedCampaign
        if self._closed:
            raise RuntimeError("Workflow selection has been closed")
        if not isinstance(problem, LoadedProblem) or problem.agent is None or problem.initial_source is None or problem.evaluation is None:
            raise TypeError("light Workflow requires explicit public, initial-source and evaluator Problem roles")
        if problem.initial_source.kind != "sol" and problem.evaluation.runtime_root is None:
            raise ValueError("light Workflow deterministic V0 requires native Atrex-Bench or SOL; use legacy for custom setup")
        campaign = ManagedCampaign.create(request, problem, self, current_settings())
        self._campaigns.add(campaign)
        return campaign

    def close(self):
        self._closed = True
        for campaign in self._campaigns:
            campaign.close_runtime()
        self._campaigns.clear()
