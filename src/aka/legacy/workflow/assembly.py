"""Workflow borrows explicitly assembled dependencies."""

from dataclasses import replace
from aka.contracts.problem import LoadedProblem
from aka.contracts.workflow import CampaignRequest, ProblemSource
from .dependencies import CampaignDependencies


class LegacyWorkflow:
    def __init__(self, dependencies: CampaignDependencies):
        self._dependencies = dependencies
        self.agent = dependencies.agent
        self._closed = False

    def close(self):
        self._closed = True

    def _check_open(self):
        if self._closed:
            raise RuntimeError("Workflow selection has been closed")

    def campaign(self, request: CampaignRequest):
        self._check_open()
        if not isinstance(request.problem, ProblemSource):
            raise TypeError("CampaignRequest.problem must be a ProblemSource")
        problem = self._load_problem(request)
        return self._campaign(request, problem)

    def _campaign(self, request: CampaignRequest, problem: LoadedProblem):
        self._check_open()
        from .campaign import Campaign

        return Campaign(
            request, replace(self._dependencies, workflow=self._run_campaign,
                             agent_resources=self._resources_for(request)), problem
        )

    def _resources_for(self, request):
        """Business owns target binding; Bootstrap never decodes Workflow argv."""
        owner = self._dependencies.agent_resources
        if owner is None:
            return None  # Explicit legacy/in-process compatibility path.
        import json
        import os
        from aka.bootstrap.continuation import base_payload, read
        from aka.resources import restore_bound
        payload = read(os.environ)
        value = json.loads(payload) if payload else None
        if value and value.get("version") == 3:
            binding = value["agent_resources"]
            if (binding["workspace"] != str(request.workspace.resolve())
                    or binding["campaign_id"] != request.name):
                raise ValueError("campaign differs from its frozen agent resources")
            bound = restore_bound(binding, base_selection=json.loads(base_payload(payload)))
            # Reconstructed providers must describe the same frozen collection.
            if owner.freeze() != bound.resource_snapshot():
                raise ValueError("selected agent resource contributions changed")
            return bound
        return owner.bind(request.workspace, request.name, base_selection=value)

    def _load_problem(self, request: CampaignRequest) -> LoadedProblem:
        """Capture or restore run inputs before campaign-specific preparation."""
        self._check_open()
        problem = self._dependencies.problem.load(
            request.problem.directory,
            optimization_mode=request.settings.optimization_mode,
            snapshot_directory=request.problem.snapshot_directory,
        )
        if not isinstance(problem, LoadedProblem):
            raise TypeError("ProblemProvider.load must return LoadedProblem")
        if problem.initial_source is None or problem.evaluation is None:
            raise TypeError(
                "Workflow requires explicit Problem initial source and evaluation roles"
            )
        return problem

    def _run_campaign(self, campaign):
        self._check_open()
        return self._dependencies.workflow(campaign)
