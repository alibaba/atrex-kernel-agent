"""Workflow borrows explicitly assembled dependencies."""

from dataclasses import replace
from aka.contracts.workflow import CampaignRequest
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
        from .campaign import Campaign

        return Campaign(
            request, replace(self._dependencies, workflow=self._run_campaign)
        )

    def _run_campaign(self, campaign):
        self._check_open()
        return self._dependencies.workflow(campaign)
