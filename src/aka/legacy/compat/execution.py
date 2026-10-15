"""Historical context conversion at the Workflow execution seam."""
from aka.legacy.workflow.execution import CampaignExecution as _CampaignExecution


class CampaignExecution(_CampaignExecution):
    def __init__(self, context):
        super().__init__(getattr(context, "settings", context))
