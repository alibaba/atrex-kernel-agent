"""Resource telemetry paths contributed to the selected Workspace."""
from aka.contracts.workspace import WorkspaceExclusion


RUNTIME_EXCLUSIONS = (
    WorkspaceExclusion(".gpu_wiki_profile", subtree=True),
    WorkspaceExclusion("trace-retention-manifest.json"),
)
