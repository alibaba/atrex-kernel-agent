"""Agent-owned transient workspace paths contributed to the selected Workspace."""
from aka.contracts.workspace import WorkspaceExclusion


RUNTIME_EXCLUSIONS = (
    WorkspaceExclusion(".aka-agent-observations", subtree=True),
    WorkspaceExclusion(".aka-agent-sessions", subtree=True),
    WorkspaceExclusion(".aka-agent-session-errors", subtree=True),
)
