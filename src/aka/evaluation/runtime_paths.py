"""Evaluation-owned runtime locations offered to the host explicitly."""
from aka.contracts.workspace import WorkspaceExclusion

VERIFY_DIR = "verification_artifacts/.atrex_long_horizon_verify"
RUNTIME_EXCLUSIONS = (WorkspaceExclusion("atrex-bench", subtree=True),)
