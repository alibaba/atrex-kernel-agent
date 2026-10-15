"""Resources paths implementation and callable interface."""
from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from aka.legacy.resources.checkout import checkout_root


REPO_ROOT = checkout_root()


PROMPTS_DIR = Path(str(files("aka.legacy.workflow").joinpath("prompts")))
RESOURCE_PROMPTS_DIR = Path(str(files("aka.legacy.resources").joinpath("prompts")))
PROBLEM_PROMPTS_DIR = Path(str(files("aka.legacy.task.problem").joinpath("prompts")))


WORKSPACE_INIT = REPO_ROOT / "reference" / "workspace_init.sh"




ATREX_BENCH_HARNESS = REPO_ROOT / "reference" / "atrex_bench_test_kernel.py"


PROFILE_DRIVER = REPO_ROOT / "reference" / "profile_driver.py"


SANDBOX_TOOL = REPO_ROOT / "tools" / "sandbox.py"


SANDBOX_SAFETY_BOUNDARY_PROMPT = RESOURCE_PROMPTS_DIR / "sandbox_safety_boundary.md"


SANDBOX_FULL_WORKFLOW_PROMPT = RESOURCE_PROMPTS_DIR / "sandbox_full_workflow.md"


DEPENDENCY_REVIEW_PROMPT = PROMPTS_DIR / "dependency_review.md"


AGENT_PROBLEM_GENERATION_PROMPT = PROBLEM_PROMPTS_DIR / "generalize_agent_problem.md"
