"""Shared constants for the optimization orchestrator.

Repository paths, policy defaults, and workspace state filenames used by campaigns and sessions.
"""

from __future__ import annotations

from pathlib import Path
from aka.legacy.evaluation.resources import profile_driver_script
from aka.light.evaluation.private_harness import (
    NATIVE_EVALUATOR_HARNESS as ATREX_BENCH_HARNESS,
    SOL_EVALUATOR_HARNESS as SOL_HARNESS,
)
from aka.legacy.acceptance.review import (
    DEPENDENCY_REVIEW_SCHEMA_VERSION as DEPENDENCY_REVIEW_SCHEMA_VERSION,
)
from aka.contracts.evaluation import (
    DEFAULT_ABBA_REPEATS,
    DEFAULT_EVALUATION_STEP_TIMEOUT,
    derived_evaluation_timeout,
)
from aka.light.workspace.resource_catalog import resource

import aka.light.agent.runtime as _agent_runtime

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
WORKSPACE_INIT = resource("reference/workspace_init.sh")
SOL_SEED = resource("reference/sol_seed.py")
PROFILE_DRIVER = profile_driver_script()
SANDBOX_TOOL = REPO_ROOT / "evaluation" / "gateway.py"
SANDBOX_SAFETY_BOUNDARY_PROMPT = PROMPTS_DIR / "sandbox_safety_boundary.md"
SANDBOX_FULL_WORKFLOW_PROMPT = PROMPTS_DIR / "sandbox_full_workflow.md"
SUPPLEMENTAL_REPAIR_PREFIX = "Supplemental numerical probes"
SUPPLEMENTAL_PENDING_PREFIX = "Supplemental validation is incomplete"
DEFAULT_OUTPUT_LIMIT_RECOVERY_RETRIES = 5
DEFAULT_VERIFY_REPEATS = DEFAULT_ABBA_REPEATS
# Upstream v0.4 still names this Campaign field ``verify_run_timeout``.
# Keep the structural field while binding it to Light's one Runtime-owned
# per-Shape evaluation budget.
DEFAULT_VERIFY_RUN_TIMEOUT = DEFAULT_EVALUATION_STEP_TIMEOUT
DEPENDENCY_REVIEW_PROMPT = resource("prompts/dependency_review.md")
RESEARCH_CLAIM_REVIEW_PROMPT = resource("prompts/research_claim_review.md")
AGENT_PROBLEM_GENERATION_PROMPT = resource("prompts/generalize_agent_problem.md")
DEFAULT_PROBLEM_GENERATION_TIMEOUT_S = 1_800
ATREX_PRIVATE_REFERENCE_ENV = "ATREX_PRIVATE_REFERENCE_DIR"
IMMUTABLE_BASELINE_PATHS = (
    "WORKSPACE.md",
    "config.json",
    "reference.py",
    "input.py",
    "agent_problem.json",
    "shapes.json",
    "metadata.json",
    "roofline.json",
    "workload.jsonl",
    "definition.json",
    "valid.py",
)
AGENT_CLI_CHOICES = _agent_runtime.SUPPORTED_RUNTIME_IDS
NVIDIA_FRAMEWORKS = ("Triton", "CuteDSL", "Cuda", "TileLang")
AMD_FRAMEWORKS = ("Triton", "FlyDSL", "TileLang")
# CuteDSL is omitted: that path has never been validated on PPU silicon.
PPU_FRAMEWORKS = ("Triton", "Cuda", "TileLang")
DEFAULT_FRAMEWORKS = ("Triton", "TileLang")
DEFAULT_SANDBOX_TIMEOUT = derived_evaluation_timeout()


STALL_STATE_FILE = ".orchestrator_state.json"
