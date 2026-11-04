"""Shared constants for the optimization orchestrator.

Repository paths, policy defaults, and workspace state filenames used by campaigns and sessions.
"""

from __future__ import annotations

from pathlib import Path
from aka.resources.bundles import resource_path
from aka.evaluation.resources import profile_driver_script
from aka.task.problem.resources import native_harness_script
from aka.workflow.legacy.review_rules.review import DEPENDENCY_REVIEW_SCHEMA_VERSION
from aka.contracts.evaluation import (
    DEFAULT_ABBA_REPEATS,
    DEFAULT_ABBA_RUN_TIMEOUT,
    derived_evaluation_timeout,
)
from aka.resources.bundles import resource_path as resource

from aka.agent import SUPPORTED_RUNTIME_IDS

REPO_ROOT = resource_path("")
PROMPTS_DIR = resource_path("prompts/research-episodes")
WORKSPACE_INIT = REPO_ROOT / "reference" / "workspace_init.sh"
SOL_SEED = REPO_ROOT / "reference" / "sol_seed.py"
ATREX_BENCH_HARNESS = native_harness_script()
SOL_HARNESS = resource("reference/test_kernel.py")
PROFILE_DRIVER = profile_driver_script()
SANDBOX_SAFETY_BOUNDARY_PROMPT = PROMPTS_DIR / "sandbox_safety_boundary.md"
SANDBOX_FULL_WORKFLOW_PROMPT = PROMPTS_DIR / "sandbox_full_workflow.md"
CONVERT_PERF_TOL = (
    0.05  # triton->gluon is a direct translation: gluon must be within +5% of triton
)
DEFAULT_CONVERT_AFTER = (
    3  # mandatory Triton->Gluon escalation after three consecutive stalls
)
SUPPLEMENTAL_REPAIR_PREFIX = "Supplemental numerical probes"
SUPPLEMENTAL_PENDING_PREFIX = "Supplemental validation is incomplete"
DEFAULT_HANDOFF_RESUMES = 2
DEFAULT_VERIFY_REPEATS = DEFAULT_ABBA_REPEATS
DEFAULT_VERIFY_RUN_TIMEOUT = DEFAULT_ABBA_RUN_TIMEOUT
FRAMEWORK_BASELINE_FILE = "framework_baseline.json"
FRAMEWORK_BASELINE_VERSION = (
    1  # the framework baseline always occupies v1, retries overwrite it
)
FRAMEWORK_BASELINE_TIMEOUT_S = 10800
FRAMEWORK_BASELINE_MODES = ("auto", "always", "never")
FRAMEWORK_BASELINE_CATEGORY = "framework_baseline"
DEPENDENCY_REVIEW_TIMEOUT_S = 600
DEPENDENCY_REVIEW_PROMPT = PROMPTS_DIR / "dependency_review.md"
AGENT_PROBLEM_GENERATION_PROMPT = PROMPTS_DIR / "generalize_agent_problem.md"
DEFAULT_PROBLEM_GENERATION_TIMEOUT_S = 1_800
ATREX_PRIVATE_REFERENCE_ENV = "ATREX_PRIVATE_REFERENCE_DIR"
IMMUTABLE_BASELINE_PATHS = (
    "test_kernel.py",
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
    # The profiling entry lives outside kernel.py so no candidate rewrite can silently
    # remove the ability to profile; it is ground truth like the evaluator harness.
    "profile_driver.py",
    "memory/v0.json",
    ".atrex_versions/v0.json",
)
TEST_RESULT_PREFIX = "[test_kernel] RESULT_JSON="
AGENT_CLI_CHOICES = SUPPORTED_RUNTIME_IDS
NVIDIA_FRAMEWORKS = ("Triton", "CuteDSL", "Cuda", "TileLang")
AMD_FRAMEWORKS = ("Triton", "FlyDSL", "TileLang")
# CuteDSL is omitted: that path has never been validated on PPU silicon.
PPU_FRAMEWORKS = ("Triton", "Cuda", "TileLang")
DEFAULT_FRAMEWORKS = ("Triton", "TileLang")
DEFAULT_SANDBOX_TIMEOUT = derived_evaluation_timeout()


STALL_STATE_FILE = ".orchestrator_state.json"
