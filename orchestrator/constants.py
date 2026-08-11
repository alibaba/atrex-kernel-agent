"""Shared constants for the optimization orchestrator.

Repository paths, policy defaults, and workspace state filenames used by campaigns and sessions.
"""
from __future__ import annotations

from pathlib import Path

from . import agent_runtime as _agent_runtime

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
WORKSPACE_INIT = REPO_ROOT / "reference" / "workspace_init.sh"
SOL_SEED = REPO_ROOT / "reference" / "sol_seed.py"
ATREX_BENCH_HARNESS = REPO_ROOT / "reference" / "atrex_bench_test_kernel.py"
PROFILE_DRIVER = REPO_ROOT / "reference" / "profile_driver.py"
SANDBOX_TOOL = REPO_ROOT / "tools" / "sandbox.py"
SANDBOX_DIRECTIVE_PROMPT = PROMPTS_DIR / "sandbox_directive.md"
HUMANIZE_DIR = REPO_ROOT / "3rdparty" / "humanize"
CONVERT_PERF_TOL = 0.05   # triton->gluon is a direct translation: gluon must be within +5% of triton
DEFAULT_CONVERT_AFTER = 3     # mandatory Triton->Gluon escalation after three consecutive stalls
DEFAULT_HANDOFF_RESUMES = 2
DEFAULT_VERIFY_REPEATS = 2
DEFAULT_VERIFY_RUN_TIMEOUT = 120
FRAMEWORK_BASELINE_FILE = "framework_baseline.json"
FRAMEWORK_BASELINE_VERSION = 1     # the framework baseline always occupies v1, retries overwrite it
FRAMEWORK_BASELINE_TIMEOUT_S = 10800
FRAMEWORK_BASELINE_MODES = ("auto", "always", "never")
FRAMEWORK_BASELINE_CATEGORY = "framework_baseline"
DEPENDENCY_REVIEW_SCHEMA_VERSION = 1
DEPENDENCY_REVIEW_TIMEOUT_S = 600
DEPENDENCY_REVIEW_PROMPT = PROMPTS_DIR / "dependency_review.md"
IMMUTABLE_BASELINE_PATHS = (
    "test_kernel.py", "reference.py", "input.py", "shapes.json", "metadata.json",
    "roofline.json", "workload.jsonl", "definition.json", "valid.py",
    # The profiling entry lives outside kernel.py so no candidate rewrite can silently
    # remove the ability to profile; it is ground truth like the evaluator harness.
    "profile_driver.py",
    "memory/v0.json",
)
TEST_RESULT_PREFIX = "[test_kernel] RESULT_JSON="
AGENT_CLI_CHOICES = _agent_runtime.SUPPORTED_RUNTIME_IDS
NVIDIA_FRAMEWORKS = ("Triton", "CuteDSL", "Cuda")
AMD_FRAMEWORKS = ("Triton", "FlyDSL")
DEFAULT_FRAMEWORKS = ("Triton",)
DEFAULT_SANDBOX_TIMEOUT = 600
MAX_SANDBOX_TIMEOUT = 600


STALL_STATE_FILE = ".orchestrator_state.json"
