"""Workflow constants implementation and callable interface."""
from __future__ import annotations

import aka.legacy.agent as _agent_runtime


from aka.legacy.acceptance.policy import CONVERT_PERF_TOL


DEFAULT_CONVERT_AFTER = (
    3  # mandatory Triton->Gluon escalation after three consecutive stalls
)


SUPPLEMENTAL_REPAIR_PREFIX = "Supplemental numerical probes"


SUPPLEMENTAL_PENDING_PREFIX = "Supplemental validation is incomplete"


DEFAULT_HANDOFF_RESUMES = 2


DEFAULT_FAST_EPISODES = 2


DEFAULT_FAST_TRIALS = 5


DEFAULT_VERIFY_REPEATS = 2


DEFAULT_VERIFY_RUN_TIMEOUT = 120


FRAMEWORK_BASELINE_VERSION = (
    1  # the framework baseline always occupies v1, retries overwrite it
)


FRAMEWORK_BASELINE_TIMEOUT_S = 10800


FRAMEWORK_BASELINE_MODES = ("auto", "always", "never")


FRAMEWORK_BASELINE_CATEGORY = "framework_baseline"


from aka.legacy.acceptance.review import DEPENDENCY_REVIEW_SCHEMA_VERSION


DEPENDENCY_REVIEW_TIMEOUT_S = 600


ATREX_PRIVATE_REFERENCE_ENV = "ATREX_PRIVATE_REFERENCE_DIR"


TEST_RESULT_PREFIX = "[test_kernel] RESULT_JSON="


AGENT_CLI_CHOICES = _agent_runtime.SUPPORTED_RUNTIME_IDS
