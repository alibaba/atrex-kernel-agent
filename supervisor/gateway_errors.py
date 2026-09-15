"""Error reasons shared by the local scheduler and trusted Gateway consumer."""

# These failures do not contain an authoritative Candidate verdict.
# Keep command_failed/profiler_failed, output limits and request validation separate:
# those can be caused by submitted code or arguments and must not be blindly retried.
LOCAL_INFRASTRUCTURE_REASONS = frozenset({
    "scheduler_stopped",
    "scheduler_restarted",
    "execution_error",
    "invalid_eval_result",
    "evaluator_failed",
    "invalid_diagnostic_result",
    "diagnostic_failed",
})

# A whole-job deadline does not establish whether code or infrastructure caused
# the timeout. It is not the evaluator's Candidate timeout and has a separate,
# conservative retry allowance; it must not become a permanent Candidate verdict.
COMMAND_TIMEOUT_REASON = "command_timeout"
DEFAULT_COMMAND_TIMEOUT_RETRIES = 1
