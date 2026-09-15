"""Error reasons shared by the local scheduler and trusted Gateway consumer."""

# These failures do not contain an authoritative Candidate verdict. In particular,
# command_timeout is the whole-job deadline, not the evaluator's Candidate timeout.
# Keep command_failed/profiler_failed, output limits and request validation separate:
# those can be caused by submitted code or arguments and must not be blindly retried.
LOCAL_INFRASTRUCTURE_REASONS = frozenset({
    "scheduler_stopped",
    "scheduler_restarted",
    "command_timeout",
    "execution_error",
    "invalid_eval_result",
    "evaluator_failed",
    "invalid_diagnostic_result",
    "diagnostic_failed",
})
