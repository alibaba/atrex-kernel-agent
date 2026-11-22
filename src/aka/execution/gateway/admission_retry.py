"""Bounded Agate recovery; explicit infrastructure errors always permit retry."""
from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping

from aka.contracts.execution_recovery import InfrastructureRetryPolicy

POLICY_ENV = "AKA_GATEWAY_ADMISSION_RETRY_POLICY"
RETRYABLE_HTTP = frozenset({429, 502, 503, 504})


def policy_from_environment(environment=None):
    raw = (os.environ if environment is None else environment).get(POLICY_ENV)
    if raw is None:
        return InfrastructureRetryPolicy()
    value = json.loads(raw)
    return InfrastructureRetryPolicy(tuple(value["delays"]), value["max_elapsed_s"])


def ambiguous_dev_submission(body, *, kind):
    """Only this explicit Dev failure permits a fresh, potentially duplicate Job."""
    if kind != "dev" or not isinstance(body, dict):
        return False
    if body.get("status") != "failed":
        return False
    error = body.get("error", body)
    return isinstance(error, dict) and (
        error.get("reason") == "dev_submission_ambiguous"
        or error.get("code") == "dev_submission_ambiguous"
    )


def infrastructure_failure(body):
    """Use the backend's explicit class, never words in stdout or stderr."""
    return (isinstance(body, Mapping) and isinstance(body.get("error"), Mapping)
            and body["error"].get("error_class") == "infra")


def retryable_job_failure(body, *, kind):
    if not isinstance(body, dict) or body.get("status") in {"cancelled", "canceled"}:
        return False
    # A GET failure says nothing about the accepted job's outcome. Do not
    # replay it under a new key when collection ends before a terminal state.
    if body.get("job_id") and body.get("status") not in {"failed", "succeeded"}:
        return False
    return infrastructure_failure(body) or ambiguous_dev_submission(body, kind=kind)


def fresh_key_required(body):
    """A confirmed failed Job needs a new physical submission.

    Without a Job ID retain the stable key, so admission uncertainty can still
    reconcile to the same submission even when the backend says never retry.
    """
    if not isinstance(body, dict):
        return False
    if body.get("job_id"):
        return True
    return any(fresh_key_required(body.get(key)) for key in ("error", "details"))


def retry_delay(error, index, policy):
    """Explicit infra errors override backend replay advice, within one budget."""
    if index >= len(policy.delays):
        return None
    try:
        body = json.loads(error.detail)
    except (TypeError, ValueError):
        body = None  # A proxy may return an HTML 503 rather than Agate JSON.

    def prohibited(value):
        if isinstance(value, dict):
            if (value.get("job_id") or value.get("execution_state") == "unknown"
                    or value.get("reason") == "dev_submission_ambiguous"
                    or value.get("code") in {"dev_submission_ambiguous", "replay-forbidden"}
                    or value.get("retry_advice") == "never"
                    or value.get("replay_policy") not in (None, "allow", "legacy")):
                return True
            return any(prohibited(item) for item in value.values())
        return isinstance(value, list) and any(prohibited(item) for item in value)

    if not infrastructure_failure(body) and (error.status not in RETRYABLE_HTTP or prohibited(body)):
        return None
    delay = policy.delays[index]
    retry_after = getattr(error, "retry_after_s", None)
    if type(retry_after) in (int, float) and math.isfinite(retry_after):
        delay = max(delay, retry_after)
    return delay
