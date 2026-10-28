"""Map prepared requests to the selected gateway operation and actual budgets."""
from __future__ import annotations

from dataclasses import asdict
import json

from aka.contracts.execution import ExecutionBudget, UnsupportedExecution
from aka.contracts.execution_environment import environment_document
from aka.execution.outcomes import emit_fact

LIMITS_ENV = "AKA_EXECUTION_ENVIRONMENT_LIMITS"


def operation_request(request, kind):
    """Return an independent wire request; Eval-only controls stay on Eval."""
    value = json.loads(json.dumps(request))
    if kind == "run":
        return value
    common = {"name", "spec", "candidate", "env_vars", "requirements", "deps_mode"}
    if kind == "profile":
        allowed = common | {"reference", "options", "level", "profiler", "counters", "kernel_regex",
                            "kernel_name", "top_kernels", "source", "launch_skip", "launch_count"}
        value = {key: item for key, item in value.items() if key in allowed}
        value["options"] = {key: item for key, item in value.get("options", {}).items()
                            if key in {"bench_iters", "timeout_s"}}
        return value
    if kind not in {"check", "disassemble"}:
        raise ValueError("unsupported prepared gateway operation")
    if value.get("spec", {}).get("num_gpus", 1) != 1:
        raise UnsupportedExecution("selected compile/disassemble adapter supports one GPU")
    allowed = common | {"init_kwargs", "timeout_s"} | ({"arch", "sanitize"} if kind == "check" else {"fmt"})
    return {key: item for key, item in value.items() if key in allowed}


def hard_limit(environment, kind):
    try:
        limits = json.loads(environment.get(LIMITS_ENV, "{}"))
    except (ValueError, TypeError):
        raise ValueError("invalid frozen backend limits") from None
    if not isinstance(limits, dict):
        raise ValueError("frozen backend limits must be an object")
    keys = ("max_job_timeout_s_dev", "dev_timeout_s") if kind == "dev" else ("max_job_timeout_s", "job_timeout_s")
    value = next((limits[key] for key in keys if key in limits), None)
    if value is not None and (type(value) is not int or value < 1):
        raise ValueError("backend job hard limit must be positive")
    return value


def effective_budget(requested, *, scope="job", fallback=600, limit=None, emit=True):
    value = fallback if requested is None else requested
    if type(value) is not int or value < 1:
        raise ValueError("execution budget must be positive")
    effective = min(value, limit) if limit is not None else value
    budget = ExecutionBudget(scope, requested, effective, limit,
                             "backend-hard-limit" if effective < value else "fallback" if requested is None else "requested")
    if emit:
        emit_fact({"event": "budget", **asdict(budget)})
    return budget


def bound_limits(environment, support):
    """Never accept caller-injected limit authority over the selected backend."""
    environment = dict(environment)
    limits = support.environment.declarations.get("limits", {})
    environment[LIMITS_ENV] = json.dumps(environment_document(limits), sort_keys=True)
    return environment
