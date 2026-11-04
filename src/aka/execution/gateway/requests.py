"""Map prepared requests to the selected gateway operation and actual budgets."""
from __future__ import annotations

from dataclasses import asdict
import json

from aka.contracts.execution import ExecutionBudget, UnsupportedExecution
from aka.contracts.execution_environment import environment_document
from aka.execution.outcomes import emit_fact

LIMITS_ENV = "AKA_EXECUTION_ENVIRONMENT_LIMITS"
NATIVE_ABBA = "same_allocation_abba"
MAX_TYPED_GATEWAY_JOB_TIMEOUT = 10_800


def typed_gateway_job_timeout(command_timeout, queue_wait_grace=None, *, limit=None):
    """Reject typed-job deadlines that cannot contain the requested allocation."""
    if type(command_timeout) is not int or command_timeout <= 0:
        raise ValueError("Gateway command timeout must be a positive integer")
    if queue_wait_grace is not None and (type(queue_wait_grace) is not int or queue_wait_grace < 0):
        raise ValueError("Gateway queue wait grace must be a non-negative integer")
    maximum = min(MAX_TYPED_GATEWAY_JOB_TIMEOUT, limit) if limit is not None else MAX_TYPED_GATEWAY_JOB_TIMEOUT
    requested = command_timeout
    if requested > maximum:
        raise ValueError(
            f"Gateway allocation exceeds the {maximum}-second typed-job limit: "
            f"attempt timeout {command_timeout}s")
    return requested


def validate_native_abba(request):
    """Validate the typed Eval extension without interpreting measurement output."""
    abba = request.get("abba") if isinstance(request, dict) else None
    if not isinstance(abba, dict) or set(abba) != {"baseline", "repeats"}:
        raise ValueError("same-allocation ABBA requires baseline and SDK block repetitions")
    baseline = abba["baseline"]
    if isinstance(baseline, str):
        valid_baseline = bool(baseline.strip())
    else:
        from .source_archives import source_descriptor
        valid_baseline = source_descriptor(baseline)
    if not valid_baseline:
        raise ValueError("same-allocation ABBA baseline must be nonempty source or an archive descriptor")
    if type(abba["repeats"]) is not int or abba["repeats"] < 1:
        raise ValueError("same-allocation ABBA SDK block repetitions must be positive")


def prepared_operation(prepared, selected):
    """Bind a logical operation to its immutable typed envelope."""
    if (not isinstance(prepared, dict) or prepared.get("schema_version") != 1
            or prepared.get("kind") not in {"run", "profile", "check", "disassemble"}):
        raise ValueError("unsupported prepared gateway request")
    kind = prepared["kind"]
    request = prepared.get("request")
    if not isinstance(request, dict):
        raise ValueError("prepared gateway request must contain a request object")
    if selected == NATIVE_ABBA:
        if kind != "run":
            raise ValueError("same-allocation ABBA requires a typed Eval request")
        validate_native_abba(request)
    else:
        if selected not in {"auto", kind}:
            raise ValueError("prepared request operation differs from selected operation")
        if "abba" in request:
            raise ValueError("ABBA requests require the same_allocation_abba operation")
    return kind


def operation_request(request, kind):
    """Return an independent wire request; Eval-only controls stay on Eval."""
    value = json.loads(json.dumps(request))
    if kind == NATIVE_ABBA:
        validate_native_abba(value)
        return value
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


def wait_budget_records(timeout, *, scope="wait", emit=True):
    """An absent client deadline must not be reported as a 600-second fallback."""
    return () if timeout is None else (effective_budget(timeout, scope=scope, emit=emit),)


def bound_limits(environment, support):
    """Never accept caller-injected limit authority over the selected backend."""
    environment = dict(environment)
    limits = support.environment.declarations.get("limits", {})
    environment[LIMITS_ENV] = json.dumps(environment_document(limits), sort_keys=True)
    return environment
