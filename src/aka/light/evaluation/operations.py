"""Additional GPU operations, separate from measurement storage and acceptance.

The gateway argument is the running executor module's public API. Passing that
instance keeps job tracking and SDK cancellation shared with the module entry
point, which may be loaded as __main__ rather than supervisor.gateway.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from aka.contracts.evaluation import (
    ABBA_SCHEDULING_OVERHEAD,
    DEFAULT_EVALUATION_STEP_TIMEOUT,
    MAX_TYPED_GATEWAY_JOB_TIMEOUT,
)
from aka.light.evaluation.policy import validate_abba_repeats

from aka.light.evaluation.projection import (
    GATEWAY_ERROR_PREFIX,
    SOURCE_ERROR_PREFIX,
    candidate_source_rejection,
    emit_gateway_error,
    gateway_error_result,
)


def diagnostic(gateway, args, workspace: Path, queue_wait_grace: int) -> int:
    if args.command:
        raise ValueError("Check/Disassemble do not accept a command override")
    request = gateway.build_typed_request(
        workspace, args.hardware, args.timeout, args.env, [], args.kind,
        arch=args.arch, sanitize=args.sanitize, disassembly_format=args.disassembly_format,
        requirements=args.requirement or (), deps_mode=args.deps_mode,
        ssh=bool(args.ssh),
    )
    from aka.light.evaluation.agate_native import apply_request
    request = apply_request(request, args, args.kind)
    if args.dry_run:
        print(json.dumps({"kind": args.kind, "candidate_bytes": len(request["candidate"].encode())}))
        return 0
    if args.ssh:
        from aka.light.evaluation.ssh_transport import run
        process = run(gateway, args, workspace, request, args.kind)
    else:
        # All Agate Gateway diagnostics share the typed SDK lifecycle.  The
        # explicit SSH transport above is independent of Agate and does not
        # reintroduce CLI or direct-HTTP Gateway fallbacks.
        from aka.light.evaluation.source_transport import run as run_tree
        process = run_tree(gateway, args, request, args.kind, queue_wait_grace)
    source_error = candidate_source_rejection(process)
    if source_error is not None:
        print(SOURCE_ERROR_PREFIX + json.dumps(source_error))
        return process.returncode or 1
    job = gateway.parse_job_response(process.stdout or "")
    if not job or job.get("status") != "succeeded" or not isinstance(job.get("result"), dict):
        if not emit_gateway_error(process, args.kind):
            print(GATEWAY_ERROR_PREFIX + json.dumps(gateway_error_result({
                "operation": args.kind, "error": process.stderr or "Gateway returned no result", "outcome": "unknown",
            })))
        return process.returncode or 1
    prefix = "[sandbox] CHECK_JSON=" if args.kind == "check" else "[sandbox] DISASSEMBLE_JSON="
    print(prefix + json.dumps(job["result"]))
    return 0 if job["result"].get("passed", True) else 1


def validate_comparison(args) -> None:
    if args.kind != "run" or args.evaluation_mode == "correctness_only":
        raise ValueError("ABBA requires --kind run in full mode")
    validate_abba_repeats(args.comparison_repeats)
    if (args.command or args.evaluation_input_path or args.evaluation_shapes_path
            or args.shape_id):
        raise ValueError("ABBA uses the canonical full contract; command/input/shape overrides are unsupported")
    comparison_run_timeout(args)


def comparison_run_timeout(args) -> int:
    available = (
        args.timeout - ABBA_SCHEDULING_OVERHEAD
    ) // (2 * args.comparison_repeats)
    # Recorded acceptance emits the explicit Agent-visible override. Preserve
    # it exactly so the prompt/request identity matches the job actually run.
    seconds = getattr(args, "comparison_run_timeout", None)
    if seconds is not None:
        if type(seconds) is not int or seconds <= 0:
            raise ValueError("--comparison-run-timeout must be a positive integer")
        return seconds
    # The Campaign step timeout is an upper bound, not a promise that it fits
    # every ABBA schedule. Derive a safe implicit value when the Agent leaves
    # the option unset. An explicit value is authoritative and is never
    # clamped to this pre-existing allocation.
    seconds = min(
        getattr(
            args,
            "evaluation_step_timeout",
            DEFAULT_EVALUATION_STEP_TIMEOUT,
        ),
        available,
    )
    if seconds <= 0:
        raise ValueError(
            "The configured ABBA allocation leaves no positive per-run budget; "
            "increase the Campaign evaluation allocation"
        )
    return seconds


def comparison_allocation_timeout(args, per_run_timeout: int | None = None) -> int:
    """Return an allocation that honors an explicit Agent per-run timeout."""
    seconds = (
        comparison_run_timeout(args)
        if per_run_timeout is None
        else per_run_timeout
    )
    schedule_timeout = (
        2 * args.comparison_repeats * seconds + ABBA_SCHEDULING_OVERHEAD
    )
    allocation_timeout = max(args.timeout, schedule_timeout)
    if allocation_timeout > MAX_TYPED_GATEWAY_JOB_TIMEOUT:
        explicit = getattr(args, "comparison_run_timeout", None)
        if explicit is not None and schedule_timeout > MAX_TYPED_GATEWAY_JOB_TIMEOUT:
            raise ValueError(
                f"--comparison-run-timeout {explicit} seconds with "
                f"--comparison-repeats {args.comparison_repeats} requires a "
                f"{schedule_timeout}-second ABBA allocation "
                f"(2 * {args.comparison_repeats} * {explicit} + "
                f"{ABBA_SCHEDULING_OVERHEAD} seconds scheduling overhead), exceeding "
                f"Agate's {MAX_TYPED_GATEWAY_JOB_TIMEOUT}-second typed-job limit. "
                "Runtime did not clamp the supplied value and no Job was submitted; "
                "reduce --comparison-run-timeout or --comparison-repeats."
            )
        raise ValueError(
            f"The configured {allocation_timeout}-second ABBA allocation exceeds "
            f"Agate's {MAX_TYPED_GATEWAY_JOB_TIMEOUT}-second typed-job limit. "
            "Runtime did not clamp the allocation and no Job was submitted; reduce the "
            "Campaign evaluation timeout."
        )
    return allocation_timeout


def compare(gateway, args, workspace: Path, queue_wait_grace: int) -> int:
    from aka.light.evaluation.native_abba import compare as native_compare
    return native_compare(gateway, args, workspace, queue_wait_grace)
