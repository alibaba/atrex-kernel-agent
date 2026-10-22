"""Legacy CLI composition: Evaluation policy over Execution transport primitives.

This adapter is used only by the retained CLI; Experiment's prepared-evaluation
chain continues to own its input/output persistence through the shared contracts.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
from aka.legacy.execution.gateway import sandbox as transport
from aka.legacy.execution.outcomes import emit_fact, emit_diagnostic
from aka.legacy.evaluation.gateway_request import prepare_request, _candidate_languages
from aka.legacy.evaluation.gateway_measurement import GatewayMeasurement, prepare_dev_output
from aka.legacy.evaluation.metadata import distributed_evaluation_world_size


def _requirements(workspace):
    path = os.environ.get("ATREX_SANDBOX_REQUIREMENTS_FILE", "").strip()
    if not path:
        return []
    solution = transport._json_object(workspace / "solution.json") or {}
    candidate = (workspace / "kernel.py").read_text(encoding="utf-8")
    return transport._runtime_requirements(path,
        candidate_languages=_candidate_languages(solution, candidate_sources=(candidate,)))


def run_typed_gateway(
    args: argparse.Namespace,
    workspace: Path,
    command_parts: list[str],
    kind: str,
    sync_paths: list[str],
    queue_wait_grace: int,
) -> int | None:
    """Run agate run/profile, returning None only for a documented dev fallback."""
    generalized = transport._is_generalized_workspace(workspace)
    from aka.legacy.task.problem.inputs import PerformanceBaselineError

    try:
        request = prepare_request(
            workspace,
            args.hardware,
            args.timeout,
            args.env,
            transport._command_parts(command_parts),
            kind,
            reference_dir=transport._private_reference_dir(workspace) or workspace,
            requirements=_requirements(workspace),
            profiler=args.profiler,
            profile_level=args.profile_level,
            counters=args.profile_counter,
            kernel_regex=args.kernel_regex,
            top_kernels=args.top_kernels,
        )
    except PerformanceBaselineError as exc:
        diagnostic = {
            "schema": "optimization.gateway-protocol-failure/v1",
            "agate_return_code": 2,
            "response_type": "local-validation",
            "reason": "invalid-performance-baseline",
            "failure_class": "request",
            "metadata_invalid": True,
            "metadata_reason": exc.reason,
            "shape_id": exc.shape_id,
            "hardware": exc.hardware,
        }
        emit_diagnostic(diagnostic)
        print(f"[sandbox] {exc}; GPU submission skipped", file=sys.stderr)
        print(
            "[sandbox] gateway_protocol_failure="
            + json.dumps(diagnostic, sort_keys=True),
            file=sys.stderr,
            flush=True,
        )
        return 2
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        print(
            f"[sandbox] {kind} interface unsupported for this workspace: {exc}; using dev",
            file=sys.stderr,
        )
        return None

    measurement = GatewayMeasurement(request, workspace=workspace, hardware=args.hardware,
        private=generalized, batch_size=getattr(args, "shape_batch_size", transport.DEFAULT_EVAL_SHAPE_BATCH_SIZE),
        profiling=kind != "run")
    shape_batches = measurement.batches
    batched = len(shape_batches) > 1

    if args.dry_run:
        print(
            json.dumps(
                {
                    "hardware": args.hardware,
                    "url": args.url or None,
                    "gateway_profile": args.gateway_profile,
                    "workspace": str(workspace),
                    "kind": kind,
                    "num_gpus": request.get("spec", {}).get("num_gpus", 1),
                    "fallback_kind": "dev",
                    "candidate_bytes": len(request["candidate"].encode("utf-8")),
                    "shape_count": (
                        "private"
                        if transport._is_generalized_workspace(workspace)
                        else len(request["reference"]["shapes"])
                    ),
                    "shape_batch_count": len(shape_batches),
                    "mode": request.get("mode"),
                    "options": request["options"],
                    "sync": sync_paths,
                },
                indent=2,
            )
        )
        return 0

    agate_executable = transport._find_agate()
    try:
        with tempfile.TemporaryDirectory(prefix="atrex-shape-batches-") as temp_dir:
            batch_root = Path(temp_dir)

            def run_batch(
                item: tuple[int, list[str]],
            ) -> subprocess.CompletedProcess[str]:
                batch_index, shape_ids = item
                batch_request = measurement.request_for(batch_index)
                if args.url and agate_executable is None:
                    return transport._run_direct_job(
                        url=args.url,
                        kind="eval" if kind == "run" else kind,
                        payload=batch_request,
                        timeout=args.timeout,
                        queue_wait_grace=queue_wait_grace,
                    )
                if agate_executable is None:
                    raise FileNotFoundError("agate")
                reference_dir = None
                if kind == "run":
                    # Even a single targeted batch needs its filtered reference
                    # directory. Otherwise the agate CLI would reload the original
                    # full shapes.json and defeat --shape-id smoke selection.
                    reference_dir = batch_root / f"batch-{batch_index:04d}"
                    measurement.write_reference(batch_index, reference_dir)
                agate = transport._typed_agate_command(
                    agate_executable,
                    args,
                    workspace,
                    kind,
                    batch_request,
                    queue_wait_grace,
                    reference_dir,
                )
                return transport._run_agate_with_cancel_retry(
                    agate=agate,
                    executable=agate_executable,
                    url=args.url,
                    gateway_profile=args.gateway_profile,
                    command_timeout=transport._gateway_job_timeout(
                        args.timeout, queue_wait_grace
                    ),
                    wait_budget=args.timeout + queue_wait_grace,
                )

            batch_items = list(enumerate(shape_batches))
            if batched:
                print(
                    f"[sandbox] running {len(shape_batches)} shape batches "
                    f"with {min(transport.DEFAULT_EVAL_BATCH_WORKERS, len(shape_batches))} workers",
                    file=sys.stderr,
                    flush=True,
                )
            from aka.legacy.execution.dispatch import run_batches
            processes = run_batches(
                batch_items, run_batch, max_workers=transport.DEFAULT_EVAL_BATCH_WORKERS,
            )
    except transport.GatewayHTTPError as exc:
        if transport._typed_fallback_allowed(exc):
            print(
                f"[sandbox] gateway {kind} interface unavailable ({exc}); using dev",
                file=sys.stderr,
            )
            return None
        if exc.status in {429, 502, 503, 504}:
            emit_diagnostic({"failure_class": "infrastructure", "reason": "transport-unavailable"})
            print(transport.INFRASTRUCTURE_MARKER, file=sys.stderr)
            return transport.ENVIRONMENT_TEMPFAIL
        if generalized:
            raise SystemExit(
                f"sandbox: generalized {kind} gateway request failed; "
                "hidden-case details withheld"
            ) from exc
        raise SystemExit(f"sandbox: {kind} gateway request failed: {exc}") from exc
    except FileNotFoundError as exc:
        raise SystemExit(
            "sandbox: agate not found and no explicit --url was provided; "
            "install atrex-gateway-client first"
        ) from exc

    jobs: list[dict[str, Any]] = []
    for proc in processes:
        job = transport._job_response(proc.stdout or "")
        if job is not None and transport._report_cancellation(job):
            return proc.returncode or 1
        detail = (proc.stderr or "") + (proc.stdout or "")
        if proc.returncode and transport._typed_fallback_allowed(detail):
            print(
                f"[sandbox] gateway {kind} interface rejected this request; using dev",
                file=sys.stderr,
            )
            return None
        if proc.stderr and not generalized:
            print("\n".join(line for line in proc.stderr.rstrip().splitlines()
                              if line != transport.INFRASTRUCTURE_MARKER), file=sys.stderr)
        if job is None:
            if transport._report_agate_transport_failure(proc):
                return transport.ENVIRONMENT_TEMPFAIL
            if proc.stdout and not generalized:
                print(proc.stdout.rstrip())
            elif generalized:
                print(
                    "[sandbox] generalized gateway response unavailable; evaluator details withheld",
                    file=sys.stderr,
                )
            return proc.returncode or 2
        if job.get("status") != "succeeded" or not isinstance(job.get("result"), dict):
            if transport._report_infrastructure_failure(job):
                return transport.ENVIRONMENT_TEMPFAIL
            if generalized:
                print(
                    "[sandbox] generalized evaluation failed; hidden-case details withheld; "
                    f"job_id={job.get('job_id')}",
                    file=sys.stderr,
                )
            else:
                print(json.dumps(job, ensure_ascii=False))
            return proc.returncode or 1
        jobs.append(job)

    print(
        f"[sandbox] gateway_kind={kind} "
        + (
            f"job_ids={','.join(str(job.get('job_id')) for job in jobs)}"
            if batched
            else f"job_id={jobs[0].get('job_id')}"
        ),
        file=sys.stderr,
    )
    if kind == "run":
        result = measurement.interpret(jobs)
        emit_fact({"event": "result_available"})

        print(
            transport.TEST_RESULT_PREFIX + json.dumps(result, ensure_ascii=False, allow_nan=False)
        )
        return 0 if result["all_pass"] else 1

    transport._record_profile_job(jobs[0], workspace, sync_paths)
    print(transport.PROFILE_RESULT_PREFIX + json.dumps(jobs[0]["result"], ensure_ascii=False))
    return 0


class MeasurementAdapter:
    def num_gpus(self, workspace):
        metadata = transport._json_object(
            transport._evaluator_input_path(workspace, "metadata.json", required=False))
        return distributed_evaluation_world_size(metadata)

    run_typed = staticmethod(run_typed_gateway)
    dev_output = staticmethod(prepare_dev_output)


def main(argv=None):
    return transport.main(argv, measurement_adapter=MeasurementAdapter())


if __name__ == "__main__":
    raise SystemExit(main())
