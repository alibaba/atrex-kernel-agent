"""Additional GPU operations, separate from measurement storage and acceptance.

The gateway argument is the running executor module's public API. Passing that
instance keeps job tracking/cancellation shared with the CLI entry point, which
may be loaded as __main__ rather than supervisor.gateway.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _workload_shape_ids(path: Path) -> list[str]:
    ids, seen = [], set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            location = f"{path.name} line {line_number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{location}: invalid JSON ({exc.msg}); ask the operator to repair the workload contract"
                ) from exc
            if not isinstance(record, dict) or not isinstance(record.get("uuid"), str) or not record["uuid"].strip():
                raise ValueError(
                    f"{location}: expected an object with a non-empty string 'uuid'; "
                    "ask the operator to repair the workload contract"
                )
            if record["uuid"] in seen:
                raise ValueError(f"{location}: duplicate uuid; ask the operator to repair the workload contract")
            seen.add(record["uuid"])
            ids.append(record["uuid"])
    if not ids:
        raise ValueError(f"{path.name}: no workload entries; ask the operator to repair the workload contract")
    return ids


def _abba_batch_error(gateway, batch: str, message: str, stderr) -> RuntimeError:
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    detail = gateway.bounded_actionable_diagnostic(stderr, limit=4000) or "(empty)"
    return RuntimeError(
        f"{batch} {message}; no completed comparison is available. "
        "Inspect the Supervisor diagnostics and remote job state; do not resubmit blindly.\n"
        f"Nested stderr:\n{detail}"
    )


def diagnostic(gateway, args, workspace: Path, queue_wait_grace: int) -> int:
    if args.command or args.ssh:
        raise ValueError("Check/Disassemble require the typed Gateway; no command or SSH transport")
    request = gateway.build_typed_request(
        workspace, args.hardware, args.timeout, args.env, [], args.kind,
        arch=args.arch, sanitize=args.sanitize, disassembly_format=args.disassembly_format,
        requirements=args.requirement or (), deps_mode=args.deps_mode,
    )
    if args.dry_run:
        print(json.dumps({"kind": args.kind, "candidate_bytes": len(request["candidate"].encode())}))
        return 0
    executable = gateway.find_agate()
    if args.url and executable is None:
        process = gateway.run_direct_job(
            url=args.url, kind="compile" if args.kind == "check" else "disassemble",
            payload=request, timeout=args.timeout, queue_wait_grace=queue_wait_grace,
        )
    else:
        if executable is None:
            raise ValueError("Install the Agate client on the Supervisor or configure --sandbox-url")
        with tempfile.TemporaryDirectory(prefix="aka-diagnostic-") as directory:
            command = gateway.build_typed_agate_command(executable, args, workspace, args.kind, request,
                                                   queue_wait_grace, request_sidecar_dir=Path(directory))
            process = gateway.run_agate_with_cancel_retry(
                agate=command, executable=executable, url=args.url,
                gateway_profile=args.gateway_profile,
                command_timeout=gateway.gateway_job_timeout(args.timeout, queue_wait_grace),
                wait_budget=args.timeout + queue_wait_grace,
                request_identity={"kind": args.kind, "request": request},
            )
    job = gateway.parse_job_response(process.stdout or "")
    if not job or job.get("status") != "succeeded" or not isinstance(job.get("result"), dict):
        print(json.dumps({"status": "failed", "operation": args.kind,
                          "error": (job or {}).get("error") or "Gateway returned no result"}))
        return process.returncode or 1
    prefix = "[sandbox] CHECK_JSON=" if args.kind == "check" else "[sandbox] DISASSEMBLE_JSON="
    print(prefix + json.dumps(job["result"]))
    return 0


def validate_comparison(args) -> None:
    if args.kind != "run" or args.evaluation_mode == "correctness_only":
        raise ValueError("--baseline-path requires --kind run in full mode")
    if not 1 <= args.comparison_repeats <= 20:
        raise ValueError("--comparison-repeats must be in 1..20")
    if (args.command or args.evaluation_input_path or args.evaluation_shapes_path
            or args.shape_id or args.multi_seed is not None):
        raise ValueError("ABBA uses the canonical full contract; command/input/shape/seed overrides are unsupported")
    comparison_run_timeout(args)


def comparison_run_timeout(args) -> int:
    available = (args.timeout - 30) // (2 * args.comparison_repeats)
    requested = getattr(args, "comparison_run_timeout", None)
    seconds = min(120, available) if requested is None else requested
    if seconds <= 0 or seconds > available:
        raise ValueError("--comparison-run-timeout must be positive and fit the full ABBA allocation schedule")
    return seconds


def compare(gateway, args, workspace: Path, queue_wait_grace: int) -> int:
    """Checkpoint each completed physical batch before running the next one."""
    from long_horizon.verifier import verification_schedule, parse_abba_payload, merge_abba_batch_payloads
    from supervisor.projection import abba
    from supervisor.abba_checkpoints import AbbaBatchStore, validate_batch
    from supervisor.measurement_records import JOB_ROOT_ENV

    validate_comparison(args)
    baseline = gateway.read_workspace_override(
        workspace, args.baseline_path, field="baseline-path", max_bytes=16 * 1024 * 1024,
    )
    schedule = verification_schedule(args.comparison_repeats)
    per_run = comparison_run_timeout(args)
    root = gateway.private_reference_dir(workspace) or workspace
    sol = (workspace / "workload.jsonl").is_file()
    if sol:
        ids = _workload_shape_ids(workspace / "workload.jsonl")
    else:
        ids = sorted(gateway.read_json_object(root / "shapes.json", required=True), key=gateway.shape_id_sort_key)
    if args.dry_run:
        print(json.dumps({"kind": "same_allocation_abba", "shape_count": len(ids),
                          "comparison_repeats": args.comparison_repeats}))
        return 0
    control = workspace / "verification_artifacts" / "agent-comparison"
    control.mkdir(parents=True, exist_ok=True)
    shutil.copy2(gateway.REPO_ROOT / "long_horizon/remote_abba.py", control / "test_kernel.py")
    (control / "snapshots").mkdir(exist_ok=True)
    (control / "snapshots/baseline.py").write_text(baseline)
    (control / "snapshots/candidate.py").write_bytes((workspace / "kernel.py").read_bytes())
    command = ["python3", "test_kernel.py", "--no-memory", "--version", args.version or "vcompare"]
    if args.timed_runs is not None and not sol:
        command += ["--timed-runs", str(args.timed_runs)]
    batches = [ids] if sol else gateway.batch_shape_ids(ids, args.shape_batch_size)
    payloads = []
    checkpoints = AbbaBatchStore(Path(os.environ[JOB_ROOT_ENV])) if os.environ.get(JOB_ROOT_ENV) else None
    for index, shapes in enumerate(batches):
        request = control / f"request-{index}.json"
        result = control / f"result-{index}.json"
        request.write_text(json.dumps({
            "schema_version": 1, "schedule": schedule,
            "manifests": {"incumbent": {"kernel.py": "snapshots/baseline.py"},
                          "candidate": {"kernel.py": "snapshots/candidate.py"}},
            "command": command + ([] if sol else [v for sid in shapes for v in ("--shape-id", sid)]),
            "run_timeout_seconds": per_run,
        }))
        nested = [sys.executable, str(gateway.REPO_ROOT / "supervisor/gateway.py"),
                  "--workspace", str(workspace), "--kind", "dev", "--hardware", args.hardware,
                  "--timeout", str(args.timeout), "--no-sync"]
        for option, value in (("--url", args.url), ("--gateway-profile", args.gateway_profile),
                              ("--ssh", args.ssh), ("--ssh-init", args.ssh_init),
                              ("--health-command", args.health_command)):
            if value:
                nested += [option, str(value)]
        if args.ssh_gpu is not None:
            nested += ["--ssh-gpu", str(args.ssh_gpu)]
        for bind in args.ssh_runtime_bind or ():
            nested += ["--ssh-runtime-bind", bind]
        for item in args.env:
            nested += ["--env", item]
        nested += ["--", "python3", str((control / "test_kernel.py").relative_to(workspace)),
                   str(request.relative_to(workspace)), str(result.relative_to(workspace))]
        batch = f"ABBA batch {index + 1}/{len(batches)}"
        identity = {"schedule": schedule, "shape_ids": shapes, "batch": index}
        if checkpoints:
            saved = checkpoints.load(identity)
            if saved is not None:
                payloads.append(saved["payload"])
                continue
        try:
            process = subprocess.run(nested, cwd=workspace, env=os.environ.copy(), capture_output=True,
                                     text=True, timeout=args.timeout + queue_wait_grace + 120)
        except subprocess.TimeoutExpired as exc:
            raise _abba_batch_error(
                gateway, batch, "timed out; remote execution may have completed", exc.stderr,
            ) from exc
        if process.returncode:
            raise _abba_batch_error(gateway, batch, f"exited with code {process.returncode}", process.stderr)
        try:
            payload = parse_abba_payload(process.stdout)
            validate_batch(payload, schedule, shapes)
            if checkpoints:
                checkpoints.save(identity, payload, stdout=process.stdout, stderr=process.stderr)
            payloads.append(payload)
        except ValueError as exc:
            raise _abba_batch_error(gateway, batch, f"returned an invalid result ({exc})", process.stderr) from exc
    try:
        value = abba(merge_abba_batch_payloads(payloads, schedule, ids), schedule, ids, args.comparison_repeats)
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise RuntimeError(
            "ABBA results could not be combined into a valid comparison. "
            "Inspect the Supervisor diagnostics and remote job state; do not resubmit blindly. "
            f"Details: {gateway.bounded_actionable_diagnostic(exc, limit=4000)}"
        ) from exc
    print("[sandbox] ABBA_JSON=" + json.dumps(value))
    return 0 if value["correct"] else 1
