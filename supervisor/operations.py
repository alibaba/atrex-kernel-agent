"""Additional GPU operations, separate from measurement storage and acceptance."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def diagnostic(gateway, args, workspace: Path, queue_wait_grace: int) -> int:
    if args.command or args.ssh:
        raise ValueError("Check/Disassemble require the typed Gateway; no command or SSH transport")
    request = gateway._typed_request(
        workspace, args.hardware, args.timeout, args.env, [], args.kind,
        arch=args.arch, sanitize=args.sanitize, disassembly_format=args.disassembly_format,
        requirements=args.requirement or (), deps_mode=args.deps_mode,
    )
    if args.dry_run:
        print(json.dumps({"kind": args.kind, "candidate_bytes": len(request["candidate"].encode())}))
        return 0
    executable = gateway._find_agate()
    if args.url and executable is None:
        process = gateway._run_direct_job(
            url=args.url, kind="compile" if args.kind == "check" else "disassemble",
            payload=request, timeout=args.timeout, queue_wait_grace=queue_wait_grace,
        )
    else:
        if executable is None:
            raise ValueError("Install the Agate client on the Supervisor or configure --sandbox-url")
        with tempfile.TemporaryDirectory(prefix="aka-diagnostic-") as directory:
            command = gateway._typed_agate_command(executable, args, workspace, args.kind, request,
                                                   queue_wait_grace, request_sidecar_dir=Path(directory))
            process = gateway._run_agate_with_cancel_retry(
                agate=command, executable=executable, url=args.url,
                gateway_profile=args.gateway_profile,
                command_timeout=gateway._gateway_job_timeout(args.timeout, queue_wait_grace),
                wait_budget=args.timeout + queue_wait_grace,
            )
    job = gateway._job_response(process.stdout or "")
    if not job or job.get("status") != "succeeded" or not isinstance(job.get("result"), dict):
        print(json.dumps({"status": "failed", "operation": args.kind,
                          "error": (job or {}).get("error") or "Gateway returned no result"}))
        return process.returncode or 1
    prefix = "[sandbox] CHECK_JSON=" if args.kind == "check" else "[sandbox] DISASSEMBLE_JSON="
    print(prefix + json.dumps(job["result"]))
    return 0


def compare(gateway, args, workspace: Path, queue_wait_grace: int) -> int:
    """Use the existing same-allocation AB/BA runner, without PR4 caching/repeats."""
    from long_horizon.verifier import verification_schedule, _payload_from_stdout, _merge_batch_payloads
    from supervisor.projection import abba

    if args.kind != "run" or args.evaluation_mode == "correctness_only":
        raise ValueError("--baseline-path requires --kind run in full mode")
    if not 1 <= args.comparison_repeats <= 20:
        raise ValueError("--comparison-repeats must be in 1..20")
    if (args.command or args.evaluation_input_path or args.evaluation_shapes_path
            or args.shape_id or args.multi_seed is not None):
        raise ValueError("ABBA uses the canonical full contract; command/input/shape/seed overrides are unsupported")
    baseline = gateway._read_workspace_override(
        workspace, args.baseline_path, field="baseline-path", max_bytes=16 * 1024 * 1024,
    )
    schedule = verification_schedule(args.comparison_repeats)
    per_run = min(120, (args.timeout - 30) // len(schedule))
    if per_run <= 0:
        raise ValueError("ABBA schedule does not fit the configured timeout")
    root = gateway._private_reference_dir(workspace) or workspace
    sol = (workspace / "workload.jsonl").is_file()
    if sol:
        ids = [json.loads(line)["uuid"] for line in (workspace / "workload.jsonl").read_text().splitlines() if line.strip()]
    else:
        ids = sorted(gateway._json_object(root / "shapes.json", required=True), key=gateway._sort_shape_id)
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
    batches = [ids] if sol else gateway._shape_batches(ids, args.shape_batch_size)
    payloads = []
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
        process = subprocess.run(nested, cwd=workspace, env=os.environ.copy(), capture_output=True,
                                 text=True, timeout=args.timeout + queue_wait_grace + 120)
        if process.returncode:
            raise RuntimeError("ABBA batch failed; no completed comparison is available")
        payloads.append(_payload_from_stdout(process.stdout))
    value = abba(_merge_batch_payloads(payloads, schedule, ids), schedule, ids, args.comparison_repeats)
    print("[sandbox] ABBA_JSON=" + json.dumps(value))
    return 0 if value["correct"] else 1
