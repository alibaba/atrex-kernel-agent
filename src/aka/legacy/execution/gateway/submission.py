"""Submit one immutable prepared gateway request and return the raw receipt."""
import json
import os
import sys
from pathlib import Path
import tempfile
from . import sandbox as transport
from .returned_files import emit
from aka.legacy.execution.outcomes import emit_fact, emit_diagnostic


def run_prepared(args, workspace, sync_paths, grace):
    size = os.fstat(args.prepared_request_fd).st_size
    if size > 64 * 1024 * 1024:
        raise ValueError("prepared gateway request exceeds 64 MiB")
    prepared = json.loads(os.pread(args.prepared_request_fd, size, 0))
    if prepared.get("schema_version") != 1 or prepared.get("kind") not in {"run", "profile"}:
        raise ValueError("unsupported prepared gateway request")
    kind, request = prepared["kind"], prepared["request"]
    if args.ssh or args.keep_pod or args.input:
        raise ValueError("prepared typed requests do not support SSH, keep-pod or custom input")
    executable = transport._find_agate()
    try:
        if args.url and executable is None:
            process = transport._run_direct_job(url=args.url, kind="eval" if kind == "run" else kind,
                payload=request, timeout=args.timeout, queue_wait_grace=grace)
        elif executable is None:
            raise FileNotFoundError("agate not found and no explicit --url was provided")
        else:
            with tempfile.TemporaryDirectory(prefix="aka-submission-") as temp:
                directory = Path(temp)
                reference = directory / "reference"
                reference.mkdir()
                for name, content in prepared["reference_files"].items():
                    if Path(name).name != name:
                        raise ValueError("invalid prepared reference filename")
                    (reference / name).write_text(content, encoding="utf-8")
                (directory / "kernel.py").write_text(request["candidate"], encoding="utf-8")
                requirements = request.get("requirements", [])
                requirement_path = directory / "requirements.txt"
                requirement_path.write_text("\n".join(requirements) + "\n", encoding="utf-8")
                argv = transport._typed_agate_command(executable, args, directory, kind, request, grace,
                    reference, requirements_file=requirement_path)
                process = transport._run_agate_with_cancel_retry(agate=argv, executable=executable,
                    url=args.url, gateway_profile=args.gateway_profile,
                    command_timeout=transport._gateway_job_timeout(args.timeout, grace),
                    wait_budget=args.timeout + grace)
    except transport.GatewayHTTPError as error:
        # No second route is submitted here. The owning experiment receives admission facts.
        emit_diagnostic({"failure_class": "infrastructure" if error.status in {429,502,503,504} else "request",
                         "http_status": error.status, "reason": "gateway-http-rejection"})
        if error.status in {429,502,503,504}:
            print(transport.INFRASTRUCTURE_MARKER, file=sys.stderr)
            return transport.ENVIRONMENT_TEMPFAIL
        print(f"HTTP {error.status}", file=sys.stderr)
        return 2
    job = transport._job_response(process.stdout or "")
    if job is not None and transport._report_cancellation(job):
        return process.returncode or 1
    if job is not None and transport._report_infrastructure_failure(job):
        return transport.ENVIRONMENT_TEMPFAIL
    if job is None and transport._report_agate_transport_failure(process):
        return transport.ENVIRONMENT_TEMPFAIL
    if job is not None and job.get("status") == "succeeded" and isinstance(job.get("result"), dict):
        emit_fact({"event": "result_available"})
        if kind == "profile":
            files = []
            for item in job["result"].get("artifacts") or []:
                if not isinstance(item, dict):
                    continue
                name = transport._safe_relative(str(item.get("name") or ""))
                with tempfile.TemporaryDirectory(prefix="aka-profile-download-") as temp:
                    target = Path(temp) / name
                    transport._download_oss_artifact(item, target)
                    files.append((name, target.read_bytes(), 0o600))
            emit(files)
    if process.stdout:
        print(process.stdout.rstrip())
    if process.stderr:
        print(process.stderr.rstrip(), file=sys.stderr)
    return process.returncode or (0 if job and job.get("status") == "succeeded" else 1)
