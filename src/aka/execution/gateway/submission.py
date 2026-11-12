"""Submit one immutable prepared gateway request and return the raw receipt."""
import json
import os
import sys
from pathlib import Path
import tempfile
from . import sandbox as transport
from aka.execution.configuration import client_wait_timeout
from .requests import wait_budget_records
from .returned_files import emit
from aka.execution.outcomes import emit_fact, emit_diagnostic
from .requests import NATIVE_ABBA, operation_request, prepared_operation, effective_budget, hard_limit, typed_gateway_job_timeout


def run_prepared(args, workspace, sync_paths, grace):
    size = os.fstat(args.prepared_request_fd).st_size
    if size > 64 * 1024 * 1024:
        raise ValueError("prepared gateway request exceeds 64 MiB")
    prepared = json.loads(os.pread(args.prepared_request_fd, size, 0))
    return submit_prepared(args, workspace, grace, prepared)


def submit_prepared(args, workspace, grace, prepared):
    kind = prepared_operation(prepared, args.kind)
    request = operation_request(prepared["request"], kind, ssh=bool(args.ssh))
    if args.ssh:
        if args.input:
            raise ValueError("prepared SSH inputs must be frozen by Evaluation")
        from .source_archives import read_archives, validate_archives
        archives = read_archives(prepared)
        validate_archives(prepared, archives, args.kind, ssh=True)
        if archives:
            from aka.evaluation.source_archives import archive_source
            request['__atrex_eval_archives'] = {archive.path: {name: content.decode('utf-8')
                for name, content in archive_source(archive).files.items() if content is not None}
                for archive in archives}
        from aka.execution.ssh.transport import run
        from aka.contracts.execution import ExecutionUnavailable
        try:
            process = run(transport, args, workspace, request, kind)
        except (transport.SSHTransportError, ExecutionUnavailable) as error:
            emit_diagnostic({"failure_class": "infrastructure", "reason": "ssh_evidence_unavailable"})
            print(str(error), file=sys.stderr)
            return transport.ENVIRONMENT_TEMPFAIL
        emit(getattr(process, 'output_files', ()))
        if process.stdout:
            print(process.stdout.rstrip())
        if process.stderr:
            print(process.stderr.rstrip(), file=sys.stderr)
        return process.returncode
    wait_budget_records(client_wait_timeout(args.timeout, grace))
    if kind in {"run", "profile"}:
        limit = hard_limit(os.environ, kind)
        requested = typed_gateway_job_timeout(args.timeout, grace, limit=limit)
        budget = effective_budget(requested, limit=limit)
        request.setdefault("options", {})["timeout_s"] = budget.effective
    else:
        emit_diagnostic({"failure_class":"protocol", "reason":"job-runtime-budget-unconfirmed"})
    if args.input:
        raise ValueError("prepared typed requests do not support custom input")
    from .source_archives import read_archives, validate_archives, run_archived, SourceArchiveError
    archives = read_archives(prepared)
    validate_archives(prepared, archives, args.kind)
    executable = transport._find_agate()
    try:
        if archives:
            process = run_archived(args, workspace, grace, request, archives, transport=transport)
        elif args.url and executable is None:
            process = transport._run_direct_job(url=args.url, kind={"run":"eval", "check":"compile"}.get(kind, kind),
                payload=request, timeout=args.timeout, queue_wait_grace=grace)
        elif executable is None:
            raise FileNotFoundError("agate not found and no explicit --url was provided")
        else:
            with tempfile.TemporaryDirectory(prefix="aka-submission-") as temp:
                directory = Path(temp)
                reference = directory / "reference"
                reference.mkdir()
                for name, content in prepared.get("reference_files", {}).items():
                    if Path(name).name != name:
                        raise ValueError("invalid prepared reference filename")
                    (reference / name).write_text(content, encoding="utf-8")
                (directory / "kernel.py").write_text(request["candidate"], encoding="utf-8")
                requirements = request.get("requirements", [])
                requirement_path = directory / "requirements.txt"
                requirement_path.write_text("\n".join(requirements) + "\n", encoding="utf-8")
                initial = directory / "init_kwargs.json" if "init_kwargs" in request else None
                if initial is not None:
                    initial.write_text(json.dumps(request["init_kwargs"]), encoding="utf-8")
                argv = transport._typed_agate_command(executable, args, directory, kind, request, grace,
                    reference, requirements_file=requirement_path, init_kwargs_file=initial)
                process = transport._run_agate_with_cancel_retry(agate=argv, executable=executable,
                    url=args.url, gateway_profile=args.gateway_profile,
                    command_timeout=transport._gateway_job_timeout(args.timeout, grace),
                    wait_budget=client_wait_timeout(args.timeout, grace))
    except SourceArchiveError as error:
        emit_diagnostic({"failure_class": "infrastructure", "reason": error.code,
                         **({"http_status": error.status} if error.status is not None else {})})
        print(transport.INFRASTRUCTURE_MARKER, file=sys.stderr)
        return transport.ENVIRONMENT_TEMPFAIL
    except transport.GatewayHTTPError as error:
        from .diagnostics import report_typed_limitation
        report_typed_limitation(f"http {error.status}")
        # No second route is submitted here. The owning experiment receives admission facts.
        emit_diagnostic({"failure_class": "infrastructure" if error.status in {429,502,503,504} else "request",
                         "http_status": error.status, "reason": "gateway-http-rejection"})
        if error.status in {429,502,503,504}:
            print(transport.INFRASTRUCTURE_MARKER, file=sys.stderr)
            return transport.ENVIRONMENT_TEMPFAIL
        print(f"HTTP {error.status}", file=sys.stderr)
        return 2
    job = transport._job_response(process.stdout or "")
    # Partial native measurement evidence belongs to Evaluation even when the
    # transport classifies cancellation or infrastructure failure below.
    if args.kind == NATIVE_ABBA:
        if process.stdout:
            print(process.stdout.rstrip())
        if process.stderr:
            print(process.stderr.rstrip(), file=sys.stderr)
    if process.returncode:
        from .diagnostics import report_typed_limitation
        report_typed_limitation((process.stdout or "") + "\n" + (process.stderr or ""))
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
                from aka.execution.command_transport import raw_profile_path
                if getattr(args, "include_raw_profile", None) is False and raw_profile_path(name):
                    continue
                with tempfile.TemporaryDirectory(prefix="aka-profile-download-") as temp:
                    target = Path(temp) / name
                    transport._download_oss_artifact(item, target)
                    from aka.execution.command_transport import output_limit
                    if target.stat().st_size > output_limit(args):
                        emit_diagnostic({"failure_class": "output", "reason": "output-file-size-limit"})
                        continue
                    files.append((name, target.read_bytes(), 0o600))
            emit(files)
    if process.stdout and args.kind != NATIVE_ABBA:
        print(process.stdout.rstrip())
    if process.stderr and args.kind != NATIVE_ABBA:
        print(process.stderr.rstrip(), file=sys.stderr)
    return process.returncode or (0 if job and job.get("status") == "succeeded" else 1)


def run_kernel_operation(args, workspace, grace):
    """Check/disassemble use their own source request, without an Eval harness."""
    import argparse
    parser = argparse.ArgumentParser(prog="sandbox " + args.kind)
    parser.add_argument("candidate")
    parser.add_argument("--init-kwargs")
    if args.kind == "check":
        parser.add_argument("--arch")
        parser.add_argument("--sanitize", choices=("memcheck", "racecheck", "initcheck", "synccheck"))
    else:
        parser.add_argument("--fmt", choices=("sass", "ptx", "isa", "auto"), default="auto")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    options = parser.parse_args(command)
    candidate = (workspace / options.candidate).resolve()
    candidate.relative_to(workspace)
    solution = transport._json_object(workspace / "solution.json") or {}
    request = {"spec": {"target_hardware": [args.hardware], "languages": transport._candidate_languages(solution)},
               "candidate": candidate.read_text(encoding="utf-8"), "env_vars": transport._parse_env_items(args.env)}
    requirements = json.loads(os.environ.get("AKA_GATEWAY_DEV_DEPENDENCIES", "[]"))
    if requirements:
        request["requirements"] = requirements
    for field in ("arch", "sanitize", "fmt"):
        value = getattr(options, field, None)
        if value:
            request[field] = value
    if options.init_kwargs:
        path = (workspace / options.init_kwargs).resolve()
        path.relative_to(workspace)
        request["init_kwargs"] = transport._json_object(path, required=True)
    if args.dry_run:
        print(json.dumps(operation_request(request, args.kind), indent=2))
        return 0
    return submit_prepared(args, workspace, grace,
        {"schema_version": 1, "kind": args.kind, "request": request})
