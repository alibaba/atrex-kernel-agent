"""Physical-job recovery: poll known IDs; resubmit only confirmed infra outcomes."""
from __future__ import annotations

import json
import base64
import hashlib
import io
import math
import os
import subprocess
import sys
import time
import tarfile
from pathlib import Path

from supervisor.measurement_records import JOB_ROOT_ENV, EvidenceUnavailable, digest, private_write, read_json

def retry_kind(job: object) -> str | None:
    if not isinstance(job, dict) or job.get("status") not in {"failed", "cancelled", "canceled"}:
        return None
    error = job.get("error") or {}
    if not isinstance(error, dict):
        return None
    if error.get("reason") == "command_timeout":
        return "timeout"
    details = error.get("details")
    if error.get("error_class") == "infra" or (isinstance(details, dict) and details.get("failure_origin") == "infrastructure"):
        return "infra"
    if job.get("status") in {"cancelled", "canceled"} and not error and not job.get("result"):
        return "cancelled"
    if error.get("reason") == "submit_failed" and "Version check returned 404" in str(error.get("message", "")):
        return "infra"
    if error.get("reason") == "timeout" and "never started executing" in str(error.get("message", "")):
        return "infra"
    return None


def cacheable_job(job: object, identity: dict | None = None) -> bool:
    if not isinstance(job, dict) or retry_kind(job) is not None:
        return False
    if job.get("status") == "succeeded":
        result = job.get("result")
        if not isinstance(result, dict) or not result or job.get("error"):
            return False
        request = (identity or {}).get("request", (identity or {}).get("payload", {}))
        if (identity or {}).get("kind") in {"run", "eval"} and "reference" in request:
            try:
                return complete_evaluation(result, request)
            except (TypeError, AttributeError):
                return False
        return True
    if job.get("status") != "failed":
        return False
    error = job.get("error") or {}
    # Unknown/infra/cancelled outcomes are never permanent Candidate failures.
    return isinstance(error, dict) and error.get("error_class") in {
        "candidate", "compilation", "correctness", "validation",
    }


def complete_evaluation(result: dict, request: dict) -> bool:
    """Missing stage/Shape data is not a completed Candidate rejection."""
    shapes = request["reference"].get("shapes", {})
    passed = result.get("passed")
    if not shapes or not isinstance(passed, dict):
        return False
    compile_result = passed.get("compile")
    if not isinstance(compile_result, dict):
        return False
    correctness = passed.get("correctness") or {}
    performance = (result.get("performance") or {}).get("shapes", {})
    for shape in shapes:
        compilation = compile_result if "status" in compile_result else compile_result.get(shape, {})
        if compilation.get("status") == "failed":
            continue
        if compilation.get("status") != "passed":
            return False
        status = correctness.get(shape, {}).get("status")
        if status == "failed":
            continue
        if status != "passed":
            return False
        if request.get("mode") == "correctness_only":
            continue
        samples = performance.get(shape, {}).get("samples", [])
        values = [row.get("end_to_end_time_ms") for row in samples if isinstance(row, dict)]
        if not values or any(isinstance(value, bool) or not isinstance(value, (int, float))
                             or not math.isfinite(value) or value <= 0 for value in values):
            return False
    return True


def job_from_process(process: subprocess.CompletedProcess) -> dict | None:
    try:
        value = json.loads(process.stdout or "")
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def bundle_digest(data: bytes) -> str:
    """Ignore gzip/tar timestamps, but retain exact uploaded bytes and modes."""
    entries = {}
    total = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive:
            if member.isdir():
                continue
            if not member.isfile() or member.name in entries or len(entries) >= 8192:
                raise ValueError("Invalid Gateway bundle identity")
            total += member.size
            if total > 128 * 1024 * 1024:
                raise ValueError("Gateway bundle identity exceeds its size limit")
            with archive.extractfile(member) as source:
                entries[member.name] = {"sha256": hashlib.sha256(source.read()).hexdigest(), "mode": member.mode}
    return digest(entries)


def payload_identity(payload: dict) -> dict:
    """Canonicalize multipart workspace bundles without changing the sent request."""
    value = dict(payload)
    # Gateway display labels default to the temporary staging directory name.
    # They must not turn a resumed logical task into a fresh physical job.
    value.pop("name", None)
    if isinstance(value.get("reference"), dict):
        value["reference"] = dict(value["reference"])
        value["reference"].pop("operator", None)
    files = dict(value.get("files") or {})
    for prefix in ("__atrex_workspace.tar.gz.b64", "__atrex_bench_runtime.tar.gz.b64"):
        names = sorted(name for name in files if name == prefix or name.startswith(prefix + ".part"))
        if names:
            encoded = "".join(files.pop(name) for name in names)
            files[prefix] = {"bundle_digest": bundle_digest(base64.b64decode(encoded))}
    if files:
        value["files"] = files
    return value


def command_identity(command: list[str]) -> list:
    """Hash sidecar contents, not per-request temporary directory names."""
    values = []
    for item in command:
        path = Path(item)
        if path.is_absolute() and path.is_file():
            from orchestrator.session_tail import read_regular_bytes
            data = read_regular_bytes(path, limit=64 * 1024 * 1024 + 1)
            if len(data) > 64 * 1024 * 1024:
                raise ValueError("Gateway sidecar exceeds identity limit")
            if path.name.endswith(".tar.gz.b64"):
                checksum = bundle_digest(base64.b64decode(data))
            else:
                checksum = hashlib.sha256(data).hexdigest()
            values.append({"file": path.name, "sha256": checksum})
        elif path.is_absolute() and path.is_dir():
            from orchestrator.session_tail import read_regular_bytes
            files = {}
            for source in sorted(path.rglob("*")):
                if source.is_file():
                    if len(files) >= 4096:
                        raise ValueError("Gateway reference directory exceeds identity limit")
                    files[source.relative_to(path).as_posix()] = hashlib.sha256(
                        read_regular_bytes(source, limit=16 * 1024 * 1024 + 1)).hexdigest()
            values.append({"directory": files})
        else:
            values.append(item)
    return values


def execute_job(identity: object, submit, poll, *, wait_budget: float) -> subprocess.CompletedProcess:
    """Persist each submission before polling. Never guess after an uncertain POST."""
    directory = Path(os.environ[JOB_ROOT_ENV]) / "jobs" / digest(identity) if os.environ.get(JOB_ROOT_ENV) else None
    state = {}
    if directory:
        try:
            state = read_json(directory / "state.json")
        except FileNotFoundError:
            pass
        if state.get("phase") == "submitting":
            raise EvidenceUnavailable("Prior Gateway submission has no confirmed job ID; operator reconciliation required")
        if state and state.get("identity") != identity:
            raise EvidenceUnavailable("Gateway job checkpoint identity is inconsistent; operator reconciliation required")
    deadline = time.monotonic() + wait_budget
    submission = state.get("submission", 0)
    retries = state.get("retries", {})

    def save(phase, process=None):
        value = {"phase": phase, "submission": submission, "retries": retries, "identity": identity}
        if process is not None:
            value["process"] = {"returncode": process.returncode, "stdout": process.stdout, "stderr": process.stderr}
        if directory:
            private_write(directory / "state.json", value)
            if process is not None:
                private_write(directory / f"submission-{submission:04d}-{phase}.json", value)
        return value

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Gateway request retry/poll budget exhausted")
        if state.get("phase") in {"accepted", "terminal"}:
            data = state["process"]
            process = subprocess.CompletedProcess([], data["returncode"], data["stdout"], data["stderr"])
        else:
            submission += 1
            save("submitting")
            process = submit()
            job = job_from_process(process)
            if not job or not isinstance(job.get("job_id"), str):
                # No reliable ID: retain the uncertain submission, never cache it
                # as a Candidate failure or automatically submit again.
                save("submitting", process)
                return process
            state = save("accepted", process)
        job = job_from_process(process)
        if state.get("phase") != "terminal":
            process = poll(process, max(1, int(deadline - time.monotonic())))
            job = job_from_process(process)
            if not job or job.get("status") not in {"succeeded", "failed", "cancelled", "canceled"}:
                # Keep the accepted ID so a later request polls that same job.
                return process
            state = save("terminal", process)
        kind = retry_kind(job)
        if kind is None or (kind in {"timeout", "cancelled"} and retries.get(kind, 0) >= 1):
            return process
        retries[kind] = retries.get(kind, 0) + 1
        count = sum(retries.values())
        delay = min(5 * 2 ** min(count - 1, 4), 60)
        if deadline - time.monotonic() <= delay:
            return process
        print(f"[sandbox] Gateway {kind} failure; resubmitting after {delay}s (retry {count})",
              file=sys.stderr, flush=True)
        # Every returned infra terminal includes a known outcome. In particular
        # logs_unavailable + backend_state=succeeded needs a NEW job, not get().
        state = save("retry_pending")
        time.sleep(delay)
