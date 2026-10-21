"""Physical-job recovery: poll known IDs; retry confirmed rejections/infra outcomes."""
from __future__ import annotations

import json
import base64
import hashlib
import hmac
import io
import math
import os
import re
import subprocess
import sys
import time
import tarfile
import uuid
from pathlib import Path

from aka.light.supervisor.measurement_records import JOB_ROOT_ENV, EvidenceUnavailable, digest, private_write, read_json
from aka.light.evaluation.evaluation_failures import embedded_device_failure


def retry_delay(count: int) -> float:
    return float(5 * 2 ** (count - 1)) if count < 5 else 60.0


def request_is_infrastructure(process) -> bool:
    payload = job_from_process(process)
    if not isinstance(payload, dict):
        return False
    error = payload.get("error", payload)
    if isinstance(error, dict) and isinstance(error.get("details"), dict) and error["details"].get("cause_class") == "InvalidURL":
        return False
    return isinstance(error, dict) and (
        error.get("error_class") == "infra" or error.get("reason") == "oss_unavailable"
    )


class SubmissionUncertain(RuntimeError):
    """A transport failure; replay is allowed only with a persisted wire key."""

    def __init__(self, process):
        self.process = process
        super().__init__("Gateway submission outcome is unknown")


class EvaluationDeviceUnavailable(EvidenceUnavailable):
    """Bounded device recovery exhausted; never negative Kernel evidence."""


class SubmissionRejected(RuntimeError):
    """The submit transport received a definitive rejection, not an accepted job."""

    def __init__(self, process: subprocess.CompletedProcess, status: int | None = None):
        self.process = process
        self.status = status
        self.retryable = status in {408, 429, 503} or request_is_infrastructure(process)
        super().__init__(f"Gateway rejected submission with HTTP {status}" if status is not None
                         else "Agate CLI rejected submission before dispatch")


def submission_rejection(process: subprocess.CompletedProcess) -> SubmissionRejected | None:
    """Recognize pre-dispatch CLI errors; arbitrary stderr is not proof of safety."""
    if process.returncode == 0:
        return None
    response = job_from_process(process)
    streams = (process.stdout or "") + "\n" + (process.stderr or "")
    if (response and response.get("job_id")) or re.search(r"\b(?:submitted\s+)?job_id[=:]\s*\S+", streams):
        return None
    # Agate's GatewayError formatter, used by its --no-wait submit commands.
    match = re.search(r"(?m)^agate: \[([45]\d{2})\](?:\s|$)", process.stderr or "")
    if match:
        status = int(match[1])
        return SubmissionRejected(process, status) if (status < 500 and status != 408) or status == 503 else None
    # argparse exits before args.func can dispatch. Do not generalize this to
    # arbitrary code-2 errors: they can also occur after a request was accepted.
    if (process.returncode == 2 and re.match(r"usage: agate\b", process.stderr or "")
            and re.search(r"(?m)^agate(?: [\w-]+)*: error: ", process.stderr or "")):
        return SubmissionRejected(process)
    # Also accept structured CLI HTTP-error envelopes, but never infer a
    # pre-dispatch rejection from error_class=infra or exit code alone.
    if response and response.get("error"):
        status = response.get("http_status", response.get("status_code"))
        if isinstance(status, int) and not isinstance(status, bool) and 400 <= status < 600:
            return SubmissionRejected(process, status) if (status < 500 and status != 408) or status == 503 else None
    return None


def retry_kind(job: object) -> str | None:
    """Match Runtime: only explicitly failed infra Jobs get a replacement."""
    if not isinstance(job, dict) or job.get("status") != "failed":
        return None
    error = job.get("error") or {}
    if not isinstance(error, dict):
        return None
    if error.get("error_class") == "infra":
        return "infra"
    return None


def eval_retry_kind(job: object) -> str | None:
    """Evaluation shares the same explicit infrastructure policy as other jobs."""
    return retry_kind(job)


def cacheable_job(job: object, identity: dict | None = None) -> bool:
    if not isinstance(job, dict) or retry_kind(job) is not None or embedded_device_failure(job):
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
    if request.get("abba"):
        from aka.light.evaluation.native_abba import parse_native_abba
        try:
            parse_native_abba({"status": "succeeded", "result": result},
                              request["abba"]["repeats"] * 2, list(request["reference"]["shapes"]))
            return True
        except (ValueError, TypeError, KeyError, AttributeError):
            return False
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
    archives = value.get("__atrex_eval_archives")
    if isinstance(archives, dict):
        # Exact content already lives in the Measurement Record. Physical job
        # checkpoints need only stable digests, not another entire source tree.
        value["__atrex_eval_archives"] = {
            name: {path: hashlib.sha256(text.encode()).hexdigest() for path, text in files.items()}
            for name, files in archives.items()
        }
    if isinstance(value.get("env_vars"), dict):
        environment = value.pop("env_vars")
        encoded = json.dumps(environment, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False).encode()
        # The empty-key fallback preserves standalone/cooperative CLI behavior;
        # only Supervisor-supplied keys make this fingerprint an authenticity check.
        key = bytes.fromhex(os.environ["ATREX_AKA_IDENTITY_KEY"]) if os.environ.get(
            "ATREX_AKA_IDENTITY_KEY"
        ) else b""
        value["env_names"] = sorted(environment)
        value["env_fingerprint"] = hmac.new(key, encoded, hashlib.sha256).hexdigest()
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


def private_value_fingerprint(value: str) -> str:
    """Bind a private launcher value to a checkpoint without storing it there."""
    encoded = os.environ.get("ATREX_AKA_IDENTITY_KEY", "")
    key = bytes.fromhex(encoded) if encoded else b""
    return hmac.new(key, value.encode(), hashlib.sha256).hexdigest()


def command_identity(command: list[str]) -> list:
    """Hash sidecar contents, not per-request temporary directory names."""
    values = []
    for item in command:
        path = Path(item)
        if path.is_absolute() and path.is_file():
            from aka.light.orchestrator.session_tail import read_regular_bytes
            data = read_regular_bytes(path, limit=64 * 1024 * 1024 + 1)
            if len(data) > 64 * 1024 * 1024:
                raise ValueError("Gateway sidecar exceeds identity limit")
            if path.name.endswith(".tar.gz.b64"):
                checksum = bundle_digest(base64.b64decode(data))
            else:
                checksum = hashlib.sha256(data).hexdigest()
            values.append({"file": path.name, "sha256": checksum})
        elif path.is_absolute() and path.is_dir():
            from aka.light.orchestrator.session_tail import read_regular_bytes
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


def execute_job(
    identity: object,
    submit,
    poll,
    *,
    wait_budget: float,
    retry_classifier=retry_kind,
    max_retries: int | None = None,
    idempotent_submission: bool = False,
    max_device_retries: int | None = None,
    device_retry_delay: float | None = None,
) -> subprocess.CompletedProcess:
    """Persist each submission before polling. Never guess after an uncertain POST."""
    retry_classifier = retry_kind if retry_classifier is None else retry_classifier
    directory = Path(os.environ[JOB_ROOT_ENV]) / "jobs" / digest(identity) if os.environ.get(JOB_ROOT_ENV) else None
    state = {}
    if directory:
        try:
            state = read_json(directory / "state.json")
        except FileNotFoundError:
            pass
        if state and state.get("identity") != identity:
            raise EvidenceUnavailable("Gateway job checkpoint identity is inconsistent; operator reconciliation required")
    deadline = time.monotonic() + wait_budget
    submission = state.get("submission", 0)
    retries = state.get("retries", {})
    wire_key = state.get("idempotency_key")
    if idempotent_submission and not wire_key and state.get("phase") == "submitting":
        raise EvidenceUnavailable("Old submission has no persisted wire key; operator reconciliation required")
    wire_key = wire_key or uuid.uuid4().hex

    def save(phase, process=None):
        value = {"phase": phase, "submission": submission, "retries": retries, "identity": identity}
        if idempotent_submission:
            value["idempotency_key"] = wire_key
        if process is not None:
            value["process"] = {"returncode": process.returncode, "stdout": process.stdout, "stderr": process.stderr}
        if directory:
            private_write(directory / "state.json", value)
            if process is not None:
                private_write(directory / f"submission-{submission:04d}-{phase}.json", value)
        return value

    rejection = None
    if state.get("phase") == "submitting":
        data = state.get("process")
        if isinstance(data, dict):
            rejection = submission_rejection(subprocess.CompletedProcess(
                [], data["returncode"], data["stdout"], data["stderr"]))
        if rejection is None and not idempotent_submission:
            raise EvidenceUnavailable("Prior Gateway submission has no confirmed job ID; operator reconciliation required")
        # Recover old checkpoints only when they actually contain a definitive
        # rejection. A bare pre-POST marker still has an unknown outcome.

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise EvidenceUnavailable(
                "Gateway request retry/poll deadline reached; saved Job state requires reconciliation"
            )
        if rejection is None:
            if state.get("phase") in {"accepted", "terminal"}:
                data = state["process"]
                process = subprocess.CompletedProcess([], data["returncode"], data["stdout"], data["stderr"])
            else:
                submission += 1
                save("submitting")
                try:
                    process = submit(wire_key) if idempotent_submission else submit()
                except SubmissionRejected as error:
                    rejection = error
                    process = error.process
                except SubmissionUncertain as error:
                    process = error.process
                job = job_from_process(process)
                if rejection is None and (not job or not isinstance(job.get("job_id"), str) or not job["job_id"]):
                    rejection = submission_rejection(process)
                    if rejection is None:
                        # No reliable ID or explicit rejection: retain the
                        # uncertain submission, never guess that retry is safe.
                        save("submitting", process)
                        if not idempotent_submission:
                            return process
                        count = retries.get("transport", 0) + 1
                        retries["transport"] = count
                        save("submitting", process)
                        delay = retry_delay(count)
                        if deadline - time.monotonic() <= delay:
                            raise EvidenceUnavailable("Gateway submission pending reconciliation; retry deadline reached")
                        print(f"[sandbox] Submission transport unavailable; retrying the SAME idempotency key in {delay}s",
                              file=sys.stderr, flush=True)
                        time.sleep(delay)
                        continue
                if rejection is None:
                    state = save("accepted", process)
        if rejection is not None:
            process = rejection.process
            state = save("rejected", process)
            if not rejection.retryable:
                return process
            rejection = None
            kind = "submission"
        else:
            job = job_from_process(process)
            if state.get("phase") != "terminal":
                process = poll(process, max(1, int(deadline - time.monotonic())))
                job = job_from_process(process)
                if not job or job.get("status") not in {"succeeded", "failed", "cancelled", "canceled"}:
                    # Keep the accepted ID so a later request polls that same job.
                    return process
                state = save("terminal", process)
            kind = retry_classifier(job)
            diagnostic = (embedded_device_failure(job)
                          if max_device_retries is not None and isinstance(job, dict) else None)
            if kind is None and diagnostic is not None:
                assert max_device_retries is not None
                # Keep the rejected raw Job in submission-N-terminal.json.
                if retries.get("device", 0) >= max_device_retries:
                    raise EvaluationDeviceUnavailable(
                        "Evaluation device unavailable; automatic retries exhausted. "
                        "No Kernel correctness verdict was recorded. Ask the operator to restore GPU availability."
                    )
                kind = "device"
            if kind is not None and idempotent_submission:
                # Stable replacement identity for the failed job, including crash replay.
                error = job.get("error")
                reason = error.get("reason") if isinstance(error, dict) else None
                prefix = "eval-device-retry:" if kind == "device" else (
                    "logs-retry:" if reason == "logs_unavailable" else "infra-retry:"
                )
                wire_key = prefix + hashlib.sha256(job["job_id"].encode()).hexdigest()
        if (
            kind is None
            or (max_retries is not None and sum(retries.values()) >= max_retries)
        ):
            return process
        retries[kind] = retries.get(kind, 0) + 1
        count = sum(retries.values())
        delay = device_retry_delay if kind == "device" and device_retry_delay is not None else retry_delay(count)
        if deadline - time.monotonic() <= delay:
            if kind == "device":
                raise EvaluationDeviceUnavailable("Evaluation device unavailable; retry deadline reached. No Kernel verdict was recorded.")
            return process
        print(f"[sandbox] Gateway {kind} failure; resubmitting after {delay}s (retry {count})",
              file=sys.stderr, flush=True)
        # Rejections and retryable infra terminals have a known outcome.
        # logs_unavailable + backend_state=succeeded needs a NEW job, not get().
        state = save("retry_pending")
        time.sleep(delay)
