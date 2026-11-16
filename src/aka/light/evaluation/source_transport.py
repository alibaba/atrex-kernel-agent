"""Native Eval archives and Runtime-owned whole-tree diagnostics.

Logical task identity contains exact sources, not expiring OSS URLs. Persist the
uploaded wire request before POST so an uncertain submission reuses that same
reservation and idempotency key on recovery.
"""
from __future__ import annotations
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import time
import urllib.parse
import zipfile

from aka.light.workspace.sources import archive_bytes

ARCHIVES = "__atrex_eval_archives"
DIAGNOSTIC = "__atrex_source_diagnostic"
MAX_INLINE_DEV_BYTES = 120 * 1024
PROFILES = {
    "pre": "https://pre-atrex-gateway.alibaba-inc.com",
    "prod": "https://atrex-gateway.alibaba-inc.com",
}
TERMINAL_JOB_STATES = {"succeeded", "failed", "cancelled", "canceled"}
SHORT_GET_TIMEOUT_SECONDS = 30
SHORT_POLL_INTERVAL_SECONDS = 2.0


def attach(request, workspace, *, side="candidate", source=None):
    from aka.light.workspace.sources import Artifact, read
    artifact = Artifact(source) if source is not None else read(workspace)
    packaged = artifact.execution_source(f"archives/{side}.tar.gz")
    if packaged.archives:
        request.setdefault(ARCHIVES, {}).update(packaged.archives)
    if side == "candidate":
        request["candidate"] = packaged.candidate
    else:
        request["abba"]["baseline"] = packaged.candidate
    requirements = list(dict.fromkeys([*request.get("requirements", []), *packaged.requirements]))
    if requirements:
        request["requirements"] = requirements
    return request


def _dev_archive(request):
    """Use Runtime's verified bootstrap for oversized Dev source payloads."""
    from aka.light.evaluation import oss_remote
    command = request.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ValueError("OSS Dev transport requires a nonempty command")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as bundle:
        for name, text in sorted(request["files"].items()):
            oss_remote.validate_path(name)
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system, info.external_attr = 3, 0o100644 << 16
            bundle.writestr(info, text.encode(), zipfile.ZIP_DEFLATED, 9)
    data = output.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    archive, driver = f"__atrex_oss_payload_{digest}.zip", f"__atrex_oss_unpack_{digest}.py"
    occupied = set(request["files"]) | {v["path"] for v in request.get("oss_files", [])}
    if archive in occupied or driver in occupied:
        raise ValueError("OSS transport staging paths conflict with request files")
    wire = dict(request, files={driver: Path(oss_remote.__file__).read_text()},
                command=f"python3 {shlex.quote(driver)} {shlex.quote(archive)} {digest} && (\n{command}\n)")
    return {archive: data}, wire


def uploaded_request(request, wire_key, *, json_call, upload, kind="eval"):
    from aka.light.experiment.store import JOB_ROOT_ENV, read_json, private_write
    files = request.get("files")
    large_dev = (kind == "dev" and isinstance(files, dict)
                 and all(isinstance(k, str) and isinstance(v, str) for k, v in files.items())
                 and sum(len(v.encode()) for v in files.values()) > MAX_INLINE_DEV_BYTES)
    if ARCHIVES not in request and not large_dev:
        return dict(request, idempotency_key=wire_key)
    key = hashlib.sha256(wire_key.encode()).hexdigest()
    root = os.environ.get(JOB_ROOT_ENV)
    path = Path(root) / "uploads" / (key + ".json") if root else None
    logical = hashlib.sha256(json.dumps([kind, request], sort_keys=True).encode()).hexdigest()
    if path is not None:
        try:
            saved = read_json(path)
        except FileNotFoundError:
            pass
        else:
            if saved["logical"] != logical:
                raise ValueError("Persisted Eval upload belongs to a different source request")
            return saved["wire"]
    if large_dev:
        packed, wire = _dev_archive(request)
    else:
        if kind not in {"eval", "profile"} or request.get("oss_files"):
            raise ValueError("Native Eval archives cannot mix existing upload references or Dev tasks")
        packed = {name: archive_bytes(files) for name, files in sorted(request[ARCHIVES].items())}
        wire = {k: v for k, v in request.items() if k != ARCHIVES}
    descriptors = [{"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                   for name, data in packed.items()]
    reservation = json_call("POST", "/v1/uploads", {"kind": kind, "spec": request["spec"], "files": descriptors}, 30)
    uploads = reservation.get("uploads")
    if not reservation.get("job_id") or not isinstance(uploads, list) or len(uploads) != len(packed):
        raise ValueError("Source upload reservation is incomplete")
    for descriptor, item in zip(descriptors, uploads, strict=True):
        if item.get("path") != descriptor["path"] or not item.get("put_url") or not item.get("upload_ref"):
            raise ValueError("Source upload reservation identity mismatch")
    for item in uploads:
        upload(item["put_url"], packed[item["path"]])
    wire["oss_files"] = [*wire.get("oss_files", []),
                         *({"path": v["path"], "upload_ref": v["upload_ref"]} for v in uploads)]
    wire["idempotency_key"] = "oss:" + hashlib.sha256(json.dumps([wire_key, wire], sort_keys=True).encode()).hexdigest()
    if path is not None:
        private_write(path, {"logical": logical, "wire": wire})
    return wire


def _agate_configuration(environment=None):
    """Read bounded Agate configuration with CLI-compatible precedence."""
    from aka.light.agent.session_tail import read_regular_bytes

    environment = os.environ if environment is None else environment
    configured_home = environment.get("HOME")
    home = Path(configured_home) if configured_home else Path.home()
    try:
        value = json.loads(
            read_regular_bytes(home / ".atrex/config.json", limit=64 * 1024)
        )
    except FileNotFoundError:
        return {}
    except (OSError, ValueError, UnicodeError) as error:
        from aka.light.runtime.request_errors import GatewayConfigurationError

        raise GatewayConfigurationError(
            "Supervisor Agate config is unreadable or invalid.",
            next_action=(
                "Ask the operator to repair ~/.atrex/config.json; do not change "
                "credentials or bypass the Runtime."
            ),
        ) from error
    if not isinstance(value, dict):
        from aka.light.runtime.request_errors import GatewayConfigurationError

        raise GatewayConfigurationError(
            "Supervisor Agate config must be a JSON object."
        )
    return value


def _load_sdk_client(
    args,
    environment=None,
    *,
    authority_url=None,
    authority_profile=None,
):
    """Construct the public Agate SDK client without exposing credentials."""
    minimum_agate_version = "0.14.30"
    try:
        from aka.light.agent.preflight import (
            MINIMUM_AGATE_VERSION,
            check_minimum_agate_version,
        )

        minimum_agate_version = MINIMUM_AGATE_VERSION
        check_minimum_agate_version()
        from atrex_gateway_client import AkSkAuth, Client, NoAuth, TokenAuth
        from atrex_gateway_client.errors import GatewayError, TransportError
    except ModuleNotFoundError as error:
        from aka.light.runtime.request_errors import GatewayConfigurationError

        raise GatewayConfigurationError(
            "atrex-gateway-client"
            f">={minimum_agate_version} is required for Gateway operations.",
            next_action=(
                "Ask the operator to install the declared Atrex AKA Light dependencies "
                "and restart the Supervisor; do not change Agent arguments or fall back to Dev."
            ),
        ) from error
    except RuntimeError as error:
        from aka.light.runtime.request_errors import GatewayConfigurationError

        raise GatewayConfigurationError(
            str(error),
            next_action=(
                f"Ask the operator to install atrex-gateway-client>={minimum_agate_version} "
                "and restart the Supervisor; do not retry or change Agent arguments."
            ),
        ) from error

    environment = os.environ if environment is None else environment
    config = _agate_configuration(environment)
    url = (
        authority_url
        or getattr(args, "url", None)
        or PROFILES.get(authority_profile or getattr(args, "gateway_profile", None))
        or environment.get("AGATE_URL")
        or config.get("url")
        or PROFILES["prod"]
    )
    parts = urllib.parse.urlsplit(url) if isinstance(url, str) else None
    if (
        parts is None
        or parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username
        or parts.password
    ):
        from aka.light.runtime.request_errors import GatewayConfigurationError

        raise GatewayConfigurationError("Agate base URL must be an HTTP(S) origin.")

    token = environment.get("AGATE_TOKEN") or config.get("token")
    access_key = environment.get("AGATE_AK") or config.get("ak")
    secret_key = environment.get("AGATE_SK") or config.get("sk")
    if any(
        value is not None and not isinstance(value, str)
        for value in (token, access_key, secret_key)
    ):
        from aka.light.runtime.request_errors import GatewayConfigurationError

        raise GatewayConfigurationError("Agate credential fields must be strings.")
    if token:
        auth = TokenAuth(token)
    elif access_key and secret_key:
        auth = AkSkAuth(access_key, secret_key)
    elif access_key or secret_key:
        from aka.light.runtime.request_errors import GatewayConfigurationError

        raise GatewayConfigurationError(
            "Agate access key and secret key must be configured together."
        )
    else:
        auth = NoAuth()

    class LightAgateClient(Client):
        @staticmethod
        def _overload_delay(error, attempt=0):
            # Runtime owns visible durable retries; do not sleep inside the SDK.
            return None

    client = LightAgateClient(
        url.rstrip("/"),
        auth=auth,
        timeout=min(600, max(1, int(getattr(args, "timeout", 600)))),
        transport_retries=0,
    )
    return client, url.rstrip("/"), GatewayError, TransportError


def validate_gateway_sdk(
    args,
    environment,
    *,
    authority_url=None,
    authority_profile=None,
):
    """Fail before staging when the configured SDK client is unavailable."""
    _load_sdk_client(
        args,
        environment,
        authority_url=authority_url,
        authority_profile=authority_profile,
    )


validate_eval_sdk = validate_gateway_sdk


def _sdk_upload(client, url, data):
    path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="atrex-light-source-", suffix=".archive", delete=False
        ) as output:
            output.write(data)
            path = output.name
        client.upload_file(url, path)
    finally:
        if path is not None:
            Path(path).unlink(missing_ok=True)


def _sdk_uploaded_request(client, request, wire_key, *, kind="eval"):
    def reserve(method, path, body, timeout):
        if method != "POST" or path != "/v1/uploads":
            raise ValueError("Unexpected Agate SDK upload reservation request")
        spec = body.get("spec") if isinstance(body, dict) else None
        targets = spec.get("target_hardware") if isinstance(spec, dict) else None
        if (
            not isinstance(targets, list)
            or len(targets) != 1
            or not isinstance(targets[0], str)
        ):
            raise ValueError("Agate SDK upload requires exactly one target hardware")
        return client.prepare_uploads(
            targets[0],
            body["files"],
            kind=body["kind"],
            num_gpus=spec.get("num_gpus", 1),
            placement=request.get("placement"),
        )

    return uploaded_request(
        request,
        wire_key,
        kind=kind,
        json_call=reserve,
        upload=lambda url, data: _sdk_upload(client, url, data),
    )


def _sdk_error_process(error, *, stage):
    payload = getattr(error, "payload", None)
    if not isinstance(payload, dict):
        payload = {
            "reason": "transport_failed",
            "error_class": "infra",
            "message": "Agate SDK transport failed",
        }
    status = getattr(error, "status", None)
    envelope = {"status": "failed", "error": payload}
    if isinstance(status, int):
        envelope["http_status"] = status
    stderr = f"agate-sdk: {stage} failed"
    if isinstance(status, int):
        stderr += f" (HTTP {status})"
    return subprocess.CompletedProcess(
        ["agate-sdk", stage], 1, json.dumps(envelope), stderr
    )


def run_sdk_job(
    gateway,
    args,
    request,
    sdk_kind,
    queue_wait_grace,
    *,
    identity_extra=None,
    eval_semantics=False,
):
    """Submit and collect one public Gateway job through the Agate SDK."""
    from aka.light.execution.gateway_jobs import (
        SubmissionRejected,
        SubmissionUncertain,
        annotate_deadline_timeout,
        configured_max_deadline_retries,
        configured_max_infra_retries,
        eval_retry_kind,
        execute_job,
        payload_identity,
        private_value_fingerprint,
        retry_kind,
    )
    from aka.light.experiment.store import EvidenceUnavailable

    client, url, GatewayError, TransportError = _load_sdk_client(args)
    prepared_wires = {}

    def submit(wire_key):
        try:
            wire = prepared_wires.get(wire_key)
            if wire is None:
                wire = _sdk_uploaded_request(
                    client, request, wire_key, kind=sdk_kind
                )
                prepared_wires[wire_key] = wire
        except GatewayError as error:
            process = _sdk_error_process(error, stage="upload")
            status = getattr(error, "status", None)
            retryable = (
                status is None
                or status in {408, 429}
                or (isinstance(status, int) and status >= 500)
            )
            raise SubmissionRejected(
                process, 503 if retryable else status
            ) from None
        except (OSError, ValueError) as error:
            raise SubmissionRejected(
                subprocess.CompletedProcess(
                    ["agate-sdk", "upload"],
                    1,
                    "",
                    f"source preparation failed: {error}",
                ),
                400,
            ) from None

        try:
            accepted = client.submit_job(sdk_kind, wire)
        except TransportError as error:
            raise SubmissionUncertain(
                _sdk_error_process(error, stage="submit")
            ) from None
        except GatewayError as error:
            process = _sdk_error_process(error, stage="submit")
            status = getattr(error, "status", None)
            if isinstance(status, int) and (status < 500 or status == 503):
                raise SubmissionRejected(process, status) from None
            raise SubmissionUncertain(process) from None
        job_id = accepted.get("job_id") if isinstance(accepted, dict) else None
        if not isinstance(job_id, str) or not job_id:
            raise SubmissionUncertain(
                subprocess.CompletedProcess(
                    ["agate-sdk", "submit"],
                    75,
                    "",
                    "Agate SDK returned no confirmed job ID",
                )
            )
        return subprocess.CompletedProcess(
            ["agate-sdk", "submit"], 0, json.dumps(accepted), ""
        )

    def poll(
        initial,
        remaining,
        *,
        recovery_probe=False,
        observation_callback=None,
    ):
        _ = recovery_probe
        accepted = json.loads(initial.stdout)
        job_id = accepted["job_id"]
        deadline = time.monotonic() + remaining
        job = accepted
        try:
            while True:
                if observation_callback is not None:
                    observation_callback(job)
                if job.get("status") in TERMINAL_JOB_STATES:
                    return subprocess.CompletedProcess(
                        ["agate-sdk", "get", job_id],
                        0 if job.get("status") == "succeeded" else 1,
                        json.dumps(job),
                        "",
                    )
                available = deadline - time.monotonic()
                if available <= 0:
                    raise EvidenceUnavailable(
                        "Gateway collection deadline reached; the accepted Job is saved for reconciliation"
                    )
                try:
                    observed = client.get_job(
                        job_id,
                        wait=False,
                        timeout=min(SHORT_GET_TIMEOUT_SECONDS, max(1, available)),
                    )
                    if isinstance(observed, dict):
                        job = observed
                        if observation_callback is not None:
                            observation_callback(job)
                except GatewayError as error:
                    status = getattr(error, "status", None)
                    if (
                        isinstance(status, int)
                        and status < 500
                        and status not in {404, 408, 429}
                    ):
                        return _sdk_error_process(error, stage="get")
                    # The accepted Job continues independently of this GET.
                    # Preserve the same Job ID and spend the remaining
                    # collection budget on short snapshot polls instead of
                    # converting one transient transport/visibility failure
                    # into an Agent-visible timeout.
                    _ = error
                if job.get("status") in TERMINAL_JOB_STATES:
                    continue
                delay = min(
                    SHORT_POLL_INTERVAL_SECONDS,
                    max(0.0, deadline - time.monotonic()),
                )
                if delay > 0:
                    time.sleep(delay)
        except (KeyboardInterrupt, SystemExit):
            try:
                client.cancel_job(job_id)
            except (GatewayError, OSError, ValueError):
                pass
            raise

    identity = {
        "url_fingerprint": private_value_fingerprint(url),
        "kind": sdk_kind,
        "payload": payload_identity(request),
    }
    if identity_extra:
        identity["execution_context"] = identity_extra
    process = execute_job(
        identity,
        submit,
        poll,
        wait_budget=args.timeout + queue_wait_grace,
        retry_classifier=eval_retry_kind if eval_semantics else retry_kind,
        max_infra_retries=configured_max_infra_retries(),
        max_deadline_retries=configured_max_deadline_retries(
            terminal_evaluation=eval_semantics,
        ),
        idempotent_submission=True,
        max_device_retries=10 if eval_semantics else None,
        device_retry_delay=60 if eval_semantics else None,
        capture_phase_observations=True,
    )
    timeout_s = request.get("timeout_s")
    if timeout_s is None and isinstance(request.get("options"), dict):
        timeout_s = request["options"].get("timeout_s")
    return annotate_deadline_timeout(process, timeout_s)


def run_eval_sdk(
    gateway,
    args,
    request,
    queue_wait_grace,
    *,
    identity_extra=None,
):
    """Submit inline and source-tree Eval/ABBA through one SDK lifecycle."""
    return run_sdk_job(
        gateway,
        args,
        request,
        "eval",
        queue_wait_grace,
        identity_extra=identity_extra,
        eval_semantics=True,
    )


def diagnostic_request(request, workspace, kind, *, timeout=600):
    from aka.light.workspace.sources import read
    artifact = read(workspace)
    route = artifact.layout.diagnostic_route(kind)
    if route in {"single", "typed"}:
        return attach(request, workspace, source=artifact.snapshot)
    files = artifact.files
    reference = request.get("reference", request.get("diagnostic_reference"))
    parameters = {key: request[key] for key in ("profiler", "level", "counters", "kernel_regex", "kernel_name",
                  "source", "launch_skip", "launch_count", "top_kernels", "arch", "sanitize", "fmt") if key in request}
    timeout = min(10800, timeout + 30)
    diagnostic = {"operation": kind, **artifact.diagnostic_source(), "requirements": request.get("requirements", []),
                  "shapes": reference["shapes"], "parameters": parameters, "lock_clocks": True,
                  "timeout_s": max(1, timeout - 30)}
    from aka.legacy.evaluation.resources import ncu_profile_script
    payload = {"spec": request["spec"], "command": "python3 __atrex_diagnostic.py request.json",
        "files": {**{"candidate/" + k: v.decode() for k, v in files.items()},
            "input.py": reference["input_py"], "request.json": json.dumps(diagnostic),
            "__atrex_diagnostic.py": Path(__file__).with_name("source_diagnostics.py").read_text(),
            "ncu_profile.py": ncu_profile_script().read_text(),
            "__atrex_abba.py": Path(__file__).with_name("abba_remote.py").read_text()},
        "env_vars": request.get("env_vars", {}), "timeout_s": timeout, "recycle": True,
        "dev_intent": "sanitize" if parameters.get("sanitize") else "profile_adhoc"}
    return dict(request, **{DIAGNOSTIC: payload})


def run(gateway, args, request, kind, queue_wait_grace, *, identity_extra=None):
    """Run typed and source-tree operations through one SDK lifecycle."""
    diagnostic = request.get(DIAGNOSTIC)
    if kind == "run" and diagnostic is None:
        return run_eval_sdk(
            gateway,
            args,
            request,
            queue_wait_grace,
            identity_extra=identity_extra,
        )
    sdk_kind = "dev" if diagnostic is not None else {
        "check": "compile",
        "disassemble": "disassemble",
        "profile": "profile",
    }.get(kind, kind)
    process = run_sdk_job(
        gateway,
        args,
        diagnostic or request,
        sdk_kind,
        queue_wait_grace,
        identity_extra=identity_extra,
    )
    if diagnostic and process.returncode == 0:
        from .source_diagnostics import DIAGNOSTIC_PREFIX
        from aka.light.experiment.store import EvidenceUnavailable
        job = gateway.parse_job_response(process.stdout)
        if not job or not isinstance(job.get("result"), dict):
            raise EvidenceUnavailable("Source-tree diagnostic returned no complete Job result")
        for line in reversed((job.get("result", {}).get("stdout") or "").splitlines()):
            if line.startswith(DIAGNOSTIC_PREFIX):
                try:
                    result = json.loads(line[len(DIAGNOSTIC_PREFIX):])
                    if result.get("operation") != kind or not isinstance(result.get("passed"), bool):
                        continue
                except (ValueError, TypeError, AttributeError):
                    continue
                job["source_tree_execution"] = job["result"]
                job["result"] = result
                return subprocess.CompletedProcess(process.args, 0 if result["passed"] else 1, json.dumps(job), process.stderr)
        raise EvidenceUnavailable("Source-tree diagnostic returned no complete result; inspect its saved Job")
    return process
