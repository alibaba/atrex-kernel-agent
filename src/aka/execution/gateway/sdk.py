"""The shared Execution owner's only Agate control-plane transport."""
import json
import os
import subprocess
import tempfile
import urllib.parse
from pathlib import Path
from types import SimpleNamespace
from aka.contracts.execution import ExecutionUnavailable

PROFILES = {"pre": "https://pre-atrex-gateway.alibaba-inc.com", "prod": "https://atrex-gateway.alibaba-inc.com"}
class GatewayConfigurationError(ExecutionUnavailable):
    def __init__(self, message, **details):
        super().__init__(message)
        self.details = details

def _agate_configuration(environment=None):
    """Read bounded Agate configuration with CLI-compatible precedence."""
    from aka.core.filesystem import read_regular_bytes

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
        from aka.execution.gateway.sdk import GatewayConfigurationError

        raise GatewayConfigurationError(
            "Supervisor Agate config is unreadable or invalid.",
            next_action=(
                "Ask the operator to repair ~/.atrex/config.json; do not change "
                "credentials or bypass the Runtime."
            ),
        ) from error
    if not isinstance(value, dict):
        from aka.execution.gateway.sdk import GatewayConfigurationError

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
        from aka.agent.preflight import (
            MINIMUM_AGATE_VERSION,
            check_minimum_agate_version,
        )

        minimum_agate_version = MINIMUM_AGATE_VERSION
        check_minimum_agate_version()
        from atrex_gateway_client import AkSkAuth, Client, NoAuth, TokenAuth
        from atrex_gateway_client.errors import GatewayError, TransportError
    except ModuleNotFoundError as error:
        from aka.execution.gateway.sdk import GatewayConfigurationError

        raise GatewayConfigurationError(
            "atrex-gateway-client"
            f">={minimum_agate_version} is required for Gateway operations.",
            next_action=(
                "Ask the operator to install the declared Atrex AKA Light dependencies "
                "and restart the Supervisor; do not change Agent arguments or fall back to Dev."
            ),
        ) from error
    except RuntimeError as error:
        from aka.execution.gateway.sdk import GatewayConfigurationError

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
        from aka.execution.gateway.sdk import GatewayConfigurationError

        raise GatewayConfigurationError("Agate base URL must be an HTTP(S) origin.")

    token = environment.get("AGATE_TOKEN") or config.get("token")
    access_key = environment.get("AGATE_AK") or config.get("ak")
    secret_key = environment.get("AGATE_SK") or config.get("sk")
    if any(
        value is not None and not isinstance(value, str)
        for value in (token, access_key, secret_key)
    ):
        from aka.execution.gateway.sdk import GatewayConfigurationError

        raise GatewayConfigurationError("Agate credential fields must be strings.")
    if token:
        auth = TokenAuth(token)
    elif access_key and secret_key:
        auth = AkSkAuth(access_key, secret_key)
    elif access_key or secret_key:
        from aka.execution.gateway.sdk import GatewayConfigurationError

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

def _sdk_error_process(error, *, stage):
    payload = getattr(error, "payload", None)
    if payload is None:
        payload = {
            "code": "agate_sdk_transport_failed",
            "error_class": "infra",
            "message": str(error),
            "exception_type": type(error).__name__,
        }
    status = getattr(error, "status", None)
    envelope = {"status": "failed", "error": payload}
    if isinstance(status, int):
        envelope["http_status"] = status
    return subprocess.CompletedProcess(
        ["agate-sdk", stage], 1, json.dumps(envelope), ""
    )

def client(*, endpoint="", profile="", timeout=30, environment=None):
    return _load_sdk_client(SimpleNamespace(url=endpoint, gateway_profile=profile, timeout=timeout), environment)[0]


def route(endpoint="", profile="", environment=None):
    env = os.environ if environment is None else environment
    config = _agate_configuration(env)
    return endpoint or PROFILES.get(profile) or env.get("AGATE_URL") or config.get("url") or PROFILES["prod"]


def request(endpoint, method, path, payload, timeout, *, environment=None, allow_list=False):
    """Job recovery belongs to the job owner; retry Env/upload infra here."""
    if urllib.parse.urlsplit(path).path.startswith('/v1/jobs/'):
        return _request_once(endpoint, method, path, payload, timeout,
                             environment=environment, allow_list=allow_list)
    import time
    from .admission_retry import infrastructure_failure, policy_from_environment, retry_delay
    from .errors import GatewayHTTPError
    policy = policy_from_environment(environment)
    deadline = time.monotonic() + timeout
    recovery_deadline = None
    for index in range(len(policy.delays) + 1):
        failure = None
        try:
            result = _request_once(endpoint, method, path, payload, max(0, deadline - time.monotonic()),
                                   environment=environment, allow_list=allow_list)
        except GatewayHTTPError as error:
            failure = error
            delay = retry_delay(error, index, policy)
        else:
            if not infrastructure_failure(result):
                return result
            delay = policy.delays[index] if index < len(policy.delays) else None
        if recovery_deadline is None:
            recovery_deadline = time.monotonic() + policy.max_elapsed_s
        limit = min(deadline, recovery_deadline)
        if delay is not None and time.monotonic() + delay < limit:
            time.sleep(delay)
            if time.monotonic() < limit:
                continue
        if failure is not None:
            raise failure
        return result


def _request_once(endpoint, method, path, payload, timeout, *, environment=None, allow_list=False):
    """Adapt the retained owner request protocol to public SDK methods only."""
    from atrex_gateway_client.errors import GatewayError
    from .errors import GatewayHTTPError
    api = client(endpoint=endpoint, timeout=timeout, environment=environment)
    parsed = urllib.parse.urlsplit(path)
    bits = [urllib.parse.unquote(x) for x in parsed.path.strip('/').split('/')]
    params = urllib.parse.parse_qs(parsed.query)
    force = params.get('force', ['false'])[0] == 'true'
    try:
        if bits == ['v1','uploads'] and method == 'POST':
            spec = payload.get("spec", {})
            hardware = spec.get("target_hardware", [])
            if not isinstance(hardware, list) or len(hardware) != 1:
                raise ValueError("Agate upload reservation requires one target hardware")
            return api.prepare_uploads(hardware[0], payload["files"],
                kind=payload.get("kind", "dev"), num_gpus=spec.get("num_gpus", 1),
                **({"placement": payload["placement"]} if "placement" in payload else {}))
        if bits[:2] == ['v1','jobs']:
            if method == 'POST' and len(bits)==3:
                return api.submit_job(bits[2], payload)
            if method == 'GET' and len(bits)==3:
                return api.get_job(bits[2], wait=False, timeout=timeout)
            if method == 'POST' and len(bits)==4 and bits[3]=='cancel':
                return api.cancel_job(bits[2])
        if bits[:2] == ['v1','env'] and method == 'GET':
            from aka.core.service_audit import audit_service_exchange
            import uuid
            name = ('list_env' if len(bits) == 2 else 'get_env' if len(bits) == 3
                    else 'get_capabilities' if len(bits) == 4 and bits[3] == 'capabilities' else None)
            if name is not None:
                exchange_id = uuid.uuid4().hex
                arguments = {"force": force, **({"gpu": bits[2]} if len(bits) > 2 else {})}
                audit_service_exchange(service="agate", exchange_id=exchange_id,
                    request={"method": name, **arguments})
                try:
                    result = getattr(api, name)(*([bits[2]] if len(bits) > 2 else []), force=force)
                except Exception as error:
                    audit_service_exchange(service="agate", exchange_id=exchange_id,
                        response={"status": "error", "http_status": getattr(error, "status", None),
                            "payload": getattr(error, "payload", None), "error_type": type(error).__name__})
                    raise
                audit_service_exchange(service="agate", exchange_id=exchange_id, response=result)
                return result
        raise ValueError(f"No supported Agate SDK operation for {method} {parsed.path}")
    except GatewayError as error:
        response = _sdk_error_process(error, stage="request")
        raise GatewayHTTPError(getattr(error,'status',None), response.stdout,
            retry_after_s=getattr(error, 'retry_after_s', None)) from error


def upload_bytes(url, data, *, timeout):
    api = client(timeout=timeout)
    path = None
    try:
        with tempfile.NamedTemporaryFile(prefix='aka-sdk-upload-', delete=False) as output:
            path = Path(output.name)
            output.write(data)
        api.upload_file(url, str(path))
    finally:
        if path is not None:path.unlink(missing_ok=True)


def submission_rejection(process):
    if process.returncode == 0:
        return None
    try:
        value = json.loads(process.stdout)
    except (ValueError, TypeError):
        return None
    if not isinstance(value, dict) or value.get("job_id"):
        return None
    status = value.get("http_status", value.get("status_code"))
    if type(status) is int and 400 <= status < 600 and (status < 500 and status != 408 or status == 503):
        return SimpleNamespace(process=process, status=status, retryable=status in {429,503})
    return None
