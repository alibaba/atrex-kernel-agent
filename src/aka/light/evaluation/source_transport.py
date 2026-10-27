"""Native Eval archives and Runtime-owned whole-tree diagnostics.

Logical task identity contains exact sources, not expiring OSS URLs. Persist the
uploaded wire request before POST so an uncertain submission reuses that same
reservation and idempotency key on recovery.
"""
from __future__ import annotations
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import urllib.parse
import zipfile

from aka.light.workspace.candidate import archive_bytes

ARCHIVES = "__atrex_eval_archives"
DIAGNOSTIC = "__atrex_source_diagnostic"
MAX_INLINE_DEV_BYTES = 4 * 1024 * 1024


class SourceUploadError(RuntimeError):
    """Archive preparation failed before the GPU job submission began."""

    def __init__(self, status=None):
        self.status = status
        super().__init__("Source archive upload failed" + (f" (HTTP {status})" if status else " (transport)"))


def attach(request, workspace, *, side="candidate", source=None):
    from aka.light.workspace.candidate import Artifact, read
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


def upload_bytes(url, data):
    # Match Agate's presigned PUT contract. urllib adds an unsigned Content-Type
    # to byte bodies, which makes OSS reject the signature. Never send Gateway
    # authorization or follow redirects with the storage credential.
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
        raise ValueError("Invalid source archive upload URL")
    connection = (http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection)(
        parts.hostname, parts.port, timeout=120)
    try:
        target = parts.path + ("?" + parts.query if parts.query else "")
        connection.request("PUT", target, body=data, headers={"Content-Length": str(len(data))})
        response = connection.getresponse()
        response.read(4096)
        if not 200 <= response.status < 300:
            raise SourceUploadError(response.status)
    except (OSError, http.client.HTTPException):
        # Exception strings may include the temporary credential-bearing URL.
        raise SourceUploadError() from None
    finally:
        connection.close()


def diagnostic_request(request, workspace, kind, *, timeout=600):
    from aka.light.workspace.candidate import read
    artifact = read(workspace)
    route = artifact.layout.diagnostic_route(kind)
    if route == "single":
        return attach(request, workspace, source=artifact.snapshot)
    if route == "typed":
        return attach(request, workspace, source=artifact.snapshot)
    files = artifact.files
    if request.get("profiler", "ncu") != "ncu" or request.get("fmt") == "isa":
        raise ValueError("Source-tree diagnostics currently support NVIDIA NCU and auto/sass/ptx")
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
    url = gateway._resolved_gateway_url(gateway.find_agate(), url=args.url, profile=args.gateway_profile)
    if not url:
        raise ValueError("Whole-tree transport requires a resolvable Agate URL and Supervisor HTTP credentials")
    diagnostic = request.get(DIAGNOSTIC)
    process = gateway.run_direct_job(url=url, kind="dev" if diagnostic else "profile" if kind == "profile" else "eval", payload=diagnostic or request,
        timeout=args.timeout, queue_wait_grace=queue_wait_grace, max_device_retries=10,
        device_retry_delay=60, identity_extra=identity_extra)
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
