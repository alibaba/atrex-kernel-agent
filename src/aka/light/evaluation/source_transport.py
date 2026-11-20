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
import zipfile

from aka.task.source.formats.common import archive_bytes

ARCHIVES = "__atrex_eval_archives"
ATTACHMENTS = "__atrex_eval_attachments"
MAX_INLINE_DEV_BYTES = 120 * 1024

def _dev_archive(request):
    """Use Runtime's verified bootstrap for oversized Dev source payloads."""
    from aka.light.evaluation import oss_remote
    command = request.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ValueError("OSS Dev transport requires a nonempty command")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as bundle:
        from aka.light.execution.package import decode_files
        for name, content in sorted(decode_files(request["files"]).items()):
            oss_remote.validate_path(name)
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system, info.external_attr = 3, 0o100644 << 16
            bundle.writestr(info, content, zipfile.ZIP_DEFLATED, 9)
    data = output.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    archive, driver = f"__atrex_oss_payload_{digest}.zip", f"__atrex_oss_unpack_{digest}.py"
    occupied = set(request["files"]) | {v["path"] for v in request.get("oss_files", [])}
    if archive in occupied or driver in occupied:
        raise ValueError("OSS transport staging paths conflict with request files")
    wire = dict(request, files={driver: Path(oss_remote.__file__).read_text()},
                command=f"python3 {shlex.quote(driver)} {shlex.quote(archive)} {digest} && (\n{command}\n)")
    return {archive: data}, wire


def uploaded_request(request, wire_key, *, json_call, upload, kind="eval", state_directory=None):
    from aka.light.tasks.content import atomic_json
    files = request.get("files")
    large_dev = (kind == "dev" and isinstance(files, dict)
                 and (any(not isinstance(v, str) for v in files.values())
                      or sum(len(v.encode()) for v in files.values()) > MAX_INLINE_DEV_BYTES))
    if ARCHIVES not in request and ATTACHMENTS not in request and not large_dev:
        return dict(request, idempotency_key=wire_key)
    key = hashlib.sha256(wire_key.encode()).hexdigest()
    root = state_directory or os.environ.get("AKA_JOB_ROOT")
    path = Path(root).resolve() / "uploads" / (key + ".json") if root else None
    logical = hashlib.sha256(json.dumps([kind, request], sort_keys=True).encode()).hexdigest()
    if path is not None:
        try:
            saved = json.loads(path.read_text())
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
        from aka.light.execution.package import decode_files
        packed = {name: archive_bytes(files) for name, files in sorted(request.get(ARCHIVES, {}).items())}
        packed.update(decode_files(request.get(ATTACHMENTS, {})))
        wire = {k: v for k, v in request.items() if k not in {ARCHIVES, ATTACHMENTS}}
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
        atomic_json(path, {"logical": logical, "wire": wire})
    return wire
