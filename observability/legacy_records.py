"""Read-only compatibility for main's v1 measurement records."""
import hashlib, json, re, uuid
from pathlib import Path
from aka.core.filesystem import read_regular_bytes
from aka.task.source.formats import Artifact, MAX_SNAPSHOT

RECORD_ID = re.compile(r"gateway-[0-9a-f]{32}")
KERNEL_ID = re.compile(r"kernel-[0-9a-f]{32}")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_identity(source):
    checksum = "sha256:" + hashlib.sha256(source).hexdigest()
    return {"kernel_id": "kernel-" + uuid.uuid5(uuid.NAMESPACE_URL, f"urn:atrex:kernel-source:{checksum}").hex,
            "kernel_artifact_digest": checksum}


class LegacyMeasurementReader:
    def __init__(self, root):
        self.root = Path(root)

    def record_ids(self):
        return tuple(p.name for p in sorted((self.root / "records").glob("gateway-*")) if p.is_dir() and not p.is_symlink() and RECORD_ID.fullmatch(p.name))

    def record_directory(self, record_id):
        if not RECORD_ID.fullmatch(record_id):
            raise ValueError("invalid v1 record identity")
        return self.root / "records" / record_id

    def read_kernel(self, kernel_id):
        if not KERNEL_ID.fullmatch(kernel_id):
            raise ValueError("invalid v1 kernel identity")
        source = read_regular_bytes(self.root / "kernels" / kernel_id / "kernel.py", limit=MAX_SNAPSHOT + 1)
        if len(source) > MAX_SNAPSHOT or source_identity(source)["kernel_id"] != kernel_id:
            raise ValueError("v1 kernel identity or size is invalid")
        return source

    def read(self, record_id):
        record = json.loads(read_regular_bytes(self.record_directory(record_id) / "record.json", limit=32 * 1024 * 1024))
        if record.get("gateway_record_id") != record_id or record.get("schema_version") != 1:
            raise ValueError("v1 record identity or version is invalid")
        if record.get("task_digest") != digest(record["request"]) or record.get("result_digest") != digest(record["response"]):
            raise ValueError("v1 record integrity check failed")
        for side, identity in record["kernels"].items():
            source = self.read_kernel(identity["kernel_id"])
            if source_identity(source) != identity or not Artifact(source).verify_measurement_inputs(record["request"], side):
                raise ValueError("v1 record source evidence changed")
        return record
