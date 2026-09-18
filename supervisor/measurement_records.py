"""Private, durable measurement facts and task reservations (not Agent journals)."""
from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import re
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from orchestrator.agent_home import open_private_directory
from orchestrator.durable_state import durable_write_json, durable_write_text
from orchestrator.session_tail import read_regular_bytes

RECORD_ID = re.compile(r"gateway-[0-9a-f]{32}")
KERNEL_ID = re.compile(r"kernel-[0-9a-f]{32}")
JOB_ROOT_ENV = "ATREX_AKA_MEASUREMENT_JOB_ROOT"
MAX_RECORD_BYTES = 32 * 1024 * 1024


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def source_identity(source: bytes) -> dict:
    checksum = "sha256:" + hashlib.sha256(source).hexdigest()
    return {"kernel_id": "kernel-" + uuid.uuid5(uuid.NAMESPACE_URL, f"urn:atrex:kernel-source:{checksum}").hex,
            "kernel_artifact_digest": checksum}


def read_json(path: Path, *, limit: int = MAX_RECORD_BYTES) -> dict:
    data = read_regular_bytes(path, limit=limit + 1)
    if len(data) > limit:
        raise ValueError("Measurement evidence exceeds its size limit")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("Measurement evidence must be an object")
    return value


def private_write(path: Path, value: object) -> None:
    # Store trees are Supervisor-owned; reject links before atomic publication.
    with open_private_directory(path.parent):
        durable_write_json(path, value)


def private_write_bytes(path: Path, data: bytes) -> None:
    with open_private_directory(path.parent) as parent:
        temporary = ".record-" + uuid.uuid4().hex
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path.name, src_dir_fd=parent, dst_dir_fd=parent)
            os.fsync(parent)
        finally:
            try:
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass


class EvidenceUnavailable(RuntimeError):
    """A storage outage must not cause a new GPU submission."""


class DuplicateTask(RuntimeError):
    def __init__(self, record_id: str | None):
        self.record_id = record_id
        super().__init__("Identical Gateway task is already recorded" if record_id else
                         "Identical Gateway task is already running")

    def response(self) -> dict:
        return {"code": "duplicate_gateway_task", "message": str(self),
                "gateway_record_id": self.record_id,
                "next_action": (
                    "Read the recorded result: python3 tools/sandbox.py --kind record-read --record-id "
                    + self.record_id if self.record_id else
                    "Wait for the original request; do not launch a second identical job.")}


class MeasurementStore:
    def __init__(self, root: Path):
        self.root = root
        with open_private_directory(root):
            pass

    def kernel(self, source: bytes) -> dict:
        identity = source_identity(source)
        directory = self.root / "kernels" / identity["kernel_id"]
        with open_private_directory(directory):
            pass
        path = directory / "kernel.py"
        # Atomic identical writes are harmless; an existing corrupted artifact
        # is not silently repaired or presented as measured source.
        try:
            existing = read_regular_bytes(path, limit=16 * 1024 * 1024 + 1)
        except FileNotFoundError:
            durable_write_text(path, source.decode("utf-8"))
        else:
            if existing != source:
                raise ValueError("Kernel artifact does not match its content identity")
        return identity

    def read_kernel(self, kernel_id: str) -> bytes:
        if not KERNEL_ID.fullmatch(kernel_id):
            raise ValueError("Use a kernel- ID returned by a Gateway measurement")
        try:
            source = read_regular_bytes(self.root / "kernels" / kernel_id / "kernel.py",
                                        limit=16 * 1024 * 1024 + 1)
        except FileNotFoundError as error:
            raise ValueError("Kernel is not available in this Campaign; use an ID returned here") from error
        if len(source) > 16 * 1024 * 1024 or source_identity(source)["kernel_id"] != kernel_id:
            raise ValueError("Kernel artifact identity or size is invalid")
        return source

    def read(self, record_id: str) -> dict:
        if not RECORD_ID.fullmatch(record_id):
            raise ValueError("Use a gateway- ID returned by a Gateway measurement")
        record = read_json(self.root / "records" / record_id / "record.json")
        if record.get("gateway_record_id") != record_id or record.get("schema_version") != 1:
            raise ValueError("Gateway record identity or version is invalid")
        if record.get("task_digest") != digest(record["request"]):
            raise ValueError("Gateway record request identity is inconsistent")
        if record.get("result_digest") != digest(record["response"]):
            raise ValueError("Gateway record result integrity check failed")
        for side, identity in record["kernels"].items():
            source = self.read_kernel(identity["kernel_id"])
            filename = "kernel.py" if side == "candidate" else record["request"]["options"]["baseline_path"]
            if (source_identity(source) != identity
                    or hashlib.sha256(source).hexdigest() != record["request"]["inputs"][filename]):
                raise ValueError("Gateway record refers to a different Kernel")
        return record

    def _cached(self, record_id: str, task_digest: str) -> dict | None:
        # Broken/old indexes are misses. Transient IO failures retain the index
        # and stop dispatch, including on the publication path.
        for delay in (0, .05, .1):
            if delay:
                time.sleep(delay)
            try:
                record = self.read(record_id)
                if record.get("task_digest") != task_digest or record.get("cacheable") is not True:
                    return None
                return record
            except (ValueError, KeyError, TypeError, FileNotFoundError, NotADirectoryError, IsADirectoryError):
                return None
            except OSError as error:
                if error.errno == errno.ELOOP:
                    return None
                last_error = error
        raise EvidenceUnavailable(
            f"Recorded measurement {record_id} is temporarily unreadable; "
            "no replacement job was submitted. Restore storage access, then retry."
        ) from last_error

    @contextmanager
    def reserve(self, request: dict, kernels: dict):
        key = digest(request)
        with open_private_directory(self.root / "tasks") as directory:
            descriptor = os.open(key + ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600, dir_fd=directory)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise DuplicateTask(None) from None
            marker = self.root / "tasks" / (key + ".json")
            try:
                state = read_json(marker, limit=4096)
            except (FileNotFoundError, ValueError):
                state = {}
            record_id = state.get("gateway_record_id")
            if isinstance(record_id, str) and RECORD_ID.fullmatch(record_id):
                cached = self._cached(record_id, key)
                if cached is not None:
                    raise DuplicateTask(record_id)
            # Resume an interrupted request's physical jobs/checkpoints only
            # when its exact request identity remains intact.
            directory = self.root / "records" / str(record_id)
            resumable = False
            if isinstance(record_id, str) and RECORD_ID.fullmatch(record_id) and state.get("status") == "running":
                try:
                    resumable = read_json(directory / "request.json") == request
                except (FileNotFoundError, ValueError):
                    pass
            if not resumable:
                record_id = "gateway-" + uuid.uuid4().hex
                directory = self.root / "records" / record_id
                private_write(directory / "request.json", request)
            private_write(marker, {"status": "running", "gateway_record_id": record_id})
            yield MeasurementTask(self, key, record_id, directory, request, kernels, marker)
        finally:
            os.close(descriptor)

    def kernel_records(self, kernel_id: str) -> list[dict]:
        self.read_kernel(kernel_id)  # Validate identity and visibility first.
        rows = []
        for path in sorted((self.root / "records").glob("gateway-*/record.json")):
            record = self.read(path.parent.name)
            if any(item["kernel_id"] == kernel_id for item in record["kernels"].values()):
                rows.append({"gateway_record_id": record["gateway_record_id"],
                             "operation": record["operation"], "created_at": record["created_at"]})
        return rows


class MeasurementTask:
    def __init__(self, store, key, record_id, directory, request, kernels, marker):
        self.store, self.key, self.record_id = store, key, record_id
        self.directory, self.request, self.kernels, self.marker = directory, request, kernels, marker

    def finish(self, response: dict, *, cacheable: bool, pending: bool = False) -> dict:
        record = {"schema_version": 1, "gateway_record_id": self.record_id,
                  "task_digest": self.key, "request": self.request, "kernels": self.kernels,
                  "operation": self.request["operation"], "response": response,
                  "result_digest": digest(response), "cacheable": cacheable,
                  "created_at": datetime.now(timezone.utc).isoformat()}
        private_write(self.directory / "record.json", record)
        # Keep a reference even if post-write validation encounters transient IO.
        private_write(self.marker, {"status": "validation_pending", "gateway_record_id": self.record_id})
        if cacheable and self.store._cached(self.record_id, self.key) is None:
            raise ValueError("Recorded measurement failed validation")
        private_write(self.marker, {"status": "running" if pending else "completed" if cacheable else "not_cacheable",
                                    "gateway_record_id": self.record_id})
        return record
