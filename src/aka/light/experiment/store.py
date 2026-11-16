"""Private, durable measurement facts and task reservations (not Agent journals)."""
from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import logging
import math
import os
import re
import secrets
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from aka.light.agent.home import open_private_directory
from aka.legacy.task.persistence import durable_write_json, durable_write_text
from aka.light.agent.session_tail import read_regular_bytes
from aka.light.workspace.sources import MAX_SNAPSHOT, Artifact

RECORD_ID = re.compile(r"gateway-[0-9a-f]{32}")
KERNEL_ID = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
JOB_ROOT_ENV = "ATREX_AKA_MEASUREMENT_JOB_ROOT"
ABBA_BATCH_ROOT_ENV = "ATREX_AKA_ABBA_BATCH_ROOT"
BATCH_ROOT_ENV = "ATREX_AKA_SHAPE_BATCH_ROOT"
RUNTIME_REQUEST_ID_ENV = "ATREX_AKA_RUNTIME_REQUEST_ID"
MAX_RECORD_BYTES = 32 * 1024 * 1024
MAX_RECORD_HEADER_BYTES = 16 * 1024
RECORD_HEADER_FIELDS = frozenset({"gateway_record_id", "operation", "created_at", "kernels"})


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def source_identity(source: bytes, kernel_id: str) -> dict:
    if not isinstance(kernel_id, str) or KERNEL_ID.fullmatch(kernel_id) is None:
        raise ValueError("kernel_commit_id must be a complete Git commit ID")
    checksum = "sha256:" + hashlib.sha256(source).hexdigest()
    return {"kernel_id": kernel_id, "kernel_artifact_digest": checksum}


def read_json(path: Path, *, limit: int = MAX_RECORD_BYTES) -> dict:
    data = read_regular_bytes(path, limit=limit + 1)
    if len(data) > limit:
        raise ValueError("Measurement evidence exceeds its size limit")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("Measurement evidence must be an object")
    return value


def read_record_header(path: Path) -> dict:
    """Read bounded listing metadata, not a validated measurement.

    private_write sorts JSON keys, placing these fields before request/response.
    Stop decoding once the header is complete; never load a response to list it.
    Noncanonical or oversized headers are skipped by the listing caller.
    """
    # Canonical records use ASCII JSON escapes, so a bounded prefix cannot
    # split a UTF-8 code point in one of our records.
    text = read_regular_bytes(path, limit=MAX_RECORD_HEADER_BYTES).decode("utf-8")
    decoder = json.JSONDecoder()
    header, seen = {}, set()

    def whitespace(index):
        while index < len(text) and text[index] in " \t\r\n":
            index += 1
        return index

    index = whitespace(0)
    if text[index:index + 1] != "{":
        raise ValueError("Gateway record header must be an object")
    index = whitespace(index + 1)
    while True:
        key, index = decoder.raw_decode(text, index)
        if not isinstance(key, str) or key in seen:
            raise ValueError("Gateway record header has an invalid or duplicate key")
        seen.add(key)
        index = whitespace(index)
        if text[index:index + 1] != ":":
            raise ValueError("Gateway record header is malformed")
        # Refuse to decode large payload fields if required metadata is absent
        # or misplaced. The listing must not turn into a full-record read.
        if key not in RECORD_HEADER_FIELDS and key != "cacheable":
            raise ValueError("Gateway record listing header is incomplete")
        value, index = decoder.raw_decode(text, whitespace(index + 1))
        if key in RECORD_HEADER_FIELDS:
            header[key] = value
        index = whitespace(index)
        separator = text[index:index + 1]
        if separator not in {",", "}"}:
            raise ValueError("Gateway record header is malformed or exceeds its size limit")
        if RECORD_HEADER_FIELDS <= header.keys():
            return header
        if separator != ",":
            raise ValueError("Gateway record listing header is incomplete")
        index = whitespace(index + 1)


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


class RecordNotFound(ValueError):
    """A well-formed ID does not exist in this Campaign."""


class DuplicateTask(RuntimeError):
    def __init__(self, record_id: str | None):
        self.record_id = record_id
        super().__init__("Identical Gateway task is already recorded" if record_id else
                         "Identical Gateway task is already running")

    def response(self) -> dict:
        raise RuntimeError(
            "Completed exact Gateway tasks are internal cache hits, not Agent errors"
        )


class TaskRunning(DuplicateTask):
    """An identical local call still owns its reservation, not a cached result."""

    def __init__(self, deadline: float | None = None, request_id: str = ""):
        super().__init__(None)
        self.deadline = deadline
        self.request_id = request_id

    def response(self) -> dict:
        remaining = (max(0, math.ceil(self.deadline - time.monotonic()))
                     if self.deadline is not None else None)
        message = "The same Agate call is still running. No new Agate task was submitted."
        if remaining:
            message += (f" The original request has about {remaining} seconds left before its deadline. "
                        "Expect a result or timeout response within that budget, plus cleanup time; "
                        "this is not a guarantee of successful Agate completion.")
        elif remaining == 0:
            message += " Its request deadline has elapsed; cleanup is still in progress. Completion time is unknown."
        else:
            message += " Its remaining request time is unavailable; completion time is unknown."
        return {
            "code": "gateway_task_running", "message": message,
            "status": "running", "gateway_record_id": None,
            "original_request_id": self.request_id or None,
            "remaining_seconds": remaining, "estimate_basis": "runtime_request_deadline",
            "completion_guaranteed": False,
            "next_action": "Wait for the original call's response; do not submit this same task again while it is running.",
        }


def _running_task(descriptor: int) -> TaskRunning:
    # Lock metadata is advisory and bounded. Missing/old metadata must never
    # turn a live reservation into a cache hit or invent a fresh deadline.
    try:
        value = json.loads(os.pread(descriptor, 4096, 0))
        deadline, request_id = value.get("deadline"), value.get("request_id", "")
        if type(deadline) not in (int, float) or not math.isfinite(deadline):
            deadline = None
        if not isinstance(request_id, str):
            request_id = ""
        return TaskRunning(deadline, request_id)
    except (OSError, ValueError, TypeError, AttributeError):
        return TaskRunning()


class MeasurementStore:
    def __init__(self, root: Path):
        self.root = root
        self._active_tasks: dict[str, TaskRunning] = {}
        self._active_lock = threading.Lock()
        with open_private_directory(root) as directory:
            lock = os.open("identity.lock", os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW,
                           0o600, dir_fd=directory)
            try:
                fcntl.flock(lock, fcntl.LOCK_EX)
                try:
                    descriptor = os.open(
                        "identity.key", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                        0o600, dir_fd=directory,
                    )
                except FileExistsError:
                    pass
                else:
                    with os.fdopen(descriptor, "wb") as output:
                        output.write(secrets.token_bytes(32))
                        output.flush()
                        os.fsync(output.fileno())
                    os.fsync(directory)
                self.identity_key = read_regular_bytes(root / "identity.key", limit=33)
            finally:
                os.close(lock)
        if len(self.identity_key) != 32:
            raise ValueError("Measurement identity key is invalid; restore its private state")

    def kernel(
        self,
        source: bytes,
        kernel_id: str,
    ) -> dict:
        identity = source_identity(source, kernel_id)
        directory = self.root / "kernels" / identity["kernel_id"]
        with open_private_directory(directory):
            pass
        path = directory / "kernel.py"
        # Atomic identical writes are harmless; an existing corrupted artifact
        # is not silently repaired or presented as measured source.
        try:
            existing = read_regular_bytes(path, limit=MAX_SNAPSHOT + 1)
        except FileNotFoundError:
            durable_write_text(path, source.decode("utf-8"))
        else:
            if existing != source:
                raise ValueError("Git commit ID is already bound to different Kernel source")
        digest_directory = (
            self.root
            / "source-index"
            / identity["kernel_artifact_digest"].removeprefix("sha256:")
        )
        with open_private_directory(digest_directory):
            pass
        marker = digest_directory / kernel_id
        try:
            value = read_regular_bytes(marker, limit=65).decode()
        except FileNotFoundError:
            durable_write_text(marker, kernel_id)
        else:
            if value != kernel_id:
                raise ValueError("Kernel source index is inconsistent")
        return identity

    def find_kernel_id(self, source: bytes) -> str:
        """Resolve an already recorded source to one preserved Git commit ID."""
        digest_directory = self.root / "source-index" / hashlib.sha256(source).hexdigest()
        matches = []
        for marker in digest_directory.glob("*"):
            if not marker.is_file() or KERNEL_ID.fullmatch(marker.name) is None:
                continue
            try:
                existing = read_regular_bytes(
                    self.root / "kernels" / marker.name / "kernel.py",
                    limit=MAX_SNAPSHOT + 1,
                )
            except (OSError, ValueError):
                continue
            if existing == source:
                matches.append(marker.name)
        if not matches:
            raise RecordNotFound(
                "Kernel source has no recorded Git commit ID; commit and evaluate it first"
            )
        return sorted(matches)[0]

    def read_kernel(self, kernel_id: str) -> bytes:
        if not KERNEL_ID.fullmatch(kernel_id):
            raise ValueError("Use a Git commit ID returned by a Gateway measurement")
        try:
            source = read_regular_bytes(self.root / "kernels" / kernel_id / "kernel.py",
                                        limit=MAX_SNAPSHOT + 1)
        except FileNotFoundError as error:
            raise RecordNotFound("Kernel is not available in this Campaign; use an ID returned here") from error
        if len(source) > MAX_SNAPSHOT:
            raise ValueError("Kernel artifact size is invalid")
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
            if source_identity(source, identity["kernel_id"]) != identity or not Artifact(source).verify_measurement_inputs(record["request"], side):
                raise ValueError("Gateway record omits or changes candidate source evidence")
        return record

    def record_ids(self) -> list[str]:
        """List validated terminal Record identities without exposing payloads."""
        values = []
        for path in (self.root / "records").glob("gateway-*/record.json"):
            record_id = path.parent.name
            if not RECORD_ID.fullmatch(record_id):
                continue
            try:
                record = self.read(record_id)
            except (OSError, ValueError, KeyError, TypeError, RecursionError):
                logging.getLogger(__name__).warning(
                    "Skipping unreadable Gateway record while exporting shared evidence: %s",
                    record_id,
                )
                continue
            if record.get("gateway_record_id") == record_id:
                values.append(record_id)
        return sorted(values)

    def import_record(self, source: "MeasurementStore", record_id: str) -> dict:
        """Import immutable evidence without importing its task/cache identity."""
        record = source.read(record_id)
        for identity in record["kernels"].values():
            kernel_source = source.read_kernel(identity["kernel_id"])
            if self.kernel(kernel_source, identity["kernel_id"]) != identity:
                raise ValueError("Imported Kernel identity differs from the Gateway Record")
        destination = self.root / "records" / record_id / "record.json"
        try:
            existing = self.read(record_id)
        except (FileNotFoundError, RecordNotFound):
            private_write(destination, record)
            existing = self.read(record_id)
        if existing != record:
            raise ValueError(f"Gateway Record collision while importing {record_id}")
        return existing

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
                if record["operation"] in {"evaluate", "same_allocation_abba"}:
                    from aka.light.evaluation.public_result import cacheable_evaluation
                    if not cacheable_evaluation(record["response"], record["request"]):
                        # Older Dev-backed evaluators cached allocation success
                        # even when no evaluation receipt existed. Keep the old
                        # record intact, but do not use it as a permanent verdict.
                        return None
                from aka.light.evaluation.evaluation_failures import embedded_device_failure
                from aka.light.evaluation.result_json import parse_evaluation_result_line
                negative = False
                for line in record["response"].get("stdout", "").splitlines():
                    result = parse_evaluation_result_line(line)
                    if result is not None:
                        if embedded_device_failure({"status": "succeeded", "result": result}):
                            return None
                        negative = negative or result.get("all_pass", result.get("correct")) is False
                        continue
                    prefix = "[sandbox] ABBA_JSON="
                    if line.startswith(prefix):
                        result = json.loads(line[len(prefix):])
                        if not isinstance(result, dict):
                            return None
                        if embedded_device_failure({"status": "succeeded", "result": result}):
                            return None
                        negative = negative or result.get("all_pass", result.get("correct")) is False
                if negative:
                    # Old projections could hide the device error. Inspect sealed
                    # terminal jobs only for failed Eval/ABBA records, not all history.
                    for path in (self.root / "records" / record_id).glob("**/jobs/*/state.json"):
                        state = read_json(path)
                        process = state.get("process", {})
                        try:
                            job = json.loads(process.get("stdout", ""))
                        except ValueError:
                            continue
                        if isinstance(job, dict) and embedded_device_failure(job):
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

    def check_running(self, request: dict) -> None:
        """Probe this Supervisor's active calls without taking their file lock."""
        key = digest(request)
        with self._active_lock:
            active = self._active_tasks.get(key)
        if active is not None:
            raise TaskRunning(active.deadline, active.request_id)

    @contextmanager
    def reserve(self, request: dict, kernels: dict, *, deadline: float | None = None, request_id: str = ""):
        key = digest(request)
        # Gateway calls may deliberately have no local deadline while the
        # Runtime owns durable remote retries.  JSON has no portable infinity
        # value, and waiting callers must see that as an unknown completion
        # time rather than a fabricated finite budget.
        reservation_deadline = (
            deadline
            if type(deadline) in (int, float) and math.isfinite(deadline)
            else None
        )
        with open_private_directory(self.root / "tasks") as directory:
            descriptor = os.open(key + ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600, dir_fd=directory)
        acquired = False
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise _running_task(descriptor) from None
            acquired = True
            # Publish only the current owner's supervisory deadline, outside
            # task identity. Waiting callers must not reset it with their own.
            os.ftruncate(descriptor, 0)
            os.write(descriptor, json.dumps({"deadline": reservation_deadline, "request_id": request_id},
                                            allow_nan=False).encode())
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
            if (isinstance(record_id, str) and RECORD_ID.fullmatch(record_id)
                    and state.get("status") in {"running", "validation_pending"}):
                try:
                    resumable = read_json(directory / "request.json") == request
                except (FileNotFoundError, ValueError):
                    pass
            if not resumable:
                record_id = "gateway-" + uuid.uuid4().hex
                directory = self.root / "records" / record_id
                private_write(directory / "request.json", request)
            private_write(marker, {"status": "running", "gateway_record_id": record_id})
            with self._active_lock:
                self._active_tasks[key] = TaskRunning(reservation_deadline, request_id)
            yield MeasurementTask(
                self,
                key,
                record_id,
                directory,
                request,
                kernels,
                marker,
                request_id=request_id,
            )
        finally:
            try:
                if acquired:
                    with self._active_lock:
                        self._active_tasks.pop(key, None)
                    os.ftruncate(descriptor, 0)
            finally:
                os.close(descriptor)

    def kernel_records(self, kernel_id: str) -> list[dict]:
        self.read_kernel(kernel_id)  # Validate identity and visibility first.
        rows = []
        skipped = 0
        for path in (self.root / "records").glob("gateway-*/record.json"):
            try:
                if not RECORD_ID.fullmatch(path.parent.name):
                    raise ValueError("Invalid Gateway record directory")
                record = read_record_header(path)
                if (record["gateway_record_id"] != path.parent.name
                        or not isinstance(record["operation"], str) or not record["operation"]
                        or not isinstance(record["created_at"], str)
                        or not isinstance(record["kernels"], dict)):
                    raise ValueError("Invalid Gateway record header")
                datetime.fromisoformat(record["created_at"])
                kernel_ids = []
                for item in record["kernels"].values():
                    if (not isinstance(item, dict) or not isinstance(item.get("kernel_id"), str)
                            or not KERNEL_ID.fullmatch(item["kernel_id"])):
                        raise ValueError("Invalid Kernel identity in Gateway record header")
                    kernel_ids.append(item["kernel_id"])
            except (OSError, ValueError, KeyError, TypeError, RecursionError):
                skipped += 1
                continue
            if kernel_id in kernel_ids:
                rows.append({"gateway_record_id": record["gateway_record_id"],
                             "operation": record["operation"], "created_at": record["created_at"]})
        if skipped:
            logging.getLogger(__name__).warning(
                "Skipped %d unreadable Gateway record headers while listing Kernel %s", skipped, kernel_id)
        return sorted(rows, key=lambda row: row["gateway_record_id"])


class MeasurementTask:
    def __init__(
        self,
        store,
        key,
        record_id,
        directory,
        request,
        kernels,
        marker,
        *,
        request_id="",
    ):
        self.store, self.key, self.record_id = store, key, record_id
        self.directory, self.request, self.kernels, self.marker = directory, request, kernels, marker
        self.request_id = request_id

    def finish(self, response: dict, *, cacheable: bool, pending: bool = False) -> dict:
        from aka.light.evaluation.public_result import seal_response
        response = seal_response(response, self.request, record_id=self.record_id, kernels=self.kernels)
        if self.request["operation"] in {"evaluate", "same_allocation_abba"}:
            from aka.light.evaluation.public_result import cacheable_evaluation
            cacheable = cacheable and cacheable_evaluation(response, self.request)
        record = {"schema_version": 1, "gateway_record_id": self.record_id,
                  "task_digest": self.key, "request": self.request, "kernels": self.kernels,
                  "operation": self.request["operation"], "response": response,
                  "result_digest": digest(response), "cacheable": cacheable,
                  "created_at": datetime.now(timezone.utc).isoformat()}
        if self.request_id:
            record["runtime_request_id"] = self.request_id
        # Match private_write's actual encoding, including envelope and escaping.
        # Refuse before publication: reserve() can recover this running task's
        # existing Jobs instead of leaving an unreadable validation_pending record.
        if len(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()) + 1 > MAX_RECORD_BYTES:
            raise EvidenceUnavailable("Measurement record exceeds its size limit; existing job evidence retained")
        private_write(self.directory / "record.json", record)
        # Keep a reference even if post-write validation encounters transient IO.
        private_write(self.marker, {"status": "validation_pending", "gateway_record_id": self.record_id})
        if cacheable and self.store._cached(self.record_id, self.key) is None:
            raise ValueError("Recorded measurement failed validation")
        private_write(self.marker, {"status": "running" if pending else "completed" if cacheable else "not_cacheable",
                                    "gateway_record_id": self.record_id})
        return record
