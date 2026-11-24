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
import sys
import time
import uuid
from contextlib import contextmanager
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

from aka.core.filesystem import open_private_directory
from aka.core.filesystem import durable_write_json, durable_write_text
from aka.core.filesystem import read_regular_bytes
from aka.task.source.formats import MAX_SNAPSHOT, Artifact

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



class MeasurementWaitTimeout(TimeoutError):
    """This caller's budget expired while another request owns the measurement."""


def _running_record(descriptor: int) -> str | None:
    # The owner publishes this after its reservation is initialized. A brief
    # empty/partial write is not evidence that a second job should be started.
    try:
        value = json.loads(os.pread(descriptor, 4096, 0))
        record_id = value.get("gateway_record_id")
        return record_id if isinstance(record_id, str) and RECORD_ID.fullmatch(record_id) else None
    except (OSError, ValueError, TypeError, AttributeError):
        return None


class MeasurementStore:
    def __init__(self, root: Path):
        root = Path(root).expanduser().resolve()
        self.root = root
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

    def execution_records(self):
        """Use one Campaign namespace for the public Experiment execution ledger.

        Attempts, preparation failures, raw outputs and source artifacts belong
        here even when no successful measurement or accepted submission exists.
        A lineage workspace is an execution location, never a storage namespace.
        """
        from .file_records import FileExperimentRecords

        return FileExperimentRecords(
            self.root / "executions", "campaign", namespace_directory=False,
            artifact_root=self.root,
        )

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

    def _cached(self, record_id: str, task_digest: str, *, joined: bool = False) -> dict | None:
        # Broken/old indexes are misses. Transient IO failures retain the index
        # and stop dispatch, including on the publication path.
        for delay in (0, .05, .1):
            if delay:
                time.sleep(delay)
            try:
                record = self.read(record_id)
                if record.get("task_digest") != task_digest:
                    return None
                # Concurrent callers share the original response, including a
                # non-cacheable infrastructure failure. Later calls still obey
                # normal cache/recovery policy.
                if joined:
                    return record
                if record.get("cacheable") is not True:
                    return None
                if record["operation"] in {"evaluate", "same_allocation_abba"}:
                    from aka.evaluation.tool_results import cacheable_evaluation
                    if not cacheable_evaluation(record["response"], record["request"]):
                        # Older Dev-backed evaluators cached allocation success
                        # even when no evaluation receipt existed. Keep the old
                        # record intact, but do not use it as a permanent verdict.
                        return None
                from aka.evaluation.device_failures import embedded_device_failure
                from aka.evaluation.result_json import parse_evaluation_result_line
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

    @contextmanager
    def reserve(self, request: dict, kernels: dict, *, deadline: float | None = None,
                request_id: str = "", check_wait=None, reusable: bool = True):
        """Join an identical active request, or own its execution/recovery.

        Poll only the local reservation so cancellation and the caller's own
        budget remain effective. No duplicate GPU request is submitted.
        """
        key = digest(request)
        if not reusable:
            # One-shot observations cannot be joined or resumed by another call.
            # Keep their complete immutable Record, without permanent task inodes.
            record_id = "gateway-" + uuid.uuid4().hex
            directory = self.root / "records" / record_id
            private_write(directory / "request.json", request)
            task = MeasurementTask(self, key, record_id, directory, request, kernels,
                                   None, request_id=request_id)
            try:
                yield task
            finally:
                if not (directory / "record.json").exists():
                    # Preserve interrupted calls as discoverable audit facts.
                    # Do not invent a remote Job outcome or hide the caller's
                    # original exception if the audit disk also failed.
                    try:
                        task.finish({"operation": request["operation"], "status": "interrupted",
                                     "error": {"code": "runtime_request_interrupted",
                                               "message": "Runtime request ended before a result was archived."}},
                                    cacheable=False)
                    except Exception:
                        logging.getLogger(__name__).exception("Could not archive interrupted Runtime request %s", record_id)
            return
        wait_deadline = deadline if type(deadline) in (int, float) and math.isfinite(deadline) else None
        with open_private_directory(self.root / "tasks") as directory:
            try:
                descriptor = os.open(key + ".lock", os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW,
                                     0o600, dir_fd=directory)
            except FileExistsError:
                # Open an existing reservation separately from creation. This
                # also avoids APFS openat(O_CREAT) races between lineages.
                descriptor = os.open(key + ".lock", os.O_RDWR | os.O_NOFOLLOW, dir_fd=directory)
        acquired = False
        waited = False
        joined_record_id = None
        try:
            while True:
                if check_wait is not None:
                    check_wait()
                if wait_deadline is not None and time.monotonic() >= wait_deadline:
                    raise MeasurementWaitTimeout(
                        "Runtime request deadline expired while waiting for an identical measurement. "
                        "This caller submitted no new job and did not cancel the original request."
                    )
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except BlockingIOError:
                    waited = True
                    joined_record_id = joined_record_id or _running_record(descriptor)
                    delay = .05 if wait_deadline is None else min(.05, max(0, wait_deadline - time.monotonic()))
                    time.sleep(delay)
            # Do not expose stale reservation metadata if this caller becomes
            # the recovery owner after an interrupted request.
            os.ftruncate(descriptor, 0)
            marker = self.root / "tasks" / (key + ".json")
            try:
                state = read_json(marker, limit=4096)
            except (FileNotFoundError, ValueError):
                state = {}
            record_id = state.get("gateway_record_id")
            if waited:
                # If the owner finished before publishing lock metadata, the
                # marker under the now-acquired lock identifies its last result.
                joined_record_id = joined_record_id or record_id
                if isinstance(joined_record_id, str) and RECORD_ID.fullmatch(joined_record_id):
                    joined = self._cached(joined_record_id, key, joined=True)
                    if joined is not None:
                        raise DuplicateTask(joined_record_id)
                    if (self.root / "records" / joined_record_id / "record.json").exists():
                        raise EvidenceUnavailable(
                            "The shared measurement result failed integrity validation; no replacement job was submitted."
                        )
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
            os.write(descriptor, json.dumps({"gateway_record_id": record_id,
                                            "request_id": request_id}).encode())
            yield MeasurementTask(
                self,
                key,
                record_id,
                directory,
                request,
                kernels,
                marker,
                request_id=request_id,
                resumed=resumable,
            )
        finally:
            try:
                if acquired:
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
        resumed=False,
    ):
        self.store, self.key, self.record_id = store, key, record_id
        self.directory, self.request, self.kernels, self.marker = directory, request, kernels, marker
        self.request_id = request_id
        self.resumed = resumed

    def finish(self, response: dict, *, cacheable: bool, pending: bool = False) -> dict:
        from aka.evaluation.tool_results import seal_response
        response = seal_response(response, self.request, record_id=self.record_id, kernels=self.kernels)
        if self.request["operation"] in {"evaluate", "same_allocation_abba"}:
            from aka.evaluation.tool_results import cacheable_evaluation
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
        if self.marker is not None:
            private_write(self.marker, {"status": "validation_pending", "gateway_record_id": self.record_id})
        if cacheable and self.store._cached(self.record_id, self.key) is None:
            raise ValueError("Recorded measurement failed validation")
        if self.marker is not None:
            private_write(self.marker, {"status": "running" if pending else "completed" if cacheable else "not_cacheable",
                                        "gateway_record_id": self.record_id})
        return record


class MeasurementReadView(MeasurementStore):
    """Bounded request-local LRU for immutable records and source artifacts."""
    def __init__(self, root, *, max_entries=128, max_bytes=64 * 1024 * 1024):
        super().__init__(root)
        self._cache = OrderedDict()
        self._cache_bytes = 0
        self._max_entries = max_entries
        self._max_bytes = max_bytes

    def _read_cached(self, key, read):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key][0]
        value = read()
        # Account for decoded Python objects, not only serialized JSON bytes.
        # Oversized values remain readable but never occupy the cache.
        pending, seen, size = [value], set(), 0
        while pending and size <= self._max_bytes:
            item = pending.pop()
            if id(item) in seen:
                continue
            seen.add(id(item))
            size += sys.getsizeof(item)
            if isinstance(item, dict):
                pending.extend(item.keys())
                pending.extend(item.values())
            elif isinstance(item, (list, tuple)):
                pending.extend(item)
        if size <= self._max_bytes and self._max_entries > 0:
            while self._cache and (len(self._cache) >= self._max_entries
                                  or self._cache_bytes + size > self._max_bytes):
                _, (_, discarded_size) = self._cache.popitem(last=False)
                self._cache_bytes -= discarded_size
            self._cache[key] = value, size
            self._cache_bytes += size
        return value

    def read(self, record_id):
        return self._read_cached(("record", record_id), lambda: MeasurementStore.read(self, record_id))

    def read_kernel(self, kernel_id):
        return self._read_cached(("kernel", kernel_id), lambda: MeasurementStore.read_kernel(self, kernel_id))
