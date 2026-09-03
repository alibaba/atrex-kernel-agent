"""Durable, PID-reuse-safe ownership for recovery handoff processes."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

HANDOFF_ID_ENV = "ATREX_ENVIRONMENT_RESTART_HANDOFF_ID"
HANDOFF_LOCK_FD_ENV = "ATREX_ENVIRONMENT_RESTART_LOCK_FD"
STATE_FILE_ENV = "ATREX_ENVIRONMENT_STATE_FILE"
REGISTRY_DIRECTORY = "restart-processes"
_HANDOFF_ID_PATTERN = re.compile(r"[a-f0-9]{32}")


@dataclass(frozen=True)
class ProcessRecord:
    pid: int
    ppid: int
    pgid: int
    start_token: str
    role: str


@dataclass(frozen=True)
class TerminationResult:
    complete: bool
    remaining_pids: tuple[int, ...] = ()
    errors: tuple[str, ...] = ()


class _DarwinProcBSDInfo(ctypes.Structure):
    _fields_ = [
        ("pbi_flags", ctypes.c_uint32),
        ("pbi_status", ctypes.c_uint32),
        ("pbi_xstatus", ctypes.c_uint32),
        ("pbi_pid", ctypes.c_uint32),
        ("pbi_ppid", ctypes.c_uint32),
        ("pbi_uid", ctypes.c_uint32),
        ("pbi_gid", ctypes.c_uint32),
        ("pbi_ruid", ctypes.c_uint32),
        ("pbi_rgid", ctypes.c_uint32),
        ("pbi_svuid", ctypes.c_uint32),
        ("pbi_svgid", ctypes.c_uint32),
        ("rfu_1", ctypes.c_uint32),
        ("pbi_comm", ctypes.c_char * 16),
        ("pbi_name", ctypes.c_char * 32),
        ("pbi_nfiles", ctypes.c_uint32),
        ("pbi_pgid", ctypes.c_uint32),
        ("pbi_pjobc", ctypes.c_uint32),
        ("e_tdev", ctypes.c_uint32),
        ("e_tpgid", ctypes.c_uint32),
        ("pbi_nice", ctypes.c_int32),
        ("pbi_start_tvsec", ctypes.c_uint64),
        ("pbi_start_tvusec", ctypes.c_uint64),
    ]


def _darwin_process(pid: int) -> ProcessRecord | None:
    if sys.platform != "darwin":
        return None
    try:
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
        proc_pidinfo = libproc.proc_pidinfo
        proc_pidinfo.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        proc_pidinfo.restype = ctypes.c_int
        info = _DarwinProcBSDInfo()
        size = ctypes.sizeof(info)
        written = proc_pidinfo(pid, 3, 0, ctypes.byref(info), size)
    except (AttributeError, OSError):
        return None
    if written != size or info.pbi_pid != pid or info.pbi_status == 5:
        return None
    token = f"darwin:{info.pbi_start_tvsec}:{info.pbi_start_tvusec}"
    return ProcessRecord(pid, info.pbi_ppid, info.pbi_pgid, token, "")


def _values(environment: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if environment is None else environment


def recovery_pass_fds(environment: Mapping[str, str] | None = None) -> tuple[int, ...]:
    """Return the validated inherited handoff descriptor for a child Popen."""
    raw = _values(environment).get(HANDOFF_LOCK_FD_ENV, "")
    try:
        descriptor = int(raw)
    except ValueError:
        return ()
    if descriptor <= 2:
        return ()
    try:
        os.fstat(descriptor)
    except OSError:
        return ()
    return (descriptor,)


def _linux_process(pid: int, boot_id: str) -> ProcessRecord | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return None
    closing = raw.rfind(")")
    if closing < 0:
        return None
    fields = raw[closing + 2 :].split()
    try:
        state = fields[0]
        ppid = int(fields[1])
        pgid = int(fields[2])
        start_ticks = fields[19]
    except (IndexError, ValueError):
        return None
    if state == "Z":
        return None
    return ProcessRecord(pid, ppid, pgid, f"linux:{boot_id}:{start_ticks}", "")


def _process_table() -> dict[int, ProcessRecord]:
    proc = Path("/proc")
    boot_id_path = Path("/proc/sys/kernel/random/boot_id")
    if proc.is_dir() and boot_id_path.is_file():
        try:
            boot_id = boot_id_path.read_text(encoding="utf-8").strip()
            pids = [int(path.name) for path in proc.iterdir() if path.name.isdigit()]
        except (OSError, ValueError):
            pass
        else:
            records = (_linux_process(pid, boot_id) for pid in pids)
            return {record.pid: record for record in records if record is not None}

    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,pgid=,stat=,lstart="],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if result.returncode != 0:
        return {}
    records: dict[int, ProcessRecord] = {}
    for line in result.stdout.splitlines():
        fields = line.split(maxsplit=4)
        if len(fields) != 5 or fields[3].startswith("Z"):
            continue
        try:
            pid, ppid, pgid = (int(value) for value in fields[:3])
        except ValueError:
            continue
        darwin = _darwin_process(pid)
        if darwin is not None:
            records[pid] = darwin
            continue
        records[pid] = ProcessRecord(
            pid, ppid, pgid, f"ps:{fields[4].strip()}", ""
        )
    return records


def process_start_token(pid: int) -> str | None:
    record = _process_table().get(pid)
    return record.start_token if record is not None else None


def _registry_path(state_dir: Path, handoff_id: str) -> Path:
    if _HANDOFF_ID_PATTERN.fullmatch(handoff_id) is None:
        raise ValueError("invalid recovery handoff id")
    return state_dir.resolve() / REGISTRY_DIRECTORY / handoff_id


def _write_private_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    temporary.replace(path)


def _record_process(
    state_dir: Path,
    handoff_id: str,
    record: ProcessRecord,
    role: str,
) -> Path:
    directory = _registry_path(state_dir, handoff_id)
    digest = hashlib.sha256(record.start_token.encode()).hexdigest()[:16]
    path = directory / f"{record.pid}-{digest}.json"
    if path.is_file():
        return path
    safe_role = re.sub(r"[^A-Za-z0-9_.-]", "_", role)[:80] or "process"
    _write_private_json(
        path,
        {
            "schema_version": 1,
            "handoff_id": handoff_id,
            "pid": record.pid,
            "ppid": record.ppid,
            "pgid": record.pgid,
            "start_token": record.start_token,
            "role": safe_role,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return path


def register_process(
    state_dir: Path,
    handoff_id: str,
    pid: int,
    role: str,
) -> Path:
    record = _process_table().get(pid)
    if record is None:
        raise ProcessLookupError(pid)
    return _record_process(state_dir, handoff_id, record, role)


def register_recovery_process(
    pid: int,
    role: str,
    environment: Mapping[str, str] | None = None,
) -> Path | None:
    values = _values(environment)
    handoff_id = values.get(HANDOFF_ID_ENV, "")
    state_file = values.get(STATE_FILE_ENV, "")
    if not handoff_id or not state_file:
        return None
    return register_process(
        Path(state_file).expanduser().resolve().parent, handoff_id, pid, role
    )


def record_process_tree(
    state_dir: Path,
    handoff_id: str,
    root_pid: int,
) -> int:
    """Persist identities for the root and all currently visible descendants."""
    table = _process_table()
    if root_pid not in table:
        return 0
    children: dict[int, list[int]] = {}
    for record in table.values():
        children.setdefault(record.ppid, []).append(record.pid)
    pending = [root_pid]
    seen: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        pending.extend(children.get(pid, ()))
    for pid in seen:
        _record_process(state_dir, handoff_id, table[pid], "observed-descendant")
    return len(seen)


def _load_records(
    state_dir: Path, handoff_id: str
) -> tuple[list[ProcessRecord], list[str]]:
    directory = _registry_path(state_dir, handoff_id)
    if not directory.is_dir():
        return [], []
    records: list[ProcessRecord] = []
    errors: list[str] = []
    for path in sorted(directory.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if (
                not isinstance(value, dict)
                or value.get("schema_version") != 1
                or value.get("handoff_id") != handoff_id
            ):
                raise ValueError("schema or handoff id mismatch")
            pid = value["pid"]
            ppid = value["ppid"]
            pgid = value["pgid"]
            start_token = value["start_token"]
            role = value["role"]
            if (
                not all(isinstance(item, int) and item > 0 for item in (pid, pgid))
                or not isinstance(ppid, int)
                or ppid < 0
                or not isinstance(start_token, str)
                or not start_token
                or not isinstance(role, str)
            ):
                raise ValueError("invalid process identity")
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{path.name}: {exc}")
            continue
        records.append(ProcessRecord(pid, ppid, pgid, start_token, role))
    return records, errors


def matching_processes(state_dir: Path, handoff_id: str) -> tuple[ProcessRecord, ...]:
    records, _errors = _load_records(state_dir, handoff_id)
    table = _process_table()
    matches = []
    for record in records:
        current = table.get(record.pid)
        if current is not None and current.start_token == record.start_token:
            matches.append(
                ProcessRecord(
                    current.pid,
                    current.ppid,
                    current.pgid,
                    current.start_token,
                    record.role,
                )
            )
    return tuple(matches)


def process_identity_matches(
    state_dir: Path, handoff_id: str, pid: int
) -> bool:
    return any(record.pid == pid for record in matching_processes(state_dir, handoff_id))


def _signal_groups(groups: set[int], sig: signal.Signals) -> list[str]:
    errors: list[str] = []
    own_group = os.getpgrp()
    for pgid in sorted(groups):
        if pgid == own_group:
            errors.append(f"refusing to signal monitor process group {pgid}")
            continue
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            pass
        except OSError as exc:
            errors.append(f"cannot signal process group {pgid}: {exc}")
    return errors


def terminate_processes(
    state_dir: Path,
    handoff_id: str,
    *,
    root_pid: int | None,
    require_registered_owner: bool,
    grace_seconds: float = 5.0,
) -> TerminationResult:
    """Terminate every registered group without trusting a reused PID."""
    if root_pid is not None:
        record_process_tree(state_dir, handoff_id, root_pid)
    records, load_errors = _load_records(state_dir, handoff_id)
    if load_errors:
        return TerminationResult(False, errors=tuple(load_errors))

    table = _process_table()
    matches = [
        table[record.pid]
        for record in records
        if record.pid in table
        and table[record.pid].start_token == record.start_token
    ]
    if require_registered_owner and not matches:
        return TerminationResult(
            False,
            errors=("handoff ownership is active but no registered process identity matches",),
        )
    groups = {record.pgid for record in matches}
    errors = _signal_groups(groups, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if root_pid is not None:
            record_process_tree(state_dir, handoff_id, root_pid)
        remaining = matching_processes(state_dir, handoff_id)
        current_table = _process_table()
        lingering = any(record.pgid in groups for record in current_table.values())
        if not remaining and not lingering:
            break
        time.sleep(0.05)

    remaining = matching_processes(state_dir, handoff_id)
    kill_groups = groups | {record.pgid for record in remaining}
    current_table = _process_table()
    lingering_before_kill = {
        record.pgid for record in current_table.values() if record.pgid in kill_groups
    }
    if remaining or lingering_before_kill:
        errors.extend(_signal_groups(kill_groups, signal.SIGKILL))
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            current_table = _process_table()
            if not matching_processes(state_dir, handoff_id) and not any(
                record.pgid in kill_groups for record in current_table.values()
            ):
                break
            time.sleep(0.05)

    remaining = matching_processes(state_dir, handoff_id)
    current_table = _process_table()
    lingering_groups = {
        record.pgid for record in current_table.values() if record.pgid in kill_groups
    }
    if lingering_groups:
        errors.append(
            "owned process groups still exist after SIGKILL: "
            + ", ".join(str(value) for value in sorted(lingering_groups))
        )
    return TerminationResult(
        not remaining and not lingering_groups and not errors,
        remaining_pids=tuple(sorted(record.pid for record in remaining)),
        errors=tuple(errors),
    )


def clear_process_registry(state_dir: Path, handoff_id: str) -> None:
    directory = _registry_path(state_dir, handoff_id)
    if not directory.is_dir():
        return
    for path in directory.iterdir():
        if path.is_file():
            path.unlink()
    try:
        directory.rmdir()
    except OSError:
        return
    parent = directory.parent
    try:
        parent.rmdir()
    except OSError:
        pass
