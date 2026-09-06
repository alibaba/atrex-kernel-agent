"""Durable, PID-reuse-safe ownership for recovery handoff processes."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import select
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from orchestrator.durable_state import (  # noqa: E402
    durable_rmdir,
    durable_unlink,
    durable_write_json,
    durable_write_text,
)

HANDOFF_ID_ENV = "ATREX_ENVIRONMENT_RESTART_HANDOFF_ID"
HANDOFF_LOCK_FD_ENV = "ATREX_ENVIRONMENT_RESTART_LOCK_FD"
STATE_FILE_ENV = "ATREX_ENVIRONMENT_STATE_FILE"
REGISTRY_DIRECTORY = "restart-processes"
ACTIVE_MARKER = "active.json"
STOP_MARKER = "stopped.json"
TERMINATE_REQUEST = "terminate.request"
RESTART_PRIMARY_PID = "restart.primary.pid"
RESTART_EXIT = "restart.exit.json"
RESTART_COMPLETE = "restart.complete.json"
_HANDOFF_ID_PATTERN = re.compile(r"[a-f0-9]{32}")


@dataclass(frozen=True)
class ProcessRecord:
    pid: int
    ppid: int
    pgid: int
    start_token: str
    role: str
    owner_kind: str = ""
    owner_pid: int = 0


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


def _current_process(pid: int) -> ProcessRecord | None:
    proc = Path("/proc")
    boot_id_path = Path("/proc/sys/kernel/random/boot_id")
    if proc.is_dir() and boot_id_path.is_file():
        try:
            boot_id = boot_id_path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return _linux_process(pid, boot_id)
    darwin = _darwin_process(pid)
    if darwin is not None or sys.platform == "darwin":
        return darwin
    return _process_table().get(pid)


def _registry_path(state_dir: Path, handoff_id: str) -> Path:
    if _HANDOFF_ID_PATTERN.fullmatch(handoff_id) is None:
        raise ValueError("invalid recovery handoff id")
    return state_dir.resolve() / REGISTRY_DIRECTORY / handoff_id


def _write_private_json(path: Path, value: object) -> None:
    durable_write_json(path, value)


def _record_process(
    state_dir: Path,
    handoff_id: str,
    record: ProcessRecord,
    role: str,
    *,
    owner_kind: str,
    owner_pid: int = 0,
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
            "schema_version": 3,
            "handoff_id": handoff_id,
            "pid": record.pid,
            "ppid": record.ppid,
            "pgid": record.pgid,
            "start_token": record.start_token,
            "role": safe_role,
            "owner_kind": owner_kind,
            "owner_pid": owner_pid,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return path


def _register_session_owner(
    pid: int,
    role: str,
    environment: Mapping[str, str] | None = None,
) -> Path | None:
    values = _values(environment)
    handoff_id = values.get(HANDOFF_ID_ENV, "")
    state_file = values.get(STATE_FILE_ENV, "")
    if not handoff_id or not state_file:
        return None
    record = _current_process(pid)
    if record is None:
        raise ProcessLookupError(pid)
    return _record_process(
        Path(state_file).expanduser().resolve().parent,
        handoff_id,
        record,
        role,
        owner_kind="session-wrapper",
    )


def _register_session_primary(
    pid: int,
    role: str,
    owner_pid: int,
    environment: Mapping[str, str] | None = None,
) -> Path | None:
    values = _values(environment)
    handoff_id = values.get(HANDOFF_ID_ENV, "")
    state_file = values.get(STATE_FILE_ENV, "")
    if not handoff_id or not state_file:
        return None
    record = _current_process(pid)
    if record is None:
        raise ProcessLookupError(pid)
    return _record_process(
        Path(state_file).expanduser().resolve().parent,
        handoff_id,
        record,
        role,
        owner_kind="session-primary",
        owner_pid=owner_pid,
    )


def _register_session_guardian(
    pid: int,
    role: str,
    owner_pid: int,
    environment: Mapping[str, str] | None = None,
) -> Path | None:
    values = _values(environment)
    handoff_id = values.get(HANDOFF_ID_ENV, "")
    state_file = values.get(STATE_FILE_ENV, "")
    if not handoff_id or not state_file:
        return None
    record = _current_process(pid)
    if record is None:
        raise ProcessLookupError(pid)
    return _record_process(
        Path(state_file).expanduser().resolve().parent,
        handoff_id,
        record,
        role,
        owner_kind="session-guardian",
        owner_pid=owner_pid,
    )


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
                or value.get("schema_version") not in {1, 2, 3}
                or value.get("handoff_id") != handoff_id
            ):
                raise ValueError("schema or handoff id mismatch")
            pid = value["pid"]
            ppid = value["ppid"]
            pgid = value["pgid"]
            start_token = value["start_token"]
            role = value["role"]
            owner_kind = value.get("owner_kind", "legacy-identity")
            owner_pid = value.get("owner_pid", 0)
            if (
                not all(isinstance(item, int) and item > 0 for item in (pid, pgid))
                or not isinstance(ppid, int)
                or ppid < 0
                or not isinstance(start_token, str)
                or not start_token
                or not isinstance(role, str)
                or not isinstance(owner_kind, str)
                or not isinstance(owner_pid, int)
                or owner_pid < 0
            ):
                raise ValueError("invalid process identity")
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{path.name}: {exc}")
            continue
        records.append(
            ProcessRecord(pid, ppid, pgid, start_token, role, owner_kind, owner_pid)
        )
    return records, errors


def matching_processes(state_dir: Path, handoff_id: str) -> tuple[ProcessRecord, ...]:
    records, _errors = _load_records(state_dir, handoff_id)
    matches = []
    for record in records:
        current = _current_process(record.pid)
        if current is not None and current.start_token == record.start_token:
            matches.append(
                ProcessRecord(
                    current.pid,
                    current.ppid,
                    current.pgid,
                    current.start_token,
                    record.role,
                    record.owner_kind,
                    record.owner_pid,
                )
            )
    return tuple(matches)


def process_identity_matches(
    state_dir: Path, handoff_id: str, pid: int
) -> bool:
    return any(record.pid == pid for record in matching_processes(state_dir, handoff_id))


def terminate_processes(
    state_dir: Path,
    handoff_id: str,
    *,
    require_registered_owner: bool,
    grace_seconds: float = 5.0,
) -> TerminationResult:
    """Ask registered session owners to terminate without signalling a stale PID."""
    records, load_errors = _load_records(state_dir, handoff_id)
    if load_errors:
        return TerminationResult(False, errors=tuple(load_errors))

    matches = []
    for record in records:
        current = _current_process(record.pid)
        if current is not None and current.start_token == record.start_token:
            matches.append(
                ProcessRecord(
                    current.pid,
                    current.ppid,
                    current.pgid,
                    current.start_token,
                    record.role,
                    record.owner_kind,
                    record.owner_pid,
                )
            )
    if require_registered_owner and not matches:
        return TerminationResult(
            False,
            errors=("handoff ownership is active but no registered process identity matches",),
        )
    stable_owner_kinds = {
        "session-wrapper",
        "session-primary",
        "session-guardian",
    }
    legacy = [
        record.pid
        for record in matches
        if record.owner_kind not in stable_owner_kinds
    ]
    if legacy:
        error = (
            "live handoff records predate stable session ownership; "
            "refusing speculative process-group termination"
        )
        return TerminationResult(
            False,
            remaining_pids=tuple(sorted(legacy)),
            errors=(error,),
        )
    cooperative_owners = [
        record
        for record in matches
        if record.owner_kind in {"session-wrapper", "session-guardian"}
    ]
    if cooperative_owners:
        _write_private_json(
            _registry_path(state_dir, handoff_id) / TERMINATE_REQUEST,
            {
                "schema_version": 1,
                "handoff_id": handoff_id,
                "grace_seconds": max(0.0, grace_seconds),
                "requested_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    deadline = time.monotonic() + (
        max(0.0, grace_seconds) if cooperative_owners else 0.0
    )
    while time.monotonic() < deadline:
        remaining = matching_processes(state_dir, handoff_id)
        if not remaining:
            break
        time.sleep(0.05)

    remaining = matching_processes(state_dir, handoff_id)
    # Guardians live outside the target groups and retain cleanup ownership if a
    # wrapper dies. Force only wrapper/primary groups here so a guardian can still
    # observe controller death, clean the target group, and commit completion.
    owned_groups = {
        record.pgid
        for record in remaining
        if record.owner_kind in {"session-wrapper", "session-primary"}
    }
    for pgid in owned_groups:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    force_deadline = time.monotonic() + 5.0
    while owned_groups and time.monotonic() < force_deadline:
        remaining = matching_processes(state_dir, handoff_id)
        owned_groups = {
            record.pgid
            for record in remaining
            if record.owner_kind in {"session-wrapper", "session-primary"}
        }
        if owned_groups:
            time.sleep(0.05)
    for pgid in owned_groups:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if owned_groups:
        kill_deadline = time.monotonic() + 5.0
        while time.monotonic() < kill_deadline:
            remaining = matching_processes(state_dir, handoff_id)
            if not remaining:
                break
            time.sleep(0.05)

    guardian_deadline = time.monotonic() + max(0.0, grace_seconds) + 10.0
    remaining = matching_processes(state_dir, handoff_id)
    while remaining and time.monotonic() < guardian_deadline:
        time.sleep(0.05)
        remaining = matching_processes(state_dir, handoff_id)
    errors = ()
    if remaining:
        errors = (
            "registered session owners did not honor the termination request",
        )
    return TerminationResult(
        not remaining,
        remaining_pids=tuple(sorted(record.pid for record in remaining)),
        errors=errors,
    )


def _kill_owned_wrapper(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def spawn_owned_session(
    command: Sequence[str | os.PathLike[str]],
    *,
    role: str,
    environment: Mapping[str, str] | None = None,
    finalize_handoff: bool = False,
    registered_callback: Callable[[subprocess.Popen[Any]], None] | None = None,
    **popen_options: Any,
) -> subprocess.Popen[Any]:
    """Start a session only after its stable wrapper identity is registered."""
    forbidden = {"env", "start_new_session", "close_fds", "pass_fds"}
    overlap = forbidden.intersection(popen_options)
    if overlap:
        raise TypeError(
            "spawn_owned_session owns Popen options: " + ", ".join(sorted(overlap))
        )
    argv = [os.fspath(item) for item in command]
    if not argv:
        raise ValueError("owned session command must not be empty")
    values = _values(environment)
    handoff_id = values.get(HANDOFF_ID_ENV, "")
    if not handoff_id:
        return subprocess.Popen(
            argv,
            env=environment,
            start_new_session=True,
            close_fds=True,
            **popen_options,
        )
    if _HANDOFF_ID_PATTERN.fullmatch(handoff_id) is None:
        raise ValueError("invalid recovery handoff id")
    state_file = values.get(STATE_FILE_ENV, "")
    handoff_fds = recovery_pass_fds(values)
    if not state_file or len(handoff_fds) != 1:
        raise RuntimeError("active recovery handoff lacks state or lock ownership")

    start_read, start_write = os.pipe()
    os.set_inheritable(start_read, True)
    wrapper = [
        str(Path(sys.executable).resolve()),
        str(Path(__file__).resolve()),
        "--owned-session",
        str(start_read),
        "1" if finalize_handoff else "0",
        role,
        "--",
        *argv,
    ]
    process: subprocess.Popen[Any] | None = None
    try:
        process = subprocess.Popen(
            wrapper,
            env=environment,
            start_new_session=True,
            close_fds=True,
            pass_fds=(*handoff_fds, start_read),
            **popen_options,
        )
        _register_session_owner(process.pid, role, values)
        if registered_callback is not None:
            registered_callback(process)
        os.write(start_write, b"1")
    except BaseException:
        if process is not None:
            _kill_owned_wrapper(process)
        raise
    finally:
        os.close(start_read)
        os.close(start_write)
    return process


def clear_process_registry(state_dir: Path, handoff_id: str) -> None:
    directory = _registry_path(state_dir, handoff_id)
    if not directory.is_dir():
        return
    for path in directory.iterdir():
        if path.is_file():
            durable_unlink(path)
    try:
        durable_rmdir(directory)
    except OSError:
        return
    parent = directory.parent
    try:
        durable_rmdir(parent)
    except OSError:
        pass


def _live_group_members(pgid: int, *, exclude: set[int]) -> tuple[int, ...]:
    members = []
    for record in _process_table().values():
        if record.pgid != pgid or record.pid in exclude:
            continue
        current = _current_process(record.pid)
        if current is not None and current.start_token == record.start_token:
            members.append(record.pid)
    return tuple(sorted(members))


def _signal_group_members(pgid: int, sig: signal.Signals) -> None:
    for pid in _live_group_members(pgid, exclude={os.getpid()}):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def _close_handoff_lock() -> None:
    raw = os.environ.pop(HANDOFF_LOCK_FD_ENV, "")
    try:
        descriptor = int(raw)
    except ValueError:
        return
    if descriptor > 2:
        try:
            os.close(descriptor)
        except OSError:
            pass


def _termination_grace_seconds(handoff_id: str) -> float | None:
    state_file = os.environ.get(STATE_FILE_ENV, "")
    if not state_file:
        return None
    path = (
        _registry_path(Path(state_file).expanduser().resolve().parent, handoff_id)
        / TERMINATE_REQUEST
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        grace_seconds = value["grace_seconds"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return None
    if (
        value.get("schema_version") != 1
        or value.get("handoff_id") != handoff_id
        or not isinstance(grace_seconds, (int, float))
        or isinstance(grace_seconds, bool)
        or grace_seconds < 0
    ):
        return None
    return float(grace_seconds)


def _root_state_dir() -> Path | None:
    state_file = os.environ.get(STATE_FILE_ENV, "")
    if not state_file:
        return None
    return Path(state_file).expanduser().resolve().parent


def _write_root_event(
    name: str,
    handoff_id: str,
    primary_pid: int,
    returncode: int,
    *,
    wrapper_pid: int | None = None,
    completed_by: int | None = None,
) -> None:
    state_dir = _root_state_dir()
    if state_dir is None:
        return
    event = {
        "schema_version": 1,
        "handoff_id": handoff_id,
        "wrapper_pid": wrapper_pid or os.getpid(),
        "primary_pid": primary_pid,
        "returncode": returncode,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    if completed_by is not None:
        event["completed_by"] = completed_by
    _write_private_json(state_dir / name, event)


def _read_root_event(path: Path, handoff_id: str) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("handoff_id") != handoff_id
        or not isinstance(value.get("primary_pid"), int)
        or isinstance(value.get("primary_pid"), bool)
        or value["primary_pid"] <= 0
        or not isinstance(value.get("returncode"), int)
        or isinstance(value.get("returncode"), bool)
    ):
        return None
    return value


def _cleanup_owned_group(pgid: int, grace_seconds: float) -> bool:
    members = _live_group_members(pgid, exclude={os.getpid()})
    if not members:
        return True
    _signal_group_members(pgid, signal.SIGTERM)
    deadline = time.monotonic() + max(0.0, grace_seconds)
    while time.monotonic() < deadline:
        if not _live_group_members(pgid, exclude={os.getpid()}):
            return True
        time.sleep(0.05)
    _signal_group_members(pgid, signal.SIGKILL)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not _live_group_members(pgid, exclude={os.getpid()}):
            return True
        time.sleep(0.05)
    return not _live_group_members(pgid, exclude={os.getpid()})


def _request_other_sessions_stop(
    handoff_id: str,
    grace_seconds: float,
    *,
    exclude_pids: set[int] | None = None,
) -> bool:
    state_dir = _root_state_dir()
    if state_dir is None:
        return True
    _records, errors = _load_records(state_dir, handoff_id)
    if errors:
        print(
            "[recovery-owner] cannot complete corrupt process registry: "
            + "; ".join(errors),
            file=sys.stderr,
            flush=True,
        )
        return False
    excluded = {os.getpid()} if exclude_pids is None else set(exclude_pids)
    excluded.add(os.getpid())
    live_others = [
        record
        for record in matching_processes(state_dir, handoff_id)
        if record.pid not in excluded
    ]
    if not live_others:
        return True
    _write_private_json(
        _registry_path(state_dir, handoff_id) / TERMINATE_REQUEST,
        {
            "schema_version": 1,
            "handoff_id": handoff_id,
            "grace_seconds": max(0.0, grace_seconds),
            "requested_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    deadline = time.monotonic() + max(0.0, grace_seconds) + 5.0
    while time.monotonic() < deadline:
        if not [
            record
            for record in matching_processes(state_dir, handoff_id)
            if record.pid not in excluded
        ]:
            return True
        time.sleep(0.05)
    return False


def _owned_command_entry(argv: list[str]) -> int:
    if len(argv) < 4 or argv[0] != "--owned-command" or argv[2] != "--":
        return 125
    try:
        start_fd = int(argv[1])
    except ValueError:
        return 125
    command = argv[3:]
    if start_fd <= 2 or not command:
        return 125
    try:
        permitted = os.read(start_fd, 1)
    except OSError:
        return 125
    finally:
        try:
            os.close(start_fd)
        except OSError:
            pass
    if permitted != b"1":
        return 125
    try:
        os.execvpe(command[0], command, os.environ)
    except OSError as exc:
        print(
            f"[recovery-primary] cannot exec command: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 127


def _cleanup_guardian_entry(argv: list[str]) -> int:
    if (
        len(argv) != 5
        or argv[0] != "--cleanup-guardian"
        or argv[4] not in {"0", "1"}
    ):
        return 125
    try:
        control_fd = int(argv[1])
        wrapper_pid = int(argv[2])
        target_pgid = int(argv[3])
        finalize_handoff = argv[4] == "1"
    except ValueError:
        return 125
    if control_fd <= 2 or wrapper_pid <= 0 or target_pgid <= 0:
        return 125
    try:
        permitted = os.read(control_fd, 1)
        if permitted != b"1":
            return 125
        while True:
            readable, _writable, _errors = select.select([control_fd], [], [], 0.1)
            if not readable:
                continue
            message = os.read(control_fd, 1)
            if message == b"A":
                return 0
            if message == b"C":
                if not finalize_handoff:
                    return 0
                state_dir = _root_state_dir()
                handoff_id = os.environ.get(HANDOFF_ID_ENV, "")
                if (
                    state_dir is not None
                    and _HANDOFF_ID_PATTERN.fullmatch(handoff_id)
                    and _read_root_event(state_dir / RESTART_COMPLETE, handoff_id)
                    is not None
                ):
                    return 0
                break
            if not message:
                break
            return 125

        handoff_id = os.environ.get(HANDOFF_ID_ENV, "")
        if _HANDOFF_ID_PATTERN.fullmatch(handoff_id) is None:
            return 125
        requested_grace = _termination_grace_seconds(handoff_id)
        grace_seconds = requested_grace if requested_grace is not None else 5.0
        group_clean = _cleanup_owned_group(target_pgid, grace_seconds)
        sessions_clean = True
        if finalize_handoff:
            sessions_clean = _request_other_sessions_stop(
                handoff_id, grace_seconds, exclude_pids={os.getpid()}
            )
        state_dir = _root_state_dir()
        if finalize_handoff and group_clean and sessions_clean and state_dir is not None:
            exit_event = _read_root_event(state_dir / RESTART_EXIT, handoff_id)
            if exit_event is not None:
                _write_root_event(
                    RESTART_COMPLETE,
                    handoff_id,
                    exit_event["primary_pid"],
                    exit_event["returncode"],
                    wrapper_pid=wrapper_pid,
                    completed_by=os.getpid(),
                )
        return 0 if group_clean and sessions_clean else 1
    finally:
        try:
            os.close(control_fd)
        except OSError:
            pass
        _close_handoff_lock()


def _owned_session_entry(argv: list[str]) -> int:
    if len(argv) < 6 or argv[0] != "--owned-session" or argv[4] != "--":
        print("invalid recovery owner invocation", file=sys.stderr, flush=True)
        return 125
    try:
        start_fd = int(argv[1])
    except ValueError:
        return 125
    finalize_handoff = argv[2] == "1"
    role = argv[3]
    command = argv[5:]
    if start_fd <= 2 or not command:
        return 125
    try:
        permitted = os.read(start_fd, 1)
    except OSError:
        return 125
    finally:
        try:
            os.close(start_fd)
        except OSError:
            pass
    if permitted != b"1":
        return 125

    stop_signal = 0

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal stop_signal

        stop_signal = stop_signal or signum

    for handled in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(handled, request_stop)

    group = os.getpgrp()
    handoff_id = os.environ.get(HANDOFF_ID_ENV, "")
    guardian_read, guardian_write = os.pipe()
    os.set_inheritable(guardian_read, True)
    guardian_command = [
        str(Path(sys.executable).resolve()),
        str(Path(__file__).resolve()),
        "--cleanup-guardian",
        str(guardian_read),
        str(os.getpid()),
        str(group),
        "1" if finalize_handoff else "0",
    ]
    guardian: subprocess.Popen[Any] | None = None
    try:
        guardian = subprocess.Popen(
            guardian_command,
            start_new_session=True,
            close_fds=True,
            pass_fds=(*recovery_pass_fds(), guardian_read),
        )
        _register_session_guardian(
            guardian.pid, f"{role}-cleanup", os.getpid()
        )
        os.write(guardian_write, b"1")
    except OSError as exc:
        print(
            f"[recovery-owner] cannot start cleanup guardian: {exc}",
            file=sys.stderr,
            flush=True,
        )
        if guardian is not None:
            try:
                os.killpg(guardian.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            guardian.wait()
        os.close(guardian_read)
        os.close(guardian_write)
        _close_handoff_lock()
        return 127
    finally:
        try:
            os.close(guardian_read)
        except OSError:
            pass

    def release_guardian(message: bytes | None) -> None:
        nonlocal guardian_write

        if guardian_write <= 2:
            return
        try:
            if message is not None:
                os.write(guardian_write, message)
        except OSError:
            pass
        finally:
            os.close(guardian_write)
            guardian_write = -1

    primary_read, primary_write = os.pipe()
    os.set_inheritable(primary_read, True)
    primary_command = [
        str(Path(sys.executable).resolve()),
        str(Path(__file__).resolve()),
        "--owned-command",
        str(primary_read),
        "--",
        *command,
    ]
    child: subprocess.Popen[Any] | None = None
    try:
        child = subprocess.Popen(
            primary_command,
            close_fds=True,
            pass_fds=(*recovery_pass_fds(), primary_read),
        )
        _register_session_primary(child.pid, f"{role}-primary", os.getpid())
        state_dir = _root_state_dir()
        if finalize_handoff and state_dir is not None:
            durable_write_text(state_dir / RESTART_PRIMARY_PID, str(child.pid) + "\n")
        os.write(primary_write, b"1")
    except OSError as exc:
        print(
            f"[recovery-owner] cannot start command: {exc}", file=sys.stderr, flush=True
        )
        if child is not None:
            _signal_group_members(group, signal.SIGKILL)
            child.wait()
        release_guardian(b"A")
        if guardian is not None:
            guardian.wait()
        _close_handoff_lock()
        return 127
    except BaseException:
        if child is not None:
            _signal_group_members(group, signal.SIGKILL)
            child.wait()
        release_guardian(b"A")
        if guardian is not None:
            guardian.wait()
        _close_handoff_lock()
        raise
    finally:
        os.close(primary_read)
        os.close(primary_write)

    forwarded_signal = 0
    termination_deadline: float | None = None
    while child.poll() is None:
        requested_grace = _termination_grace_seconds(os.environ.get(HANDOFF_ID_ENV, ""))
        requested_signal = stop_signal or (
            signal.SIGTERM if requested_grace is not None else 0
        )
        if requested_signal and forwarded_signal != requested_signal:
            _signal_group_members(group, signal.Signals(requested_signal))
            forwarded_signal = requested_signal
            termination_deadline = time.monotonic() + (
                requested_grace if requested_grace is not None else 5.0
            )
        if (
            termination_deadline is not None
            and time.monotonic() >= termination_deadline
        ):
            _signal_group_members(group, signal.SIGKILL)
        time.sleep(0.05)
    if finalize_handoff and _HANDOFF_ID_PATTERN.fullmatch(handoff_id):
        _write_root_event(RESTART_EXIT, handoff_id, child.pid, child.returncode)
    requested_grace = _termination_grace_seconds(handoff_id)
    grace_seconds = requested_grace if requested_grace is not None else 5.0
    group_clean = _cleanup_owned_group(group, grace_seconds)
    sessions_clean = True
    if finalize_handoff and _HANDOFF_ID_PATTERN.fullmatch(handoff_id):
        sessions_clean = _request_other_sessions_stop(
            handoff_id,
            grace_seconds,
            exclude_pids={os.getpid(), guardian.pid},
        )
        if group_clean and sessions_clean:
            _write_root_event(RESTART_COMPLETE, handoff_id, child.pid, child.returncode)
    cleanup_committed = group_clean and sessions_clean
    release_guardian(b"C" if cleanup_committed else None)
    if guardian is not None:
        try:
            guardian.wait(timeout=max(0.0, grace_seconds) + 15.0)
        except subprocess.TimeoutExpired:
            pass
    _close_handoff_lock()
    return child.returncode if child.returncode >= 0 else 128 - child.returncode


if __name__ == "__main__":
    entry = sys.argv[1:]
    if entry and entry[0] == "--owned-command":
        raise SystemExit(_owned_command_entry(entry))
    if entry and entry[0] == "--cleanup-guardian":
        raise SystemExit(_cleanup_guardian_entry(entry))
    raise SystemExit(_owned_session_entry(entry))
