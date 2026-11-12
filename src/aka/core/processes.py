"""Read-only local process identities shared by process owners."""
from __future__ import annotations
import ctypes
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class ProcessRecord:
    pid: int
    ppid: int
    pgid: int
    start_token: str
    role: str
    owner_kind: str = ""
    owner_pid: int = 0


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


def _darwin_session_processes() -> dict[int, ProcessRecord]:
    """Read a PID census without spawning ps on each suspension poll."""
    try:
        list_pids = ctypes.CDLL("/usr/lib/libproc.dylib").proc_listallpids
        list_pids.argtypes = [ctypes.c_void_p, ctypes.c_int]
        list_pids.restype = ctypes.c_int
        count = list_pids(None, 0)
        if count <= 0:
            return {}
        capacity = count + 64
        for _ in range(2):
            pids = (ctypes.c_int * capacity)()
            count = list_pids(pids, ctypes.sizeof(pids))
            if count <= 0:
                return {}
            if count < capacity:
                records = (_darwin_process(pid) for pid in pids[:count] if pid > 0)
                return {record.pid: record for record in records if record is not None}
            capacity *= 2
    except (AttributeError, OSError):
        pass
    return {}


def process_snapshot() -> dict[int, ProcessRecord]:
    if sys.platform == "darwin":
        records = _darwin_session_processes()
        if records:
            return records
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


def process_identity(pid: int) -> ProcessRecord | None:
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
    return process_snapshot().get(pid)
