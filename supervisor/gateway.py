#!/usr/bin/env python3
# Copyright 2026 Alibaba Group.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Supervisor-private GPU execution engine.

The coding Agent never imports or mounts this module. Its public
``tools/sandbox.py`` command is a standard-library HTTP facade that forwards
the requested arguments to the scoped Supervisor Runtime.

In gateway mode, native Atrex-Bench correctness/performance commands use
``agate run``; profiling, compile/sanitize checks, and assembly inspection use
the typed ``profile``, ``compile``, and ``disassemble`` jobs. ``dev`` remains the
compatibility escape hatch for workloads those typed interfaces cannot represent
(for example SOL-ExecBench, source-correlated custom profiling, or a community
gateway that explicitly returns ``kind_not_supported``). OpenSSH mode executes
the same allowlisted command bundle through a portable remote runner. Every
invocation is stateless; callers must not rely on remote filesystem persistence.

Examples::

    python tools/sandbox.py --kind run --hardware REMOTE_GPU --no-sync -- python test_kernel.py --no-memory
    python tools/sandbox.py --kind run --hardware REMOTE_GPU --mode correctness_only --no-sync
    python tools/sandbox.py --kind run --hardware REMOTE_GPU \
        --input-path scratch/custom-input.py --shapes-path scratch/custom-shapes.json --no-sync
    python tools/sandbox.py --kind run --hardware REMOTE_GPU --mode full \
        --baseline-path scratch/baseline.py --comparison-repeats 2 --no-sync
    python tools/sandbox.py --kind profile --hardware REMOTE_GPU --sync profiles/v1 -- \
        bash tools/profile_nvidia.sh profile_driver.py --output-dir profiles/v1 --source
    python tools/sandbox.py --kind profile --hardware REMOTE_ACCELERATOR --gateway-profile pre --sync profiles/v1 -- \
        bash tools/profile_kernel.sh profile_driver.py --output-dir profiles/v1
    python tools/sandbox.py --kind check --hardware REMOTE_GPU --no-sync
    python tools/sandbox.py --kind check --hardware REMOTE_GPU \
        --requirement 'custom-kernel-package==1' --deps-mode no_deps --no-sync
    python tools/sandbox.py --kind disassemble --hardware REMOTE_GPU --format isa --no-sync
    python tools/sandbox.py --kind env --env-gpu REMOTE_GPU --env-capabilities
    python tools/sandbox.py --kind record-read --record-id gateway-<timestamp>-<digest>
    python tools/sandbox.py --kind record-read --record-id kernel-<timestamp>-<id> \
        --view gateway-records
    python tools/sandbox.py --kind record-read --record-id kernel-<timestamp>-<id> \
        --view source --output-path scratch/prior-kernel.py
    python tools/sandbox.py --kind run --hardware H20 --ssh gpu-host --no-sync -- \
        python test_kernel.py --no-memory

``ATREX_SANDBOX_GPU``, ``ATREX_SANDBOX_PROFILE``, ``ATREX_SANDBOX_URL``,
``ATREX_SANDBOX_SSH_GPU``, and ``ATREX_SANDBOX_TIMEOUT`` provide defaults for
the corresponding flags.  A
localhost gateway uses the same transport as a remote worker, for example
``ATREX_SANDBOX_GPU=local`` plus
``ATREX_SANDBOX_URL=http://127.0.0.1:8000``.  Authentication and any remaining
URL resolution stay agate's responsibility (AGATE_* or ~/.atrex/config.json).
With a standard agate gateway profile, synchronized remote files are packed once
on the worker, transferred through OSS, integrity-checked, and extracted locally.
Custom endpoints selected by URL, ``AGATE_URL``, or agate config retain inline
transport because gateways do not currently advertise OSS capability.

``ATREX_SANDBOX_SSH`` selects a standard OpenSSH target (including aliases from
``~/.ssh/config``). ``ATREX_SANDBOX_SSH_INIT`` optionally activates the remote
runtime before each command and health probe. SSH jobs are always executed in a
Bubblewrap namespace with no network, no host home, and only explicitly bound
runtime paths and one explicitly assigned physical NVIDIA GPU. SSH, gateway
profile, and gateway URL transports are mutually exclusive.
"""

from __future__ import annotations

import argparse
import ast
import base64
import fcntl
import hashlib
import io
import json
import math
import os
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orchestrator.constants import ATREX_BENCH_RUNTIME_ENV  # noqa: E402
from orchestrator.durable_state import (  # noqa: E402
    durable_write_json,
    ensure_private_directory,
    fsync_directory,
)
from orchestrator.optimization_policy import (  # noqa: E402
    MODE_STATE_ENV,
    read_policy_file,
    read_workspace_policy,
)
from orchestrator.ssh_health import (  # noqa: E402
    DEFAULT_SSH_HEALTH_COMMAND,
    combined_health_command,
)

from supervisor.errors import (  # noqa: E402
    ESCALATE_RUNTIME,
    AgentRequestError,
    RuntimeStateError,
    error_response,
)
from supervisor.runner_assets import (  # noqa: E402
    PROFILE_DRIVER,
    RUNNERS_ROOT,
    evaluation_inputs,
)

DEFAULT_SYNC_PATHS: tuple[str, ...] = ()
INPUT_SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".atrex_environment",
    ".atrex_long_horizon",
    # Memory is optimizer state owned and updated by the local agent.  The pod
    # receives only code/harness inputs and returns test output/profile files.
    "memory",
    # Runtime/knowledge symlinks are useful to the local agent but are not
    # required by correctness, performance, or profiler commands in the pod.
    ".claude",
    ".qoder",
    ".agents",
    "gpu-wiki",
    "reference-projects",
    "skills",
    # Plans are local campaign inputs for the agent, never runtime inputs for
    # the command executing in the GPU pod. In particular, preserved
    # implementation patches can be large enough to push agate's single
    # uploaded-file argument past Linux MAX_ARG_STRLEN.
    "plans",
    # Older resumable workspaces may retain this former plan-plugin cache. It
    # is never a GPU runtime input and can contain large preserved patches.
    ".humanize",
}
INPUT_SKIP_PATHS = {
    # A pod must not recursively submit another sandbox job, and memory updates
    # are deliberately local-only.  Omitting these also leaves useful headroom
    # below the gateway worker's per-argument limit.
    "tools/sandbox.py",
    "tools/local_gateway.py",
    "tools/memory_manager.py",
    # The durable host-side monitor is never invoked inside a GPU worker.  It
    # can grow the materialized tools bundle enough to exceed agate's
    # per-argument limit despite being unrelated to validation.
    "tools/monitor_optimize_tasks.py",
    # Duplicate of kernel.py from a prior session — not a runtime input.
    "_cute_fa_kernel.py",
    # Exploratory test/debug scripts that are not part of the evaluation harness.
    "test_triton_dot.py",
    "test_triton_dot2.py",
    "valid.py",
}
INPUT_SKIP_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".ncu-rep",
    ".att",
    ".pftrace",
    ".otf2",
    # Campaign documentation, plans, and prior profile reports are local agent
    # state.  Remote correctness/profile commands only need executable sources
    # and harness inputs; omitting Markdown also keeps agate's uploaded file
    # arguments below the worker's argv size limit on long-running campaigns.
    ".md",
}
OUTPUT_BEGIN = "__ATREX_SANDBOX_OUTPUT_BEGIN__"
OUTPUT_END = "__ATREX_SANDBOX_OUTPUT_END__"
DEFAULT_COMMAND_TIMEOUT = 600
DEFAULT_EVAL_SHAPE_BATCH_SIZE = 4
DEFAULT_EVAL_BATCH_WORKERS = 4
DEFAULT_ABBA_BATCH_WORKERS = 16
DEFAULT_INFRASTRUCTURE_RETRIES = 5
DEFAULT_INFRASTRUCTURE_RETRY_SECONDS = 5
MEASUREMENT_REPETITIONS = 3
FP4_MAX_REL_L2 = 0.2
MAX_COMMAND_TIMEOUT = 600
DEFAULT_QUEUE_WAIT_GRACE = 14_400
MAX_GATEWAY_JOB_TIMEOUT = 10_800
MAX_DEV_JOB_TIMEOUT = 600
MAX_HTTP_REQUEST_TIMEOUT = 600
SSH_CONNECT_TIMEOUT = 15
ENVIRONMENT_TEMPFAIL = 75
SSH_RUNTIME_BINDS_ENV = "ATREX_SANDBOX_SSH_RUNTIME_BINDS"
SSH_GPU_ENV = "ATREX_SANDBOX_SSH_GPU"
SSH_WATCHDOG_SOURCE = r"""
import os
import signal
import subprocess
import sys

timeout = int(sys.argv[1])
process = subprocess.Popen(sys.argv[2:], start_new_session=True)
try:
    status = process.wait(timeout=timeout)
except subprocess.TimeoutExpired:
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    print(f"[sandbox] remote command timed out after {timeout}s", file=sys.stderr)
    status = 124
raise SystemExit(status)
""".strip()
AGATE_WAIT_SLICE_SECONDS = 300
RUNTIME_CHUNK_BYTES = 20 * 1024
# Small inline workspace bundles stay below Linux MAX_ARG_STRLEN; larger
# bundles use agate's OSS attachment transport.
WORKSPACE_CHUNK_BYTES = 20 * 1024
OSS_WORKSPACE_THRESHOLD_BYTES = 120 * 1024
OSS_OUTPUT_ARCHIVE = "__atrex_outputs.tar.gz"
SUBMITTED_JOB_RE = re.compile(r"\bsubmitted job_id=([A-Za-z0-9_.-]+); polling\.\.\.")
ACTIVE_AGATE_JOBS: dict[str, tuple[str, str, str | None]] = {}
ACTIVE_AGATE_JOBS_LOCK = Lock()
EVALUATION_INPUT_PATHS = frozenset(
    {
        "agent_problem.json",
        "definition.json",
        "input.py",
        "kernel.py",
        "metadata.json",
        "reference.py",
        "roofline.json",
        "shapes.json",
        "solution.json",
        "test_kernel.py",
        "workload.jsonl",
        "config.json",
    }
)
CANDIDATE_RUNTIME_INPUT_PATHS = frozenset(
    {
        "agent_problem.json",
        "definition.json",
        "input.py",
        "kernel.py",
        "reference.py",
        "shapes.json",
        "solution.json",
        "workload.jsonl",
    }
)
NVIDIA_PROFILE_TOOL_INPUT_PATHS = frozenset(
    {
        "tools/profile_nvidia.sh",
        "tools/classify_ncu.py",
    }
)
AMD_PROFILE_TOOL_INPUT_PATHS = frozenset({"tools/profile_kernel.sh"})
OUTPUT_PATH_FLAGS = frozenset({"-o", "--output", "--output-dir"})
TEST_RESULT_PREFIX = "[test_kernel] RESULT_JSON="
ABBA_RESULT_PREFIX = "__ATREX_LONG_HORIZON_ABBA_RESULT__="
PROFILE_RESULT_PREFIX = "[sandbox] PROFILE_JSON="
CHECK_RESULT_PREFIX = "[sandbox] CHECK_JSON="
DISASSEMBLY_RESULT_PREFIX = "[sandbox] DISASSEMBLY_JSON="
ENV_RESULT_PREFIX = "[sandbox] ENV_JSON="
ABBA_RESULT_PUBLIC_PREFIX = "[sandbox] ABBA_JSON="
RECORD_RESULT_PREFIX = "[sandbox] RECORD_JSON="
DEV_RECORD_RESULT_PREFIX = "[sandbox] DEV_RECORD_JSON="
MAX_AGENT_PROFILE_KERNELS = 32
MAX_AGENT_PROFILE_METRICS = 64
MAX_AGENT_EVALUATION_SHAPES = 256
MAX_AGENT_DIAGNOSTICS = 64
MAX_AGENT_DISASSEMBLY_BYTES = 256 * 1024
MAX_CUSTOM_INPUT_SOURCE_BYTES = 128 * 1024
MAX_CUSTOM_SHAPES_BYTES = 256 * 1024
MAX_KERNEL_SOURCE_BYTES = 24 * 1024 * 1024
MAX_GATEWAY_RECORD_BYTES = 1024 * 1024
MAX_AGENT_DEV_STDOUT_BYTES = 64 * 1024
MAX_AGENT_DEV_STDERR_BYTES = 16 * 1024
DIAGNOSTIC_KINDS = frozenset({"check", "disassemble"})
DEPENDENCY_KINDS = frozenset({"profile", *DIAGNOSTIC_KINDS})
TYPED_KINDS = frozenset({"run", "profile", *DIAGNOSTIC_KINDS})
KERNEL_ARTIFACT_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
KERNEL_RECORD_ID_RE = re.compile(r"kernel-[0-9]+-[0-9a-f]{12}")
TYPED_FALLBACK_REASONS = (
    "kind_not_supported",
    "invalid_source",
    "source validation failed",
    "deps_install_failed",
    "gateway_dependency_install_failed",
    "http 404",
    "http 413",
    "http 501",
)
AGENT_PROBLEM_FILENAME = "agent_problem.json"
PRIVATE_REFERENCE_ENV = "ATREX_PRIVATE_REFERENCE_DIR"
PRIVATE_EVALUATOR_FILENAMES = ("shapes.json", "metadata.json", "roofline.json")
PRIVATE_PROFILE_CASE_FILENAME = ".atrex_private_profile_case.json"
EPISODE_EVALUATIONS_PATH = ".atrex_long_horizon/evaluations.jsonl"
GATEWAY_RECORDS_PATH = ".atrex_long_horizon/gateway-records"
SUPERVISOR_EVIDENCE_ROOT_ENV = "ATREX_AKA_SUPERVISOR_EVIDENCE_ROOT"
SUPERVISOR_HISTORY_ROOT_ENV = "ATREX_AKA_SUPERVISOR_HISTORY_ROOT"
REUSE_GATEWAY_RESULTS_ENV = "ATREX_AKA_REUSE_GATEWAY_RESULTS"
INTERNAL_MEASUREMENT_ENV = "ATREX_AKA_INTERNAL_MEASUREMENT"
COMPARISON_RUN_TIMEOUT_ENV = "ATREX_AKA_COMPARISON_RUN_TIMEOUT"
SUPERVISOR_MEASUREMENT_PREFIX = "__ATREX_SUPERVISOR_MEASUREMENT__="
GATEWAY_RECORD_ID_RE = re.compile(r"gateway-[0-9]+-[0-9a-f]{12}")
PROFILE_ENVIRONMENT_KEYS = (
    "PROFILE_ITERS",
    "PROFILE_WARMUP",
    "PROFILE_WORKLOAD_IDX",
    "PROFILE_SHAPE_ID",
    "PROFILE_DEVICE",
)


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"path must be relative to the workspace: {value!r}")
    normalized = path.as_posix()
    if normalized in ("", "."):
        raise ValueError(f"path must not resolve to the workspace root: {value!r}")
    return normalized


def _read_workspace_override(
    workspace: Path,
    value: str,
    *,
    field: str,
    max_bytes: int,
) -> str:
    """Read one regular Agent-authored override without following it outside the workspace."""
    normalized = _safe_relative(value)
    parts = PurePosixPath(normalized).parts
    if parts[0] in INPUT_SKIP_DIRS or normalized in INPUT_SKIP_PATHS:
        raise ValueError(f"--{field} cannot read Runtime or protected workspace state")
    root = workspace.resolve()
    path = workspace / normalized
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"--{field} must name a regular workspace file")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ValueError(f"--{field} resolves outside the workspace") from exc
    payload = resolved.read_bytes()
    if len(payload) > max_bytes:
        raise ValueError(f"--{field} exceeds the {max_bytes} byte limit")
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"--{field} must be UTF-8") from exc


def _find_agate() -> str | None:
    """Find agate beside the active Python before consulting the shell PATH."""
    adjacent = Path(sys.executable).resolve().parent / "agate"
    if adjacent.is_file() and os.access(adjacent, os.X_OK):
        return str(adjacent)
    return shutil.which("agate")


def _uses_standard_oss_gateway(agate_executable: str, *, url: str, profile: str | None) -> bool:
    """Return whether agate resolves to one of its standard gateway profiles."""
    if url:
        return False
    if profile in {"pre", "prod"}:
        return True

    def resolved_url(*options: str) -> str | None:
        try:
            completed = subprocess.run(
                [agate_executable, "config", *options],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            config = json.loads(completed.stdout) if completed.returncode == 0 else {}
            value = config.get("url") if isinstance(config, dict) else None
            return value.rstrip("/") if isinstance(value, str) else None
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            return None

    selected = resolved_url()
    standard = {
        value
        for value in (
            resolved_url("--profile", "pre"),
            resolved_url("--profile", "prod"),
        )
        if value is not None
    }
    return selected is not None and selected in standard


def _resolved_gateway_url(
    agate_executable: str | None, *, url: str, profile: str | None
) -> str | None:
    """Resolve Agate's selected endpoint without exposing it to the Agent."""
    if url:
        return url.rstrip("/")
    if agate_executable is None:
        return None
    command = [agate_executable, "config"]
    if profile:
        command += ["--profile", profile]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        value = json.loads(completed.stdout) if completed.returncode == 0 else {}
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return None
    selected = value.get("url") if isinstance(value, dict) else None
    return selected.rstrip("/") if isinstance(selected, str) and selected else None


def _is_loopback_gateway_url(value: str) -> bool:
    """Whether an endpoint is the trusted localhost development Gateway."""
    try:
        host = urllib.parse.urlsplit(value).hostname
    except ValueError:
        return False
    return host in {"127.0.0.1", "::1", "localhost"}


def _walk_files(root: Path) -> Iterable[Path]:
    """Yield regular files below root without following directory symlinks."""
    for current, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [
            name
            for name in dirs
            if name not in INPUT_SKIP_DIRS and not (Path(current) / name).is_symlink()
        ]
        for name in files:
            path = Path(current) / name
            if path.is_file() and not path.is_symlink():
                yield path


def _make_input_bundle(
    workspace: Path,
    max_file_bytes: int,
    input_paths: Iterable[str] = (),
    injected_inputs: dict[str, Path] | None = None,
    injected_payloads: dict[str, bytes] | None = None,
) -> tuple[str, int, list[str]]:
    """Return a base64 tarball containing only explicitly selected inputs."""
    archive = io.BytesIO()
    seen: set[str] = set()
    skipped: list[str] = []
    count = 0
    selected_inputs = frozenset(input_paths)

    def add_file(tf: tarfile.TarFile, path: Path, arcname: str) -> None:
        nonlocal count
        if (
            arcname in seen
            or arcname in INPUT_SKIP_PATHS
            or path.suffix in INPUT_SKIP_SUFFIXES
            or arcname not in selected_inputs
        ):
            return
        try:
            size = path.stat().st_size
        except OSError as exc:
            skipped.append(f"{arcname} ({exc})")
            return
        if size > max_file_bytes:
            skipped.append(f"{arcname} ({size} bytes > input limit)")
            return
        tf.add(path, arcname=arcname, recursive=False)
        seen.add(arcname)
        count += 1

    def add_tree(tf: tarfile.TarFile, source: Path, prefix: str = "") -> None:
        if not source.is_dir():
            return
        for path in _walk_files(source):
            rel = path.relative_to(source).as_posix()
            arcname = f"{prefix}/{rel}" if prefix else rel
            add_file(tf, path, arcname)

    def add_payload(tf: tarfile.TarFile, payload: bytes, arcname: str) -> None:
        nonlocal count
        if arcname in seen or arcname not in selected_inputs:
            return
        if len(payload) > max_file_bytes:
            skipped.append(f"{arcname} ({len(payload)} bytes > input limit)")
            return
        info = tarfile.TarInfo(arcname)
        info.size = len(payload)
        info.mode = 0o400
        tf.addfile(info, io.BytesIO(payload))
        seen.add(arcname)
        count += 1

    with tarfile.open(fileobj=archive, mode="w:gz") as tf:
        # Evaluator-only inputs are added before the public workspace tree so a candidate-created
        # file with the same name cannot shadow the orchestrator-owned private test set.
        for arcname, path in (injected_inputs or {}).items():
            add_file(tf, path, arcname)
        for arcname, payload in (injected_payloads or {}).items():
            add_payload(tf, payload, arcname)
        # Profile wrappers remain trusted even when tools/ is Agent-writable.
        # Supply only selected helpers, before any same-named workspace file.
        for arcname in sorted(selected_inputs):
            if (arcname in NVIDIA_PROFILE_TOOL_INPUT_PATHS | AMD_PROFILE_TOOL_INPUT_PATHS
                    or arcname.startswith("tools/ncu_helpers/")):
                add_file(tf, REPO_ROOT / arcname, arcname)
        add_tree(tf, workspace)
        # Supervisor worktrees may still use the sealed tool link; Agent views
        # contain real writable directories and contribute their selected scripts.
        workspace_tools = workspace / "tools"
        if workspace_tools.is_symlink() or not workspace_tools.exists():
            add_tree(tf, REPO_ROOT / "tools", "tools")
        # ``skills/`` is normally a runtime symlink and is intentionally skipped during the
        # workspace walk.  Materialize only explicitly selected skill files so a profiling
        # snapshot can use its backend on the worker without uploading every installed skill.
        if any(path.startswith("skills/") for path in selected_inputs):
            add_tree(tf, workspace / "skills", "skills")
    return base64.b64encode(archive.getvalue()).decode("ascii"), count, skipped


def _declared_candidate_sources(workspace: Path) -> set[str]:
    """Return candidate sources declared by solution.json."""
    selected: set[str] = set()
    solution_path = workspace / "solution.json"
    if solution_path.is_file():
        try:
            solution = json.loads(solution_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid workspace solution.json: {exc}") from exc
        sources = solution.get("sources", []) if isinstance(solution, dict) else []
        if not isinstance(sources, list):
            raise RuntimeError("workspace solution.json sources must be a list of paths")
        for source in sources:
            if isinstance(source, str):
                source_path = source
            elif isinstance(source, dict) and isinstance(source.get("path"), str):
                source_path = source["path"]
            else:
                raise RuntimeError(
                    "workspace solution.json source entries must be paths or path objects"
                )
            selected.add(_safe_relative(source_path))

    return selected


def _evaluation_input_paths(workspace: Path, command: Iterable[str] = ()) -> frozenset[str]:
    """Return only files required by the immutable evaluator."""
    selected = set(EVALUATION_INPUT_PATHS) | _declared_candidate_sources(workspace)
    referenced = _referenced_workspace_inputs(workspace, _command_parts(list(command)))
    selected.update(referenced)
    for path in referenced:
        if "verification_artifacts" not in PurePosixPath(path).parts:
            continue
        snapshots = PurePosixPath(path).parent / "snapshots"
        if (workspace / snapshots).is_dir():
            selected.update(_expand_workspace_input(workspace, snapshots.as_posix()))
    return frozenset(selected)


def _candidate_runtime_input_paths(workspace: Path) -> set[str]:
    """Return candidate and workload modules needed by profile/import commands."""
    selected = {path for path in CANDIDATE_RUNTIME_INPUT_PATHS if (workspace / path).is_file()}
    selected.update(_declared_candidate_sources(workspace))
    return selected


def _expand_workspace_input(workspace: Path, value: str) -> set[str]:
    """Expand one explicitly named workspace file or directory."""
    normalized = _safe_relative(value)
    source = workspace / normalized
    if source.is_file():
        return {normalized}
    if source.is_dir():
        return {
            f"{normalized}/{path.relative_to(source).as_posix()}" for path in _walk_files(source)
        }
    raise ValueError(f"sandbox input does not exist: {value!r}")


def _parsed_command_parts(parts: list[str]) -> tuple[list[str], bool]:
    """Return command words and whether a single shell string stayed opaque."""
    command = parts[1:] if parts and parts[0] == "--" else list(parts)
    if len(command) != 1:
        return command, False
    try:
        parsed = shlex.split(command[0])
    except ValueError:
        return command, True
    if shlex.join(parsed) == command[0]:
        return parsed, False
    return command, True


def _command_parts(parts: list[str]) -> list[str]:
    return _parsed_command_parts(parts)[0]


def _python_inline_imports(parts: list[str]) -> set[str]:
    """Return top-level modules imported by a direct ``python -c`` command."""
    if not parts or not re.fullmatch(r"python(?:[0-9.]+)?", Path(parts[0]).name):
        return set()
    try:
        code_index = parts.index("-c") + 1
        tree = ast.parse(parts[code_index])
    except (ValueError, IndexError, SyntaxError):
        return set()
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
    return imported


def _referenced_workspace_inputs(workspace: Path, parts: list[str]) -> set[str]:
    """Return existing workspace paths explicitly referenced by the command."""
    selected: set[str] = set()
    skip_next = False
    for index, token in enumerate(parts):
        if skip_next:
            skip_next = False
            continue
        if token in OUTPUT_PATH_FLAGS:
            skip_next = True
            continue
        if any(token.startswith(flag + "=") for flag in OUTPUT_PATH_FLAGS):
            continue
        # Code supplied to python/shell -c is not a path. Inputs opened from a
        # custom code string must be declared explicitly with --input.
        if index > 0 and parts[index - 1] in {"-c", "--command"}:
            continue
        try:
            normalized = _safe_relative(token)
        except ValueError:
            continue
        source = workspace / normalized
        if source.is_file():
            selected.add(normalized)
        elif source.is_dir():
            selected.update(_expand_workspace_input(workspace, normalized))
    return selected


def _command_input_paths(
    workspace: Path,
    command: list[str],
    explicit_inputs: Iterable[str] = (),
) -> frozenset[str]:
    """Build the minimal allowlist for a non-evaluator sandbox command.

    Arbitrary commands intentionally start with an empty workspace. Existing
    paths named on the command line are uploaded automatically, while hidden or
    dynamically opened dependencies must be declared with ``--input``.
    """
    parts = _command_parts(command)
    selected: set[str] = set()
    for value in explicit_inputs:
        selected.update(_expand_workspace_input(workspace, value))
    selected.update(_referenced_workspace_inputs(workspace, parts))

    basenames = {Path(token).name for token in parts}
    imports = _python_inline_imports(parts)
    candidate_command = bool(
        imports & {"kernel", "input", "reference"}
        or basenames
        & {
            "kernel.py",
            "profile_driver.py",
            "profile_entry.py",
            "profile_nvidia.sh",
            "profile_kernel.sh",
            "extract_ttgir.py",
        }
        or any("harness" in PurePosixPath(path).parts for path in selected)
    )
    if candidate_command:
        selected.update(_candidate_runtime_input_paths(workspace))

    if basenames & {"profile_nvidia.sh", "profile_entry.py"}:
        selected.update(NVIDIA_PROFILE_TOOL_INPUT_PATHS)
        ncu_helpers = REPO_ROOT / "tools" / "ncu_helpers"
        if ncu_helpers.is_dir():
            selected.update(
                f"tools/ncu_helpers/{path.relative_to(ncu_helpers).as_posix()}"
                for path in _walk_files(ncu_helpers)
            )
    if basenames & {"profile_kernel.sh", "profile_entry.py"}:
        selected.update(AMD_PROFILE_TOOL_INPUT_PATHS)

    if "profile_driver.py" in basenames or "profile_entry.py" in basenames:
        selected.add("profile_driver.py")
    if "profile_entry.py" in basenames:
        selected.add("profile_entry.py")

    # Profile drivers can have sibling helper modules imported by name. Upload
    # that small harness directory, never the complete profiles tree.
    for path in tuple(selected):
        path_parts = PurePosixPath(path).parts
        if "harness" not in path_parts:
            continue
        harness_index = path_parts.index("harness")
        harness_dir = PurePosixPath(*path_parts[: harness_index + 1]).as_posix()
        if (workspace / harness_dir).is_dir():
            selected.update(_expand_workspace_input(workspace, harness_dir))
    return frozenset(selected)


def _supervisor_runtime_inputs(command: list[str]) -> dict[str, Path]:
    """Inject opaque Runtime evidence named by a legacy remote command.

    The path is intentionally absent from the coding Agent's filesystem.  The
    trusted host-side proxy recognizes the stable argument spelling and places
    the authoritative file into the remote GPU bundle under that same name.
    """
    root_value = os.environ.get(SUPERVISOR_EVIDENCE_ROOT_ENV, "").strip()
    if not root_value:
        return {}
    requested = False
    for token in _command_parts(command):
        if token == EPISODE_EVALUATIONS_PATH or token.endswith("=" + EPISODE_EVALUATIONS_PATH):
            requested = True
            break
    if not requested:
        return {}
    source = Path(root_value) / "evaluations.jsonl"
    if not source.is_file() or source.is_symlink():
        raise ValueError("Supervisor evaluation evidence is unavailable")
    return {EPISODE_EVALUATIONS_PATH: source}


def _standard_command_name(value: str, names: set[str]) -> str | None:
    """Return a command name only for PATH lookup or a conventional system path."""
    name = Path(value).name
    if name in names and value in {name, f"/bin/{name}", f"/usr/bin/{name}"}:
        return name
    return None


def _command_executable_index(command: list[str], *, typed_launcher: bool = False) -> int | None:
    """Skip supported shell assignments, env, and execution wrappers."""

    def assignment_end(start: int, *, shell_prefix: bool = False) -> int | None:
        while start < len(command):
            name, separator, _ = command[start].partition("=")
            if not separator or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
                break
            if shell_prefix and shlex.quote(command[start]) != command[start]:
                break
            if typed_launcher and name not in {
                "PYTHONDONTWRITEBYTECODE",
                "PYTHONUNBUFFERED",
            }:
                return None
            start += 1
        return start

    index = assignment_end(0, shell_prefix=True)
    if index is None:
        return None
    if index < len(command) and command[index] in {"env", "/usr/bin/env"}:
        index += 1
        while index < len(command):
            option = command[index]
            if option == "--":
                index += 1
                break
            if option == "-" or option == "--ignore-environment" or option == "--debug":
                if typed_launcher:
                    return None
                index += 1
                continue
            if re.fullmatch(r"-[iv]+", option):
                if typed_launcher:
                    return None
                index += 1
                continue
            if option in {"-C", "--chdir", "-u", "--unset"}:
                if index + 1 >= len(command):
                    return None
                if typed_launcher:
                    return None
                index += 2
                continue
            if (option.startswith(("-C", "-u")) and len(option) > 2) or (
                option.startswith(("--chdir=", "--unset=")) and option.partition("=")[2]
            ):
                if typed_launcher:
                    return None
                index += 1
                continue
            if option.startswith("-"):
                return None
            break
        index = assignment_end(index)
        if index is None:
            return None
    while index < len(command):
        launcher = command[index]
        wrapper = _standard_command_name(
            launcher,
            {"command", "exec", "nice", "nohup", "stdbuf", "time", "timeout"},
        )
        if wrapper is None:
            break
        if typed_launcher:
            return None
        index += 1

        if wrapper == "command":
            while index < len(command):
                option = command[index]
                if option == "--":
                    index += 1
                    break
                if option == "-p":
                    index += 1
                    continue
                if option.startswith("-"):
                    return None
                break
        elif wrapper == "exec":
            while index < len(command):
                option = command[index]
                if option == "--":
                    index += 1
                    break
                if option == "-a":
                    if index + 1 >= len(command):
                        return None
                    index += 2
                    continue
                if re.fullmatch(r"-[cl]+", option):
                    index += 1
                    continue
                if option.startswith("-"):
                    return None
                break
        elif wrapper == "nice":
            while index < len(command):
                option = command[index]
                if option == "--":
                    index += 1
                    break
                if option in {"-n", "--adjustment"}:
                    if index + 1 >= len(command):
                        return None
                    index += 2
                    continue
                if re.fullmatch(r"-(?:n)?\d+", option) or option.startswith("--adjustment="):
                    index += 1
                    continue
                if option.startswith("-"):
                    return None
                break
        elif wrapper == "time":
            while index < len(command):
                option = command[index]
                if option == "--":
                    index += 1
                    break
                if option in {"-f", "--format", "-o", "--output"}:
                    if index + 1 >= len(command):
                        return None
                    index += 2
                    continue
                if (
                    re.fullmatch(r"-[fo].+", option)
                    or option.startswith("--format=")
                    or option.startswith("--output=")
                    or re.fullmatch(r"-[apqv]+", option)
                    or option in {"--append", "--portability", "--quiet", "--verbose"}
                ):
                    index += 1
                    continue
                if option.startswith("-"):
                    return None
                break
        elif wrapper == "timeout":
            while index < len(command):
                option = command[index]
                if option == "--":
                    index += 1
                    break
                if option in {"-k", "--kill-after", "-s", "--signal"}:
                    if index + 1 >= len(command):
                        return None
                    index += 2
                    continue
                if (
                    re.fullmatch(r"-[ks].+", option)
                    or option.startswith(("--kill-after=", "--signal="))
                    or option in {"--foreground", "--preserve-status", "--verbose"}
                ):
                    index += 1
                    continue
                if option.startswith("-"):
                    return None
                break
            if index >= len(command):
                return None
            index += 1
        elif wrapper == "stdbuf":
            while index < len(command):
                option = command[index]
                if option == "--":
                    index += 1
                    break
                if option in {"-e", "--error", "-i", "--input", "-o", "--output"}:
                    if index + 1 >= len(command):
                        return None
                    index += 2
                    continue
                if re.fullmatch(r"-[eio].+", option) or option.startswith(
                    ("--error=", "--input=", "--output=")
                ):
                    index += 1
                    continue
                if option.startswith("-"):
                    return None
                break
        else:
            if index < len(command) and command[index] == "--":
                index += 1
            elif index < len(command) and command[index].startswith("-"):
                return None
    if index < len(command) and command[index] in {"env", "/usr/bin/env"}:
        nested = _command_executable_index(command[index:], typed_launcher=typed_launcher)
        return index + nested if nested is not None else None
    return index if index < len(command) else None


def _python_script_index(
    parts: list[str], script_name: str, *, typed_launcher: bool = False
) -> int | None:
    """Locate a Python script; optionally require a prefix typed run may omit."""
    command, opaque = _parsed_command_parts(parts)
    if opaque:
        return None
    index = _command_executable_index(command, typed_launcher=typed_launcher)

    if index is None or re.fullmatch(r"python(?:3(?:\.\d+)*)?", Path(command[index]).name) is None:
        return None
    index += 1
    while index < len(command):
        option = command[index]
        if option == "--":
            index += 1
            break
        if re.fullmatch(r"-[bBdEiIOPqRsSuvx]+", option):
            if typed_launcher and re.fullmatch(r"-[Bu]+", option) is None:
                return None
            index += 1
            continue
        if option in {"-W", "-X"}:
            if index + 1 >= len(command):
                return None
            if typed_launcher:
                return None
            index += 2
            continue
        if len(option) > 2 and option.startswith(("-W", "-X")):
            if typed_launcher:
                return None
            index += 1
            continue
        if option == "--check-hash-based-pycs":
            if index + 1 >= len(command) or command[index + 1] not in {
                "always",
                "default",
                "never",
            }:
                return None
            if typed_launcher:
                return None
            index += 2
            continue
        if option.startswith("-"):
            return None
        break

    if index < len(command) and Path(command[index]).name == script_name:
        return index
    return None


def _test_kernel_script_index(parts: list[str], *, typed_launcher: bool = False) -> int | None:
    return _python_script_index(parts, "test_kernel.py", typed_launcher=typed_launcher)


def _is_test_kernel_command(parts: list[str]) -> bool:
    return _test_kernel_script_index(parts) is not None


def _shell_command_operand(command: list[str], executable_index: int) -> tuple[str, int] | None:
    """Locate a shell script or the command string consumed by ``-c``."""
    shell = _standard_command_name(command[executable_index], {"bash", "sh"})
    if shell is None:
        return None
    index = executable_index + 1
    while index < len(command):
        option = command[index]
        if option == "--":
            index += 1
            break
        if shell == "bash" and option in {"--init-file", "--rcfile"}:
            if index + 1 >= len(command):
                return None
            index += 2
            continue
        if shell == "bash" and (
            option.startswith("--init-file=") or option.startswith("--rcfile=")
        ):
            index += 1
            continue
        if shell == "bash" and option in {
            "--debug",
            "--debugger",
            "--login",
            "--noediting",
            "--noprofile",
            "--norc",
            "--posix",
            "--protected",
            "--restricted",
            "--verbose",
        }:
            index += 1
            continue
        if shell == "bash" and option in {
            "--dump-po-strings",
            "--dump-strings",
            "--help",
            "--version",
            "--wordexp",
        }:
            return None
        if re.fullmatch(r"-[abefhiklmpruvxBCHP]*c", option):
            return ("command", index + 1) if index + 1 < len(command) else None
        if re.fullmatch(r"-[abefhiklmpruvxBCHP]*[oO]", option):
            if index + 1 >= len(command):
                return None
            index += 2
            continue
        if re.fullmatch(r"-[abefhiklmpruvxBCHP]+", option):
            index += 1
            continue
        if option.startswith("-"):
            return None
        break
    return ("script", index) if index < len(command) else None


def _is_profile_command(parts: list[str]) -> bool:
    """Return whether argv invokes one of the repository profiler wrappers."""
    command, opaque = _parsed_command_parts(parts)
    if opaque:
        return False
    if any(
        _python_script_index(command, name) is not None
        for name in ("profile_driver.py", "profile_entry.py")
    ):
        return True
    index = _command_executable_index(command)
    if index is None:
        return False
    frontend = _standard_command_name(command[index], {"ncu", "nsys", "rocprofv3"})
    if frontend is not None and (
        frontend != "nsys" or (index + 1 < len(command) and command[index + 1] == "profile")
    ):
        for nested_index in range(index + 1, len(command)):
            if _python_script_index(command[nested_index:], "profile_driver.py") is not None:
                return True
    wrappers = {"tools/profile_nvidia.sh", "tools/profile_kernel.sh"}
    executable = PurePosixPath(command[index]).as_posix()
    if Path(executable).name == "profile_driver.py" or executable in wrappers:
        return True
    operand = _shell_command_operand(command, index)
    return bool(
        operand
        and operand[0] == "script"
        and PurePosixPath(command[operand[1]]).as_posix() in wrappers
    )


def _mentions_evaluator_target(value: str) -> bool:
    """Find target names in shell text without pretending to parse shell grammar."""
    unquoted = value.translate(str.maketrans("", "", "\\'\""))
    return (
        re.search(
            r"(?<![A-Za-z0-9_.-])"
            r"(?:test_kernel\.py|profile_driver\.py|profile_nvidia\.sh|profile_kernel\.sh)"
            r"(?![A-Za-z0-9_.-])",
            unquoted,
        )
        is not None
    )


def _is_unsafe_target_command(parts: list[str]) -> bool:
    """Reject target-bearing commands outside the supported launcher grammar."""
    command, _ = _parsed_command_parts(parts)
    if _is_test_kernel_command(command) or _is_profile_command(command):
        return False
    return any(_mentions_evaluator_target(token) for token in command)


def _option_value(parts: list[str], name: str, default: Any = None) -> Any:
    """Read a simple ``--flag value``/``--flag=value`` option from command argv."""
    command = _command_parts(parts)
    for index, token in enumerate(command):
        if token == name:
            return command[index + 1] if index + 1 < len(command) else default
        if token.startswith(name + "="):
            return token.split("=", 1)[1]
    return default


def _option_values(parts: list[str], name: str) -> list[str]:
    """Read every repeated ``--flag value``/``--flag=value`` option."""
    command = _command_parts(parts)
    values: list[str] = []
    for index, token in enumerate(command):
        if token == name:
            if index + 1 >= len(command):
                raise ValueError(f"{name} requires a value")
            values.append(command[index + 1])
        elif token.startswith(name + "="):
            values.append(token.split("=", 1)[1])
    return values


def _json_object(path: Path, *, required: bool = False) -> dict[str, Any] | None:
    if not path.is_file():
        if required:
            raise ValueError(f"required typed-gateway input is missing: {path.name}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON in {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _distributed_evaluation_world_size(metadata: object) -> int:
    """Return the GPU count declared by a single-node evaluator contract."""
    if not isinstance(metadata, dict):
        return 1
    benchmark_contract = metadata.get("benchmark_contract")
    if not isinstance(benchmark_contract, dict):
        return 1
    evaluation = benchmark_contract.get("distributed_evaluation")
    if evaluation is None:
        return 1
    if not isinstance(evaluation, dict):
        raise ValueError("benchmark_contract.distributed_evaluation must be an object")
    if evaluation.get("launcher") != "torchrun":
        raise ValueError("benchmark_contract.distributed_evaluation.launcher must be 'torchrun'")
    if evaluation.get("backend") != "nccl":
        raise ValueError("benchmark_contract.distributed_evaluation.backend must be 'nccl'")
    world_size = evaluation.get("world_size")
    if isinstance(world_size, bool) or not isinstance(world_size, int):
        raise ValueError("benchmark_contract.distributed_evaluation.world_size must be an integer")
    if world_size != 2:
        raise ValueError("benchmark_contract.distributed_evaluation.world_size must be 2")
    return world_size


def _workspace_num_gpus(workspace: Path) -> int:
    metadata = _json_object(_evaluator_input_path(workspace, "metadata.json", required=False))
    return _distributed_evaluation_world_size(metadata)


def _is_fp4_dtype(value: object) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.casefold().replace("-", "").replace("_", "")
    return "fp4" in normalized or "float4" in normalized


def _metadata_has_fp4_dtype(metadata: object) -> bool:
    if isinstance(metadata, dict):
        if any(_is_fp4_dtype(metadata.get(field)) for field in ("dtype", "dtype_compute")):
            return True
        return any(_metadata_has_fp4_dtype(value) for value in metadata.values())
    if isinstance(metadata, list):
        return any(_metadata_has_fp4_dtype(value) for value in metadata)
    return False


def _fp4_correctness_max_rel_l2(
    metadata: dict[str, Any] | None,
    operator: object = None,
) -> float | None:
    return FP4_MAX_REL_L2 if _metadata_has_fp4_dtype(metadata) or _is_fp4_dtype(operator) else None


def _is_generalized_workspace(workspace: Path) -> bool:
    """Return whether production policy enables private exact-case handling."""
    policy = os.environ.get(MODE_STATE_ENV)
    state = read_policy_file(Path(policy)) if policy else read_workspace_policy(workspace)
    return state.get("mode") == "production" and (workspace / AGENT_PROBLEM_FILENAME).is_file()


def _private_reference_dir(workspace: Path) -> Path | None:
    """Resolve private evaluator inputs only for a generalized production workspace."""
    if not _is_generalized_workspace(workspace):
        return None
    raw = os.environ.get(PRIVATE_REFERENCE_ENV, "")
    if not raw:
        raise ValueError(
            f"{PRIVATE_REFERENCE_ENV} is required for generalized Atrex-Bench evaluation"
        )
    private_dir = Path(raw).expanduser().resolve()
    if not private_dir.is_dir():
        raise ValueError("configured private Atrex-Bench reference directory is missing")
    private_problem = private_dir / AGENT_PROBLEM_FILENAME
    public_problem = workspace / AGENT_PROBLEM_FILENAME
    # A user-provided contract remains evaluator-owned and must match byte-for-byte.
    # An automatically authored production contract intentionally exists only in the
    # campaign workspace, so absence from the detailed-shape source is valid.
    if private_problem.is_file() and (private_problem.read_bytes() != public_problem.read_bytes()):
        raise ValueError(
            "workspace agent_problem.json does not match the evaluator-owned public contract"
        )
    return private_dir


def _evaluator_input_path(workspace: Path, filename: str, *, required: bool) -> Path:
    private_dir = _private_reference_dir(workspace)
    path = (private_dir / filename) if private_dir is not None else (workspace / filename)
    if required and not path.is_file():
        raise ValueError(f"required evaluator input is missing: {filename}")
    return path


def _private_evaluator_inputs(workspace: Path) -> dict[str, Path]:
    private_dir = _private_reference_dir(workspace)
    if private_dir is None:
        return {}
    inputs: dict[str, Path] = {}
    for filename in PRIVATE_EVALUATOR_FILENAMES:
        path = private_dir / filename
        if filename in {"shapes.json", "metadata.json"} and not path.is_file():
            raise ValueError(f"required private evaluator input is missing: {filename}")
        if path.is_file():
            inputs[filename] = path
    return inputs


def _sort_shape_id(shape_id: str) -> tuple[int, object]:
    return (0, int(shape_id)) if shape_id.isdigit() else (1, shape_id)


def _private_profile_case(
    workspace: Path, env_items: Iterable[str], requested_shape_id: str | None = None
) -> tuple[str, bytes] | None:
    """Materialize exactly one private real shape for an ephemeral remote profile."""
    private_dir = _private_reference_dir(workspace)
    if private_dir is None:
        return None
    shapes = _json_object(private_dir / "shapes.json", required=True)
    if not shapes:
        raise ValueError("private shapes.json must contain a non-empty object")
    environment = _parse_env_items(env_items)
    shape_id = (
        requested_shape_id
        or environment.get("PROFILE_SHAPE_ID")
        or sorted((str(value) for value in shapes), key=_sort_shape_id)[0]
    )
    entry = shapes.get(shape_id)
    if not isinstance(entry, dict):
        raise ValueError(f"PROFILE_SHAPE_ID={shape_id!r} is not a real evaluator shape id")
    payload = {
        "schema_version": 1,
        "shape_id": shape_id,
        "init_kwargs": entry.get("init_kwargs") or {},
        "input_kwargs": entry.get("input_kwargs") or {},
    }
    return (
        PRIVATE_PROFILE_CASE_FILENAME,
        (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"),
    )


def _typed_workspace_limitation(workspace: Path, command: list[str], kind: str) -> str | None:
    """Explain why a typed Gateway source contract cannot represent a workspace."""
    required = (
        ("kernel.py",) if kind in DIAGNOSTIC_KINDS else ("kernel.py", "reference.py", "input.py")
    )
    missing = [name for name in required if not (workspace / name).is_file()]
    if missing:
        return "missing " + ", ".join(missing)
    try:
        _evaluator_input_path(workspace, "shapes.json", required=True)
    except ValueError as exc:
        return str(exc)
    if (
        kind == "run"
        and _is_test_kernel_command(command)
        and _test_kernel_script_index(command, typed_launcher=True) is None
    ):
        return "evaluator launcher semantics require the dev route"
    if kind not in DIAGNOSTIC_KINDS and (workspace / "workload.jsonl").is_file():
        return "SOL-ExecBench workload.jsonl is not supported by the Atrex-Bench typed API"

    solution = _json_object(workspace / "solution.json")
    if solution is not None:
        sources = solution.get("sources")
        if isinstance(sources, list):
            source_paths = {
                str(item.get("path"))
                for item in sources
                if isinstance(item, dict) and item.get("path")
            }
            if source_paths - {"kernel.py"}:
                return "solution.json declares auxiliary candidate sources"

    try:
        tree = ast.parse((workspace / "kernel.py").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError):
        tree = None
    if tree is not None:
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
        local_imports = sorted(
            root
            for root in imported_roots
            if root not in {"input", "reference", "kernel"}
            and (
                (workspace / f"{root}.py").is_file() or (workspace / root / "__init__.py").is_file()
            )
        )
        if local_imports:
            return "candidate imports local auxiliary modules: " + ", ".join(local_imports)

    if kind in DIAGNOSTIC_KINDS:
        return None

    # These test-harness controls do not exist in the typed request contract.
    # Preserve their exact semantics through dev instead of silently dropping them.
    unsupported_options = (
        "--seed",
        "--workspace",
        "--candidate-timeout-s",
        "--perf-timeout-s",
    )
    for option in unsupported_options:
        if _option_value(command, option) is not None:
            return f"{option} is not supported by the typed API"
    warmup = _option_value(command, "--warmup")
    if warmup is not None and str(warmup) != "5":
        return "non-default --warmup is not supported by the typed API"
    for option, default in (("--atol", 1e-2), ("--rtol", 0.05)):
        value = _option_value(command, option)
        if value is not None:
            try:
                matches_default = float(value) == default
            except (TypeError, ValueError):
                matches_default = False
            if not matches_default:
                return f"non-default {option} is not exposed by agate run"
    return None


def _requested_gateway_kind(requested: str, command: list[str]) -> str:
    if requested != "auto":
        return requested
    if _is_test_kernel_command(command):
        return "run"
    if _is_profile_command(command):
        return "profile"
    return "dev"


def _parse_env_items(items: Iterable[str]) -> dict[str, str]:
    env_vars: dict[str, str] = {}
    for item in items:
        if "=" not in item or item.startswith("="):
            raise ValueError(f"invalid --env {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"invalid --env key {key!r}")
        env_vars[key] = value
    return env_vars


def _with_inherited_profile_environment(items: Iterable[str]) -> list[str]:
    """Forward the documented PROFILE_* shell assignments without forwarding secrets."""
    result = list(items)
    configured = _parse_env_items(result)
    for key in PROFILE_ENVIRONMENT_KEYS:
        if key not in configured and key in os.environ:
            result.append(f"{key}={os.environ[key]}")
    return result


def _profile_command_environment(items: Iterable[str]) -> tuple[list[str], list[str]]:
    """Move PROFILE_* controls into the uploaded command for a dev fallback.

    The gateway intentionally accepts only a small environment-variable allowlist,
    which does not include the profiler driver's local PROFILE_* controls.  A
    generalized profile already falls back to an uploaded dev command so it can
    consume one privately injected real shape.  Prefix those non-secret controls
    on that command instead of asking the gateway API to inject them.
    """
    command_environment: list[str] = []
    gateway_environment: list[str] = []
    for item in items:
        key = item.split("=", 1)[0]
        if key in PROFILE_ENVIRONMENT_KEYS:
            command_environment.append(item)
        else:
            gateway_environment.append(item)
    return command_environment, gateway_environment


def _typed_request(
    workspace: Path,
    hardware: str,
    timeout: int,
    env_items: list[str],
    command: list[str],
    kind: str,
    *,
    profiler: str | None = None,
    profile_level: str = "sol",
    counters: Iterable[str] = (),
    kernel_regex: str | None = None,
    kernel_name: str | None = None,
    profile_source: bool = False,
    launch_skip: int | None = None,
    launch_count: int | None = None,
    profile_shape_id: str | None = None,
    top_kernels: int | None = None,
    arch: str | None = None,
    sanitize: str | None = None,
    disassembly_format: str = "auto",
    requirements: Iterable[str] = (),
    deps_mode: str | None = None,
    evaluation_input_path: str | None = None,
    evaluation_shapes_path: str | None = None,
    evaluation_mode: str | None = None,
) -> dict[str, Any]:
    """Build a public typed request without importing the agate package."""
    dependency_specs = list(requirements)
    if len(dependency_specs) > 128:
        raise ValueError("at most 128 --requirement values are allowed")
    for requirement in dependency_specs:
        if (
            not requirement.strip()
            or len(requirement) > 2048
            or "\x00" in requirement
            or "\n" in requirement
            or "\r" in requirement
        ):
            raise ValueError(
                "each --requirement must be one non-empty PEP 508 string of at most 2048 characters"
            )
    if deps_mode not in {None, "freeze_installed", "no_deps"}:
        raise ValueError("--deps-mode must be freeze_installed or no_deps")
    if evaluation_mode not in {None, "full", "correctness_only"}:
        raise ValueError("--mode must be full or correctness_only")
    if kernel_regex and kernel_name:
        raise ValueError("--kernel-regex and --kernel-name are mutually exclusive")
    if profile_level == "deep" and not (kernel_regex or kernel_name):
        raise ValueError("deep profile requires --kernel-regex or --kernel-name")
    if launch_skip is not None and launch_skip < 0:
        raise ValueError("--launch-skip must be non-negative")
    if launch_count is not None and launch_count <= 0:
        raise ValueError("--launch-count must be positive")

    shapes = _json_object(
        _evaluator_input_path(workspace, "shapes.json", required=True), required=True
    )
    assert shapes is not None
    custom_input_source: str | None = None
    custom_shapes = False
    if evaluation_input_path is not None:
        custom_input_source = _read_workspace_override(
            workspace,
            evaluation_input_path,
            field="input-path",
            max_bytes=MAX_CUSTOM_INPUT_SOURCE_BYTES,
        )
        if not custom_input_source.strip():
            raise ValueError("--input-path must contain a non-empty input generator")
    if evaluation_shapes_path is not None:
        shapes_source = _read_workspace_override(
            workspace,
            evaluation_shapes_path,
            field="shapes-path",
            max_bytes=MAX_CUSTOM_SHAPES_BYTES,
        )
        try:
            custom_shape_value = json.loads(shapes_source)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--shapes-path must contain valid JSON: {exc}") from exc
        if not isinstance(custom_shape_value, dict) or not custom_shape_value:
            raise ValueError("--shapes-path must contain a non-empty JSON object")
        for shape_id, shape in custom_shape_value.items():
            try:
                int(shape_id)
            except (TypeError, ValueError) as exc:
                raise ValueError("--shapes-path keys must be integer-parseable strings") from exc
            if not isinstance(shape, dict):
                raise ValueError("each --shapes-path record must be a JSON object")
        shapes = custom_shape_value
        custom_shapes = True
    solution = _json_object(workspace / "solution.json") or {}
    languages = solution.get("languages")
    if not isinstance(languages, list):
        languages = []
    spec: dict[str, Any] = {
        "languages": [str(value) for value in languages],
        "target_hardware": [hardware],
    }

    if kind in DIAGNOSTIC_KINDS:
        if not shapes:
            raise ValueError("diagnostic evaluator Shape contract is empty")
        selected_shape_id = sorted((str(value) for value in shapes), key=_sort_shape_id)[0]
        shape = shapes.get(selected_shape_id)
        if not isinstance(shape, dict):
            raise ValueError(f"shape {selected_shape_id!r} must be an object")
        init_value = shape.get("init_kwargs")
        if init_value is not None and not isinstance(init_value, dict):
            raise ValueError(f"shape {selected_shape_id!r} init_kwargs must be an object or null")
        request: dict[str, Any] = {
            "spec": {"target_hardware": [hardware]},
            "candidate": (workspace / "kernel.py").read_text(encoding="utf-8"),
            "env_vars": _parse_env_items(env_items),
            "init_kwargs": dict(init_value or {}),
            # Local Gateway uses this evaluator-owned input to perform a real
            # one-Shape launch/JIT probe. Remote Agate ignores this extension
            # because it already owns the diagnostic execution contract.
            "diagnostic_reference": {
                "input_py": _evaluator_input_path(workspace, "input.py", required=True).read_text(
                    encoding="utf-8"
                ),
                "shapes": {selected_shape_id: shape},
            },
            "shape_id": selected_shape_id,
        }
        if dependency_specs:
            request["requirements"] = dependency_specs
        if deps_mode is not None:
            request["deps_mode"] = deps_mode
        if kind == "check":
            if arch:
                request["arch"] = arch
            if sanitize:
                request["sanitize"] = sanitize
        else:
            request["fmt"] = disassembly_format
        return request

    requested_shape_ids = _option_values(command, "--shape-id")
    if requested_shape_ids:
        unknown_shape_ids = [shape_id for shape_id in requested_shape_ids if shape_id not in shapes]
        if unknown_shape_ids:
            raise ValueError("unknown --shape-id values: " + ", ".join(unknown_shape_ids))
        # Preserve command order while dropping accidental duplicates. The typed
        # gateway must honor the adapter's targeted-smoke contract instead of
        # silently expanding a one-shape request back to the complete workload.
        requested_shape_ids = list(dict.fromkeys(requested_shape_ids))
        shapes = {shape_id: shapes[shape_id] for shape_id in requested_shape_ids}
    try:
        multi_seed = int(_option_value(command, "--multi-seed", 0))
        bench_iters = int(_option_value(command, "--timed-runs", 20))
        atol = float(_option_value(command, "--atol", 1e-2))
        rtol = float(_option_value(command, "--rtol", 0.05))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid evaluator command option: {exc}") from exc
    if multi_seed < 0 or bench_iters < 1:
        raise ValueError("--multi-seed must be non-negative and --timed-runs must be positive")

    reference: dict[str, Any] = {
        "operator": workspace.name,
        "reference_py": (workspace / "reference.py").read_text(encoding="utf-8"),
        "input_py": (
            custom_input_source
            if custom_input_source is not None
            else (workspace / "input.py").read_text(encoding="utf-8")
        ),
        "shapes": shapes,
    }
    if custom_input_source is None and not custom_shapes:
        for filename, field in (
            ("metadata.json", "metadata"),
            ("roofline.json", "roofline"),
        ):
            value = _json_object(_evaluator_input_path(workspace, filename, required=False))
            if value is not None:
                if requested_shape_ids and isinstance(value.get("shapes"), dict):
                    value = dict(value)
                    value["shapes"] = {
                        shape_id: value["shapes"][shape_id]
                        for shape_id in requested_shape_ids
                        if shape_id in value["shapes"]
                    }
                    if filename == "metadata.json" and "num_shapes" in value:
                        value["num_shapes"] = len(requested_shape_ids)
                reference[field] = value

    num_gpus = _distributed_evaluation_world_size(reference.get("metadata"))
    if num_gpus > 1:
        spec["num_gpus"] = num_gpus
    request: dict[str, Any] = {
        "name": f"{workspace.name}_{kind}",
        "spec": spec,
        "candidate": (workspace / "kernel.py").read_text(encoding="utf-8"),
        "reference": reference,
        "options": {
            "num_correctness_cases": 1 + multi_seed,
            "bench_iters": bench_iters,
            "atol": atol,
            "rtol": rtol,
            "timeout_s": timeout,
        },
        "env_vars": _parse_env_items(env_items),
    }
    correctness_max_rel_l2 = _fp4_correctness_max_rel_l2(
        reference.get("metadata") if isinstance(reference.get("metadata"), dict) else None,
        reference.get("operator"),
    )
    if kind == "run" and correctness_max_rel_l2 is not None:
        request["options"]["correctness_max_rel_l2"] = correctness_max_rel_l2
    if kind == "run":
        version = str(_option_value(command, "--version", "v0"))
        request["mode"] = evaluation_mode or (
            "correctness_only" if multi_seed > 0 and version not in {"v0", "v1"} else "full"
        )
    else:
        if profiler:
            request["profiler"] = profiler
        if profile_level:
            request["level"] = profile_level
        if counters:
            request["counters"] = list(counters)
        if kernel_regex:
            request["kernel_regex"] = kernel_regex
        if kernel_name:
            request["kernel_name"] = kernel_name
        if profile_source:
            request["source"] = True
        if launch_skip is not None:
            request["launch_skip"] = launch_skip
        if launch_count is not None:
            request["launch_count"] = launch_count
        if profile_shape_id is not None:
            if profile_shape_id not in shapes:
                raise ValueError(
                    f"--profile-shape-id {profile_shape_id!r} is not an evaluator-owned Shape id"
                )
            request["shape_id"] = profile_shape_id
            request["reference"]["shapes"] = {profile_shape_id: shapes[profile_shape_id]}
            for field in ("metadata", "roofline"):
                document = request["reference"].get(field)
                if isinstance(document, dict) and isinstance(document.get("shapes"), dict):
                    document = dict(document)
                    document["shapes"] = (
                        {profile_shape_id: document["shapes"][profile_shape_id]}
                        if profile_shape_id in document["shapes"]
                        else {}
                    )
                    if field == "metadata" and "num_shapes" in document:
                        document["num_shapes"] = 1
                    request["reference"][field] = document
        if top_kernels is not None:
            request["top_kernels"] = top_kernels
        if dependency_specs:
            request["requirements"] = dependency_specs
        if deps_mode is not None:
            request["deps_mode"] = deps_mode
    return request


def _make_atrex_bench_runtime_bundle(
    runtime_root: Path | None, *, evaluator_only: bool = False
) -> str | None:
    """Package the private canonical evaluator separately from Agent state.

    The compressed runtime is split into multiple uploaded files by ``main``
    because agate's worker places each file value in one Linux argv entry.
    """
    if runtime_root is None:
        return None
    runtime_root = runtime_root.resolve()
    run_eval = runtime_root / "scripts" / "run_eval.py"
    package = runtime_root / "src" / "atrex_bench"
    utils_module = package / "utils.py"
    sdk_module = package / "sdk.py"
    if (
        not package.is_dir()
        or not run_eval.is_file()
        or (evaluator_only and not utils_module.is_file())
    ):
        raise RuntimeError(f"invalid Supervisor Atrex-Bench runtime: {runtime_root}")

    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tf:
        if evaluator_only:
            evaluator_files = [package / "__init__.py", utils_module]
            # Newer Atrex-Bench releases re-export the Python evaluation API
            # from ``atrex_bench.__init__``.  Keep the file optional so the
            # evaluator-only bundle remains compatible with older releases
            # that predate sdk.py.
            if sdk_module.is_file():
                evaluator_files.append(sdk_module)
            shape_contracts = package / "shape_contracts.py"
            if shape_contracts.is_file():
                evaluator_files.append(shape_contracts)
            evaluator_files.extend(_walk_files(package / "eval"))
            # Newer Atrex-Bench releases moved the CLI implementations into
            # ``atrex_bench.cli`` and kept ``scripts/run_eval.py`` as a thin
            # compatibility wrapper.  _walk_files yields nothing when the
            # directory is absent, so releases without ``cli`` stay compatible.
            evaluator_files.extend(_walk_files(package / "cli"))
            tf.add(run_eval, arcname="atrex-bench/scripts/run_eval.py", recursive=False)
            for path in evaluator_files:
                relative = path.relative_to(package).as_posix()
                tf.add(
                    path,
                    arcname=f"atrex-bench/src/atrex_bench/{relative}",
                    recursive=False,
                )
        else:
            tf.add(run_eval, arcname="atrex-bench/scripts/run_eval.py", recursive=False)
            for path in _walk_files(package):
                relative = path.relative_to(package).as_posix()
                tf.add(
                    path,
                    arcname=f"atrex-bench/src/atrex_bench/{relative}",
                    recursive=False,
                )
    return base64.b64encode(archive.getvalue()).decode("ascii")


def _private_atrex_bench_runtime() -> Path | None:
    """Resolve evaluator code supplied only by the Supervisor process."""
    raw = os.environ.get(ATREX_BENCH_RUNTIME_ENV, "").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute() or path.is_symlink():
        raise ValueError("Supervisor Atrex-Bench runtime must be an absolute real path")
    return path


REMOTE_COLLECTOR = r"""#!/usr/bin/env python3
import base64
import io
import json
import sys
import tarfile
from pathlib import Path, PurePosixPath

BEGIN = "__ATREX_SANDBOX_OUTPUT_BEGIN__"
END = "__ATREX_SANDBOX_OUTPUT_END__"
RAW = {".ncu-rep", ".att", ".pftrace", ".otf2"}

root = Path(sys.argv[1]).resolve()
cfg = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
max_bytes = int(cfg["max_file_bytes"])
transport = cfg.get("transport", "inline")
include_raw = bool(cfg["include_raw_profile"]) or transport == "oss"
skipped = []
seen = set()

def safe(value):
    p = PurePosixPath(value)
    return bool(value) and not p.is_absolute() and ".." not in p.parts

def add_file(tf, path):
    rel = path.relative_to(root).as_posix()
    if rel in seen or path.is_symlink() or not path.is_file():
        return
    size = path.stat().st_size
    if (not include_raw and path.suffix in RAW) or size > max_bytes:
        skipped.append(f"{rel} ({size} bytes)")
        return
    tf.add(path, arcname=rel, recursive=False)
    seen.add(rel)

def collect(tf):
    for value in cfg["paths"]:
        if not safe(value):
            continue
        path = root / value
        if path.is_file():
            add_file(tf, path)
        elif path.is_dir():
            for child in path.rglob("*"):
                add_file(tf, child)

if transport == "oss":
    with tarfile.open(Path(sys.argv[3]), mode="w:gz") as tf:
        collect(tf)
elif transport == "inline":
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tf:
        collect(tf)
    print(BEGIN)
    print(base64.b64encode(archive.getvalue()).decode("ascii"))
    print(END)
elif transport == "ssh":
    with tarfile.open(Path(sys.argv[3]), mode="w:gz") as tf:
        collect(tf)
elif transport != "none":
    raise ValueError(f"unsupported output transport: {transport!r}")
if skipped:
    print("[sandbox] artifacts not returned: " + ", ".join(skipped), file=sys.stderr)
"""


def _runner_source() -> str:
    return r"""#!/usr/bin/env bash
set -uo pipefail
mkdir -p workspace
ws_parts=(__atrex_workspace.tar.gz.b64.part*)
if [[ -e "${ws_parts[0]}" ]]; then
    if ! cat "${ws_parts[@]}" | base64 -d | tar -xzf - -C workspace; then
        echo "[sandbox] failed to unpack workspace" >&2
        exit 97
    fi
elif [[ -f __atrex_workspace.tar.gz.b64 ]]; then
    if ! base64 -d __atrex_workspace.tar.gz.b64 | tar -xzf - -C workspace; then
        echo "[sandbox] failed to unpack workspace" >&2
        exit 97
    fi
fi
runtime_parts=(__atrex_bench_runtime.tar.gz.b64.part*)
if [[ -e "${runtime_parts[0]}" ]]; then
    if ! cat "${runtime_parts[@]}" | base64 -d | tar -xzf - -C workspace; then
        echo "[sandbox] failed to unpack Atrex-Bench evaluator runtime" >&2
        exit 97
    fi
fi
cd workspace
set +e
bash ../__atrex_command.sh
command_status=$?
cd ..
python __atrex_collect.py workspace __atrex_outputs.json __atrex_outputs.tar.gz
collect_status=$?
if [[ $collect_status -ne 0 ]]; then
    exit 98
fi
exit $command_status
"""


def _extract_output_tar(tf: tarfile.TarFile, workspace: Path) -> None:
    """Safely extract a sandbox-owned output archive into ``workspace``."""
    workspace_root = workspace.resolve()
    for member in tf.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"unsafe artifact path returned by sandbox: {member.name!r}")
        if member.issym() or member.islnk():
            raise RuntimeError(f"sandbox artifact links are not accepted: {member.name!r}")
        target = workspace_root / path.as_posix()
        try:
            target.resolve(strict=False).relative_to(workspace_root)
        except ValueError as exc:
            raise RuntimeError(
                f"sandbox artifact resolves outside workspace: {member.name!r}"
            ) from exc
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not member.isfile():
            continue
        source = tf.extractfile(member)
        if source is None:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read())
        try:
            target.chmod(member.mode & 0o777)
        except OSError:
            pass


def _extract_output_archive(archive: Path, workspace: Path) -> None:
    with tarfile.open(archive, mode="r:gz") as tf:
        _extract_output_tar(tf, workspace)


def _extract_outputs(stdout: str, workspace: Path) -> str:
    """Extract a legacy inline archive and return stdout without framing."""
    if OUTPUT_BEGIN not in stdout or OUTPUT_END not in stdout:
        raise RuntimeError("sandbox response did not contain an artifact frame")
    command_stdout, framed = stdout.rsplit(OUTPUT_BEGIN, 1)
    encoded, trailing = framed.split(OUTPUT_END, 1)
    if trailing.strip():
        command_stdout += trailing
    payload = base64.b64decode("".join(encoded.split()), validate=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tf:
        _extract_output_tar(tf, workspace)
    return command_stdout.rstrip("\n")


def _oss_artifact(job: dict[str, Any], name: str) -> dict[str, Any]:
    artifacts = (job.get("result") or {}).get("artifacts") or []
    for artifact in artifacts:
        if isinstance(artifact, dict) and artifact.get("name") == name:
            return artifact
    raise RuntimeError(f"gateway returned no OSS artifact named {name!r}")


def _download_oss_artifact(artifact: dict[str, Any], destination: Path) -> None:
    """Download one presigned OSS artifact and verify gateway metadata."""
    name = str(artifact.get("name") or "artifact")
    url = artifact.get("url")
    if not isinstance(url, str):
        raise RuntimeError(f"OSS artifact {name!r} has no download URL")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise RuntimeError(f"OSS artifact {name!r} has an invalid download URL")

    expected_bytes = artifact.get("bytes")
    expected_sha256 = artifact.get("sha256")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".part",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            digest = hashlib.sha256()
            downloaded = 0
            with urllib.request.urlopen(url, timeout=MAX_HTTP_REQUEST_TIMEOUT) as response:
                final_url = urllib.parse.urlsplit(response.geturl())
                if final_url.scheme not in {"http", "https"} or not final_url.hostname:
                    raise RuntimeError(f"OSS artifact {name!r} redirected to an invalid URL")
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
                    downloaded += len(chunk)
        if (
            isinstance(expected_bytes, int)
            and not isinstance(expected_bytes, bool)
            and downloaded != expected_bytes
        ):
            raise RuntimeError(
                f"OSS artifact {name!r} size mismatch: "
                f"expected {expected_bytes}, received {downloaded}"
            )
        if (
            isinstance(expected_sha256, str)
            and expected_sha256
            and digest.hexdigest() != expected_sha256.casefold()
        ):
            raise RuntimeError(f"OSS artifact {name!r} sha256 mismatch")
        os.replace(temporary, destination)
        temporary = None
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(f"failed to download OSS artifact {name!r}: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _command_text(parts: list[str]) -> str:
    if parts and parts[0] == "--":
        parts = parts[1:]
    if not parts:
        raise ValueError("a command is required after --")
    # A single argument is commonly a deliberately quoted shell pipeline.
    return parts[0] if len(parts) == 1 else shlex.join(parts)


class _AgentArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        # Avoid dumping every Supervisor-only flag for a single bad Agent argument.
        self.exit(2, json.dumps(error_response(message, code="invalid_arguments")) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = _AgentArgumentParser(
        description="Run correctness, performance, or profile commands on a remote GPU sandbox.",
        allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--hardware",
        default=os.environ.get("ATREX_SANDBOX_GPU", ""),
        help="Remote GPU hardware token, e.g. REMOTE_GPU (default: ATREX_SANDBOX_GPU).",
    )
    parser.add_argument(
        "--kind",
        choices=(
            "auto",
            "run",
            "profile",
            "check",
            "disassemble",
            "dev",
            "env",
            "record-read",
        ),
        default="auto",
        help=(
            "Use run/profile without a command; the Supervisor supplies execution drivers. "
            "auto classifies explicit remote commands (default: auto). check "
            "and disassemble are typed diagnostics over the exact kernel.py."
        ),
    )
    parser.add_argument(
        "--record-id",
        default=None,
        metavar="ID",
        help=(
            "Read a Gateway record (gateway-...) or select a Kernel view (kernel-...)."
        ),
    )
    parser.add_argument(
        "--view",
        choices=("source", "gateway-records"),
        default=None,
        help="Required view when --record-id names a Kernel; invalid for Gateway records.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        metavar="SCRATCH_PATH",
        help="Workspace-relative scratch/ destination required by the Kernel source view.",
    )
    parser.add_argument(
        "--gateway-profile",
        choices=("pre", "prod"),
        default=None,
        help="Gateway endpoint profile (default: ATREX_SANDBOX_PROFILE, then normal agate resolution).",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Explicit gateway URL (default: ATREX_SANDBOX_URL; overrides environment profile/config).",
    )
    parser.add_argument(
        "--ssh",
        default=None,
        metavar="[USER@]HOST",
        help=(
            "OpenSSH target for Bubblewrap-isolated remote GPU execution "
            "(default: ATREX_SANDBOX_SSH). Reuses ~/.ssh/config and is "
            "mutually exclusive with gateway endpoint options."
        ),
    )
    parser.add_argument(
        "--ssh-init",
        default=os.environ.get("ATREX_SANDBOX_SSH_INIT", ""),
        metavar="COMMAND",
        help=(
            "Remote shell initialization run before commands and probes, e.g. a "
            "Conda activation (default: ATREX_SANDBOX_SSH_INIT)."
        ),
    )
    parser.add_argument(
        "--ssh-runtime-bind",
        action="append",
        default=None,
        metavar="REMOTE_PATH[=SANDBOX_PATH]",
        help=(
            "Read-only runtime directory exposed inside the SSH Bubblewrap sandbox "
            "(repeatable; default: ATREX_SANDBOX_SSH_RUNTIME_BINDS JSON array)."
        ),
    )
    parser.add_argument(
        "--ssh-gpu",
        default=None,
        metavar="INDEX",
        help=(
            "Physical NVIDIA GPU index exposed to an SSH job (required with --ssh; "
            f"default: {SSH_GPU_ENV}). MIG selectors are rejected until capability-node "
            "assignment is implemented."
        ),
    )
    parser.add_argument(
        "--health-command",
        default=os.environ.get("ATREX_SANDBOX_HEALTH_COMMAND", DEFAULT_SSH_HEALTH_COMMAND),
        metavar="COMMAND",
        help=(
            "Remote GPU health probe used to distinguish candidate failures from "
            "environment failures (default: ATREX_SANDBOX_HEALTH_COMMAND)."
        ),
    )
    parser.add_argument(
        "--runtime-health-command",
        default=os.environ.get("ATREX_SANDBOX_RUNTIME_HEALTH_COMMAND", ""),
        metavar="COMMAND",
        help="Additional trusted evaluator/framework probe; replayed by recovery.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Check SSH health before optimization and record failures for recovery.",
    )
    parser.add_argument(
        "--check-health",
        action="store_true",
        help="Run only the configured SSH health probe; do not upload a workspace.",
    )
    parser.add_argument(
        "--workspace", default=".", help="Local workspace to upload (default: cwd)."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("ATREX_SANDBOX_TIMEOUT", str(DEFAULT_COMMAND_TIMEOUT))),
        help=(
            "Remote command execution timeout in seconds, 1..600 "
            "(default: 600; queue wait is budgeted separately)."
        ),
    )
    parser.add_argument(
        "--shape-batch-size",
        type=int,
        default=int(
            os.environ.get("ATREX_EVAL_SHAPE_BATCH_SIZE", str(DEFAULT_EVAL_SHAPE_BATCH_SIZE))
        ),
        help="Maximum Atrex-Bench shapes per concurrent eval job (default: 4).",
    )
    parser.add_argument(
        "--mode",
        dest="evaluation_mode",
        choices=("full", "correctness_only"),
        default=None,
        help=(
            "Typed --kind run mode. correctness_only skips performance measurement; "
            "the default preserves the standard evaluator workflow."
        ),
    )
    parser.add_argument("--version", default=None, help="Evaluation version label, e.g. v3.")
    parser.add_argument(
        "--multi-seed", type=int, default=None,
        help="Additional correctness seeds for --kind run.",
    )
    parser.add_argument(
        "--shape-id", action="append", default=[],
        help="Opaque evaluation Shape id (repeatable; default: all).",
    )
    parser.add_argument(
        "--timed-runs", type=int, default=None,
        help="Native evaluator timing iterations for --kind run.",
    )
    parser.add_argument(
        "--baseline-path",
        default=None,
        metavar="PATH",
        help=(
            "Workspace-relative baseline Kernel for an Agent-requested same-allocation "
            "ABBA comparison. Valid only with --kind run --mode full."
        ),
    )
    parser.add_argument(
        "--comparison-repeats",
        type=int,
        default=2,
        metavar="N",
        help="A/B pair count for --baseline-path (default: 2; schedule alternates AB/BA).",
    )
    parser.add_argument(
        "--input-path",
        dest="evaluation_input_path",
        default=None,
        metavar="PATH",
        help=(
            "Workspace-relative custom _make_inputs Python source for --kind run (at most 128 KiB)."
        ),
    )
    parser.add_argument(
        "--shapes-path",
        dest="evaluation_shapes_path",
        default=None,
        metavar="PATH",
        help=("Workspace-relative custom Shape JSON object for --kind run (at most 256 KiB)."),
    )
    parser.add_argument(
        "--sync",
        action="append",
        default=[],
        metavar="PATH",
        help="Relative result path to copy back (repeatable; default: no download).",
    )
    parser.add_argument("--no-sync", action="store_true", help="Do not copy any files back.")
    parser.add_argument(
        "--inline-output",
        action="store_true",
        help=(
            "Return synchronized files through the legacy stdout archive instead of "
            "agate OSS (default: use OSS with standard agate gateway profiles; "
            "custom gateway URLs and config use inline output)."
        ),
    )
    parser.add_argument(
        "--include-raw-profile",
        action="store_true",
        help=(
            "Include raw .ncu-rep/ATT artifacts with legacy inline output "
            "(OSS output includes them by default)."
        ),
    )
    parser.add_argument(
        "--profile-level",
        choices=("survey", "sol", "deep"),
        default="sol",
        help="Typed profile funnel level (default: sol).",
    )
    parser.add_argument(
        "--profiler",
        choices=("ncu", "rocprofv3"),
        default=None,
        help="Typed profile backend (default: gateway vendor auto-detection).",
    )
    parser.add_argument(
        "--profile-counter",
        action="append",
        default=[],
        metavar="METRIC",
        help="Typed profile metric/counter (repeatable).",
    )
    parser.add_argument(
        "--kernel-regex",
        default=None,
        help="Typed profile kernel regex (mutually exclusive with --kernel-name).",
    )
    parser.add_argument(
        "--kernel-name",
        default=None,
        help="Typed profile exact kernel name (mutually exclusive with --kernel-regex).",
    )
    parser.add_argument(
        "--profile-source",
        action="store_true",
        help="Request source-correlated profiler output.",
    )
    parser.add_argument(
        "--launch-skip",
        type=int,
        default=None,
        help="Profiler launches to skip before collection (default: Gateway policy).",
    )
    parser.add_argument(
        "--launch-count",
        type=int,
        default=None,
        help="Profiler launches to collect (default: Gateway policy).",
    )
    parser.add_argument(
        "--profile-shape-id",
        default=None,
        metavar="ID",
        help="Opaque evaluator Shape id to use for this profile (default: first Shape).",
    )
    parser.add_argument(
        "--top-kernels",
        type=int,
        default=None,
        help="Limit the typed profile result to the N hottest kernels.",
    )
    parser.add_argument(
        "--arch",
        default=None,
        help="Optional exact GPU architecture for --kind check, for example sm_90.",
    )
    parser.add_argument(
        "--sanitize",
        choices=("memcheck", "racecheck", "initcheck", "synccheck"),
        default=None,
        help="Run the selected compute-sanitizer mode with --kind check.",
    )
    parser.add_argument(
        "--requirement",
        action="append",
        default=[],
        metavar="SPEC",
        help=(
            "Additional PEP 508 dependency for a typed profile, check, or disassemble job "
            "(repeatable; installed only in that Gateway job)."
        ),
    )
    parser.add_argument(
        "--deps-mode",
        choices=("freeze_installed", "no_deps"),
        default=None,
        help=(
            "Dependency policy for --requirement: freeze_installed reuses satisfied "
            "base packages; no_deps installs only the named requirements."
        ),
    )
    parser.add_argument(
        "--format",
        dest="disassembly_format",
        choices=("sass", "ptx", "isa", "auto"),
        default="auto",
        help="Assembly format for --kind disassemble (default: auto).",
    )
    parser.add_argument(
        "--env-gpu",
        default=None,
        metavar="GPU",
        help="With --kind env, inspect one selectable environment instead of listing all.",
    )
    parser.add_argument(
        "--env-capabilities",
        action="store_true",
        help="With --kind env --env-gpu, return frameworks, profilers, and limits.",
    )
    parser.add_argument(
        "--env-force",
        action="store_true",
        help="With --kind env, request a fresh Gateway environment probe.",
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Additional required workspace file or directory for a non-evaluator command "
            "(repeatable). Arbitrary commands otherwise receive only paths referenced by "
            "their argv; the full workspace is never uploaded implicitly."
        ),
    )
    parser.add_argument(
        "--max-input-file-mb",
        type=int,
        default=16,
        help="Skip individual workspace input files larger than this (default: 16 MiB).",
    )
    parser.add_argument(
        "--max-output-file-mb",
        type=int,
        default=512,
        help="Skip individual returned artifacts larger than this (default: 512 MiB).",
    )
    parser.add_argument("-e", "--env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument(
        "--keep-pod",
        action="store_true",
        help="Ask the gateway not to recycle the pod; filesystem persistence is still not assumed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Package and print the request summary only.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command after --.")
    return parser


class SSHTransportError(RuntimeError):
    """A failure to connect to or transfer data through OpenSSH."""


def _validate_ssh_target(target: str) -> str:
    value = target.strip()
    if not value or value.startswith("-"):
        raise ValueError("SSH target must be a non-option host or [user@]host")
    if any(character.isspace() or ord(character) < 32 for character in value):
        raise ValueError("SSH target must not contain whitespace or control characters")
    return value


def _environment_ssh_runtime_binds() -> list[str]:
    raw = os.environ.get(SSH_RUNTIME_BINDS_ENV, "").strip()
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{SSH_RUNTIME_BINDS_ENV} must be a JSON array") from exc
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{SSH_RUNTIME_BINDS_ENV} must be a JSON array of strings")
    return value


def _ssh_runtime_bind(value: str) -> tuple[str, str]:
    source, separator, destination = value.partition("=")
    if not separator:
        destination = source
    paths = []
    for label, raw in (("source", source), ("destination", destination)):
        path = PurePosixPath(raw)
        if not raw or not path.is_absolute() or ".." in path.parts:
            raise ValueError(
                f"SSH runtime bind {label} must be an absolute path without '..': {raw!r}"
            )
        paths.append(path.as_posix())
    forbidden = {
        "/",
        "/atrex",
        "/bin",
        "/dev",
        "/etc",
        "/lib",
        "/lib64",
        "/proc",
        "/sbin",
        "/sys",
        "/tmp",
        "/usr",
        "/home",
        "/root",
    }
    reserved_trees = {
        "atrex",
        "bin",
        "dev",
        "etc",
        "lib",
        "lib64",
        "proc",
        "sbin",
        "sys",
        "tmp",
        "usr",
    }
    source_path = PurePosixPath(paths[0])
    source_parts = source_path.parts
    source_forbidden_trees = {
        "/dev",
        "/etc",
        "/proc",
        "/root",
        "/sys",
        "/var/lib",
        "/var/log",
        "/var/run",
    }
    broad_source_roots = {
        "/",
        "/atrex",
        "/bin",
        "/home",
        "/lib",
        "/lib64",
        "/opt",
        "/sbin",
        "/srv",
        "/tmp",
        "/usr",
        "/var",
    }
    sensitive_components = {".aws", ".config", ".docker", ".gnupg", ".kube", ".ssh"}
    if paths[0] in broad_source_roots or any(
        paths[0] == root or paths[0].startswith(root + "/") for root in source_forbidden_trees
    ):
        raise ValueError(f"SSH runtime bind source is sensitive or too broad: {paths[0]!r}")
    if any(part in sensitive_components for part in source_parts):
        raise ValueError(f"SSH runtime bind source contains a sensitive directory: {paths[0]!r}")
    # Home-directory binds are limited to conventional virtual-environment roots;
    # arbitrary project/home subtrees are not runtime allowlists.
    if (
        len(source_parts) > 1
        and source_parts[1] == "home"
        and source_path.name not in {".venv", "venv"}
        and source_path.parent.name != "envs"
    ):
        raise ValueError(
            "SSH runtime bind source below /home must be a .venv/venv or a direct "
            f"Conda envs child: {paths[0]!r}"
        )

    destination_parts = PurePosixPath(paths[1]).parts
    if paths[1] in forbidden or (
        len(destination_parts) > 1 and destination_parts[1] in reserved_trees
    ):
        raise ValueError(f"SSH runtime bind destination is reserved: {paths[1]!r}")
    return paths[0], paths[1]


SSH_RUNTIME_RESOLVER_SOURCE = r"""
import json
import os
import sys

resolved = []
for source in sys.argv[1:]:
    real = os.path.realpath(source)
    if not os.path.isdir(real):
        print(f"runtime bind is not a directory: {source}", file=sys.stderr)
        raise SystemExit(2)
    resolved.append(real)
print(json.dumps(resolved, separators=(",", ":")))
""".strip()


def _resolve_ssh_runtime_binds(ssh: str, target: str, runtime_binds: list[str]) -> list[str]:
    """Resolve remote symlinks and re-apply the source denylist to their targets."""
    parsed = [_ssh_runtime_bind(value) for value in runtime_binds]
    if not parsed:
        return []
    try:
        result = subprocess.run(
            [
                *_ssh_base(ssh, target),
                shlex.join(
                    [
                        "python3",
                        "-c",
                        SSH_RUNTIME_RESOLVER_SOURCE,
                        *[source for source, _destination in parsed],
                    ]
                ),
            ],
            capture_output=True,
            text=True,
            timeout=SSH_CONNECT_TIMEOUT + 15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SSHTransportError(f"cannot resolve SSH runtime binds: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-1000:]
        raise SSHTransportError(f"cannot resolve SSH runtime binds: {detail}")
    try:
        payload = result.stdout.strip().splitlines()[-1]
        resolved = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SSHTransportError("SSH runtime bind resolver returned invalid JSON") from exc
    except IndexError as exc:
        raise SSHTransportError("SSH runtime bind resolver returned no paths") from exc
    if (
        not isinstance(resolved, list)
        or len(resolved) != len(parsed)
        or not all(isinstance(item, str) for item in resolved)
    ):
        raise SSHTransportError("SSH runtime bind resolver returned an invalid path list")
    validated: list[str] = []
    for source, (_declared_source, destination) in zip(resolved, parsed):
        try:
            validated_source, validated_destination = _ssh_runtime_bind(f"{source}={destination}")
        except ValueError as exc:
            raise SSHTransportError(
                f"resolved SSH runtime bind is unsafe: {source!r}: {exc}"
            ) from exc
        validated.append(f"{validated_source}={validated_destination}")
    return validated


SSH_GPU_RESOLVER_SOURCE = r"""
import subprocess
import sys

index = int(sys.argv[1])
result = subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=index,uuid,mig.mode.current",
        "--format=csv,noheader,nounits",
    ],
    capture_output=True,
    text=True,
)
if result.returncode:
    sys.stderr.write(result.stderr or result.stdout)
    raise SystemExit(result.returncode)
for line in result.stdout.splitlines():
    fields = [field.strip() for field in line.split(",", 2)]
    if len(fields) != 3 or fields[0] != str(index):
        continue
    if fields[2].lower() == "enabled":
        print("MIG-enabled GPUs are unsupported without capability-node assignment", file=sys.stderr)
        raise SystemExit(2)
    if fields[1].startswith("GPU-"):
        print(fields[1])
        raise SystemExit(0)
print(f"physical NVIDIA GPU index {index} was not found", file=sys.stderr)
raise SystemExit(2)
""".strip()


def _ssh_gpu_index(value: str) -> int:
    if not re.fullmatch(r"[0-9]+", value.strip()):
        raise ValueError(
            "SSH GPU must be a physical NVIDIA index; MIG/UUID selectors are not yet supported"
        )
    index = int(value)
    if index > 31:
        raise ValueError("SSH GPU index must be in the range 0..31")
    return index


def _resolve_ssh_gpu(ssh: str, target: str, gpu_index: int) -> str:
    """Resolve an assigned physical index to a stable CUDA visibility UUID."""
    try:
        result = subprocess.run(
            [
                *_ssh_base(ssh, target),
                shlex.join(["python3", "-c", SSH_GPU_RESOLVER_SOURCE, str(gpu_index)]),
            ],
            capture_output=True,
            text=True,
            timeout=SSH_CONNECT_TIMEOUT + 15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SSHTransportError(f"cannot resolve assigned SSH GPU: {exc}") from exc
    uuid = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    if result.returncode != 0 or not re.fullmatch(r"GPU-[0-9A-Fa-f-]+", uuid):
        detail = (result.stderr or result.stdout).strip()[-1000:]
        raise SSHTransportError(f"cannot resolve assigned SSH GPU: {detail}")
    return uuid


def _ssh_bwrap_command(
    runtime_binds: list[str],
    command: list[str],
    *,
    gpu_index: int,
    gpu_uuid: str,
    remote_dir: str | None = None,
) -> str:
    """Build the only remote execution path: a restricted Bubblewrap namespace."""
    binds = [_ssh_runtime_bind(value) for value in runtime_binds]
    args = [
        "bwrap",
        "--die-with-parent",
        "--new-session",
        "--unshare-net",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--ro-bind",
        "/usr",
        "/usr",
        "--ro-bind-try",
        "/bin",
        "/bin",
        "--ro-bind-try",
        "/sbin",
        "/sbin",
        "--ro-bind-try",
        "/lib",
        "/lib",
        "--ro-bind-try",
        "/lib64",
        "/lib64",
        "--ro-bind",
        "/etc",
        "/etc",
        "--ro-bind-try",
        "/sys",
        "/sys",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/dev/shm",
        "--tmpfs",
        "/tmp",
        "--dir",
        "/tmp/atrex-home",
    ]
    for device in (
        "/dev/nvidiactl",
        "/dev/nvidia-uvm",
        "/dev/nvidia-uvm-tools",
        "/dev/nvidia-modeset",
    ):
        args.extend(("--dev-bind-try", device, device))
    assigned_device = f"/dev/nvidia{gpu_index}"
    args.extend(("--dev-bind", assigned_device, assigned_device))

    created: set[str] = set()
    for source, destination in binds:
        parent = PurePosixPath(destination).parent
        parents = list(reversed(parent.parents)) + [parent]
        for directory in parents:
            rendered = directory.as_posix()
            if rendered == "/" or rendered in created:
                continue
            args.extend(("--dir", rendered))
            created.add(rendered)
        args.extend(("--ro-bind", source, destination))
    if remote_dir is not None:
        if not re.fullmatch(r"/tmp/atrex-sandbox\.[A-Za-z0-9._-]+", remote_dir):
            raise ValueError("unsafe SSH workspace path")
        args.extend(("--dir", "/atrex", "--bind", remote_dir, "/atrex", "--chdir", "/atrex"))
    args.extend(
        (
            "env",
            "-i",
            "HOME=/tmp/atrex-home",
            "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "LANG=C.UTF-8",
            f"CUDA_VISIBLE_DEVICES={gpu_uuid}",
            *command,
        )
    )
    return shlex.join(args)


def _ssh_base(executable: str, target: str) -> list[str]:
    return [
        executable,
        "-o",
        f"ConnectTimeout={SSH_CONNECT_TIMEOUT}",
        target,
    ]


def _ssh_shell_script(init_command: str, command: str) -> str:
    lines = ["set -eo pipefail"]
    if init_command.strip():
        lines.append(init_command)
    lines.append(command)
    return "\n".join(lines) + "\n"


def _run_ssh_health(
    target: str,
    init_command: str,
    health_command: str,
    runtime_binds: list[str],
    gpu_index: int,
    *,
    timeout: int = 60,
) -> subprocess.CompletedProcess[str]:
    ssh = shutil.which("ssh")
    if ssh is None:
        raise SSHTransportError("ssh executable not found on PATH")
    try:
        resolved_binds = _resolve_ssh_runtime_binds(ssh, target, runtime_binds)
        gpu_uuid = _resolve_ssh_gpu(ssh, target, gpu_index)
        script = _ssh_shell_script(init_command, health_command)
        remote_command = _ssh_bwrap_command(
            resolved_binds,
            [
                "python3",
                "-c",
                SSH_WATCHDOG_SOURCE,
                str(timeout),
                "bash",
                "-lc",
                script,
            ],
            gpu_index=gpu_index,
            gpu_uuid=gpu_uuid,
        )
    except ValueError as exc:
        raise SSHTransportError(str(exc)) from exc
    try:
        return subprocess.run(
            [*_ssh_base(ssh, target), remote_command],
            capture_output=True,
            text=True,
            timeout=timeout + SSH_CONNECT_TIMEOUT + 10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SSHTransportError(f"SSH health probe failed: {exc}") from exc


def _remote_temp_dir(stdout: str) -> str:
    value = stdout.strip().splitlines()[-1] if stdout.strip() else ""
    if not re.fullmatch(r"/tmp/atrex-sandbox\.[A-Za-z0-9._-]+", value):
        raise SSHTransportError("remote mktemp returned an unsafe directory")
    return value


def _best_effort_ssh_cleanup(ssh: str, target: str, remote_dir: str) -> bool:
    if not re.fullmatch(r"/tmp/atrex-sandbox\.[A-Za-z0-9._-]+", remote_dir):
        return False
    try:
        result = subprocess.run(
            [
                *_ssh_base(ssh, target),
                "rm -rf -- " + shlex.quote(remote_dir),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _run_ssh_job(
    *,
    target: str,
    init_command: str,
    runtime_binds: list[str],
    gpu_index: int,
    timeout: int,
    env_items: list[str],
    upload_paths: list[Path],
    temp: Path,
    workspace: Path,
    sync_outputs: bool,
) -> subprocess.CompletedProcess[str]:
    """Upload one stateless sandbox allocation, execute it, and retrieve outputs."""
    ssh = shutil.which("ssh")
    scp = shutil.which("scp")
    if ssh is None or scp is None:
        missing = "ssh" if ssh is None else "scp"
        raise SSHTransportError(f"{missing} executable not found on PATH")

    resolved_binds = _resolve_ssh_runtime_binds(ssh, target, runtime_binds)
    gpu_uuid = _resolve_ssh_gpu(ssh, target, gpu_index)

    try:
        environment = _parse_env_items(env_items)
    except ValueError as exc:
        raise SystemExit(f"sandbox: {exc}") from exc
    entry_lines = ["#!/usr/bin/env bash", "set -eo pipefail"]
    if init_command.strip():
        entry_lines.append(init_command)
    entry_lines.append('cd "$1"')
    for key, value in environment.items():
        entry_lines.append(f"export {key}={shlex.quote(value)}")
    entry_lines.append(
        "exec "
        + shlex.join(
            [
                "python3",
                "-c",
                SSH_WATCHDOG_SOURCE,
                str(timeout),
                "bash",
                "__atrex_runner.sh",
            ]
        )
    )
    entry_path = temp / "ssh_entry.sh"
    entry_path.write_text("\n".join(entry_lines) + "\n", encoding="utf-8")

    result: subprocess.CompletedProcess[str] | None = None
    cleanup_succeeded = False
    try:
        create = subprocess.run(
            [
                *_ssh_base(ssh, target),
                "mktemp -d /tmp/atrex-sandbox.XXXXXXXXXX",
            ],
            capture_output=True,
            text=True,
            timeout=SSH_CONNECT_TIMEOUT + 10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SSHTransportError(f"cannot create remote workspace: {exc}") from exc
    if create.returncode != 0:
        detail = (create.stderr or create.stdout).strip()[-1000:]
        raise SSHTransportError(f"cannot create remote workspace: {detail}")
    remote_dir = _remote_temp_dir(create.stdout)
    try:
        transfer = subprocess.run(
            [
                scp,
                "-q",
                *[str(path) for path in upload_paths],
                str(entry_path),
                f"{target}:{remote_dir}/",
            ],
            capture_output=True,
            text=True,
            timeout=max(60, timeout),
        )
        if transfer.returncode != 0:
            detail = (transfer.stderr or transfer.stdout).strip()[-1000:]
            raise SSHTransportError(f"cannot upload sandbox inputs: {detail}")

        try:
            remote_command = _ssh_bwrap_command(
                resolved_binds,
                ["bash", "/atrex/ssh_entry.sh", "/atrex"],
                gpu_index=gpu_index,
                gpu_uuid=gpu_uuid,
                remote_dir=remote_dir,
            )
        except ValueError as exc:
            raise SSHTransportError(str(exc)) from exc
        try:
            result = subprocess.run(
                [*_ssh_base(ssh, target), remote_command],
                capture_output=True,
                text=True,
                timeout=timeout + SSH_CONNECT_TIMEOUT + 15,
            )
        except subprocess.TimeoutExpired as exc:
            raise SSHTransportError(f"SSH command wait timed out: {exc}") from exc

        if sync_outputs:
            local_archive = temp / "ssh_outputs.tar.gz"
            download = subprocess.run(
                [
                    scp,
                    "-q",
                    f"{target}:{remote_dir}/{OSS_OUTPUT_ARCHIVE}",
                    str(local_archive),
                ],
                capture_output=True,
                text=True,
                timeout=max(60, timeout),
            )
            if download.returncode != 0:
                detail = (download.stderr or download.stdout).strip()[-1000:]
                execution_detail = (result.stderr or result.stdout).strip()[-1000:]
                if result.returncode == 0:
                    raise SSHTransportError(
                        "cannot download sandbox outputs: "
                        f"{detail}; remote_exit={result.returncode}; "
                        f"remote_output={execution_detail}"
                    )
                warning = f"[sandbox] output archive unavailable after failed command: {detail}"
                result = subprocess.CompletedProcess(
                    args=result.args,
                    returncode=result.returncode,
                    stdout=result.stdout,
                    stderr="\n".join(part for part in (result.stderr, warning) if part),
                )
            else:
                try:
                    _extract_output_archive(local_archive, workspace)
                except (OSError, RuntimeError, tarfile.TarError) as exc:
                    raise SSHTransportError(f"cannot extract sandbox outputs: {exc}") from exc
    except (OSError, subprocess.TimeoutExpired) as exc:
        # scp can time out during either upload or download. Both are transport
        # failures, even if the remote candidate itself has already exited.
        raise SSHTransportError(f"SSH transfer failed: {exc}") from exc
    finally:
        cleanup_succeeded = _best_effort_ssh_cleanup(ssh, target, remote_dir)
        if not cleanup_succeeded:
            _record_pending_ssh_cleanup(target=target, remote_dir=remote_dir)
    if not cleanup_succeeded:
        raise SSHTransportError(
            f"remote workspace cleanup failed and was queued for recovery: {remote_dir}"
        )
    if result is None:
        raise SSHTransportError("SSH job produced no result")
    return result


def _environment_failure_path() -> Path | None:
    value = os.environ.get("ATREX_ENVIRONMENT_STATE_FILE", "").strip()
    return Path(value).expanduser().resolve() if value else None


def _record_pending_ssh_cleanup(*, target: str, remote_dir: str) -> None:
    state_file = _environment_failure_path()
    if state_file is None:
        print(
            f"[sandbox] WARNING: remote workspace cleanup failed: {target}:{remote_dir}",
            file=sys.stderr,
        )
        return
    digest = hashlib.sha256(f"{target}\0{remote_dir}".encode("utf-8")).hexdigest()[:16]
    path = state_file.parent / f"cleanup-{digest}.json"
    payload = {
        "schema_version": 1,
        "transport": "ssh",
        "target": target,
        "remote_dir": remote_dir,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    durable_write_json(path, payload, indent=2, ensure_ascii=False)


def _record_environment_failure(
    *, target: str, stage: str, detail: str, health_status: int | None = None
) -> None:
    path = _environment_failure_path()
    if path is None:
        return
    payload = {
        "schema_version": 1,
        "status": "blocked",
        "transport": "ssh",
        "target": target,
        "stage": stage,
        "detail": " ".join(detail.split())[-2000:],
        "health_status": health_status,
        "detected_at": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
    }
    durable_write_json(path, payload, indent=2, ensure_ascii=False)


def _auth_headers() -> dict[str, str]:
    """Generate token or AK/SK headers matching agate's auth precedence."""
    import hashlib

    config: dict[str, Any] = {}
    try:
        value = json.loads((Path.home() / ".atrex" / "config.json").read_text(encoding="utf-8"))
        if isinstance(value, dict):
            config = value
    except (OSError, json.JSONDecodeError):
        pass
    private_token = os.environ.get("AGATE_TOKEN", "") or str(config.get("token") or "")
    if private_token:
        return {"Authorization": f"Bearer {private_token}"}
    ak = os.environ.get("AGATE_AK", "") or str(config.get("ak") or "")
    sk = os.environ.get("AGATE_SK", "") or str(config.get("sk") or "")
    if not ak or not sk:
        return {}
    ts = str(int(time.time() * 1000))
    token = hashlib.md5(f"{ak}::{sk}::{ts}".encode()).hexdigest()
    return {"Access-Key": ak, "Timestamp": ts, "Token": token}


class GatewayHTTPError(RuntimeError):
    def __init__(self, status: int, detail: str) -> None:
        self.status = status
        self.detail = detail
        super().__init__(f"gateway HTTP {status}: {detail}")


def _gateway_json(
    base_url: str,
    method: str,
    path: str,
    payload: dict | None,
    timeout: float,
) -> dict:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = dict(_auth_headers())
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        method=method,
        data=body,
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise GatewayHTTPError(exc.code, detail) from exc
    if not isinstance(result, dict):
        raise RuntimeError("gateway returned a non-object JSON response")
    return result


def _run_direct_job(
    *,
    url: str,
    kind: str,
    payload: dict[str, Any],
    timeout: int,
    queue_wait_grace: int,
) -> subprocess.CompletedProcess[str]:
    """Submit and wait for a public job, resubmitting infrastructure failures."""
    deadline = time.monotonic() + timeout + queue_wait_grace
    notes: list[str] = []
    retries = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 1:
            raise TimeoutError("gateway retry budget exhausted before submission")
        try:
            accepted = _gateway_json(url, "POST", f"/v1/jobs/{kind}", payload, 30)
        except GatewayHTTPError as exc:
            if not _retryable_gateway_http_error(exc) or retries >= DEFAULT_INFRASTRUCTURE_RETRIES:
                raise
            retries += 1
            delay = _retry_delay(retries, deadline - time.monotonic())
            notes.append(
                f"[sandbox] Gateway HTTP {exc.status}; retrying submission "
                f"({retries}/{DEFAULT_INFRASTRUCTURE_RETRIES})"
            )
            if delay:
                time.sleep(delay)
            continue
        job_id = accepted.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise RuntimeError(f"gateway submission returned no job_id: {accepted}")
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"gateway job {job_id} exceeded client timeout")
                wait_for = min(30.0, remaining)
                try:
                    job = _gateway_json(
                        url,
                        "GET",
                        f"/v1/jobs/{job_id}?wait=true&timeout={wait_for:.3f}",
                        None,
                        wait_for + 10,
                    )
                except GatewayHTTPError as exc:
                    if not _retryable_gateway_http_error(exc):
                        raise
                    delay = _retry_delay(1, deadline - time.monotonic())
                    notes.append(
                        f"[sandbox] Gateway poll HTTP {exc.status}; continuing the same job"
                    )
                    if delay:
                        time.sleep(delay)
                    continue
                if job.get("status") in ("succeeded", "failed", "cancelled"):
                    if (
                        _infrastructure_failure(job)
                        and retries < DEFAULT_INFRASTRUCTURE_RETRIES
                        and deadline - time.monotonic() > 1
                    ):
                        retries += 1
                        delay = _retry_delay(retries, deadline - time.monotonic())
                        notes.append(
                            "[sandbox] Gateway infrastructure failure "
                            f"({_infrastructure_reason(job)}); resubmitting a fresh job "
                            f"({retries}/{DEFAULT_INFRASTRUCTURE_RETRIES})"
                        )
                        if delay:
                            time.sleep(delay)
                        break
                    return subprocess.CompletedProcess(
                        args=["direct-gateway", kind, job_id],
                        returncode=0 if job.get("status") == "succeeded" else 1,
                        stdout=json.dumps(job),
                        stderr="\n".join(notes),
                    )
        except BaseException:
            try:
                _gateway_json(url, "POST", f"/v1/jobs/{job_id}/cancel", {}, 10)
            except Exception:
                pass
            raise


def _run_environment_query(args: argparse.Namespace) -> int:
    """Expose the read-only Agate environment contract through the Supervisor."""
    if args.ssh:
        raise SystemExit("sandbox: --kind env requires an Agate Gateway endpoint")
    executable = _find_agate()
    url = _resolved_gateway_url(
        executable,
        url=args.url,
        profile=args.gateway_profile,
    )
    if url:
        path = "/v1/env"
        if args.env_gpu:
            encoded = urllib.parse.quote(args.env_gpu, safe="")
            path = f"/v1/env/{encoded}"
            if args.env_capabilities:
                path += "/capabilities"
        if args.env_force:
            path += "?force=true"
        try:
            value = _gateway_json(url, "GET", path, None, 120 if args.env_force else 30)
        except GatewayHTTPError as exc:
            raise SystemExit(f"sandbox: env gateway request failed: {exc}") from exc
    else:
        if executable is None:
            raise SystemExit("sandbox: agate not found and no explicit --url was provided")
        command = [executable, "env"]
        if args.gateway_profile:
            command += ["--profile", args.gateway_profile]
        if args.env_gpu:
            command.append(args.env_gpu)
        if args.env_capabilities:
            command += ["--frameworks", "--limits"]
        if args.env_force:
            command.append("--force")
        command.append("--json")
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=180 if args.env_force else 60,
            check=False,
        )
        if completed.returncode:
            raise SystemExit(
                "sandbox: env query failed: "
                + _bounded_text(completed.stderr or completed.stdout, 2000)
            )
        try:
            value = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise SystemExit("sandbox: env query returned invalid JSON") from exc
    print(ENV_RESULT_PREFIX + json.dumps(value, ensure_ascii=False))
    return 0


def _run_direct_gateway(
    *,
    url: str,
    hardware: str,
    timeout: int,
    queue_wait_grace: int,
    env_items: list[str],
    files: dict[str, Path],
    command: str,
    num_gpus: int = 1,
) -> subprocess.CompletedProcess[str]:
    """Use the public dev-job HTTP API when the optional agate CLI is absent."""
    try:
        env_vars = _parse_env_items(env_items)
    except ValueError as exc:
        raise SystemExit(f"sandbox: {exc}") from exc
    spec: dict[str, Any] = {"target_hardware": [hardware]}
    if num_gpus > 1:
        spec["num_gpus"] = num_gpus
    return _run_direct_job(
        url=url,
        kind="dev",
        timeout=timeout,
        queue_wait_grace=queue_wait_grace,
        payload={
            "spec": spec,
            "command": command,
            "timeout_s": timeout,
            "env_vars": env_vars,
            "files": {name: path.read_text(encoding="utf-8") for name, path in files.items()},
        },
    )


def _job_response(stdout: str) -> dict | None:
    """Return an agate job response when stdout is complete JSON."""
    try:
        result = json.loads(stdout)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(result, dict) or not isinstance(result.get("job_id"), str):
        return None
    return result


def _cancelled_without_outcome(job: dict | None) -> bool:
    """Return whether a job was cancelled before producing any outcome.

    The production gateway can occasionally cancel a queued job before an
    attempt starts.  Such a response has no command result and no gateway
    error, so it says nothing about the submitted kernel.  A cancellation
    carrying either field is a real terminal outcome and must not be retried.
    """
    return bool(
        job and job.get("status") == "cancelled" and not job.get("result") and not job.get("error")
    )


def _ray_submit_version_mismatch(job: dict | None) -> bool:
    error = job.get("error") if job else None
    return bool(
        isinstance(error, dict)
        and error.get("reason") == "submit_failed"
        and "Version check returned 404" in str(error.get("message", ""))
    )


def _queue_timeout_before_start(job: dict | None) -> bool:
    error = job.get("error") if job else None
    return bool(
        isinstance(error, dict)
        and error.get("reason") == "timeout"
        and "never started executing" in str(error.get("message", ""))
    )


def _infrastructure_failure(job: dict | None) -> bool:
    """Return whether a terminal response is unrelated to candidate quality.

    Infrastructure failures must be resubmitted as new jobs.  Polling the same
    terminal job cannot recover missing logs, a failed Ray runtime environment,
    or another backend-originated failure.
    """
    if not job or job.get("status") not in {"failed", "cancelled"}:
        return False
    error = job.get("error")
    if not isinstance(error, dict):
        return _cancelled_without_outcome(job)
    details = error.get("details")
    details = details if isinstance(details, dict) else {}
    return bool(
        error.get("error_class") == "infra"
        or details.get("failure_origin") == "infrastructure"
        or _ray_submit_version_mismatch(job)
        or _queue_timeout_before_start(job)
    )


def _infrastructure_reason(job: dict | None) -> str:
    error = job.get("error") if job else None
    if isinstance(error, dict):
        return str(error.get("reason") or error.get("error_class") or "infrastructure")
    return "cancelled_without_outcome"


def _agate_transport_failure(completed: subprocess.CompletedProcess[str], job: dict | None) -> bool:
    if _infrastructure_failure(job):
        return True
    if job is not None or completed.returncode == 0:
        return False
    detail = ((completed.stderr or "") + "\n" + (completed.stdout or "")).casefold()
    markers = (
        "http 408",
        "http 425",
        "http 429",
        "http 500",
        "http 502",
        "http 503",
        "http 504",
        "connection reset",
        "connection refused",
        "temporarily unavailable",
        "timed out",
        "timeout",
    )
    return any(marker in detail for marker in markers)


def _agate_failure_reason(completed: subprocess.CompletedProcess[str], job: dict | None) -> str:
    if job is not None:
        return _infrastructure_reason(job)
    detail = _bounded_text(completed.stderr or completed.stdout or "transport failure", 200)
    return " ".join(detail.split())


def _retry_delay(retry_number: int, remaining: float) -> float:
    requested = min(
        60,
        DEFAULT_INFRASTRUCTURE_RETRY_SECONDS * (2 ** max(0, retry_number - 1)),
    )
    return max(0.0, min(float(requested), remaining - 1.0))


def _retryable_gateway_http_error(error: GatewayHTTPError) -> bool:
    return error.status in {408, 425, 429, 500, 502, 503, 504}


def _l20n_failover_command(agate: list[str]) -> list[str] | None:
    try:
        gpu_index = agate.index("--gpu") + 1
    except (ValueError, IndexError):
        return None
    if agate[gpu_index].casefold() != "l20n":
        return None
    fallback = list(agate)
    fallback[gpu_index] = "l20n-ray"
    return fallback


def _submitted_job_id(proc: subprocess.CompletedProcess[str]) -> str | None:
    """Recover the job id printed by agate before it starts polling."""
    match = SUBMITTED_JOB_RE.search((proc.stderr or "") + "\n" + (proc.stdout or ""))
    return match.group(1) if match else None


def _track_agate_job(job_id: str, executable: str, url: str, gateway_profile: str | None) -> None:
    with ACTIVE_AGATE_JOBS_LOCK:
        ACTIVE_AGATE_JOBS[job_id] = (executable, url, gateway_profile)


def _forget_agate_job(job_id: str) -> None:
    with ACTIVE_AGATE_JOBS_LOCK:
        ACTIVE_AGATE_JOBS.pop(job_id, None)


def _cancel_active_agate_jobs() -> None:
    with ACTIVE_AGATE_JOBS_LOCK:
        jobs = list(ACTIVE_AGATE_JOBS.items())
    for job_id, (executable, url, gateway_profile) in jobs:
        command = [executable, "cancel"]
        if url:
            command += ["--url", url]
        elif gateway_profile:
            command += ["--profile", gateway_profile]
        command += ["--http-timeout", "10", job_id]
        try:
            subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass


def _interrupt_active_agate_jobs(_signum: int, _frame: object) -> None:
    _cancel_active_agate_jobs()
    raise KeyboardInterrupt


def _gateway_job_timeout(command_timeout: int, queue_wait_grace: int) -> int:
    """Budget typed gateway queueing separately from evaluator runtime.

    Typed eval/profile jobs accept a larger enclosing deadline than their evaluator
    timeout.  Give that job deadline as much of the configured queue grace as the
    service permits.
    """
    return min(MAX_GATEWAY_JOB_TIMEOUT, command_timeout + queue_wait_grace)


def _dev_gateway_job_timeout(command_timeout: int) -> int:
    """Return a service-valid deadline for an agate dev job.

    Unlike typed eval/profile jobs, the dev API currently validates ``timeout_s``
    against a hard 600-second ceiling.  Passing the longer client-side queue wait
    budget through ``--job-timeout`` is rejected at submission time with HTTP 422,
    so keep queue grace exclusively in ``--wait-timeout`` for this route.
    """
    return min(MAX_DEV_JOB_TIMEOUT, command_timeout)


def _resume_interrupted_agate_wait(
    *,
    executable: str,
    url: str,
    gateway_profile: str | None,
    command_timeout: int,
    wait_budget: int,
    elapsed: float,
    initial: subprocess.CompletedProcess[str],
) -> subprocess.CompletedProcess[str]:
    """Wait for an already-submitted job without resubmitting it.

    Submission is deliberately non-blocking so the sandbox knows the job id and
    can cancel it if its own parent terminates the sandbox while it is polling.
    """
    initial_job = _job_response(initial.stdout or "")
    if initial_job and initial_job.get("status") in {
        "succeeded",
        "failed",
        "cancelled",
    }:
        return initial
    job_id = initial_job.get("job_id") if initial_job else _submitted_job_id(initial)
    remaining = int(wait_budget - elapsed)
    if not job_id or remaining <= 0:
        return initial

    get_command = [executable, "get"]
    if url:
        get_command += ["--url", url]
    elif gateway_profile:
        get_command += ["--profile", gateway_profile]
    note = f"submitted job_id={job_id}; polling..."
    stderr_parts = [part.rstrip() for part in (initial.stderr, note) if part]
    deadline = time.monotonic() + remaining
    resumed = initial
    while (remaining := int(deadline - time.monotonic())) > 0:
        resumed = subprocess.run(
            [
                *get_command,
                "--http-timeout",
                str(MAX_HTTP_REQUEST_TIMEOUT),
                "--wait-timeout",
                str(min(AGATE_WAIT_SLICE_SECONDS, remaining)),
                "--job-timeout",
                str(command_timeout),
                "--wait",
                job_id,
            ],
            capture_output=True,
            text=True,
        )
        if resumed.stderr:
            stderr_parts.append(resumed.stderr.rstrip())
        job = _job_response(resumed.stdout or "")
        if job and job.get("status") in {"succeeded", "failed", "cancelled"}:
            break
        if not job:
            time.sleep(min(2, max(0, deadline - time.monotonic())))
    return subprocess.CompletedProcess(
        args=resumed.args,
        returncode=resumed.returncode,
        stdout=resumed.stdout,
        stderr="\n".join(stderr_parts),
    )


def _run_agate_once(
    *,
    agate: list[str],
    executable: str,
    url: str,
    gateway_profile: str | None,
    command_timeout: int,
    wait_budget: int,
) -> subprocess.CompletedProcess[str]:
    """Submit one agate job, then wait while keeping its id available for cleanup."""
    wait_started = time.monotonic()
    submitted = subprocess.run([*agate, "--no-wait"], capture_output=True, text=True)
    job = _job_response(submitted.stdout or "")
    if submitted.returncode or not job:
        return submitted
    job_id = job["job_id"]
    _track_agate_job(job_id, executable, url, gateway_profile)
    try:
        return _resume_interrupted_agate_wait(
            executable=executable,
            url=url,
            gateway_profile=gateway_profile,
            command_timeout=command_timeout,
            wait_budget=wait_budget,
            elapsed=time.monotonic() - wait_started,
            initial=submitted,
        )
    finally:
        _forget_agate_job(job_id)


def _run_agate_with_cancel_retry(
    *,
    agate: list[str],
    executable: str,
    url: str,
    gateway_profile: str | None,
    command_timeout: int,
    wait_budget: int,
) -> subprocess.CompletedProcess[str]:
    """Retry terminal infrastructure failures by submitting fresh Gateway jobs."""
    deadline = time.monotonic() + wait_budget
    stderr_parts: list[str] = []
    retries = 0
    active_agate = list(agate)
    while True:
        remaining = int(deadline - time.monotonic())
        if remaining <= 1:
            return subprocess.CompletedProcess(
                args=active_agate,
                returncode=1,
                stdout="",
                stderr="\n".join([*stderr_parts, "[sandbox] gateway retry budget exhausted"]),
            )
        completed = _run_agate_once(
            agate=active_agate,
            executable=executable,
            url=url,
            gateway_profile=gateway_profile,
            command_timeout=command_timeout,
            wait_budget=remaining,
        )
        if completed.stderr:
            stderr_parts.append(completed.stderr.rstrip())
        job = _job_response(completed.stdout or "")
        if not _agate_transport_failure(completed, job):
            return subprocess.CompletedProcess(
                args=completed.args,
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr="\n".join(stderr_parts),
            )
        if retries >= DEFAULT_INFRASTRUCTURE_RETRIES:
            return subprocess.CompletedProcess(
                args=completed.args,
                returncode=completed.returncode or 1,
                stdout=completed.stdout,
                stderr="\n".join(stderr_parts),
            )

        if deadline - time.monotonic() <= 1:
            return subprocess.CompletedProcess(
                args=completed.args,
                returncode=completed.returncode or 1,
                stdout=completed.stdout,
                stderr="\n".join(stderr_parts),
            )

        if _ray_submit_version_mismatch(job):
            fallback = _l20n_failover_command(active_agate)
            if fallback is not None:
                active_agate = fallback
        retries += 1
        delay = _retry_delay(retries, deadline - time.monotonic())
        note = (
            "[sandbox] Gateway infrastructure failure "
            f"({_agate_failure_reason(completed, job)}); resubmitting a fresh job "
            f"({retries}/{DEFAULT_INFRASTRUCTURE_RETRIES})"
        )
        if delay:
            note += f" in {delay:g}s"
        stderr_parts.append(note)
        if delay:
            time.sleep(delay)


def _append_typed_dependency_options(
    command: list[str],
    request: dict[str, Any],
    request_sidecar_dir: Path | None,
) -> None:
    """Append Agate dependency switches without exposing private sidecars."""
    requirements = request.get("requirements")
    if isinstance(requirements, list) and requirements:
        if request_sidecar_dir is None:
            raise ValueError("typed dependencies require a private sidecar directory")
        request_sidecar_dir.mkdir(parents=True, exist_ok=True)
        requirements_path = request_sidecar_dir / "requirements.txt"
        requirements_path.write_text(
            "\n".join(map(str, requirements)) + "\n",
            encoding="utf-8",
        )
        command += ["--requirements", str(requirements_path)]
    deps_mode = request.get("deps_mode")
    if isinstance(deps_mode, str):
        command += ["--deps-mode", deps_mode]


def _typed_agate_command(
    executable: str,
    args: argparse.Namespace,
    workspace: Path,
    kind: str,
    request: dict[str, Any],
    queue_wait_grace: int,
    reference_dir: Path | None = None,
    request_sidecar_dir: Path | None = None,
) -> list[str]:
    """Build an Agate CLI invocation for one typed request."""
    candidate_path = workspace / "kernel.py"
    if request_sidecar_dir is not None:
        request_sidecar_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = request_sidecar_dir / "kernel.py"
        candidate_path.write_text(request["candidate"], encoding="utf-8")
    agate_kind = "check" if kind == "check" else kind
    command = [executable, agate_kind]
    if args.url:
        command += ["--url", args.url]
    elif args.gateway_profile:
        command += ["--profile", args.gateway_profile]
    if kind in DIAGNOSTIC_KINDS:
        if request_sidecar_dir is None:
            raise ValueError("typed diagnostics require a private sidecar directory")
        request_sidecar_dir.mkdir(parents=True, exist_ok=True)
        command += ["--gpu", args.hardware]
        if kind == "check":
            command.append(str(candidate_path))
            if args.arch:
                command += ["--arch", args.arch]
            if args.sanitize:
                command += ["--sanitize", args.sanitize]
        else:
            command += [
                "--candidate",
                str(candidate_path),
                "--fmt",
                args.disassembly_format,
            ]
        init_kwargs_path = request_sidecar_dir / "init-kwargs.json"
        init_kwargs_path.write_text(
            json.dumps(request.get("init_kwargs") or {}, ensure_ascii=False),
            encoding="utf-8",
        )
        command += ["--init-kwargs", str(init_kwargs_path)]
        _append_typed_dependency_options(command, request, request_sidecar_dir)
        command += [
            "--http-timeout",
            str(MAX_HTTP_REQUEST_TIMEOUT),
            "--wait-timeout",
            str(args.timeout + queue_wait_grace),
            "--job-timeout",
            str(_gateway_job_timeout(args.timeout, queue_wait_grace)),
        ]
        for item in args.env:
            command += ["--env-var", item]
        return command

    options = request["options"]
    # Generalized workspaces deliberately expose only agent_problem.json to the
    # optimization agent.  The agate client still needs the evaluator-owned
    # shapes/reference files locally to assemble its typed eval payload, so point
    # --reference-dir at the private source while keeping the candidate in the
    # public workspace.  The private directory is never copied into the workspace.
    reference_dir = reference_dir or _private_reference_dir(workspace) or workspace
    command += ["--gpu", args.hardware]
    num_gpus = request.get("spec", {}).get("num_gpus", 1)
    if num_gpus > 1:
        if kind != "run":
            raise ValueError("Agate Profile CLI does not support multi-GPU requests")
        command += ["--num-gpus", str(num_gpus)]
    command += [
        "--candidate",
        str(candidate_path),
        "--reference-dir",
        str(reference_dir),
        "--operator",
        str(request["reference"]["operator"]),
        "--num-correctness-cases",
        str(options["num_correctness_cases"]),
        "--bench-iters",
        str(options["bench_iters"]),
        "--http-timeout",
        str(MAX_HTTP_REQUEST_TIMEOUT),
        "--wait-timeout",
        str(args.timeout + queue_wait_grace),
        "--job-timeout",
        str(_gateway_job_timeout(args.timeout, queue_wait_grace)),
    ]
    if kind == "run":
        command += ["--mode", str(request.get("mode") or "full"), "--set", "warmup_iters=5"]
    correctness_max_rel_l2 = options.get("correctness_max_rel_l2")
    if kind == "run" and correctness_max_rel_l2 is not None:
        command += [
            "--set",
            f"correctness_max_rel_l2={json.dumps(correctness_max_rel_l2)}",
        ]
    for item in args.env:
        command += ["--env-var", item]
    if kind == "profile":
        _append_typed_dependency_options(command, request, request_sidecar_dir)
        command += ["--level", args.profile_level]
        if args.profiler:
            command += ["--profiler", args.profiler]
        for counter in args.profile_counter:
            command += ["--counter", counter]
        if args.kernel_regex:
            command += ["--kernel-regex", args.kernel_regex]
        if args.kernel_name:
            command += ["--kernel-name", args.kernel_name]
        if args.profile_source:
            command.append("--source")
        if args.launch_skip is not None:
            command += ["--launch-skip", str(args.launch_skip)]
        if args.launch_count is not None:
            command += ["--launch-count", str(args.launch_count)]
        if args.top_kernels is not None:
            command += ["--top-kernels", str(args.top_kernels)]
    return command


def _typed_fallback_allowed(detail: object) -> bool:
    text = str(detail).lower()
    return any(reason in text for reason in TYPED_FALLBACK_REASONS)


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _request_shape_ids(request: dict[str, Any]) -> list[str]:
    shapes = request["reference"]["shapes"]

    def sort_key(shape_id: str) -> tuple[int, object]:
        return (0, int(shape_id)) if shape_id.isdigit() else (1, shape_id)

    return sorted((str(shape_id) for shape_id in shapes), key=sort_key)


def _shape_batches(shape_ids: list[str], batch_size: int) -> list[list[str]]:
    return [
        shape_ids[offset : offset + batch_size] for offset in range(0, len(shape_ids), batch_size)
    ]


def _shape_batch_request(request: dict[str, Any], shape_ids: list[str]) -> dict[str, Any]:
    batched = dict(request)
    reference = dict(request["reference"])
    reference["shapes"] = {
        shape_id: request["reference"]["shapes"][shape_id] for shape_id in shape_ids
    }
    for field in ("metadata", "roofline"):
        payload = reference.get(field)
        if not isinstance(payload, dict):
            continue
        payload = dict(payload)
        if isinstance(payload.get("shapes"), dict):
            payload["shapes"] = {
                shape_id: payload["shapes"][shape_id]
                for shape_id in shape_ids
                if shape_id in payload["shapes"]
            }
        if field == "metadata" and "num_shapes" in payload:
            payload["num_shapes"] = len(shape_ids)
        reference[field] = payload
    batched["reference"] = reference
    return batched


def _shape_batch_reference(reference: dict[str, Any], destination: Path) -> None:
    destination.mkdir()
    for filename, field in (
        ("reference.py", "reference_py"),
        ("input.py", "input_py"),
        ("shapes.json", "shapes"),
        ("metadata.json", "metadata"),
        ("roofline.json", "roofline"),
    ):
        value = reference.get(field)
        if value is not None:
            (destination / filename).write_text(
                value if isinstance(value, str) else json.dumps(value),
                encoding="utf-8",
            )


def _compile_failures(compile_result: object, shape_ids: list[str]) -> list[str]:
    """Return compile failures for aggregate and per-shape evaluator schemas.

    Older gateway/evaluator versions returned one ``{"status": ...}`` object,
    while current Atrex-Bench returns ``{shape_id: {"status": ...}}``.  Keep
    accepting the aggregate form, but require every expected shape to pass when
    the result is shape-scoped.
    """
    if not isinstance(compile_result, dict):
        compile_result = {}

    if "status" in compile_result:
        if compile_result.get("status") == "passed":
            return []
        return [
            "compile: "
            + str(compile_result.get("reason") or compile_result.get("status") or "did not pass")
        ]

    failures: list[str] = []
    for shape_id in shape_ids:
        status = compile_result.get(shape_id)
        status = status if isinstance(status, dict) else {}
        if status.get("status") != "passed":
            failures.append(
                f"sid={shape_id}: compile "
                + str(status.get("reason") or status.get("status") or "missing")
            )
    return failures


def _bounded_actionable_diagnostic(value: object, *, limit: int = 6000) -> str:
    """Keep useful compiler/exception context without emitting unbounded tracebacks."""
    text = str(value or "").replace("\x00", "").strip()
    if len(text) <= limit:
        return text
    head = min(1000, limit // 3)
    tail = limit - head - len("\n... diagnostic truncated ...\n")
    return text[:head] + "\n... diagnostic truncated ...\n" + text[-tail:]


def _looks_like_candidate_exception(value: object) -> bool:
    """Recognize build/import/driver failures without exposing arbitrary case errors."""
    text = str(value or "")
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "compilation error",
            "compileerror",
            "cuda_error_",
            "cudaerror",
            "culaunchkernel",
            "cumodule",
            "invalid context",
            "kernelparams",
            "modulenotfounderror:",
            "importerror:",
            "nvrtc",
            "syntaxerror:",
            "undefined symbol",
        )
    )


def _compile_diagnostics(compile_result: object, shape_ids: list[str]) -> list[dict[str, str]]:
    """Extract compile/import failures that are safe and useful in generalized mode."""
    if not isinstance(compile_result, dict):
        return []
    if "status" in compile_result:
        if compile_result.get("status") == "passed":
            return []
        message = _bounded_actionable_diagnostic(
            compile_result.get("reason") or compile_result.get("status")
        )
        return [{"stage": "compile_import", "shape_id": "", "message": message}] if message else []

    diagnostics: list[dict[str, str]] = []
    seen_messages: set[str] = set()
    for shape_id in shape_ids:
        status = compile_result.get(shape_id)
        status = status if isinstance(status, dict) else {}
        if status.get("status") == "passed":
            continue
        message = _bounded_actionable_diagnostic(
            status.get("reason") or status.get("status") or "missing"
        )
        if message in seen_messages:
            continue
        seen_messages.add(message)
        diagnostics.append(
            {
                "stage": "compile_import",
                "shape_id": shape_id,
                "message": message,
            }
        )
    return diagnostics


def _candidate_exception_diagnostics(
    correctness_status: dict[str, Any],
    correctness_shapes: dict[str, Any],
    shape_ids: list[str],
) -> list[dict[str, str]]:
    """Surface traceback-backed candidate failures without exposing numeric case data."""
    diagnostics: list[dict[str, str]] = []
    seen: set[str] = set()
    for shape_id in shape_ids:
        status = correctness_status.get(shape_id)
        status = status if isinstance(status, dict) else {}
        reasons: list[object] = [status.get("reason")]
        shape_result = correctness_shapes.get(shape_id)
        shape_result = shape_result if isinstance(shape_result, dict) else {}
        cases = shape_result.get("cases")
        for case in cases if isinstance(cases, list) else []:
            if isinstance(case, dict):
                reasons.append(case.get("error"))
        for reason in reasons:
            if not _looks_like_candidate_exception(reason):
                continue
            message = _bounded_actionable_diagnostic(reason)
            if not message or message in seen:
                continue
            seen.add(message)
            diagnostics.append(
                {
                    "stage": "candidate_runtime",
                    "shape_id": shape_id,
                    "message": message,
                }
            )
    return diagnostics


PERFORMANCE_OBJECTIVE = "shape_speedup_arithmetic_mean"


def _metadata_shape_latency_us(metadata: object, shape_id: str) -> float | None:
    """Read one authoritative production latency from Atrex-Bench metadata."""
    if not isinstance(metadata, dict):
        return None
    shapes = metadata.get("shapes")
    shape = shapes.get(shape_id) if isinstance(shapes, dict) else None
    if not isinstance(shape, dict):
        return None
    production = shape.get("production_performance")
    if not isinstance(production, dict):
        return None
    direct = _finite_number(production.get("performance_us"))
    if direct is not None and direct > 0.0:
        return direct
    nested = [
        value
        for entry in production.values()
        if isinstance(entry, dict)
        if (value := _finite_number(entry.get("performance_us"))) is not None and value > 0.0
    ]
    return nested[0] if len(nested) == 1 else None


def _metadata_speedup_mean(
    metadata: object,
    shape_ids: list[str],
    latency_by_shape: dict[str, float],
) -> tuple[float | None, list[str]]:
    if any(shape_id not in latency_by_shape for shape_id in shape_ids):
        return None, []
    speedups: list[float] = []
    failures: list[str] = []
    for shape_id in shape_ids:
        reference_us = _metadata_shape_latency_us(metadata, shape_id)
        if reference_us is None:
            failures.append(
                f"sid={shape_id}: metadata has no unambiguous positive "
                "production_performance.performance_us"
            )
            continue
        speedups.append(reference_us / latency_by_shape[shape_id])
    if failures or len(speedups) != len(shape_ids) or not speedups:
        return None, failures
    return sum(speedups) / len(speedups), []


def _optimizer_result_from_eval(
    payload: dict[str, Any],
    shape_ids: list[str],
    metadata: object,
    *,
    require_performance: bool = True,
) -> dict[str, Any]:
    """Convert the typed gateway's Atrex-Bench result to optimizer RESULT_JSON."""
    failures: list[str] = []
    if payload.get("error"):
        failures.append("evaluation: " + str(payload["error"]))
    passed = payload.get("passed")
    passed = passed if isinstance(passed, dict) else {}
    compile_result = passed.get("compile")
    failures.extend(_compile_failures(compile_result, shape_ids))
    actionable_diagnostics = _compile_diagnostics(compile_result, shape_ids)

    correctness_status = passed.get("correctness")
    correctness_status = correctness_status if isinstance(correctness_status, dict) else {}
    correctness = payload.get("correctness")
    correctness = correctness if isinstance(correctness, dict) else {}
    correctness_shapes = correctness.get("shapes")
    correctness_shapes = correctness_shapes if isinstance(correctness_shapes, dict) else {}
    actionable_diagnostics.extend(
        _candidate_exception_diagnostics(correctness_status, correctness_shapes, shape_ids)
    )
    max_abs = 0.0
    max_rel = 0.0
    for shape_id in shape_ids:
        status = correctness_status.get(shape_id)
        status = status if isinstance(status, dict) else {}
        if status.get("status") != "passed":
            failures.append(
                f"sid={shape_id}: correctness "
                + str(status.get("reason") or status.get("status") or "missing")
            )
        shape_result = correctness_shapes.get(shape_id)
        shape_result = shape_result if isinstance(shape_result, dict) else {}
        cases = shape_result.get("cases")
        for case in cases if isinstance(cases, list) else []:
            if not isinstance(case, dict):
                continue
            outputs = case.get("outputs")
            for output in outputs if isinstance(outputs, list) else []:
                if not isinstance(output, dict):
                    continue
                abs_diff = _finite_number(output.get("max_elementwise_abs_diff"))
                rel_diff = _finite_number(output.get("max_elementwise_rel_diff"))
                if abs_diff is not None:
                    max_abs = max(max_abs, abs_diff)
                if rel_diff is not None:
                    max_rel = max(max_rel, rel_diff)

    latency_by_shape: dict[str, float] = {}
    if require_performance:
        performance = payload.get("performance")
        performance = performance if isinstance(performance, dict) else {}
        performance_shapes = performance.get("shapes")
        performance_shapes = performance_shapes if isinstance(performance_shapes, dict) else {}
        for shape_id in shape_ids:
            shape_result = performance_shapes.get(shape_id)
            shape_result = shape_result if isinstance(shape_result, dict) else {}
            sample_ms: list[float] = []
            samples = shape_result.get("samples")
            for sample in samples if isinstance(samples, list) else []:
                if not isinstance(sample, dict):
                    continue
                value = _finite_number(sample.get("end_to_end_time_ms"))
                if value is not None and value > 0.0:
                    sample_ms.append(value)
            if shape_result.get("error") is not None or not sample_ms:
                failures.append(
                    f"sid={shape_id}: performance "
                    + str(shape_result.get("error") or "has no valid samples")
                )
                continue
            latency_by_shape[shape_id] = statistics.median(sample_ms) * 1000.0

    latencies = [
        latency_by_shape[shape_id] for shape_id in shape_ids if shape_id in latency_by_shape
    ]
    complete = len(latencies) == len(shape_ids)
    geomean = (
        math.exp(sum(math.log(value) for value in latencies) / len(latencies))
        if complete and latencies
        else 0.0
    )
    arithmetic = sum(latencies) / len(latencies) if complete and latencies else 0.0
    speedup_mean, metadata_failures = (
        _metadata_speedup_mean(metadata, shape_ids, latency_by_shape)
        if require_performance
        else (None, [])
    )
    failures.extend(metadata_failures)
    return {
        "all_pass": not failures,
        "failures": failures,
        "latency_us_geomean": geomean,
        "latency_us_arith_mean": arithmetic,
        "latency_us_by_shape": latency_by_shape,
        "speedup_vs_ref_mean": speedup_mean,
        "speedup_vs_ref_geomean": None,
        "performance_score": speedup_mean,
        "performance_objective": PERFORMANCE_OBJECTIVE,
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
        "evaluator": "atrex-gpu-gateway/run",
        "eval_id": payload.get("eval_id"),
        "actionable_diagnostics": actionable_diagnostics[:8],
    }


def _merge_optimizer_results(
    results: list[dict[str, Any]],
    shape_ids: list[str],
    metadata: object,
    *,
    require_performance: bool = True,
) -> dict[str, Any]:
    latency_by_shape = {
        str(shape_id): float(latency)
        for result in results
        for shape_id, latency in (result.get("latency_us_by_shape") or {}).items()
    }
    latencies = [
        latency_by_shape[shape_id] for shape_id in shape_ids if shape_id in latency_by_shape
    ]
    complete = not require_performance or len(latencies) == len(shape_ids)
    actionable_diagnostics: list[dict[str, str]] = []
    seen_diagnostics: set[tuple[str, str]] = set()
    for result in results:
        if len(actionable_diagnostics) >= 8:
            break
        for diagnostic in result.get("actionable_diagnostics") or []:
            if not isinstance(diagnostic, dict):
                continue
            normalized = {
                "stage": str(diagnostic.get("stage") or "candidate_runtime"),
                "shape_id": str(diagnostic.get("shape_id") or ""),
                "message": str(diagnostic.get("message") or ""),
            }
            key = (
                normalized["stage"],
                normalized["message"],
            )
            if not normalized["message"] or key in seen_diagnostics:
                continue
            seen_diagnostics.add(key)
            actionable_diagnostics.append(normalized)
            if len(actionable_diagnostics) >= 8:
                break

    speedup_mean, metadata_failures = (
        _metadata_speedup_mean(metadata, shape_ids, latency_by_shape)
        if require_performance
        else (None, [])
    )
    failures = [str(failure) for result in results for failure in (result.get("failures") or [])]
    for failure in metadata_failures:
        if failure not in failures:
            failures.append(failure)

    return {
        "all_pass": complete
        and (not require_performance or speedup_mean is not None)
        and all(result.get("all_pass") for result in results),
        "failures": failures,
        "latency_us_geomean": (
            math.exp(sum(math.log(value) for value in latencies) / len(latencies))
            if complete and latencies
            else 0.0
        ),
        "latency_us_arith_mean": (
            sum(latencies) / len(latencies) if complete and latencies else 0.0
        ),
        "latency_us_by_shape": latency_by_shape,
        "speedup_vs_ref_mean": speedup_mean,
        "speedup_vs_ref_geomean": None,
        "performance_score": speedup_mean,
        "performance_objective": PERFORMANCE_OBJECTIVE,
        "max_abs_err": max(float(result.get("max_abs_err") or 0.0) for result in results),
        "max_rel_err": max(float(result.get("max_rel_err") or 0.0) for result in results),
        "evaluator": "atrex-gpu-gateway/run/batched",
        "eval_id": results[-1].get("eval_id"),
        "shape_batch_count": len(results),
        "actionable_diagnostics": actionable_diagnostics,
    }


def _mask_generalized_result(workspace: Path, result: dict[str, Any]) -> dict[str, Any]:
    """Hide exact inputs and failures but retain real latency keyed by opaque shape id."""
    result = _with_workspace_performance_score(workspace, result)
    if not _is_generalized_workspace(workspace):
        return result
    masked = dict(result)
    if result.get("failures"):
        if result.get("actionable_diagnostics"):
            masked["failures"] = [
                "candidate compile/import/runtime failure; see actionable_diagnostics"
            ]
        else:
            masked["failures"] = [
                "one or more hidden evaluator cases failed; reproduce within the public shape_domain"
            ]
    masked["hidden_case_details"] = "shape inputs and failure details withheld"
    masked["shape_ids_are_opaque"] = True
    return masked


def _kernel_artifact_digest(kernel_bytes: bytes) -> str:
    return "sha256:" + hashlib.sha256(kernel_bytes).hexdigest()


def _store_kernel_artifact_at_root(
    evidence_root: Path, kernel_bytes: bytes
) -> dict[str, str]:
    """Store exact Kernel bytes privately and return their stable opaque identity."""
    digest = _kernel_artifact_digest(kernel_bytes)
    hexadecimal = digest.removeprefix("sha256:")
    artifact_dir = evidence_root / "kernel-artifacts" / "sha256" / hexadecimal
    if artifact_dir.is_symlink():
        raise RuntimeError("Kernel Artifact directory cannot be a symlink")
    ensure_private_directory(artifact_dir)
    lock_path = artifact_dir / ".identity.lock"
    lock_descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        destination = artifact_dir / "kernel.py"
        if destination.exists():
            if destination.is_symlink() or not destination.is_file():
                raise RuntimeError("Kernel Artifact path is unsafe")
            if destination.read_bytes() != kernel_bytes:
                raise RuntimeError("Kernel Artifact digest collision")
        else:
            temporary = artifact_dir / ".kernel.py.tmp"
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    descriptor = -1
                    stream.write(kernel_bytes)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, destination)
                fsync_directory(artifact_dir)
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
                temporary.unlink(missing_ok=True)

        identity_path = artifact_dir / "identity.json"
        if identity_path.exists():
            identity = _read_json_file(identity_path, max_bytes=4096)
            kernel_id = identity.get("kernel_id")
            recorded_digest = identity.get("kernel_artifact_digest")
            if (
                not isinstance(kernel_id, str)
                or KERNEL_RECORD_ID_RE.fullmatch(kernel_id) is None
                or recorded_digest != digest
            ):
                raise RuntimeError("Kernel identity record is invalid")
        else:
            kernel_id = f"kernel-{time.time_ns()}-{os.urandom(6).hex()}"
            durable_write_json(
                identity_path,
                {
                    "kernel_id": kernel_id,
                    "kernel_artifact_digest": digest,
                },
                indent=2,
                ensure_ascii=False,
            )
        return {"kernel_id": kernel_id, "kernel_artifact_digest": digest}
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)


def _record_episode_evaluation(
    workspace: Path,
    result: dict[str, Any],
    *,
    gateway_kind: str,
    job_id: object = None,
    private_result: object | None = None,
    gateway_task_digest: str | None = None,
    measurement_repetitions: list[dict[str, Any]] | None = None,
    kernel_subjects: dict[str, str] | None = None,
    kernel_bytes: bytes | None = None,
) -> dict[str, str] | None:
    """Persist the exact Kernel bytes and normalized Gateway result.

    The append-only index preserves measured candidate facts for reports and recovery.
    Each row now points at immutable per-call artifacts so later analysis never has
    to reconstruct a measured Kernel from the mutable worktree or Agent prose.
    """
    try:
        if kernel_bytes is None:
            kernel_bytes = (workspace / "kernel.py").read_bytes()
    except OSError:
        return None
    kernel_sha256 = hashlib.sha256(kernel_bytes).hexdigest()
    record_id = f"gateway-{time.time_ns()}-{kernel_sha256[:12]}"
    evidence_root_value = os.environ.get(SUPERVISOR_EVIDENCE_ROOT_ENV, "").strip()
    raw_evidence_root = Path(evidence_root_value) if evidence_root_value else None
    evidence_root = raw_evidence_root.resolve() if raw_evidence_root is not None else workspace
    if evidence_root_value and (
        raw_evidence_root is None
        or not raw_evidence_root.is_absolute()
        or raw_evidence_root.is_symlink()
    ):
        raise RuntimeError("Supervisor evidence root must be an absolute real path")
    record_root = (
        evidence_root / "gateway-records"
        if evidence_root_value
        else evidence_root / GATEWAY_RECORDS_PATH
    )
    evidence_store_root = record_root.parent
    kernel_artifact = _store_kernel_artifact_at_root(evidence_store_root, kernel_bytes)
    kernel_artifact_digest = kernel_artifact["kernel_artifact_digest"]
    kernel_id = kernel_artifact["kernel_id"]
    kernel_subject_ids: dict[str, str] = {}
    for role, hexadecimal in (kernel_subjects or {}).items():
        subject_digest = f"sha256:{hexadecimal}"
        subject = (
            kernel_artifact
            if subject_digest == kernel_artifact_digest
            else _kernel_identity_for_digest(workspace, subject_digest)
        )
        kernel_subject_ids[role] = subject["kernel_id"]
    record_dir = record_root / record_id
    kernel_path = record_dir / "kernel.py"
    result_path = record_dir / "result.json"
    ensure_private_directory(record_dir)
    temporary_kernel = record_dir / ".kernel.py.tmp"
    descriptor = os.open(temporary_kernel, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(kernel_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_kernel, kernel_path)
        fsync_directory(record_dir)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary_kernel.unlink(missing_ok=True)
    record_value: dict[str, Any] = {
        "gateway_kind": gateway_kind,
        "job_id": str(job_id) if job_id else None,
        "kernel_sha256": kernel_sha256,
        "kernel_id": kernel_id,
        "kernel_artifact_digest": kernel_artifact_digest,
        "result": result,
    }
    if isinstance(private_result, dict) and isinstance(private_result.get("status"), str):
        # Preserve execution completion independently of correctness/probe payloads.
        record_value["execution_status"] = private_result["status"]
    if kernel_subjects:
        record_value["kernel_subjects"] = kernel_subjects
        record_value["kernel_subject_ids"] = kernel_subject_ids
    if gateway_task_digest is not None:
        record_value["gateway_task_digest"] = gateway_task_digest
    durable_write_json(
        result_path,
        record_value,
        indent=2,
        ensure_ascii=False,
    )
    raw_result_path: Path | None = None
    if private_result is not None:
        raw_result_path = record_dir / "raw-result.json"
        durable_write_json(
            raw_result_path,
            private_result,
            indent=2,
            ensure_ascii=False,
        )
    payload = {
        "schema_version": 3,
        "record_id": record_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gateway_kind": gateway_kind,
        "job_id": str(job_id) if job_id else None,
        "kernel_sha256": kernel_sha256,
        "kernel_id": kernel_id,
        "kernel_artifact_digest": kernel_artifact_digest,
        "kernel_artifact": str(kernel_path.relative_to(evidence_root)),
        "result_artifact": str(result_path.relative_to(evidence_root)),
        "result": result,
    }
    if gateway_task_digest is not None:
        payload["gateway_task_digest"] = gateway_task_digest
    if measurement_repetitions:
        payload["measurement_repetitions"] = measurement_repetitions
    if kernel_subjects:
        payload["kernel_subjects"] = kernel_subjects
        payload["kernel_subject_ids"] = kernel_subject_ids
    if raw_result_path is not None:
        payload["raw_result_artifact"] = str(raw_result_path.relative_to(evidence_root))
    path = (
        evidence_root / "evaluations.jsonl"
        if evidence_root_value
        else evidence_root / EPISODE_EVALUATIONS_PATH
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, line)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {
        "record_id": record_id,
        "kernel_sha256": kernel_sha256,
        "kernel_id": kernel_id,
        "kernel_artifact_digest": kernel_artifact_digest,
    }


def _supervisor_evaluations_path(workspace: Path) -> Path:
    value = os.environ.get(SUPERVISOR_EVIDENCE_ROOT_ENV, "").strip()
    if value:
        root = Path(value)
        if not root.is_absolute() or root.is_symlink():
            raise RuntimeError("Supervisor evidence root must be an absolute real path")
        return root.resolve() / "evaluations.jsonl"
    return workspace / EPISODE_EVALUATIONS_PATH


def _gateway_record_root(workspace: Path) -> Path:
    return _supervisor_evaluations_path(workspace).parent / "gateway-records"


def _kernel_artifact_root(workspace: Path) -> Path:
    return _supervisor_evaluations_path(workspace).parent / "kernel-artifacts" / "sha256"


def _historical_evidence_roots() -> list[Path]:
    value = os.environ.get(SUPERVISOR_HISTORY_ROOT_ENV, "").strip()
    if not value:
        return []
    root = Path(value)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        return []
    return [
        path
        for path in sorted(root.glob("e*/supervisor_runtime"))
        if path.is_dir() and not path.is_symlink()
    ]


def _gateway_record_roots(workspace: Path) -> list[Path]:
    return [
        *(root / "gateway-records" for root in _historical_evidence_roots()),
        _gateway_record_root(workspace),
    ]


def _kernel_artifact_roots(workspace: Path) -> list[Path]:
    return [
        *(root / "kernel-artifacts" / "sha256" for root in _historical_evidence_roots()),
        _kernel_artifact_root(workspace),
    ]


def _store_kernel_artifact(workspace: Path, kernel_bytes: bytes) -> dict[str, str]:
    return _store_kernel_artifact_at_root(
        _supervisor_evaluations_path(workspace).parent,
        kernel_bytes,
    )


def _read_json_file(path: Path, *, max_bytes: int) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Gateway record file is missing: {path.name}")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ValueError(f"Gateway record file cannot be inspected: {path.name}") from exc
    if size > max_bytes:
        raise ValueError(f"Gateway record file exceeds {max_bytes} bytes: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Gateway record file is invalid: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Gateway record file must contain an object: {path.name}")
    return value


def _kernel_identity_for_digest(workspace: Path, digest: str) -> dict[str, str]:
    if KERNEL_ARTIFACT_DIGEST_RE.fullmatch(digest) is None:
        raise ValueError("private Kernel Artifact Digest is invalid")
    identity: dict[str, Any] | None = None
    for root in _kernel_artifact_roots(workspace):
        path = root / digest.removeprefix("sha256:") / "identity.json"
        try:
            identity = _read_json_file(path, max_bytes=4096)
        except ValueError:
            continue
        break
    if identity is None:
        raise ValueError("Kernel Artifact Digest does not exist in visible history")
    kernel_id = identity.get("kernel_id")
    if (
        not isinstance(kernel_id, str)
        or KERNEL_RECORD_ID_RE.fullmatch(kernel_id) is None
        or identity.get("kernel_artifact_digest") != digest
    ):
        raise ValueError("Kernel identity record is invalid")
    return {"kernel_id": kernel_id, "kernel_artifact_digest": digest}


def _load_kernel_identity(workspace: Path, kernel_id: str) -> dict[str, str]:
    if KERNEL_RECORD_ID_RE.fullmatch(kernel_id) is None:
        raise ValueError("Kernel ID has an invalid format")
    for root in _kernel_artifact_roots(workspace):
        if root.is_symlink() or not root.is_dir():
            continue
        for artifact_dir in root.iterdir():
            if (
                artifact_dir.is_symlink()
                or not artifact_dir.is_dir()
                or re.fullmatch(r"[0-9a-f]{64}", artifact_dir.name) is None
            ):
                continue
            identity_path = artifact_dir / "identity.json"
            if not identity_path.is_file() or identity_path.is_symlink():
                continue
            try:
                identity = _read_json_file(identity_path, max_bytes=4096)
            except ValueError:
                continue
            if identity.get("kernel_id") != kernel_id:
                continue
            digest = identity.get("kernel_artifact_digest")
            if digest != f"sha256:{artifact_dir.name}":
                raise ValueError("Kernel identity record is inconsistent")
            return {"kernel_id": kernel_id, "kernel_artifact_digest": str(digest)}
    raise ValueError("Kernel record does not exist in visible history")


def _valid_kernel_sha256(value: object) -> str | None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        return None
    return value


def _load_gateway_record(workspace: Path, record_id: str) -> dict[str, Any]:
    """Load one immutable Supervisor-owned Gateway record without exposing raw evidence."""
    if GATEWAY_RECORD_ID_RE.fullmatch(record_id) is None:
        raise ValueError("Gateway record ID has an invalid format")
    record_dir: Path | None = None
    for root in _gateway_record_roots(workspace):
        candidate = root / record_id
        if candidate.is_dir() and not candidate.is_symlink():
            record_dir = candidate
            break
    if record_dir is None:
        raise ValueError("Gateway record does not exist in visible history")
    value = _read_json_file(record_dir / "result.json", max_bytes=MAX_GATEWAY_RECORD_BYTES)
    gateway_kind = value.get("gateway_kind")
    result = value.get("result")
    kernel_sha256 = _valid_kernel_sha256(value.get("kernel_sha256"))
    if not isinstance(gateway_kind, str) or not isinstance(result, dict):
        raise ValueError("Gateway record has an invalid result contract")
    if kernel_sha256 is None:
        raise ValueError("Gateway record has an invalid Kernel digest")
    kernel_artifact_digest = value.get("kernel_artifact_digest")
    if not isinstance(kernel_artifact_digest, str) or KERNEL_ARTIFACT_DIGEST_RE.fullmatch(
        kernel_artifact_digest
    ) is None:
        kernel_artifact_digest = f"sha256:{kernel_sha256}"
    kernel_id = value.get("kernel_id")
    if not isinstance(kernel_id, str) or KERNEL_RECORD_ID_RE.fullmatch(kernel_id) is None:
        try:
            source = (record_dir / "kernel.py").read_bytes()
        except OSError as exc:
            raise ValueError("Gateway record Kernel source is missing") from exc
        identity = _store_kernel_artifact(workspace, source)
        if identity["kernel_artifact_digest"] != kernel_artifact_digest:
            raise ValueError("Gateway record Kernel identity is inconsistent")
        kernel_id = identity["kernel_id"]
    else:
        identity = _kernel_identity_for_digest(workspace, kernel_artifact_digest)
        if identity["kernel_id"] != kernel_id:
            raise ValueError("Gateway record Kernel ID is inconsistent")
    kernel_subject_ids = value.get("kernel_subject_ids")
    if not isinstance(kernel_subject_ids, dict):
        kernel_subject_ids = {}
    if gateway_kind == "same_allocation_abba" and not kernel_subject_ids:
        raw_subjects = value.get("kernel_subjects")
        if isinstance(raw_subjects, dict):
            for role, hexadecimal in raw_subjects.items():
                digest = f"sha256:{hexadecimal}"
                try:
                    subject = _kernel_identity_for_digest(workspace, digest)
                except ValueError:
                    if role != "incumbent":
                        raise
                    raw_path = record_dir / "raw-result.json"
                    raw = _read_json_file(raw_path, max_bytes=64 * 1024 * 1024)
                    source = raw.get("baseline_source")
                    if not isinstance(source, str):
                        raise ValueError("ABBA Gateway record omitted its Incumbent Kernel")
                    subject = _store_kernel_artifact(workspace, source.encode("utf-8"))
                    if subject["kernel_artifact_digest"] != digest:
                        raise ValueError("ABBA Incumbent Kernel digest is inconsistent")
                kernel_subject_ids[str(role)] = subject["kernel_id"]
    if gateway_kind == "same_allocation_abba" and set(kernel_subject_ids) != {
        "incumbent",
        "candidate",
    }:
        raise ValueError("ABBA Gateway record has an invalid Kernel identity contract")
    if any(
        not isinstance(subject_id, str)
        or KERNEL_RECORD_ID_RE.fullmatch(subject_id) is None
        for subject_id in kernel_subject_ids.values()
    ):
        raise ValueError("ABBA Gateway record contains an invalid Kernel ID")
    if kernel_subject_ids and kernel_subject_ids.get("candidate") != kernel_id:
        raise ValueError("ABBA Candidate Kernel ID disagrees with its Gateway record")
    return {
        "record_id": record_id,
        "gateway_kind": gateway_kind,
        "kernel_sha256": kernel_sha256,
        "kernel_id": kernel_id,
        "kernel_artifact_digest": kernel_artifact_digest,
        "kernel_subjects": value.get("kernel_subjects"),
        "gateway_task_digest": value.get("gateway_task_digest"),
        "kernel_subject_ids": kernel_subject_ids,
        "result": result,
        "record_dir": record_dir,
    }


def _latency_us_by_shape(value: object) -> dict[str, float]:
    if not isinstance(value, dict) or not isinstance(value.get("latency_us_by_shape"), dict):
        return {}
    return {
        str(shape_id): number
        for shape_id, latency in value["latency_us_by_shape"].items()
        if (number := _finite_number(latency)) is not None and number > 0.0
    }


def _replace_shape_latencies(
    result: dict[str, Any], replacements: dict[str, float]
) -> dict[str, Any]:
    updated = dict(result)
    values = _latency_us_by_shape(result)
    accepted = {
        shape_id: latency
        for shape_id, latency in replacements.items()
        if shape_id in values and latency > 0.0 and math.isfinite(latency)
    }
    values.update(accepted)
    updated["latency_us_by_shape"] = values
    if accepted and values:
        latencies = list(values.values())
        updated["latency_us_geomean"] = (
            latencies[0]
            if len(latencies) == 1
            else math.exp(sum(math.log(value) for value in latencies) / len(latencies))
        )
        updated["latency_us_arith_mean"] = sum(latencies) / len(latencies)
    return updated


def _measurement_aggregation_summary() -> dict[str, Any]:
    return {
        "repetitions": MEASUREMENT_REPETITIONS,
        "method": "per_shape_median",
    }


def _median_latency_by_shape(
    samples: list[dict[str, float]], shape_ids: list[str]
) -> dict[str, float]:
    if len(samples) != MEASUREMENT_REPETITIONS:
        raise RuntimeError(f"measurement aggregation requires {MEASUREMENT_REPETITIONS} samples")
    expected = set(shape_ids)
    if not expected or any(set(sample) != expected for sample in samples):
        raise RuntimeError("repeated measurements have inconsistent Shape coverage")
    return {
        shape_id: statistics.median(sample[shape_id] for sample in samples)
        for shape_id in shape_ids
    }


def _gateway_task_digest(
    gateway_kind: str,
    kernel_sha256: str,
    request_identity: object,
    *,
    baseline_sha256: str | None = None,
    measurement_repetitions: int = MEASUREMENT_REPETITIONS,
) -> str:
    payload = json.dumps(
        {
            "measurement_contract": 2,
            "measurement_repetitions": measurement_repetitions,
            "gateway_kind": gateway_kind,
            "kernel_sha256": kernel_sha256,
            "baseline_sha256": baseline_sha256,
            "request": request_identity,
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _gateway_task_root(workspace: Path) -> Path:
    root = _supervisor_evaluations_path(workspace).parent / "gateway-tasks"
    ensure_private_directory(root)
    return root


@contextmanager
def _locked_gateway_task_root(workspace: Path) -> Iterator[Path]:
    root = _gateway_task_root(workspace)
    lock_path = root / ".lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield root
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _reserve_gateway_task(workspace: Path, task_digest: str) -> tuple[str | None, str | None]:
    """Atomically reserve a task or return its previous public Gateway record ID."""
    owner = f"{os.getpid()}-{time.time_ns()}"
    with _locked_gateway_task_root(workspace) as root:
        marker = root / f"{task_digest}.json"
        if marker.is_file():
            try:
                state = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError("Supervisor Gateway task marker is corrupt") from exc
            if not isinstance(state, dict):
                raise RuntimeError("Supervisor Gateway task marker is invalid")
            record_id = state.get("gateway_record_id")
            if state.get("status") == "completed" and isinstance(record_id, str):
                return None, record_id
            pid = state.get("pid")
            if isinstance(pid, int) and pid > 0:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    return None, None
                else:
                    return None, None
        # Duplicate detection is shared by Agents and trusted consumers. The reuse
        # flag controls the response, not visibility of completed historical tasks.
        for evidence in reversed(_historical_evidence_roots()):
            previous = evidence / "gateway-tasks" / f"{task_digest}.json"
            if previous.is_file() and not previous.is_symlink():
                state = _read_json_file(previous, max_bytes=4096)
                if state.get("status") == "completed" and isinstance(
                    state.get("gateway_record_id"), str
                ):
                    return None, state["gateway_record_id"]
        durable_write_json(
            marker,
            {
                "status": "running",
                "owner": owner,
                "pid": os.getpid(),
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            ensure_ascii=False,
        )
    return owner, None


def _complete_gateway_task(
    workspace: Path,
    task_digest: str,
    owner: str,
    gateway_record_id: str,
) -> None:
    with _locked_gateway_task_root(workspace) as root:
        marker = root / f"{task_digest}.json"
        try:
            state = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("Supervisor Gateway task reservation was lost") from exc
        if not isinstance(state, dict) or state.get("owner") != owner:
            raise RuntimeError("Supervisor Gateway task is owned by another request")
        durable_write_json(
            marker,
            {
                "status": "completed",
                "gateway_record_id": gateway_record_id,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            ensure_ascii=False,
        )


def _abandon_gateway_task(workspace: Path, task_digest: str, owner: str) -> None:
    with _locked_gateway_task_root(workspace) as root:
        marker = root / f"{task_digest}.json"
        try:
            state = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(state, dict) and state.get("owner") == owner:
            marker.unlink(missing_ok=True)
            fsync_directory(root)


@dataclass
class _GatewayTask:
    workspace: Path
    digest: str | None
    kernel_bytes: bytes
    owner: str | None
    previous_record_id: str | None
    record_id: str | None = None

    def record(self, result: dict[str, Any], *, cache: bool = True, **metadata: Any) -> dict[str, str]:
        record = _record_episode_evaluation(
            self.workspace, result, gateway_task_digest=self.digest,
            kernel_bytes=self.kernel_bytes, **metadata,
        )
        if record is None:
            raise RuntimeError("Gateway task could not preserve its measured Kernel")
        self.record_id = record["record_id"]
        if cache and self.digest is not None and self.owner is not None:
            _complete_gateway_task(self.workspace, self.digest, self.owner, self.record_id)
        return record


@contextmanager
def _gateway_task(
    workspace: Path, digest: str | None, kernel_bytes: bytes,
) -> Iterator[_GatewayTask]:
    owner, previous = _reserve_gateway_task(workspace, digest) if digest is not None else (None, None)
    try:
        yield _GatewayTask(workspace, digest, kernel_bytes, owner, previous)
    finally:
        # Completed markers no longer contain owner, so this only releases unfinished work.
        if digest is not None and owner is not None:
            _abandon_gateway_task(workspace, digest, owner)


def _bundle_task_inputs(bundle: str | None) -> tuple[dict[str, Any], bytes | None]:
    """Hash the uploaded file contents, not tar/gzip timestamps or host paths."""
    if bundle is None:
        return {}, None
    inputs: dict[str, Any] = {}
    kernel_bytes = None
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(bundle)), mode="r:gz") as archive:
        for member in archive:
            if not member.isfile():
                raise ValueError("Gateway input bundle contains a non-file entry")
            stream = archive.extractfile(member)
            if stream is None or member.name in inputs:
                raise ValueError("Gateway input bundle contains an invalid or repeated entry")
            with stream:
                content = stream.read()
            inputs[member.name] = {
                "sha256": hashlib.sha256(content).hexdigest(),
                "mode": member.mode,
            }
            if member.name == "kernel.py":
                kernel_bytes = content
    return inputs, kernel_bytes


def _bounded_text(value: object, limit: int = 1000) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _add_gateway_record_identity(
    projected: dict[str, Any], record: dict[str, Any] | None
) -> None:
    if not record:
        return
    projected["gateway_record_id"] = record["record_id"]
    kernel_id = record.get("kernel_id")
    if isinstance(kernel_id, str):
        projected["kernel_id"] = kernel_id


def _agent_evaluation_result(
    result: dict[str, Any], record: dict[str, Any] | None
) -> dict[str, Any]:
    """Return only decision-relevant evaluator facts to the Agent."""
    projected: dict[str, Any] = {
        key: result[key]
        for key in (
            "all_pass",
            "latency_us_geomean",
            "latency_us_arith_mean",
            "performance_score",
            "performance_objective",
            "max_abs_err",
            "max_rel_err",
            "mode",
            "input_scope",
        )
        if key in result
    }
    by_shape = result.get("latency_us_by_shape")
    if isinstance(by_shape, dict):
        projected["latency_us_by_shape"] = {
            _bounded_text(shape_id, 128): number
            for shape_id, raw_number in list(by_shape.items())[:MAX_AGENT_EVALUATION_SHAPES]
            if (number := _finite_number(raw_number)) is not None and number >= 0.0
        }
        if len(by_shape) > MAX_AGENT_EVALUATION_SHAPES:
            projected["shape_results_omitted"] = len(by_shape) - MAX_AGENT_EVALUATION_SHAPES
    projected["failures"] = [_bounded_text(value) for value in (result.get("failures") or [])[:8]]
    projected["actionable_diagnostics"] = [
        {
            key: _bounded_text(value)
            for key, value in diagnostic.items()
            if key in {"stage", "shape_id", "message"}
        }
        for diagnostic in (result.get("actionable_diagnostics") or [])[:8]
        if isinstance(diagnostic, dict)
    ]
    if "hidden_case_details" in result:
        projected["hidden_case_details"] = result["hidden_case_details"]
    if result.get("shape_ids_are_opaque") is True:
        projected["shape_ids_are_opaque"] = True
    _add_gateway_record_identity(projected, record)
    return projected


def _profile_duration_us(kernel: dict[str, Any]) -> float | None:
    duration_us = _finite_number(kernel.get("duration_us"))
    if duration_us is not None and duration_us >= 0.0:
        return duration_us
    duration = _finite_number(kernel.get("duration"))
    unit = kernel.get("duration_unit")
    if duration is None or duration < 0.0 or not isinstance(unit, str):
        return None
    scale = {"ns": 0.001, "us": 1.0, "ms": 1000.0, "s": 1_000_000.0}.get(unit.casefold())
    return None if scale is None else duration * scale


def _agent_profile_metric(value: object) -> object | None:
    """Project one requested profiler counter without serving raw profiler text."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value if math.isfinite(float(value)) else None
    if isinstance(value, str):
        return _bounded_text(value, 256)
    if not isinstance(value, dict):
        return None
    projected: dict[str, object] = {}
    for key, item in value.items():
        if key not in {"value", "unit"} or isinstance(item, bool):
            continue
        if isinstance(item, float) and not math.isfinite(item):
            continue
        if item is None or isinstance(item, (int, float)):
            projected[key] = item
        elif isinstance(item, str):
            projected[key] = _bounded_text(item, 256)
    return projected or None


def _agent_profile_kernel(raw: dict[str, Any], *, total_duration_us: float) -> dict[str, Any]:
    """Create an explicit, bounded allowlist for one profiled Kernel."""
    projected: dict[str, Any] = {}
    name = raw.get("name", raw.get("kernel_name"))
    if isinstance(name, str) and name:
        projected["name"] = _bounded_text(name, 512)
    duration_us = _profile_duration_us(raw)
    if duration_us is not None:
        projected["duration_us"] = duration_us
        if total_duration_us > 0.0:
            projected["duration_share_pct"] = duration_us * 100.0 / total_duration_us

    aliases = {
        "mem_sol_pct": "memory_sol_pct",
        "registers": "registers_per_thread",
        "smem_bytes": "shared_memory_bytes",
    }
    numeric_fields = (
        "compute_sol_pct",
        "memory_sol_pct",
        "registers_per_thread",
        "shared_memory_bytes",
        "occupancy_pct",
        "achieved_occupancy_pct",
        "dram_throughput_pct",
        "memory_throughput_pct",
        "sm_throughput_pct",
        "waves_per_sm",
    )
    for field in numeric_fields:
        value = _finite_number(raw.get(field))
        if value is None:
            source = next((source for source, target in aliases.items() if target == field), None)
            value = _finite_number(raw.get(source)) if source else None
        if value is not None:
            projected[field] = value
    bound = raw.get("bound")
    if isinstance(bound, str) and bound:
        projected["bound"] = _bounded_text(bound, 64)
    elif "compute_sol_pct" in projected and "memory_sol_pct" in projected:
        projected["bound"] = (
            "compute" if projected["compute_sol_pct"] > projected["memory_sol_pct"] else "memory"
        )

    metrics = raw.get("metrics")
    if isinstance(metrics, dict):
        projected_metrics: dict[str, object] = {}
        for key, value in list(metrics.items())[:MAX_AGENT_PROFILE_METRICS]:
            metric = _agent_profile_metric(value)
            if metric is not None:
                projected_metrics[_bounded_text(key, 256)] = metric
        if projected_metrics:
            projected["metrics"] = projected_metrics
    return projected


def _agent_clock_lock(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    projected = {
        key: (_bounded_text(item, 256) if isinstance(item, str) else item)
        for key, item in value.items()
        if key in {"requested", "applied", "locked", "supported", "status", "reason"}
        and isinstance(item, (bool, str, int, float))
    }
    return projected or None


def _agent_profile_result(result: dict[str, Any], record: dict[str, Any] | None) -> dict[str, Any]:
    kernels_value = result.get("kernels")
    kernels = (
        [item for item in kernels_value if isinstance(item, dict)]
        if isinstance(kernels_value, list)
        else []
    )
    durations = [
        duration for kernel in kernels if (duration := _profile_duration_us(kernel)) is not None
    ]
    total_duration_us = sum(durations)
    projected_kernels = [
        _agent_profile_kernel(kernel, total_duration_us=total_duration_us)
        for kernel in kernels[:MAX_AGENT_PROFILE_KERNELS]
    ]
    projected_kernels = [kernel for kernel in projected_kernels if kernel]

    projected: dict[str, Any] = {}
    for key in ("status", "passed", "shape_id", "profile_level", "level"):
        value = result.get(key)
        if isinstance(value, (bool, str, int, float)) and not (
            isinstance(value, float) and not math.isfinite(value)
        ):
            projected[key] = _bounded_text(value, 256) if isinstance(value, str) else value
    exit_code = result.get("exit_code")
    if isinstance(exit_code, int) and not isinstance(exit_code, bool):
        projected["exit_code"] = exit_code
    clock_lock = _agent_clock_lock(result.get("clock_lock"))
    if clock_lock:
        projected["clock_lock"] = clock_lock
    summary = result.get("summary")
    if isinstance(summary, str) and summary:
        projected["summary"] = _bounded_text(summary, 2000)
    error = result.get("error")
    if error:
        projected["error"] = _bounded_text(error, 1000)

    projected["kernel_count"] = len(kernels)
    if total_duration_us > 0.0:
        projected["total_duration_us"] = total_duration_us
        dominant = max(
            kernels,
            key=lambda kernel: _profile_duration_us(kernel) or 0.0,
        )
        dominant_name = dominant.get("name", dominant.get("kernel_name"))
        if isinstance(dominant_name, str) and dominant_name:
            projected["dominant_kernel"] = _bounded_text(dominant_name, 512)

    weighted_duration = 0.0
    weighted_sol = 0.0
    weighted_compute = 0.0
    weighted_memory = 0.0
    for kernel in projected_kernels:
        duration = _finite_number(kernel.get("duration_us"))
        compute = _finite_number(kernel.get("compute_sol_pct"))
        memory = _finite_number(kernel.get("memory_sol_pct"))
        if duration is None or compute is None or memory is None:
            continue
        weighted_duration += duration
        weighted_sol += max(compute, memory) * duration
        weighted_compute += compute * duration
        weighted_memory += memory * duration
    if weighted_duration > 0.0:
        projected["weighted_sol_pct"] = weighted_sol / weighted_duration
        projected["dominant_bound"] = "compute" if weighted_compute > weighted_memory else "memory"
    projected["kernels"] = projected_kernels
    if len(kernels) > len(projected_kernels):
        projected["kernels_omitted"] = len(kernels) - len(projected_kernels)
    _add_gateway_record_identity(projected, record)
    return projected


def _agent_diagnostic_item(value: object) -> object | None:
    """Project one compiler diagnostic without exposing request or worker state."""
    if isinstance(value, str):
        return _bounded_text(value, 2000)
    if not isinstance(value, dict):
        return None
    projected: dict[str, object] = {}
    text_fields = {
        "severity",
        "level",
        "stage",
        "message",
        "file",
        "kernel",
        "name",
        "kind",
    }
    numeric_fields = {
        "line",
        "column",
        "registers",
        "registers_per_thread",
        "spills",
        "spill_bytes",
        "shared_memory_bytes",
        "local_memory_bytes",
        "stack_frame_bytes",
    }
    for key in text_fields:
        item = value.get(key)
        if isinstance(item, str) and item:
            projected[key] = _bounded_text(item, 2000 if key == "message" else 512)
    for key in numeric_fields:
        item = value.get(key)
        if (
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and (not isinstance(item, float) or math.isfinite(item))
        ):
            projected[key] = item
    return projected or None


def _agent_check_result(result: dict[str, Any], record: dict[str, Any] | None) -> dict[str, Any]:
    """Return only actionable compilation and sanitizer facts to the Agent."""
    projected: dict[str, Any] = {}
    for key in (
        "status",
        "ok",
        "passed",
        "compile_ok",
        "launch_ok",
        "asm_available",
        "correctness_checked",
        "sanitize",
        "sanitizer_passed",
        "scope",
        "shape_id",
        "failure_stage",
    ):
        if key not in result:
            continue
        value = result.get(key)
        if value is None or isinstance(value, (bool, int, float, str)):
            if isinstance(value, float) and not math.isfinite(value):
                continue
            projected[key] = _bounded_text(value, 512) if isinstance(value, str) else value
    for key in (
        "registers",
        "registers_per_thread",
        "spills",
        "spill_bytes",
        "shared_memory_bytes",
        "local_memory_bytes",
        "stack_frame_bytes",
    ):
        value = result.get(key)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and (not isinstance(value, float) or math.isfinite(value))
        ):
            projected[key] = value
    diagnostics = result.get("diagnostics")
    if isinstance(diagnostics, list):
        projected_diagnostics = [
            item
            for raw in diagnostics[:MAX_AGENT_DIAGNOSTICS]
            if (item := _agent_diagnostic_item(raw)) is not None
        ]
        projected["diagnostics"] = projected_diagnostics
        if len(diagnostics) > MAX_AGENT_DIAGNOSTICS:
            projected["diagnostics_omitted"] = len(diagnostics) - MAX_AGENT_DIAGNOSTICS
    error = result.get("error")
    if error:
        projected["error"] = _bounded_text(error, 4000)
    _add_gateway_record_identity(projected, record)
    return projected


def _bounded_utf8_text(
    value: str,
    limit: int,
    *,
    marker: str = "assembly",
) -> tuple[str, int]:
    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) <= limit:
        return value, 0
    marker_bytes = f"\n... <{marker} truncated by Supervisor Runtime> ...\n".encode()
    budget = max(0, limit - len(marker_bytes))
    head_size = budget * 2 // 3
    tail_size = budget - head_size
    head = encoded[:head_size].decode("utf-8", errors="ignore")
    tail = encoded[-tail_size:].decode("utf-8", errors="ignore")
    text = head + marker_bytes.decode() + tail
    return text, len(encoded) - len(text.encode("utf-8"))


def _agent_disassembly_result(
    result: dict[str, Any], record: dict[str, Any] | None
) -> dict[str, Any]:
    """Return bounded SASS/PTX text plus useful Kernel resource facts."""
    projected: dict[str, Any] = {}
    for key in (
        "status",
        "ok",
        "passed",
        "compile_ok",
        "launch_ok",
        "scope",
        "shape_id",
        "format",
        "arch",
        "failure_stage",
    ):
        if key not in result:
            continue
        value = result.get(key)
        if value is None or isinstance(value, (bool, int, float, str)):
            if isinstance(value, float) and not math.isfinite(value):
                continue
            projected[key] = _bounded_text(value, 512) if isinstance(value, str) else value
    if "format" not in projected and isinstance(result.get("fmt"), str):
        projected["format"] = _bounded_text(result["fmt"], 64)

    kernels = result.get("kernels")
    if isinstance(kernels, list):
        public_kernels = [
            _agent_profile_kernel(kernel, total_duration_us=0.0)
            for kernel in kernels[:MAX_AGENT_PROFILE_KERNELS]
            if isinstance(kernel, dict)
        ]
        projected["kernels"] = [kernel for kernel in public_kernels if kernel]
        if len(kernels) > MAX_AGENT_PROFILE_KERNELS:
            projected["kernels_omitted"] = len(kernels) - MAX_AGENT_PROFILE_KERNELS

    assembly: str | None = None
    exports = result.get("exports")
    assembly_name: str | None = None
    if isinstance(exports, dict):
        preferred = str(result.get("format") or "") + ".txt"
        candidates = [preferred, "sass.txt", "ptx.txt"]
        for name in candidates:
            exported = exports.get(name)
            if isinstance(exported, dict) and isinstance(exported.get("text"), str):
                assembly = exported["text"]
                assembly_name = name
                break
            if isinstance(exported, str):
                assembly = exported
                assembly_name = name
                break
    recorded_assembly: dict[str, Any] | None = None
    if assembly is None and isinstance(result.get("assembly"), dict):
        recorded_assembly = result["assembly"]
        if isinstance(recorded_assembly.get("text"), str):
            assembly = recorded_assembly["text"]
            assembly_name = str(recorded_assembly.get("format") or "unknown")
    if assembly is None:
        for key in ("assembly", "asm", "sass", "ptx", "text"):
            value = result.get(key)
            if isinstance(value, str) and value:
                assembly = value
                assembly_name = key
                break
    if assembly is not None:
        text, omitted = _bounded_utf8_text(assembly, MAX_AGENT_DISASSEMBLY_BYTES)
        recorded_size = (
            recorded_assembly.get("size_bytes")
            if recorded_assembly is not None
            else None
        )
        size_bytes = (
            recorded_size
            if isinstance(recorded_size, int)
            and not isinstance(recorded_size, bool)
            and recorded_size >= len(text.encode("utf-8", errors="replace"))
            else len(assembly.encode("utf-8", errors="replace"))
        )
        recorded_omitted = (
            recorded_assembly.get("bytes_omitted")
            if recorded_assembly is not None
            else None
        )
        bytes_omitted = (
            recorded_omitted
            if isinstance(recorded_omitted, int)
            and not isinstance(recorded_omitted, bool)
            and recorded_omitted > 0
            else omitted
        )
        projected["assembly"] = {
            "format": str(result.get("format") or assembly_name or "unknown").removesuffix(".txt"),
            "text": text,
            "size_bytes": size_bytes,
            "truncated": bool(bytes_omitted)
            or bool(recorded_assembly and recorded_assembly.get("truncated") is True),
        }
        if bytes_omitted:
            projected["assembly"]["bytes_omitted"] = bytes_omitted
    error = result.get("error")
    if error:
        projected["error"] = _bounded_text(error, 4000)
    _add_gateway_record_identity(projected, record)
    return projected


def _agent_gateway_failure(
    job: dict[str, Any], record: dict[str, Any] | None, *, generalized: bool
) -> dict[str, Any]:
    error = job.get("error")
    error = error if isinstance(error, dict) else {}
    error_class = str(error.get("error_class") or "unknown")
    infrastructure = error_class == "infra"
    unknown = error_class == "unknown"
    projected = error_response(
        "hidden evaluator case failed"
        if generalized and not infrastructure
        else _bounded_text(error.get("message") or "Gateway job failed"),
        code="gateway_infrastructure" if infrastructure else "gateway_job_failed",
        repairable=not (infrastructure or unknown),
        error_class=error_class,
        reason=_bounded_text(error.get("reason") or "unknown", 128),
        next_action=(
            "The Supervisor's configured retry policy has ended for this request. Do not "
            "change the Kernel to fix this infrastructure failure or start a retry loop. "
            + ESCALATE_RUNTIME
            if infrastructure else
            "The failure cause is unclassified; do not assume a Kernel bug. Preserve the "
            "Gateway Record ID and report the blocker if diagnostics cannot identify the cause."
            if unknown else
            "Inspect the returned diagnostics and correct the Kernel or probe before another "
            "request. Do not repeat the same failing request unchanged; hidden case inputs "
            "remain private."
        ),
    )
    projected["status"] = str(job.get("status") or "failed")
    _add_gateway_record_identity(projected, record)
    return projected


def _reject_duplicate_task(previous_record_id: str | None) -> None:
    raise SystemExit(json.dumps(error_response(
        "sandbox: duplicate Gateway task rejected; " + (
            f"previous gateway_record_id={previous_record_id}"
            if previous_record_id else "the identical task is currently running"
        ),
        code="duplicate_gateway_task", repairable=False,
        next_action=(
            "Read the existing result: python3 tools/sandbox.py --kind record-read "
            f"--record-id {previous_record_id}"
            if previous_record_id else
            "Wait for the original request to complete. Do not launch another identical job."
        ),
        **({"gateway_record_id": previous_record_id} if previous_record_id else {}),
    )))


def _reusable_gateway_record(
    workspace: Path, task_digest: str, record_id: str | None,
) -> dict[str, Any]:
    """Trusted consumers may reuse facts; Agents keep the existing duplicate error."""
    if os.environ.get(REUSE_GATEWAY_RESULTS_ENV) != "1" or record_id is None:
        _reject_duplicate_task(record_id)
    record = _load_gateway_record(workspace, record_id)
    if record["gateway_task_digest"] != task_digest:
        raise RuntimeError("Gateway cache record does not match the requested measurement contract")
    source = (record["record_dir"] / "kernel.py").read_bytes()
    if hashlib.sha256(source).hexdigest() != record["kernel_sha256"]:
        raise RuntimeError("Gateway cache Kernel source does not match its recorded digest")
    return record


def _record_exit_code(record: dict[str, Any]) -> int:
    result = record["result"]
    if result.get("status") in {"failed", "cancelled", "error"} or result.get("error"):
        return 1
    if record["gateway_kind"] == "run":
        return 0 if result.get("all_pass") is True else 1
    if record["gateway_kind"] == "same_allocation_abba":
        return 0 if result.get("correct") is True else 1
    for key in ("all_pass", "correct", "passed", "ok"):
        if result.get(key) is False:
            return 1
    code = result.get("exit_code")
    return code if isinstance(code, int) and not isinstance(code, bool) else 0


def _emit_supervisor_measurement(
    workspace: Path, record_id: str, *, reused: bool,
) -> None:
    """Emit an untrimmed private receipt, never through the Agent HTTP projection."""
    if os.environ.get(REUSE_GATEWAY_RESULTS_ENV) != "1":
        return
    record = _load_gateway_record(workspace, record_id)
    print(SUPERVISOR_MEASUREMENT_PREFIX + json.dumps({
        "gateway_record_id": record_id,
        "gateway_kind": record["gateway_kind"],
        "kernel_sha256": record["kernel_sha256"],
        "kernel_subjects": record["kernel_subjects"],
        "result": record["result"],
        "artifact": str(record["record_dir"] / "result.json"),
        "reused": reused,
    }, ensure_ascii=False, allow_nan=False))


def _gateway_execution_identity(args: argparse.Namespace) -> dict[str, Any]:
    """Measurement-affecting transport/environment, excluding credentials and output paths."""
    return {
        "hardware": args.hardware,
        "url": args.url or os.environ.get("AGATE_URL", ""),
        "profile": args.gateway_profile,
        "ssh": args.ssh,
        "ssh_gpu": args.ssh_gpu,
        "ssh_init": args.ssh_init,
        "ssh_runtime_binds": args.ssh_runtime_bind,
        "environment": _parse_env_items(args.env),
        "requirements": args.requirement,
        "deps_mode": args.deps_mode,
    }


def _comparison_command(command: list[str], *, sol: bool = False) -> list[str]:
    """Version labels/output flags do not identify a measurement; seeds/iters do."""
    cleaned = ["python3", "test_kernel.py", "--no-memory"]
    skip = False
    for token in command[2:]:
        if skip:
            skip = False
        elif token in {"--version", "--multi-seed", "--timed-runs", "--shape-id"}:
            skip = True
        elif token == "--no-memory" or any(
            token.startswith(option + "=")
            for option in ("--version", "--multi-seed", "--timed-runs", "--shape-id")
        ):
            continue
        else:
            cleaned.append(token)
    options = [("--multi-seed", 0)]
    if not sol:
        options.append(("--timed-runs", 100))
    elif _option_values(command, "--timed-runs"):
        raise ValueError("SOL's evaluator does not support --timed-runs")
    for option, default in options:
        cleaned += [option, str(int(_option_value(command, option, default)))]
    return cleaned


def _comparison_input_identity(staged: Path) -> dict[str, str]:
    paths = {
        filename: staged / filename
        for filename in _evaluation_input_paths(staged)
        if filename != "kernel.py" and (staged / filename).is_file()
    }
    paths.update(evaluation_inputs(staged))
    paths["remote_abba.py"] = REPO_ROOT / "long_horizon" / "remote_abba.py"
    runtime = _private_atrex_bench_runtime()
    if runtime is not None:
        for path in runtime.rglob("*"):
            relative = path.relative_to(runtime)
            if path.is_file() and not any(
                part in {".git", "__pycache__"} for part in relative.parts
            ) and path.suffix not in {".pyc", ".pyo"}:
                paths[f"evaluator/{relative.as_posix()}"] = path
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}


def _agent_dev_probe_result(result: dict[str, Any]) -> dict[str, Any]:
    """Project an arbitrary Dev command without pretending it is an evaluation."""
    projected: dict[str, Any] = {}
    status = result.get("status")
    if isinstance(status, str):
        projected["status"] = _bounded_text(status, 64)
    if isinstance(result.get("error"), dict):
        failure = _agent_gateway_failure(result, None, generalized=False)
        projected.update({key: failure[key] for key in ("ok", "repairable", "error")})
    command = result.get("command")
    if isinstance(command, str):
        projected["command"] = _bounded_text(command, 4096)
    exit_code = result.get("exit_code")
    if isinstance(exit_code, int) and not isinstance(exit_code, bool):
        projected["exit_code"] = exit_code
    for key, limit in (
        ("stdout", MAX_AGENT_DEV_STDOUT_BYTES),
        ("stderr", MAX_AGENT_DEV_STDERR_BYTES),
    ):
        value = result.get(key)
        if not isinstance(value, str):
            continue
        text, omitted = _bounded_utf8_text(value, limit, marker=f"Dev {key}")
        projected[key] = text
        if omitted:
            truncated = projected.setdefault("truncated", {})
            if isinstance(truncated, dict):
                truncated[f"{key}_bytes_omitted"] = omitted
    synced_paths = result.get("synced_paths")
    if isinstance(synced_paths, list):
        projected["synced_paths"] = [
            _bounded_text(value, 512)
            for value in synced_paths[:64]
            if isinstance(value, str)
        ]
    return projected


def _gateway_record_kernel_subjects(record: dict[str, Any]) -> dict[str, str]:
    candidate = record["kernel_sha256"]
    raw_subjects = record.get("kernel_subjects")
    subjects = raw_subjects if isinstance(raw_subjects, dict) else {}
    incumbent = _valid_kernel_sha256(subjects.get("incumbent"))
    recorded_candidate = _valid_kernel_sha256(subjects.get("candidate"))
    if recorded_candidate is not None and recorded_candidate != candidate:
        raise ValueError("ABBA Candidate digest disagrees with its Gateway record")
    if incumbent is None and record.get("gateway_kind") == "same_allocation_abba":
        raw_path = record["record_dir"] / "raw-result.json"
        if raw_path.is_file() and not raw_path.is_symlink():
            try:
                raw = _read_json_file(raw_path, max_bytes=64 * 1024 * 1024)
            except ValueError:
                raw = {}
            incumbent = _valid_kernel_sha256(raw.get("baseline_sha256"))
    if incumbent is None:
        raise ValueError("ABBA Gateway record omitted its Incumbent Kernel digest")
    return {"incumbent": incumbent, "candidate": recorded_candidate or candidate}


def _gateway_record_public_result(
    record_id: str,
    record: dict[str, Any],
    *,
    generalized: bool = False,
) -> dict[str, Any]:
    """Return the same bounded semantic result family Agent received when it was recorded."""
    gateway_kind = record["gateway_kind"]
    stored = record["result"]
    if stored.get("status") in {"failed", "cancelled", "error"} and isinstance(
        stored.get("error"), dict
    ):
        operation = "evaluate" if gateway_kind == "run" else gateway_kind
        projected = _agent_gateway_failure(stored, None, generalized=generalized)
        if gateway_kind == "dev":
            projected.update(_agent_dev_probe_result(stored))
    elif gateway_kind == "run":
        operation = "evaluate"
        projected = (
            _agent_gateway_failure(stored, None, generalized=generalized)
            if stored.get("status") == "failed" and "error" in stored
            else _agent_evaluation_result(stored, None)
        )
    elif gateway_kind == "same_allocation_abba":
        operation = gateway_kind
        projected = _agent_abba_result(stored)
    elif gateway_kind == "profile":
        operation = gateway_kind
        projected = _agent_profile_result(stored, None)
    elif gateway_kind == "dev":
        operation = gateway_kind
        projected = (
            _agent_evaluation_result(stored, None)
            if "all_pass" in stored
            else _agent_dev_probe_result(stored)
        )
    elif gateway_kind == "check":
        operation = gateway_kind
        projected = _agent_check_result(stored, None)
    elif gateway_kind == "disassemble":
        operation = gateway_kind
        projected = _agent_disassembly_result(stored, None)
    else:
        raise ValueError(f"Gateway record kind is not Agent-readable: {gateway_kind}")

    raw_status = projected.pop("status", stored.get("status", None))
    projected.pop("operation", None)
    projected.pop("gateway_record_id", None)
    projected.pop("kernel_sha256", None)
    projected.pop("kernel_artifact_digest", None)
    projected.pop("kernel_trial_id", None)
    projected.pop("kernel_id", None)
    status = (
        "failed"
        if isinstance(raw_status, str)
        and raw_status in {"failed", "error", "cancelled", "canceled"}
        else "completed"
    )
    public: dict[str, Any] = {
        "record_type": "gateway_result",
        "gateway_record_id": record_id,
        "operation": operation,
        "status": status,
        "kernel_id": record["kernel_id"],
        "result": projected,
    }
    if gateway_kind == "same_allocation_abba":
        public["kernels"] = {
            name: {"kernel_id": kernel_id}
            for name, kernel_id in record["kernel_subject_ids"].items()
        }
    return public


def _read_gateway_record(workspace: Path, record_id: str) -> int:
    try:
        record = _load_gateway_record(workspace, record_id)
        public = _gateway_record_public_result(
            record_id,
            record,
            generalized=_is_generalized_workspace(workspace),
        )
    except ValueError as exc:
        raise SystemExit(f"sandbox: cannot read Gateway record: {exc}") from exc
    print(
        RECORD_RESULT_PREFIX
        + json.dumps(public, ensure_ascii=False, allow_nan=False)
    )
    return 0


def _visible_gateway_records(workspace: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in _gateway_record_roots(workspace):
        if root.is_symlink() or not root.is_dir():
            continue
        for path in sorted(root.iterdir(), key=lambda item: item.name):
            if (
                not path.is_dir()
                or path.is_symlink()
                or GATEWAY_RECORD_ID_RE.fullmatch(path.name) is None
                or path.name in seen
            ):
                continue
            try:
                records.append(_load_gateway_record(workspace, path.name))
            except ValueError:
                continue
            seen.add(path.name)
    return records


def _gateway_record_index_entry(
    record: dict[str, Any], *, role: str
) -> dict[str, Any]:
    public = _gateway_record_public_result(record["record_id"], record)
    return {
        "gateway_record_id": record["record_id"],
        "operation": public["operation"],
        "status": public["status"],
        "role": role,
    }


def _kernel_gateway_records(
    workspace: Path, kernel_id: str
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for record in _visible_gateway_records(workspace):
        if record["gateway_kind"] == "same_allocation_abba":
            roles = [
                role
                for role, subject_id in record["kernel_subject_ids"].items()
                if subject_id == kernel_id
            ]
        elif record["kernel_id"] == kernel_id:
            roles = ["workspace_snapshot" if record["gateway_kind"] == "dev" else "subject"]
        else:
            roles = []
        results.extend(
            _gateway_record_index_entry(record, role=role) for role in roles
        )
    return results


def _kernel_artifact_bytes(workspace: Path, kernel_id: str) -> bytes:
    identity = _load_kernel_identity(workspace, kernel_id)
    digest = identity["kernel_artifact_digest"]
    hexadecimal = digest.removeprefix("sha256:")
    path = next(
        (
            candidate
            for root in _kernel_artifact_roots(workspace)
            if (candidate := root / hexadecimal / "kernel.py").is_file()
            and not candidate.is_symlink()
        ),
        _kernel_artifact_root(workspace) / hexadecimal / "kernel.py",
    )
    if not path.is_file() or path.is_symlink():
        # Backward-compatible lookup for records created before the content store existed.
        record = next(
            (
                item
                for item in _visible_gateway_records(workspace)
                if item["kernel_id"] == kernel_id
            ),
            None,
        )
        if record is None:
            raise ValueError("Kernel Artifact does not exist in this Agent workspace")
        path = record["record_dir"] / "kernel.py"
        if not path.is_file() or path.is_symlink():
            raise ValueError("Kernel Artifact source is missing")
    try:
        source = path.read_bytes()
    except OSError as exc:
        raise ValueError("Kernel Artifact source cannot be read") from exc
    if _kernel_artifact_digest(source) != digest:
        raise ValueError("Kernel Artifact source failed digest verification")
    return source


def _scratch_output_path(workspace: Path, value: str) -> tuple[Path, str]:
    relative = PurePosixPath(value)
    if relative.is_absolute() or not relative.parts or relative.parts[0] != "scratch":
        raise ValueError("--output-path must be a workspace-relative path under scratch/")
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("--output-path contains an unsafe path component")
    target = workspace.joinpath(*relative.parts)
    scratch = workspace / "scratch"
    if scratch.exists() and (scratch.is_symlink() or not scratch.is_dir()):
        raise ValueError("workspace scratch path is unsafe")
    scratch.mkdir(mode=0o700, exist_ok=True)
    current = scratch
    for part in relative.parts[1:-1]:
        current = current / part
        if current.exists() and (current.is_symlink() or not current.is_dir()):
            raise ValueError("--output-path traverses an unsafe directory")
        current.mkdir(mode=0o700, exist_ok=True)
    if target.exists() and (target.is_symlink() or not target.is_file()):
        raise ValueError("--output-path names an unsafe existing path")
    return target, relative.as_posix()


def _read_kernel_source(workspace: Path, kernel_id: str, output_path: str) -> int:
    try:
        source = _kernel_artifact_bytes(workspace, kernel_id)
        destination, public_path = _scratch_output_path(workspace, output_path)
    except ValueError as exc:
        raise SystemExit(f"sandbox: cannot read Kernel source: {exc}") from exc
    temporary = destination.parent / f".{destination.name}.{os.getpid()}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(source)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        fsync_directory(destination.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    public = {
        "record_type": "kernel_source",
        "status": "written",
        "kernel_id": kernel_id,
        "file": public_path,
        "size_bytes": len(source),
    }
    print(RECORD_RESULT_PREFIX + json.dumps(public, ensure_ascii=False))
    return 0


def _read_kernel_gateway_records(workspace: Path, kernel_id: str) -> int:
    try:
        _load_kernel_identity(workspace, kernel_id)
    except ValueError as exc:
        raise SystemExit(f"sandbox: cannot read Kernel Gateway records: {exc}") from exc
    public = {
        "record_type": "kernel_gateway_records",
        "kernel_id": kernel_id,
        "gateway_records": _kernel_gateway_records(workspace, kernel_id),
    }
    print(RECORD_RESULT_PREFIX + json.dumps(public, ensure_ascii=False))
    return 0


def _agent_retry_notes(stderr: str) -> str:
    """Keep retry decisions, not transport polling chatter."""
    retained: list[str] = []
    for line in stderr.splitlines():
        if "[sandbox] Gateway" not in line and "[sandbox] gateway" not in line:
            continue
        normalized = _bounded_text(" ".join(line.split()), 500)
        if normalized and normalized not in retained:
            retained.append(normalized)
        if len(retained) >= 8:
            break
    return "\n".join(retained)


def _record_result_lines(
    workspace: Path, stdout: str, *, gateway_kind: str, task: _GatewayTask | None = None,
    job: dict[str, Any] | None = None,
) -> str:
    """Record and compact ordinary RESULT_JSON emitted by a dev evaluator."""
    lines = stdout.splitlines()
    for index in range(len(lines) - 1, -1, -1):
        line = lines[index]
        if not line.startswith(TEST_RESULT_PREFIX):
            continue
        try:
            result = json.loads(line[len(TEST_RESULT_PREFIX) :])
        except json.JSONDecodeError:
            return stdout
        if isinstance(result, dict):
            record = (
                task.record(result, gateway_kind=gateway_kind, private_result=job,
                            job_id=job.get("job_id") if job else None)
                if task is not None else
                _record_episode_evaluation(workspace, result, gateway_kind=gateway_kind)
            )
            lines[index] = TEST_RESULT_PREFIX + json.dumps(
                _agent_evaluation_result(result, record),
                ensure_ascii=False,
                allow_nan=False,
            )
        return "\n".join(lines)
    return stdout


def _positive_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number > 0.0 and math.isfinite(number) else None


def _with_workspace_performance_score(workspace: Path, result: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy per-shape results to the shared arithmetic-mean score."""
    if result.get("performance_objective") == PERFORMANCE_OBJECTIVE:
        return result
    candidate_by_shape = result.get("latency_us_by_shape")
    if not isinstance(candidate_by_shape, dict) or not candidate_by_shape:
        return result
    try:
        baseline = _json_object(workspace / "memory" / "v0.json") or {}
    except ValueError:
        return result
    performance = baseline.get("performance")
    performance = performance if isinstance(performance, dict) else {}
    baseline_by_shape = performance.get("latency_us_by_shape")
    if not isinstance(baseline_by_shape, dict) or set(candidate_by_shape) != set(baseline_by_shape):
        return result
    speedups: list[float] = []
    for shape_id, raw_candidate in candidate_by_shape.items():
        candidate = _positive_number(raw_candidate)
        reference = _positive_number(baseline_by_shape.get(shape_id))
        if candidate is None or reference is None:
            return result
        speedups.append(reference / candidate)
    if not speedups:
        return result
    score = sum(speedups) / len(speedups)
    hydrated = dict(result)
    hydrated["speedup_vs_ref_mean"] = score
    hydrated["speedup_vs_ref_geomean"] = None
    hydrated["performance_score"] = score
    hydrated["performance_objective"] = PERFORMANCE_OBJECTIVE
    return hydrated


def _hydrate_result_lines(workspace: Path, stdout: str) -> str:
    """Return ordinary dev evaluator output with the shared score contract."""
    normalized: list[str] = []
    for line in stdout.splitlines():
        if not line.startswith(TEST_RESULT_PREFIX):
            normalized.append(line)
            continue
        try:
            result = json.loads(line[len(TEST_RESULT_PREFIX) :])
        except json.JSONDecodeError:
            normalized.append(line)
            continue
        if not isinstance(result, dict):
            normalized.append(line)
            continue
        normalized.append(
            TEST_RESULT_PREFIX
            + json.dumps(
                _with_workspace_performance_score(workspace, result),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    return "\n".join(normalized)


def _hydrate_abba_result_lines(workspace: Path, stdout: str) -> str:
    """Normalize candidate-only ABBA run results for long-lived supervisors."""
    normalized: list[str] = []
    for line in stdout.splitlines():
        if not line.startswith(ABBA_RESULT_PREFIX):
            normalized.append(line)
            continue
        try:
            payload = json.loads(line[len(ABBA_RESULT_PREFIX) :])
        except json.JSONDecodeError:
            normalized.append(line)
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("runs"), list):
            normalized.append(line)
            continue
        changed = False
        runs: list[object] = []
        for row in payload["runs"]:
            if not isinstance(row, dict) or not isinstance(row.get("result"), dict):
                runs.append(row)
                continue
            result = _with_workspace_performance_score(workspace, row["result"])
            if result is row["result"]:
                runs.append(row)
                continue
            updated = dict(row)
            updated["result"] = result
            runs.append(updated)
            changed = True
        if not changed:
            normalized.append(line)
            continue
        updated_payload = dict(payload)
        updated_payload["runs"] = runs
        normalized.append(
            ABBA_RESULT_PREFIX + json.dumps(updated_payload, ensure_ascii=False, allow_nan=False)
        )
    return "\n".join(normalized)


def _record_profile_job(
    job: dict[str, Any],
    workspace: Path,
    sync_paths: list[str],
    public_result: dict[str, Any],
) -> None:
    """Persist only the public Profile projection in the Agent workspace."""
    for relative in sync_paths:
        path = PurePosixPath(relative)
        if not path.parts or path.parts[0] != "scratch":
            continue
        output_dir = workspace / path.as_posix()
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / "gateway_profile.json"
        target.write_text(
            json.dumps(
                {"status": job.get("status"), "result": public_result},
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        for artifact in (job.get("result") or {}).get("artifacts") or []:
            if not isinstance(artifact, dict):
                continue
            try:
                artifact_name = _safe_relative(str(artifact.get("name") or ""))
            except ValueError as exc:
                raise RuntimeError(f"invalid profile artifact name: {exc}") from exc
            _download_oss_artifact(artifact, output_dir / artifact_name)


def _copy_evaluation_workspace(source: Path, destination: Path) -> None:
    """Stage only evaluator inputs for a Supervisor-private ABBA workspace."""
    for relative in _evaluation_input_paths(source):
        src = source / relative
        if not src.is_file() or src.is_symlink():
            continue
        dst = destination / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    for relative, src in evaluation_inputs(source).items():
        shutil.copy2(src, destination / relative)
    baseline_memory = source / "memory" / "v0.json"
    if baseline_memory.is_file() and not baseline_memory.is_symlink():
        target = destination / "memory" / "v0.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(baseline_memory, target)


def _abba_side_metrics(
    rows: list[dict[str, Any]], side: str, repeats: int, shape_ids: list[str]
) -> dict[str, Any]:
    selected = [row for row in rows if row.get("revision") == side]
    successful = [
        row
        for row in selected
        if row.get("exit_code") == 0
        and isinstance(row.get("result"), dict)
        and row["result"].get("all_pass") is True
    ]
    by_shape: dict[str, float] = {}
    for shape_id in shape_ids:
        values = [
            _positive_number(row["result"].get("latency_us_by_shape", {}).get(shape_id))
            for row in successful
            if isinstance(row["result"].get("latency_us_by_shape"), dict)
        ]
        numbers = [value for value in values if value is not None]
        if len(numbers) == repeats:
            by_shape[shape_id] = math.exp(sum(math.log(value) for value in numbers) / len(numbers))
    run_latencies = [
        _positive_number(row["result"].get("latency_us_geomean")) for row in successful
    ]
    valid_run_latencies = [value for value in run_latencies if value is not None]
    latency = (
        math.exp(sum(math.log(value) for value in valid_run_latencies) / len(valid_run_latencies))
        if len(valid_run_latencies) == repeats
        else None
    )
    scores = [_positive_number(row["result"].get("performance_score")) for row in successful]
    score = (
        sum(value for value in scores if value is not None) / repeats
        if len(scores) == repeats and all(value is not None for value in scores)
        else None
    )
    return {
        "correct": (
            len(successful) == repeats and set(by_shape) == set(shape_ids)
            and latency is not None
        ),
        "latency_us_geomean": latency,
        "latency_us_by_shape": by_shape,
        "performance_score": score,
        "max_abs_err": max(
            (row["result"].get("max_abs_err") or 0 for row in successful), default=0,
        ),
        "max_rel_err": max(
            (row["result"].get("max_rel_err") or 0 for row in successful), default=0,
        ),
    }


def _agent_abba_public_result(
    payload: dict[str, Any],
    schedule: list[dict[str, int | str]],
    shape_ids: list[str],
    repeats: int,
) -> dict[str, Any]:
    rows = payload.get("runs")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("ABBA result omitted its measurement rows")
    if [{"revision": row.get("revision"), "repeat": row.get("repeat")} for row in rows] != schedule:
        raise ValueError("ABBA result did not execute the exact requested schedule")
    baseline = _abba_side_metrics(rows, "incumbent", repeats, shape_ids)
    candidate = _abba_side_metrics(rows, "candidate", repeats, shape_ids)
    baseline_us = _positive_number(baseline.get("latency_us_geomean"))
    candidate_us = _positive_number(candidate.get("latency_us_geomean"))
    correct = (
        baseline["correct"] is True and candidate["correct"] is True and not payload.get("error")
    )
    speedup = baseline_us / candidate_us if correct and baseline_us and candidate_us else None
    return {
        "status": "succeeded" if correct and payload.get("error") is None else "failed",
        "operation": "same_allocation_abba",
        "correct": correct,
        "comparison": {"method": "abba", "repeats": repeats},
        "schedule": [
            {
                "side": "A" if step["revision"] == "incumbent" else "B",
                "repeat": step["repeat"],
            }
            for step in schedule
        ],
        "baseline": baseline,
        "candidate": candidate,
        "speedup": speedup,
        "improvement_pct": (speedup - 1.0) * 100.0 if speedup is not None else None,
        "measurements": [
            {
                "side": "A" if row.get("revision") == "incumbent" else "B",
                "repeat": row.get("repeat"),
                "correct": isinstance(row.get("result"), dict)
                and row["result"].get("all_pass") is True,
                "latency_us_geomean": (
                    row["result"].get("latency_us_geomean")
                    if isinstance(row.get("result"), dict)
                    else None
                ),
                "latency_us_by_shape": (
                    row["result"].get("latency_us_by_shape", {})
                    if isinstance(row.get("result"), dict)
                    else {}
                ),
            }
            for row in rows
        ],
        "shape_batch_count": payload.get("shape_batch_count", 1),
        "error": _bounded_text(payload.get("error"), 1000) if payload.get("error") else None,
    }


def _aggregate_agent_abba_results(
    samples: list[dict[str, Any]], shape_ids: list[str], *, metadata: object = None,
) -> dict[str, Any]:
    if len(samples) != MEASUREMENT_REPETITIONS:
        raise RuntimeError(f"ABBA aggregation requires {MEASUREMENT_REPETITIONS} measurements")
    failed = next((sample for sample in samples if sample.get("correct") is not True), None)
    updated = dict(failed if failed is not None else samples[-1])
    if failed is None:
        for side in ("baseline", "candidate"):
            side_samples = [
                _latency_us_by_shape(sample.get(side)) for sample in samples
            ]
            selected_side = updated.get(side)
            if not isinstance(selected_side, dict):
                raise RuntimeError(f"ABBA measurement omitted {side} result")
            updated[side] = _replace_shape_latencies(
                selected_side,
                _median_latency_by_shape(side_samples, shape_ids),
            )
            score, _ = _metadata_speedup_mean(
                metadata, shape_ids, _latency_us_by_shape(updated[side]),
            )
            if score is None and metadata is None:
                # SOL supplies its reference-based score, not Atrex-Bench production metadata.
                scores = [
                    _positive_number(sample[side].get("performance_score")) for sample in samples
                ]
                if all(value is not None for value in scores):
                    score = sum(value for value in scores if value is not None) / len(scores)
            updated[side]["performance_score"] = score
            if score is not None:
                updated[side]["performance_objective"] = PERFORMANCE_OBJECTIVE
            for key in ("max_abs_err", "max_rel_err"):
                updated[side][key] = max(sample[side].get(key) or 0 for sample in samples)
    baseline = updated.get("baseline")
    candidate = updated.get("candidate")
    baseline_us = (
        _positive_number(baseline.get("latency_us_geomean")) if isinstance(baseline, dict) else None
    )
    candidate_us = (
        _positive_number(candidate.get("latency_us_geomean"))
        if isinstance(candidate, dict)
        else None
    )
    speedup = baseline_us / candidate_us if baseline_us and candidate_us else None
    updated["speedup"] = speedup
    updated["improvement_pct"] = (speedup - 1.0) * 100.0 if speedup is not None else None
    updated["measurements"] = [
        {
            "repetition": ordinal,
            "baseline": _latency_us_by_shape(sample.get("baseline")),
            "candidate": _latency_us_by_shape(sample.get("candidate")),
        }
        for ordinal, sample in enumerate(samples, start=1)
    ]
    updated["measurement_aggregation"] = _measurement_aggregation_summary()
    return updated


def _agent_abba_result(result: dict[str, Any]) -> dict[str, Any]:
    """Hide repetition and batching mechanics from the Agent-facing ABBA result."""
    return {
        key: value
        for key, value in result.items()
        if key not in {"measurements", "shape_batch_count", "measurement_aggregation"}
    }


def _run_agent_abba(
    args: argparse.Namespace,
    workspace: Path,
    command_parts: list[str],
    queue_wait_grace: int,
) -> int:
    """Shared A/B measurement service; callers never acquire promotion authority."""
    from long_horizon.verifier import (
        _merge_batch_payloads,
        _payload_from_stdout,
        verification_schedule,
    )

    if args.evaluation_mode == "correctness_only":
        raise SystemExit("sandbox: ABBA comparison requires --mode full")
    if not 1 <= args.comparison_repeats <= 20:
        raise SystemExit("sandbox: --comparison-repeats must be in 1..20")
    baseline_source = _read_workspace_override(
        workspace,
        args.baseline_path,
        field="baseline-path",
        max_bytes=MAX_KERNEL_SOURCE_BYTES,
    )
    candidate_source = (workspace / "kernel.py").read_bytes().decode("utf-8")
    schedule = verification_schedule(args.comparison_repeats)
    per_run_timeout = int(os.environ.get(COMPARISON_RUN_TIMEOUT_ENV, min(120, args.timeout)))
    if per_run_timeout <= 0:
        raise ValueError("ABBA per-run timeout must be positive")
    if per_run_timeout * len(schedule) + 30 > args.timeout:
        raise SystemExit(
            "sandbox: ABBA schedule does not fit the configured allocation timeout; "
            "reduce --comparison-repeats or increase the Supervisor timeout"
        )
    reference_root = _private_reference_dir(workspace) or workspace
    sol = (workspace / "workload.jsonl").is_file()
    if sol:
        if args.evaluation_input_path or args.evaluation_shapes_path:
            raise ValueError(
                "SOL ABBA uses the complete canonical workload, not custom input overrides"
            )
        workloads = [
            json.loads(line) for line in (workspace / "workload.jsonl").read_text().splitlines()
            if line.strip()
        ]
        if any(not isinstance(row, dict) or not isinstance(row.get("uuid"), str)
               or not row["uuid"] for row in workloads):
            raise ValueError("SOL workloads require non-empty UUIDs")
        shapes = {row["uuid"]: {} for row in workloads}
        if len(shapes) != len(workloads):
            raise ValueError("SOL workloads require distinct non-empty UUIDs")
    else:
        shapes = _json_object(reference_root / "shapes.json", required=True)
        assert shapes is not None
    if not shapes:
        raise SystemExit("sandbox: ABBA evaluator Shape contract is empty")
    requested = _option_values(command_parts, "--shape-id")
    if sol and requested:
        raise ValueError("SOL ABBA requires the complete workload")
    if requested and not args.evaluation_shapes_path:
        unknown = [shape_id for shape_id in requested if shape_id not in shapes]
        if unknown:
            raise SystemExit("sandbox: unknown --shape-id values: " + ", ".join(unknown))
        shape_ids = list(dict.fromkeys(requested))
    elif not requested:
        shape_ids = sorted((str(value) for value in shapes), key=_sort_shape_id)
    else:
        shape_ids = []
    evaluator_command = _command_parts(command_parts) or [
        "python3",
        "test_kernel.py",
        "--version",
        "vlong",
        "--no-memory",
    ]
    if not _is_test_kernel_command(evaluator_command):
        raise SystemExit("sandbox: ABBA requires the canonical test_kernel.py evaluator command")
    cleaned_command: list[str] = []
    skip_next = False
    for token in evaluator_command:
        if skip_next:
            skip_next = False
            continue
        if token == "--shape-id":
            skip_next = True
            continue
        if token.startswith("--shape-id="):
            continue
        cleaned_command.append(token)
    evaluator_command = _comparison_command(cleaned_command, sol=sol)
    with tempfile.TemporaryDirectory(prefix="atrex-agent-abba-") as temporary:
        staged = Path(temporary) / workspace.name
        staged.mkdir()
        _copy_evaluation_workspace(workspace, staged)
        for filename in (
            "reference.py",
            "input.py",
            "shapes.json",
            "metadata.json",
            "roofline.json",
        ):
            source = reference_root / filename
            if source.is_file():
                shutil.copy2(source, staged / filename)
        if args.evaluation_input_path:
            (staged / "input.py").write_text(
                _read_workspace_override(
                    workspace,
                    args.evaluation_input_path,
                    field="input-path",
                    max_bytes=MAX_CUSTOM_INPUT_SOURCE_BYTES,
                ),
                encoding="utf-8",
            )
        if args.evaluation_shapes_path:
            custom_shapes = _read_workspace_override(
                workspace,
                args.evaluation_shapes_path,
                field="shapes-path",
                max_bytes=MAX_CUSTOM_SHAPES_BYTES,
            )
            parsed_shapes = json.loads(custom_shapes)
            if not isinstance(parsed_shapes, dict) or not parsed_shapes:
                raise SystemExit("sandbox: --shapes-path must contain a non-empty object")
            (staged / "shapes.json").write_text(custom_shapes, encoding="utf-8")
            if requested:
                unknown = [shape_id for shape_id in requested if shape_id not in parsed_shapes]
                if unknown:
                    raise SystemExit(
                        "sandbox: unknown custom --shape-id values: " + ", ".join(unknown)
                    )
                shape_ids = list(dict.fromkeys(requested))
            else:
                shape_ids = sorted((str(value) for value in parsed_shapes), key=_sort_shape_id)
        control = staged / "verification_artifacts" / "agent-comparison"
        control.mkdir(parents=True)
        driver = control / "test_kernel.py"
        shutil.copy2(REPO_ROOT / "long_horizon" / "remote_abba.py", driver)
        snapshots = control / "snapshots"
        snapshots.mkdir()
        (snapshots / "baseline.py").write_text(baseline_source, encoding="utf-8")
        (snapshots / "candidate.py").write_text(candidate_source, encoding="utf-8")
        manifests = {
            "incumbent": {"kernel.py": "snapshots/baseline.py"},
            "candidate": {"kernel.py": "snapshots/candidate.py"},
        }
        baseline_sha256 = hashlib.sha256(baseline_source.encode()).hexdigest()
        candidate_sha256 = hashlib.sha256(candidate_source.encode()).hexdigest()
        evaluation_inputs = _comparison_input_identity(staged)
        task_digest = _gateway_task_digest(
            "same_allocation_abba",
            candidate_sha256,
            {
                "execution": _gateway_execution_identity(args),
                "shape_ids": shape_ids,
                "schedule": schedule,
                "command": evaluator_command,
                "evaluation_inputs": evaluation_inputs,
                "per_run_timeout": per_run_timeout,
                "allocation_timeout": args.timeout,
            },
            baseline_sha256=baseline_sha256,
        )
        task_owner, previous_record_id = _reserve_gateway_task(workspace, task_digest)
        if task_owner is None:
            previous = _reusable_gateway_record(workspace, task_digest, previous_record_id)
            _emit_supervisor_measurement(workspace, previous["record_id"], reused=True)
            return 0 if previous["result"].get("correct") is True else 1

        from supervisor.abba_checkpoints import AbbaBatchStore, validate_batch

        checkpoints = AbbaBatchStore(
            _supervisor_evaluations_path(workspace).parent, _historical_evidence_roots(),
        )

        def run_batch(item: tuple[str, int, list[str]]) -> dict[str, Any]:
            phase, index, batch = item
            identity = {
                "checkpoint_contract": 1,
                "comparison_task_digest": task_digest,
                "measurement_repetition": phase,
                "batch_index": index,
                "shape_ids": batch,
                "schedule": schedule,
            }
            cached = checkpoints.load(identity)
            if cached is not None:
                print(
                    f"[sandbox] ABBA {phase} batch {index + 1}: reusing completed measurement",
                    file=sys.stderr,
                )
                return cached
            request_path = control / f"request-{phase}-{index:04d}.json"
            result_path = control / f"result-{phase}-{index:04d}.json"
            command = list(evaluator_command)
            for shape_id in ([] if sol else batch):
                command += ["--shape-id", shape_id]
            request_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "schedule": schedule,
                        "manifests": manifests,
                        "command": command,
                        "run_timeout_seconds": per_run_timeout,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            nested = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--workspace",
                str(staged),
                "--hardware",
                args.hardware,
                "--timeout",
                str(args.timeout),
                "--kind",
                "dev",
                "--no-sync",
            ]
            if args.url:
                nested += ["--url", args.url]
            elif args.gateway_profile:
                nested += ["--gateway-profile", args.gateway_profile]
            elif args.ssh:
                nested += ["--ssh", args.ssh]
                if args.ssh_init:
                    nested += ["--ssh-init", args.ssh_init]
                if args.ssh_gpu is not None:
                    nested += ["--ssh-gpu", str(args.ssh_gpu)]
                if args.health_command:
                    nested += ["--health-command", args.health_command]
                for bind in args.ssh_runtime_bind or []:
                    nested += ["--ssh-runtime-bind", bind]
            for item in args.env:
                nested += ["--env", item]
            for requirement in args.requirement or []:
                nested += ["--requirement", requirement]
            if args.deps_mode:
                nested += ["--deps-mode", args.deps_mode]
            nested += [
                "--",
                "python3",
                "verification_artifacts/agent-comparison/test_kernel.py",
                f"verification_artifacts/agent-comparison/{request_path.name}",
                f"verification_artifacts/agent-comparison/{result_path.name}",
            ]
            environment = os.environ.copy()
            environment.pop("ATREX_AKA_RUNTIME_URL", None)
            environment.pop("ATREX_AKA_RUNTIME_TOKEN", None)
            environment.pop(REUSE_GATEWAY_RESULTS_ENV, None)
            # This is one constituent measurement of the outer reserved ABBA task.
            environment[INTERNAL_MEASUREMENT_ENV] = "1"
            environment.pop(COMPARISON_RUN_TIMEOUT_ENV, None)
            # The staging tree already contains the exact private/custom input contract.
            environment.pop(PRIVATE_REFERENCE_ENV, None)
            completed = subprocess.run(
                nested,
                cwd=staged,
                env=environment,
                capture_output=True,
                text=True,
                timeout=args.timeout + queue_wait_grace + 120,
                check=False,
            )
            if completed.returncode:
                raise RuntimeError(
                    "ABBA batch failed: "
                    + _bounded_text(completed.stderr or completed.stdout, 3000)
                )
            payload = validate_batch(_payload_from_stdout(completed.stdout), schedule, batch)
            return checkpoints.save(
                identity, payload, stdout=completed.stdout, stderr=completed.stderr,
            )

        def execute(selected_shape_ids: list[str], phase: str) -> dict[str, Any]:
            selected_batches = (
                [selected_shape_ids] if sol else _shape_batches(selected_shape_ids, 1)
            )
            with ThreadPoolExecutor(
                max_workers=1 if args.ssh else min(DEFAULT_ABBA_BATCH_WORKERS, len(selected_batches))
            ) as executor:
                batches = list(
                    executor.map(
                        run_batch,
                        [(phase, index, batch) for index, batch in enumerate(selected_batches)],
                    )
                )
            return {
                "batches": batches,
                "payload": _merge_batch_payloads(
                    [batch["payload"] for batch in batches], schedule, selected_shape_ids,
                ),
            }

        try:
            repetition_checkpoints = [
                execute(shape_ids, f"measurement-{ordinal}")
                for ordinal in range(1, MEASUREMENT_REPETITIONS + 1)
            ]
            raw_repetitions = [item["payload"] for item in repetition_checkpoints]
            public_repetitions = [
                _agent_abba_public_result(
                    payload,
                    schedule,
                    shape_ids,
                    args.comparison_repeats,
                )
                for payload in raw_repetitions
            ]
            public = _aggregate_agent_abba_results(
                public_repetitions, shape_ids, metadata=_json_object(staged / "metadata.json"),
            )
        except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
            _abandon_gateway_task(workspace, task_digest, task_owner)
            raise SystemExit(
                json.dumps(error_response(
                    "ABBA comparison failed; no comparison result was recorded: "
                    + _bounded_text(exc, 1500),
                    code="abba_comparison_unavailable", repairable=False,
                    next_action=(
                        "Do not interpret a missing comparison as a Kernel regression. "
                        "Completed Shape batches are saved; retrying the unchanged comparison "
                        "will reuse them and submit only missing batches. "
                        + ESCALATE_RUNTIME
                    ),
                ))
            ) from exc
        except BaseException:
            _abandon_gateway_task(workspace, task_digest, task_owner)
            raise
    try:
        baseline_kernel = _store_kernel_artifact(
            workspace, baseline_source.encode("utf-8")
        )
        agent_public = _agent_abba_result(public)
        record = _record_episode_evaluation(
            workspace,
            agent_public,
            gateway_kind="same_allocation_abba",
            private_result={
                "measurement_payloads": raw_repetitions,
                "measurement_results": public_repetitions,
                "physical_batches": [
                    batch
                    for repetition in repetition_checkpoints
                    for batch in repetition["batches"]
                ],
                "baseline_source": baseline_source,
                "baseline_sha256": baseline_sha256,
            },
            gateway_task_digest=task_digest,
            measurement_repetitions=public_repetitions,
            kernel_subjects={
                "incumbent": baseline_sha256,
                "candidate": candidate_sha256,
            },
            kernel_bytes=candidate_source.encode("utf-8"),
        )
        if record is None:
            raise RuntimeError("ABBA result could not preserve the measured Kernel")
        _complete_gateway_task(workspace, task_digest, task_owner, record["record_id"])
    except (OSError, RuntimeError, ValueError) as exc:
        _abandon_gateway_task(workspace, task_digest, task_owner)
        raise SystemExit(
            json.dumps(error_response(
                "ABBA comparison result could not be recorded.",
                code="gateway_record_unavailable", repairable=False, next_action=ESCALATE_RUNTIME,
            ))
        ) from exc
    except BaseException:
        _abandon_gateway_task(workspace, task_digest, task_owner)
        raise
    _add_gateway_record_identity(agent_public, record)
    _emit_supervisor_measurement(workspace, record["record_id"], reused=False)
    agent_public["kernels"] = {
        "incumbent": {"kernel_id": baseline_kernel["kernel_id"]},
        "candidate": {"kernel_id": record["kernel_id"]},
    }
    print(
        ABBA_RESULT_PUBLIC_PREFIX
        + json.dumps(agent_public, ensure_ascii=False, allow_nan=False)
    )
    return 0 if agent_public["correct"] else 1


def _execute_typed_processes(
    args: argparse.Namespace,
    workspace: Path,
    kind: str,
    request: dict[str, Any],
    shape_batches: list[list[str]],
    queue_wait_grace: int,
    diagnostic_url: str | None,
    *,
    phase: str,
) -> list[subprocess.CompletedProcess[str]]:
    """Execute one logical typed operation over an explicit set of Shape batches."""
    batched = len(shape_batches) > 1
    agate_executable = _find_agate()
    with tempfile.TemporaryDirectory(prefix=f"atrex-{phase}-batches-") as temp_dir:
        batch_root = Path(temp_dir)

        def run_batch(item: tuple[int, list[str]]) -> subprocess.CompletedProcess[str]:
            batch_index, shape_ids = item
            batch_request = _shape_batch_request(request, shape_ids) if kind == "run" else request
            if (args.url and agate_executable is None) or diagnostic_url:
                direct_payload = batch_request
                direct_url = diagnostic_url or args.url
                if kind in DIAGNOSTIC_KINDS and not _is_loopback_gateway_url(direct_url):
                    direct_payload = {
                        key: value
                        for key, value in batch_request.items()
                        if key not in {"diagnostic_reference", "shape_id"}
                    }
                return _run_direct_job(
                    url=direct_url,
                    kind=("eval" if kind == "run" else "compile" if kind == "check" else kind),
                    payload=direct_payload,
                    timeout=args.timeout,
                    queue_wait_grace=queue_wait_grace,
                )
            if agate_executable is None:
                raise FileNotFoundError("agate")
            reference_dir = None
            if kind == "run" or (
                kind == "profile" and isinstance(batch_request.get("shape_id"), str)
            ):
                reference_dir = batch_root / f"batch-{batch_index:04d}"
                _shape_batch_reference(batch_request["reference"], reference_dir)
            agate = _typed_agate_command(
                agate_executable,
                args,
                workspace,
                kind,
                batch_request,
                queue_wait_grace,
                reference_dir,
                batch_root / f"request-{batch_index:04d}",
            )
            return _run_agate_with_cancel_retry(
                agate=agate,
                executable=agate_executable,
                url=args.url,
                gateway_profile=args.gateway_profile,
                command_timeout=_gateway_job_timeout(args.timeout, queue_wait_grace),
                wait_budget=args.timeout + queue_wait_grace,
            )

        batch_items = list(enumerate(shape_batches))
        if batched:
            print(
                f"[sandbox] running {len(shape_batches)} {phase} shape batches "
                f"with {min(DEFAULT_EVAL_BATCH_WORKERS, len(shape_batches))} workers",
                file=sys.stderr,
                flush=True,
            )
            with ThreadPoolExecutor(
                max_workers=min(DEFAULT_EVAL_BATCH_WORKERS, len(shape_batches))
            ) as executor:
                return list(executor.map(run_batch, batch_items))
        return [run_batch(batch_items[0])]


def _aggregate_typed_runs(
    workspace: Path,
    request: dict[str, Any],
    shape_ids: list[str],
    samples: list[tuple[dict[str, Any], list[dict[str, Any]]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Aggregate exactly three complete ordinary Evaluate measurements."""
    if len(samples) != MEASUREMENT_REPETITIONS:
        raise RuntimeError(f"Evaluate aggregation requires {MEASUREMENT_REPETITIONS} measurements")
    failed = next((result for result, _jobs in samples if result.get("all_pass") is not True), None)
    selected = dict(failed if failed is not None else samples[-1][0])
    if failed is None:
        medians = _median_latency_by_shape(
            [_latency_us_by_shape(result) for result, _jobs in samples],
            shape_ids,
        )
        selected = _replace_shape_latencies(selected, medians)
        metadata = request["reference"].get("metadata")
        score, _failures = _metadata_speedup_mean(
            metadata, shape_ids, _latency_us_by_shape(selected)
        )
        if score is not None:
            selected["speedup_vs_ref_mean"] = score
            selected["speedup_vs_ref_geomean"] = None
            selected["performance_score"] = score
        elif "performance_score" in selected:
            for key in (
                "speedup_vs_ref_mean",
                "speedup_vs_ref_geomean",
                "performance_score",
                "performance_objective",
            ):
                selected.pop(key, None)
            selected = _with_workspace_performance_score(workspace, selected)
    selected["measurement_aggregation"] = _measurement_aggregation_summary()
    private = {
        "measurement_aggregation": _measurement_aggregation_summary(),
        "repetitions": [
            {"result": result, "jobs": jobs} for result, jobs in samples
        ],
    }
    return selected, private


def _run_typed_gateway(
    args: argparse.Namespace,
    workspace: Path,
    command_parts: list[str],
    kind: str,
    sync_paths: list[str],
    queue_wait_grace: int,
) -> int | None:
    """Run one typed Agate operation.

    Run/Profile may use the legacy Dev compatibility route when the remote
    endpoint lacks their source contract. Check/Disassemble fail closed because
    treating a diagnostic as an arbitrary shell command changes its semantics.
    """
    custom_input_scope = bool(args.evaluation_input_path or args.evaluation_shapes_path)
    strict_evaluation = kind == "run" and bool(args.evaluation_mode or custom_input_scope)
    try:
        request = _typed_request(
            workspace,
            args.hardware,
            args.timeout,
            args.env,
            command_parts,
            kind,
            profiler=args.profiler,
            profile_level=args.profile_level,
            counters=args.profile_counter,
            kernel_regex=args.kernel_regex,
            kernel_name=args.kernel_name,
            profile_source=args.profile_source,
            launch_skip=args.launch_skip,
            launch_count=args.launch_count,
            profile_shape_id=args.profile_shape_id,
            top_kernels=args.top_kernels,
            arch=args.arch,
            sanitize=args.sanitize,
            disassembly_format=args.disassembly_format,
            requirements=args.requirement,
            deps_mode=args.deps_mode,
            evaluation_input_path=args.evaluation_input_path,
            evaluation_shapes_path=args.evaluation_shapes_path,
            evaluation_mode=args.evaluation_mode,
        )
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        if kind in DIAGNOSTIC_KINDS or strict_evaluation:
            raise SystemExit(f"sandbox: {kind} source contract is invalid: {exc}") from exc
        print(
            f"[sandbox] {kind} interface unsupported for this workspace: {exc}; using dev",
            file=sys.stderr,
        )
        return None

    expected_shape_ids = [] if kind in DIAGNOSTIC_KINDS else _request_shape_ids(request)
    shape_batches = (
        _shape_batches(
            expected_shape_ids,
            getattr(args, "shape_batch_size", DEFAULT_EVAL_SHAPE_BATCH_SIZE),
        )
        if kind == "run" and request.get("mode") != "correctness_only"
        else [expected_shape_ids]
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "hardware": args.hardware,
                    "url": args.url or None,
                    "gateway_profile": args.gateway_profile,
                    "workspace": str(workspace),
                    "kind": kind,
                    "num_gpus": request.get("spec", {}).get("num_gpus", 1),
                    "fallback_kind": (
                        None if kind in DIAGNOSTIC_KINDS or strict_evaluation else "dev"
                    ),
                    "candidate_bytes": len(request["candidate"].encode("utf-8")),
                    "shape_count": (
                        "private"
                        if _is_generalized_workspace(workspace)
                        else (
                            None
                            if kind in DIAGNOSTIC_KINDS
                            else len(request["reference"]["shapes"])
                        )
                    ),
                    "shape_batch_count": len(shape_batches),
                    "mode": request.get("mode"),
                    "options": request.get("options"),
                    "sanitize": request.get("sanitize"),
                    "format": request.get("fmt"),
                    "requirements": request.get("requirements", []),
                    "deps_mode": request.get("deps_mode"),
                    "sync": sync_paths,
                },
                indent=2,
            )
        )
        return 0

    agate_executable = _find_agate()
    diagnostic_url = (
        _resolved_gateway_url(
            agate_executable,
            url=args.url,
            profile=args.gateway_profile,
        )
        if kind in DIAGNOSTIC_KINDS
        else None
    )
    repetitions = MEASUREMENT_REPETITIONS if kind == "run" and request.get("mode") != "correctness_only" else 1
    kernel_bytes = request["candidate"].encode("utf-8")
    identity = {key: value for key, value in request.items() if key != "name"}
    if kind in DIAGNOSTIC_KINDS:
        # Compile/disassembly timeouts are CLI execution options, not source payload fields.
        identity["execution_timeout_s"] = args.timeout
    task_digest = _gateway_task_digest(kind, hashlib.sha256(kernel_bytes).hexdigest(), {
        "request": identity,
        "execution": _gateway_execution_identity(args),
    }, measurement_repetitions=repetitions)
    with _gateway_task(workspace, task_digest, kernel_bytes) as task:
        if task.owner is None:
            previous = _reusable_gateway_record(workspace, task_digest, task.previous_record_id)
            _emit_supervisor_measurement(workspace, previous["record_id"], reused=True)
            return _record_exit_code(previous)
        return _execute_typed_gateway(
            args, workspace, kind, request, shape_batches, sync_paths, queue_wait_grace,
            diagnostic_url, repetitions, task,
        )


def _execute_typed_gateway(
    args: argparse.Namespace,
    workspace: Path,
    kind: str,
    request: dict[str, Any],
    shape_batches: list[list[str]],
    sync_paths: list[str],
    queue_wait_grace: int,
    diagnostic_url: str | None,
    repetitions: int,
    task: _GatewayTask,
) -> int | None:
    generalized = _is_generalized_workspace(workspace)
    custom_input_scope = bool(args.evaluation_input_path or args.evaluation_shapes_path)
    strict_evaluation = kind == "run" and bool(args.evaluation_mode or custom_input_scope)
    expected_shape_ids = [shape for batch in shape_batches for shape in batch]
    batched = len(shape_batches) > 1
    task_digest, task_owner = task.digest, task.owner
    try:
        process_groups = [
            _execute_typed_processes(
                args,
                workspace,
                kind,
                request,
                shape_batches,
                queue_wait_grace,
                diagnostic_url,
                phase=(
                    f"measurement-{ordinal}"
                    if repetitions > 1
                    else "primary"
                ),
            )
            for ordinal in range(1, repetitions + 1)
        ]
    except GatewayHTTPError as exc:
        if task_digest is not None and task_owner is not None:
            _abandon_gateway_task(workspace, task_digest, task_owner)
        if not (kind in DIAGNOSTIC_KINDS or strict_evaluation) and _typed_fallback_allowed(exc):
            print(
                f"[sandbox] gateway {kind} interface unavailable ({exc}); using dev",
                file=sys.stderr,
            )
            return None
        infrastructure = _retryable_gateway_http_error(exc) or exc.status in {401, 403}
        message = (
            f"Gateway HTTP {exc.status} could not complete {kind}."
            if generalized or infrastructure else
            f"{kind} gateway request failed: {_bounded_text(exc, 1500)}"
        )
        raise SystemExit(json.dumps(error_response(
            message,
            code="gateway_unavailable" if infrastructure else "gateway_request_rejected",
            repairable=not infrastructure,
            next_action=ESCALATE_RUNTIME if infrastructure else (
                "Check the operation arguments and public source/input contract before retrying. "
                "Hidden evaluator inputs and Supervisor configuration must not be changed."
            ),
        ))) from exc
    except FileNotFoundError as exc:
        if task_digest is not None and task_owner is not None:
            _abandon_gateway_task(workspace, task_digest, task_owner)
        raise RuntimeStateError(
            "The Supervisor's Agate client or endpoint is unavailable; "
            "this is not an Agent dependency.",
            code="gateway_dependency_unavailable",
        ) from exc

    job_groups: list[list[dict[str, Any]]] = []
    for processes in process_groups:
        jobs: list[dict[str, Any]] = []
        for proc in processes:
            detail = (proc.stderr or "") + (proc.stdout or "")
            if proc.returncode and _typed_fallback_allowed(detail):
                if task_digest is not None and task_owner is not None:
                    _abandon_gateway_task(workspace, task_digest, task_owner)
                if kind in DIAGNOSTIC_KINDS or strict_evaluation:
                    raise SystemExit(
                        f"sandbox: typed {kind} is unavailable: {_bounded_text(detail, 2000)}"
                    )
                print(
                    f"[sandbox] gateway {kind} interface rejected this request; using dev",
                    file=sys.stderr,
                )
                return None
            if proc.stderr and not generalized:
                retry_notes = _agent_retry_notes(proc.stderr)
                if retry_notes:
                    print(retry_notes, file=sys.stderr)
            job = _job_response(proc.stdout or "")
            if job is None:
                if task_digest is not None and task_owner is not None:
                    _abandon_gateway_task(workspace, task_digest, task_owner)
                message = (
                    f"Supervisor Agate client returned no valid Job response for {kind} "
                    f"(exit={proc.returncode})."
                )
                if not generalized and detail.strip():
                    message += " " + _bounded_text(detail.strip(), 2000)
                print(json.dumps(error_response(
                    message, code="gateway_client_error", repairable=False,
                    next_action=(
                        "Ask the operator to check the Supervisor's Agate CLI compatibility "
                        "and transport diagnostics. Do not change Kernel code to fix this "
                        "error or blindly repeat the same request."
                    ),
                ), ensure_ascii=False), file=sys.stderr)
                return proc.returncode or 2
            if job.get("status") != "succeeded" or not isinstance(job.get("result"), dict):
                failure = {
                    "status": job.get("status"),
                    "error": job.get("error"),
                }
                error = job.get("error")
                infrastructure = isinstance(error, dict) and error.get("error_class") == "infra"
                record = task.record(
                    failure,
                    gateway_kind=kind,
                    job_id=job.get("job_id"),
                    private_result=job,
                    cache=not infrastructure and job.get("status") in {"failed", "cancelled"},
                )
                if record is not None and not infrastructure:
                    _emit_supervisor_measurement(workspace, record["record_id"], reused=False)
                print(
                    json.dumps(
                        _agent_gateway_failure(
                            job,
                            record,
                            generalized=(
                                generalized
                                and not custom_input_scope
                                and kind not in DIAGNOSTIC_KINDS
                            ),
                        ),
                        ensure_ascii=False,
                    )
                )
                return proc.returncode or 1
            jobs.append(job)
        job_groups.append(jobs)

    print(
        f"[sandbox] gateway {kind} completed in "
        f"{sum(len(jobs) for jobs in job_groups)} batch job(s)",
        file=sys.stderr,
    )
    if kind == "run":
        metadata = request["reference"].get("metadata")
        require_performance = request.get("mode") != "correctness_only"
        normalized_repetitions: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        for jobs in job_groups:
            batch_results = [
                _optimizer_result_from_eval(
                    job["result"],
                    shape_ids,
                    metadata,
                    require_performance=require_performance,
                )
                for job, shape_ids in zip(jobs, shape_batches, strict=True)
            ]
            normalized = (
                _merge_optimizer_results(
                    batch_results,
                    expected_shape_ids,
                    metadata,
                    require_performance=require_performance,
                )
                if batched
                else batch_results[0]
            )
            normalized_repetitions.append((normalized, jobs))
        normalized_result = normalized_repetitions[-1][0]
        private_result: dict[str, Any] = {"jobs": job_groups[-1]}
        if require_performance:
            try:
                normalized_result, private_result = _aggregate_typed_runs(
                    workspace,
                    request,
                    expected_shape_ids,
                    normalized_repetitions,
                )
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                if task_digest is not None and task_owner is not None:
                    _abandon_gateway_task(workspace, task_digest, task_owner)
                raise SystemExit(
                    "sandbox: Evaluate measurement aggregation failed: "
                    + _bounded_text(exc, 3000)
                ) from exc
        result = (
            normalized_result
            if custom_input_scope
            else _mask_generalized_result(workspace, normalized_result)
        )
        if request.get("mode") == "correctness_only" or custom_input_scope:
            result["mode"] = request.get("mode", "full")
            result["input_scope"] = "custom" if custom_input_scope else "contract"
        if request.get("mode") == "correctness_only":
            result["latency_us_geomean"] = None
            result["latency_us_arith_mean"] = None
            result["latency_us_by_shape"] = {}
            result.pop("performance_score", None)
            result.pop("performance_objective", None)
        record = task.record(
            result,
            gateway_kind=kind,
            job_id=",".join(
                str(job.get("job_id")) for jobs in job_groups for job in jobs
            ),
            private_result=private_result,
            measurement_repetitions=[
                result for result, _jobs in normalized_repetitions
            ]
            if repetitions > 1
            else None,
        )
        if record is not None:
            _emit_supervisor_measurement(workspace, record["record_id"], reused=False)
        print(
            TEST_RESULT_PREFIX
            + json.dumps(
                _agent_evaluation_result(result, record),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
        return 0 if result["all_pass"] else 1

    if kind in DIAGNOSTIC_KINDS:
        jobs = job_groups[0]
        raw_result = jobs[0]["result"]
        public_result = (
            _agent_check_result(raw_result, None)
            if kind == "check"
            else _agent_disassembly_result(raw_result, None)
        )
        diagnostic_record = task.record(
            public_result,
            gateway_kind=kind,
            job_id=jobs[0].get("job_id"),
            private_result=jobs[0],
        )
        public_result = (
            _agent_check_result(raw_result, diagnostic_record)
            if kind == "check"
            else _agent_disassembly_result(raw_result, diagnostic_record)
        )
        prefix = CHECK_RESULT_PREFIX if kind == "check" else DISASSEMBLY_RESULT_PREFIX
        print(prefix + json.dumps(public_result, ensure_ascii=False))
        passed = raw_result.get("passed", raw_result.get("ok", True))
        return 0 if passed is not False else 1

    jobs = job_groups[0]
    profile_record = task.record(
        jobs[0]["result"],
        gateway_kind=kind,
        job_id=jobs[0].get("job_id"),
        private_result=jobs[0],
    )
    public_profile = _agent_profile_result(jobs[0]["result"], profile_record)
    _record_profile_job(jobs[0], workspace, sync_paths, public_profile)
    print(
        PROFILE_RESULT_PREFIX
        + json.dumps(
            public_profile,
            ensure_ascii=False,
        )
    )
    return 0


def _evaluation_command(args: argparse.Namespace) -> list[str]:
    """Translate Agent options to the private evaluator's remote command."""
    command = ["python3", "test_kernel.py", "--no-memory"]
    for option, value in (
        ("--version", args.version),
        ("--multi-seed", args.multi_seed),
        ("--timed-runs", args.timed_runs),
    ):
        if value is not None:
            command += [option, str(value)]
    for shape_id in args.shape_id:
        command += ["--shape-id", shape_id]
    return command


def _profile_fallback_command(args: argparse.Namespace, sync_paths: list[str]) -> list[str]:
    """Build a remote-only profiler wrapper without requiring an Agent driver."""
    if args.profile_counter or args.top_kernels:
        raise ValueError("custom counters/top-kernels require the typed Profile route")
    output_dir = sync_paths[0] if sync_paths else "scratch/profile"
    command = ["python3", "profile_entry.py", "--output-dir", output_dir]
    for option, value in (
        ("--profiler", args.profiler),
        ("--kernel-name", args.kernel_name),
        ("--kernel-regex", args.kernel_regex),
        ("--launch-skip", args.launch_skip),
        ("--launch-count", args.launch_count),
    ):
        if value is not None:
            command += [option, str(value)]
    if args.profile_source:
        command.append("--source")
    return command


def _main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.kind == "record-read":
        if args.command not in ([], ["--"]):
            raise SystemExit("sandbox: --kind record-read does not accept a command")
        workspace = Path(args.workspace).resolve()
        if not workspace.is_dir():
            raise SystemExit(f"sandbox: workspace not found: {workspace}")
        if not args.record_id:
            raise SystemExit("sandbox: --kind record-read requires --record-id")
        if GATEWAY_RECORD_ID_RE.fullmatch(args.record_id):
            if args.view is not None or args.output_path is not None:
                raise SystemExit(
                    "sandbox: --view and --output-path are invalid for a Gateway record"
                )
            return _read_gateway_record(workspace, args.record_id)
        if KERNEL_RECORD_ID_RE.fullmatch(args.record_id):
            if args.view is None:
                raise SystemExit(
                    "sandbox: a Kernel record requires --view source or --view gateway-records"
                )
            if args.view == "source":
                if not args.output_path:
                    raise SystemExit(
                        "sandbox: Kernel source view requires --output-path scratch/<file>"
                    )
                return _read_kernel_source(workspace, args.record_id, args.output_path)
            if args.output_path is not None:
                raise SystemExit(
                    "sandbox: --output-path is valid only with --view source"
                )
            return _read_kernel_gateway_records(workspace, args.record_id)
        raise SystemExit(
            "sandbox: --record-id must be a gateway-... record or kernel-... Kernel ID"
        )
    if args.record_id is not None or args.view is not None or args.output_path is not None:
        raise SystemExit(
            "sandbox: --record-id, --view, and --output-path require --kind record-read"
        )
    if not args.hardware and args.kind != "env":
        raise SystemExit("sandbox: --hardware or ATREX_SANDBOX_GPU is required")
    # Explicit endpoint flags override inherited sandbox endpoint variables.  This
    # matters when a long-lived optimization shell switches between a remote
    # profile and localhost without first scrubbing its environment.
    explicit_endpoints = sum(
        value is not None for value in (args.url, args.gateway_profile, args.ssh)
    )
    if explicit_endpoints > 1:
        raise SystemExit("sandbox: --ssh, --url, and --gateway-profile are mutually exclusive")
    if args.ssh is not None:
        args.url = ""
        args.gateway_profile = None
    elif args.url is not None:
        args.ssh = ""
        args.gateway_profile = None
    elif args.gateway_profile is not None:
        args.ssh = ""
        args.url = ""
    else:
        args.ssh = os.environ.get("ATREX_SANDBOX_SSH", "")
        args.url = os.environ.get("ATREX_SANDBOX_URL", "")
        args.gateway_profile = os.environ.get("ATREX_SANDBOX_PROFILE") or None
        if sum(bool(value) for value in (args.ssh, args.url, args.gateway_profile)) > 1:
            raise SystemExit(
                "sandbox: ATREX_SANDBOX_SSH, ATREX_SANDBOX_URL, and "
                "ATREX_SANDBOX_PROFILE are mutually exclusive"
            )
    if args.ssh:
        try:
            args.ssh = _validate_ssh_target(args.ssh)
            args.ssh_gpu = _ssh_gpu_index(
                args.ssh_gpu if args.ssh_gpu is not None else os.environ.get(SSH_GPU_ENV, "")
            )
            if args.ssh_runtime_bind is None:
                args.ssh_runtime_bind = _environment_ssh_runtime_binds()
            for runtime_bind in args.ssh_runtime_bind:
                _ssh_runtime_bind(runtime_bind)
        except ValueError as exc:
            raise SystemExit(f"sandbox: {exc}") from exc
        if not args.health_command.strip():
            raise SystemExit("sandbox: --health-command must not be empty with --ssh")
    elif args.ssh_runtime_bind:
        raise SystemExit("sandbox: --ssh-runtime-bind requires --ssh")
    elif args.ssh_gpu is not None:
        raise SystemExit("sandbox: --ssh-gpu requires --ssh")
    if not 1 <= args.timeout <= MAX_COMMAND_TIMEOUT:
        raise SystemExit(
            f"sandbox: --timeout must be in the gateway-supported range 1..{MAX_COMMAND_TIMEOUT}"
        )
    if args.shape_batch_size <= 0:
        raise SystemExit("sandbox: --shape-batch-size must be positive")
    try:
        queue_wait_grace = int(
            os.environ.get("ATREX_SANDBOX_QUEUE_WAIT_GRACE", str(DEFAULT_QUEUE_WAIT_GRACE))
        )
    except ValueError as exc:
        raise SystemExit("sandbox: ATREX_SANDBOX_QUEUE_WAIT_GRACE must be an integer") from exc
    if queue_wait_grace < 0:
        raise SystemExit("sandbox: ATREX_SANDBOX_QUEUE_WAIT_GRACE must be non-negative")
    if args.max_input_file_mb <= 0 or args.max_output_file_mb <= 0:
        raise SystemExit("sandbox: file size limits must be positive")
    args.health_command = combined_health_command(args.health_command, args.runtime_health_command)
    if args.kind == "env":
        if args.command not in ([], ["--"]):
            raise SystemExit("sandbox: --kind env does not accept a command")
        if args.env_capabilities and not args.env_gpu:
            raise SystemExit("sandbox: --env-capabilities requires --env-gpu")
        return _run_environment_query(args)
    if args.check_health or args.preflight:
        if not args.ssh:
            raise SystemExit("sandbox: --check-health/--preflight requires --ssh")
        try:
            health = _run_ssh_health(
                args.ssh,
                args.ssh_init,
                args.health_command,
                args.ssh_runtime_bind,
                args.ssh_gpu,
            )
        except SSHTransportError as exc:
            health = subprocess.CompletedProcess(
                args=["ssh", args.ssh], returncode=1, stdout="", stderr=str(exc)
            )
        if health.stdout:
            print(health.stdout.rstrip())
        if health.stderr:
            print(health.stderr.rstrip(), file=sys.stderr)
        if args.preflight and health.returncode != 0:
            _record_environment_failure(
                target=args.ssh,
                stage="preflight",
                detail=(health.stderr or health.stdout or "health probe failed")[-2000:],
                health_status=health.returncode,
            )
            return ENVIRONMENT_TEMPFAIL
        return health.returncode
    try:
        command = (
            ""
            if args.kind in TYPED_KINDS and args.command in ([], ["--"])
            else _command_text(args.command)
        )
        sync_paths = (
            []
            if args.no_sync
            else [_safe_relative(path) for path in (args.sync or list(DEFAULT_SYNC_PATHS))]
        )
    except ValueError as exc:
        raise SystemExit(f"sandbox: {exc}") from exc

    workspace = Path(args.workspace).resolve()
    if any(PurePosixPath(path).parts[0] == "memory" for path in sync_paths):
        raise SystemExit("sandbox: memory/ is local optimizer state and cannot be synchronized")
    if not workspace.is_dir():
        raise SystemExit(f"sandbox: workspace not found: {workspace}")
    if _is_unsafe_target_command(args.command):
        raise SystemExit(
            "sandbox: evaluator and profile targets must use a supported launcher "
            "with separate arguments after --"
        )

    gateway_kind = _requested_gateway_kind(args.kind, args.command)
    evaluation_options = bool(
        args.version is not None or args.multi_seed is not None
        or args.timed_runs is not None or args.shape_id
    )
    if evaluation_options and (gateway_kind != "run" or command):
        raise SystemExit(
            "sandbox: --version/--multi-seed/--timed-runs/--shape-id require "
            "--kind run without a command after --"
        )
    if (args.multi_seed is not None and args.multi_seed < 0) or (
        args.timed_runs is not None and args.timed_runs < 1
    ):
        raise SystemExit("sandbox: --multi-seed must be >= 0 and --timed-runs must be >= 1")
    if gateway_kind == "run" and not command:
        args.command = _evaluation_command(args)
        command = _command_text(args.command)
    if gateway_kind in DIAGNOSTIC_KINDS and command:
        raise SystemExit(
            f"sandbox: --kind {gateway_kind} operates on kernel.py directly; "
            "do not provide a command after --"
        )
    if args.arch and gateway_kind != "check":
        raise SystemExit("sandbox: --arch is only valid with --kind check")
    if args.sanitize and gateway_kind != "check":
        raise SystemExit("sandbox: --sanitize is only valid with --kind check")
    if args.disassembly_format != "auto" and gateway_kind != "disassemble":
        raise SystemExit("sandbox: --format is only valid with --kind disassemble")
    if args.kernel_regex and args.kernel_name:
        raise SystemExit("sandbox: --kernel-regex and --kernel-name are mutually exclusive")
    if args.launch_skip is not None and args.launch_skip < 0:
        raise SystemExit("sandbox: --launch-skip must be non-negative")
    if args.launch_count is not None and args.launch_count <= 0:
        raise SystemExit("sandbox: --launch-count must be positive")
    profile_only = (
        args.kernel_regex
        or args.kernel_name
        or args.profile_source
        or args.launch_skip is not None
        or args.launch_count is not None
        or args.profile_shape_id is not None
    )
    if profile_only and gateway_kind != "profile":
        raise SystemExit("sandbox: profile selectors require --kind profile")
    if args.env_gpu or args.env_capabilities or args.env_force:
        raise SystemExit("sandbox: --env-* options require --kind env")
    if (args.requirement or args.deps_mode) and gateway_kind not in DEPENDENCY_KINDS:
        raise SystemExit(
            "sandbox: --requirement and --deps-mode are only valid with "
            "--kind profile, --kind check, or --kind disassemble"
        )
    if (
        args.evaluation_mode or args.evaluation_input_path or args.evaluation_shapes_path
    ) and gateway_kind != "run":
        raise SystemExit(
            "sandbox: --mode, --input-path, and --shapes-path are only valid with --kind run"
        )
    if args.baseline_path is not None and gateway_kind != "run":
        raise SystemExit("sandbox: --baseline-path is only valid with --kind run")
    if args.baseline_path is None and args.comparison_repeats != 2:
        raise SystemExit("sandbox: --comparison-repeats is only valid with --baseline-path")
    evaluator_command = _is_test_kernel_command(args.command)
    profile_command = _is_profile_command(args.command)
    profile_request = gateway_kind == "profile" or profile_command
    if profile_request:
        try:
            args.env = _with_inherited_profile_environment(args.env)
        except ValueError as exc:
            raise SystemExit(f"sandbox: {exc}") from exc
    if args.baseline_path is not None:
        return _run_agent_abba(
            args,
            workspace,
            args.command,
            queue_wait_grace,
        )
    typed_limitation: str | None = None
    typed_fallback_kind: str | None = None
    num_gpus = 1
    if gateway_kind in TYPED_KINDS or evaluator_command or profile_request:
        try:
            num_gpus = _workspace_num_gpus(workspace)
        except ValueError as exc:
            raise SystemExit(f"sandbox: invalid distributed evaluator contract: {exc}") from exc
    if gateway_kind in TYPED_KINDS:
        if args.ssh:
            typed_limitation = "SSH uses the portable sandbox command runner"
        elif (
            gateway_kind == "profile"
            and args.profile_level == "deep"
            and not (args.kernel_regex or args.kernel_name)
        ):
            raise SystemExit(
                "sandbox: --profile-level deep requires --kernel-regex or --kernel-name"
            )
        elif args.keep_pod:
            typed_limitation = "--keep-pod is only supported by dev"
        elif args.input:
            typed_limitation = "custom --input files are only supported by dev"
        elif gateway_kind == "profile" and args.include_raw_profile:
            typed_limitation = "--include-raw-profile requires the custom dev profiler wrapper"
        else:
            try:
                typed_limitation = _typed_workspace_limitation(
                    workspace, args.command, gateway_kind
                )
            except ValueError as exc:
                typed_limitation = str(exc)
        if gateway_kind in DIAGNOSTIC_KINDS and typed_limitation is not None:
            raise SystemExit(f"sandbox: --kind {gateway_kind} is unavailable: {typed_limitation}")
        if typed_limitation is None:
            typed_result = _run_typed_gateway(
                args,
                workspace,
                args.command,
                gateway_kind,
                sync_paths,
                queue_wait_grace,
            )
            if typed_result is not None:
                return typed_result
            if gateway_kind in DIAGNOSTIC_KINDS or (
                gateway_kind == "run"
                and (
                    args.evaluation_mode
                    or args.evaluation_input_path
                    or args.evaluation_shapes_path
                )
            ):
                raise SystemExit(f"sandbox: --kind {gateway_kind} produced no typed result")
            typed_limitation = (
                f"gateway {gateway_kind} route unavailable or rejected the source contract"
            )
        if gateway_kind == "run" and (
            args.evaluation_mode or args.evaluation_input_path or args.evaluation_shapes_path
        ):
            raise SystemExit(
                "sandbox: custom-input or explicit-mode evaluation requires the "
                f"typed run route: {typed_limitation}"
            )
        typed_fallback_kind = gateway_kind
        gateway_kind = "dev"

    if gateway_kind == "dev" and profile_request and not command:
        try:
            args.command = _profile_fallback_command(args, sync_paths)
        except ValueError as exc:
            raise SystemExit(f"sandbox: {exc}") from exc
        command = _command_text(args.command)
        profile_command = True
    if gateway_kind == "dev" and profile_command and args.profile_shape_id is not None:
        args.env = [item for item in args.env if not item.startswith("PROFILE_SHAPE_ID=")]
        args.env.append("PROFILE_SHAPE_ID=" + args.profile_shape_id)

    if gateway_kind == "dev" and profile_request and num_gpus > 1:
        limitation = f" ({typed_limitation})" if typed_limitation else ""
        raise SystemExit(
            "sandbox: distributed profile commands require the typed profile route"
            f"{limitation}; "
            "the dev route does not launch ranks"
        )
    if args.ssh and evaluator_command and num_gpus > 1:
        raise SystemExit(
            "sandbox: distributed evaluator commands are not supported by the SSH "
            "runner; it exposes only one GPU"
        )
    if typed_fallback_kind is not None:
        if args.ssh:
            print(
                f"[sandbox] {typed_fallback_kind} kind uses the isolated OpenSSH runner",
                file=sys.stderr,
            )
        else:
            print(
                f"[sandbox] {typed_fallback_kind} interface unsupported "
                f"({typed_limitation}); using dev",
                file=sys.stderr,
            )

    if evaluator_command:
        selected = set(_evaluation_input_paths(workspace, args.command))
        try:
            for value in args.input:
                selected.update(_expand_workspace_input(workspace, value))
        except ValueError as exc:
            raise SystemExit(f"sandbox: {exc}") from exc
        selected_inputs = frozenset(selected)
    else:
        try:
            selected_inputs = _command_input_paths(
                workspace,
                args.command,
                args.input,
            )
        except ValueError as exc:
            raise SystemExit(f"sandbox: {exc}") from exc
    try:
        injected_inputs = _private_evaluator_inputs(workspace) if evaluator_command else {}
        if evaluator_command:
            injected_inputs.update(evaluation_inputs(workspace))
        if "profile_driver.py" in selected_inputs:
            injected_inputs["profile_driver.py"] = PROFILE_DRIVER
        if "profile_entry.py" in selected_inputs:
            injected_inputs["profile_entry.py"] = RUNNERS_ROOT / "profile_entry.py"
        injected_inputs.update(_supervisor_runtime_inputs(args.command))
        selected_inputs = frozenset((*selected_inputs, *injected_inputs))
        injected_payloads: dict[str, bytes] = {}
        if profile_command and _is_generalized_workspace(workspace):
            profile_case = _private_profile_case(workspace, args.env, args.profile_shape_id)
            if profile_case is not None:
                filename, payload = profile_case
                injected_payloads[filename] = payload
                selected_inputs = frozenset((*selected_inputs, filename))
        bundle, file_count, skipped = _make_input_bundle(
            workspace,
            args.max_input_file_mb * 1024 * 1024,
            selected_inputs,
            injected_inputs,
            injected_payloads,
        )
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise SystemExit(f"sandbox: cannot prepare evaluator inputs: {exc}") from exc
    if evaluator_command:
        runtime_bundle = _make_atrex_bench_runtime_bundle(
            _private_atrex_bench_runtime(),
            evaluator_only=evaluator_command,
        )
    else:
        # Profile drivers and ad-hoc dev commands never import the evaluator;
        # uploading it pushes the gateway's ray-submit argv past MAX_ARG limits.
        runtime_bundle = None
    bundle_bytes = len(bundle.encode("ascii"))
    runtime_bundle_bytes = len(runtime_bundle.encode("ascii")) if runtime_bundle else 0
    gateway_environment = list(args.env)
    if profile_command and _is_generalized_workspace(workspace):
        command_environment, gateway_environment = _profile_command_environment(args.env)
        if command_environment:
            command = shlex.join(["env", *command_environment]) + " " + command
    agate_executable = None if args.ssh else _find_agate()
    direct_http = bool(args.url and agate_executable is None)
    standard_oss_gateway = bool(
        agate_executable
        and _uses_standard_oss_gateway(
            agate_executable,
            url=args.url,
            profile=args.gateway_profile,
        )
    )
    oss_workspace = bool(standard_oss_gateway and bundle_bytes > OSS_WORKSPACE_THRESHOLD_BYTES)
    workspace_transport = (
        "ssh" if args.ssh else ("oss" if oss_workspace else ("http" if direct_http else "inline"))
    )
    if not sync_paths:
        output_transport = "none"
    elif args.ssh:
        output_transport = "ssh"
    # Custom gateways do not advertise OSS capability yet, so both input and
    # output stay inline unless agate resolves to one of its standard profiles.
    elif standard_oss_gateway and not args.inline_output:
        output_transport = "oss"
    else:
        output_transport = "inline"
    if direct_http and bundle_bytes > 20 * 1024 * 1024:
        raise SystemExit(
            f"sandbox: packaged payload is {bundle_bytes / 1024:.1f} KiB, "
            "above the 20 MiB direct gateway request limit"
        )
    print(
        f"[sandbox] sandbox_kind=dev hardware={args.hardware} files={file_count} "
        f"payload={bundle_bytes / 1024:.1f} KiB "
        f"input_transport={workspace_transport} "
        f"output_transport={output_transport} "
        f"atrex_runtime={runtime_bundle_bytes / 1024:.1f} KiB command={command!r}",
        file=sys.stderr,
    )
    if skipped:
        print("[sandbox] inputs skipped: " + ", ".join(skipped), file=sys.stderr)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "hardware": args.hardware,
                    "ssh": args.ssh or None,
                    "ssh_init": bool(args.ssh_init),
                    "ssh_isolation": "bubblewrap" if args.ssh else None,
                    "ssh_runtime_binds": args.ssh_runtime_bind or [],
                    "url": args.url or None,
                    "gateway_profile": args.gateway_profile,
                    "workspace": str(workspace),
                    "kind": "dev",
                    "num_gpus": num_gpus,
                    "requested_kind": args.kind,
                    "typed_fallback_reason": typed_limitation,
                    "files": file_count,
                    "payload_bytes": bundle_bytes,
                    "workspace_transport": workspace_transport,
                    "output_transport": output_transport,
                    "atrex_runtime_payload_bytes": runtime_bundle_bytes,
                    "sync": sync_paths,
                    "command": command,
                },
                indent=2,
            )
        )
        return 0

    output_cfg = {
        "paths": sync_paths,
        "max_file_bytes": args.max_output_file_mb * 1024 * 1024,
        "include_raw_profile": args.include_raw_profile,
        "transport": output_transport,
    }
    inputs, kernel_bytes = _bundle_task_inputs(bundle)
    runtime_inputs, _ = _bundle_task_inputs(runtime_bundle)
    if kernel_bytes is None:
        kernel_bytes = (workspace / "kernel.py").read_bytes()
    task_digest = None if os.environ.get(INTERNAL_MEASUREMENT_ENV) == "1" else _gateway_task_digest(
        "dev", hashlib.sha256(kernel_bytes).hexdigest(), {
            "command": command,
            "inputs": inputs,
            "runtime_inputs": runtime_inputs,
            "execution": _gateway_execution_identity(args),
            "environment": _parse_env_items(gateway_environment),
            "num_gpus": num_gpus,
            "timeout": args.timeout,
            "keep_pod": args.keep_pod,
            "outputs": output_cfg,
            "profile_request": profile_request,
            "evaluator_command": evaluator_command,
            "runner_sha256": hashlib.sha256(_runner_source().encode()).hexdigest(),
            "collector_sha256": hashlib.sha256(REMOTE_COLLECTOR.encode()).hexdigest(),
        }, measurement_repetitions=1,
    )
    with _gateway_task(workspace, task_digest, kernel_bytes) as task:
        if task_digest is not None and task.owner is None:
            previous = _reusable_gateway_record(workspace, task_digest, task.previous_record_id)
            _emit_supervisor_measurement(workspace, previous["record_id"], reused=True)
            return _record_exit_code(previous)
        with tempfile.TemporaryDirectory(prefix="atrex-sandbox-") as temp_dir:
            temp = Path(temp_dir)
            command_path = temp / "command.sh"
            collector_path = temp / "collect.py"
            outputs_path = temp / "outputs.json"
            runtime_part_paths: list[Path] = []
            workspace_part_paths: list[Path] = []
            # Chunk workspace bundle when it exceeds MAX_ARG_STRLEN safe limit
            # (same pattern as runtime chunking). The runner concatenates parts.
            if not args.ssh and not oss_workspace and len(bundle) > WORKSPACE_CHUNK_BYTES:
                for index, offset in enumerate(range(0, len(bundle), WORKSPACE_CHUNK_BYTES)):
                    part_path = temp / f"atrex_workspace.part{index:03d}"
                    part_path.write_text(
                        bundle[offset : offset + WORKSPACE_CHUNK_BYTES],
                        encoding="ascii",
                    )
                    workspace_part_paths.append(part_path)
            else:
                bundle_path = temp / "workspace.tar.gz.b64"
                bundle_path.write_text(bundle, encoding="ascii")
            command_path.write_text(
                "#!/usr/bin/env bash\nset -o pipefail\n" + command + "\n", encoding="utf-8"
            )
            collector_path.write_text(REMOTE_COLLECTOR, encoding="utf-8")
            outputs_path.write_text(json.dumps(output_cfg), encoding="utf-8")
            if runtime_bundle:
                runtime_chunk_bytes = len(runtime_bundle) if args.ssh else RUNTIME_CHUNK_BYTES
                for index, offset in enumerate(range(0, len(runtime_bundle), runtime_chunk_bytes)):
                    part_path = temp / f"atrex_runtime.part{index:03d}"
                    part_path.write_text(
                        runtime_bundle[offset : offset + runtime_chunk_bytes],
                        encoding="ascii",
                    )
                    runtime_part_paths.append(part_path)

            if args.kind == "profile":
                dev_intent = "profile_adhoc"
            elif args.kind == "run":
                dev_intent = "custom_harness"
            else:
                dev_intent = "other"
            agate = [
                agate_executable or "agate",
                "dev",
                "--intent",
                dev_intent,
                "--note",
                f"tools/sandbox.py {args.kind} compatibility path",
            ]
            if args.url:
                agate += ["--url", args.url]
            elif args.gateway_profile:
                agate += ["--profile", args.gateway_profile]
            agate += ["--gpu", args.hardware]
            if num_gpus > 1:
                agate += ["--num-gpus", str(num_gpus)]
            agate += [
                "--dev-timeout",
                str(args.timeout),
                "--http-timeout",
                str(MAX_HTTP_REQUEST_TIMEOUT),
                "--wait-timeout",
                str(args.timeout + queue_wait_grace),
                "--job-timeout",
                str(_dev_gateway_job_timeout(args.timeout)),
            ]
            if oss_workspace:
                agate += ["--oss-file", f"__atrex_workspace.tar.gz.b64={bundle_path}"]
            elif workspace_part_paths:
                for index, part_path in enumerate(workspace_part_paths):
                    agate += [
                        "--file",
                        f"__atrex_workspace.tar.gz.b64.part{index:03d}={part_path}",
                    ]
            else:
                agate += ["--file", f"__atrex_workspace.tar.gz.b64={bundle_path}"]
            agate += [
                "--file",
                f"__atrex_command.sh={command_path}",
                "--file",
                f"__atrex_collect.py={collector_path}",
                "--file",
                f"__atrex_outputs.json={outputs_path}",
            ]
            for index, part_path in enumerate(runtime_part_paths):
                agate += [
                    "--file",
                    f"__atrex_bench_runtime.tar.gz.b64.part{index:03d}={part_path}",
                ]
            for item in gateway_environment:
                if "=" not in item or item.startswith("="):
                    raise SystemExit(f"sandbox: invalid --env {item!r}; expected KEY=VALUE")
                agate += ["--env-var", item]
            if args.keep_pod:
                agate.append("--no-recycle")
            if output_transport == "oss":
                agate += ["--oss-output", OSS_OUTPUT_ARCHIVE]
            agate.append("bash __atrex_runner.sh")

            # The runner is uploaded separately after the command has been assembled.
            runner_path = temp / "runner.sh"
            runner_path.write_text(_runner_source(), encoding="utf-8")
            agate[-1:-1] = ["--file", f"__atrex_runner.sh={runner_path}"]

            if args.ssh:
                upload_dir = temp / "ssh-upload"
                upload_dir.mkdir()
                uploads: list[tuple[Path, str]] = [
                    (command_path, "__atrex_command.sh"),
                    (collector_path, "__atrex_collect.py"),
                    (outputs_path, "__atrex_outputs.json"),
                    (runner_path, "__atrex_runner.sh"),
                ]
                uploads.extend(
                    (path, f"__atrex_bench_runtime.tar.gz.b64.part{index:03d}")
                    for index, path in enumerate(runtime_part_paths)
                )
                uploads.extend(
                    (path, f"__atrex_workspace.tar.gz.b64.part{index:03d}")
                    for index, path in enumerate(workspace_part_paths)
                )
                if not workspace_part_paths:
                    uploads.append((bundle_path, "__atrex_workspace.tar.gz.b64"))
                upload_paths = []
                for source, remote_name in uploads:
                    destination = upload_dir / remote_name
                    shutil.copy2(source, destination)
                    upload_paths.append(destination)
                try:
                    ssh_result = _run_ssh_job(
                        target=args.ssh,
                        init_command=args.ssh_init,
                        runtime_binds=args.ssh_runtime_bind,
                        gpu_index=args.ssh_gpu,
                        timeout=args.timeout,
                        env_items=gateway_environment,
                        upload_paths=upload_paths,
                        temp=temp,
                        workspace=workspace,
                        sync_outputs=bool(sync_paths),
                    )
                except SSHTransportError as exc:
                    _record_environment_failure(
                        target=args.ssh,
                        stage="transport",
                        detail=str(exc),
                    )
                    print(f"sandbox: SSH environment unavailable: {exc}", file=sys.stderr)
                    return ENVIRONMENT_TEMPFAIL
                if ssh_result.returncode != 0:
                    try:
                        health = _run_ssh_health(
                            args.ssh,
                            args.ssh_init,
                            args.health_command,
                            args.ssh_runtime_bind,
                            args.ssh_gpu,
                        )
                    except SSHTransportError as exc:
                        health = subprocess.CompletedProcess(
                            args=["ssh", args.ssh],
                            returncode=1,
                            stdout="",
                            stderr=str(exc),
                        )
                    if health.returncode != 0:
                        detail = (health.stderr or health.stdout or "health probe failed")[-2000:]
                        _record_environment_failure(
                            target=args.ssh,
                            stage="post-command-health",
                            detail=detail,
                            health_status=health.returncode,
                        )
                        print(
                            "sandbox: remote command failed and the GPU environment health "
                            "probe also failed; optimization recovery requested",
                            file=sys.stderr,
                        )
                        return ENVIRONMENT_TEMPFAIL
                job = {
                    "job_id": f"ssh-{os.getpid()}",
                    "status": "succeeded" if ssh_result.returncode == 0 else "failed",
                    "result": {
                        "stdout": ssh_result.stdout,
                        "stderr": ssh_result.stderr,
                        "exit_code": ssh_result.returncode,
                    },
                }
                proc = subprocess.CompletedProcess(
                    args=ssh_result.args,
                    returncode=ssh_result.returncode,
                    stdout=json.dumps(job),
                    stderr="",
                )
            elif direct_http:
                print(
                    "[sandbox] agate CLI not found; using direct gateway HTTP API",
                    file=sys.stderr,
                )
                try:
                    direct_files = {
                        "__atrex_command.sh": command_path,
                        "__atrex_collect.py": collector_path,
                        "__atrex_outputs.json": outputs_path,
                        "__atrex_runner.sh": runner_path,
                    }
                    if workspace_part_paths:
                        direct_files.update(
                            {
                                f"__atrex_workspace.tar.gz.b64.part{index:03d}": path
                                for index, path in enumerate(workspace_part_paths)
                            }
                        )
                    else:
                        direct_files["__atrex_workspace.tar.gz.b64"] = bundle_path
                    direct_files.update(
                        {
                            f"__atrex_bench_runtime.tar.gz.b64.part{index:03d}": path
                            for index, path in enumerate(runtime_part_paths)
                        }
                    )
                    proc = _run_direct_gateway(
                        url=args.url,
                        hardware=args.hardware,
                        timeout=args.timeout,
                        queue_wait_grace=queue_wait_grace,
                        env_items=gateway_environment,
                        files=direct_files,
                        command="bash __atrex_runner.sh",
                        num_gpus=num_gpus,
                    )
                except (OSError, RuntimeError, TimeoutError) as exc:
                    raise RuntimeStateError(
                        "The Supervisor could not complete the Gateway transport request.",
                        code="gateway_unavailable",
                    ) from exc
            else:
                try:
                    proc = _run_agate_with_cancel_retry(
                        agate=agate,
                        executable=agate_executable or "agate",
                        url=args.url,
                        gateway_profile=args.gateway_profile,
                        command_timeout=_dev_gateway_job_timeout(args.timeout),
                        wait_budget=args.timeout + queue_wait_grace,
                    )
                except FileNotFoundError as exc:
                    raise RuntimeStateError(
                        "The Supervisor's Agate client or endpoint is unavailable; "
                        "this is not an Agent dependency.",
                        code="gateway_dependency_unavailable",
                    ) from exc

        hide_evaluator_details = evaluator_command and _is_generalized_workspace(workspace)
        if proc.stderr and not hide_evaluator_details:
            visible_stderr = (
                _agent_retry_notes(proc.stderr)
                if evaluator_command or profile_request
                else proc.stderr.rstrip()
            )
            if visible_stderr:
                print(visible_stderr, file=sys.stderr)
        try:
            job = json.loads(proc.stdout)
        except json.JSONDecodeError:
            print(json.dumps(error_response(
                "Gateway returned no valid job response; the execution outcome is unknown.",
                code="gateway_response_invalid", repairable=False, next_action=ESCALATE_RUNTIME,
            )))
            return proc.returncode or 2
        if not isinstance(job, dict):
            raise RuntimeStateError(
                "Gateway returned a non-object job response.", code="gateway_response_invalid"
            )

        if job.get("status") != "succeeded" and (
            isinstance(job.get("error"), dict) or not isinstance(job.get("result"), dict)
        ):
            failure = {"status": job.get("status"), "error": job.get("error") or {}}
            if not evaluator_command and not profile_request:
                remote = job.get("result")
                remote = remote if isinstance(remote, dict) else {}
                failure.update({
                    "command": command,
                    "exit_code": remote.get("exit_code", proc.returncode or 1),
                    "stdout": remote.get("stdout", ""),
                    "stderr": remote.get("stderr", ""),
                })
            error = job.get("error")
            infrastructure = isinstance(error, dict) and error.get("error_class") == "infra"
            record = task.record(
                failure, gateway_kind="profile" if profile_request else "dev",
                job_id=job.get("job_id"), private_result=job,
                cache=not infrastructure and job.get("status") in {"failed", "cancelled"},
            )
            public = _agent_gateway_failure(job, record, generalized=hide_evaluator_details)
            if not evaluator_command and not profile_request:
                public.update(_agent_dev_probe_result(failure))
            print(json.dumps(public, ensure_ascii=False))
            return proc.returncode or 1

        result = job.get("result") or {}
        remote_stdout = str(result.get("stdout") or "")
        remote_stderr = str(result.get("stderr") or "")
        try:
            if output_transport == "oss":
                artifact = _oss_artifact(job, OSS_OUTPUT_ARCHIVE)
                with tempfile.TemporaryDirectory(prefix="atrex-oss-output-") as temp_dir:
                    archive = Path(temp_dir) / OSS_OUTPUT_ARCHIVE
                    _download_oss_artifact(artifact, archive)
                    _extract_output_archive(archive, workspace)
                command_stdout = remote_stdout.rstrip("\n")
            elif output_transport == "inline":
                command_stdout = _extract_outputs(remote_stdout, workspace)
            elif output_transport == "ssh":
                command_stdout = remote_stdout.rstrip("\n")
            else:
                command_stdout = remote_stdout.rstrip("\n")
        except (RuntimeError, ValueError, tarfile.TarError) as exc:
            if remote_stdout and not hide_evaluator_details:
                print(remote_stdout.rstrip())
            if remote_stderr and not hide_evaluator_details:
                print(remote_stderr.rstrip(), file=sys.stderr)
            print(f"sandbox: {exc}; job_id={job.get('job_id')}", file=sys.stderr)
            return int(result.get("exit_code") or proc.returncode or 2)
        if profile_request:
            profile_result = dict(result)
            profile_result["status"] = job.get("status")
            if command_stdout:
                profile_result["summary"] = _bounded_text(command_stdout[-4000:], 2000)
            profile_record = task.record(
                profile_result,
                gateway_kind="profile",
                job_id=job.get("job_id"),
                private_result=job,
            )
            command_stdout = PROFILE_RESULT_PREFIX + json.dumps(
                _agent_profile_result(profile_result, profile_record),
                ensure_ascii=False,
            )
        if evaluator_command:
            command_stdout = _hydrate_abba_result_lines(workspace, command_stdout)
            command_stdout = _hydrate_result_lines(workspace, command_stdout)
            command_stdout = _record_result_lines(
                workspace, command_stdout, gateway_kind="dev", task=task, job=job,
            )
        remote_rc = result.get("exit_code")
        dev_record: dict[str, str] | None = None
        if not profile_request and (
            not evaluator_command or (task.digest is not None and task.record_id is None)
        ):
            dev_result = _agent_dev_probe_result(
                {
                    "status": job.get("status"),
                    "command": command,
                    "exit_code": (
                        remote_rc
                        if isinstance(remote_rc, int) and not isinstance(remote_rc, bool)
                        else proc.returncode
                    ),
                    "stdout": command_stdout,
                    "stderr": remote_stderr,
                    "synced_paths": sync_paths,
                }
            )
            dev_record = task.record(
                dev_result,
                gateway_kind="dev",
                job_id=job.get("job_id"),
                private_result=job,
            )
        if hide_evaluator_details:
            command_stdout = "\n".join(
                line
                for line in command_stdout.splitlines()
                if line.startswith((TEST_RESULT_PREFIX, ABBA_RESULT_PREFIX))
            )
        if command_stdout:
            print(command_stdout)
        if dev_record is not None:
            print(
                DEV_RECORD_RESULT_PREFIX
                + json.dumps(
                    {
                        "gateway_record_id": dev_record["record_id"],
                        "kernel_id": dev_record["kernel_id"],
                    },
                    ensure_ascii=False,
                )
            )
        if remote_stderr and not hide_evaluator_details:
            has_result = any(
                line.startswith((TEST_RESULT_PREFIX, ABBA_RESULT_PREFIX))
                for line in command_stdout.splitlines()
            )
            remote_rc = result.get("exit_code")
            if (
                (evaluator_command and not has_result)
                or (profile_request and isinstance(remote_rc, int) and remote_rc != 0)
                or (not evaluator_command and not profile_request)
            ):
                print(_bounded_text(remote_stderr.rstrip(), 2000), file=sys.stderr)
        if isinstance(remote_rc, int):
            return remote_rc
        return 0 if job.get("status") == "succeeded" else (proc.returncode or 1)


def _sandbox_telemetry_category(arguments: list[str]) -> str:
    local_kind = _option_value(arguments, "--kind")
    if local_kind == "record-read":
        return "record_read"
    names = {Path(value).name for value in arguments}
    if names & {"profile_nvidia.sh", "profile_kernel.sh"}:
        return "profile"
    if "test_kernel.py" in names:
        return "correctness" if "--multi-seed" in arguments else "benchmark"
    return "dev"


def _append_sandbox_telemetry(event: str, **fields: object) -> None:
    trace = os.environ.get("ATREX_TELEMETRY_TRACE")
    if not trace:
        return
    payload = {
        "schema_version": "atrex_iteration_event_v1",
        "campaign_id": os.environ.get("ATREX_TELEMETRY_CAMPAIGN_ID", "campaign"),
        "iteration_id": os.environ.get("ATREX_TELEMETRY_ITERATION_ID", "unknown"),
        "attempt_id": os.environ.get("ATREX_TELEMETRY_ATTEMPT_ID", "attempt"),
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "monotonic_seconds": time.monotonic(),
        "source": "sandbox",
        "measurement": "exact",
        **fields,
    }
    path = Path(trace)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    arguments = list(argv if argv is not None else sys.argv[1:])
    operation_id = f"sandbox-{os.getpid()}-{time.monotonic_ns()}"
    category = _sandbox_telemetry_category(arguments)
    started = time.monotonic()
    previous_handlers = {
        signum: signal.signal(signum, _interrupt_active_agate_jobs)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    _append_sandbox_telemetry(
        "sandbox_operation_started",
        operation_id=operation_id,
        category=category,
    )
    try:
        returncode = _main(argv)
    except BaseException as exc:
        _cancel_active_agate_jobs()
        _append_sandbox_telemetry(
            "sandbox_operation_completed",
            operation_id=operation_id,
            category=category,
            duration_seconds=round(time.monotonic() - started, 6),
            status="failed",
            failure_type=type(exc).__name__,
        )
        if isinstance(exc, (AgentRequestError, RuntimeStateError)):
            raise SystemExit(json.dumps(exc.response, ensure_ascii=False)) from None
        if isinstance(exc, SystemExit) and isinstance(exc.code, str):
            try:
                value = json.loads(exc.code)
            except ValueError:
                value = None
            if not (isinstance(value, dict) and isinstance(value.get("error"), dict)):
                raise SystemExit(json.dumps(error_response(exc.code), ensure_ascii=False)) from None
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    _append_sandbox_telemetry(
        "sandbox_operation_completed",
        operation_id=operation_id,
        category=category,
        duration_seconds=round(time.monotonic() - started, 6),
        status="succeeded" if returncode == 0 else "failed",
        exit_status=returncode,
    )
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
