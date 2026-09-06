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

"""Run optimizer GPU work through a gateway or an OpenSSH GPU host.

In gateway mode, native Atrex-Bench correctness/performance commands use
``agate run`` and profiling commands use ``profile``. ``dev`` remains the
compatibility escape hatch for workloads those typed interfaces cannot represent
(for example SOL-ExecBench, source-correlated custom profiling, or a community
gateway that explicitly returns ``kind_not_supported``). OpenSSH mode executes
the same allowlisted command bundle through a portable remote runner. Every
invocation is stateless; callers must not rely on remote filesystem persistence.

Examples::

    python tools/sandbox.py --kind run --hardware REMOTE_GPU --no-sync -- python test_kernel.py --no-memory
    python tools/sandbox.py --kind profile --hardware REMOTE_GPU --sync profiles/v1 -- \
        bash tools/profile_nvidia.sh profile_driver.py --output-dir profiles/v1 --source
    python tools/sandbox.py --kind profile --hardware REMOTE_ACCELERATOR --gateway-profile pre --sync profiles/v1 -- \
        bash tools/profile_kernel.sh profile_driver.py --output-dir profiles/v1
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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orchestrator.durable_state import durable_write_json  # noqa: E402
from orchestrator.ssh_health import (  # noqa: E402
    DEFAULT_SSH_HEALTH_COMMAND,
    combined_health_command,
)

DEFAULT_SYNC_PATHS = ("profiles",)
INPUT_SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".atrex_environment",
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
TYPED_KINDS = frozenset({"run", "profile"})
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
MODE_STATE_FILENAME = ".orchestrator_mode.json"
PRIVATE_REFERENCE_ENV = "ATREX_PRIVATE_REFERENCE_DIR"
PRIVATE_EVALUATOR_FILENAMES = ("shapes.json", "metadata.json", "roofline.json")
PRIVATE_PROFILE_CASE_FILENAME = ".atrex_private_profile_case.json"
EPISODE_EVALUATIONS_PATH = ".atrex_long_horizon/evaluations.jsonl"
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


def _find_agate() -> str | None:
    """Find agate beside the active Python before consulting the shell PATH."""
    adjacent = Path(sys.executable).resolve().parent / "agate"
    if adjacent.is_file() and os.access(adjacent, os.X_OK):
        return str(adjacent)
    return shutil.which("agate")


def _uses_standard_oss_gateway(
    agate_executable: str, *, url: str, profile: str | None
) -> bool:
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
        add_tree(tf, workspace)
        # Optimization workspaces receive tools/ as a symlink.  Materialize the
        # small tool directory so remote profile commands are self-contained.
        workspace_tools = workspace / "tools"
        if workspace_tools.is_symlink() or not workspace_tools.exists():
            add_tree(tf, REPO_ROOT / "tools", "tools")
        # ``skills/`` is normally a runtime symlink and is intentionally skipped during the
        # workspace walk.  Materialize only explicitly selected skill files so a profiling
        # snapshot can use its backend on the worker without uploading every installed skill.
        if any(path.startswith("skills/") for path in selected_inputs):
            add_tree(tf, REPO_ROOT / "skills", "skills")
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
            raise RuntimeError(
                "workspace solution.json sources must be a list of paths"
            )
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


def _evaluation_input_paths(
    workspace: Path, command: Iterable[str] = ()
) -> frozenset[str]:
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
    selected = {
        path for path in CANDIDATE_RUNTIME_INPUT_PATHS if (workspace / path).is_file()
    }
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
            f"{normalized}/{path.relative_to(source).as_posix()}"
            for path in _walk_files(source)
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
            "profile_nvidia.sh",
            "profile_kernel.sh",
            "extract_ttgir.py",
        }
        or any("harness" in PurePosixPath(path).parts for path in selected)
    )
    if candidate_command:
        selected.update(_candidate_runtime_input_paths(workspace))

    if "profile_nvidia.sh" in basenames:
        selected.update(NVIDIA_PROFILE_TOOL_INPUT_PATHS)
        ncu_helpers = REPO_ROOT / "tools" / "ncu_helpers"
        if ncu_helpers.is_dir():
            selected.update(
                f"tools/ncu_helpers/{path.relative_to(ncu_helpers).as_posix()}"
                for path in _walk_files(ncu_helpers)
            )
    if "profile_kernel.sh" in basenames:
        selected.update(AMD_PROFILE_TOOL_INPUT_PATHS)

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


def _standard_command_name(value: str, names: set[str]) -> str | None:
    """Return a command name only for PATH lookup or a conventional system path."""
    name = Path(value).name
    if name in names and value in {name, f"/bin/{name}", f"/usr/bin/{name}"}:
        return name
    return None


def _command_executable_index(
    command: list[str], *, typed_launcher: bool = False
) -> int | None:
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
                option.startswith(("--chdir=", "--unset="))
                and option.partition("=")[2]
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
                if (
                    re.fullmatch(r"-(?:n)?\d+", option)
                    or option.startswith("--adjustment=")
                ):
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
        nested = _command_executable_index(
            command[index:], typed_launcher=typed_launcher
        )
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

    if (
        index is None
        or re.fullmatch(r"python(?:3(?:\.\d+)*)?", Path(command[index]).name) is None
    ):
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


def _test_kernel_script_index(
    parts: list[str], *, typed_launcher: bool = False
) -> int | None:
    return _python_script_index(
        parts, "test_kernel.py", typed_launcher=typed_launcher
    )


def _is_test_kernel_command(parts: list[str]) -> bool:
    return _test_kernel_script_index(parts) is not None


def _shell_command_operand(
    command: list[str], executable_index: int
) -> tuple[str, int] | None:
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
    if _python_script_index(command, "profile_driver.py") is not None:
        return True
    index = _command_executable_index(command)
    if index is None:
        return False
    frontend = _standard_command_name(command[index], {"ncu", "nsys", "rocprofv3"})
    if frontend is not None and (
        frontend != "nsys"
        or (index + 1 < len(command) and command[index + 1] == "profile")
    ):
        for nested_index in range(index + 1, len(command)):
            if (
                _python_script_index(
                    command[nested_index:], "profile_driver.py"
                )
                is not None
            ):
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
    return re.search(
        r"(?<![A-Za-z0-9_.-])"
        r"(?:test_kernel\.py|profile_driver\.py|profile_nvidia\.sh|profile_kernel\.sh)"
        r"(?![A-Za-z0-9_.-])",
        unquoted,
    ) is not None


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
        raise ValueError(
            "benchmark_contract.distributed_evaluation.launcher must be 'torchrun'"
        )
    if evaluation.get("backend") != "nccl":
        raise ValueError(
            "benchmark_contract.distributed_evaluation.backend must be 'nccl'"
        )
    world_size = evaluation.get("world_size")
    if isinstance(world_size, bool) or not isinstance(world_size, int):
        raise ValueError(
            "benchmark_contract.distributed_evaluation.world_size must be an integer"
        )
    if world_size != 2:
        raise ValueError(
            "benchmark_contract.distributed_evaluation.world_size must be 2"
        )
    return world_size


def _workspace_num_gpus(workspace: Path) -> int:
    metadata = _json_object(
        _evaluator_input_path(workspace, "metadata.json", required=False)
    )
    return _distributed_evaluation_world_size(metadata)


def _is_fp4_dtype(value: object) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.casefold().replace("-", "").replace("_", "")
    return "fp4" in normalized or "float4" in normalized


def _metadata_has_fp4_dtype(metadata: object) -> bool:
    if isinstance(metadata, dict):
        if any(
            _is_fp4_dtype(metadata.get(field))
            for field in ("dtype", "dtype_compute")
        ):
            return True
        return any(_metadata_has_fp4_dtype(value) for value in metadata.values())
    if isinstance(metadata, list):
        return any(_metadata_has_fp4_dtype(value) for value in metadata)
    return False


def _fp4_correctness_max_rel_l2(
    metadata: dict[str, Any] | None,
    operator: object = None,
) -> float | None:
    return (
        FP4_MAX_REL_L2
        if _metadata_has_fp4_dtype(metadata) or _is_fp4_dtype(operator)
        else None
    )


def _is_generalized_workspace(workspace: Path) -> bool:
    """Return whether production policy enables private exact-case handling."""
    state = _json_object(workspace / MODE_STATE_FILENAME) or {}
    return (
        state.get("mode") == "production"
        and (workspace / AGENT_PROBLEM_FILENAME).is_file()
    )


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
    if private_problem.is_file() and (
        private_problem.read_bytes() != public_problem.read_bytes()
    ):
        raise ValueError(
            "workspace agent_problem.json does not match the evaluator-owned public contract"
        )
    return private_dir


def _evaluator_input_path(workspace: Path, filename: str, *, required: bool) -> Path:
    private_dir = _private_reference_dir(workspace)
    path = (
        (private_dir / filename) if private_dir is not None else (workspace / filename)
    )
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
    workspace: Path, env_items: Iterable[str]
) -> tuple[str, bytes] | None:
    """Materialize exactly one private real shape for an ephemeral remote profile."""
    private_dir = _private_reference_dir(workspace)
    if private_dir is None:
        return None
    shapes = _json_object(private_dir / "shapes.json", required=True)
    if not shapes:
        raise ValueError("private shapes.json must contain a non-empty object")
    environment = _parse_env_items(env_items)
    shape_id = environment.get("PROFILE_SHAPE_ID") or sorted(
        (str(value) for value in shapes), key=_sort_shape_id
    )[0]
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


def _typed_workspace_limitation(
    workspace: Path, command: list[str], kind: str
) -> str | None:
    """Explain why the typed run/profile source contract cannot represent a workspace."""
    required = ("kernel.py", "reference.py", "input.py")
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
    if kind == "profile" and _is_generalized_workspace(workspace):
        return "generalized tasks inject one private real shape through the dev profile route"
    if (workspace / "workload.jsonl").is_file():
        return (
            "SOL-ExecBench workload.jsonl is not supported by the Atrex-Bench typed API"
        )

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
                imported_roots.update(
                    alias.name.split(".", 1)[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
        local_imports = sorted(
            root
            for root in imported_roots
            if root not in {"input", "reference", "kernel"}
            and (
                (workspace / f"{root}.py").is_file()
                or (workspace / root / "__init__.py").is_file()
            )
        )
        if local_imports:
            return "candidate imports local auxiliary modules: " + ", ".join(
                local_imports
            )

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
    top_kernels: int | None = None,
) -> dict[str, Any]:
    """Build the public run/profile request without importing the agate package."""
    shapes = _json_object(
        _evaluator_input_path(workspace, "shapes.json", required=True), required=True
    )
    assert shapes is not None
    requested_shape_ids = _option_values(command, "--shape-id")
    if requested_shape_ids:
        unknown_shape_ids = [
            shape_id for shape_id in requested_shape_ids if shape_id not in shapes
        ]
        if unknown_shape_ids:
            raise ValueError(
                "unknown --shape-id values: " + ", ".join(unknown_shape_ids)
            )
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
        raise ValueError(
            "--multi-seed must be non-negative and --timed-runs must be positive"
        )

    solution = _json_object(workspace / "solution.json") or {}
    languages = solution.get("languages")
    if not isinstance(languages, list):
        languages = []
    reference: dict[str, Any] = {
        "operator": workspace.name,
        "reference_py": (workspace / "reference.py").read_text(encoding="utf-8"),
        "input_py": (workspace / "input.py").read_text(encoding="utf-8"),
        "shapes": shapes,
    }
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

    spec: dict[str, Any] = {
        "languages": [str(value) for value in languages],
        "target_hardware": [hardware],
    }
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
        reference.get("metadata")
        if isinstance(reference.get("metadata"), dict)
        else None,
        reference.get("operator"),
    )
    if kind == "run" and correctness_max_rel_l2 is not None:
        request["options"]["correctness_max_rel_l2"] = correctness_max_rel_l2
    if kind == "run":
        version = str(_option_value(command, "--version", "v0"))
        request["mode"] = (
            "correctness_only"
            if multi_seed > 0 and version not in {"v0", "v1"}
            else "full"
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
        if top_kernels is not None:
            request["top_kernels"] = top_kernels
    return request


def _make_atrex_bench_runtime_bundle(
    workspace: Path, *, evaluator_only: bool = False
) -> str | None:
    """Package the canonical native evaluator separately from workspace state.

    The compressed runtime is split into multiple uploaded files by ``main``
    because agate's worker places each file value in one Linux argv entry.
    """
    runtime_link = workspace / "atrex-bench"
    if not runtime_link.is_dir():
        return None
    runtime_root = runtime_link
    run_eval = runtime_root / "scripts" / "run_eval.py"
    package = runtime_root / "src" / "atrex_bench"
    utils_module = package / "utils.py"
    sdk_module = package / "sdk.py"
    if (
        not package.is_dir()
        or not run_eval.is_file()
        or (evaluator_only and not utils_module.is_file())
    ):
        raise RuntimeError(
            f"invalid workspace Atrex-Bench runtime link: {runtime_link}"
        )

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
            raise RuntimeError(
                f"unsafe artifact path returned by sandbox: {member.name!r}"
            )
        if member.issym() or member.islnk():
            raise RuntimeError(
                f"sandbox artifact links are not accepted: {member.name!r}"
            )
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
            with urllib.request.urlopen(
                url, timeout=MAX_HTTP_REQUEST_TIMEOUT
            ) as response:
                final_url = urllib.parse.urlsplit(response.geturl())
                if final_url.scheme not in {"http", "https"} or not final_url.hostname:
                    raise RuntimeError(
                        f"OSS artifact {name!r} redirected to an invalid URL"
                    )
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run correctness, performance, or profile commands on a remote GPU sandbox.",
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
        choices=("auto", "run", "profile", "dev"),
        default="auto",
        help=(
            "Gateway interface to use. auto routes test_kernel.py to run, profiler "
            "wrappers to profile, and other commands to dev (default: auto). Typed "
            "jobs fall back to dev only when their source contract is unsupported."
        ),
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
        default=os.environ.get(
            "ATREX_SANDBOX_HEALTH_COMMAND", DEFAULT_SSH_HEALTH_COMMAND
        ),
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
        default=int(
            os.environ.get("ATREX_SANDBOX_TIMEOUT", str(DEFAULT_COMMAND_TIMEOUT))
        ),
        help=(
            "Remote command execution timeout in seconds, 1..600 "
            "(default: 600; queue wait is budgeted separately)."
        ),
    )
    parser.add_argument(
        "--shape-batch-size",
        type=int,
        default=int(
            os.environ.get(
                "ATREX_EVAL_SHAPE_BATCH_SIZE", str(DEFAULT_EVAL_SHAPE_BATCH_SIZE)
            )
        ),
        help="Maximum Atrex-Bench shapes per concurrent eval job (default: 4).",
    )
    parser.add_argument(
        "--sync",
        action="append",
        default=[],
        metavar="PATH",
        help="Relative profile/result path to copy back (repeatable; default: profiles).",
    )
    parser.add_argument(
        "--no-sync", action="store_true", help="Do not copy any files back."
    )
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
        help="Typed profile kernel regex (required by a deep profile).",
    )
    parser.add_argument(
        "--top-kernels",
        type=int,
        default=None,
        help="Limit the typed profile result to the N hottest kernels.",
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
        paths[0] == root or paths[0].startswith(root + "/")
        for root in source_forbidden_trees
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


def _resolve_ssh_runtime_binds(
    ssh: str, target: str, runtime_binds: list[str]
) -> list[str]:
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
    if not isinstance(resolved, list) or len(resolved) != len(parsed) or not all(
        isinstance(item, str) for item in resolved
    ):
        raise SSHTransportError("SSH runtime bind resolver returned an invalid path list")
    validated: list[str] = []
    for source, (_declared_source, destination) in zip(resolved, parsed):
        try:
            validated_source, validated_destination = _ssh_runtime_bind(
                f"{source}={destination}"
            )
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
                shlex.join(
                    ["python3", "-c", SSH_GPU_RESOLVER_SOURCE, str(gpu_index)]
                ),
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
        args.extend(
            ("--dir", "/atrex", "--bind", remote_dir, "/atrex", "--chdir", "/atrex")
        )
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
                warning = (
                    "[sandbox] output archive unavailable after failed command: "
                    f"{detail}"
                )
                result = subprocess.CompletedProcess(
                    args=result.args,
                    returncode=result.returncode,
                    stdout=result.stdout,
                    stderr="\n".join(
                        part for part in (result.stderr, warning) if part
                    ),
                )
            else:
                try:
                    _extract_output_archive(local_archive, workspace)
                except (OSError, RuntimeError, tarfile.TarError) as exc:
                    raise SSHTransportError(
                        f"cannot extract sandbox outputs: {exc}"
                    ) from exc
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

    private_token = os.environ.get("AGATE_TOKEN", "")
    if private_token:
        return {"Authorization": f"Bearer {private_token}"}
    ak = os.environ.get("AGATE_AK", "")
    sk = os.environ.get("AGATE_SK", "")
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
    """Submit and wait for any public gateway job kind through HTTP."""
    prior_note = ""
    for submission in range(2):
        accepted = _gateway_json(url, "POST", f"/v1/jobs/{kind}", payload, 30)
        job_id = accepted.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise RuntimeError(f"gateway submission returned no job_id: {accepted}")
        deadline = time.monotonic() + timeout + queue_wait_grace
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"gateway job {job_id} exceeded client timeout")
                wait_for = min(30.0, remaining)
                job = _gateway_json(
                    url,
                    "GET",
                    f"/v1/jobs/{job_id}?wait=true&timeout={wait_for:.3f}",
                    None,
                    wait_for + 10,
                )
                if job.get("status") in ("succeeded", "failed", "cancelled"):
                    if submission == 0 and _cancelled_without_outcome(job):
                        prior_note = (
                            f"[sandbox] gateway cancelled job_id={job_id} without a "
                            "result/error; resubmitted once"
                        )
                        break
                    return subprocess.CompletedProcess(
                        args=["direct-gateway", kind, job_id],
                        returncode=0 if job.get("status") == "succeeded" else 1,
                        stdout=json.dumps(job),
                        stderr=prior_note,
                    )
        except BaseException:
            try:
                _gateway_json(url, "POST", f"/v1/jobs/{job_id}/cancel", {}, 10)
            except Exception:
                pass
            raise

    raise AssertionError(
        "unreachable: direct gateway retry loop returned no terminal job"
    )


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
            "files": {
                name: path.read_text(encoding="utf-8") for name, path in files.items()
            },
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
        job
        and job.get("status") == "cancelled"
        and not job.get("result")
        and not job.get("error")
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


def _track_agate_job(
    job_id: str, executable: str, url: str, gateway_profile: str | None
) -> None:
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
    job_id = (
        initial_job.get("job_id") if initial_job else _submitted_job_id(initial)
    )
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
    """Retry transient gateway-only failures without blaming the submitted kernel."""
    deadline = time.monotonic() + wait_budget
    first = _run_agate_once(
        agate=agate,
        executable=executable,
        url=url,
        gateway_profile=gateway_profile,
        command_timeout=command_timeout,
        wait_budget=wait_budget,
    )
    first_job = _job_response(first.stdout or "")
    fallback = (
        _l20n_failover_command(agate)
        if _ray_submit_version_mismatch(first_job)
        else None
    )
    if fallback is not None:
        agate = fallback
        print(
            "[sandbox] L20N submit cluster unavailable; retrying on l20n-ray",
            file=sys.stderr,
            flush=True,
        )
        first = _run_agate_once(
            agate=agate,
            executable=executable,
            url=url,
            gateway_profile=gateway_profile,
            command_timeout=command_timeout,
            wait_budget=max(1, int(deadline - time.monotonic())),
        )
        first_job = _job_response(first.stdout or "")
    while _ray_submit_version_mismatch(first_job) or _queue_timeout_before_start(
        first_job
    ):
        remaining = int(deadline - time.monotonic())
        if remaining <= 1:
            return first
        delay = min(60, remaining - 1) if _ray_submit_version_mismatch(first_job) else 0
        if delay:
            print(
                f"[sandbox] gateway Ray submit compatibility outage; retrying in {delay}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
        else:
            print(
                "[sandbox] gateway queue timeout before job start; resubmitting",
                file=sys.stderr,
                flush=True,
            )
        first = _run_agate_once(
            agate=agate,
            executable=executable,
            url=url,
            gateway_profile=gateway_profile,
            command_timeout=command_timeout,
            wait_budget=remaining - delay,
        )
        first_job = _job_response(first.stdout or "")
    if not _cancelled_without_outcome(first_job):
        return first

    first_job_id = first_job.get("job_id")
    second = _run_agate_once(
        agate=agate,
        executable=executable,
        url=url,
        gateway_profile=gateway_profile,
        command_timeout=command_timeout,
        wait_budget=wait_budget,
    )
    note = (
        f"[sandbox] gateway cancelled job_id={first_job_id} without a result/error; "
        "resubmitted once"
    )
    stderr_parts = [
        part.rstrip() for part in (first.stderr, note, second.stderr) if part
    ]
    return subprocess.CompletedProcess(
        args=second.args,
        returncode=second.returncode,
        stdout=second.stdout,
        stderr="\n".join(stderr_parts),
    )


def _typed_agate_command(
    executable: str,
    args: argparse.Namespace,
    workspace: Path,
    kind: str,
    request: dict[str, Any],
    queue_wait_grace: int,
    reference_dir: Path | None = None,
) -> list[str]:
    """Build an agate run/profile invocation for a typed request."""
    command = [executable, kind]
    if args.url:
        command += ["--url", args.url]
    elif args.gateway_profile:
        command += ["--profile", args.gateway_profile]
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
        command += ["--num-gpus", str(num_gpus)]
    command += [
        "--candidate",
        str(workspace / "kernel.py"),
        "--reference-dir",
        str(reference_dir),
        "--operator",
        str(request["reference"]["operator"]),
        "--mode",
        str(request.get("mode") or "full"),
        "--num-correctness-cases",
        str(options["num_correctness_cases"]),
        "--bench-iters",
        str(options["bench_iters"]),
        "--set",
        "warmup_iters=5",
        "--http-timeout",
        str(MAX_HTTP_REQUEST_TIMEOUT),
        "--wait-timeout",
        str(args.timeout + queue_wait_grace),
        "--job-timeout",
        str(_gateway_job_timeout(args.timeout, queue_wait_grace)),
    ]
    correctness_max_rel_l2 = options.get("correctness_max_rel_l2")
    if correctness_max_rel_l2 is not None:
        command += [
            "--set",
            f"correctness_max_rel_l2={json.dumps(correctness_max_rel_l2)}",
        ]
    for item in args.env:
        command += ["--env-var", item]
    if kind == "profile":
        command += ["--level", args.profile_level]
        if args.profiler:
            command += ["--profiler", args.profiler]
        for counter in args.profile_counter:
            command += ["--counter", counter]
        if args.kernel_regex:
            command += ["--kernel-regex", args.kernel_regex]
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
        shape_ids[offset : offset + batch_size]
        for offset in range(0, len(shape_ids), batch_size)
    ]


def _shape_batch_request(
    request: dict[str, Any], shape_ids: list[str]
) -> dict[str, Any]:
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


def _shape_batch_reference(
    reference: dict[str, Any], destination: Path
) -> None:
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
            + str(
                compile_result.get("reason")
                or compile_result.get("status")
                or "did not pass"
            )
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


def _compile_diagnostics(
    compile_result: object, shape_ids: list[str]
) -> list[dict[str, str]]:
    """Extract compile/import failures that are safe and useful in generalized mode."""
    if not isinstance(compile_result, dict):
        return []
    if "status" in compile_result:
        if compile_result.get("status") == "passed":
            return []
        message = _bounded_actionable_diagnostic(
            compile_result.get("reason") or compile_result.get("status")
        )
        return (
            [{"stage": "compile_import", "shape_id": "", "message": message}]
            if message
            else []
        )

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
        if (value := _finite_number(entry.get("performance_us"))) is not None
        and value > 0.0
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
    correctness_status = (
        correctness_status if isinstance(correctness_status, dict) else {}
    )
    correctness = payload.get("correctness")
    correctness = correctness if isinstance(correctness, dict) else {}
    correctness_shapes = correctness.get("shapes")
    correctness_shapes = (
        correctness_shapes if isinstance(correctness_shapes, dict) else {}
    )
    actionable_diagnostics.extend(
        _candidate_exception_diagnostics(
            correctness_status, correctness_shapes, shape_ids
        )
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
        performance_shapes = (
            performance_shapes if isinstance(performance_shapes, dict) else {}
        )
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
        latency_by_shape[shape_id]
        for shape_id in shape_ids
        if shape_id in latency_by_shape
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
        latency_by_shape[shape_id]
        for shape_id in shape_ids
        if shape_id in latency_by_shape
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
    failures = [
        str(failure)
        for result in results
        for failure in (result.get("failures") or [])
    ]
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
        "max_abs_err": max(
            float(result.get("max_abs_err") or 0.0) for result in results
        ),
        "max_rel_err": max(
            float(result.get("max_rel_err") or 0.0) for result in results
        ),
        "evaluator": "atrex-gpu-gateway/run/batched",
        "eval_id": results[-1].get("eval_id"),
        "shape_batch_count": len(results),
        "actionable_diagnostics": actionable_diagnostics,
    }


def _mask_generalized_result(
    workspace: Path, result: dict[str, Any]
) -> dict[str, Any]:
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
    return masked


def _record_episode_evaluation(
    workspace: Path,
    result: dict[str, Any],
    *,
    gateway_kind: str,
    job_id: object = None,
) -> None:
    """Persist exact optimizer-facing results for terminal memory construction.

    Long-horizon workers use ``--no-memory`` because canonical memory belongs to the
    supervisor.  Keep their complete evaluator results in excluded episode runtime
    state so a pivot or interruption can still record this round's real per-shape
    performance without parsing model-authored journal prose.
    """
    runtime = workspace / ".atrex_long_horizon"
    if not (runtime / "journal.json").is_file():
        return
    try:
        kernel_sha256 = hashlib.sha256((workspace / "kernel.py").read_bytes()).hexdigest()
    except OSError:
        kernel_sha256 = None
    payload = {
        "schema_version": 2,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gateway_kind": gateway_kind,
        "job_id": str(job_id) if job_id else None,
        "kernel_sha256": kernel_sha256,
        "result": result,
    }
    path = workspace / EPISODE_EVALUATIONS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, line)
    finally:
        os.close(descriptor)


def _record_result_lines(workspace: Path, stdout: str, *, gateway_kind: str) -> None:
    """Record the last ordinary RESULT_JSON emitted by a dev evaluator command."""
    for line in reversed(stdout.splitlines()):
        if not line.startswith(TEST_RESULT_PREFIX):
            continue
        try:
            result = json.loads(line[len(TEST_RESULT_PREFIX) :])
        except json.JSONDecodeError:
            return
        if isinstance(result, dict):
            _record_episode_evaluation(
                workspace, result, gateway_kind=gateway_kind
            )
        return


def _positive_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number > 0.0 and math.isfinite(number) else None


def _with_workspace_performance_score(
    workspace: Path, result: dict[str, Any]
) -> dict[str, Any]:
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
    if (
        not isinstance(baseline_by_shape, dict)
        or set(candidate_by_shape) != set(baseline_by_shape)
    ):
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
            ABBA_RESULT_PREFIX
            + json.dumps(updated_payload, ensure_ascii=False, allow_nan=False)
        )
    return "\n".join(normalized)


def _record_profile_job(
    job: dict[str, Any], workspace: Path, sync_paths: list[str]
) -> None:
    """Persist the typed profile response where the local optimization session expects it."""
    for relative in sync_paths:
        path = PurePosixPath(relative)
        if not path.parts or path.parts[0] != "profiles":
            continue
        output_dir = workspace / path.as_posix()
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / "gateway_profile.json"
        target.write_text(
            json.dumps(job, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        for artifact in (job.get("result") or {}).get("artifacts") or []:
            if not isinstance(artifact, dict):
                continue
            try:
                artifact_name = _safe_relative(str(artifact.get("name") or ""))
            except ValueError as exc:
                raise RuntimeError(f"invalid profile artifact name: {exc}") from exc
            _download_oss_artifact(artifact, output_dir / artifact_name)


def _run_typed_gateway(
    args: argparse.Namespace,
    workspace: Path,
    command_parts: list[str],
    kind: str,
    sync_paths: list[str],
    queue_wait_grace: int,
) -> int | None:
    """Run agate run/profile, returning None only for a documented dev fallback."""
    generalized = _is_generalized_workspace(workspace)
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
            top_kernels=args.top_kernels,
        )
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        print(
            f"[sandbox] {kind} interface unsupported for this workspace: {exc}; using dev",
            file=sys.stderr,
        )
        return None

    expected_shape_ids = _request_shape_ids(request)
    shape_batches = (
        _shape_batches(
            expected_shape_ids,
            getattr(args, "shape_batch_size", DEFAULT_EVAL_SHAPE_BATCH_SIZE),
        )
        if kind == "run"
        else [expected_shape_ids]
    )
    batched = len(shape_batches) > 1

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
                    "fallback_kind": "dev",
                    "candidate_bytes": len(request["candidate"].encode("utf-8")),
                    "shape_count": (
                        "private"
                        if _is_generalized_workspace(workspace)
                        else len(request["reference"]["shapes"])
                    ),
                    "shape_batch_count": len(shape_batches),
                    "mode": request.get("mode"),
                    "options": request["options"],
                    "sync": sync_paths,
                },
                indent=2,
            )
        )
        return 0

    agate_executable = _find_agate()
    try:
        with tempfile.TemporaryDirectory(prefix="atrex-shape-batches-") as temp_dir:
            batch_root = Path(temp_dir)

            def run_batch(
                item: tuple[int, list[str]],
            ) -> subprocess.CompletedProcess[str]:
                batch_index, shape_ids = item
                batch_request = (
                    _shape_batch_request(request, shape_ids) if batched else request
                )
                if args.url and agate_executable is None:
                    return _run_direct_job(
                        url=args.url,
                        kind="eval" if kind == "run" else kind,
                        payload=batch_request,
                        timeout=args.timeout,
                        queue_wait_grace=queue_wait_grace,
                    )
                if agate_executable is None:
                    raise FileNotFoundError("agate")
                reference_dir = None
                if kind == "run":
                    # Even a single targeted batch needs its filtered reference
                    # directory. Otherwise the agate CLI would reload the original
                    # full shapes.json and defeat --shape-id smoke selection.
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
                )
                return _run_agate_with_cancel_retry(
                    agate=agate,
                    executable=agate_executable,
                    url=args.url,
                    gateway_profile=args.gateway_profile,
                    command_timeout=_gateway_job_timeout(
                        args.timeout, queue_wait_grace
                    ),
                    wait_budget=args.timeout + queue_wait_grace,
                )

            batch_items = list(enumerate(shape_batches))
            if batched:
                print(
                    f"[sandbox] running {len(shape_batches)} shape batches "
                    f"with {min(DEFAULT_EVAL_BATCH_WORKERS, len(shape_batches))} workers",
                    file=sys.stderr,
                    flush=True,
                )
                with ThreadPoolExecutor(
                    max_workers=min(DEFAULT_EVAL_BATCH_WORKERS, len(shape_batches))
                ) as executor:
                    processes = list(executor.map(run_batch, batch_items))
            else:
                processes = [run_batch(batch_items[0])]
    except GatewayHTTPError as exc:
        if _typed_fallback_allowed(exc):
            print(
                f"[sandbox] gateway {kind} interface unavailable ({exc}); using dev",
                file=sys.stderr,
            )
            return None
        if generalized:
            raise SystemExit(
                f"sandbox: generalized {kind} gateway request failed; "
                "hidden-case details withheld"
            ) from exc
        raise SystemExit(f"sandbox: {kind} gateway request failed: {exc}") from exc
    except FileNotFoundError as exc:
        raise SystemExit(
            "sandbox: agate not found and no explicit --url was provided; "
            "install atrex-gateway-client first"
        ) from exc

    jobs: list[dict[str, Any]] = []
    for proc in processes:
        detail = (proc.stderr or "") + (proc.stdout or "")
        if proc.returncode and _typed_fallback_allowed(detail):
            print(
                f"[sandbox] gateway {kind} interface rejected this request; using dev",
                file=sys.stderr,
            )
            return None
        if proc.stderr and not generalized:
            print(proc.stderr.rstrip(), file=sys.stderr)
        job = _job_response(proc.stdout or "")
        if job is None:
            if proc.stdout and not generalized:
                print(proc.stdout.rstrip())
            elif generalized:
                print(
                    "[sandbox] generalized gateway response unavailable; evaluator details withheld",
                    file=sys.stderr,
                )
            return proc.returncode or 2
        if job.get("status") != "succeeded" or not isinstance(job.get("result"), dict):
            if generalized:
                print(
                    "[sandbox] generalized evaluation failed; hidden-case details withheld; "
                    f"job_id={job.get('job_id')}",
                    file=sys.stderr,
                )
            else:
                print(json.dumps(job, ensure_ascii=False))
            return proc.returncode or 1
        jobs.append(job)

    print(
        f"[sandbox] gateway_kind={kind} "
        + (
            f"job_ids={','.join(str(job.get('job_id')) for job in jobs)}"
            if batched
            else f"job_id={jobs[0].get('job_id')}"
        ),
        file=sys.stderr,
    )
    if kind == "run":
        metadata = request["reference"].get("metadata")
        require_performance = request.get("mode") != "correctness_only"
        batch_results = [
            _optimizer_result_from_eval(
                job["result"],
                shape_ids,
                metadata,
                require_performance=require_performance,
            )
            for job, shape_ids in zip(jobs, shape_batches)
        ]
        result = _mask_generalized_result(
            workspace,
            _merge_optimizer_results(
                batch_results,
                expected_shape_ids,
                metadata,
                require_performance=require_performance,
            )
            if batched
            else batch_results[0],
        )
        _record_episode_evaluation(
            workspace,
            result,
            gateway_kind=kind,
            job_id=",".join(str(job.get("job_id")) for job in jobs),
        )
        print(
            TEST_RESULT_PREFIX + json.dumps(result, ensure_ascii=False, allow_nan=False)
        )
        return 0 if result["all_pass"] else 1

    _record_profile_job(jobs[0], workspace, sync_paths)
    print(PROFILE_RESULT_PREFIX + json.dumps(jobs[0]["result"], ensure_ascii=False))
    return 0


def _main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.hardware:
        raise SystemExit("sandbox: --hardware or ATREX_SANDBOX_GPU is required")
    # Explicit endpoint flags override inherited sandbox endpoint variables.  This
    # matters when a long-lived optimization shell switches between a remote
    # profile and localhost without first scrubbing its environment.
    explicit_endpoints = sum(
        value is not None for value in (args.url, args.gateway_profile, args.ssh)
    )
    if explicit_endpoints > 1:
        raise SystemExit(
            "sandbox: --ssh, --url, and --gateway-profile are mutually exclusive"
        )
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
                args.ssh_gpu
                if args.ssh_gpu is not None
                else os.environ.get(SSH_GPU_ENV, "")
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
            "sandbox: --timeout must be in the gateway-supported range "
            f"1..{MAX_COMMAND_TIMEOUT}"
        )
    if args.shape_batch_size <= 0:
        raise SystemExit("sandbox: --shape-batch-size must be positive")
    try:
        queue_wait_grace = int(
            os.environ.get(
                "ATREX_SANDBOX_QUEUE_WAIT_GRACE", str(DEFAULT_QUEUE_WAIT_GRACE)
            )
        )
    except ValueError as exc:
        raise SystemExit(
            "sandbox: ATREX_SANDBOX_QUEUE_WAIT_GRACE must be an integer"
        ) from exc
    if queue_wait_grace < 0:
        raise SystemExit("sandbox: ATREX_SANDBOX_QUEUE_WAIT_GRACE must be non-negative")
    if args.max_input_file_mb <= 0 or args.max_output_file_mb <= 0:
        raise SystemExit("sandbox: file size limits must be positive")
    args.health_command = combined_health_command(
        args.health_command, args.runtime_health_command
    )
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
        command = _command_text(args.command)
        sync_paths = (
            []
            if args.no_sync
            else [
                _safe_relative(path) for path in (args.sync or list(DEFAULT_SYNC_PATHS))
            ]
        )
    except ValueError as exc:
        raise SystemExit(f"sandbox: {exc}") from exc

    workspace = Path(args.workspace).resolve()
    if any(PurePosixPath(path).parts[0] == "memory" for path in sync_paths):
        raise SystemExit(
            "sandbox: memory/ is local optimizer state and cannot be synchronized"
        )
    if not workspace.is_dir():
        raise SystemExit(f"sandbox: workspace not found: {workspace}")
    if _is_unsafe_target_command(args.command):
        raise SystemExit(
            "sandbox: evaluator and profile targets must use a supported launcher "
            "with separate arguments after --"
        )

    gateway_kind = _requested_gateway_kind(args.kind, args.command)
    evaluator_command = _is_test_kernel_command(args.command)
    profile_command = _is_profile_command(args.command)
    profile_request = gateway_kind == "profile" or profile_command
    if profile_command:
        try:
            args.env = _with_inherited_profile_environment(args.env)
        except ValueError as exc:
            raise SystemExit(f"sandbox: {exc}") from exc
    typed_limitation: str | None = None
    typed_fallback_kind: str | None = None
    num_gpus = 1
    if gateway_kind in TYPED_KINDS or evaluator_command or profile_request:
        try:
            num_gpus = _workspace_num_gpus(workspace)
        except ValueError as exc:
            raise SystemExit(
                f"sandbox: invalid distributed evaluator contract: {exc}"
            ) from exc
    if gateway_kind in TYPED_KINDS:
        if args.ssh:
            typed_limitation = "SSH uses the portable sandbox command runner"
        elif (
            gateway_kind == "profile"
            and args.profile_level == "deep"
            and not args.kernel_regex
        ):
            raise SystemExit("sandbox: --profile-level deep requires --kernel-regex")
        elif args.keep_pod:
            typed_limitation = "--keep-pod is only supported by dev"
        elif args.input:
            typed_limitation = "custom --input files are only supported by dev"
        elif gateway_kind == "profile" and args.include_raw_profile:
            typed_limitation = (
                "--include-raw-profile requires the custom dev profiler wrapper"
            )
        else:
            try:
                typed_limitation = _typed_workspace_limitation(
                    workspace, args.command, gateway_kind
                )
            except ValueError as exc:
                typed_limitation = str(exc)
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
            typed_limitation = f"gateway {gateway_kind} route unavailable or rejected the source contract"
        typed_fallback_kind = gateway_kind
        gateway_kind = "dev"

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
        injected_inputs = (
            _private_evaluator_inputs(workspace) if evaluator_command else {}
        )
        injected_payloads: dict[str, bytes] = {}
        if profile_command and _is_generalized_workspace(workspace):
            profile_case = _private_profile_case(workspace, args.env)
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
            workspace,
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
        command_environment, gateway_environment = _profile_command_environment(
            args.env
        )
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
    oss_workspace = bool(
        standard_oss_gateway and bundle_bytes > OSS_WORKSPACE_THRESHOLD_BYTES
    )
    workspace_transport = (
        "ssh"
        if args.ssh
        else ("oss" if oss_workspace else ("http" if direct_http else "inline"))
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
            for index, offset in enumerate(
                range(0, len(bundle), WORKSPACE_CHUNK_BYTES)
            ):
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
            runtime_chunk_bytes = (
                len(runtime_bundle) if args.ssh else RUNTIME_CHUNK_BYTES
            )
            for index, offset in enumerate(
                range(0, len(runtime_bundle), runtime_chunk_bytes)
            ):
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
                raise SystemExit(
                    f"sandbox: direct gateway request failed: {exc}"
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
                raise SystemExit(
                    "sandbox: agate not found and no explicit --url was provided; "
                    "install atrex-gateway-client first"
                ) from exc

    hide_evaluator_details = evaluator_command and _is_generalized_workspace(workspace)
    if proc.stderr and not hide_evaluator_details:
        print(proc.stderr.rstrip(), file=sys.stderr)
    try:
        job = json.loads(proc.stdout)
    except json.JSONDecodeError:
        if proc.stdout and not hide_evaluator_details:
            print(proc.stdout.rstrip())
        elif hide_evaluator_details:
            print(
                "sandbox: generalized gateway response unavailable; evaluator details withheld",
                file=sys.stderr,
            )
        return proc.returncode or 2

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
    if evaluator_command:
        command_stdout = _hydrate_abba_result_lines(workspace, command_stdout)
        command_stdout = _hydrate_result_lines(workspace, command_stdout)
        _record_result_lines(workspace, command_stdout, gateway_kind="dev")
    if hide_evaluator_details:
        command_stdout = "\n".join(
            line
            for line in command_stdout.splitlines()
            if line.startswith((TEST_RESULT_PREFIX, ABBA_RESULT_PREFIX))
        )
    if command_stdout:
        print(command_stdout)
    if remote_stderr and not hide_evaluator_details:
        print(remote_stderr.rstrip(), file=sys.stderr)
    remote_rc = result.get("exit_code")
    if isinstance(remote_rc, int):
        return remote_rc
    return 0 if job.get("status") == "succeeded" else (proc.returncode or 1)


def _sandbox_telemetry_category(arguments: list[str]) -> str:
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
