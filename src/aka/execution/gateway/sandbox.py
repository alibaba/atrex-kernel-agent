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

    python -m aka.legacy.workflow.measurement_cli --kind run --hardware REMOTE_GPU --no-sync -- python test_kernel.py --no-memory
    python -m aka.legacy.workflow.measurement_cli --kind profile --hardware REMOTE_GPU --sync profiles/v1 -- \
        bash profiling/nvidia/profile_nvidia.sh profile_driver.py --output-dir profiles/v1 --source
    python -m aka.legacy.workflow.measurement_cli --kind profile --hardware REMOTE_ACCELERATOR --gateway-profile pre --sync profiles/v1 -- \
        bash profiling/amd/profile_kernel.sh profile_driver.py --output-dir profiles/v1
    python -m aka.legacy.workflow.measurement_cli --kind run --hardware H20 --ssh gpu-host --no-sync -- \
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
import fcntl
import io
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import Any, Iterable

from aka.core.filesystem import durable_write_json  # noqa: E402
from aka.execution.configuration import queue_wait_grace as configured_queue_wait_grace  # noqa: E402
from aka.execution.health import DEFAULT_SSH_HEALTH_COMMAND, combined_health_command

from aka.execution.command_syntax import (
    _command_executable_index as _command_executable_index,
    command_parts as _command_parts,
    is_profile_command as _is_profile_command,
    is_test_kernel_command as _is_test_kernel_command,
    is_unsafe_target_command as _is_unsafe_target_command,
    _mentions_evaluator_target as _mentions_evaluator_target,
    option_value as _option_value,
    _parsed_command_parts as _parsed_command_parts,
    _python_script_index as _python_script_index,
    requested_operation as _requested_gateway_kind,
    _shell_command_operand as _shell_command_operand,
    _standard_command_name as _standard_command_name,
    test_kernel_script_index as _test_kernel_script_index,
)
from aka.execution.outcomes import emit_fact, emit_diagnostic, backend_attempt


DEFAULT_SYNC_PATHS = ("profiles",)
INPUT_SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".atrex_environment",
    ".atrex_plugins",
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
MAX_HTTP_REQUEST_TIMEOUT = 600
SSH_CONNECT_TIMEOUT = 15
ENVIRONMENT_TEMPFAIL = 75
INFRASTRUCTURE_MARKER = "__ATREX_INFRASTRUCTURE_UNAVAILABLE__"
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
        "profiling/nvidia/profile_nvidia.sh",
        "profiling/nvidia/classify_ncu.py",
    }
)
AMD_PROFILE_TOOL_INPUT_PATHS = frozenset({"profiling/amd/profile_kernel.sh"})
OUTPUT_PATH_FLAGS = frozenset({"-o", "--output", "--output-dir"})
TEST_RESULT_PREFIX = "[test_kernel] RESULT_JSON="
ABBA_RESULT_PREFIX = "__ATREX_LONG_HORIZON_ABBA_RESULT__="
NUMERICAL_RESULT_PREFIX = "__ATREX_NUMERICAL_RESULT__="
PROFILE_RESULT_PREFIX = "[sandbox] PROFILE_JSON="
TYPED_KINDS = frozenset({"run", "profile"})
EPISODE_EVALUATIONS_PATH = ".atrex_long_horizon/evaluations.jsonl"


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"path must be relative to the workspace: {value!r}")
    normalized = path.as_posix()
    if normalized in ("", "."):
        raise ValueError(f"path must not resolve to the workspace root: {value!r}")
    return normalized


def _find_agate() -> str | None:
    """Require an explicit campaign wrapper; auto-discover only when unset."""
    configured = os.environ.get("ATREX_AGATE_EXECUTABLE", "").strip()
    if configured:
        executable = shutil.which(os.path.expanduser(configured))
        if executable is None:
            raise FileNotFoundError(f"ATREX_AGATE_EXECUTABLE is not executable: {configured}")
        return executable
    adjacent = Path(sys.executable).absolute().parent / "agate"
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
    selected_inputs = frozenset(_safe_relative(value) for value in input_paths)

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
            # Explicit Resources mounts may be leaf symlinks. Archive their
            # declared bytes as regular files, never host absolute link targets.
            source = path.resolve(strict=True)
            if not source.is_file():
                return
            size = source.stat().st_size
        except OSError as exc:
            skipped.append(f"{arcname} ({exc})")
            return
        if max_file_bytes is not None and size > max_file_bytes:
            skipped.append(f"{arcname} ({size} bytes > input limit)")
            return
        tf.add(source, arcname=arcname, recursive=False)
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
        if max_file_bytes is not None and len(payload) > max_file_bytes:
            skipped.append(f"{arcname} ({len(payload)} bytes > input limit)")
            return
        info = tarfile.TarInfo(arcname)
        info.size = len(payload)
        info.mode = 0o400
        tf.addfile(info, io.BytesIO(payload))
        seen.add(arcname)
        count += 1

    with tarfile.open(fileobj=archive, mode="w:gz", dereference=True) as tf:
        # Evaluator-only inputs are added before the public workspace tree so a candidate-created
        # file with the same name cannot shadow the orchestrator-owned private test set.
        for arcname, path in (injected_inputs or {}).items():
            add_file(tf, path, arcname)
        for arcname, payload in (injected_payloads or {}).items():
            add_payload(tf, payload, arcname)
        add_tree(tf, workspace)
        # Resources has installed explicit immutable mounts. Materialize only
        # allowlisted members, including members beneath those selected symlinks.
        for arcname in sorted(selected_inputs):
            add_file(tf, workspace / arcname, arcname)
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
        ncu_helpers = workspace / "profiling/nvidia/ncu_helpers"
        if ncu_helpers.is_dir():
            selected.update(
                f"profiling/nvidia/ncu_helpers/{path.relative_to(ncu_helpers).as_posix()}"
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


REQUIREMENTS_LANGUAGES_PREFIX = "# aka-candidate-languages:"


def _candidate_languages(
    solution: dict[str, Any], *, candidate_sources: Iterable[str] = ()
) -> list[str]:
    """Return the candidate languages declared by current or legacy manifests."""
    spec = solution.get("spec")
    languages = spec.get("languages") if isinstance(spec, dict) else None
    if not isinstance(languages, list):
        languages = solution.get("languages")
    if isinstance(languages, list):
        return [str(value) for value in languages]
    # Preserve the pre-refactor fallback for legacy manifests that omitted the
    # language declaration while importing the internal pinned framework.
    if any("flydsl" in source.casefold() for source in candidate_sources):
        return ["flydsl"]
    return []


def _runtime_requirements(
    path: str, *, candidate_languages: Iterable[str] = ()
) -> list[str]:
    """Read runtime pins selected for the candidate's declared languages."""
    try:
        contents = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read runtime requirements file {path}: {exc}") from exc
    if len(contents.encode("utf-8")) > 128 * 1024:
        raise ValueError("runtime requirements file exceeds 128 KiB")
    from aka.execution.environment_query import select_requirements
    return list(select_requirements(contents, candidate_languages))

















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


REMOTE_COLLECTOR = r"""#!/usr/bin/env python3
import base64
import io
import json
import sys
import tarfile
from pathlib import Path, PurePosixPath

BEGIN = "__ATREX_SANDBOX_OUTPUT_BEGIN__"
END = "__ATREX_SANDBOX_OUTPUT_END__"
RAW_DEFAULT = {".ncu-rep", ".att", ".pftrace", ".otf2"}

root = Path(sys.argv[1]).resolve()
cfg = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
max_bytes = int(cfg["max_file_bytes"])
transport = cfg.get("transport", "inline")
include_raw = (bool(cfg["include_raw_profile"]) if cfg["include_raw_profile"] is not None else transport == "oss")
RAW = set(cfg.get("raw_profile_suffixes", RAW_DEFAULT | {".nsys-rep"})) if cfg["include_raw_profile"] is False else RAW_DEFAULT
skipped = []
seen = set()

def safe(value):
    p = PurePosixPath(value)
    return bool(value) and not p.is_absolute() and ".." not in p.parts

def add_file(tf, path):
    rel = path.relative_to(root).as_posix()
    if rel in cfg.get("private_paths", ()) or rel in seen or path.is_symlink() or not path.is_file():
        return
    size = path.stat().st_size
    raw = path.suffix.lower() in RAW or (cfg["include_raw_profile"] is False and any(p.lower() == "att" for p in path.parts))
    if (not include_raw and raw) or size > max_bytes:
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


def _extract_output_tar(tf: tarfile.TarFile, workspace: Path) -> list:
    """Safely extract a sandbox-owned output archive into ``workspace``."""
    workspace_root = workspace.resolve()
    files = []
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
            continue
        if not member.isfile():
            continue
        source = tf.extractfile(member)
        if source is None:
            continue
        files.append((path.as_posix(), source.read(), member.mode & 0o777))


    return files

def _extract_output_archive(archive: Path, workspace: Path) -> None:
    with tarfile.open(archive, mode="r:gz") as tf:
        from .returned_files import emit
        emit(_extract_output_tar(tf, workspace))


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
        from .returned_files import emit
        emit(_extract_output_tar(tf, workspace))
    return command_stdout.rstrip("\n")


def _oss_artifact(job: dict[str, Any], name: str) -> dict[str, Any]:
    artifacts = (job.get("result") or {}).get("artifacts") or []
    for artifact in artifacts:
        if isinstance(artifact, dict) and artifact.get("name") == name:
            return artifact
    raise RuntimeError(f"gateway returned no OSS artifact named {name!r}")


def _download_oss_artifact(artifact: dict[str, Any], destination: Path, *, timeout=None) -> None:
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
    wait_budget = MAX_HTTP_REQUEST_TIMEOUT if timeout is None else timeout
    deadline = time.monotonic() + wait_budget
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
                url, timeout=wait_budget
            ) as response:
                final_url = urllib.parse.urlsplit(response.geturl())
                if final_url.scheme not in {"http", "https"} or not final_url.hostname:
                    raise RuntimeError(
                        f"OSS artifact {name!r} redirected to an invalid URL"
                    )
                while chunk := response.read(1024 * 1024):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("artifact collection exceeded its wait budget")
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
    from aka.execution.cli_options import register_transport_options
    register_transport_options(parser)
    parser.add_argument("--prepared-request-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--prepared-inputs-fd", type=int, help=argparse.SUPPRESS)
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
        "--profile-level",
        choices=("survey", "sol", "deep"),
        default="sol",
        help="Typed profile funnel level (default: sol).",
    )
    parser.add_argument(
        "--profiler",
        choices=("ncu", "nsys", "rocprofv3", "acu"),
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
            emit_fact({"event": "submitted", "identifier": remote_dir, "backend": "ssh", "operation": "dev"})
            result = subprocess.run(
                [*_ssh_base(ssh, target), remote_command],
                capture_output=True,
                text=True,
                timeout=timeout + SSH_CONNECT_TIMEOUT + 15,
            )
        except subprocess.TimeoutExpired as exc:
            raise SSHTransportError(f"SSH command wait timed out: {exc}") from exc
        # An SSH client exit is a command observation, not proof of remote process-tree termination.
        emit_fact({"event": "attempt", "identifier": remote_dir, "backend": "ssh", "operation": "dev",
                   "disposition": "unknown", "termination": None,
                   "exit_code": result.returncode if result.returncode != 255 else None, "diagnostics": []})

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
        emit_fact({"event": "cleanup", "status": "completed" if cleanup_succeeded else "failed",
                   "code": "ssh-workspace-cleanup"})
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


def _auth_headers(environment=None) -> dict[str, str]:
    """Generate token or AK/SK headers matching agate's auth precedence."""
    import hashlib

    environment = os.environ if environment is None else environment
    private_token = environment.get("AGATE_TOKEN", "")
    if private_token:
        return {"Authorization": f"Bearer {private_token}"}
    ak = environment.get("AGATE_AK", "")
    sk = environment.get("AGATE_SK", "")
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
    *, allow_list: bool = False, environment=None,
) -> dict | list:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = dict(_auth_headers(environment))
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
    if not isinstance(result, dict) and not (allow_list and isinstance(result, list)):
        raise RuntimeError("gateway returned a non-object JSON response")
    return result


def _run_direct_job(
    *,
    url: str,
    kind: str,
    payload: dict[str, Any],
    timeout: int,
    queue_wait_grace: int,
    on_submitted=None,
) -> subprocess.CompletedProcess[str]:
    """Submit and wait for any public gateway job kind through HTTP."""
    import uuid
    from .jobs import GatewayJobs
    from aka.execution.outcomes import observe_attempt
    attempt_id = uuid.uuid4().hex
    endpoint, profile = _submission_route(url)
    identity = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    key = payload.get("idempotency_key")
    deadline = time.monotonic() + timeout + queue_wait_grace
    emit_fact({"event": "attempt_started", "attempt_id": attempt_id, "operation": kind, "backend": "gateway"})
    emit_fact({"event": "submission_intent", "attempt_id": attempt_id, "request_identity": identity,
               "idempotency_key": key})
    try:
        accepted = _gateway_json(endpoint or url, "POST", f"/v1/jobs/{kind}", payload,
                                 min(30, max(0, deadline - time.monotonic())))
    except GatewayHTTPError as error:
        try:
            native = json.loads(error.detail)
        except (TypeError, ValueError):
            native = error.detail
        if isinstance(native, dict):
            native = native.get("error", native)
        observe_attempt(subprocess.CompletedProcess((), 1, json.dumps({"error": native}), f"agate: [{error.status}]"),
                        operation=kind, attempt_id=attempt_id)
        raise
    job_id = accepted.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        raise RuntimeError("gateway submission returned no confirmed job identity")
    emit_fact({"event": "submitted", "attempt_id": attempt_id, "identifier": job_id,
               "operation": kind, "backend": "gateway", "endpoint": endpoint, "profile": profile,
               "request_identity": identity, "idempotency_key": key})
    if on_submitted is not None:
        on_submitted(accepted)
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"gateway job {job_id} exceeded client timeout")
            wait_for = min(30.0, remaining)
            job = _gateway_json(endpoint or url, "GET",
                f"/v1/jobs/{job_id}?wait=true&timeout={wait_for:.3f}", None, min(wait_for + 10, remaining))
            if job.get("status") in ("succeeded", "failed", "cancelled"):
                process = subprocess.CompletedProcess(args=["direct-gateway", kind, job_id],
                    returncode=0 if job.get("status") == "succeeded" else 1, stdout=json.dumps(job), stderr="")
                observe_attempt(process, operation=kind, attempt_id=attempt_id)
                return process
    except BaseException:
        emit_fact({"event": "cancel_requested", "identifier": job_id})
        from aka.contracts.execution_recovery import JobReference
        reference = JobReference("gateway", job_id, kind, endpoint or url, profile, identity, key)
        jobs = GatewayJobs(endpoint=reference.endpoint, profile=profile, http=_gateway_json)
        _observe_cancel_cleanup(jobs.cancel(reference, timeout=10), job_id, kind, attempt_id)
        raise


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
            "timeout_s": _dev_gateway_job_timeout(timeout),
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


def _report_infrastructure_failure(job: dict) -> bool:
    """Preserve the failure category without revealing private evaluator details."""
    from .protocol import terminal_cancellation
    if terminal_cancellation(job):
        return False
    error = job.get("error")
    if (isinstance(error, dict) and (error.get("error_class") or error.get("class")) == "infra"
            or _queue_timeout_before_start(job) or _cancelled_without_outcome(job)):
        emit_diagnostic({"failure_class": "infrastructure", "reason": "transport-unavailable"})
        print(INFRASTRUCTURE_MARKER, file=sys.stderr)
        return True
    return False


def _report_cancellation(job: dict) -> bool:
    """Keep terminal cancellation visible when evaluator content is withheld."""
    from .protocol import protocol_failure, terminal_cancellation
    if not terminal_cancellation(job):
        return False
    diagnostic = protocol_failure(
        subprocess.CompletedProcess([], 1, json.dumps(job), ""), lambda _: None)
    emit_diagnostic(diagnostic)
    print("[sandbox] gateway_protocol_failure=" + json.dumps(diagnostic, sort_keys=True),
          file=sys.stderr, flush=True)
    return True


def _report_agate_transport_failure(proc: subprocess.CompletedProcess) -> bool:
    """Inspect CLI transport diagnostics only when no remote job envelope exists."""
    if _job_response(proc.stdout or "") is not None:
        return False
    detail = proc.stderr or ""
    if (proc.returncode == ENVIRONMENT_TEMPFAIL and INFRASTRUCTURE_MARKER in detail.splitlines()
            or re.search(r"\bHTTP(?: Error)?\s*[:=]?\s*(429|502|503|504)\b", detail)
            or any(message in detail for message in (
                "Connection refused", "Connection reset by peer", "Temporary failure in name resolution"))):
        emit_diagnostic({"failure_class": "infrastructure", "reason": "transport-unavailable"})
        print(INFRASTRUCTURE_MARKER, file=sys.stderr)
        return True
    return False


def _ray_submit_version_mismatch(job: dict | None) -> bool:
    from .protocol import terminal_cancellation
    if terminal_cancellation(job):
        return False
    error = job.get("error") if job else None
    return bool(
        isinstance(error, dict)
        and error.get("reason") == "submit_failed"
        and "Version check returned 404" in str(error.get("message", ""))
    )


def _queue_timeout_before_start(job: dict | None) -> bool:
    from .protocol import terminal_cancellation
    if terminal_cancellation(job):
        return False
    error = job.get("error") if job else None
    return bool(
        isinstance(error, dict)
        and error.get("reason") == "timeout"
        and "never started executing" in str(error.get("message", ""))
    )


def _submitted_job_id(proc: subprocess.CompletedProcess[str]) -> str | None:
    """Recover the job id printed by agate before it starts polling."""
    match = SUBMITTED_JOB_RE.search((proc.stderr or "") + "\n" + (proc.stdout or ""))
    return match.group(1) if match else None


def _track_agate_job(
    job_id: str, executable: str, url: str, gateway_profile: str | None, operation: str = "unknown",
    request_identity: str = "",
) -> None:
    endpoint, profile = _submission_route(url, gateway_profile)
    emit_fact({"event": "submitted", "identifier": job_id, "backend": "gateway", "operation": operation,
               "endpoint": endpoint, "profile": profile, "request_identity": request_identity})
    with ACTIVE_AGATE_JOBS_LOCK:
        ACTIVE_AGATE_JOBS[job_id] = (executable, endpoint, profile)


def _submission_route(url="", profile=None):
    """Use the selected provider route even when the CLI flags use a profile alias."""
    from .jobs import selected_route
    route = json.loads(os.environ.get("AKA_EXECUTION_GATEWAY_ROUTE", "{}"))
    if route.get("endpoint"):
        return route["endpoint"], route.get("profile", "")
    endpoint, profile, _ = selected_route(url, profile or "", environment=os.environ,
                                        executable=_find_agate())
    return endpoint, profile


def _observe_cancel_cleanup(result, job_id, operation="cancel", attempt_id=None):
    from aka.execution.outcomes import observe_attempt
    if result.stdout:
        observe_attempt(subprocess.CompletedProcess((), result.transport_returncode, result.stdout, ""),
                        operation=operation, attempt_id=attempt_id)
    failed = result.facts.remote_cleanup == "failed" or result.facts.disposition == "unknown"
    emit_fact({"event": "cleanup", "identifier": job_id,
               "status": "failed" if failed else "completed", "code": "cancel-confirmation-incomplete"})
    for item in result.diagnostics:
        emit_diagnostic({"failure_class": "cleanup", "reason": item.code, "job_id": job_id})


def _forget_agate_job(job_id: str) -> None:
    with ACTIVE_AGATE_JOBS_LOCK:
        ACTIVE_AGATE_JOBS.pop(job_id, None)


def _cancel_active_agate_jobs() -> None:
    with ACTIVE_AGATE_JOBS_LOCK:
        jobs = list(ACTIVE_AGATE_JOBS.items())
    for job_id, (executable, url, gateway_profile) in jobs:
        emit_fact({"event": "cancel_requested", "identifier": job_id})
        try:
            from .jobs import GatewayJobs
            from aka.contracts.execution_recovery import JobReference
            reference = JobReference("gateway", job_id, "cancel", url, gateway_profile or "")
            jobs = GatewayJobs(endpoint=url, profile=gateway_profile or "", executable=executable)
            _observe_cancel_cleanup(jobs.cancel(reference, timeout=10), job_id)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            emit_fact({"event": "cleanup", "identifier": job_id, "status": "failed", "code": type(error).__name__})


def _interrupt_active_agate_jobs(_signum: int, _frame: object) -> None:
    _cancel_active_agate_jobs()
    raise KeyboardInterrupt


def _gateway_job_timeout(command_timeout: int, queue_wait_grace: int) -> int:
    """Queue grace extends client waiting; only advertised limits bound runtime."""
    from .requests import effective_budget, hard_limit
    return effective_budget(command_timeout, limit=hard_limit(os.environ, "run")).effective


def _dev_gateway_job_timeout(command_timeout: int) -> int:
    """Respect this backend's declared dev ceiling without imposing a default."""
    from .requests import effective_budget, hard_limit
    return effective_budget(command_timeout, limit=hard_limit(os.environ, "dev")).effective


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


def _run_agate_once_unchecked(
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
    # Large concurrent upload/submit requests can overload gateway ingress before
    # jobs reach its GPU queue. Serialize only this short admission phase across
    # local sandbox processes; release before polling so GPU jobs remain parallel.
    endpoint = url or gateway_profile or os.environ.get("AGATE_URL") or "default"
    key = hashlib.sha256(endpoint.encode()).hexdigest()[:16]
    files = []
    for argument in agate[2:]:
        path = Path(argument.split("=", 1)[-1])
        try:
            if path.is_file():
                files.append((argument, hashlib.sha256(path.read_bytes()).hexdigest()))
        except (OSError, ValueError):
            pass
    identity = hashlib.sha256(json.dumps({"argv": agate, "files": files}, sort_keys=True).encode()).hexdigest()
    emit_fact({"event": "submission_intent", "request_identity": identity, "idempotency_key": None})
    lock_path = Path(tempfile.gettempdir()) / f"atrex-agate-submit-{os.getuid()}-{key}.lock"
    with lock_path.open("a") as admission:
        fcntl.flock(admission, fcntl.LOCK_EX)
        remaining = wait_budget - (time.monotonic() - wait_started)
        if remaining <= 0:
            return subprocess.CompletedProcess(agate, ENVIRONMENT_TEMPFAIL, "", INFRASTRUCTURE_MARKER)
        try:
            submitted = subprocess.run(
                [*agate, "--no-wait"], capture_output=True, text=True,
                timeout=min(MAX_HTTP_REQUEST_TIMEOUT, remaining))
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(agate, ENVIRONMENT_TEMPFAIL, "", INFRASTRUCTURE_MARKER)
    job = _job_response(submitted.stdout or "")
    job_id = job.get("job_id") if job else _submitted_job_id(submitted)
    if not isinstance(job_id, str) or not job_id:
        return submitted
    _track_agate_job(job_id, executable, url, gateway_profile, agate[1], identity)
    endpoint, profile = _submission_route(url, gateway_profile)
    try:
        completed = _resume_interrupted_agate_wait(
            executable=executable,
            url=endpoint,
            gateway_profile=profile,
            command_timeout=command_timeout,
            wait_budget=wait_budget,
            elapsed=time.monotonic() - wait_started,
            initial=submitted,
        )
        final_job = _job_response(completed.stdout or "")
        if final_job and final_job.get("status") in {"succeeded", "failed", "cancelled"}:
            _forget_agate_job(job_id)
        return completed
    except BaseException:
        # The signal/final cleanup handler still owns this accepted job.
        raise


def _run_agate_with_cancel_retry(
    *,
    agate: list[str],
    executable: str,
    url: str,
    gateway_profile: str | None,
    command_timeout: int,
    wait_budget: int,
) -> subprocess.CompletedProcess[str]:
    """Submit once; Experiment owns any new measurement attempt."""
    return _run_agate_once(agate=agate, executable=executable, url=url,
        gateway_profile=gateway_profile, command_timeout=command_timeout, wait_budget=wait_budget)


def _typed_agate_command(
    executable: str,
    args: argparse.Namespace,
    workspace: Path,
    kind: str,
    request: dict[str, Any],
    queue_wait_grace: int,
    reference_dir: Path | None = None,
    *, requirements_file: Path | None = None, init_kwargs_file: Path | None = None,
) -> list[str]:
    """Build an agate run/profile invocation for a typed request."""
    command = [executable, kind]
    if args.url:
        command += ["--url", args.url]
    elif args.gateway_profile:
        command += ["--profile", args.gateway_profile]
    from .requests import operation_request
    request = operation_request(request, kind)
    options = request.get("options", {})
    # Generalized workspaces deliberately expose only agent_problem.json to the
    # optimization agent.  The agate client still needs the evaluator-owned
    # shapes/reference files locally to assemble its typed eval payload, so point
    # --reference-dir at the private source while keeping the candidate in the
    # public workspace.  The private directory is never copied into the workspace.
    reference_dir = (reference_dir or workspace) if kind in {"run", "profile"} else workspace
    command += ["--gpu", args.hardware]
    num_gpus = request.get("spec", {}).get("num_gpus", 1)
    if num_gpus > 1 and kind in {"run", "profile"}:
        command += ["--num-gpus", str(num_gpus)]
    if kind == "run" and request.get("atrex_bench_version"):
        command += ["--atrex-bench-version", request["atrex_bench_version"]]
    requirements_file = str(requirements_file) if requirements_file is not None else os.environ.get("ATREX_SANDBOX_REQUIREMENTS_FILE", "").strip()
    if requirements_file:
        requirements = _runtime_requirements(
            requirements_file,
            candidate_languages=request.get("spec", {}).get("languages", ()),
        )
        # CLI submission reloads this file; refuse to silently change the runtime
        # captured in a persisted typed request (the direct API uses that snapshot).
        if requirements != request.get("requirements", []):
            raise ValueError(
                f"runtime requirements file {requirements_file} differs from the typed request"
            )
        if requirements:
            command += ["--requirements", requirements_file]
    elif request.get("requirements"):
        raise ValueError(
            "ATREX_SANDBOX_REQUIREMENTS_FILE is required to replay the typed request's runtime pins"
        )
    command += ([str(workspace / "kernel.py")] if kind == "check" else ["--candidate", str(workspace / "kernel.py")])
    if kind in {"run", "profile"}:
        command += ["--reference-dir", str(reference_dir), "--operator", str(request["reference"]["operator"])]
    if kind == "run":
        command += ["--mode", str(request.get("mode") or "full"),
                    "--num-correctness-cases", str(options["num_correctness_cases"]),
                    "--bench-iters", str(options["bench_iters"])]
        warmup = os.environ.get("AKA_GATEWAY_WARMUP_ITERS", "5")
        if warmup != "default":
            command += ["--set", "warmup_iters=" + warmup]
        if "abba" in request:
            from .requests import validate_native_abba
            validate_native_abba(request)
            baseline = workspace / "baseline.py"
            baseline.write_text(request["abba"]["baseline"], encoding="utf-8")
            command += ["--baseline", str(baseline), "--abba-repeats", str(request["abba"]["repeats"])]
    elif kind == "profile" and "bench_iters" in options:
        command += ["--bench-iters", str(options["bench_iters"])]
    command += ["--http-timeout", str(MAX_HTTP_REQUEST_TIMEOUT),
                "--wait-timeout", str(args.timeout + queue_wait_grace)]
    if kind in {"run", "profile"}:
        command += ["--job-timeout", str(_gateway_job_timeout(args.timeout, queue_wait_grace))]
    if kind == "check":
        for field in ("arch", "sanitize"):
            if request.get(field):
                command += ["--" + field, str(request[field])]
    if kind == "disassemble" and request.get("fmt"):
        command += ["--fmt", str(request["fmt"])]
    if init_kwargs_file is not None and kind in {"check", "disassemble"}:
        command += ["--init-kwargs", str(init_kwargs_file)]
    correctness_max_rel_l2 = options.get("correctness_max_rel_l2")
    if kind == "run" and correctness_max_rel_l2 is not None:
        command += [
            "--set",
            f"correctness_max_rel_l2={json.dumps(correctness_max_rel_l2)}",
        ]
    if kind == "run":
        # Keep the Evaluation policy explicit even for replayed older requests.
        # The extra torch.profiler pass is reserved for explicit profiling.
        command += ["--set", "skip_kernel_attribution=true"]
    for item in args.env:
        command += ["--env-var", item]
    if kind == "profile":
        command += ["--level", request.get("level", args.profile_level)]
        profiler = request.get("profiler", args.profiler)
        if profiler:
            command += ["--profiler", profiler]
        for counter in request.get("counters", args.profile_counter):
            command += ["--counter", counter]
        kernel_regex = request.get("kernel_regex", args.kernel_regex)
        if kernel_regex:
            command += ["--kernel-regex", kernel_regex]
        top_kernels = request.get("top_kernels", args.top_kernels)
        if top_kernels is not None:
            command += ["--top-kernels", str(top_kernels)]
    return command





def _main(argv: list[str] | None = None, *, parsed_arguments=None) -> int:
    args = build_parser().parse_args(argv) if parsed_arguments is None else parsed_arguments
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
    if args.kind == "same_allocation_abba" and (
            args.prepared_request_fd is None or args.ssh or args.preflight or args.check_health):
        raise SystemExit("sandbox: same-allocation ABBA requires a prepared typed gateway request")
    if args.prepared_inputs_fd is not None and (
            args.ssh or args.prepared_request_fd is not None or args.kind in {"check", "disassemble"}
            or args.preflight or args.check_health):
        raise SystemExit("sandbox: selected transport or operation cannot accept closed prepared inputs")
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
    if args.timeout < 1:
        raise SystemExit("sandbox: --timeout must be positive")
    if args.shape_batch_size is not None and args.shape_batch_size <= 0:
        raise SystemExit("sandbox: --shape-batch-size must be positive")
    try:
        queue_wait_grace = configured_queue_wait_grace(os.environ)
    except ValueError as exc:
        raise SystemExit(f"sandbox: {exc}") from exc
    from .requests import effective_budget
    effective_budget(args.timeout + (0 if args.ssh else queue_wait_grace), scope="wait")
    if args.ssh:
        effective_budget(args.timeout, scope="command")
    if args.max_input_file_mb <= 0 or args.max_output_file_mb <= 0 or (args.max_output_file_bytes is not None and args.max_output_file_bytes <= 0):
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
        command = "" if args.prepared_request_fd is not None else _command_text(args.command)
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

    if args.prepared_request_fd is not None:
        from .submission import run_prepared
        return run_prepared(args, workspace, sync_paths, queue_wait_grace)
    if args.kind in {"check", "disassemble"}:
        from .submission import run_kernel_operation
        return run_kernel_operation(args, workspace, queue_wait_grace)
    gateway_kind = "dev"
    evaluator_command = _is_test_kernel_command(args.command)
    if evaluator_command:
        command = (
            f"AKA_TARGET_HARDWARE={shlex.quote(args.hardware)} " + command
        )
    profile_command = _is_profile_command(args.command)
    profile_request = gateway_kind == "profile" or profile_command
    num_gpus = args.num_gpus
    typed_limitation = None
    if num_gpus < 1:
        raise SystemExit("sandbox: --num-gpus must be positive")
    if profile_request and num_gpus > 1:
        raise SystemExit("sandbox: distributed profile commands require the typed profile route")
    if args.ssh and evaluator_command and num_gpus > 1:
        raise SystemExit("sandbox: distributed evaluator commands are not supported by the SSH runner")

    private_paths = ()
    if args.prepared_inputs_fd is not None:
        if args.ssh or args.prepared_request_fd is not None:
            raise SystemExit("sandbox: closed prepared inputs do not support SSH or typed requests")
        from aka.execution.prepared_inputs import read as read_prepared_inputs, archive
        prepared = read_prepared_inputs(args.prepared_inputs_fd)
        limit = args.max_input_file_mb * 1024 * 1024
        bundle = archive(prepared.files, max_file_bytes=limit)
        runtime_bundle = archive(prepared.runtime_files, max_file_bytes=None) if prepared.runtime_files else None
        file_count, skipped = len(prepared.files), []
        private_paths = prepared.private_paths
    else:
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
                selected_inputs = _command_input_paths(workspace, args.command, args.input)
            except ValueError as exc:
                raise SystemExit(f"sandbox: {exc}") from exc
        try:
            bundle, file_count, skipped = _make_input_bundle(
                workspace, args.max_input_file_mb * 1024 * 1024, selected_inputs)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise SystemExit(f"sandbox: cannot prepare evaluator inputs: {exc}") from exc
        runtime_bundle = _make_atrex_bench_runtime_bundle(workspace, evaluator_only=True) if evaluator_command else None
    bundle_bytes = len(bundle.encode("ascii"))
    runtime_bundle_bytes = len(runtime_bundle.encode("ascii")) if runtime_bundle else 0
    gateway_environment = list(args.env)
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

    from aka.execution.command_transport import RAW_PROFILE_SUFFIXES
    output_cfg = {
        "paths": sync_paths,
        "private_paths": private_paths,
        "max_file_bytes": args.max_output_file_bytes if args.max_output_file_bytes is not None else args.max_output_file_mb * 1024 * 1024,
        "include_raw_profile": args.include_raw_profile,
        "raw_profile_suffixes": RAW_PROFILE_SUFFIXES,
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
        from .recovery import _patch_dev_command_script
        _patch_dev_command_script(command_path, workspace)
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
            f"aka.execution.gateway.sandbox {args.kind} command path",
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
            except GatewayHTTPError as exc:
                if exc.status in {429, 502, 503, 504}:
                    emit_diagnostic({"failure_class": "infrastructure", "reason": "transport-unavailable"})
                    print(INFRASTRUCTURE_MARKER, file=sys.stderr)
                    return ENVIRONMENT_TEMPFAIL
                raise SystemExit(f"sandbox: direct gateway request failed: {exc}") from exc
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

    hide_evaluator_details = evaluator_command and bool(private_paths)
    if proc.stderr and not hide_evaluator_details:
        print("\n".join(line for line in proc.stderr.rstrip().splitlines()
                              if line != INFRASTRUCTURE_MARKER), file=sys.stderr)
    try:
        job = json.loads(proc.stdout)
    except json.JSONDecodeError:
        if _report_agate_transport_failure(proc):
            return ENVIRONMENT_TEMPFAIL
        if proc.stdout and not hide_evaluator_details:
            print(proc.stdout.rstrip())
        elif hide_evaluator_details:
            print(
                "sandbox: generalized gateway response unavailable; evaluator details withheld",
                file=sys.stderr,
            )
        return proc.returncode or 2

    if _report_cancellation(job):
        return proc.returncode or 1
    if _report_infrastructure_failure(job):
        return ENVIRONMENT_TEMPFAIL
    result = job.get("result") or {}
    remote_stdout = str(result.get("stdout") or "")
    remote_stderr = "\n".join(line for line in str(result.get("stderr") or "").splitlines()
                              if line != INFRASTRUCTURE_MARKER)
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


def main(argv: list[str] | None = None, *, parsed_arguments=None) -> int:
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
        returncode = (_main(argv) if parsed_arguments is None
                      else _main(argv, parsed_arguments=parsed_arguments))
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


def _run_agate_once(*args, **kwargs):
    from .protocol import protocol_failure
    from aka.execution.outcomes import observe_attempt
    agate = kwargs.get("agate", ())
    operation = agate[1] if len(agate) > 1 else "unknown"
    with backend_attempt(operation):
        process = _run_agate_once_unchecked(*args, **kwargs)
        observe_attempt(process, operation=operation)
        diagnostic = protocol_failure(process, _submitted_job_id)
        if diagnostic is not None:
            emit_diagnostic(diagnostic)
            print("[sandbox] gateway_protocol_failure=" + json.dumps(diagnostic, sort_keys=True), file=sys.stderr, flush=True)
        return process


if __name__ == "__main__":
    raise SystemExit(main())
