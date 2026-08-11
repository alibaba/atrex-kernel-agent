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

"""Run optimizer GPU work through the matching atrex-gpu-gateway interface.

Native Atrex-Bench correctness/performance commands use ``agate run`` and profiling
commands use ``profile``.  ``dev`` remains the compatibility escape hatch for
workloads those typed interfaces cannot represent (for example SOL-ExecBench,
source-correlated custom profiling or a
community gateway that explicitly returns ``kind_not_supported``).  A new pod
may be selected for every invocation; callers must not rely on remote filesystem
persistence.

Examples::

    python tools/sandbox.py --kind run --hardware REMOTE_GPU --no-sync -- python test_kernel.py --no-memory
    python tools/sandbox.py --kind profile --hardware REMOTE_GPU --sync profiles/v1 -- \
        bash tools/profile_nvidia.sh profile_driver.py --output-dir profiles/v1 --source
    python tools/sandbox.py --kind profile --hardware REMOTE_ACCELERATOR --gateway-profile pre --sync profiles/v1 -- \
        bash tools/profile_kernel.sh profile_driver.py --output-dir profiles/v1

``ATREX_SANDBOX_GPU``, ``ATREX_SANDBOX_PROFILE``, ``ATREX_SANDBOX_URL``, and
``ATREX_SANDBOX_TIMEOUT`` provide defaults for the corresponding flags.  A
localhost gateway uses the same transport as a remote worker, for example
``ATREX_SANDBOX_GPU=local`` plus
``ATREX_SANDBOX_URL=http://127.0.0.1:8000``.  Authentication and any remaining
URL resolution stay agate's responsibility (AGATE_* or ~/.atrex/config.json).
"""

from __future__ import annotations

import argparse
import ast
import base64
import io
import json
import math
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SYNC_PATHS = ("profiles",)
INPUT_SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
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
    # Plans and humanize state are local campaign inputs for the agent, never
    # runtime inputs for the command executing in the GPU pod.  In particular,
    # preserved implementation patches can be large enough to push agate's
    # single uploaded-file argument past Linux MAX_ARG_STRLEN.
    "plans",
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
    ".pyc", ".pyo", ".ncu-rep", ".att", ".pftrace", ".otf2",
    # Campaign documentation, plans, and prior profile reports are local agent
    # state.  Remote correctness/profile commands only need executable sources
    # and harness inputs; omitting Markdown also keeps agate's uploaded file
    # arguments below the worker's argv size limit on long-running campaigns.
    ".md",
}
OUTPUT_BEGIN = "__ATREX_SANDBOX_OUTPUT_BEGIN__"
OUTPUT_END = "__ATREX_SANDBOX_OUTPUT_END__"
DEFAULT_COMMAND_TIMEOUT = 600
MAX_COMMAND_TIMEOUT = 600
DEFAULT_QUEUE_WAIT_GRACE = 14_400
MAX_HTTP_REQUEST_TIMEOUT = 600
RUNTIME_CHUNK_BYTES = 20 * 1024
# Workspace bundle uses a smaller chunk size because agate embeds the file
# content in one argv entry alongside other metadata, and the total must stay
# below Linux MAX_ARG_STRLEN (128 KiB).
WORKSPACE_CHUNK_BYTES = 20 * 1024
SUBMITTED_JOB_RE = re.compile(r"\bsubmitted job_id=([A-Za-z0-9_.-]+); polling\.\.\.")
EVALUATION_INPUT_PATHS = frozenset(
    {
        "definition.json",
        "input.py",
        "kernel.py",
        "metadata.json",
        "reference.py",
        "shapes.json",
        "solution.json",
        "test_kernel.py",
        "workload.jsonl",
    }
)
CANDIDATE_RUNTIME_INPUT_PATHS = frozenset(
    {
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
PROFILE_RESULT_PREFIX = "[sandbox] PROFILE_JSON="
TYPED_KINDS = frozenset({"run", "profile"})
TYPED_FALLBACK_REASONS = (
    "kind_not_supported",
    "invalid_source",
    "source validation failed",
    "http 404",
    "http 413",
    "http 501",
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


def _walk_files(root: Path) -> Iterable[Path]:
    """Yield regular files below root without following directory symlinks."""
    for current, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [
            name for name in dirs
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

    with tarfile.open(fileobj=archive, mode="w:gz") as tf:
        add_tree(tf, workspace)
        # Optimization workspaces receive tools/ as a symlink.  Materialize the
        # small tool directory so remote profile commands are self-contained.
        workspace_tools = workspace / "tools"
        if workspace_tools.is_symlink() or not workspace_tools.exists():
            add_tree(tf, REPO_ROOT / "tools", "tools")
    return base64.b64encode(archive.getvalue()).decode("ascii"), count, skipped


def _declared_candidate_sources(workspace: Path) -> set[str]:
    """Return candidate sources declared by solution.json plus verification artifacts."""
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

    verification_sources = workspace / "verification_artifacts"
    if verification_sources.is_dir():
        for path in _walk_files(verification_sources):
            selected.add(path.relative_to(workspace).as_posix())

    return selected


def _evaluation_input_paths(workspace: Path) -> frozenset[str]:
    """Return only files required by the immutable evaluator."""
    return frozenset(set(EVALUATION_INPUT_PATHS) | _declared_candidate_sources(workspace))


def _candidate_runtime_input_paths(workspace: Path) -> set[str]:
    """Return candidate and workload modules needed by profile/import commands."""
    selected = {
        path for path in CANDIDATE_RUNTIME_INPUT_PATHS
        if (workspace / path).is_file()
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


def _command_parts(parts: list[str]) -> list[str]:
    return parts[1:] if parts and parts[0] == "--" else list(parts)


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
        or basenames & {
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


def _is_test_kernel_command(parts: list[str]) -> bool:
    command = _command_parts(parts)
    return (
        len(command) >= 2
        and Path(command[0]).name in {"python", "python3", "python3.10", "python3.12"}
        and Path(command[1]).name == "test_kernel.py"
    )


def _is_profile_command(parts: list[str]) -> bool:
    """Return whether argv invokes one of the repository profiler wrappers."""
    return any(
        Path(token).name in {"profile_nvidia.sh", "profile_kernel.sh"}
        for token in _command_parts(parts)
    )


def _option_value(parts: list[str], name: str, default: Any = None) -> Any:
    """Read a simple ``--flag value``/``--flag=value`` option from command argv."""
    command = _command_parts(parts)
    for index, token in enumerate(command):
        if token == name:
            return command[index + 1] if index + 1 < len(command) else default
        if token.startswith(name + "="):
            return token.split("=", 1)[1]
    return default


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


def _typed_workspace_limitation(workspace: Path, command: list[str]) -> str | None:
    """Explain why the typed run/profile source contract cannot represent a workspace."""
    required = ("kernel.py", "reference.py", "input.py", "shapes.json")
    missing = [name for name in required if not (workspace / name).is_file()]
    if missing:
        return "missing " + ", ".join(missing)
    if (workspace / "workload.jsonl").is_file():
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
                (workspace / f"{root}.py").is_file()
                or (workspace / root / "__init__.py").is_file()
            )
        )
        if local_imports:
            return "candidate imports local auxiliary modules: " + ", ".join(local_imports)

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
    if warmup is not None and str(warmup) != "10":
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
        env_vars[key] = value
    return env_vars


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
    shapes = _json_object(workspace / "shapes.json", required=True)
    assert shapes is not None
    try:
        multi_seed = int(_option_value(command, "--multi-seed", 0))
        bench_iters = int(_option_value(command, "--timed-runs", 100))
        atol = float(_option_value(command, "--atol", 1e-2))
        rtol = float(_option_value(command, "--rtol", 0.05))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid evaluator command option: {exc}") from exc
    if multi_seed < 0 or bench_iters < 1:
        raise ValueError("--multi-seed must be non-negative and --timed-runs must be positive")

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
    for filename, field in (("metadata.json", "metadata"), ("roofline.json", "roofline")):
        value = _json_object(workspace / filename)
        if value is not None:
            reference[field] = value

    request: dict[str, Any] = {
        "name": f"{workspace.name}_{kind}",
        "spec": {
            "languages": [str(value) for value in languages],
            "target_hardware": [hardware],
        },
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
    if kind == "run":
        request["mode"] = "full"
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
    if not runtime_link.is_symlink():
        return None
    runtime_root = runtime_link.resolve()
    run_eval = runtime_root / "scripts" / "run_eval.py"
    package = runtime_root / "src" / "atrex_bench"
    utils_module = package / "utils.py"
    sdk_module = package / "sdk.py"
    if (
        not package.is_dir()
        or not run_eval.is_file()
        or (evaluator_only and not utils_module.is_file())
    ):
        raise RuntimeError(f"invalid workspace Atrex-Bench runtime link: {runtime_link}")

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


REMOTE_COLLECTOR = r'''#!/usr/bin/env python3
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
include_raw = bool(cfg["include_raw_profile"])
archive = io.BytesIO()
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

with tarfile.open(fileobj=archive, mode="w:gz") as tf:
    for value in cfg["paths"]:
        if not safe(value):
            continue
        path = root / value
        if path.is_file():
            add_file(tf, path)
        elif path.is_dir():
            for child in path.rglob("*"):
                add_file(tf, child)

if skipped:
    print("[sandbox] artifacts not returned: " + ", ".join(skipped), file=sys.stderr)
print(BEGIN)
print(base64.b64encode(archive.getvalue()).decode("ascii"))
print(END)
'''


def _runner_source() -> str:
    return r'''#!/usr/bin/env bash
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
python __atrex_collect.py workspace __atrex_outputs.json
collect_status=$?
if [[ $collect_status -ne 0 ]]; then
    exit 98
fi
exit $command_status
'''


def _extract_outputs(stdout: str, workspace: Path) -> str:
    """Extract the returned archive and return command stdout without framing."""
    if OUTPUT_BEGIN not in stdout or OUTPUT_END not in stdout:
        raise RuntimeError("sandbox response did not contain an artifact frame")
    command_stdout, framed = stdout.rsplit(OUTPUT_BEGIN, 1)
    encoded, trailing = framed.split(OUTPUT_END, 1)
    if trailing.strip():
        command_stdout += trailing
    payload = base64.b64decode("".join(encoded.split()), validate=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tf:
        for member in tf.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError(f"unsafe artifact path returned by sandbox: {member.name!r}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"sandbox artifact links are not accepted: {member.name!r}")
            target = workspace / path.as_posix()
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
    return command_stdout.rstrip("\n")


def _command_text(parts: list[str]) -> str:
    if parts and parts[0] == "--":
        parts = parts[1:]
    if not parts:
        raise ValueError("a command is required after --")
    # A single argument is commonly a deliberately quoted shell pipeline.
    return parts[0] if len(parts) == 1 else shlex.join(parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run correctness, performance, or profile commands in an agate GPU sandbox.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--hardware",
        default=os.environ.get("ATREX_SANDBOX_GPU", ""),
        help="Gateway GPU hardware token, e.g. REMOTE_GPU (default: ATREX_SANDBOX_GPU).",
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
    parser.add_argument("--workspace", default=".", help="Local workspace to upload (default: cwd).")
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
        "--sync", action="append", default=[], metavar="PATH",
        help="Relative profile/result path to copy back (repeatable; default: profiles).",
    )
    parser.add_argument("--no-sync", action="store_true", help="Do not copy any files back.")
    parser.add_argument(
        "--include-raw-profile", action="store_true",
        help="Return raw .ncu-rep/ATT artifacts (can make the gateway response very large).",
    )
    parser.add_argument(
        "--profile-level", choices=("survey", "sol", "deep"), default="sol",
        help="Typed profile funnel level (default: sol).",
    )
    parser.add_argument(
        "--profiler", choices=("ncu", "rocprofv3"), default=None,
        help="Typed profile backend (default: gateway vendor auto-detection).",
    )
    parser.add_argument(
        "--profile-counter", action="append", default=[], metavar="METRIC",
        help="Typed profile metric/counter (repeatable).",
    )
    parser.add_argument(
        "--kernel-regex", default=None,
        help="Typed profile kernel regex (required by a deep profile).",
    )
    parser.add_argument(
        "--top-kernels", type=int, default=None,
        help="Limit the typed profile result to the N hottest kernels.",
    )
    parser.add_argument(
        "--input", action="append", default=[], metavar="PATH",
        help=(
            "Additional required workspace file or directory for a non-evaluator command "
            "(repeatable). Arbitrary commands otherwise receive only paths referenced by "
            "their argv; the full workspace is never uploaded implicitly."
        ),
    )
    parser.add_argument(
        "--max-input-file-mb", type=int, default=16,
        help="Skip individual workspace input files larger than this (default: 16 MiB).",
    )
    parser.add_argument(
        "--max-output-file-mb", type=int, default=8,
        help="Skip individual returned artifacts larger than this (default: 8 MiB).",
    )
    parser.add_argument("-e", "--env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument(
        "--keep-pod", action="store_true",
        help="Ask the gateway not to recycle the pod; filesystem persistence is still not assumed.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Package and print the request summary only.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command after --.")
    return parser


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

    raise AssertionError("unreachable: direct gateway retry loop returned no terminal job")


def _run_direct_gateway(
    *,
    url: str,
    hardware: str,
    timeout: int,
    queue_wait_grace: int,
    env_items: list[str],
    files: dict[str, Path],
    command: str,
) -> subprocess.CompletedProcess[str]:
    """Use the public dev-job HTTP API when the optional agate CLI is absent."""
    try:
        env_vars = _parse_env_items(env_items)
    except ValueError as exc:
        raise SystemExit(f"sandbox: {exc}") from exc
    return _run_direct_job(
        url=url,
        kind="dev",
        timeout=timeout,
        queue_wait_grace=queue_wait_grace,
        payload={
        "spec": {"target_hardware": [hardware]},
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
        job
        and job.get("status") == "cancelled"
        and not job.get("result")
        and not job.get("error")
    )


def _submitted_job_id(proc: subprocess.CompletedProcess[str]) -> str | None:
    """Recover the job id printed by agate before it starts polling."""
    match = SUBMITTED_JOB_RE.search((proc.stderr or "") + "\n" + (proc.stdout or ""))
    return match.group(1) if match else None


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
    """Continue waiting for an already-submitted job without resubmitting it.

    A long-lived ``agate dev`` polling process can occasionally receive SIGTERM
    while its remote job continues running.  In that case Python normalizes the
    child's ``-SIGTERM`` return code to exit 241, and treating it as a kernel
    failure loses a perfectly valid later RESULT_JSON.  Agate prints the job id
    before polling, so attach to that same job with ``agate get --wait`` for the
    remainder of the original client-side budget.
    """
    if _job_response(initial.stdout or "") is not None:
        return initial
    job_id = _submitted_job_id(initial)
    remaining = int(wait_budget - elapsed)
    if not job_id or remaining <= 0:
        return initial

    get_command = [executable, "get"]
    if url:
        get_command += ["--url", url]
    elif gateway_profile:
        get_command += ["--profile", gateway_profile]
    get_command += [
        "--http-timeout", str(MAX_HTTP_REQUEST_TIMEOUT),
        "--wait-timeout", str(max(1, remaining)),
        "--job-timeout", str(command_timeout),
        "--wait",
        job_id,
    ]
    resumed = subprocess.run(get_command, capture_output=True, text=True)
    note = (
        f"[sandbox] agate polling exited {initial.returncode}; "
        f"resumed existing job_id={job_id} without resubmission"
    )
    stderr_parts = [part.rstrip() for part in (initial.stderr, note, resumed.stderr) if part]
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
    """Submit one agate job and preserve the existing interrupted-wait recovery."""
    wait_started = time.monotonic()
    proc = subprocess.run(agate, capture_output=True, text=True)
    return _resume_interrupted_agate_wait(
        executable=executable,
        url=url,
        gateway_profile=gateway_profile,
        command_timeout=command_timeout,
        wait_budget=wait_budget,
        elapsed=time.monotonic() - wait_started,
        initial=proc,
    )


def _run_agate_with_cancel_retry(
    *,
    agate: list[str],
    executable: str,
    url: str,
    gateway_profile: str | None,
    command_timeout: int,
    wait_budget: int,
) -> subprocess.CompletedProcess[str]:
    """Retry once only when a cancelled job produced no result or error."""
    first = _run_agate_once(
        agate=agate,
        executable=executable,
        url=url,
        gateway_profile=gateway_profile,
        command_timeout=command_timeout,
        wait_budget=wait_budget,
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
    stderr_parts = [part.rstrip() for part in (first.stderr, note, second.stderr) if part]
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
) -> list[str]:
    """Build an agate run/profile invocation for a typed request."""
    command = [executable, kind]
    if args.url:
        command += ["--url", args.url]
    elif args.gateway_profile:
        command += ["--profile", args.gateway_profile]
    options = request["options"]
    command += [
        "--gpu", args.hardware,
        "--candidate", str(workspace / "kernel.py"),
        "--reference-dir", str(workspace),
        "--operator", str(request["reference"]["operator"]),
        "--num-correctness-cases", str(options["num_correctness_cases"]),
        "--bench-iters", str(options["bench_iters"]),
        "--http-timeout", str(MAX_HTTP_REQUEST_TIMEOUT),
        "--wait-timeout", str(args.timeout + queue_wait_grace),
        "--job-timeout", str(args.timeout),
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


def _expected_shape_ids(workspace: Path) -> list[str]:
    shapes = _json_object(workspace / "shapes.json", required=True)
    assert shapes is not None

    def sort_key(shape_id: str) -> tuple[int, object]:
        return (0, int(shape_id)) if shape_id.isdigit() else (1, shape_id)

    return sorted((str(shape_id) for shape_id in shapes), key=sort_key)


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


def _optimizer_result_from_eval(
    payload: dict[str, Any], shape_ids: list[str]
) -> dict[str, Any]:
    """Convert the typed gateway's Atrex-Bench result to optimizer RESULT_JSON."""
    failures: list[str] = []
    if payload.get("error"):
        failures.append("evaluation: " + str(payload["error"]))
    passed = payload.get("passed")
    passed = passed if isinstance(passed, dict) else {}
    failures.extend(_compile_failures(passed.get("compile"), shape_ids))

    correctness_status = passed.get("correctness")
    correctness_status = correctness_status if isinstance(correctness_status, dict) else {}
    correctness = payload.get("correctness")
    correctness = correctness if isinstance(correctness, dict) else {}
    correctness_shapes = correctness.get("shapes")
    correctness_shapes = correctness_shapes if isinstance(correctness_shapes, dict) else {}
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

    performance = payload.get("performance")
    performance = performance if isinstance(performance, dict) else {}
    performance_shapes = performance.get("shapes")
    performance_shapes = performance_shapes if isinstance(performance_shapes, dict) else {}
    latency_by_shape: dict[str, float] = {}
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
    return {
        "all_pass": not failures,
        "failures": failures,
        "latency_us_geomean": geomean,
        "latency_us_arith_mean": arithmetic,
        "latency_us_by_shape": latency_by_shape,
        "speedup_vs_ref_geomean": None,
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
        "evaluator": "atrex-gpu-gateway/run",
        "eval_id": payload.get("eval_id"),
    }


def _record_profile_job(job: dict[str, Any], workspace: Path, sync_paths: list[str]) -> None:
    """Persist the typed profile response where the local optimization session expects it."""
    for relative in sync_paths:
        path = PurePosixPath(relative)
        if not path.parts or path.parts[0] != "profiles":
            continue
        target = workspace / path.as_posix() / "gateway_profile.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(job, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


def _run_typed_gateway(
    args: argparse.Namespace,
    workspace: Path,
    command_parts: list[str],
    kind: str,
    sync_paths: list[str],
    queue_wait_grace: int,
) -> int | None:
    """Run agate run/profile, returning None only for a documented dev fallback."""
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
        print(f"[sandbox] {kind} interface unsupported for this workspace: {exc}; using dev", file=sys.stderr)
        return None

    if args.dry_run:
        print(json.dumps({
            "hardware": args.hardware,
            "url": args.url or None,
            "gateway_profile": args.gateway_profile,
            "workspace": str(workspace),
            "kind": kind,
            "fallback_kind": "dev",
            "candidate_bytes": len(request["candidate"].encode("utf-8")),
            "shape_count": len(request["reference"]["shapes"]),
            "options": request["options"],
            "sync": sync_paths,
        }, indent=2))
        return 0

    agate_executable = _find_agate()
    try:
        if args.url and agate_executable is None:
            proc = _run_direct_job(
                url=args.url,
                kind="eval" if kind == "run" else kind,
                payload=request,
                timeout=args.timeout,
                queue_wait_grace=queue_wait_grace,
            )
        else:
            if agate_executable is None:
                raise FileNotFoundError("agate")
            agate = _typed_agate_command(
                agate_executable, args, workspace, kind, request, queue_wait_grace
            )
            proc = _run_agate_with_cancel_retry(
                agate=agate,
                executable=agate_executable,
                url=args.url,
                gateway_profile=args.gateway_profile,
                command_timeout=args.timeout,
                wait_budget=args.timeout + queue_wait_grace,
            )
    except GatewayHTTPError as exc:
        if _typed_fallback_allowed(exc):
            print(f"[sandbox] gateway {kind} interface unavailable ({exc}); using dev", file=sys.stderr)
            return None
        raise SystemExit(f"sandbox: {kind} gateway request failed: {exc}") from exc
    except FileNotFoundError as exc:
        raise SystemExit(
            "sandbox: agate not found and no explicit --url was provided; "
            "install atrex-gateway-client first"
        ) from exc

    if proc.returncode and _typed_fallback_allowed((proc.stderr or "") + (proc.stdout or "")):
        if proc.stderr:
            print(proc.stderr.rstrip(), file=sys.stderr)
        print(f"[sandbox] gateway {kind} interface rejected this request; using dev", file=sys.stderr)
        return None
    if proc.stderr:
        print(proc.stderr.rstrip(), file=sys.stderr)
    job = _job_response(proc.stdout or "")
    if job is None:
        if proc.stdout:
            print(proc.stdout.rstrip())
        return proc.returncode or 2
    if job.get("status") != "succeeded" or not isinstance(job.get("result"), dict):
        print(json.dumps(job, ensure_ascii=False))
        return proc.returncode or 1

    print(
        f"[sandbox] gateway_kind={kind} job_id={job.get('job_id')}",
        file=sys.stderr,
    )
    if kind == "run":
        result = _optimizer_result_from_eval(job["result"], _expected_shape_ids(workspace))
        print(TEST_RESULT_PREFIX + json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0 if result["all_pass"] else 1

    _record_profile_job(job, workspace, sync_paths)
    print(PROFILE_RESULT_PREFIX + json.dumps(job["result"], ensure_ascii=False))
    return 0


def _main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.hardware:
        raise SystemExit("sandbox: --hardware or ATREX_SANDBOX_GPU is required")
    # Explicit endpoint flags override inherited sandbox endpoint variables.  This
    # matters when a long-lived optimization shell switches between a remote
    # profile and localhost without first scrubbing its environment.
    if args.url and args.gateway_profile:
        raise SystemExit("sandbox: --url and --gateway-profile are mutually exclusive")
    if args.url is not None:
        args.gateway_profile = None
    elif args.gateway_profile is not None:
        args.url = ""
    else:
        args.url = os.environ.get("ATREX_SANDBOX_URL", "")
        args.gateway_profile = os.environ.get("ATREX_SANDBOX_PROFILE") or None
        if args.url and args.gateway_profile:
            raise SystemExit(
                "sandbox: ATREX_SANDBOX_URL and ATREX_SANDBOX_PROFILE are mutually exclusive"
            )
    if not 1 <= args.timeout <= MAX_COMMAND_TIMEOUT:
        raise SystemExit(
            "sandbox: --timeout must be in the gateway-supported range "
            f"1..{MAX_COMMAND_TIMEOUT}"
        )
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
    try:
        command = _command_text(args.command)
        sync_paths = [] if args.no_sync else [
            _safe_relative(path) for path in (args.sync or list(DEFAULT_SYNC_PATHS))
        ]
    except ValueError as exc:
        raise SystemExit(f"sandbox: {exc}") from exc

    workspace = Path(args.workspace).resolve()
    if any(PurePosixPath(path).parts[0] == "memory" for path in sync_paths):
        raise SystemExit("sandbox: memory/ is local optimizer state and cannot be synchronized")
    if not workspace.is_dir():
        raise SystemExit(f"sandbox: workspace not found: {workspace}")

    gateway_kind = _requested_gateway_kind(args.kind, args.command)
    typed_limitation: str | None = None
    if gateway_kind in TYPED_KINDS:
        if gateway_kind == "profile" and args.profile_level == "deep" and not args.kernel_regex:
            raise SystemExit("sandbox: --profile-level deep requires --kernel-regex")
        if args.keep_pod:
            typed_limitation = "--keep-pod is only supported by dev"
        elif args.input:
            typed_limitation = "custom --input files are only supported by dev"
        elif gateway_kind == "profile" and args.include_raw_profile:
            typed_limitation = "--include-raw-profile requires the custom dev profiler wrapper"
        else:
            try:
                typed_limitation = _typed_workspace_limitation(workspace, args.command)
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
        print(
            f"[sandbox] {gateway_kind} interface unsupported ({typed_limitation}); using dev",
            file=sys.stderr,
        )
        gateway_kind = "dev"

    evaluator_command = _is_test_kernel_command(args.command)
    if evaluator_command:
        selected_inputs = _evaluation_input_paths(workspace)
    else:
        try:
            selected_inputs = _command_input_paths(
                workspace,
                args.command,
                args.input,
            )
        except ValueError as exc:
            raise SystemExit(f"sandbox: {exc}") from exc
    bundle, file_count, skipped = _make_input_bundle(
        workspace,
        args.max_input_file_mb * 1024 * 1024,
        selected_inputs,
    )
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
    agate_executable = _find_agate()
    direct_http = bool(args.url and (agate_executable is None or bundle_bytes > 50 * 1024))
    # Linux limits each individual argv entry to 128 KiB (MAX_ARG_STRLEN).
    # agate's worker materializes an uploaded file through one such argument,
    # so leave headroom for its framing instead of creating a doomed job.
    # The direct HTTP fallback does not place file contents in argv and can use
    # the gateway's normal request-body allowance instead.
    safe_bundle_bytes = 20 * 1024 * 1024 if direct_http else 120 * 1024
    if bundle_bytes > safe_bundle_bytes:
        raise SystemExit(
            f"sandbox: packaged payload is {bundle_bytes / 1024:.1f} KiB, "
            f"above the safe {safe_bundle_bytes / 1024:.0f} KiB gateway argument limit; "
            "exclude additional "
            "local-only workspace artifacts"
        )
    print(
        f"[sandbox] gateway_kind=dev hardware={args.hardware} files={file_count} "
        f"payload={bundle_bytes / 1024:.1f} KiB "
        f"atrex_runtime={runtime_bundle_bytes / 1024:.1f} KiB command={command!r}",
        file=sys.stderr,
    )
    if skipped:
        print("[sandbox] inputs skipped: " + ", ".join(skipped), file=sys.stderr)
    if args.dry_run:
        print(json.dumps({
            "hardware": args.hardware,
            "url": args.url or None,
            "gateway_profile": args.gateway_profile,
            "workspace": str(workspace),
            "kind": "dev",
            "requested_kind": args.kind,
            "typed_fallback_reason": typed_limitation,
            "files": file_count,
            "payload_bytes": bundle_bytes,
            "atrex_runtime_payload_bytes": runtime_bundle_bytes,
            "sync": sync_paths,
            "command": command,
        }, indent=2))
        return 0

    output_cfg = {
        "paths": sync_paths,
        "max_file_bytes": args.max_output_file_mb * 1024 * 1024,
        "include_raw_profile": args.include_raw_profile,
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
        if len(bundle) > WORKSPACE_CHUNK_BYTES:
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
        command_path.write_text("#!/usr/bin/env bash\nset -o pipefail\n" + command + "\n", encoding="utf-8")
        collector_path.write_text(REMOTE_COLLECTOR, encoding="utf-8")
        outputs_path.write_text(json.dumps(output_cfg), encoding="utf-8")
        if runtime_bundle:
            for index, offset in enumerate(range(0, len(runtime_bundle), RUNTIME_CHUNK_BYTES)):
                part_path = temp / f"atrex_runtime.part{index:03d}"
                part_path.write_text(
                    runtime_bundle[offset : offset + RUNTIME_CHUNK_BYTES],
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
        agate += [
            "--gpu", args.hardware,
            "--dev-timeout", str(args.timeout),
            "--http-timeout", str(MAX_HTTP_REQUEST_TIMEOUT),
            "--wait-timeout", str(args.timeout + queue_wait_grace),
            "--job-timeout", str(args.timeout),
        ]
        if workspace_part_paths:
            for index, part_path in enumerate(workspace_part_paths):
                agate += [
                    "--file",
                    f"__atrex_workspace.tar.gz.b64.part{index:03d}={part_path}",
                ]
        else:
            agate += ["--file", f"__atrex_workspace.tar.gz.b64={bundle_path}"]
        agate += [
            "--file", f"__atrex_command.sh={command_path}",
            "--file", f"__atrex_collect.py={collector_path}",
            "--file", f"__atrex_outputs.json={outputs_path}",
        ]
        for index, part_path in enumerate(runtime_part_paths):
            agate += [
                "--file",
                f"__atrex_bench_runtime.tar.gz.b64.part{index:03d}={part_path}",
            ]
        for item in args.env:
            if "=" not in item or item.startswith("="):
                raise SystemExit(f"sandbox: invalid --env {item!r}; expected KEY=VALUE")
            agate += ["--env-var", item]
        if args.keep_pod:
            agate.append("--no-recycle")
        agate.append("bash __atrex_runner.sh")

        # The runner is uploaded separately after the command has been assembled.
        runner_path = temp / "runner.sh"
        runner_path.write_text(_runner_source(), encoding="utf-8")
        agate[-1:-1] = ["--file", f"__atrex_runner.sh={runner_path}"]

        if direct_http:
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
                    env_items=args.env,
                    files=direct_files,
                    command="bash __atrex_runner.sh",
                )
            except (OSError, RuntimeError, TimeoutError) as exc:
                raise SystemExit(f"sandbox: direct gateway request failed: {exc}") from exc
        else:
            try:
                proc = _run_agate_with_cancel_retry(
                    agate=agate,
                    executable=agate_executable or "agate",
                    url=args.url,
                    gateway_profile=args.gateway_profile,
                    command_timeout=args.timeout,
                    wait_budget=args.timeout + queue_wait_grace,
                )
            except FileNotFoundError as exc:
                raise SystemExit(
                    "sandbox: agate not found and no explicit --url was provided; "
                    "install atrex-gateway-client first"
                ) from exc

    if proc.stderr:
        print(proc.stderr.rstrip(), file=sys.stderr)
    try:
        job = json.loads(proc.stdout)
    except json.JSONDecodeError:
        if proc.stdout:
            print(proc.stdout.rstrip())
        return proc.returncode or 2

    result = job.get("result") or {}
    remote_stdout = str(result.get("stdout") or "")
    remote_stderr = str(result.get("stderr") or "")
    try:
        command_stdout = _extract_outputs(remote_stdout, workspace)
    except (RuntimeError, ValueError, tarfile.TarError) as exc:
        if remote_stdout:
            print(remote_stdout.rstrip())
        if remote_stderr:
            print(remote_stderr.rstrip(), file=sys.stderr)
        print(f"sandbox: {exc}; job_id={job.get('job_id')}", file=sys.stderr)
        return int(result.get("exit_code") or proc.returncode or 2)
    if command_stdout:
        print(command_stdout)
    if remote_stderr:
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
    _append_sandbox_telemetry(
        "sandbox_operation_started",
        operation_id=operation_id,
        category=category,
    )
    try:
        returncode = _main(argv)
    except BaseException as exc:
        _append_sandbox_telemetry(
            "sandbox_operation_completed",
            operation_id=operation_id,
            category=category,
            duration_seconds=round(time.monotonic() - started, 6),
            status="failed",
            failure_type=type(exc).__name__,
        )
        raise
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
