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

"""Run a workspace command in an atrex-gpu-gateway GPU sandbox.

The gateway's sandbox primitive is ``agate dev``.  This wrapper makes it useful
for the optimizer by shipping a self-contained workspace into the pod and
copying the small, durable results back afterwards.  A new pod may be selected
for every invocation; callers must not rely on remote filesystem persistence.

Examples::

    python tools/sandbox.py --hardware REMOTE_GPU --no-sync -- python test_kernel.py --no-memory
    python tools/sandbox.py --hardware REMOTE_GPU --sync profiles/v1 -- \
        bash tools/profile_nvidia.sh kernel.py --output-dir profiles/v1 --source
    python tools/sandbox.py --hardware REMOTE_ACCELERATOR --gateway-profile pre --sync profiles/v1 -- \
        bash tools/profile_kernel.sh kernel.py --output-dir profiles/v1

``ATREX_SANDBOX_GPU``, ``ATREX_SANDBOX_PROFILE``, ``ATREX_SANDBOX_URL``, and
``ATREX_SANDBOX_TIMEOUT`` provide defaults for the corresponding flags.  A
localhost gateway uses the same transport as a remote worker, for example
``ATREX_SANDBOX_GPU=local`` plus
``ATREX_SANDBOX_URL=http://127.0.0.1:8000``.  Authentication and any remaining
URL resolution stay agate's responsibility (AGATE_* or ~/.atrex/config.json).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Iterable


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
    # The aggregate workspace can contain multiple independent bucket
    # workspaces.  A full-kernel validation needs only the aggregate sources;
    # recursively uploading every bucket would duplicate repositories and can
    # exceed the gateway request-size limit.
    "workload_buckets",
}
INPUT_SKIP_PATHS = {
    # A pod must not recursively submit another sandbox job, and memory updates
    # are deliberately local-only.  Omitting these also leaves useful headroom
    # below the gateway worker's per-argument limit.
    "tools/sandbox.py",
    "tools/local_gateway.py",
    "tools/memory_manager.py",
    # The durable host-side monitor is never invoked inside a GPU worker.  It
    # grew the materialized tools bundle enough to push large aggregate kernels
    # over agate's per-argument limit despite being unrelated to validation.
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
RUNTIME_CHUNK_BYTES = 80 * 1024
SUBMITTED_JOB_RE = re.compile(r"\bsubmitted job_id=([A-Za-z0-9_.-]+); polling\.\.\.")
DISPATCH_SIGNATURE_INPUT_PATHS = frozenset(
    {
        "definition.json",
        "input.py",
        "reference.py",
        "shapes.json",
        "tools/collect_dispatch_signatures.py",
        "workload.jsonl",
    }
)
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
    profile_input_paths: Iterable[str] = (),
    input_paths: Iterable[str] = (),
) -> tuple[str, int, list[str]]:
    """Return a base64 tarball containing the workspace and runtime tools."""
    archive = io.BytesIO()
    seen: set[str] = set()
    skipped: list[str] = []
    count = 0
    profile_roots = tuple(path.rstrip("/") for path in profile_input_paths)
    selected_inputs = frozenset(input_paths)

    def add_file(tf: tarfile.TarFile, path: Path, arcname: str) -> None:
        nonlocal count
        arc_parts = PurePosixPath(arcname).parts
        # Profile reports/metrics are synchronized back for local analysis
        # and must not be re-uploaded on every later test.  Only executable
        # profile harnesses requested by this invocation are inputs to a
        # fresh sandbox job.  Historical harnesses are local campaign state
        # and accumulate indefinitely on long-running optimizations.
        if arc_parts and arc_parts[0] == "profiles":
            if "harness" not in arc_parts or not any(
                arcname == root or arcname.startswith(root + "/")
                for root in profile_roots
            ):
                return
        if (
            arcname in seen
            or arcname in INPUT_SKIP_PATHS
            or path.suffix in INPUT_SKIP_SUFFIXES
            or (selected_inputs and arcname not in selected_inputs)
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


def _evaluation_input_paths(workspace: Path) -> frozenset[str]:
    """Return evaluator inputs plus every candidate source declared by solution.json."""
    selected = set(EVALUATION_INPUT_PATHS)
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

    aggregate_sources = workspace / "aggregate_kernels"
    if aggregate_sources.is_dir():
        for path in _walk_files(aggregate_sources):
            selected.add(path.relative_to(workspace).as_posix())
    return frozenset(selected)


def _is_test_kernel_command(parts: list[str]) -> bool:
    command = parts[1:] if parts and parts[0] == "--" else parts
    return (
        len(command) >= 2
        and Path(command[0]).name in {"python", "python3", "python3.10", "python3.12"}
        and Path(command[1]).name == "test_kernel.py"
    )


def _make_atrex_bench_runtime_bundle(
    workspace: Path, *, minimal: bool = False, evaluator_only: bool = False
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
    runtime_module = package / "eval" / "_runtime.py"
    utils_module = package / "utils.py"
    if (
        not package.is_dir()
        or (minimal and not runtime_module.is_file())
        or (not minimal and not run_eval.is_file())
        or (evaluator_only and not utils_module.is_file())
    ):
        raise RuntimeError(f"invalid workspace Atrex-Bench runtime link: {runtime_link}")

    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tf:
        if minimal:
            tf.add(
                runtime_module,
                arcname="atrex-bench/src/atrex_bench/eval/_runtime.py",
                recursive=False,
            )
        elif evaluator_only:
            evaluator_files = [package / "__init__.py", utils_module]
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
if ! base64 -d __atrex_workspace.tar.gz.b64 | tar -xzf - -C workspace; then
    echo "[sandbox] failed to unpack workspace" >&2
    exit 97
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
        "--dispatch-signatures",
        action="store_true",
        help=(
            "Upload only evaluator input generators and the minimal native "
            "runtime needed to collect deterministic dispatch signatures."
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


def _gateway_json(
    base_url: str,
    method: str,
    path: str,
    payload: dict | None,
    timeout: float,
) -> dict:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        method=method,
        data=body,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"gateway HTTP {exc.code}: {detail}") from exc
    if not isinstance(result, dict):
        raise RuntimeError("gateway returned a non-object JSON response")
    return result


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
    env_vars: dict[str, str] = {}
    for item in env_items:
        if "=" not in item or item.startswith("="):
            raise SystemExit(f"sandbox: invalid --env {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        env_vars[key] = value
    payload = {
        "spec": {"target_hardware": [hardware]},
        "command": command,
        "timeout_s": timeout,
        "env_vars": env_vars,
        "files": {name: path.read_text(encoding="utf-8") for name, path in files.items()},
    }
    prior_note = ""
    for submission in range(2):
        accepted = _gateway_json(url, "POST", "/v1/jobs/dev", payload, 30)
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
                        args=["direct-gateway", job_id],
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


def main(argv: list[str] | None = None) -> int:
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
    profile_input_paths = [
        path for path in sync_paths
        if PurePosixPath(path).parts and PurePosixPath(path).parts[0] == "profiles"
    ]
    evaluator_command = _is_test_kernel_command(args.command)
    if args.dispatch_signatures:
        selected_inputs: Iterable[str] = DISPATCH_SIGNATURE_INPUT_PATHS
    elif evaluator_command:
        selected_inputs = _evaluation_input_paths(workspace)
    else:
        selected_inputs = ()
    bundle, file_count, skipped = _make_input_bundle(
        workspace,
        args.max_input_file_mb * 1024 * 1024,
        profile_input_paths,
        selected_inputs,
    )
    runtime_bundle = _make_atrex_bench_runtime_bundle(
        workspace,
        minimal=args.dispatch_signatures,
        evaluator_only=evaluator_command,
    )
    bundle_bytes = len(bundle.encode("ascii"))
    runtime_bundle_bytes = len(runtime_bundle.encode("ascii")) if runtime_bundle else 0
    agate_executable = _find_agate()
    direct_http = bool(args.url and agate_executable is None)
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
        f"[sandbox] hardware={args.hardware} files={file_count} "
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
        bundle_path = temp / "workspace.tar.gz.b64"
        command_path = temp / "command.sh"
        collector_path = temp / "collect.py"
        outputs_path = temp / "outputs.json"
        runtime_part_paths: list[Path] = []
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

        agate = [agate_executable or "agate", "dev"]
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
            "--file", f"__atrex_workspace.tar.gz.b64={bundle_path}",
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
                    "__atrex_workspace.tar.gz.b64": bundle_path,
                    "__atrex_command.sh": command_path,
                    "__atrex_collect.py": collector_path,
                    "__atrex_outputs.json": outputs_path,
                    "__atrex_runner.sh": runner_path,
                }
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


if __name__ == "__main__":
    raise SystemExit(main())
