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
import shlex
import subprocess
import sys
import tarfile
import tempfile
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


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"path must be relative to the workspace: {value!r}")
    normalized = path.as_posix()
    if normalized in ("", "."):
        raise ValueError(f"path must not resolve to the workspace root: {value!r}")
    return normalized


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
) -> tuple[str, int, list[str]]:
    """Return a base64 tarball containing the workspace and runtime tools."""
    archive = io.BytesIO()
    seen: set[str] = set()
    skipped: list[str] = []
    count = 0
    profile_roots = tuple(path.rstrip("/") for path in profile_input_paths)

    def add_tree(tf: tarfile.TarFile, source: Path, prefix: str = "") -> None:
        nonlocal count
        if not source.is_dir():
            return
        for path in _walk_files(source):
            rel = path.relative_to(source).as_posix()
            arcname = f"{prefix}/{rel}" if prefix else rel
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
                    continue
            if (
                arcname in seen
                or arcname in INPUT_SKIP_PATHS
                or path.suffix in INPUT_SKIP_SUFFIXES
            ):
                continue
            try:
                size = path.stat().st_size
            except OSError as exc:
                skipped.append(f"{arcname} ({exc})")
                continue
            if size > max_file_bytes:
                skipped.append(f"{arcname} ({size} bytes > input limit)")
                continue
            tf.add(path, arcname=arcname, recursive=False)
            seen.add(arcname)
            count += 1

    with tarfile.open(fileobj=archive, mode="w:gz") as tf:
        add_tree(tf, workspace)
        # Optimization workspaces receive tools/ as a symlink.  Materialize the
        # small tool directory so remote profile commands are self-contained.
        workspace_tools = workspace / "tools"
        if workspace_tools.is_symlink() or not workspace_tools.exists():
            add_tree(tf, REPO_ROOT / "tools", "tools")

    return base64.b64encode(archive.getvalue()).decode("ascii"), count, skipped


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
        "--timeout", type=int, default=int(os.environ.get("ATREX_SANDBOX_TIMEOUT", "600")),
        help="Remote command timeout in seconds, 1..600 (default: 600; gateway limit).",
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
    if not 1 <= args.timeout <= 600:
        raise SystemExit("sandbox: --timeout must be in the gateway-supported range 1..600")
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
    bundle, file_count, skipped = _make_input_bundle(
        workspace,
        args.max_input_file_mb * 1024 * 1024,
        profile_input_paths,
    )
    bundle_bytes = len(bundle.encode("ascii"))
    # Linux limits each individual argv entry to 128 KiB (MAX_ARG_STRLEN).
    # agate's worker materializes an uploaded file through one such argument,
    # so leave headroom for its framing instead of creating a doomed job.
    if bundle_bytes > 120 * 1024:
        raise SystemExit(
            f"sandbox: packaged payload is {bundle_bytes / 1024:.1f} KiB, "
            "above the safe 120 KiB gateway argument limit; exclude additional "
            "local-only workspace artifacts"
        )
    print(
        f"[sandbox] hardware={args.hardware} files={file_count} "
        f"payload={bundle_bytes / 1024:.1f} KiB command={command!r}",
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
        bundle_path.write_text(bundle, encoding="ascii")
        command_path.write_text("#!/usr/bin/env bash\nset -o pipefail\n" + command + "\n", encoding="utf-8")
        collector_path.write_text(REMOTE_COLLECTOR, encoding="utf-8")
        outputs_path.write_text(json.dumps(output_cfg), encoding="utf-8")

        agate = ["agate", "dev"]
        if args.url:
            agate += ["--url", args.url]
        elif args.gateway_profile:
            agate += ["--profile", args.gateway_profile]
        agate += [
            "--gpu", args.hardware,
            "--dev-timeout", str(args.timeout),
            "--timeout", str(args.timeout + 120),
            "--file", f"__atrex_workspace.tar.gz.b64={bundle_path}",
            "--file", f"__atrex_command.sh={command_path}",
            "--file", f"__atrex_collect.py={collector_path}",
            "--file", f"__atrex_outputs.json={outputs_path}",
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

        try:
            proc = subprocess.run(agate, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise SystemExit(
                "sandbox: agate not found; install atrex-gateway-client first"
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
