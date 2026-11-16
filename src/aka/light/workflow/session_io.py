"""Coding-agent session execution and sandbox I/O.

Owns session spawning and accounting, the independent production-candidate review, sandbox command
construction, and evaluator result parsing.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aka.light.agent.runtime as _agent_runtime
from aka.light.workflow.constants import (
    DEFAULT_SANDBOX_TIMEOUT,
    DEPENDENCY_REVIEW_SCHEMA_VERSION,
    REPO_ROOT,
    SANDBOX_FULL_WORKFLOW_PROMPT,
    SANDBOX_SAFETY_BOUNDARY_PROMPT,
    SANDBOX_TOOL,
)
from aka.light.execution.recovery import (environment_state_file, raise_if_environment_blocked)
from aka.light.evaluation.hardware import hardware_vendor
from aka.light.agent.recovery_processes import spawn_owned_session
from aka.light.execution.config import queue_wait_grace
from aka.light.evaluation.policy import SHAPE_BATCH_SIZE
from aka.contracts.evaluation import (
    DEFAULT_ABBA_REPEATS,
    DEFAULT_EVALUATION_STEP_TIMEOUT,
)
from aka.light.evaluation.shapes import (
    effective_shape_batch_size,
    shape_request_timeout,
)
def _status_is(value: object, expected: str) -> bool:
    """Accept a status even when a CLI accidentally stored it as a JSON-quoted string."""
    current = value
    for _ in range(2):
        if current == expected:
            return True
        if not isinstance(current, str):
            return False
        try:
            decoded = json.loads(current)
        except json.JSONDecodeError:
            return current.strip() == expected
        if decoded == current:
            return False
        current = decoded
    return current == expected


@dataclass
class SessionResult:
    exit_status: int
    timed_out: bool
    tokens: int
    stdout_tail: str
    stderr_tail: str
    session_id: str = ""
    terminal_usage: _agent_runtime.TokenUsage | None = None
    events: tuple[_agent_runtime.NormalizedAgentEvent, ...] = ()
    capabilities: _agent_runtime.AgentRuntimeCapabilities | None = None
    observation_errors: tuple[str, ...] = ()


def _render(template_path: Path, **kw: str) -> str:
    text = template_path.read_text(encoding="utf-8")
    mode_policy = kw.pop("MODE_POLICY", "")
    required = set(re.findall(r"\{\{([A-Z][A-Z0-9_]*)\}\}", text))
    missing = required - set(kw)
    if "MODE_POLICY" in missing and mode_policy:
        missing.remove("MODE_POLICY")
    if missing:
        raise ValueError(
            f"Prompt template {template_path} has unresolved fields: "
            + ", ".join(sorted(missing))
        )
    for key, val in kw.items():
        text = text.replace("{{" + key + "}}", str(val))
    if mode_policy:
        text = str(mode_policy).rstrip() + "\n\n" + text
    return text


def ensure_submodules(platform: str = "", arch: str = "") -> None:
    """Initialize submodules required by the optimization pipeline.

    The historical ncu-report-skill submodule is optional: profiling helpers
    used by Campaigns are vendored under tools/ncu_helpers/. KernelWiki's
    Agent-facing skill and the GPU Wiki JSON stores are repository-native.
    PPU campaigns also require their vendor reference projects: without those
    working trees the reference catalog contains no usable PPU implementation
    sources.
    Idempotent: already-initialized submodules are untouched.
    """
    needed = []
    if hardware_vendor(platform, arch) == "ppu":
        needed.extend(
            (path, REPO_ROOT / path / "README.md")
            for path in (
                "reference-projects/sailify",
                "reference-projects/hggc-samples",
                "reference-projects/FlashMLA-for-sail",
                "reference-projects/DeepGEMM-for-sail",
                "reference-projects/flash-attention-for-sail",
                "reference-projects/actlize",
                "reference-projects/triton-for-sail",
            )
        )
    # Internal launchers may expose a generated repository view without independent Git
    # metadata. Run submodule commands in the recorded open-source checkout while keeping
    # the current branch's intentionally small required-submodule set.
    submodule_root = REPO_ROOT
    runtime_metadata = REPO_ROOT / ".internal-runtime.json"
    if runtime_metadata.is_file():
        try:
            metadata = json.loads(runtime_metadata.read_text(encoding="utf-8"))
            recorded_root = metadata.get("open_root")
            if not isinstance(recorded_root, str) or not recorded_root.strip():
                raise ValueError("open_root is missing")
            submodule_root = Path(recorded_root).expanduser().resolve()
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"invalid internal runtime metadata: {runtime_metadata}: {exc}"
            ) from exc
    to_init = [path for path, marker in needed if not marker.exists()]
    if to_init:
        print(f"[orchestrator] initializing submodules: {to_init}", flush=True)
        cmd = ["git", "submodule", "update", "--init", "--depth", "1", "--"] + to_init
        subprocess.run(cmd, cwd=str(submodule_root), check=True)
        # verify
        for path, marker in needed:
            if not marker.exists():
                raise RuntimeError(
                    f"submodule init failed for {path} — {marker} not found. "
                    "Run `git submodule update --init` manually."
                )
        print("[orchestrator] all submodules ready", flush=True)


def run_session(
    workspace: Path,
    prompt: str,
    timeout: int | None,
    agent_cli: str = "claude",
    sandbox_hardware: str = "",
    sandbox_profile: str = "",
    sandbox_url: str = "",
    sandbox_timeout: int = DEFAULT_SANDBOX_TIMEOUT,
    reasoning_effort: str = "max",
    extra_environment: Optional[dict[str, str]] = None,
    agent_plugins: bool = True,
    sandbox_ssh: str = "",
    sandbox_ssh_init: str = "",
    sandbox_health_command: str = "",
) -> SessionResult:
    """Run one clean coding-agent session with no conversational memory from prior iterations."""
    # Kept for the dependency-review call contract. Runtime plan generation is now a
    # workspace-local skill rather than a process-level plugin.
    del agent_plugins
    session_id = str(uuid.uuid4())
    runtime = _agent_runtime.build_agent_runtime(
        agent_cli,
        process_runner=_agent_runtime.run_bounded,
    )
    result = runtime.run(
        _agent_runtime.AgentRunRequest(
            workspace=workspace,
            prompt=prompt,
            timeout_s=timeout,
            reasoning_effort=reasoning_effort,
            sandbox_hardware=sandbox_hardware,
            sandbox_profile=sandbox_profile,
            sandbox_url=sandbox_url,
            sandbox_ssh=sandbox_ssh,
            sandbox_ssh_init=sandbox_ssh_init,
            sandbox_health_command=sandbox_health_command,
            sandbox_timeout_s=sandbox_timeout,
            environment_state_file=str(environment_state_file() or ""),
            session_id=session_id,
            extra_environment=extra_environment,
        )
    )
    raise_if_environment_blocked()
    return SessionResult(
        exit_status=result.exit_status,
        timed_out=result.timed_out,
        tokens=result.tokens,
        stdout_tail=result.stdout_tail,
        stderr_tail=result.stderr_tail,
        session_id=result.session_id,
        terminal_usage=result.terminal_usage,
        events=result.events,
        capabilities=result.capabilities,
        observation_errors=result.observation_errors,
    )


def _production_review_candidate_paths(workspace: Path) -> list[Path]:
    """Return the complete, bounded source set shown to the production reviewer."""
    from aka.light.workspace.sources import policy_at, collect
    policy = policy_at(workspace)
    if policy is not None:
        return [workspace / name for name in sorted(collect(workspace, policy))]
    paths = [
        workspace / "kernel.py",
        workspace / "solution.json",
    ]
    return [path for path in paths if path.is_file()]


def _production_review_digest(
    workspace: Path,
    framework: str,
    require_gluon: bool,
    specify_dsl: bool = True,
    review_profile: str = "",
) -> str:
    digest = hashlib.sha256()
    digest.update(f"production-review-v{DEPENDENCY_REVIEW_SCHEMA_VERSION}\0".encode())
    digest.update(framework.encode("utf-8", errors="replace"))
    digest.update(b"\0")
    digest.update(review_profile.encode("utf-8", errors="replace"))
    digest.update(
        b"\0open-dsl\0"
        if not specify_dsl
        else b"\0gluon-required\0"
        if require_gluon
        else b"\0selected-framework\0"
    )
    for path in _production_review_candidate_paths(workspace):
        relative = path.relative_to(workspace).as_posix()
        contents = path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def _sandbox_endpoint(profile: str = "", url: str = "", ssh: str = "") -> str:
    """Render the endpoint clause shared by sandbox directives."""
    if ssh:
        return f" using OpenSSH target `{ssh}`"
    if url:
        return f" using gateway URL `{url}`"
    if profile:
        return f" using gateway profile `{profile}`"
    return " using agate's configured gateway"


def sandbox_directive(
    hardware: str,
    profile: str = "",
    url: str = "",
    ssh: str = "",
) -> str:
    """Execution boundary and GPU measurement guidance for optimization Episodes."""
    endpoint = _sandbox_endpoint(profile, url, ssh)
    safety = _render(
        SANDBOX_SAFETY_BOUNDARY_PROMPT, HARDWARE=hardware, ENDPOINT=endpoint
    )
    workflow = _render(
        SANDBOX_FULL_WORKFLOW_PROMPT, HARDWARE=hardware, ENDPOINT=endpoint
    )
    return f"{safety.rstrip()}\n\n{workflow.strip()}\n"


def sandbox_boundary_directive(hardware: str, profile: str = "", url: str = "", ssh: str = "") -> str:
    """Execution boundary without optimization guidance, for baseline sessions."""
    return _render(SANDBOX_SAFETY_BOUNDARY_PROMPT, HARDWARE=hardware,
                   ENDPOINT=_sandbox_endpoint(profile, url, ssh))


def _evaluation_request_shape_count(
    workspace: Path,
    private_reference_dir: Path | None,
    command: list[str],
    base_timeout: int,
    performance_timeout: int,
    grace: int,
    environment: dict[str, str],
) -> int:
    """Return the largest concrete Shape shard the child Eval will submit."""
    selected = []
    for index, value in enumerate(command):
        if value == "--shape-id" and index + 1 < len(command):
            selected.append(command[index + 1])
        elif value.startswith("--shape-id="):
            selected.append(value.split("=", 1)[1])
    selected = list(dict.fromkeys(item for item in selected if item))

    requested_batch_size = int(
        environment.get("ATREX_EVAL_SHAPE_BATCH_SIZE", str(SHAPE_BATCH_SIZE))
    )
    batch_size = effective_shape_batch_size(
        base_timeout,
        requested_batch_size,
        grace,
        performance_timeout=performance_timeout,
    )
    if selected:
        return min(len(selected), batch_size)

    shapes_path = (
        private_reference_dir / "shapes.json"
        if private_reference_dir is not None
        else workspace / "shapes.json"
    )
    try:
        shapes = json.loads(shapes_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        # The child owns custom/private request construction. Use its legal
        # maximum shard size when the parent cannot inspect the Shape contract.
        return batch_size
    shape_count = len(shapes) if isinstance(shapes, dict) else 0
    return min(shape_count, batch_size) if shape_count else batch_size


def _command_option(command: list[str], name: str) -> str | None:
    """Return the final value of a conventional ``--name VALUE`` option."""
    value = None
    for index, item in enumerate(command):
        if item == name and index + 1 < len(command):
            value = command[index + 1]
        elif item.startswith(name + "="):
            value = item.split("=", 1)[1]
    return value


def _evaluation_performance_timeout(command: list[str], step_timeout: int) -> int:
    """Return only the sequential per-Shape performance work in a request."""
    if (
        _command_option(command, "--baseline-path") is not None
        or _command_option(command, "--baseline-commit") is not None
    ):
        repeats_value = _command_option(command, "--comparison-repeats")
        repeats = int(repeats_value) if repeats_value is not None else DEFAULT_ABBA_REPEATS
        return 2 * repeats * step_timeout
    return step_timeout


def _sandbox_command(
    workspace: Path,
    hardware: str,
    profile: str,
    url: str,
    timeout: int,
    command: list[str],
    *,
    ssh: str = "",
    ssh_init: str = "",
    health_command: str = "",
    sync: tuple[str, ...] = (),
    wall_timeout: Optional[int] = None,
    gateway_kind: str = "auto",
    private_reference_dir: Path | None = None,
    preflight: bool = False,
    terminal_validation: bool = False,
    evaluation_step_timeout: int = DEFAULT_EVALUATION_STEP_TIMEOUT,
    gpu_clock_mhz: int | None = None,
    job_root: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the private GPU executor for trusted Supervisor verification."""
    if sum(bool(value) for value in (ssh, url, profile)) > 1:
        raise ValueError("ssh, url, and profile sandbox endpoints are mutually exclusive")
    cmd = [
        sys.executable,
        str(SANDBOX_TOOL),
        "--kind",
        gateway_kind,
        "--hardware",
        hardware,
        "--workspace",
        str(workspace),
        "--timeout",
        str(timeout),
        "--evaluation-step-timeout",
        str(evaluation_step_timeout),
    ]
    if gpu_clock_mhz is not None:
        if type(gpu_clock_mhz) is not int or gpu_clock_mhz <= 0:
            raise ValueError("gpu_clock_mhz must be a positive integer")
        cmd += ["--gpu-clock-mhz", str(gpu_clock_mhz)]
    if url:
        cmd += ["--url", url]
    elif profile:
        cmd += ["--gateway-profile", profile]
    elif ssh:
        cmd += ["--ssh", ssh]
    if ssh_init:
        cmd += ["--ssh-init", ssh_init]
    if health_command:
        cmd += ["--health-command", health_command]
    if sync:
        for path in sync:
            cmd += ["--sync", path]
    else:
        cmd.append("--no-sync")
    if preflight:
        cmd.append("--preflight")
    else:
        cmd += ["--", *command]
    environment = os.environ.copy()
    from aka.light.runtime.paths import supervisor_campaign_root
    from aka.light.experiment.store import JOB_ROOT_ENV
    from aka.light.execution.gateway_jobs import TERMINAL_VALIDATION_ENV
    # Bootstrap runs before the HTTP Runtime exists. Keep accepted IDs and the
    # exact uploaded wire request outside the workspace so recovery polls/replays
    # that same submission instead of creating an untracked replacement.
    environment[JOB_ROOT_ENV] = str(
        job_root
        if job_root is not None
        else supervisor_campaign_root(workspace) / "trusted-jobs"
    )
    environment.pop(TERMINAL_VALIDATION_ENV, None)
    if terminal_validation:
        environment[TERMINAL_VALIDATION_ENV] = "1"
    environment.pop("ATREX_PRIVATE_REFERENCE_DIR", None)
    if private_reference_dir is not None:
        environment["ATREX_PRIVATE_REFERENCE_DIR"] = str(private_reference_dir)
    # The gateway wait includes GPU queueing; --timeout bounds remote execution.
    # Keep the outer process alive for the same queue grace used by sandbox.py.
    if terminal_validation:
        effective_timeout = None
    elif wall_timeout is not None:
        effective_timeout = wall_timeout
    else:
        uses_ssh = bool(ssh or (not profile and not url and environment.get("ATREX_SANDBOX_SSH")))
        grace = 0 if uses_ssh else queue_wait_grace(environment)
        command_timeout = timeout
        if not uses_ssh and gateway_kind == "run":
            performance_timeout = _evaluation_performance_timeout(
                command, evaluation_step_timeout
            )
            shape_count = _evaluation_request_shape_count(
                workspace,
                private_reference_dir,
                command,
                timeout,
                performance_timeout,
                grace,
                environment,
            )
            command_timeout = shape_request_timeout(
                timeout,
                shape_count,
                performance_timeout=performance_timeout,
            )
        effective_timeout = command_timeout + grace + 240
    raise_if_environment_blocked()
    process = spawn_owned_session(
        cmd,
        role="sandbox",
        environment=environment,
        cwd=str(workspace),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    def stop_process_group() -> tuple[str, str]:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            # The private executor handles SIGTERM with bounded SSH cleanup;
            # allow that 15-second cleanup window to persist a retry marker.
            return process.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return process.communicate()

    deadline = (
        time.monotonic() + effective_timeout
        if effective_timeout is not None
        else None
    )
    try:
        while True:
            remaining = deadline - time.monotonic() if deadline is not None else None
            if remaining is not None and remaining <= 0:
                stdout, stderr = stop_process_group()
                raise subprocess.TimeoutExpired(
                    cmd, effective_timeout, output=stdout, stderr=stderr
                )
            try:
                stdout, stderr = process.communicate(
                    timeout=min(0.25, remaining) if remaining is not None else 0.25
                )
            except subprocess.TimeoutExpired:
                raise_if_environment_blocked()
                continue
            break
    except BaseException:
        # A first-run sandbox has its own session and no recovery guardian.
        # Reap it on Ctrl-C and every exceptional exit, not just wall timeout.
        try:
            stop_process_group()
        except BaseException:
            # A second interrupt must not abandon the independently owned group
            # or replace the original error. The normal path allows SSH cleanup.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        raise
    result = subprocess.CompletedProcess(
        args=cmd,
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
    )
    raise_if_environment_blocked()
    return result


def check_ssh_environment(
    workspace: Path, hardware: str, ssh: str, ssh_init: str, health_command: str
) -> None:
    """Fail into durable recovery before seeding, even when --arch was given."""
    result = _sandbox_command(
        workspace, hardware, "", "", 60, [], ssh=ssh, ssh_init=ssh_init,
        health_command=health_command, preflight=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"SSH preflight failed: {result.stderr or result.stdout}")


def _test_result_from_stdout(stdout: str) -> dict:
    """Read the structured result emitted by the active sandbox harness."""
    from aka.light.evaluation.result_json import parse_evaluation_result_line

    for line in reversed(stdout.splitlines()):
        result = parse_evaluation_result_line(line)
        if result is not None:
            return result
    raise RuntimeError("sandbox test output has no structured Evaluate result")


def _local_test_record(workspace: Path, version: str, result: dict, *, existing: dict | None = None) -> dict:
    """Build normalized evaluator facts without choosing their storage boundary."""
    if not re.fullmatch(r"v[0-9]+", version):
        raise ValueError("Invalid canonical version")
    number = int(version[1:])
    data = dict(existing or {})
    data.setdefault("version", version)
    data.setdefault("masked", False)
    data["timestamp"] = datetime.now(timezone.utc).isoformat()
    perf = data.setdefault("performance", {})
    perf["latency_us"] = result.get("latency_us_geomean", 0.0)
    perf["latency_us_geomean"] = result.get("latency_us_geomean", 0.0)
    perf["latency_us_arith_mean"] = result.get("latency_us_arith_mean", 0.0)
    by_shape = result.get("latency_us_by_shape", {})
    by_shape = by_shape if isinstance(by_shape, dict) else {}
    perf["latency_us_by_shape"] = by_shape
    perf["measurement_scope"] = "real_evaluator_shapes"
    generalized = (workspace / "agent_problem.json").is_file()
    perf["shape_ids_are_opaque"] = generalized
    if generalized:
        perf["shape_visibility"] = "hidden"
        perf["public_shape_ids"] = []
    else:
        perf["shape_visibility"] = "all"
    perf["measurement_status"] = (
        "complete" if result.get("all_pass") and by_shape else "incomplete"
    )
    perf["measured_shape_count"] = len(by_shape)
    performance_objective = result.get("performance_objective")
    perf["performance_objective"] = performance_objective
    perf["performance_score"] = result.get(
        "performance_score", result.get("speedup_vs_ref_geomean")
    )
    perf["speedup_vs_ref_mean"] = result.get("speedup_vs_ref_mean")
    perf["speedup_vs_ref_geomean"] = result.get("speedup_vs_ref_geomean")
    all_pass = bool(result.get("all_pass"))
    corr = data.setdefault("correctness", {})
    corr["status"] = "PASS" if all_pass else "FAIL"
    corr["max_abs_err"] = result.get("max_abs_err", 0.0)
    corr["max_rel_err"] = result.get("max_rel_err", 0.0)
    gate = data.setdefault("quality_gate", {})
    gate["result"] = "PASS" if all_pass else "FAIL"
    failures = result.get("failures") or []
    gate["failure_reason"] = None if all_pass else "; ".join(map(str, failures))[:2000]
    return data


def _record_local_test_result(workspace: Path, version: str, result: dict) -> Path:
    """Persist legacy non-V0 version facts; V0 is a Gateway Measurement."""
    from aka.light.workspace.versions import save_version_record, version_record_path
    number = int(version[1:]) if re.fullmatch(r"v[0-9]+", version) else -1
    if number == 0:
        raise ValueError("V0 evaluator facts must use the Campaign Measurement Store")
    path = version_record_path(workspace, number)
    try:
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        existing = {}
    data = _local_test_record(workspace, version, result, existing=existing)
    return save_version_record(workspace, number, data)


def detect_arch(
    sandbox_hardware: str = "",
    sandbox_profile: str = "",
    sandbox_url: str = "",
    sandbox_ssh: str = "",
    sandbox_ssh_init: str = "",
    sandbox_health_command: str = "",
) -> str:
    """Return the real runtime GPU architecture token (vendor-neutral), or '' if undetectable.

    NVIDIA/CUDA -> 'sm_<cap>' (e.g. 'sm_103'); AMD/ROCm -> the gfx arch (e.g. 'gfx942').
    Uses torch (get_device_capability / gcnArchName) — the AUTHORITATIVE source, which stays
    correct even when the GPU name / vendor SMI is DESENSITIZED (e.g. a target GPU reporting a
    generic compatibility alias).
    """
    code = (
        "import torch\n"
        "p=torch.cuda.get_device_properties(0)\n"
        "if getattr(torch.version,'hip',None):\n"
        "    print(getattr(p,'gcnArchName','').split(':')[0])\n"
        "else:\n"
        "    c=torch.cuda.get_device_capability(0); print('sm_%d%d'%(c[0],c[1]))\n"
    )
    if sandbox_hardware:
        try:
            with tempfile.TemporaryDirectory(prefix="atrex-arch-") as temp_dir:
                result = _sandbox_command(
                    Path(temp_dir),
                    sandbox_hardware,
                    sandbox_profile,
                    sandbox_url,
                    120,
                    ["python", "-c", code],
                    ssh=sandbox_ssh,
                    ssh_init=sandbox_ssh_init,
                    health_command=sandbox_health_command,
                )
            if result.returncode == 0:
                for line in reversed(result.stdout.splitlines()):
                    value = line.strip()
                    if re.fullmatch(r"sm_\d+|gfx[0-9a-fA-F]+", value):
                        return value
            print(
                f"[orchestrator] WARNING: sandbox arch detection failed on {sandbox_hardware}: "
                f"{result.stderr[-1000:]}",
                file=sys.stderr,
                flush=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            print(
                f"[orchestrator] WARNING: sandbox arch detection failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
        return ""

    for py in ("python", "python3", sys.executable):
        try:
            out = subprocess.run(
                [py, "-c", code], capture_output=True, text=True, timeout=120
            )
            s = out.stdout.strip()
            if s:
                return s
        except (OSError, subprocess.SubprocessError):
            continue
    return ""
