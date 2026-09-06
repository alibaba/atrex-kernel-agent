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

from . import agent_runtime as _agent_runtime
from .constants import (
    DEFAULT_SANDBOX_TIMEOUT,
    DEPENDENCY_REVIEW_SCHEMA_VERSION,
    REPO_ROOT,
    SANDBOX_FULL_WORKFLOW_PROMPT,
    SANDBOX_SAFETY_BOUNDARY_PROMPT,
    SANDBOX_TOOL,
    TEST_RESULT_PREFIX,
)
from .environment_recovery import (
    environment_state_file,
    raise_if_environment_blocked,
)
from .hardware import hardware_vendor
from .recovery_processes import spawn_owned_session
from .workspace_state import speedup_vs_reference


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
    for key, val in kw.items():
        text = text.replace("{{" + key + "}}", str(val))
    if mode_policy:
        text = str(mode_policy).rstrip() + "\n\n" + text
    return text


def ensure_submodules(platform: str = "", arch: str = "") -> None:
    """Initialize submodules required by the optimization pipeline.

    Always covers gpu-wiki/3rdparty (KernelWiki) and 3rdparty/ncu-report-skill.
    PPU campaigns also require their vendor reference projects: without those
    working trees the framework-baseline catalog silently contains no usable
    PPU implementation sources.
    Idempotent: already-initialized submodules are untouched.
    """
    needed = [
        (
            "gpu-wiki/3rdparty/",
            REPO_ROOT / "gpu-wiki" / "3rdparty" / "KernelWiki" / "README.md",
        ),
        (
            "3rdparty/ncu-report-skill",
            REPO_ROOT / "3rdparty" / "ncu-report-skill" / "SKILL.md",
        ),
    ]
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
    timeout: int,
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


_DEPENDENCY_ALLOW_CATEGORIES = {
    "toolchain_plumbing",
    "framework_runtime",
    "support_utility",
}
_DEPENDENCY_REJECT_CATEGORIES = {
    "prebuilt_compute",
    "alternate_framework",
    "hidden_dispatch",
    "external_code",
    "unresolved",
}

_PRODUCTION_REVIEW_CHECK_CATEGORIES = {
    "framework_compliance": {
        "allow": {"framework_compliant"},
        "reject": {"alternate_framework", "unresolved"},
    },
    "compute_provenance": {
        "allow": {"self_authored_compute"},
        "reject": {
            "prebuilt_compute",
            "torch_compute",
            "hidden_dispatch",
            "external_code",
            "unresolved",
        },
    },
    "dependency_inventory": {
        "allow": {"inventory_complete"},
        "reject": {"unresolved"},
    },
    "external_code_loading": {
        "allow": {"no_external_code"},
        "reject": {"external_code", "hidden_dispatch", "unresolved"},
    },
    "solution_manifest": {
        "allow": {"manifest_consistent", "not_applicable"},
        "reject": {"manifest_mismatch", "unresolved"},
    },
}


def _production_review_candidate_paths(workspace: Path) -> list[Path]:
    """Return the complete, bounded source set shown to the production reviewer."""
    paths = [
        workspace / "kernel.py",
        workspace / "solution.json",
    ]
    return [path for path in paths if path.is_file()]


def _production_review_digest(
    workspace: Path,
    framework: str,
    require_gluon: bool,
) -> str:
    digest = hashlib.sha256()
    digest.update(f"production-review-v{DEPENDENCY_REVIEW_SCHEMA_VERSION}\0".encode())
    digest.update(framework.encode("utf-8", errors="replace"))
    digest.update(b"\0gluon-required\0" if require_gluon else b"\0selected-framework\0")
    for path in _production_review_candidate_paths(workspace):
        relative = path.relative_to(workspace).as_posix()
        contents = path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def _validate_review_item(
    item: object,
    *,
    label: str,
    allow_categories: set[str],
    reject_categories: set[str],
    candidate_files: frozenset[str],
) -> tuple[str, str, str]:
    if not isinstance(item, dict):
        raise ValueError(f"production review {label} must be an object")
    decision = item.get("decision")
    category = item.get("category")
    reason = item.get("reason")
    evidence = item.get("evidence")
    if decision not in {"allow", "reject"}:
        raise ValueError(f"production review decision is invalid for {label}")
    categories = allow_categories if decision == "allow" else reject_categories
    if category not in categories:
        raise ValueError(
            f"production review category {category!r} is inconsistent with "
            f"decision {decision!r} for {label}"
        )
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"production review reason is empty for {label}")
    if (
        not isinstance(evidence, list)
        or not evidence
        or not all(isinstance(value, str) and value.strip() for value in evidence)
    ):
        raise ValueError(f"production review evidence is invalid for {label}")
    candidate_evidence_found = False
    for value in evidence:
        stripped = value.strip()
        match = re.fullmatch(
            r"candidate/(kernel\.py|solution\.json)(?::.+)?",
            stripped,
        )
        if match is not None and match.group(1) in candidate_files:
            candidate_evidence_found = True
            continue
        # The review request is trusted supervisor context rather than candidate
        # evidence.  Reviewers occasionally cite it alongside candidate source when
        # explaining the selected framework.  Permit that supplemental citation,
        # while still requiring source-backed candidate evidence for every item.
        if re.fullmatch(r"review_request\.json:[A-Za-z0-9_.-]+", stripped):
            continue
        raise ValueError(
            "production review evidence may only cite supplied candidate files "
            f"or review_request.json metadata for {label}"
        )
    if not candidate_evidence_found:
        raise ValueError(
            f"production review evidence must cite a supplied candidate file for {label}"
        )
    return decision, str(category), reason.strip()


def _validate_production_review(
    payload: object,
    *,
    candidate_files: frozenset[str] = frozenset({"kernel.py", "solution.json"}),
) -> tuple[list[str], str]:
    """Validate a complete candidate verdict and translate rejections into errors."""
    if not isinstance(payload, dict):
        raise ValueError("production review must be a JSON object")
    if payload.get("schema_version") != DEPENDENCY_REVIEW_SCHEMA_VERSION:
        raise ValueError("production review has an unsupported schema_version")
    verdict = payload.get("verdict")
    if verdict not in {"allow", "reject"}:
        raise ValueError("production review verdict must be allow or reject")
    summary = payload.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("production review summary must be non-empty")
    checks = payload.get("checks")
    if not isinstance(checks, list):
        raise ValueError("production review checks must be a list")

    expected = set(_PRODUCTION_REVIEW_CHECK_CATEGORIES)
    reviewed: set[str] = set()
    rejected: list[str] = []
    for item in checks:
        if not isinstance(item, dict):
            raise ValueError("production review check must be an object")
        check_id = item.get("id")
        if not isinstance(check_id, str) or check_id not in expected:
            raise ValueError(
                f"production review returned unexpected check id: {check_id!r}"
            )
        if check_id in reviewed:
            raise ValueError(f"production review duplicated check id: {check_id}")
        categories = _PRODUCTION_REVIEW_CHECK_CATEGORIES[check_id]
        decision, _category, reason = _validate_review_item(
            item,
            label=f"check {check_id}",
            allow_categories=categories["allow"],
            reject_categories=categories["reject"],
            candidate_files=candidate_files,
        )
        reviewed.add(check_id)
        if decision == "reject":
            rejected.append(
                "production candidate rejected by independent reviewer: "
                f"{check_id}: {reason}"
            )

    missing = sorted(expected - reviewed)
    if missing:
        raise ValueError("production review omitted check ids: " + ", ".join(missing))

    dependencies = payload.get("dependencies")
    if not isinstance(dependencies, list):
        raise ValueError("production review dependencies must be a list")
    dependency_names: set[str] = set()
    for item in dependencies:
        if not isinstance(item, dict):
            raise ValueError("production review dependency must be an object")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("production review dependency name must be non-empty")
        normalized_name = name.strip().lower()
        if normalized_name in dependency_names:
            raise ValueError(f"production review duplicated dependency: {name.strip()}")
        dependency_names.add(normalized_name)
        decision, _category, reason = _validate_review_item(
            item,
            label=f"dependency {name.strip()}",
            allow_categories=_DEPENDENCY_ALLOW_CATEGORIES,
            reject_categories=_DEPENDENCY_REJECT_CATEGORIES,
            candidate_files=candidate_files,
        )
        if decision == "reject":
            rejected.append(
                "third-party dependency rejected by independent reviewer: "
                f"{name.strip()}: {reason}"
            )

    expected_verdict = "reject" if rejected else "allow"
    if verdict != expected_verdict:
        raise ValueError(
            f"production review verdict {verdict!r} disagrees with item decisions"
        )
    return rejected, summary.strip()


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
    """Mandatory safety boundary plus full-mode workflow for full episodes."""
    endpoint = _sandbox_endpoint(profile, url, ssh)
    safety = _render(
        SANDBOX_SAFETY_BOUNDARY_PROMPT, HARDWARE=hardware, ENDPOINT=endpoint
    )
    workflow = _render(
        SANDBOX_FULL_WORKFLOW_PROMPT, HARDWARE=hardware, ENDPOINT=endpoint
    )
    return f"{safety.rstrip()}\n\n{workflow.strip()}\n"


def fast_sandbox_directive(
    hardware: str,
    profile: str = "",
    url: str = "",
    ssh: str = "",
) -> str:
    """Mandatory safety boundary for fast episodes.

    The fast episode prompt already describes the fast-specific execution
    contract (single evaluator, no multi-seed, no profile, supervisor-owned
    memory), so only the invariant safety boundary is injected here.
    """
    endpoint = _sandbox_endpoint(profile, url, ssh)
    return _render(
        SANDBOX_SAFETY_BOUNDARY_PROMPT, HARDWARE=hardware, ENDPOINT=endpoint
    )


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
) -> subprocess.CompletedProcess[str]:
    """Run one command through tools/sandbox.py and capture its user-visible output."""
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
    ]
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
    environment.pop("ATREX_PRIVATE_REFERENCE_DIR", None)
    if private_reference_dir is not None:
        environment["ATREX_PRIVATE_REFERENCE_DIR"] = str(private_reference_dir)
    effective_timeout = wall_timeout if wall_timeout is not None else timeout + 240
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
            # tools/sandbox.py handles SIGTERM by running its bounded SSH cleanup;
            # allow that 15-second cleanup window to persist a retry marker.
            return process.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return process.communicate()

    deadline = time.monotonic() + effective_timeout
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                stdout, stderr = stop_process_group()
                raise subprocess.TimeoutExpired(
                    cmd, effective_timeout, output=stdout, stderr=stderr
                )
            try:
                stdout, stderr = process.communicate(timeout=min(0.25, remaining))
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
    for line in reversed(stdout.splitlines()):
        if line.startswith(TEST_RESULT_PREFIX):
            result = json.loads(line[len(TEST_RESULT_PREFIX) :])
            if isinstance(result, dict):
                return result
    raise RuntimeError("sandbox test output has no structured RESULT_JSON line")


def _record_local_test_result(workspace: Path, version: str, result: dict) -> Path:
    """Merge a remote --no-memory test result into local optimizer memory."""
    mem_dir = workspace / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    path = mem_dir / f"{version}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        data = {}
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
    perf["shape_ids_are_opaque"] = (workspace / "agent_problem.json").is_file()
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
    perf["speedup_vs_ref_geomean"] = (
        None
        if performance_objective == "shape_speedup_arithmetic_mean"
        else speedup_vs_reference(
            workspace,
            result.get("latency_us_geomean"),
            result.get("speedup_vs_ref_geomean"),
        )
    )
    all_pass = bool(result.get("all_pass"))
    corr = data.setdefault("correctness", {})
    corr["status"] = "PASS" if all_pass else "FAIL"
    corr["max_abs_err"] = result.get("max_abs_err", 0.0)
    corr["max_rel_err"] = result.get("max_rel_err", 0.0)
    gate = data.setdefault("quality_gate", {})
    gate["result"] = "PASS" if all_pass else "FAIL"
    failures = result.get("failures") or []
    gate["failure_reason"] = None if all_pass else "; ".join(map(str, failures))[:2000]
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


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
