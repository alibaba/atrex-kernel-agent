"""Execution sandbox implementation and callable interface."""
from __future__ import annotations

import json
from aka.legacy.resources.prompts import render_prompt
from aka.legacy.resources.paths import SANDBOX_FULL_WORKFLOW_PROMPT, SANDBOX_SAFETY_BOUNDARY_PROMPT
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional
from aka.legacy.resources.paths import SANDBOX_TOOL
from aka.legacy.execution.recovery import raise_if_environment_blocked
from aka.legacy.execution.processes import spawn_owned_session
from aka.legacy.execution.configuration import queue_wait_grace
import re
import tempfile


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
    safety = render_prompt(
        SANDBOX_SAFETY_BOUNDARY_PROMPT, HARDWARE=hardware, ENDPOINT=endpoint
    )
    workflow = render_prompt(
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
    return render_prompt(SANDBOX_SAFETY_BOUNDARY_PROMPT, HARDWARE=hardware, ENDPOINT=endpoint)


def sandbox_command(
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
    cancel_event=None,
) -> subprocess.CompletedProcess[str]:
    """Run one command through tools/sandbox.py and capture its user-visible output."""
    from aka.legacy.application import execution
    from aka.legacy.execution.legacy import runtime
    environment = os.environ.copy()
    environment.pop("ATREX_PRIVATE_REFERENCE_DIR", None)
    if private_reference_dir is not None:
        environment["ATREX_PRIVATE_REFERENCE_DIR"] = str(private_reference_dir)
    request = runtime.SandboxRequest(
        workspace=workspace, launcher=(sys.executable, str(SANDBOX_TOOL)),
        hardware=hardware, profile=profile, url=url, timeout=timeout,
        command=tuple(command), environment=execution.execution_environment(environment),
        ssh=ssh, ssh_init=ssh_init, health_command=health_command, sync=sync,
        wall_timeout=wall_timeout, gateway_kind=gateway_kind, preflight=preflight,
        cancel_event=cancel_event,
    )
    hooks = runtime.ExecutionHooks(
        spawn=spawn_owned_session, check_environment=raise_if_environment_blocked,
        queue_wait_grace=queue_wait_grace,
    )
    with execution.application_execution() as executor:
        return execution.run_sandbox(executor, request, hooks=hooks)


def check_ssh_environment(
    workspace: Path, hardware: str, ssh: str, ssh_init: str, health_command: str
) -> None:
    """Fail into durable recovery before seeding, even when --arch was given."""
    result = sandbox_command(
        workspace,
        hardware,
        "",
        "",
        60,
        [],
        ssh=ssh,
        ssh_init=ssh_init,
        health_command=health_command,
        preflight=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"SSH preflight failed: {result.stderr or result.stdout}")


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
                result = sandbox_command(
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

