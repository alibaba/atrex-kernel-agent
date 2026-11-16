"""Coding-agent sessions, independent production review and Workflow result projection."""

from __future__ import annotations
from aka.acceptance.review import _validate_production_review, _validate_review_item

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from aka.contracts.agent import AgentRunRequest, AgentRuntimeCapabilities, NormalizedAgentEvent, TokenUsage
from .constants import (
    DEFAULT_SANDBOX_TIMEOUT,
    DEPENDENCY_REVIEW_SCHEMA_VERSION,
    SANDBOX_FULL_WORKFLOW_PROMPT,
    SANDBOX_SAFETY_BOUNDARY_PROMPT,
)
from aka.execution.recovery import (
    raise_if_environment_blocked,
)
from .workspace_state import speedup_vs_reference


@dataclass
class SessionResult:
    exit_status: int
    timed_out: bool
    tokens: int
    stdout_tail: str
    stderr_tail: str
    session_id: str = ""
    terminal_usage: TokenUsage | None = None
    events: tuple[NormalizedAgentEvent, ...] = ()
    capabilities: AgentRuntimeCapabilities | None = None
    observation_errors: tuple[str, ...] = ()


def _render(template_path: Path, **kw: str) -> str:
    text = template_path.read_text(encoding="utf-8")
    mode_policy = kw.pop("MODE_POLICY", "")
    for key, val in kw.items():
        text = text.replace("{{" + key + "}}", str(val))
    if mode_policy:
        text = str(mode_policy).rstrip() + "\n\n" + text
    return text


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
    *, agent=None, access=None,
) -> SessionResult:
    """Run one clean coding-agent session with no conversational memory from prior iterations."""
    # Kept for the dependency-review call contract. Runtime plan generation is now a
    # workspace-local skill rather than a process-level plugin.
    del agent_plugins, sandbox_hardware, sandbox_profile, sandbox_url, sandbox_timeout
    del sandbox_ssh, sandbox_ssh_init, sandbox_health_command
    if agent is None:
        raise ValueError("Coding sessions require the selected Agent")
    session_id = str(uuid.uuid4())
    result = agent.runtime(agent_cli).run(
        AgentRunRequest(
            workspace=workspace,
            prompt=prompt,
            timeout_s=timeout,
            reasoning_effort=reasoning_effort,
            session_id=session_id,
            environment=extra_environment,
            access=access,
        )
    )
    raise_if_environment_blocked()
    return SessionResult(
        exit_status=result.exit_status,
        timed_out=result.timed_out,
        tokens=result.terminal_usage.total_tokens or 0,
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
    from aka.task.source.formats import policy_at, collect
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


def _test_result_from_stdout(stdout: str) -> dict:
    """Read the structured result emitted by the active sandbox harness."""
    from aka.evaluation.result_json import parse_evaluation_result_line

    for line in reversed(stdout.splitlines()):
        result = parse_evaluation_result_line(line)
        if result is not None:
            return result
    raise RuntimeError("sandbox test output has no structured Evaluate result")
