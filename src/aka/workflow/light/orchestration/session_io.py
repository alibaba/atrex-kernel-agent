"""Coding-agent sessions, independent production review and Workflow result projection."""

from __future__ import annotations
from aka.workflow.legacy.review_rules.review import _validate_production_review, _validate_review_item

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
    TEST_RESULT_PREFIX,
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


def _production_review_candidate_paths(workspace: Path, *, workspace_provider) -> list[Path]:
    """Read the complete owner-validated source set shown to the reviewer."""
    source = workspace_provider.read_source(workspace)
    names = [name for name, content in source.files.items() if content is not None]
    if set(names) == {"kernel.py"}:
        manifest = workspace_provider.read_working(workspace, ("solution.json",)).files["solution.json"]
        if manifest is not None:
            names.append("solution.json")
    return [workspace / name for name in sorted(names)]


def _production_review_digest(
    workspace: Path,
    framework: str,
    require_gluon: bool,
    *, workspace_provider,
) -> str:
    digest = hashlib.sha256()
    digest.update(f"production-review-v{DEPENDENCY_REVIEW_SCHEMA_VERSION}\0".encode())
    digest.update(framework.encode("utf-8", errors="replace"))
    digest.update(b"\0gluon-required\0" if require_gluon else b"\0selected-framework\0")
    for path in _production_review_candidate_paths(workspace, workspace_provider=workspace_provider):
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
    for line in reversed(stdout.splitlines()):
        if line.startswith(TEST_RESULT_PREFIX):
            result = json.loads(line[len(TEST_RESULT_PREFIX) :])
            if isinstance(result, dict):
                return result
    raise RuntimeError("sandbox test output has no structured RESULT_JSON line")


def _record_local_test_result(workspace: Path, version: str, result: dict, *, generate_memory: bool = False) -> Path:
    """Persist evaluator facts; publish an Agent memory summary only on opt-in."""
    from aka.workflow.light.orchestration.version_records import save_version_record, version_record_path
    if not re.fullmatch(r"v[0-9]+", version):
        raise ValueError("Invalid canonical version")
    number = int(version[1:])
    path = version_record_path(workspace, number, prefer_summary=generate_memory)
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
    return save_version_record(workspace, number, data, generate_memory=generate_memory)
