"""Optimization-mode policy and production-candidate enforcement."""

from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Callable

from .git_metadata import install_git_excludes
from .durable_state import durable_write_json


OPTIMIZATION_MODE_CHOICES = ("leaderboard", "production")
MODE_STATE_FILE = ".orchestrator_mode.json"
MODE_STATE_ENV = "ATREX_AKA_MODE_STATE_FILE"
POLICY_BEGIN = "<!-- ATREX_OPTIMIZATION_MODE_POLICY_BEGIN -->"
POLICY_END = "<!-- ATREX_OPTIMIZATION_MODE_POLICY_END -->"


ProductionReviewer = Callable[[Path, str, bool], list[str]]


def workspace_policy_path(workspace: Path) -> Path:
    from .supervisor_runtime import supervisor_campaign_root

    return supervisor_campaign_root(workspace) / "optimization-policy.json"


def read_workspace_policy(workspace: Path) -> dict:
    """Read trusted policy; malformed private state must not disable private cases."""
    path = workspace_policy_path(workspace)
    return read_policy_file(path)


def read_policy_file(path: Path) -> dict:
    if path.is_symlink():
        raise RuntimeError("optimization-mode state cannot be a symlink")
    if not path.exists():
        return {}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid optimization-mode state: {path}") from exc
    if not isinstance(state, dict) or state.get("mode") not in OPTIMIZATION_MODE_CHOICES:
        raise RuntimeError(f"invalid optimization-mode state: {path}")
    return state


def _framework_key(framework: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "", framework.strip().lower())
    aliases = {
        "triton": "triton",
        "gluon": "gluon",
        "tritongluon": "gluon",
        "cutedsl": "cutedsl",
        "cute": "cutedsl",
        "cuda": "cuda",
        "cudac": "cuda",
        "flydsl": "flydsl",
        "fly": "flydsl",
    }
    return aliases.get(token, token)


def source_uses_gluon(source: str) -> bool:
    """Return whether Python source imports the Triton experimental Gluon DSL.

    Parse imports instead of searching for the word ``gluon`` so comments, strings,
    and failure notes cannot accidentally satisfy the mandatory conversion gate.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                alias.name == "triton.experimental.gluon"
                or alias.name.startswith("triton.experimental.gluon.")
                for alias in node.names
            ):
                return True
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "triton.experimental" and any(
                alias.name == "gluon" for alias in node.names
            ):
                return True
            if module == "triton.experimental.gluon" or module.startswith(
                "triton.experimental.gluon."
            ):
                return True
    return False


def optimization_mode_directive(mode: str, framework: str) -> str:
    """Describe legal implementations without exposing Supervisor gate mechanics."""
    if mode == "leaderboard":
        return (
            "## Implementation constraints\n\n"
            f"Prefer {framework}; compatible mixed/alternate implementations and preinstalled "
            "third-party libraries are allowed when supported by evidence.\n"
        )
    if mode != "production":
        raise ValueError(f"unsupported optimization mode: {mode!r}")
    if _framework_key(framework) == "triton":
        framework_rule = (
            "- Use Triton until the session explicitly requires Gluon; after conversion, stay in "
            "Gluon. Do not mix their compute kernels or use another DSL.\n"
        )
    else:
        framework_rule = (
            f"- Implement GPU computation in {framework} only; do not switch or mix DSLs.\n"
        )
    return (
        "## Implementation constraints\n\n"
        f"{framework_rule}"
        "- Write the compute kernels yourself. No prebuilt operators/math implementations, PyTorch "
        "compute fallbacks, hidden dispatch, or external implementation loading. The original V0 "
        "reference is the baseline exception, not an optimized candidate.\n"
        "- Preinstalled compiler bindings, header discovery, ABI/launch helpers, and non-compute "
        "utilities are allowed only to support your own kernels.\n"
        "- Keep `solution.json`, when present, consistent with the implementation.\n"
    )


def workspace_policy_block(mode: str, framework: str) -> str:
    directive = optimization_mode_directive(mode, framework).rstrip()
    return f"{POLICY_BEGIN}\n\n{directive}\n\n{POLICY_END}\n"


def install_workspace_policy(
    workspace: Path,
    mode: str,
    framework: str,
    *,
    agent_runtime: str | None = None,
) -> None:
    """Persist immutable mode, framework, and optional campaign runtime identity.

    Existing workspaces without ``agent_runtime`` remain readable. Their first
    explicit post-upgrade runtime is adopted before a session starts; later
    attempts to resume with another backend fail closed.
    """
    if mode not in OPTIMIZATION_MODE_CHOICES:
        raise ValueError(f"unsupported optimization mode: {mode!r}")
    requested_runtime = (
        str(agent_runtime).strip() if agent_runtime is not None else None
    )
    if agent_runtime is not None and not requested_runtime:
        raise ValueError("agent_runtime must be a non-empty runtime id")

    workspace.mkdir(parents=True, exist_ok=True)
    state_path = workspace_policy_path(workspace)
    legacy_path = workspace / MODE_STATE_FILE
    state_changed = False
    state = read_policy_file(state_path)
    if not state and legacy_path.exists():
        state = read_policy_file(legacy_path)
        state_changed = True
    if state:
        existing_mode = state.get("mode")
        existing_framework = state.get("framework")
        if existing_mode != mode or existing_framework != framework:
            raise RuntimeError(
                "workspace policy mismatch: "
                f"recorded mode/framework={existing_mode}/{existing_framework}, "
                f"requested={mode}/{framework}"
            )
        existing_runtime = str(state.get("agent_runtime") or "").strip()
        if requested_runtime and existing_runtime and existing_runtime != requested_runtime:
            raise RuntimeError(
                "workspace agent runtime mismatch: "
                f"recorded={existing_runtime}, requested={requested_runtime}; "
                "use a fresh campaign workspace to change backend"
            )
        if requested_runtime and not existing_runtime:
            state["agent_runtime"] = requested_runtime
            state_changed = True
    else:
        state = {"mode": mode, "framework": framework}
        if requested_runtime:
            state["agent_runtime"] = requested_runtime
        state_changed = True

    if state_changed:
        durable_write_json(state_path, state, indent=2)

    # Retain tracked legacy files in Git history, but never project them into an Agent.
    if legacy_path.is_file() and not legacy_path.is_symlink():
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", MODE_STATE_FILE],
            cwd=workspace, capture_output=True, check=False,
        )
        if tracked.returncode:
            legacy_path.unlink()

    claude_path = workspace / "CLAUDE.md"
    current = claude_path.read_text(encoding="utf-8") if claude_path.exists() else ""
    generated = workspace_policy_block(mode, framework)
    if POLICY_BEGIN in current and POLICY_END in current:
        before, remainder = current.split(POLICY_BEGIN, 1)
        _, after = remainder.split(POLICY_END, 1)
        current = before.rstrip() + "\n\n" + generated + after.lstrip("\n")
    else:
        current = current.rstrip() + ("\n\n" if current.strip() else "") + generated
    claude_path.write_text(current, encoding="utf-8")

    install_git_excludes(workspace)


_SUPPORTED_PRODUCTION_FRAMEWORKS = frozenset(
    {"triton", "gluon", "cutedsl", "cuda", "flydsl", "tilelang"}
)


def _has_relative_import(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            return True
    return False


def _solution_structure_violations(workspace: Path) -> list[str]:
    """Validate only manifest structure that the campaign must be able to version."""
    solution_path = workspace / "solution.json"
    if not solution_path.is_file():
        return []
    try:
        solution = json.loads(solution_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"solution.json is invalid: {exc}"]
    if not isinstance(solution, dict):
        return ["solution.json must contain a JSON object"]

    spec = solution.get("spec") or {}
    sources = solution.get("sources") or []
    external_paths: list[str] = []
    if isinstance(spec, dict):
        entry_point = spec.get("entry_point")
        if isinstance(entry_point, str) and "::" in entry_point:
            entry_path = entry_point.split("::", 1)[0].strip()
            if entry_path and entry_path != "kernel.py":
                external_paths.append(entry_path)
    if isinstance(sources, list):
        for source in sources:
            if not isinstance(source, dict):
                continue
            path = source.get("path")
            if isinstance(path, str) and path.strip() and path.strip() != "kernel.py":
                external_paths.append(path.strip())
    if external_paths:
        return [
            "solution.json references candidate source outside kernel.py that the campaign "
            "cannot version or embed: " + ", ".join(dict.fromkeys(external_paths))
        ]
    return []


def production_structure_violations(
    workspace: Path,
    framework: str,
    *,
    require_gluon: bool = False,
) -> list[str]:
    """Return only mechanically certain production-candidate violations.

    Framework ownership, compute provenance, dependency use, dynamic loading, and manifest
    semantics deliberately do not belong here. The supervisor's isolated reviewer judges
    those questions from the complete candidate.
    """
    key = _framework_key(framework)
    if key not in _SUPPORTED_PRODUCTION_FRAMEWORKS:
        return [f"unsupported production framework: {framework}"]
    kernel_path = workspace / "kernel.py"
    if not kernel_path.is_file():
        return ["kernel.py is missing"]
    source = kernel_path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source, filename=str(kernel_path))
    except SyntaxError as exc:
        return [f"kernel.py is not valid Python: {exc.msg} (line {exc.lineno})"]

    errors: list[str] = []
    if _has_relative_import(tree):
        errors.append("relative/local-module imports are not self-contained")
    if require_gluon and not source_uses_gluon(source):
        errors.append("switching back from the accepted Gluon phase to Triton is forbidden")
    errors.extend(_solution_structure_violations(workspace))
    return list(dict.fromkeys(errors))


def production_kernel_violations(
    workspace: Path,
    framework: str,
    *,
    require_gluon: bool = False,
    production_reviewer: ProductionReviewer | None = None,
) -> list[str]:
    """Return production-policy violations for the current candidate.

    Mechanically certain structure stays local. Every otherwise viable candidate is
    delegated in full through ``production_reviewer`` and fails closed when no reviewer
    is supplied. Runtime correctness and performance still use the normal sandbox.
    """
    errors = production_structure_violations(
        workspace,
        framework,
        require_gluon=require_gluon,
    )
    if errors:
        return errors
    if production_reviewer is None:
        return ["production candidate requires supervisor policy review"]
    try:
        review_errors = production_reviewer(workspace, framework, require_gluon)
    except Exception as exc:
        errors.append(
            "independent production policy review failed: "
            f"{type(exc).__name__}: {exc}"
        )
    else:
        if not isinstance(review_errors, list) or not all(
            isinstance(item, str) and item.strip() for item in review_errors
        ):
            errors.append("independent production policy review returned an invalid result")
        else:
            errors.extend(review_errors)
    return list(dict.fromkeys(errors))


def reject_production_commit(
    workspace: Path,
    version: int,
    pre_head: str,
    violations: list[str],
) -> Path:
    """Revert a violating kernel commit and preserve an actionable local record."""
    memory_path = workspace / "memory" / f"v{version}.json"
    try:
        memory = json.loads(memory_path.read_text(encoding="utf-8")) if memory_path.exists() else {}
    except (OSError, json.JSONDecodeError):
        memory = {}
    if pre_head:
        subprocess.run(
            ["git", "reset", "--hard", pre_head],
            cwd=str(workspace),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory["version"] = f"v{version}"
    memory["masked"] = False
    memory.pop("git_commit_hash", None)
    memory["quality_gate"] = {
        "result": "FAIL",
        "failure_reason": "production policy violation: " + "; ".join(violations),
    }
    memory["optimization"] = {
        "action_category": "production_policy_rejection",
        "action_description": "reverted candidate that used a forbidden dependency or wrong framework",
    }
    pitfalls = memory.setdefault("pitfalls_and_fixes", [])
    pitfalls.append({
        "error_type": "production_policy",
        "error_message": "; ".join(violations),
        "lesson": "implement the candidate directly and exclusively in the selected framework",
    })
    memory_path.write_text(json.dumps(memory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return memory_path
