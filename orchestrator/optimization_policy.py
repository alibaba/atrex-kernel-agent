"""Optimization-mode policy and production-candidate enforcement."""

from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Callable


OPTIMIZATION_MODE_CHOICES = ("leaderboard", "production")
MODE_STATE_FILE = ".orchestrator_mode.json"
POLICY_BEGIN = "<!-- ATREX_OPTIMIZATION_MODE_POLICY_BEGIN -->"
POLICY_END = "<!-- ATREX_OPTIMIZATION_MODE_POLICY_END -->"


ProductionReviewer = Callable[[Path, str, bool], list[str]]


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
    """Self-contained policy block injected into every coding-agent prompt."""
    if mode == "leaderboard":
        return (
            "## Optimization mode: leaderboard\n\n"
            "Follow the workspace `CLAUDE.md` exactly. Its existing framework guidance remains "
            "unchanged: the requested framework is a recommended direction, compatible mixed/alternate "
            "implementations are allowed when evidence supports them, and third-party helper/kernel "
            "libraries may be used.\n"
        )
    if mode != "production":
        raise ValueError(f"unsupported optimization mode: {mode!r}")
    if _framework_key(framework) == "triton":
        framework_rule = (
            "- The initial implementation framework is exactly **Triton**. After the orchestrator "
            "enters its mandatory Triton-to-Gluon conversion phase, a direct implementation in "
            "`triton.experimental.gluon` is allowed and becomes the required framework for later "
            "iterations. Do not switch early, switch back, mix Triton and Gluon compute kernels, "
            "or use any other DSL.\n"
        )
        candidate_framework = "the active Triton/Gluon phase"
    else:
        framework_rule = (
            f"- The implementation framework is exactly **{framework}**. It is a hard constraint, "
            "not a recommendation. Do not switch to another DSL, mix another kernel framework "
            "into the candidate, or replace the implementation with a prebuilt operator.\n"
        )
        candidate_framework = framework
    return (
        "## Optimization mode: production (hard gate)\n\n"
        "This generated section overrides any conflicting permissive framework or third-party-library "
        "guidance elsewhere in `CLAUDE.md`.\n\n"
        f"{framework_rule}"
        "- The V0 PyTorch reference wrapper is the only baseline exception. Every optimized candidate "
        f"committed after V0 must implement the GPU computation directly in **{candidate_framework}**.\n"
        "- The supervisor sends every production candidate to a separate, read-only policy reviewer. "
        "The reviewer judges the complete candidate by actual use, not package names: compiler bindings, "
        "header discovery, ABI/launch plumbing, and ordinary non-compute support utilities may be accepted "
        "when they only build or launch the candidate's self-authored kernel. Prebuilt kernels/operators/math "
        "implementations, alternate DSLs, hidden dispatch, PyTorch compute fallbacks, and external "
        "implementation loading remain forbidden. Ambiguous evidence is rejected.\n"
        "- Keep `solution.json` consistent with the implementation. Before committing, inspect `kernel.py` "
        "and `solution.json` against these rules. The supervisor will reject a candidate that lacks an "
        "evidence-backed production-policy verdict, even if it is faster and correct.\n"
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
    state_path = workspace / MODE_STATE_FILE
    state_changed = False
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid optimization-mode state: {state_path}") from exc
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
        state_path.write_text(
            json.dumps(state, indent=2) + "\n",
            encoding="utf-8",
        )

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

    gitignore = workspace / ".gitignore"
    ignored = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
    entry = f"/{MODE_STATE_FILE}"
    if entry not in ignored.splitlines():
        with gitignore.open("a", encoding="utf-8") as handle:
            if ignored and not ignored.endswith("\n"):
                handle.write("\n")
            handle.write("\n# orchestrator optimization-mode identity (local policy state)\n")
            handle.write(entry + "\n")


_SUPPORTED_PRODUCTION_FRAMEWORKS = frozenset(
    {"triton", "gluon", "cutedsl", "cuda", "flydsl"}
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
    memory["git_commit_hash"] = None
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
