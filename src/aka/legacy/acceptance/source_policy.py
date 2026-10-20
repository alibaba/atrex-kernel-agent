"""Acceptance source_policy implementation and callable interface."""
from __future__ import annotations

from pathlib import Path
from typing import Callable
import ast
import json
from aka.legacy.task.source.inspection import source_uses_gluon
from aka.legacy.task.source.inspection import framework_key
import subprocess


ProductionReviewer = Callable[[Path, str, bool], list[str]]


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
    key = framework_key(framework)
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
