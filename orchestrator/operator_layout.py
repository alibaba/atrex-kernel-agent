"""Detection helpers for supported operator directory layouts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


AGENT_PROBLEM_FILENAME = "agent_problem.json"
AGENT_PROBLEM_SCHEMA_VERSION = "atrex.agent_problem.v1"
GENERALIZED_AGENT_VISIBLE_FILES = (
    "reference.py",
    "input.py",
    AGENT_PROBLEM_FILENAME,
)
LEGACY_ATREX_VISIBLE_FILES = (
    "reference.py",
    "input.py",
    "shapes.json",
    "roofline.json",
    "metadata.json",
    "valid.py",
)


def is_sol_op(op_dir: Path) -> bool:
    """Return whether *op_dir* is a SOL-ExecBench operator."""
    return (op_dir / "definition.json").is_file() and (
        op_dir / "workload.jsonl"
    ).is_file()


def find_atrex_bench_root(op_dir: Path) -> Optional[Path]:
    """Return the canonical Atrex-Bench checkout owning a native shapes operator."""
    for candidate in (op_dir, *op_dir.parents):
        if (candidate / "scripts" / "run_eval.py").is_file() and (
            candidate / "src" / "atrex_bench"
        ).is_dir():
            return candidate
    return None


def validate_agent_problem(path: Path) -> dict:
    """Validate and return one public generalized Atrex-Bench problem contract."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {AGENT_PROBLEM_FILENAME}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    if payload.get("schema_version") != AGENT_PROBLEM_SCHEMA_VERSION:
        raise ValueError(
            f"{path} schema_version must be {AGENT_PROBLEM_SCHEMA_VERSION!r}, "
            f"got {payload.get('schema_version')!r}"
        )
    objective = payload.get("objective")
    if not isinstance(objective, str) or not objective.strip():
        raise ValueError(f"{path}.objective must be a non-empty string")
    for field in ("evaluation", "operator_contract", "shape_domain"):
        if not isinstance(payload.get(field), dict):
            raise ValueError(f"{path}.{field} must be a JSON object")
    evaluation = payload["evaluation"]
    if evaluation.get("exact_cases") != "private":
        raise ValueError(f"{path}.evaluation.exact_cases must be 'private'")
    if evaluation.get("development_cases_are_evaluation_cases") is not False:
        raise ValueError(
            f"{path}.evaluation.development_cases_are_evaluation_cases must be false"
        )
    for field in ("workload_profile", "distribution_profile"):
        if field in payload and not isinstance(payload[field], dict):
            raise ValueError(f"{path}.{field} must be a JSON object")
    invariants = payload.get("invariants")
    if not isinstance(invariants, list) or not all(
        isinstance(value, str) and value.strip() for value in invariants
    ):
        raise ValueError(f"{path}.invariants must be a list of non-empty strings")
    regimes = payload.get("coverage_regimes")
    if not isinstance(regimes, list) or not all(
        isinstance(value, dict) for value in regimes
    ):
        raise ValueError(f"{path}.coverage_regimes must be a list of objects")
    development_cases = payload.get("development_cases", [])
    if not isinstance(development_cases, list) or not all(
        isinstance(value, dict)
        and isinstance(value.get("init_kwargs"), (dict, type(None)))
        and isinstance(value.get("input_kwargs"), dict)
        for value in development_cases
    ):
        raise ValueError(
            f"{path}.development_cases must contain init_kwargs/input_kwargs objects"
        )
    return payload


def has_agent_problem(op_dir: Path) -> bool:
    return (op_dir / AGENT_PROBLEM_FILENAME).is_file()


def agent_visible_operator_files(op_dir: Path) -> tuple[str, ...]:
    """Return the allowlisted operator files copied into an optimization workspace."""
    if has_agent_problem(op_dir):
        validate_agent_problem(op_dir / AGENT_PROBLEM_FILENAME)
        return GENERALIZED_AGENT_VISIBLE_FILES
    return LEGACY_ATREX_VISIBLE_FILES
