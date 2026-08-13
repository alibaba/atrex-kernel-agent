"""Detection helpers for supported operator directory layouts."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Optional, cast


AGENT_PROBLEM_FILENAME = "agent_problem.json"
AGENT_PROBLEM_SCHEMA_VERSION = "atrex.agent_problem.v1"
GENERATED_AGENT_PROBLEM_FIELDS = frozenset(
    {
        "schema_version",
        "objective",
        "evaluation",
        "operator_contract",
        "workload_profile",
        "distribution_profile",
        "shape_domain",
        "invariants",
        "coverage_regimes",
        "development_cases",
    }
)
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


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def validate_private_shapes(path: Path) -> dict[str, dict[str, Any]]:
    """Validate evaluator-owned detailed cases used to derive a public problem."""
    payload = _load_json_object(path, label="shapes.json")
    if not payload:
        raise ValueError(f"{path} must contain at least one evaluator shape")
    invalid = [
        str(shape_id)
        for shape_id, entry in payload.items()
        if not isinstance(entry, dict)
        or not isinstance(entry.get("init_kwargs"), (dict, type(None)))
        or not isinstance(entry.get("input_kwargs"), dict)
    ]
    if invalid:
        raise ValueError(
            f"{path} shape entries must contain init_kwargs/input_kwargs objects: "
            + ", ".join(invalid[:8])
        )
    return cast(dict[str, dict[str, Any]], payload)


def _case_signature(value: dict[str, Any]) -> str:
    return json.dumps(
        {
            "init_kwargs": value.get("init_kwargs"),
            "input_kwargs": value.get("input_kwargs"),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def validate_agent_problem(
    path: Path, *, private_shapes_path: Path | None = None
) -> dict:
    """Validate and return one public generalized Atrex-Bench problem contract."""
    payload = _load_json_object(path, label=AGENT_PROBLEM_FILENAME)
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
    if private_shapes_path is not None:
        private_shapes = validate_private_shapes(private_shapes_path)
        hidden = {_case_signature(value) for value in private_shapes.values()}
        duplicates = [
            str(value.get("name", index))
            for index, value in enumerate(development_cases)
            if _case_signature(value) in hidden
        ]
        if duplicates:
            raise ValueError(
                f"{path}.development_cases must be synthetic and must not duplicate "
                "private evaluator cases: " + ", ".join(duplicates[:8])
            )
    return payload


def _walk_json(value: object) -> Iterator[object]:
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def validate_generated_agent_problem(path: Path, *, private_shapes_path: Path) -> dict:
    """Apply stricter completeness and no-verbatim-case checks to AKA-authored contracts."""
    payload = validate_agent_problem(path, private_shapes_path=private_shapes_path)
    unknown_fields = sorted(set(payload) - GENERATED_AGENT_PROBLEM_FIELDS)
    if unknown_fields:
        raise ValueError(
            f"{path} has unsupported fields in an AKA-generated problem: "
            + ", ".join(unknown_fields)
        )
    for field in ("operator_contract", "shape_domain"):
        if not payload[field]:
            raise ValueError(
                f"{path}.{field} must not be empty in an AKA-generated problem"
            )
    for field in ("invariants", "coverage_regimes"):
        if not payload[field]:
            raise ValueError(
                f"{path}.{field} must not be empty in an AKA-generated problem"
            )

    private_shapes = validate_private_shapes(private_shapes_path)
    hidden_entries = {_canonical_json(value) for value in private_shapes.values()}
    for node in _walk_json(payload):
        if not isinstance(node, dict):
            continue
        serialized = _canonical_json(node)
        if serialized in hidden_entries:
            raise ValueError(
                f"{path} contains a verbatim private evaluator case; generated problems "
                "must expose only a generalized domain"
            )
    return payload


def has_agent_problem(op_dir: Path) -> bool:
    return (op_dir / AGENT_PROBLEM_FILENAME).is_file()


def should_use_generalized_problem(op_dir: Path, optimization_mode: str) -> bool:
    """Return whether an operator must expose a generalized public contract.

    Private exact-case handling is a production policy, not an operator-layout side effect.
    Production uses a supplied contract when present and otherwise generates one from
    ``shapes.json`` before setup. Leaderboard always keeps its detailed shapes public.
    """
    return (
        optimization_mode == "production"
        and not is_sol_op(op_dir)
        and (op_dir / "shapes.json").is_file()
        and find_atrex_bench_root(op_dir) is not None
    )


def agent_visible_operator_files(op_dir: Path, *, generalized: bool) -> tuple[str, ...]:
    """Return the allowlisted operator files copied into an optimization workspace."""
    if generalized:
        # A production campaign may intentionally arrive here without a source problem; it
        # generates one in its workspace before any optimization session starts.
        if has_agent_problem(op_dir):
            validate_agent_problem(
                op_dir / AGENT_PROBLEM_FILENAME,
                private_shapes_path=op_dir / "shapes.json",
            )
        return GENERALIZED_AGENT_VISIBLE_FILES
    return LEGACY_ATREX_VISIBLE_FILES
