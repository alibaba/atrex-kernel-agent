"""Validate the private shape set supplied by a task contract."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, cast

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
