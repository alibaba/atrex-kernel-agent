"""Shared evaluator shapes; routing, retries and public projections stay with callers."""
from __future__ import annotations
from typing import Any


def request_shape_ids(request: dict[str, Any]) -> list[str]:
    shapes = request["reference"]["shapes"]

    def sort_key(shape_id: str) -> tuple[int, object]:
        return (0, int(shape_id)) if shape_id.isdigit() else (1, shape_id)

    available = {str(shape_id) for shape_id in shapes}
    selected = (request.get("options") or {}).get("shape_ids")
    if selected is None:
        return sorted(available, key=sort_key)
    if (
        not isinstance(selected, list)
        or not selected
        or any(not isinstance(shape_id, str) or not shape_id for shape_id in selected)
        or len(selected) != len(set(selected))
    ):
        raise ValueError("Eval options.shape_ids must be a non-empty list of distinct strings")
    try:
        for shape_id in selected:
            int(shape_id)
    except ValueError as exc:
        raise ValueError(
            "Agate 0.14.33 requires options.shape_ids to be integer-parseable strings"
        ) from exc
    unknown = [shape_id for shape_id in selected if shape_id not in available]
    if unknown:
        raise ValueError(
            "Eval options.shape_ids contains unknown Shape ids: " + ", ".join(unknown)
        )
    return list(selected)


def batch_shape_ids(shape_ids: list[str], batch_size: int) -> list[list[str]]:
    return [
        shape_ids[offset : offset + batch_size]
        for offset in range(0, len(shape_ids), batch_size)
    ]


def shape_batch_request(
    request: dict[str, Any], shape_ids: list[str]
) -> dict[str, Any]:
    batched = dict(request)
    reference = dict(request["reference"])
    reference["shapes"] = {
        shape_id: request["reference"]["shapes"][shape_id] for shape_id in shape_ids
    }
    for field in ("metadata", "roofline"):
        payload = reference.get(field)
        if not isinstance(payload, dict):
            continue
        payload = dict(payload)
        if isinstance(payload.get("shapes"), dict):
            payload["shapes"] = {
                shape_id: payload["shapes"][shape_id]
                for shape_id in shape_ids
                if shape_id in payload["shapes"]
            }
        if field == "metadata" and "num_shapes" in payload:
            payload["num_shapes"] = len(shape_ids)
        reference[field] = payload
    batched["reference"] = reference
    return batched


def shape_subset_request(
    request: dict[str, Any], shape_ids: list[str]
) -> dict[str, Any]:
    """Select original-bundle Shape IDs without rewriting evaluator inputs.

    Agate 0.14.33+ validates ``options.shape_ids`` against the complete reference
    bundle and lets Atrex-Bench/Ray Data shard that selection inside one job.
    Keeping the original reference, metadata and roofline documents intact also
    makes a subset request auditable against the exact full problem contract.
    """
    batched = dict(request)
    available = request_shape_ids({**request, "options": {
        **(request.get("options") or {}), "shape_ids": None,
    }})
    if not shape_ids or len(shape_ids) != len(set(shape_ids)):
        raise ValueError("Shape selection must be non-empty and duplicate-free")
    unknown = [shape_id for shape_id in shape_ids if shape_id not in available]
    if unknown:
        raise ValueError("Unknown Shape ids: " + ", ".join(unknown))
    options = dict(request.get("options") or {})
    if shape_ids == available:
        options.pop("shape_ids", None)
    else:
        options["shape_ids"] = list(shape_ids)
    batched["options"] = options
    return batched
