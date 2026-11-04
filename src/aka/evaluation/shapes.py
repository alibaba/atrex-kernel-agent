"""Shared evaluator shapes; routing, retries and public projections stay with callers."""
from __future__ import annotations
from typing import Any


def request_shape_ids(request: dict[str, Any]) -> list[str]:
    shapes = request["reference"]["shapes"]

    def sort_key(shape_id: str) -> tuple[int, object]:
        return (0, int(shape_id)) if shape_id.isdigit() else (1, shape_id)

    return sorted((str(shape_id) for shape_id in shapes), key=sort_key)


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
