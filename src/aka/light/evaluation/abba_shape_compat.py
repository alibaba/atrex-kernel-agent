"""Temporary full-reference Shape sharding for native Agate ABBA.

Agate ABBA currently requires every Shape in the submitted reference and
rejects ``options.shape_ids`` even when the selector names every Shape in an
outer Light shard.  Keep that service compatibility in this one module: build
an independent reference whose complete workload is exactly the requested
batch, then omit the selector.

When Agate accepts native ABBA Shape selection, delete this module and switch
``native_abba`` back to ``shapes.shape_batch_request``.
"""

from __future__ import annotations

import copy
from typing import Any

from aka.light.evaluation.shapes import shape_batch_request


def _select_shape_mapping(
    value: object,
    shape_ids: list[str],
) -> object:
    """Copy a known per-Shape mapping while retaining original Shape IDs."""
    if not isinstance(value, dict):
        return copy.deepcopy(value)
    return {
        shape_id: copy.deepcopy(value[shape_id])
        for shape_id in shape_ids
        if shape_id in value
    }


def _select_reference(request: dict[str, Any], shape_ids: list[str]) -> dict[str, Any]:
    reference = copy.deepcopy(request["reference"])
    shapes = reference.get("shapes")
    if not isinstance(shapes, dict):
        raise TypeError("ABBA reference shapes must be an object")
    missing = [shape_id for shape_id in shape_ids if shape_id not in shapes]
    if missing:
        raise ValueError(
            "ABBA Shape batch contains IDs outside the reference: "
            + ", ".join(missing)
        )
    reference["shapes"] = {
        shape_id: copy.deepcopy(shapes[shape_id]) for shape_id in shape_ids
    }

    # These are the Atrex-Bench documents with known Shape-indexed sections.
    # Keep all non-Shape metadata unchanged so the shard remains a complete
    # operator problem rather than a selector applied to the parent problem.
    for field in ("metadata", "roofline"):
        document = reference.get(field)
        if not isinstance(document, dict):
            continue
        for mapping_name in ("shapes", "cases"):
            if mapping_name in document:
                document[mapping_name] = _select_shape_mapping(
                    document[mapping_name], shape_ids
                )
        if field == "metadata" and "num_shapes" in document:
            document["num_shapes"] = len(shape_ids)
    return reference


def full_reference_abba_batch_request(
    request: dict[str, Any],
    shape_ids: list[str],
    *,
    timeout_s: int | None = None,
) -> dict[str, Any]:
    """Treat one Light batch as the complete ABBA reference workload."""
    batched = shape_batch_request(request, shape_ids, timeout_s=timeout_s)
    batched["reference"] = _select_reference(request, shape_ids)

    options = dict(batched.get("options") or {})
    options.pop("shape_ids", None)
    # Each outer request is already one Light Shape batch. Ray Data may still
    # execute that complete shard, but it must not split it again.
    options["shape_shards"] = 1
    batched["options"] = options

    overrides = dict(batched.get("runner_overrides") or {})
    overrides.pop("shape_ids", None)
    batched["runner_overrides"] = overrides
    return batched
