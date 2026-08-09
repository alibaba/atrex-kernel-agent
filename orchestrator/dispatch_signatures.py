"""Dispatch signature validation and generalized dispatch plan construction."""
from __future__ import annotations

import math

from .constants import DISPATCH_SIGNATURES_FILE, DISPATCH_VISIBILITY_POLICY
from .workload_buckets import WorkloadBucket, WorkloadSource


def _freeze_dispatch_signature(value: object) -> object:
    """Convert JSON arrays into hashable tuples while preserving scalars."""
    if isinstance(value, list):
        return tuple(_freeze_dispatch_signature(item) for item in value)
    if isinstance(value, dict):
        return tuple(
            (key, _freeze_dispatch_signature(value[key])) for key in sorted(value)
        )
    return value


def _validate_dispatch_value_signature(value: object, path: str) -> None:
    """Validate one no-sync, production-visible value signature.

    Tensor signatures intentionally contain metadata only.  A cached or agent-edited
    signature cannot add tensor values, summaries, pointers, or other evaluator-only
    information and later smuggle it into the generated production dispatcher.
    """
    if not isinstance(value, list) or not value or not isinstance(value[0], str):
        raise ValueError(f"{path} is not a tagged dispatch value signature")
    tag = value[0]
    if tag == "tensor":
        if len(value) != 6:
            raise ValueError(
                f"{path} tensor signature must contain only shape/stride/dtype/layout/requires_grad"
            )
        shape, stride, dtype, layout, requires_grad = value[1:]
        if (
            not isinstance(shape, list)
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in shape)
        ):
            raise ValueError(f"{path} tensor shape is invalid")
        if (
            not isinstance(stride, list)
            or len(stride) != len(shape)
            or any(isinstance(item, bool) or not isinstance(item, int) for item in stride)
        ):
            raise ValueError(f"{path} tensor stride is invalid")
        if not isinstance(dtype, str) or not isinstance(layout, str):
            raise ValueError(f"{path} tensor dtype/layout is invalid")
        if not isinstance(requires_grad, bool):
            raise ValueError(f"{path} tensor requires_grad flag is invalid")
        return
    if tag == "none":
        if len(value) != 1:
            raise ValueError(f"{path} none signature has extra data")
        return
    if tag == "bool":
        if len(value) != 2 or not isinstance(value[1], bool):
            raise ValueError(f"{path} bool signature is invalid")
        return
    if tag == "int":
        if len(value) != 2 or isinstance(value[1], bool) or not isinstance(value[1], int):
            raise ValueError(f"{path} int signature is invalid")
        return
    if tag == "float":
        if len(value) != 2 or not isinstance(value[1], str):
            raise ValueError(f"{path} float signature is invalid")
        rendered = value[1]
        if rendered not in {"nan", "inf", "-inf"}:
            try:
                float.fromhex(rendered)
            except ValueError as exc:
                raise ValueError(f"{path} float signature is invalid") from exc
        return
    if tag in {"str", "torch.dtype"}:
        if len(value) != 2 or not isinstance(value[1], str):
            raise ValueError(f"{path} {tag} signature is invalid")
        return
    if tag in {"tuple", "list"}:
        if len(value) != 2 or not isinstance(value[1], list):
            raise ValueError(f"{path} {tag} signature is invalid")
        for index, item in enumerate(value[1]):
            _validate_dispatch_value_signature(item, f"{path}[{index}]")
        return
    if tag == "dict":
        if len(value) != 2 or not isinstance(value[1], list):
            raise ValueError(f"{path} dict signature is invalid")
        keys: list[str] = []
        for index, item in enumerate(value[1]):
            if (
                not isinstance(item, list)
                or len(item) != 2
                or not isinstance(item[0], str)
            ):
                raise ValueError(f"{path} dict entry {index} is invalid")
            keys.append(item[0])
            _validate_dispatch_value_signature(item[1], f"{path}.{item[0]}")
        if keys != sorted(set(keys)):
            raise ValueError(f"{path} dict keys must be unique and sorted")
        return
    raise ValueError(f"{path} uses unsupported dispatch signature tag {tag!r}")


def _validate_invocation_signature(value: object, path: str) -> None:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or value[0] != "invocation"
        or not isinstance(value[1], list)
        or not isinstance(value[2], list)
    ):
        raise ValueError(f"{path} is not a valid invocation signature")
    for index, item in enumerate(value[1]):
        _validate_dispatch_value_signature(item, f"{path}.args[{index}]")
    names: list[str] = []
    for index, item in enumerate(value[2]):
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], str)
        ):
            raise ValueError(f"{path}.kwargs[{index}] is invalid")
        names.append(item[0])
        _validate_dispatch_value_signature(item[1], f"{path}.kwargs.{item[0]}")
    if names != sorted(set(names)):
        raise ValueError(f"{path} keyword names must be unique and sorted")


def validate_dispatch_signatures(
    payload: object, workload_source: WorkloadSource
) -> list[dict]:
    """Validate the sandbox collector result against immutable workload order."""
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"{DISPATCH_SIGNATURES_FILE} schema_version must be 1")
    if payload.get("kind") != workload_source.kind:
        raise ValueError(
            f"{DISPATCH_SIGNATURES_FILE} kind does not match {workload_source.kind}"
        )
    if payload.get("workload_source") != workload_source.filename:
        raise ValueError(
            f"{DISPATCH_SIGNATURES_FILE} workload source does not match "
            f"{workload_source.filename}"
        )
    visibility_policy = payload.get("visibility_policy")
    if visibility_policy not in (None, DISPATCH_VISIBILITY_POLICY):
        raise ValueError(
            f"{DISPATCH_SIGNATURES_FILE} visibility policy is unsupported: "
            f"{visibility_policy!r}"
        )
    records = payload.get("workloads")
    if not isinstance(records, list) or len(records) != len(workload_source.entries):
        raise ValueError(
            f"{DISPATCH_SIGNATURES_FILE} must contain exactly "
            f"{len(workload_source.entries)} workloads"
        )
    normalized: list[dict] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"dispatch signature record {index} is not an object")
        if record.get("index") != index or record.get("id") != workload_source.ids[index]:
            raise ValueError(
                f"dispatch signature record {index} does not match workload id "
                f"{workload_source.ids[index]!r}"
            )
        if not isinstance(record.get("init"), list) or not isinstance(
            record.get("call"), list
        ):
            raise ValueError(f"dispatch signature record {index} is missing init/call")
        _validate_invocation_signature(record["init"], f"workloads[{index}].init")
        _validate_invocation_signature(record["call"], f"workloads[{index}].call")
        normalized.append(
            {
                "index": index,
                "id": workload_source.ids[index],
                "init": record["init"],
                "call": record["call"],
            }
        )
    return normalized


def _dispatch_schema_and_features(
    init_signature: object,
    call_signature: object,
) -> tuple[object, tuple[int, ...], tuple[str, ...]]:
    """Split a runtime signature into categorical structure and integer features.

    Exact-signature dispatch makes every observed shape an isolated point.  The
    generalized dispatcher instead keeps ABI/type/layout structure categorical
    while treating tensor dimensions, tensor strides, and explicit integer
    arguments as coordinates.  A decision tree can then describe shape regimes
    with interval boundaries and extrapolate to nearby, unseen inputs.
    """

    features: list[int] = []
    descriptors: list[str] = []

    def visit(value: object, path: str) -> object:
        if not isinstance(value, (list, tuple)) or not value:
            return _freeze_dispatch_signature(value)
        tag = value[0]
        if tag == "tensor" and len(value) == 6:
            shape, stride, dtype, layout, requires_grad = value[1:]
            if not isinstance(shape, (list, tuple)) or not isinstance(
                stride, (list, tuple)
            ):
                return _freeze_dispatch_signature(value)
            for index, item in enumerate(shape):
                features.append(int(item))
                descriptors.append(f"{path}.shape[{index}]")
            for index, item in enumerate(stride):
                features.append(int(item))
                descriptors.append(f"{path}.stride[{index}]")
            return (
                "tensor",
                len(shape),
                len(stride),
                str(dtype),
                str(layout),
                bool(requires_grad),
            )
        if tag == "int" and len(value) == 2:
            features.append(int(value[1]))
            descriptors.append(f"{path}.value")
            return ("int",)
        if tag == "invocation" and len(value) == 3:
            args, kwargs = value[1], value[2]
            return (
                "invocation",
                tuple(
                    visit(item, f"{path}.args[{index}]")
                    for index, item in enumerate(args)
                ),
                tuple(
                    (str(item[0]), visit(item[1], f"{path}.kwargs.{item[0]}"))
                    for item in kwargs
                ),
            )
        if tag in {"tuple", "list"} and len(value) == 2:
            return (
                str(tag),
                tuple(
                    visit(item, f"{path}[{index}]")
                    for index, item in enumerate(value[1])
                ),
            )
        if tag == "dict" and len(value) == 2:
            return (
                "dict",
                tuple(
                    (str(item[0]), visit(item[1], f"{path}.{item[0]}"))
                    for item in value[1]
                ),
            )
        return _freeze_dispatch_signature(value)

    schema = (visit(init_signature, "init"), visit(call_signature, "call"))
    return schema, tuple(features), tuple(descriptors)


def _dispatch_tree_bucket(tree: object, features: tuple[int, ...]) -> object:
    """Evaluate one host-side generalized dispatch tree."""
    node = tree
    while isinstance(node, tuple) and node and node[0] == "split":
        _, feature_index, boundary_sum, left, right = node
        node = left if 2 * features[int(feature_index)] <= int(boundary_sum) else right
    if not isinstance(node, tuple) or len(node) != 2 or node[0] != "leaf":
        raise ValueError("invalid generalized dispatch tree")
    return node[1]


def _build_dispatch_tree(
    samples: list[tuple[tuple[int, ...], object]],
    descriptors: tuple[str, ...],
) -> object:
    """Build a deterministic axis-aligned tree that exactly fits observed owners."""
    owners_by_features: dict[tuple[int, ...], object] = {}
    for features, owner in samples:
        previous = owners_by_features.setdefault(features, owner)
        if previous != owner:
            raise ValueError(
                "workloads with the same generalized runtime features cannot be split "
                f"across buckets: {previous!r} and {owner!r}"
            )
    deduplicated = list(owners_by_features.items())

    def impurity(rows: list[tuple[tuple[int, ...], object]]) -> int:
        counts: dict[object, int] = {}
        for _features, owner in rows:
            counts[owner] = counts.get(owner, 0) + 1
        return len(rows) - max(counts.values())

    def build(rows: list[tuple[tuple[int, ...], object]]) -> object:
        owners = {owner for _features, owner in rows}
        if len(owners) == 1:
            return ("leaf", next(iter(owners)))
        if not rows or not rows[0][0]:
            raise ValueError("bucket partition has no numeric runtime feature to generalize")

        candidates: list[tuple[tuple[int, int, int, int], int, int, list, list]] = []
        for feature_index in range(len(rows[0][0])):
            values = sorted({features[feature_index] for features, _owner in rows})
            for lower, upper in zip(values, values[1:]):
                boundary_sum = lower + upper
                left = [
                    row for row in rows if 2 * row[0][feature_index] <= boundary_sum
                ]
                right = [
                    row for row in rows if 2 * row[0][feature_index] > boundary_sum
                ]
                if not left or not right:
                    continue
                descriptor = descriptors[feature_index]
                feature_priority = (
                    0 if ".shape[" in descriptor else 1 if descriptor.endswith(".value") else 2
                )
                score = (
                    impurity(left) + impurity(right),
                    abs(len(left) - len(right)),
                    feature_priority,
                    feature_index,
                )
                candidates.append((score, feature_index, boundary_sum, left, right))
        if not candidates:
            raise ValueError("bucket partition cannot be represented by generalized ranges")
        _score, feature_index, boundary_sum, left, right = min(
            candidates, key=lambda item: (item[0], item[2])
        )
        return (
            "split",
            feature_index,
            boundary_sum,
            build(left),
            build(right),
        )

    return build(deduplicated)


def _primitive_dispatch_directions(
    feature_rows: list[tuple[int, ...]],
) -> list[tuple[int, ...]]:
    """Infer primitive lattice steps while preserving observed shape relationships."""
    directions: set[tuple[int, ...]] = set()
    for position, left in enumerate(feature_rows):
        for right in feature_rows[position + 1 :]:
            delta = tuple(b - a for a, b in zip(left, right))
            divisor = 0
            for item in delta:
                divisor = math.gcd(divisor, abs(item))
            if divisor == 0:
                continue
            primitive = tuple(item // divisor for item in delta)
            first = next((item for item in primitive if item), 1)
            if first < 0:
                primitive = tuple(-item for item in primitive)
            directions.add(primitive)
    return sorted(directions, key=lambda value: (sum(abs(item) for item in value), value))


def build_generalized_dispatch_plan(
    signature_records: list[dict],
    bucket_by_index: dict[int, object],
) -> tuple[dict[object, object], dict[object, dict]]:
    """Fit range-based dispatch and certify one unseen lattice step per bucket.

    The returned evidence is also written into each bucket workspace.  It gives
    the optimizer a concrete neighboring input that its implementation must
    support instead of permitting a table of benchmark-only shape special cases.
    """
    grouped: dict[object, list[tuple[int, tuple[int, ...], object]]] = {}
    descriptors_by_schema: dict[object, tuple[str, ...]] = {}
    for record in signature_records:
        index = int(record["index"])
        schema, features, descriptors = _dispatch_schema_and_features(
            record["init"], record["call"]
        )
        previous = descriptors_by_schema.setdefault(schema, descriptors)
        if previous != descriptors:
            raise ValueError("inconsistent generalized dispatch feature schema")
        grouped.setdefault(schema, []).append(
            (index, features, bucket_by_index[index])
        )

    plan: dict[object, object] = {}
    evidence: dict[object, dict] = {}
    all_owners = set(bucket_by_index.values())
    for schema, rows in grouped.items():
        descriptors = descriptors_by_schema[schema]
        tree = _build_dispatch_tree(
            [(features, owner) for _index, features, owner in rows], descriptors
        )
        for index, features, owner in rows:
            if _dispatch_tree_bucket(tree, features) != owner:
                raise AssertionError(f"generalized dispatcher misclassified workload {index}")
        plan[schema] = tree

        observed = {features for _index, features, _owner in rows}
        directions = _primitive_dispatch_directions([features for _i, features, _o in rows])
        if not directions and descriptors:
            preferred = [
                index for index, name in enumerate(descriptors) if ".shape[" in name
            ] or list(range(len(descriptors)))
            directions = [
                tuple(1 if index == feature_index else 0 for index in range(len(descriptors)))
                for feature_index in preferred
            ]

        for workload_index, features, owner in rows:
            if owner in evidence:
                continue
            for direction in directions:
                for sign in (1, -1):
                    candidate = tuple(
                        value + sign * step for value, step in zip(features, direction)
                    )
                    if candidate in observed:
                        continue
                    if any(
                        value < 0
                        for value, descriptor in zip(candidate, descriptors)
                        if ".shape[" in descriptor or ".stride[" in descriptor
                    ):
                        continue
                    if _dispatch_tree_bucket(tree, candidate) != owner:
                        continue
                    changed = {
                        descriptor: {
                            "from": features[position],
                            "to": candidate[position],
                            "step": candidate[position] - features[position],
                        }
                        for position, descriptor in enumerate(descriptors)
                        if candidate[position] != features[position]
                    }
                    evidence[owner] = {
                        "anchor_workload_index": workload_index,
                        "unseen_one_step_neighbor": changed,
                    }
                    break
                if owner in evidence:
                    break

    missing = sorted(str(owner) for owner in all_owners - set(evidence))
    if missing:
        raise ValueError(
            "buckets have no unseen one-step generalized input region: "
            + ", ".join(missing)
            + "; merge singleton/exact-shape islands into a neighboring regime"
        )
    return plan, evidence


def validate_dispatch_bucket_compatibility(
    buckets: list[WorkloadBucket], signature_records: list[dict]
) -> None:
    """Reject partitions that are ambiguous or only recognize exact shapes."""
    owner = {
        index: bucket.name
        for bucket in buckets
        for index in bucket.workload_indices
    }
    signature_owners: dict[object, tuple[str, list[int]]] = {}
    for record in signature_records:
        index = int(record["index"])
        key = (
            _freeze_dispatch_signature(record["init"]),
            _freeze_dispatch_signature(record["call"]),
        )
        bucket_name = owner[index]
        previous = signature_owners.get(key)
        if previous is None:
            signature_owners[key] = (bucket_name, [index])
            continue
        previous_bucket, indices = previous
        if previous_bucket != bucket_name:
            raise ValueError(
                "workloads with identical runtime signatures cannot be split across "
                f"buckets: indices {indices + [index]} map to "
                f"{previous_bucket!r} and {bucket_name!r}"
            )
        indices.append(index)
    build_generalized_dispatch_plan(signature_records, owner)
