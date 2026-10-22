"""Standalone distribution/scheduling algorithms; no evaluator policy or transport."""

from __future__ import annotations
import hashlib
import json
import math


def validation_schedule(suite, shapes, rotation="", mode="light"):
    """Stress numerical risks on representative shapes; baseline still covers all shapes.

    Include the largest workload, a rotating shape and (thorough) the smallest.
    Explicit regression shape IDs override sampling. Ranks are never sampled.
    """

    def size(value):
        if isinstance(value, dict):
            return sum(size(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return sum(size(v) for v in value)
        return math.log1p(abs(value)) if type(value) in {int, float} else 0.0

    ordered = sorted(shapes, key=lambda sid: (size(shapes[sid].get("input_kwargs", {})), sid))
    if not ordered:
        raise ValueError("numerical validation requires shapes")
    if mode not in {"light", "thorough"}:
        raise ValueError("numerical mode must be light or thorough")
    schedule = []
    for index, case in enumerate(suite["cases"]):
        eligible = [
            sid
            for sid in ordered
            if matches_constraints(
                shapes[sid].get("input_kwargs", {}), case.get("input_constraints", {})
            )
        ]
        selected = case.get("shape_ids")
        if not eligible and selected is None:
            schedule.append(
                {
                    "case_id": case["id"],
                    "shape_ids": [],
                    "seeds": [],
                    "status": "unsupported",
                    "diagnosis": "no available workload meets this advisory's input constraints",
                }
            )
            continue
        if selected is None:
            selected = [eligible[-1]]
            if len(eligible) > 1:
                offset = int(
                    hashlib.sha256(f"{rotation}:{case['id']}".encode()).hexdigest()[:8],
                    16,
                )
                selected.insert(0, eligible[offset % (len(eligible) - 1)])
                if mode == "thorough" and len(eligible) > 2:
                    # Cover both size extremes plus a rotating intermediate shape.
                    selected = list(
                        dict.fromkeys(
                            [
                                eligible[0],
                                eligible[-1],
                                eligible[1 + offset % (len(eligible) - 2)],
                            ]
                        )
                    )
        if not selected or len(set(selected)) != len(selected) or set(selected) - set(eligible):
            raise ValueError(f"invalid regression shape selection for {case['id']}")
        seeds = (
            suite["seeds"]
            if mode == "thorough" or index == 0
            else [suite["seeds"][index % len(suite["seeds"])]]
        )
        # Light repeats the first risk; thorough repeats every risk with two seeds.
        seeds = seeds[:2]
        identity = json.dumps(
            {"case_id": case["id"], "shape_ids": selected, "seeds": seeds},
            sort_keys=True,
        )
        schedule.append(
            {
                "case_id": case["id"],
                "shape_ids": selected,
                "seeds": seeds,
                "selection_digest": hashlib.sha256(identity.encode()).hexdigest(),
            }
        )
    return schedule


def matches_constraints(kwargs, constraints):
    for name, bounds in constraints.items():
        value = kwargs.get(name)
        if type(value) not in {int, float, bool}:
            return False
        if "eq" in bounds and value != bounds["eq"]:
            return False
        if "min" in bounds and value < bounds["min"]:
            return False
        if "max" in bounds and value > bounds["max"]:
            return False
    return True


def tensor_leaves(value, prefix=""):
    """Use actual ABI paths (lhs.0), never infer aliases from dtype metadata."""
    if isinstance(value, dict):
        children = value.items()
    elif isinstance(value, (tuple, list)):
        children = enumerate(value)
    else:
        return {prefix: value}
    result = {}
    for key, child in children:
        result.update(tensor_leaves(child, f"{prefix}.{key}" if prefix else str(key)))
    return result


def replace_leaves(value, replacements, prefix=""):
    if prefix in replacements:
        return replacements[prefix]
    if isinstance(value, dict):
        return {
            key: replace_leaves(child, replacements, f"{prefix}.{key}" if prefix else str(key))
            for key, child in value.items()
        }
    if isinstance(value, (tuple, list)):
        return type(value)(
            replace_leaves(child, replacements, f"{prefix}.{i}") for i, child in enumerate(value)
        )
    return value


def numerical_inputs(inputs, case, seed, rank):
    """Generate values without inspecting the original sample's values or statistics."""
    import torch

    fields = case["fields"]
    leaves = tensor_leaves(inputs)
    if set(fields) - set(leaves):
        raise ValueError(
            "numerical fields do not match the actual input ABI: "
            + ", ".join(sorted(set(fields) - set(leaves)))
        )
    result = {}
    for name, rule in fields.items():
        template = leaves[name]
        if not isinstance(template, torch.Tensor):
            raise TypeError(f"numerical field {name} must be a tensor")
        kind = rule["generator"]
        field_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{rank}:{name}".encode()).digest()[:4], "big"
        )
        rng = torch.Generator(device=template.device).manual_seed(field_seed)
        shape = template.shape
        # Generate large weight tensors directly as bytes; float temporaries for them
        # would otherwise dwarf the operator's live input allocation.
        if kind == "packed_bytes":
            if template.dtype != torch.uint8:
                raise ValueError("packed_bytes requires an explicitly byte-packed input")
            values = torch.randint(
                0, 256, shape, generator=rng, device=template.device, dtype=torch.uint8
            )
        elif kind == "constant":
            value = rule["value"]
            if not template.is_floating_point() and not template.is_complex():
                lower, upper = (
                    (0, 1)
                    if template.dtype == torch.bool
                    else (
                        torch.iinfo(template.dtype).min,
                        torch.iinfo(template.dtype).max,
                    )
                )
                if not lower <= value <= upper or int(value) != value:
                    raise ValueError(
                        f"constant for {name} must be an integer in [{lower}, {upper}] "
                        f"for {template.dtype}"
                    )
            values = torch.full(shape, value, device=template.device, dtype=template.dtype)
        elif kind in {"alternating", "ramp"}:
            index = torch.arange(template.numel(), device=template.device).reshape(shape)
            if kind == "alternating":
                values = (index.remainder(2).float() * 2 - 1) * rule["amplitude"]
                if rule.get("opposite_ranks") and rank % 2:
                    values = -values
            else:
                axis = rule.get("axis", -1) % len(shape) if shape else 0
                width = shape[axis] if shape else 1
                stride = 1
                for size in shape[axis + 1 :]:
                    stride *= size
                position = index.div(stride, rounding_mode="floor").remainder(width)
                values = rule["low"] + position.float() / max(width - 1, 1) * (
                    rule["high"] - rule["low"]
                )
        else:
            unit = torch.rand(shape, generator=rng, device=template.device, dtype=torch.float32)
            if kind == "near_constant":
                values = rule["center"] + (unit * 2 - 1) * rule["amplitude"]
            elif kind == "log_uniform":
                magnitude = torch.pow(
                    2.0, rule["min_exp"] + unit * (rule["max_exp"] - rule["min_exp"])
                )
                signs = torch.randint(0, 2, shape, generator=rng, device=template.device) * 2 - 1
                values = magnitude * signs if rule.get("signed", True) else magnitude
            else:
                values = rule["low"] + unit * (rule["high"] - rule["low"])
                if kind == "sparse":
                    mask = (
                        torch.rand(shape, generator=rng, device=template.device) < rule["density"]
                    )
                    values = values * mask
        result[name] = torch.empty_strided(
            shape, template.stride(), dtype=template.dtype, device=template.device
        )
        result[name].copy_(values.reshape(shape))
        if template.is_floating_point() and not torch.isfinite(result[name].float()).all().item():
            raise ValueError(f"numerical suite generated non-finite input: {name}")
    return replace_leaves(inputs, result)
