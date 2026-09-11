#!/usr/bin/env python3
"""Supervisor-owned distribution probes inside the existing GPU evaluator allocation.

The original generator supplies only tensor metadata. Every input must be explicitly
regenerated or justified as invariant by an operator-owned numerical_suite.json.
Correctness, tolerances and input-mutation checks remain owned by the evaluator.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

PREFIX = "__ATREX_NUMERICAL_RESULT__="
GENERATORS = {"uniform", "log_uniform", "sparse", "alternating", "constant", "ramp", "packed_bytes", "near_constant"}


def validate_suite(suite):
    if suite.get("schema_version") != 1:
        raise ValueError("unsupported numerical suite schema")
    if type(suite.get("world_size")) is not int or suite["world_size"] < 1:
        raise ValueError("numerical suite requires a positive world_size")
    seeds = suite.get("seeds", [])
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError("numerical suite requires at least two distinct integer seeds")
    cases = suite.get("cases", [])
    if len(cases) < 3 or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("numerical suite requires at least three distinct distribution cases")
    families = set()
    for case in cases:
        if not case.get("purpose") or not case.get("fields"):
            raise ValueError("every numerical case needs a purpose and explicit input rules")
        for rule in case["fields"].values():
            kind = rule.get("generator")
            if kind not in GENERATORS:
                raise ValueError(f"unsupported numerical generator: {kind}")
            families.add(kind)
        if any(not isinstance(reason, str) or not reason.strip()
               for reason in case.get("preserve", {}).values()):
            raise ValueError("preserved inputs require an operator-contract justification")
    if len(families - {"constant", "packed_bytes", "near_constant"}) < 2:
        raise ValueError("numerical suite needs independent construction families, not just seeds")
    if suite.get("coverage", "compact") not in {"compact", "exhaustive"}:
        raise ValueError("numerical coverage must be compact or exhaustive")
    return suite


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
    exhaustive = suite.get("coverage", "compact") == "exhaustive"
    schedule = []
    for index, case in enumerate(suite["cases"]):
        selected = case.get("shape_ids")
        if selected is None:
            selected = list(ordered) if exhaustive else [ordered[-1]]
            if not exhaustive and len(ordered) > 1:
                offset = int(hashlib.sha256(f"{rotation}:{case['id']}".encode()).hexdigest()[:8], 16)
                selected.insert(0, ordered[offset % (len(ordered) - 1)])
                if mode == "thorough" and len(ordered) > 2:
                    # Cover both size extremes plus a rotating intermediate shape.
                    selected = list(dict.fromkeys([ordered[0], ordered[-1], ordered[1 + offset % (len(ordered) - 2)]]))
        if not selected or len(set(selected)) != len(selected) or set(selected) - set(shapes):
            raise ValueError(f"invalid regression shape selection for {case['id']}")
        seeds = suite["seeds"] if exhaustive or mode == "thorough" or index == 0 else [suite["seeds"][index % len(suite["seeds"])]]
        # Light repeats the first risk; thorough repeats every risk with two seeds.
        seeds = seeds if exhaustive else seeds[:2]
        identity = json.dumps({"case_id": case["id"], "shape_ids": selected, "seeds": seeds}, sort_keys=True)
        schedule.append({"case_id": case["id"], "shape_ids": selected, "seeds": seeds,
                         "selection_digest": hashlib.sha256(identity.encode()).hexdigest()})
    return schedule


def numerical_inputs(inputs, case, seed, rank):
    """Generate values without inspecting the original sample's values or statistics."""
    import torch
    fields, preserve = case["fields"], case.get("preserve", {})
    if set(fields) & set(preserve) or set(inputs) != set(fields) | set(preserve):
        raise ValueError("numerical suite must classify every input exactly once")
    result = dict(inputs)
    for name, rule in fields.items():
        template = inputs[name]
        if not isinstance(template, torch.Tensor):
            raise TypeError(f"numerical field {name} must be a tensor")
        kind = rule["generator"]
        field_seed = int.from_bytes(hashlib.sha256(f"{seed}:{rank}:{name}".encode()).digest()[:4], "big")
        rng = torch.Generator(device=template.device).manual_seed(field_seed)
        shape = template.shape
        # Generate large weight tensors directly as bytes; float temporaries for them
        # would otherwise dwarf the operator's live input allocation.
        if kind == "packed_bytes":
            if template.dtype != torch.uint8:
                raise ValueError("packed_bytes requires an explicitly byte-packed input")
            values = torch.randint(0, 256, shape, generator=rng, device=template.device, dtype=torch.uint8)
        elif kind == "constant":
            values = torch.full(shape, rule["value"], device=template.device, dtype=torch.float32)
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
                for size in shape[axis + 1:]:
                    stride *= size
                position = index.div(stride, rounding_mode="floor").remainder(width)
                values = rule["low"] + position.float() / max(width - 1, 1) * (rule["high"] - rule["low"])
        else:
            unit = torch.rand(shape, generator=rng, device=template.device, dtype=torch.float32)
            if kind == "near_constant":
                values = rule["center"] + (unit * 2 - 1) * rule["amplitude"]
            elif kind == "log_uniform":
                magnitude = torch.pow(2.0, rule["min_exp"] + unit * (rule["max_exp"] - rule["min_exp"]))
                signs = torch.randint(0, 2, shape, generator=rng, device=template.device) * 2 - 1
                values = magnitude * signs if rule.get("signed", True) else magnitude
            else:
                values = rule["low"] + unit * (rule["high"] - rule["low"])
                if kind == "sparse":
                    mask = torch.rand(shape, generator=rng, device=template.device) < rule["density"]
                    values = values * mask
        result[name] = torch.empty_strided(shape, template.stride(), dtype=template.dtype, device=template.device)
        result[name].copy_(values.reshape(shape))
        if template.is_floating_point() and not torch.isfinite(result[name].float()).all().item():
            raise ValueError(f"numerical suite generated non-finite input: {name}")
    return result


def install_inputs(namespace, case, seeds, receipts):
    original = namespace["_make_inputs"]
    counts = {}
    def make_inputs(**kwargs):
        signature = json.dumps(kwargs, sort_keys=True, separators=(",", ":"))
        index = counts.get(signature, 0)
        counts[signature] = index + 1
        seed = seeds[index % len(seeds)]
        rank = int(os.environ.get("RANK", "0"))
        inputs = numerical_inputs(original(**kwargs), case, seed, rank)
        row = json.dumps({"input_kwargs": kwargs, "seed": seed, "rank": rank}, separators=(",", ":")) + "\n"
        fd = os.open(receipts, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        try:
            os.write(fd, row.encode())
        finally:
            os.close(fd)
        return inputs
    namespace["_make_inputs"] = make_inputs


def run(request_path):
    request = json.loads(request_path.read_text())
    suite = validate_suite(request["suite"])
    root = Path.cwd()
    input_path = root / "input.py"
    harness = root / "test_kernel.py"
    original = input_path.read_bytes()
    original_harness = harness.read_bytes()
    rows = []
    try:
        shapes = json.loads((root / "shapes.json").read_text())
        schedule = validation_schedule(suite, shapes, request.get("rotation", ""), request.get("mode", "light"))
        # Native Atrex workspaces use the supervisor's current transport adapter.
        # This replacement exists only in this isolated allocation, not the workspace.
        snapshot = request_path.parent / "snapshots" / "evaluator.py"
        if snapshot.is_file():
            harness.write_bytes(snapshot.read_bytes())
        evaluator = suite.get("evaluator_command", [sys.executable, "test_kernel.py"])
        for plan in schedule:
            if plan["case_id"] not in request.get("case_ids", [c["id"] for c in suite["cases"]]):
                continue
            case = next(c for c in suite["cases"] if c["id"] == plan["case_id"])
            receipts = request_path.parent / "receipts.jsonl"
            receipts.unlink(missing_ok=True)
            tail = ("\nimport runpy as __numeric_runpy\n"
                    f"__numeric_runpy.run_path({str(Path(__file__).resolve())!r})['install_inputs'](globals(), {case!r}, {plan['seeds']!r}, {str(receipts.resolve())!r})\n")
            input_path.write_bytes(original + tail.encode())
            for stem in ("input", "test_kernel"):
                for cached in (root / "__pycache__").glob(f"{stem}.*.pyc"):
                    cached.unlink()
            command = [*evaluator, "--version", "vlong", "--no-memory", "--correctness-only",
                       "--multi-seed", str(len(plan["seeds"]) - 1)]
            for shape_id in plan["shape_ids"]:
                command += ["--shape-id", shape_id]
            process = subprocess.run(command, cwd=root, capture_output=True, text=True,
                                     timeout=request["per_case_timeout"])
            result = None
            for line in process.stdout.splitlines():
                if line.startswith("[test_kernel] RESULT_JSON="):
                    result = json.loads(line.split("RESULT_JSON=", 1)[1])
            observed = [json.loads(line) for line in receipts.read_text().splitlines()] if receipts.exists() else []
            key = lambda kwargs, seed, rank: (json.dumps(kwargs, sort_keys=True), seed, rank)
            seen = {key(r["input_kwargs"], r["seed"], r["rank"]) for r in observed}
            expected = {key(shapes[sid].get("input_kwargs") or {}, seed, rank)
                        for sid in plan["shape_ids"] for seed in plan["seeds"] for rank in range(suite["world_size"])}
            passed = process.returncode == 0 and isinstance(result, dict) and result.get("all_pass") is True and expected <= seen
            rows.append({"case_id": case["id"], "passed": passed, "exit_code": process.returncode,
                         "expected_probes": len(expected), "observed_probes": len(expected & seen),
                         "selection_digest": plan["selection_digest"], "shape_count": len(plan["shape_ids"]),
                         "seeds": plan["seeds"], "world_size": suite["world_size"],
                         "result": result, "numerical_metrics": (result or {}).get("numerical_metrics", {}),
                         "stderr_tail": process.stderr[-1500:] if not passed else ""})
            if not passed:
                break
        payload = {"schema_version": 1, "runs": rows, "all_pass": bool(rows) and all(r["passed"] for r in rows)}
    except Exception as exc:
        payload = {"schema_version": 1, "runs": rows, "all_pass": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        input_path.write_bytes(original)
        harness.write_bytes(original_harness)
        for stem in ("input", "test_kernel"):
            for cached in (root / "__pycache__").glob(f"{stem}.*.pyc"):
                cached.unlink()
    print(PREFIX + json.dumps(payload, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(Path(sys.argv[1])))
