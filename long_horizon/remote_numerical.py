#!/usr/bin/env python3
"""Supervisor-owned distribution probes inside the existing GPU evaluator allocation.

The original generator supplies tensor metadata and unchanged structural inputs.
Only the requested tensor leaves are regenerated; tuple/list/dict ABIs are preserved.
Supplemental floating outputs use relative L2; structural and mutation checks
remain owned by the evaluator.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
import subprocess
import sys
import time

PREFIX = "__ATREX_NUMERICAL_RESULT__="
MAX_REL_L2 = 1e-3
GENERATORS = {"uniform", "log_uniform", "sparse", "alternating", "constant", "ramp", "packed_bytes", "near_constant"}


def validate_suite(suite):
    if not isinstance(suite, dict) or suite.get("schema_version") != 1:
        raise ValueError("unsupported numerical suite schema")
    if set(suite) - {"schema_version", "world_size", "seeds", "coverage", "cases"}:
        raise ValueError("numerical suite cannot override evaluator policy")
    if type(suite.get("world_size")) is not int or not 1 <= suite["world_size"] <= 64:
        raise ValueError("numerical suite requires a positive world_size")
    seeds = suite.get("seeds", [])
    if (not isinstance(seeds, list) or len(seeds) != 2
            or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds) or len(set(seeds)) != 2):
        raise ValueError("numerical suite requires two distinct uint32 seeds")
    cases = suite.get("cases", [])
    if (not isinstance(cases, list) or not 1 <= len(cases) <= 3
            or any(not isinstance(c, dict) or not isinstance(c.get("id"), str)
                   or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", c["id"]) for c in cases)
            or len({c["id"] for c in cases}) != len(cases)):
        raise ValueError("supplemental validation requires one to three distinct cases")
    for case in cases:
        if (not isinstance(case.get("purpose"), str) or not 1 <= len(case["purpose"]) <= 2000
                or not isinstance(case.get("fields"), dict) or not 1 <= len(case["fields"]) <= 32):
            raise ValueError("every numerical case needs a purpose and explicit input rules")
        for name, rule in case["fields"].items():
            if not isinstance(name, str) or not 1 <= len(name) <= 128 or not isinstance(rule, dict):
                raise ValueError("numerical fields require bounded ABI paths and generator objects")
            kind = rule.get("generator")
            if kind not in GENERATORS:
                raise ValueError(f"unsupported numerical generator: {kind}")
            required = {"constant": {"value"}, "uniform": {"low", "high"},
                        "log_uniform": {"min_exp", "max_exp"}, "sparse": {"low", "high", "density"},
                        "alternating": {"amplitude"}, "ramp": {"low", "high"},
                        "packed_bytes": set(), "near_constant": {"center", "amplitude"}}[kind]
            optional = {"log_uniform": {"signed"}, "alternating": {"opposite_ranks"}, "ramp": {"axis"}}.get(kind, set())
            if not required <= rule.keys() or set(rule) - required - optional - {"generator"}:
                raise ValueError(f"invalid arguments for numerical generator {kind}")
            for key in required:
                if type(rule[key]) not in {int, float} or not math.isfinite(rule[key]):
                    raise ValueError("generator parameters must be finite numbers")
            for key in optional & rule.keys():
                if type(rule[key]) is not (int if key == "axis" else bool):
                    raise ValueError("invalid optional generator parameter type")
            if ("low" in rule and rule["low"] > rule["high"]
                    or "min_exp" in rule and rule["min_exp"] > rule["max_exp"]
                    or "amplitude" in rule and rule["amplitude"] < 0
                    or "density" in rule and not 0 <= rule["density"] <= 1):
                raise ValueError("invalid numerical generator range")
        if not isinstance(case.get("input_constraints", {}), dict):
            raise ValueError("input_constraints must be an object")
        for bounds in case.get("input_constraints", {}).values():
            if not isinstance(bounds, dict) or not bounds or set(bounds) - {"min", "max", "eq"}:
                raise ValueError("input constraints require min/max/eq scalar bounds")
            if any(type(value) not in {int, float, bool} or not math.isfinite(value) for value in bounds.values()):
                raise ValueError("input constraint bounds must be numeric")
    if suite.get("coverage", "compact") != "compact":
        raise ValueError("supplemental validation uses bounded compact coverage")
    if suite.get("evaluator_command") or suite.get("evaluator_files"):
        raise ValueError("supplemental probes must use the immutable evaluator")
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
    schedule = []
    for index, case in enumerate(suite["cases"]):
        eligible = [sid for sid in ordered if matches_constraints(
            shapes[sid].get("input_kwargs", {}), case.get("input_constraints", {}))]
        selected = case.get("shape_ids")
        if not eligible and selected is None:
            schedule.append({"case_id": case["id"], "shape_ids": [], "seeds": [],
                             "status": "unsupported",
                             "diagnosis": "no available workload meets this advisory's input constraints"})
            continue
        if selected is None:
            selected = [eligible[-1]]
            if len(eligible) > 1:
                offset = int(hashlib.sha256(f"{rotation}:{case['id']}".encode()).hexdigest()[:8], 16)
                selected.insert(0, eligible[offset % (len(eligible) - 1)])
                if mode == "thorough" and len(eligible) > 2:
                    # Cover both size extremes plus a rotating intermediate shape.
                    selected = list(dict.fromkeys([eligible[0], eligible[-1], eligible[1 + offset % (len(eligible) - 2)]]))
        if not selected or len(set(selected)) != len(selected) or set(selected) - set(eligible):
            raise ValueError(f"invalid regression shape selection for {case['id']}")
        seeds = suite["seeds"] if mode == "thorough" or index == 0 else [suite["seeds"][index % len(suite["seeds"])]]
        # Light repeats the first risk; thorough repeats every risk with two seeds.
        seeds = seeds[:2]
        identity = json.dumps({"case_id": case["id"], "shape_ids": selected, "seeds": seeds}, sort_keys=True)
        schedule.append({"case_id": case["id"], "shape_ids": selected, "seeds": seeds,
                         "selection_digest": hashlib.sha256(identity.encode()).hexdigest()})
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
        return {key: replace_leaves(child, replacements, f"{prefix}.{key}" if prefix else str(key))
                for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(replace_leaves(child, replacements, f"{prefix}.{i}")
                           for i, child in enumerate(value))
    return value


def numerical_inputs(inputs, case, seed, rank):
    """Generate values without inspecting the original sample's values or statistics."""
    import torch
    fields = case["fields"]
    leaves = tensor_leaves(inputs)
    if set(fields) - set(leaves):
        raise ValueError("numerical fields do not match the actual input ABI: "
                         + ", ".join(sorted(set(fields) - set(leaves))))
    result = {}
    for name, rule in fields.items():
        template = leaves[name]
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
            value = rule["value"]
            if not template.is_floating_point() and not template.is_complex():
                lower, upper = (0, 1) if template.dtype == torch.bool else (
                    torch.iinfo(template.dtype).min, torch.iinfo(template.dtype).max
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
    return replace_leaves(inputs, result)


def install_inputs(namespace, case, seeds, receipts):
    original = namespace["_make_inputs"]
    counts = {}
    def make_inputs(**kwargs):
        signature = json.dumps(kwargs, sort_keys=True, separators=(",", ":"))
        index = counts.get(signature, 0)
        counts[signature] = index + 1
        seed = seeds[index % len(seeds)]
        rank = int(os.environ.get("RANK", "0"))
        try:
            inputs = numerical_inputs(original(**kwargs), case, seed, rank)
        except (ValueError, TypeError, KeyError) as exc:
            # Preserve the generator error even when the privacy-preserving
            # evaluator suppresses its own traceback. No input values are logged.
            Path(str(receipts) + ".error").write_text(f"{type(exc).__name__}: {exc}")
            raise
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
            if plan.get("status") == "unsupported":
                rows.append(plan)
                continue
            # Evaluate one workload/seed at a time. Evaluators can stop a shape's
            # seed loop on its first mismatch; that is a counterexample, not a
            # broken probe plan. Bind each result to its own input receipt.
            expected = {(json.dumps(shapes[sid].get("input_kwargs") or {}, sort_keys=True), seed, rank)
                        for sid in plan["shape_ids"] for seed in plan["seeds"] for rank in range(suite["world_size"])}
            seen = set()
            failed_probes = 0
            results = []
            diagnosis = ""
            nonfinite_outputs = set()
            deadline = time.monotonic() + request["per_case_timeout"]
            for shape_id in plan["shape_ids"]:
                for seed in plan["seeds"]:
                    receipts = request_path.parent / "receipts.jsonl"
                    receipts.unlink(missing_ok=True)
                    input_error = Path(str(receipts) + ".error")
                    input_error.unlink(missing_ok=True)
                    tail = ("\nimport runpy as __numeric_runpy\n"
                            f"__numeric_runpy.run_path({str(Path(__file__).resolve())!r})['install_inputs'](globals(), {case!r}, {[seed]!r}, {str(receipts.resolve())!r})\n")
                    input_path.write_bytes(original + tail.encode())
                    for stem in ("input", "test_kernel"):
                        for cached in (root / "__pycache__").glob(f"{stem}.*.pyc"):
                            cached.unlink()
                    command = [*evaluator, "--version", "vlong", "--no-memory", "--correctness-only",
                               "--correctness-max-rel-l2", str(request.get("max_rel_l2", MAX_REL_L2)),
                               "--multi-seed", "0", "--seed", str(seed), "--shape-id", shape_id]
                    try:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise subprocess.TimeoutExpired(command, request["per_case_timeout"])
                        process = subprocess.run(command, cwd=root, capture_output=True, text=True,
                                                 timeout=remaining)
                    except subprocess.TimeoutExpired:
                        diagnosis = "supplemental evaluator exceeded its case timeout"
                        break
                    result = None
                    for line in process.stdout.splitlines():
                        if line.startswith("[test_kernel] RESULT_JSON="):
                            result = json.loads(line.split("RESULT_JSON=", 1)[1])
                    observed = [json.loads(line) for line in receipts.read_text().splitlines()] if receipts.exists() else []
                    actual = {(json.dumps(r["input_kwargs"], sort_keys=True), r["seed"], r["rank"]) for r in observed}
                    required = {(json.dumps(shapes[shape_id].get("input_kwargs") or {}, sort_keys=True), seed, rank)
                                for rank in range(suite["world_size"])}
                    seen.update(actual & expected)
                    if input_error.exists():
                        diagnosis = input_error.read_text()
                    elif actual != required:
                        diagnosis = "evaluator did not generate the requested workload, seed and ranks"
                    elif not isinstance(result, dict):
                        diagnosis = f"evaluator exited {process.returncode} without a result"
                    elif "unknown" in result.get("nonfinite_outputs", []):
                        diagnosis = "non-finite output diagnostic has unknown roles"
                    elif "reference" in result.get("nonfinite_outputs", []):
                        diagnosis = "reference output is non-finite under the requested probe inputs"
                    elif result.get("all_pass") is False:
                        failed_probes += 1
                    elif result.get("all_pass") is not True or process.returncode != 0:
                        diagnosis = f"evaluator returned an inconsistent result (exit={process.returncode})"
                    if isinstance(result, dict):
                        results.append(result)
                        nonfinite_outputs.update(result.get("nonfinite_outputs", []))
                    if diagnosis or failed_probes:
                        break
                if diagnosis or failed_probes:
                    break
            passed = not diagnosis and not failed_probes and seen == expected
            metrics = {}
            for result in results:
                values = {**{name: result[name] for name in ("max_abs_err", "max_rel_err") if name in result},
                          **result.get("numerical_metrics", {})}
                for name, value in values.items():
                    if value is None or not math.isfinite(value):
                        metrics[name] = None
                    elif metrics.get(name, 0) is not None:
                        metrics[name] = max(metrics.get(name, 0), value)
            rows.append({"case_id": case["id"], "passed": passed,
                         "exit_code": 0 if passed else 1,
                         "expected_probes": len(expected), "observed_probes": len(seen),
                         "failed_probes": failed_probes,
                         "selection_digest": plan["selection_digest"], "shape_count": len(plan["shape_ids"]),
                         "seeds": plan["seeds"], "world_size": suite["world_size"],
                         "result": {"all_pass": passed, "numerical_metrics": metrics,
                                    "nonfinite_outputs": sorted(nonfinite_outputs)},
                         "input_error": diagnosis})
            if not passed:
                break
        payload = {"schema_version": 1, "runs": rows, "all_pass": bool(rows) and all(r.get("passed") is True for r in rows)}
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
