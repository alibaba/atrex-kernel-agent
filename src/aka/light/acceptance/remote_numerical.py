#!/usr/bin/env python3
"""Supervisor-owned distribution probes inside the existing GPU evaluator allocation.

The original generator supplies tensor metadata and unchanged structural inputs.
Only the requested tensor leaves are regenerated; tuple/list/dict ABIs are preserved.
Supplemental floating outputs use relative L2; structural and mutation checks
remain owned by the evaluator.
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
import subprocess
import sys
import time
import traceback

# A remote bundle ships the shared module beside this script. Package imports
# use the same owner without requiring an AKA installation on GPU workers.
if not __package__ and Path(__file__).with_name("numerical_inputs.py").is_file():
    from numerical_inputs import (validation_schedule, matches_constraints, tensor_leaves, replace_leaves, numerical_inputs)
else:
    from aka.legacy.evaluation.drivers.numerical_inputs import (validation_schedule, matches_constraints, tensor_leaves, replace_leaves, numerical_inputs)

PREFIX = "__ATREX_NUMERICAL_RESULT__="
MAX_REL_L2 = 1e-3
GENERATORS = {"uniform", "log_uniform", "sparse", "alternating", "constant", "ramp", "packed_bytes", "near_constant"}


def diagnostic_text(value):
    """Bound original process evidence for persistence and diagnostic feedback."""
    raw = value if isinstance(value, bytes) else str(value or "").encode("utf-8", errors="replace")
    return {"text": raw[:8192].decode("utf-8", errors="replace"),
            "bytes": len(raw), "truncated": len(raw) > 8192}


def evaluator_failure(diagnostic):
    """Recover the actual pre-receipt failure, not an input-generation guess."""
    nested = diagnostic.get("evaluator")
    nested = nested if isinstance(nested, dict) else {}
    saved = nested.get("eval_result") or {}
    if isinstance(saved, dict) and not saved.get("truncated"):
        try:
            result = json.loads(saved.get("text", ""))
        except (ValueError, TypeError):
            result = None
        if isinstance(result, dict):
            passed = result.get("passed")
            if isinstance(passed, dict):
                for stage in ("compile", "correctness"):
                    shapes = passed.get(stage)
                    if isinstance(shapes, dict):
                        for row in shapes.values():
                            if isinstance(row, dict) and row.get("status") == "failed" and row.get("reason"):
                                return diagnostic_text(row["reason"])["text"]
            if result.get("error"):
                return diagnostic_text(result["error"])["text"]
    for source in (nested, diagnostic):
        stream = source.get("stderr")
        if isinstance(stream, dict) and stream.get("text"):
            return diagnostic_text(stream["text"])["text"]
    return ""


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
        except Exception:
            # Preserve the generator error even when the privacy-preserving
            # evaluator suppresses its own traceback. No input values are logged.
            try:
                Path(str(receipts) + ".error").write_text(traceback.format_exc())
            except OSError:
                pass  # Do not replace the original generator failure.
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
    diagnostics = []
    try:
        # The standalone driver cannot import the Supervisor package. New
        # requests carry ordinary Evaluate policy; omitted values retain the
        # Legacy default for old standalone callers.
        candidate_timeout = request.get("candidate_timeout_s")
        if candidate_timeout is not None and (
                type(candidate_timeout) not in {int, float}
                or not math.isfinite(candidate_timeout) or candidate_timeout <= 0):
            raise ValueError("candidate_timeout_s must be a positive finite number")
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
                    evaluator_diagnostics = request_path.parent / "evaluator-diagnostics.json"
                    evaluator_diagnostics.unlink(missing_ok=True)
                    tail = ("\nimport runpy as __numeric_runpy\n"
                            f"__numeric_runpy.run_path({str(Path(__file__).resolve())!r})['install_inputs'](globals(), {case!r}, {[seed]!r}, {str(receipts.resolve())!r})\n")
                    input_path.write_bytes(original + tail.encode())
                    for stem in ("input", "test_kernel"):
                        for cached in (root / "__pycache__").glob(f"{stem}.*.pyc"):
                            cached.unlink()
                    command = [*evaluator, "--version", "vlong", "--no-memory", "--correctness-only",
                               "--correctness-max-rel-l2", str(request.get("max_rel_l2", MAX_REL_L2)),
                               "--multi-seed", "0", "--seed", str(seed), "--shape-id", shape_id]
                    if candidate_timeout is not None:
                        command += ["--candidate-timeout-s", str(candidate_timeout)]
                    diagnostic = {"case_id": case["id"], "shape_id": shape_id, "seed": seed,
                                  "stage": "evaluator", "timed_out": False,
                                  "candidate_timeout_s": candidate_timeout}
                    diagnostics.append(diagnostic)
                    try:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise subprocess.TimeoutExpired(command, request["per_case_timeout"])
                        process = subprocess.run(command, cwd=root, capture_output=True, text=True,
                                                 timeout=remaining, env=os.environ | {
                                                     "ATREX_NUMERICAL_DIAGNOSTICS_PATH": str(evaluator_diagnostics.resolve()),
                                                 })
                    except subprocess.TimeoutExpired as exc:
                        diagnostic.update(timed_out=True, exit_code=None,
                                          stdout=diagnostic_text(exc.stdout), stderr=diagnostic_text(exc.stderr))
                        diagnosis = "supplemental evaluator exceeded its case timeout"
                        diagnostic["diagnosis"] = diagnosis
                        break
                    diagnostic.update(exit_code=process.returncode, stdout=diagnostic_text(process.stdout),
                                      stderr=diagnostic_text(process.stderr))
                    if evaluator_diagnostics.is_file():
                        try:
                            with evaluator_diagnostics.open("rb") as source:
                                raw_diagnostics = source.read(128 * 1024 + 1)
                            if len(raw_diagnostics) > 128 * 1024:
                                raise ValueError("nested evaluator diagnostics exceed limit")
                            diagnostic["evaluator"] = json.loads(raw_diagnostics)
                        except (OSError, ValueError, UnicodeError) as exc:
                            diagnostic["capture_error"] = diagnostic_text(exc)
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
                        with input_error.open("rb") as source:
                            raw_error = source.read(8193)
                        diagnostic["input_error"] = diagnostic_text(raw_error)
                        diagnosis = raw_error[:8192].decode("utf-8", errors="replace")
                    elif actual != required:
                        failure = evaluator_failure(diagnostic) if process.returncode else ""
                        diagnosis = (
                            f"evaluator exited {process.returncode} before the requested input receipts completed: {failure}"
                            if failure else "evaluator did not generate the requested workload, seed and ranks"
                        )
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
                    diagnostic.update(diagnosis=diagnosis, observed_probes=len(actual),
                                      expected_probes=len(required))
                    if diagnosis:
                        diagnostic["stage"] = ("input_generation" if input_error.exists() else
                                               "evaluator" if actual != required and process.returncode else
                                               "receipt_validation" if actual != required else "result_validation")
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
        diagnostics.append({"stage": "probe_driver", "traceback": diagnostic_text(traceback.format_exc())})
    finally:
        input_path.write_bytes(original)
        harness.write_bytes(original_harness)
        for stem in ("input", "test_kernel"):
            for cached in (root / "__pycache__").glob(f"{stem}.*.pyc"):
                cached.unlink()
    # Persist all attempts at the Gateway; numerical_result() selects bounded
    # failure diagnostics for both the probe planner and the optimization Agent.
    payload["private_diagnostics"] = diagnostics
    print(PREFIX + json.dumps(payload, allow_nan=False, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(Path(sys.argv[1])))
