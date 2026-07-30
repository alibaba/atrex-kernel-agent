#!/usr/bin/env python3
"""Optimizer transport adapter for the official Atrex-Bench ``run_eval``.

This file deliberately contains no candidate execution, correctness comparison,
or timing implementation.  Native ``shapes.json`` campaigns copy it to
``test_kernel.py``; it invokes the bundled, canonical Atrex-Bench
``scripts/run_eval.py`` and converts that evaluator's raw ``eval_result.json``
into the small ``RESULT_JSON`` contract consumed by the optimizer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


RESULT_PREFIX = "[test_kernel] RESULT_JSON="
ATREX_BENCH_DIR = "atrex-bench"


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _expected_shape_ids(workspace: Path) -> list[str]:
    payload = json.loads((workspace / "shapes.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise RuntimeError("shapes.json must contain a non-empty object")

    def sort_key(shape_id: str) -> tuple[int, object]:
        return (0, int(shape_id)) if shape_id.isdigit() else (1, shape_id)

    return sorted((str(shape_id) for shape_id in payload), key=sort_key)


def result_from_eval(payload: dict[str, Any], shape_ids: list[str]) -> dict[str, Any]:
    """Convert one official Atrex-Bench result into optimizer metrics."""
    failures: list[str] = []
    evaluation_error = payload.get("error")
    if evaluation_error:
        failures.append("evaluation: " + str(evaluation_error))
    passed = payload.get("passed")
    passed = passed if isinstance(passed, dict) else {}
    compile_result = passed.get("compile")
    compile_result = compile_result if isinstance(compile_result, dict) else {}
    if compile_result.get("status") != "passed":
        failures.append(
            "compile: " + str(compile_result.get("reason") or "did not pass")
        )

    correctness_status = passed.get("correctness")
    correctness_status = (
        correctness_status if isinstance(correctness_status, dict) else {}
    )
    correctness = payload.get("correctness")
    correctness = correctness if isinstance(correctness, dict) else {}
    correctness_shapes = correctness.get("shapes")
    correctness_shapes = (
        correctness_shapes if isinstance(correctness_shapes, dict) else {}
    )

    max_abs = 0.0
    max_rel = 0.0
    for shape_id in shape_ids:
        status = correctness_status.get(shape_id)
        status = status if isinstance(status, dict) else {}
        if status.get("status") != "passed":
            failures.append(
                f"sid={shape_id}: correctness "
                + str(status.get("reason") or status.get("status") or "missing")
            )

        shape_result = correctness_shapes.get(shape_id)
        shape_result = shape_result if isinstance(shape_result, dict) else {}
        cases = shape_result.get("cases")
        for case in cases if isinstance(cases, list) else []:
            if not isinstance(case, dict):
                continue
            outputs = case.get("outputs")
            for output in outputs if isinstance(outputs, list) else []:
                if not isinstance(output, dict):
                    continue
                abs_diff = _finite_number(output.get("max_elementwise_abs_diff"))
                rel_diff = _finite_number(output.get("max_elementwise_rel_diff"))
                if abs_diff is not None:
                    max_abs = max(max_abs, abs_diff)
                if rel_diff is not None:
                    max_rel = max(max_rel, rel_diff)

    performance = payload.get("performance")
    performance = performance if isinstance(performance, dict) else {}
    performance_shapes = performance.get("shapes")
    performance_shapes = (
        performance_shapes if isinstance(performance_shapes, dict) else {}
    )
    latency_by_shape: dict[str, float] = {}
    for shape_id in shape_ids:
        shape_result = performance_shapes.get(shape_id)
        shape_result = shape_result if isinstance(shape_result, dict) else {}
        error = shape_result.get("error")
        samples = shape_result.get("samples")
        sample_ms: list[float] = []
        for sample in samples if isinstance(samples, list) else []:
            if not isinstance(sample, dict):
                continue
            value = _finite_number(sample.get("end_to_end_time_ms"))
            if value is not None and value > 0.0:
                sample_ms.append(value)
        if error is not None or not sample_ms:
            failures.append(
                f"sid={shape_id}: performance " + str(error or "has no valid samples")
            )
            continue
        latency_by_shape[shape_id] = statistics.median(sample_ms) * 1000.0

    latencies = [latency_by_shape[shape_id] for shape_id in shape_ids if shape_id in latency_by_shape]
    complete_performance = len(latencies) == len(shape_ids)
    latency_geomean = (
        math.exp(sum(math.log(value) for value in latencies) / len(latencies))
        if complete_performance and latencies
        else 0.0
    )
    latency_arith_mean = (
        sum(latencies) / len(latencies) if complete_performance and latencies else 0.0
    )

    return {
        "all_pass": not failures,
        "failures": failures,
        "latency_us_geomean": latency_geomean,
        "latency_us_arith_mean": latency_arith_mean,
        "latency_us_by_shape": latency_by_shape,
        # Atrex-Bench run_eval intentionally records candidate-only timing.
        "speedup_vs_ref_geomean": None,
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
        "evaluator": "atrex-bench/run_eval",
        "eval_id": payload.get("eval_id"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the official Atrex-Bench evaluator and emit optimizer RESULT_JSON"
    )
    parser.add_argument("--version", default="v0")
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument(
        "--multi-seed",
        type=int,
        default=0,
        help="Additional correctness cases; performance is still measured once per shape",
    )
    parser.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--timed-runs", type=int, default=100)
    parser.add_argument("--candidate-timeout-s", type=float, default=20.0)
    parser.add_argument("--perf-timeout-s", type=float, default=120.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.multi_seed < 0:
        raise SystemExit("--multi-seed must be non-negative")

    workspace = Path(__file__).resolve().parent
    runtime_root = workspace / ATREX_BENCH_DIR
    evaluator = runtime_root / "scripts" / "run_eval.py"
    runtime_src = runtime_root / "src"
    if not evaluator.is_file() or not (runtime_src / "atrex_bench").is_dir():
        raise SystemExit(
            "official Atrex-Bench runtime is missing; expected "
            f"{evaluator} and {runtime_src / 'atrex_bench'}"
        )

    shape_ids = _expected_shape_ids(workspace)
    env = os.environ.copy()
    pythonpath = str(runtime_src)
    if env.get("PYTHONPATH"):
        pythonpath += os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath

    with tempfile.TemporaryDirectory(prefix="atrex-eval-") as output_dir:
        command = [
            sys.executable,
            str(evaluator),
            "--input",
            str(workspace / "kernel.py"),
            "--reference-dir",
            str(workspace),
            "--output",
            output_dir,
            "--atol",
            str(args.atol),
            "--rtol",
            str(args.rtol),
            "--num-correctness-cases",
            str(1 + args.multi_seed),
            "--warmup-iters",
            str(args.warmup),
            "--bench-iters",
            str(args.timed_runs),
            "--candidate-timeout-s",
            str(args.candidate_timeout_s),
            "--perf-timeout-s",
            str(args.perf_timeout_s),
        ]
        completed = subprocess.run(
            command,
            cwd=str(runtime_root),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.stdout:
            print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
        if completed.stderr:
            print(
                completed.stderr,
                end="" if completed.stderr.endswith("\n") else "\n",
                file=sys.stderr,
            )

        result_paths = sorted(Path(output_dir).rglob("eval_result.json"))
        if not result_paths:
            result = {
                "all_pass": False,
                "failures": [
                    f"atrex-bench run_eval exited {completed.returncode} without eval_result.json"
                ],
                "latency_us_geomean": 0.0,
                "latency_us_arith_mean": 0.0,
                "latency_us_by_shape": {},
                "speedup_vs_ref_geomean": None,
                "max_abs_err": 0.0,
                "max_rel_err": 0.0,
                "evaluator": "atrex-bench/run_eval",
                "eval_id": None,
            }
        else:
            payload = json.loads(result_paths[-1].read_text(encoding="utf-8"))
            result = result_from_eval(payload, shape_ids)
            if completed.returncode != 0 and result["all_pass"]:
                result["all_pass"] = False
                result["failures"].append(
                    f"atrex-bench run_eval exited with code {completed.returncode}"
                )

    print(RESULT_PREFIX + json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
