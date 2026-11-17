#!/usr/bin/env python3
"""metal-eval tool adapter: compile, correctness, ABBA measurement, record.

Local Apple Silicon execution backend for AKA. Deliberately self-contained
(standard library only) so it can run both as an AKA plugin tool and standalone
during bring-up. It never mutates the operator workspace; all build artifacts go
to a temporary directory owned by the invocation.

Evidence rules inherited from AKA Light:
- ABBA (A/B/B/A-style) alternation, median per block, geomean across blocks.
- The report only echoes measured facts; promotion decisions stay with the
  Supervisor/workflow, never with the tool.
- Correctness is read from the bench driver's own exit status and JSON; a
  numeric failure can never produce a measurement record with pct set.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile

DEFAULT_BUILD_ARGS = ["-std=metal4.0", "-mmacosx-version-min=27.0"]
METAL_TOOLCHAIN_HINT = (
    "xcrun could not find the Metal toolchain. On macOS run once inside Xcode, "
    "or point the toolchain bin directory via ATREX_METAL_BIN."
)


def _run(argv, cwd=None, timeout=600):
    result = subprocess.run(argv, cwd=cwd, timeout=timeout, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result


def _metal_tools():
    bin_dir = os.environ.get("ATREX_METAL_BIN")
    if bin_dir:
        return os.path.join(bin_dir, "metal"), os.path.join(bin_dir, "metallib")
    try:
        metal = subprocess.run(["xcrun", "--find", "metal"], capture_output=True,
            text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise SystemExit(METAL_TOOLCHAIN_HINT)
    return metal, os.path.join(os.path.dirname(metal), "metallib")


def _bench_json(stdout):
    start = stdout.find("{")
    if start < 0:
        raise ValueError("bench driver printed no JSON object")
    return json.loads(stdout[start:])


def evaluate(request: dict) -> dict:
    root = os.path.abspath(os.path.expanduser(request["operator_root"]))
    kernel_source = os.path.join(root, request["kernel_source"])
    bench_source = os.path.join(root, request["bench_source"])
    for path in (root, kernel_source, bench_source):
        if not os.path.exists(path):
            raise SystemExit(f"missing operator input: {path}")

    build_args = list(request.get("build_args") or DEFAULT_BUILD_ARGS)
    bench_args = [str(a) for a in (request.get("bench_args") or [])]
    blocks = int(request.get("blocks") or 4)
    iterations = int(request.get("iterations") or 10)

    workdir = tempfile.mkdtemp(prefix="atrex-metal-eval-")
    try:
        metal, metallib = _metal_tools()
        air = os.path.join(workdir, "candidate.air")
        library = os.path.join(workdir, "candidate.metallib")
        compiled = _run([metal, *build_args, "-c", kernel_source, "-o", air], timeout=300)
        if compiled.returncode != 0:
            return {"status": "compile_failed", "stderr": compiled.stderr[-4000:]}
        linked = _run([metallib, air, "-o", library], timeout=120)
        if linked.returncode != 0:
            return {"status": "metallib_failed", "stderr": linked.stderr[-4000:]}

        bench_binary = os.path.join(workdir, "bench")
        built = _run(["xcrun", "clang++", "-fobjc-arc", "-std=c++20", "-O2",
            f"-DONYX_METALLIB_PATH=\"{library}\"", bench_source, "-o", bench_binary,
            "-framework", "Foundation", "-framework", "Metal"], timeout=600)
        if built.returncode != 0:
            return {"status": "bench_build_failed", "stderr": built.stderr[-4000:]}

        variants = (request["baseline_variant"], request["candidate_variant"])
        samples = {name: [] for name in variants}
        correctness = {name: None for name in variants}
        errors = {name: None for name in variants}
        device = None
        for block in range(blocks):
            # ABBA order: baseline first on even blocks, candidate first on odd.
            order = list(variants) if block % 2 == 0 else list(reversed(variants))
            for name in order:
                result = _run([bench_binary, "--variant", name, *bench_args], timeout=900)
                try:
                    payload = _bench_json(result.stdout)
                except (ValueError, json.JSONDecodeError):
                    return {"status": "bench_output_unparseable", "variant": name,
                        "exit_code": result.returncode, "stderr": result.stderr[-2000:]}
                samples[name].append(payload["gpu_median_ms"])
                correctness[name] = bool(payload.get("correct"))
                errors[name] = payload.get("max_relative_error")
                device = payload.get("device") or device
                if result.returncode not in (0, 2):
                    return {"status": "bench_failed", "variant": name,
                        "exit_code": result.returncode, "stderr": result.stderr[-2000:]}
                if result.returncode == 2:
                    return {"status": "correctness_failed", "variant": name,
                        "max_relative_error": errors[name]}

        medians = {name: statistics.median(values) for name, values in samples.items()}
        base, cand = medians[variants[0]], medians[variants[1]]
        reduction_pct = round((base - cand) / base * 100.0, 3) if base else None
        record = {
            "status": "measured",
            "protocol": "atrex.metal-eval.abba.v0",
            "device": device,
            "blocks": blocks,
            "iterations": iterations,
            "bench_args": bench_args,
            "variants": {
                name: {"gpu_median_ms_per_block": samples[name], "gpu_median_ms": medians[name],
                       "correct": correctness[name], "max_relative_error": errors[name]}
                for name in variants},
            "latency_reduction_pct": reduction_pct,
            "decision_hint": None,
        }
        # The tool measures; it does not promote. Decision hints are diagnostic.
        if reduction_pct is not None:
            record["decision_hint"] = (
                "candidate improves" if reduction_pct > 0 else "incumbent retained")
        if request.get("output"):
            os.makedirs(os.path.dirname(os.path.abspath(request["output"])), exist_ok=True)
            with open(request["output"], "w") as handle:
                json.dump(record, handle, indent=1)
        return record
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="metal-eval")
    parser.add_argument("--input", required=True, help="path to request JSON")
    parser.add_argument("--output", help="optional response JSON path")
    args = parser.parse_args(argv)
    with open(args.input) as handle:
        request = json.load(handle)
    response = evaluate(request)
    text = json.dumps(response, indent=1)
    print(text)
    if args.output:
        with open(args.output, "w") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
