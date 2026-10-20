#!/usr/bin/env python3
"""Remote NCU collection/CSV parsing using only the Python standard library.

Uploaded as tools/ncu_profile.py by both Gateway implementations. The Agent
consumes PROFILE_JSON, never a locally imported ncu_report module. CSV handling
follows Runtime's source diagnostics: exact metric names, row/column layouts,
explicit units, finite values and bounded text exports.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

PREFIX = "[sandbox] PROFILE_JSON="
RESPONSE_LIMIT = 64 * 1024
EXPORT_LIMIT = 512 * 1024
CSV_LIMIT = 16 * 1024 * 1024


def metric_rows(text):
    fields, metrics, units = [], [], {}
    for cells in csv.reader(io.StringIO(text)):
        if "ID" in cells and "Kernel Name" in cells:
            fields, units = cells, {}
            metrics = [name for name in fields if "__" in name]
            continue
        if not fields or len(cells) != len(fields):
            continue
        row = dict(zip(fields, cells))
        if "Metric Name" in fields and "Metric Value" in fields:
            yield row
        elif not row.get("ID") and not row.get("Kernel Name"):
            units = row
        elif row.get("Kernel Name"):
            for metric in metrics:
                yield {"ID": row["ID"], "Process ID": row.get("Process ID", ""),
                       "Kernel Name": row["Kernel Name"], "Metric Name": metric,
                       "Metric Unit": units.get(metric, ""), "Metric Value": row[metric]}


def kernels(text):
    parsed = {}
    for row in metric_rows(text):
        name, metric = row.get("Kernel Name"), row.get("Metric Name")
        if not name or not metric:
            continue
        key = (row.get("Process ID", ""), row.get("ID", name))
        kernel = parsed.setdefault(key, {"name": name, "metrics": {}})
        raw = row.get("Metric Value", "")
        try:
            value = float(raw.replace(",", ""))
            value = value if math.isfinite(value) else None
        except ValueError:
            value = raw
        kernel["metrics"][metric] = {"value": value, "unit": row.get("Metric Unit", "")}
    for kernel in parsed.values():
        def number(*names):
            for name in names:
                value = kernel["metrics"].get(name, {}).get("value")
                if type(value) in (int, float):
                    return value
            return None

        duration = kernel["metrics"].get("gpu__time_duration.sum", {})
        scale = {"nsecond": .001, "ns": .001, "usecond": 1, "us": 1,
                 "msecond": 1000, "ms": 1000, "second": 1e6, "s": 1e6}.get(duration.get("unit"))
        if scale is not None and type(duration.get("value")) in (int, float):
            kernel["duration_us"] = duration["value"] * scale
        for field, names in {
            "compute_sol_pct": ("sm__throughput.avg.pct_of_peak_sustained_elapsed",),
            "memory_sol_pct": ("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",),
            "dram_throughput_pct": ("dram__throughput.avg.pct_of_peak_sustained_elapsed",
                                    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"),
            "achieved_occupancy_pct": ("sm__warps_active.avg.pct_of_peak_sustained_active",),
            "registers_per_thread": ("launch__registers_per_thread",),
            "shared_memory_bytes": ("launch__shared_mem_per_block",),
            "waves_per_sm": ("launch__waves_per_multiprocessor",),
        }.items():
            value = number(*names)
            if value is not None:
                kernel[field] = value
        def counter(names, units):
            for name in names:
                item = kernel["metrics"].get(name, {})
                value = item.get("value")
                if type(value) in (int, float) and value >= 0 and item.get("unit") in units:
                    return value
            return None

        traffic = {}
        for field, names in {
            "dram_bytes": ("dram__bytes.sum",),
            "dram_bytes_read": ("dram__bytes_read.sum", "dram__bytes_op_read.sum"),
            "dram_bytes_write": ("dram__bytes_write.sum", "dram__bytes_op_write.sum"),
            "l2_bytes": ("lts__t_bytes.sum",),
        }.items():
            value = counter(names, {"byte", "bytes", "B"})
            if value is not None:
                traffic[field] = value
        if "dram_bytes" not in traffic and {"dram_bytes_read", "dram_bytes_write"} <= traffic.keys():
            traffic["dram_bytes"] = traffic["dram_bytes_read"] + traffic["dram_bytes_write"]
        if "l2_bytes" not in traffic:
            # NCU's L2 sectors are 32 bytes (Nsight Compute Memory Tables).
            # Prefer an explicit byte counter; never add aliases or components
            # to a reported total, or mistake a sectors/second rate for traffic.
            sectors = counter(("lts__t_sectors.sum",), {"sector", "sectors"})
            if sectors is not None:
                traffic["l2_bytes"] = sectors * 32
        if kernel.get("duration_us", 0) > 0:
            traffic["duration_ns"] = kernel["duration_us"] * 1000
            if "dram_bytes" in traffic:
                traffic["achieved_dram_gbps"] = traffic["dram_bytes"] / traffic["duration_ns"]
        if traffic:
            kernel["traffic"] = traffic
    return list(parsed.values())


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def public_result(result):
    """Bound the response independently of the exported CSV/source file sizes."""
    value = copy.deepcopy(result)
    value["kernels"] = value.get("kernels", [])[:32]
    for kernel in value["kernels"]:
        kernel["name"] = kernel["name"][:512]
        metrics = kernel.get("metrics", {})
        kernel["metric_count"] = len(metrics)
        kernel["metrics"] = dict(list(metrics.items())[:64])
    def encoded_size():
        for kernel in value["kernels"]:
            kernel["metrics_omitted"] = kernel["metric_count"] - len(kernel["metrics"])
        value["kernels_omitted"] = value.get("kernel_count", len(value["kernels"])) - len(value["kernels"])
        return len((PREFIX + _json(value) + "\n").encode())

    while encoded_size() > RESPONSE_LIMIT:
        candidates = [k for k in value["kernels"] if k.get("metrics")]
        if candidates:
            largest = max(candidates, key=lambda k: len(_json(k["metrics"])))
            largest["metrics"] = dict(list(largest["metrics"].items())[:len(largest["metrics"]) // 2])
        elif len(value["kernels"]) > 1:
            value["kernels"].pop()
        else:
            raise ValueError("Profile summary exceeds response budget")
    return value


def _run(argv, *, limit=CSV_LIMIT):
    # Temporary files keep subprocess output out of unbounded Python buffers.
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        process = subprocess.run(argv, stdout=out, stderr=err, timeout=120,
                                 env=dict(os.environ, LC_ALL="C"))
        err.seek(0)
        if process.returncode:
            detail = err.read(4096).decode(errors="replace")
            raise ValueError(f"Remote NCU export failed (exit {process.returncode}): {detail}")
        out.seek(0)
        raw = out.read(limit + 1)
        if len(raw) > limit:
            raise ValueError("Remote NCU export exceeds 16 MiB; narrow the kernel filter/launch count")
        return raw


def _export(root, name, raw):
    text = raw[:EXPORT_LIMIT].decode(errors="ignore")
    (root / name).write_text(text)
    return {"path": "analysis/" + name, "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(), "truncated": len(raw) > EXPORT_LIMIT}


def has_sass(raw):
    # NCU CLI versions print PCs as bare hex columns, 0x-prefixed addresses,
    # or /*offset*/ comments. Require an instruction too: a banner mentioning
    # an address (or CUDA source alone) is not SASS evidence.
    instruction = (
        rb"(?m)^[ \t]*(?:/\*[0-9a-fA-F]+\*/|0x[0-9a-fA-F]+|[0-9a-fA-F]{4,})"
        rb"[ \t]*(?:[|:][ \t]*)?(?:@!?[UP](?:T|[0-9]+)[ \t]+)?[A-Z][A-Z0-9_.]*\b"
    )
    if re.search(instruction + rb"[^;\r\n]*;", raw):
        return True
    # The CLI's Address/Source table omits nvdisasm's semicolons and appends
    # metric columns. Require the table header as well as an instruction row.
    return bool(re.search(rb"(?m)^[ \t]*Address[ \t]+Source\b", raw)
                and re.search(instruction + rb"(?:[ \t]|$)", raw))


def _source_export(ncu, report):
    # CUDA correlation needs line info, but SASS does not. Some NCU versions
    # produce only a banner for cuda,sass on prebuilt libraries. Retry the
    # export (not collection) using the same report and label the fallback.
    for view in ("cuda,sass", "sass"):
        try:
            source = _run([ncu, "--import", str(report), "--page", "source", "--print-source", view])
        except ValueError:
            if view == "sass":
                raise
            continue
        if has_sass(source):
            return source, view
    raise ValueError("NCU returned no source/SASS evidence in either view; check report/toolchain support")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("kernel")
    parser.add_argument("--output-dir")
    parser.add_argument("--kernel-name")
    parser.add_argument("--launch-skip", type=int, default=0)
    parser.add_argument("--launch-count", type=int, default=1)
    parser.add_argument("--source", action="store_true")
    parser.add_argument("--no-classify", action="store_true")
    parser.add_argument("--diff", type=Path)
    args = parser.parse_args(argv)
    output = Path(args.output_dir or (
        "profiles/latest" if Path("memory").is_dir() and Path("profiles").is_dir() else "profiles/v0"
    ))
    try:
        if args.launch_skip < 0 or args.launch_count < 1:
            raise ValueError("launch-skip must be nonnegative and launch-count must be positive")
        ncu = shutil.which(os.environ.get("NCU_BIN", "ncu"))
        if not ncu:
            raise ValueError("NCU CLI is unavailable on the GPU worker; ask the operator to provision it")
        output.mkdir(parents=True, exist_ok=True)
        analysis = output / "analysis"
        analysis.mkdir(exist_ok=True)
        report = output / "ncu.ncu-rep"
        # Avoid mistaking an earlier file for fresh output when no kernel matches.
        report.unlink(missing_ok=True)
        command = [ncu, "--set", "full", "--force-overwrite", "--launch-skip", str(args.launch_skip),
                   "--launch-count", str(args.launch_count), "--kill", "yes", "-o", str(report)]
        if args.kernel_name:
            command += ["--kernel-name", args.kernel_name]
        if args.source:
            command += ["--section", "SourceCounters", "--import-source", "yes"]
        # The enclosing Agate/SSH request owns the collection deadline.
        collected = subprocess.run(command + [sys.executable, args.kernel])
        if collected.returncode:
            raise ValueError(f"Remote NCU collection failed (exit {collected.returncode}); inspect the preceding worker diagnostic")
        if not report.is_file():
            raise ValueError("NCU produced no report; check the kernel filter and launch range")
        raw = _run([ncu, "--import", str(report), "--page", "raw", "--csv", "--print-units", "base"])
        parsed = kernels(raw.decode(errors="replace"))
        if not parsed or not any(k["metrics"] for k in parsed):
            raise ValueError("NCU CSV contains no kernel metrics; check the kernel filter/launch range")
        parsed.sort(key=lambda k: k.get("duration_us", 0), reverse=True)
        result = {"status": "succeeded", "passed": True, "profiler": "ncu", "kernels": parsed,
                  "kernel_count": len(parsed), "total_duration_us": sum(k.get("duration_us", 0) for k in parsed),
                  "exports": {"metrics.csv": _export(analysis, "metrics.csv", raw)}}
        if args.source:
            source, view = _source_export(ncu, report)
            result["exports"]["source.txt"] = _export(analysis, "source.txt", source)
            result["source_view"] = view
            if view == "sass":
                result["warnings"] = ["CUDA/source correlation unavailable; exported SASS-only evidence from the same report"]
        # Preserve the classifier's legacy key-metric input, now derived remotely
        # from CSV. It describes the dominant launch, not all launches combined.
        key = {name: metric["value"] for name, metric in parsed[0]["metrics"].items()}
        key["__kernel_name__"] = parsed[0]["name"]
        (analysis / "metrics_key_run.json").write_text(_json(key) + "\n")
        (analysis / "metrics_key_run.txt").write_text("\n".join(f"{k}: {v}" for k, v in key.items()) + "\n")
        if args.diff:
            previous = json.loads((args.diff / "analysis/metrics_key_run.json").read_text())
            delta = {name: value - previous[name] for name, value in key.items()
                     if type(value) in (int, float) and type(previous.get(name)) in (int, float)}
            (analysis / "diff_metrics.json").write_text(_json(delta) + "\n")
        classifier = Path(__file__).with_name("classify_ncu.py")
        if not args.no_classify and classifier.is_file():
            classified = subprocess.run([sys.executable, str(classifier), "--metrics",
                str(analysis / "metrics_key_run.json"), "--output", str(output / "summary.txt")])
            if classified.returncode:
                result.setdefault("warnings", []).append("Optional symptom classification failed; use the measured metrics")
        result = public_result(result)
        (output / "profile.json").write_text(_json(result) + "\n")
        print(PREFIX + _json(result), flush=True)
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        result = {"status": "failed", "passed": False, "profiler": "ncu",
                  "error": {"reason": "remote_profile_failed", "message": str(exc)[:4096]}, "kernels": []}
        try:
            if output.is_dir():
                (output / "profile.json").write_text(_json(result) + "\n")
        except OSError:
            pass  # The stdout receipt remains authoritative if the sink is unavailable.
        print(PREFIX + _json(result), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
