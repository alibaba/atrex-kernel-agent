#!/usr/bin/env python3
"""Standard-library helpers for native ABBA summaries and source-tree diagnostics.

Clock and dependency helpers are uploaded with the diagnostic driver; this module
contains no evaluation runner or Dev ABBA entrypoint.
"""

from __future__ import annotations

import math
import os
import shutil
import statistics
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager

_CLOCK_MARKERS = (
    "ATREX_BENCH_CLOCKS_LOCKED",
    "SOL_EXECBENCH_CLOCKS_LOCKED",
)
_CLOCK_SOURCE = "ATREX_BENCH_CLOCK_LOCK_SOURCE"


def _run_clock_command(argv: list[str]) -> str:
    completed = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    if completed.returncode != 0:
        detail = next(
            (line.strip() for line in completed.stderr.splitlines() if line.strip()),
            "no stderr",
        )
        raise RuntimeError(
            f"clock command exited {completed.returncode}: {' '.join(argv)}; {detail}"
        )
    return completed.stdout


def _nvidia_selector(executable: str) -> str:
    explicit = os.environ.get("ATREX_BENCH_CLOCK_DEVICE", "").strip()
    if explicit:
        return explicit
    for name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        value = os.environ.get(name, "").strip()
        tokens = tuple(token.strip() for token in value.split(",") if token.strip())
        if len(tokens) == 1 and tokens[0].lower() not in {"all", "none", "void"}:
            return tokens[0]
    rows = [
        line.strip()
        for line in _run_clock_command(
            [
                executable,
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ]
        ).splitlines()
        if line.strip()
    ]
    if len(rows) != 1:
        raise RuntimeError(
            "clock locking requires exactly one visible NVIDIA GPU or an explicit selector"
        )
    parts = [part.strip() for part in rows[0].split(",")]
    if len(parts) != 2 or not all(parts):
        raise RuntimeError(f"could not parse NVIDIA GPU identity: {rows[0]!r}")
    return parts[1]


@contextmanager
def _clock_lock(enabled: bool) -> Iterator[dict[str, object]]:
    if not enabled:
        yield {"requested": False, "applied": False, "backend": None}
        return

    reset: list[str]
    report: dict[str, object]
    nvidia_smi = shutil.which("nvidia-smi")
    rocm_smi = shutil.which("rocm-smi")
    if nvidia_smi is not None:
        selector = _nvidia_selector(nvidia_smi)
        supported = _run_clock_command(
            [
                nvidia_smi,
                "-i",
                selector,
                "--query-supported-clocks=gr",
                "--format=csv,noheader,nounits",
            ]
        )
        clocks: list[float] = []
        for line in supported.splitlines():
            text = line.strip().split(",")[-1].strip()
            if not text:
                continue
            try:
                clocks.append(float(text))
            except ValueError:
                continue
        if not clocks:
            raise RuntimeError("nvidia-smi returned no supported graphics clocks")
        graphics_mhz = int(max(clocks))
        _run_clock_command(
            [
                nvidia_smi,
                "-i",
                selector,
                "--lock-gpu-clocks",
                f"{graphics_mhz},{graphics_mhz}",
            ]
        )
        reset = [nvidia_smi, "-i", selector, "--reset-gpu-clocks"]
        report = {
            "requested": True,
            "applied": True,
            "backend": "nvidia-smi",
            "device_selector": selector,
            "graphics_mhz": graphics_mhz,
        }
    elif rocm_smi is not None:
        _run_clock_command([rocm_smi, "--setperflevel", "high"])
        reset = [rocm_smi, "--setperflevel", "auto"]
        report = {
            "requested": True,
            "applied": True,
            "backend": "rocm-smi",
            "performance_level": "high",
        }
    else:
        raise RuntimeError("clock locking requires nvidia-smi or rocm-smi")

    saved = {name: os.environ.get(name) for name in (*_CLOCK_MARKERS, _CLOCK_SOURCE)}
    for name in _CLOCK_MARKERS:
        os.environ[name] = "1"
    os.environ[_CLOCK_SOURCE] = "atrex-runtime-abba"
    try:
        yield report
    finally:
        reset_error: BaseException | None = None
        try:
            _run_clock_command(reset)
        except BaseException as error:  # fail closed after attempting environment restoration
            reset_error = error
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if reset_error is not None:
            raise RuntimeError(f"failed to reset GPU clocks: {reset_error}") from reset_error
        report["restored"] = True


def _positive_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number > 0 and math.isfinite(number) else None


def _nonnegative_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number >= 0 and math.isfinite(number) else None


def _status_passed(value: object) -> bool:
    return isinstance(value, dict) and value.get("status") == "passed"


def _compile_passed(value: object, shape_ids: list[str]) -> bool:
    if not isinstance(value, dict):
        return False
    aggregate = value.get("status")
    if aggregate is not None:
        return bool(aggregate == "passed")
    return all(_status_passed(value.get(shape_id)) for shape_id in shape_ids)


def _correctness_summary(value: object, *, passed: bool) -> dict[str, object]:
    aliases = {
        "rel_err": {"rel_err", "relative_l2"},
        "max_abs_err": {"max_abs_err", "max_elementwise_abs_diff"},
        "max_rel_err": {"max_rel_err", "max_elementwise_rel_diff"},
    }
    return {
        "status": "PASS" if passed else "FAIL",
        **{name: _maximum_metric(value, keys) for name, keys in aliases.items()},
    }


def _maximum_metric(value: object, aliases: set[str]) -> float | None:
    maximum: float | None = None
    if isinstance(value, dict):
        for key, child in value.items():
            if key in aliases:
                number = _nonnegative_number(child)
                if number is not None:
                    maximum = number if maximum is None else max(maximum, number)
            nested = _maximum_metric(child, aliases)
            if nested is not None:
                maximum = nested if maximum is None else max(maximum, nested)
    elif isinstance(value, (list, tuple)):
        for child in value:
            nested = _maximum_metric(child, aliases)
            if nested is not None:
                maximum = nested if maximum is None else max(maximum, nested)
    return maximum


def _summarize(payload: object, shape_ids: list[str]) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {"all_pass": False, "latency_us_by_shape": {}, "error": "invalid result"}
    passed = payload.get("passed")
    passed = passed if isinstance(passed, dict) else {}
    correctness = payload.get("correctness")
    correctness = correctness if isinstance(correctness, dict) else {}
    correctness_shapes = correctness.get("shapes")
    correctness_shapes = correctness_shapes if isinstance(correctness_shapes, dict) else {}
    performance = payload.get("performance")
    performance = performance if isinstance(performance, dict) else {}
    performance_shapes = performance.get("shapes")
    performance_shapes = performance_shapes if isinstance(performance_shapes, dict) else {}

    correct = (
        payload.get("error") is None
        and _compile_passed(passed.get("compile"), shape_ids)
        and isinstance(passed.get("correctness"), dict)
        and all(_status_passed(passed["correctness"].get(shape_id)) for shape_id in shape_ids)
        and all(shape_id in correctness_shapes for shape_id in shape_ids)
    )
    latency_by_shape: dict[str, float] = {}
    sol_pct_by_shape: dict[str, float] = {}
    for shape_id in shape_ids:
        shape = performance_shapes.get(shape_id)
        if not isinstance(shape, dict) or shape.get("error") is not None:
            correct = False
            continue
        samples = shape.get("samples")
        values = []
        for sample in samples if isinstance(samples, list) else []:
            if not isinstance(sample, dict):
                continue
            number = _positive_number(sample.get("end_to_end_time_ms"))
            if number is not None:
                values.append(number)
        if not values:
            correct = False
            continue
        latency_by_shape[shape_id] = statistics.median(values) * 1000.0
        sol = shape.get("sol")
        if isinstance(sol, dict):
            percentage = _nonnegative_number(sol.get("pct"))
            if percentage is not None:
                sol_pct_by_shape[shape_id] = percentage
    correct = correct and len(latency_by_shape) == len(shape_ids) and bool(shape_ids)
    latencies = [
        latency_by_shape[shape_id] for shape_id in shape_ids if shape_id in latency_by_shape
    ]
    return {
        "all_pass": correct,
        "correctness": _correctness_summary(correctness, passed=correct),
        "latency_us_geomean": (
            math.exp(statistics.fmean(math.log(value) for value in latencies)) if correct else None
        ),
        "latency_us_by_shape": latency_by_shape,
        "sol_pct_by_shape": sol_pct_by_shape,
        "error": None if correct else "compile, correctness, or performance did not pass",
    }


def _check_requirements(requirements: list[dict[str, str]]) -> None:
    """Check the provisioned GPU environment; never install Agent-selected code."""
    from importlib.metadata import version

    from packaging.specifiers import SpecifierSet

    for requirement in requirements:
        installed = version(requirement["distribution"])
        if installed not in SpecifierSet(requirement.get("version", "")):
            raise RuntimeError(
                f"source requirement unavailable: {requirement}, installed={installed}"
            )
