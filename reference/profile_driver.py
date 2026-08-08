#!/usr/bin/env python3
"""External profiling driver for a kernel-opt workspace.

Both profiler wrappers (`tools/profile_nvidia.sh`, `tools/profile_kernel.sh`) run
`python <file>`, so the profiled file must be runnable on its own. `kernel.py` is not:
the evaluator only ever *imports* it. This driver is that runnable entry point, kept
outside `kernel.py` so no rewrite of `run()`/`Model` can silently remove the ability
to profile.

It builds real inputs from the campaign's immutable ground-truth files, warms up, and
then invokes the candidate repeatedly so the profiler captures steady-state launches.
It never writes `memory/`, never benchmarks, and never reports correctness — the
immutable `test_kernel.py` owns all of that.

Two campaign layouts are supported and detected from the files present:

- SOL-ExecBench: `definition.json` + `workload.jsonl`, candidate exposes DPS `run()`.
- Atrex-Bench:   `shapes.json` + `input.py`, candidate exposes `Model`.

Usage (from the workspace root, normally under a profiler wrapper):

    python profile_driver.py
    PROFILE_ITERS=30 python profile_driver.py
    PROFILE_WORKLOAD_IDX=2 python profile_driver.py     # SOL: pick a workload
    PROFILE_SHAPE_ID=3 python profile_driver.py         # Atrex-Bench: pick a shape

Environment:
    PROFILE_ITERS         profiled iterations (default 10)
    PROFILE_WARMUP        warmup iterations (default 3)
    PROFILE_WORKLOAD_IDX  SOL workload index (default 0)
    PROFILE_SHAPE_ID      Atrex-Bench shape id (default: first sorted key)
    PROFILE_DEVICE        torch device (default cuda:0)
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType


def _workspace_root() -> Path:
    """Return the workspace root, whether this file sits at the root or under harness/.

    A copy placed at `<profile-dir>/harness/profile_driver.py` still resolves correctly:
    the root is the nearest ancestor holding the candidate plus a ground-truth source.
    """
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if not (candidate / "kernel.py").is_file():
            continue
        if (candidate / "definition.json").is_file() or (candidate / "shapes.json").is_file():
            return candidate
    raise RuntimeError(
        "cannot locate the workspace root: no ancestor holds kernel.py plus "
        "definition.json or shapes.json"
    )


def _import_from(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so a candidate module importing itself by name still resolves.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _iterations() -> tuple[int, int]:
    warmup = max(0, int(os.environ.get("PROFILE_WARMUP", "3")))
    iters = max(1, int(os.environ.get("PROFILE_ITERS", "10")))
    return warmup, iters


def _run_sol(root: Path, device: str) -> None:
    """Drive a SOL-ExecBench candidate through its DPS `run()` entry point."""
    from sol_execbench.core.bench.io import allocate_outputs, gen_inputs
    from sol_execbench.core.data import Definition, Workload

    definition = Definition(**json.loads((root / "definition.json").read_text(encoding="utf-8")))
    workloads = [
        Workload(**json.loads(line))
        for line in (root / "workload.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not workloads:
        raise RuntimeError("workload.jsonl holds no workloads")
    index = int(os.environ.get("PROFILE_WORKLOAD_IDX", "0"))
    if not 0 <= index < len(workloads):
        raise RuntimeError(
            f"PROFILE_WORKLOAD_IDX={index} is out of range for {len(workloads)} workloads"
        )
    workload = workloads[index]

    kernel = _import_from(root / "kernel.py", "kernel")
    run = getattr(kernel, "run", None)
    if not callable(run):
        raise RuntimeError("kernel.py does not define a callable run()")

    inputs = gen_inputs(definition, workload, device=device)
    outputs = allocate_outputs(
        definition, definition.get_resolved_axes_values(workload.axes), device
    )
    _drive(lambda: run(*inputs, *outputs), f"SOL workload[{index}]")


def _run_atrex_bench(root: Path, device: str) -> None:
    """Drive an Atrex-Bench candidate through its `Model` entry point."""
    shapes = json.loads((root / "shapes.json").read_text(encoding="utf-8"))
    if not isinstance(shapes, dict) or not shapes:
        raise RuntimeError("shapes.json must contain a non-empty object")

    def sort_key(shape_id: str) -> tuple[int, object]:
        return (0, int(shape_id)) if shape_id.isdigit() else (1, shape_id)

    shape_id = os.environ.get("PROFILE_SHAPE_ID") or sorted(shapes, key=sort_key)[0]
    entry = shapes.get(str(shape_id))
    if not isinstance(entry, dict):
        raise RuntimeError(f"shapes.json has no object entry for shape id {shape_id!r}")
    init_kwargs = entry.get("init_kwargs") or {}
    input_kwargs = entry.get("input_kwargs") or {}
    if not isinstance(init_kwargs, dict) or not isinstance(input_kwargs, dict):
        raise RuntimeError(f"shapes.json[{shape_id!r}] init_kwargs/input_kwargs must be objects")

    kernel = _import_from(root / "kernel.py", "kernel")
    model_class = getattr(kernel, "Model", None)
    if model_class is None:
        raise RuntimeError("kernel.py does not define Model")
    inputs_module = _import_from(root / "input.py", "workload_input")
    make_inputs = getattr(inputs_module, "_make_inputs", None)
    if not callable(make_inputs):
        raise RuntimeError("input.py does not define callable _make_inputs(**kwargs)")

    import torch

    model = model_class(**init_kwargs).to(device).eval()
    raw = make_inputs(**input_kwargs)
    if not isinstance(raw, dict):
        raise RuntimeError(f"_make_inputs must return dict[str, Tensor], got {type(raw).__name__}")
    inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in raw.items()
    }
    with torch.no_grad():
        _drive(lambda: model(**inputs), f"Atrex-Bench shape[{shape_id}]")


def _drive(call, label: str) -> None:
    """Warm up, then invoke the candidate PROFILE_ITERS times with a sync at each end."""
    import torch

    # ROCm's torch exposes the same cuda API, so this covers both vendors. A host without an
    # accelerator cannot be profiled anyway; skipping the sync keeps the driver importable there.
    def sync() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    warmup, iters = _iterations()
    for _ in range(warmup):
        call()
    sync()
    print(
        f"[profile_driver] {label}: warmup={warmup} iters={iters}",
        file=sys.stderr,
        flush=True,
    )
    for _ in range(iters):
        call()
    sync()


def main() -> int:
    root = _workspace_root()
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    device = os.environ.get("PROFILE_DEVICE", "cuda:0")
    if (root / "definition.json").is_file() and (root / "workload.jsonl").is_file():
        _run_sol(root, device)
    elif (root / "shapes.json").is_file() and (root / "input.py").is_file():
        _run_atrex_bench(root, device)
    else:
        raise RuntimeError(
            "unrecognized workspace layout: expected definition.json+workload.jsonl (SOL) "
            "or shapes.json+input.py (Atrex-Bench)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
