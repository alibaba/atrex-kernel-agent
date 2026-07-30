#!/usr/bin/env python3
"""Collect deterministic runtime input signatures for workload dispatch.

This helper runs inside the GPU sandbox.  It uses the same input generators as
the selected evaluator, but never imports or executes a candidate kernel.  The
orchestrator stores the emitted structural signatures and generates a static
dispatcher from them.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch


RESULT_PREFIX = "[dispatch-signatures] RESULT_JSON="


def value_signature(value: Any) -> tuple[Any, ...]:
    """Return a JSON-safe structural signature without reading tensor data."""
    if isinstance(value, torch.Tensor):
        return (
            "tensor",
            tuple(int(item) for item in value.shape),
            tuple(int(item) for item in value.stride()),
            str(value.dtype),
            str(value.layout),
            bool(value.requires_grad),
        )
    if value is None:
        return ("none",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        if math.isnan(value):
            rendered = "nan"
        elif math.isinf(value):
            rendered = "inf" if value > 0 else "-inf"
        else:
            rendered = value.hex()
        return ("float", rendered)
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, torch.dtype):
        return ("torch.dtype", str(value))
    if isinstance(value, tuple):
        return ("tuple", tuple(value_signature(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(value_signature(item) for item in value))
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("dispatch dictionaries must have string keys")
        return (
            "dict",
            tuple((key, value_signature(value[key])) for key in sorted(value)),
        )
    raise TypeError(
        "unsupported dispatch input type: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def invocation_signature(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[Any, ...]:
    return (
        "invocation",
        tuple(value_signature(value) for value in args),
        tuple((key, value_signature(kwargs[key])) for key in sorted(kwargs)),
    )


def _native_signatures(workspace: Path) -> list[dict[str, Any]]:
    runtime_path = (
        workspace / "atrex-bench" / "src" / "atrex_bench" / "eval" / "_runtime.py"
    )
    if not runtime_path.is_file():
        raise FileNotFoundError("workspace is missing the Atrex-Bench evaluator runtime")
    module_name = "atrex_dispatch_runtime"
    spec = importlib.util.spec_from_file_location(module_name, runtime_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load Atrex-Bench runtime from {runtime_path}")
    runtime = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = runtime
    spec.loader.exec_module(runtime)

    shapes = json.loads((workspace / "shapes.json").read_text(encoding="utf-8"))
    if not isinstance(shapes, dict) or not shapes:
        raise ValueError("shapes.json must contain a non-empty object")

    def sort_key(shape_id: str) -> tuple[int, object]:
        return (0, int(shape_id)) if shape_id.isdigit() else (1, shape_id)

    input_module = runtime.import_module_from_path(
        workspace / "input.py", "atrex_dispatch_input"
    )
    runtime.validate_input_module(input_module)
    device = torch.device("cuda")
    records: list[dict[str, Any]] = []
    for index, shape_id in enumerate(sorted(map(str, shapes), key=sort_key)):
        shape = runtime.load_shape_spec(workspace / "reference.py", shape_id)
        init_inputs = runtime.load_shape_init_inputs(shape, device)
        call_inputs = runtime.load_shape_call_inputs(input_module, shape, device)
        records.append(
            {
                "index": index,
                "id": shape_id,
                "init": invocation_signature(init_inputs.args, init_inputs.kwargs),
                "call": invocation_signature(call_inputs.args, call_inputs.kwargs),
            }
        )
        del init_inputs, call_inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return records


def _sol_signatures(workspace: Path) -> list[dict[str, Any]]:
    from sol_execbench.core.bench.io import (  # pylint: disable=import-outside-toplevel
        allocate_outputs,
        gen_inputs,
    )
    from sol_execbench.core.data import (  # pylint: disable=import-outside-toplevel
        Definition,
        Workload,
    )

    definition = Definition(
        **json.loads((workspace / "definition.json").read_text(encoding="utf-8"))
    )
    lines = [
        line
        for line in (workspace / "workload.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        workload = Workload(**json.loads(line))
        inputs = gen_inputs(definition, workload, device="cuda:0")
        axes = definition.get_resolved_axes_values(workload.axes)
        outputs = allocate_outputs(definition, axes, "cuda:0")
        call_args = tuple(inputs) + tuple(outputs)
        records.append(
            {
                "index": index,
                "id": str(index),
                "init": invocation_signature((), {}),
                "call": invocation_signature(call_args, {}),
            }
        )
        del inputs, outputs, call_args
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=".")
    args = parser.parse_args(argv)
    workspace = Path(args.workspace).resolve()

    if (workspace / "definition.json").is_file() and (
        workspace / "workload.jsonl"
    ).is_file():
        kind = "sol"
        records = _sol_signatures(workspace)
        source = "workload.jsonl"
    elif (workspace / "shapes.json").is_file() and (
        workspace / "input.py"
    ).is_file():
        kind = "shapes"
        records = _native_signatures(workspace)
        source = "shapes.json"
    else:
        raise SystemExit("workspace has neither a SOL nor native Atrex-Bench input bundle")

    payload = {
        "schema_version": 1,
        "kind": kind,
        "workload_source": source,
        "workloads": records,
    }
    print(RESULT_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
