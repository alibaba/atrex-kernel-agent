#!/usr/bin/env python3
"""Component timing for task28 RMSNorm+quant -> C1 row-ready pipeline.

gpu-wiki archive note:
    Measures the SM120 row-ready wait-cache route. The wiki conclusion is that
    the cache recovers only a small part of the loss and remains operator
    negative.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

TASK23_DIR = (
    Path(__file__).resolve().parents[1] / "task_23_rmsnorm_mlp_parent_fusion"
)
sys.path.insert(0, str(TASK23_DIR))

from row_chunk_parent_probe import Case, time_cuda_us  # noqa: E402
from vllm._custom_ops import (  # noqa: E402
    cutlass_nvfp4_mlp_parent_fused_from_input_fp4,
    fused_add_rms_norm_scaled_fp4_quant,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1625)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--gate-up-output", type=int, default=34816)
    parser.add_argument("--down-output", type=int, default=5120)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--input-global-scale", type=float, default=1.0)
    parser.add_argument("--down-input-global-scale", type=float, default=1.0)
    parser.add_argument("--gate-up-alpha", type=float, default=1.0)
    parser.add_argument("--down-alpha", type=float, default=1.0)
    parser.add_argument("--rms-weight-offset", action="store_true")
    parser.add_argument("--chunk-rows", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, object]:
    os.environ.pop("VLLM_PROJ4_MLP_PARENT_C1_STORE_MODE", None)
    case = Case(args)
    correctness = case.correctness(args.chunk_rows)

    def quant_only():
        case.quant_residual = case.residual.clone()
        return fused_add_rms_norm_scaled_fp4_quant(
            case.input,
            case.quant_residual,
            case.rms_weight,
            case.gate_up_input_global_scale_inv,
            case.args.eps,
            rms_weight_offset=case.args.rms_weight_offset,
            is_sf_swizzled_layout=True,
        )

    case.parent_residual = case.residual.clone()
    parent_input_fp4, parent_input_sf = fused_add_rms_norm_scaled_fp4_quant(
        case.input,
        case.parent_residual,
        case.rms_weight,
        case.gate_up_input_global_scale_inv,
        case.args.eps,
        rms_weight_offset=case.args.rms_weight_offset,
        is_sf_swizzled_layout=True,
    )
    torch.cuda.synchronize(case.device)

    def parent_only():
        return cutlass_nvfp4_mlp_parent_fused_from_input_fp4(
            parent_input_fp4,
            parent_input_sf,
            case.gate_up_weight,
            case.gate_up_weight_scale,
            case.gate_up_alpha,
            case.down_weight,
            case.down_weight_scale,
            case.down_alpha,
            case.down_input_global_scale_inv,
            case.args.gate_up_output,
            case.args.down_output,
            0,
            0,
            case.dtype is torch.bfloat16,
            None,
        )

    def current_full():
        case.current_residual = case.residual.clone()
        return case.timing_current()

    def row_ready_full():
        case.row_chunk_residual = case.residual.clone()
        return case.timing_row_chunk(args.chunk_rows)

    timings: dict[str, object] = {}
    samples: dict[str, list[float]] = {}
    wall: dict[str, object] = {}
    for name, fn in (
        ("baseline_quant_only", quant_only),
        ("baseline_parent_only", parent_only),
        ("baseline_current_full", current_full),
        ("candidate_row_ready_full", row_ready_full),
    ):
        summary, raw, wall_summary = time_cuda_us(
            fn, warmup=args.warmup, repeats=args.repeats, device=case.device
        )
        timings[name] = summary
        samples[name] = raw
        wall[name] = wall_summary

    current = timings["baseline_current_full"]["median_us"]
    candidate = timings["candidate_row_ready_full"]["median_us"]
    quant = timings["baseline_quant_only"]["median_us"]
    parent = timings["baseline_parent_only"]["median_us"]

    return {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "task28_ready_wait_component_probe",
        "env": {
            "VLLM_PROJ4_RMSNORM_MLP_ROW_CHUNK_MODE": os.environ.get(
                "VLLM_PROJ4_RMSNORM_MLP_ROW_CHUNK_MODE"
            ),
            "VLLM_PROJ4_ROW_READY_QUANT_BLOCKS": os.environ.get(
                "VLLM_PROJ4_ROW_READY_QUANT_BLOCKS"
            ),
            "VLLM_PROJ4_ROW_READY_PDL_TRIGGER": os.environ.get(
                "VLLM_PROJ4_ROW_READY_PDL_TRIGGER"
            ),
            "VLLM_PROJ4_ROW_READY_WAIT_MODE": os.environ.get(
                "VLLM_PROJ4_ROW_READY_WAIT_MODE"
            ),
        },
        "device": torch.cuda.get_device_name(case.device),
        "shape": {
            "m": args.m,
            "hidden": args.hidden,
            "gate_up_output": args.gate_up_output,
            "down_output": args.down_output,
            "dtype": args.dtype,
            "chunk_rows": args.chunk_rows,
        },
        "warmup": args.warmup,
        "repeats": args.repeats,
        "correctness": correctness,
        "timings": timings,
        "wall": wall,
        "comparison": {
            "candidate_delta_us": float(candidate - current),
            "candidate_speedup_pct": float((current - candidate) / current * 100.0),
            "quant_plus_parent_us": float(quant + parent),
            "current_vs_quant_plus_parent_us": float(current - (quant + parent)),
        },
        "samples": samples,
    }


def main() -> None:
    args = parse_args()
    result = run(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
