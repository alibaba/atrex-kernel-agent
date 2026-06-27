#!/usr/bin/env python3
"""Probe task23 row-chunk RMSNorm -> NVFP4 MLP parent pipeline.

gpu-wiki archive note:
    Diagnostic probe for the SM120 row-chunk PDL/parent pipeline. The archived
    result was bit-exact but slower, so use this to study the failure mode and
    not as a deployment recipe.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import torch

import vllm._C  # noqa: F401
import vllm._C_stable_libtorch  # noqa: F401
from vllm._custom_ops import (
    cutlass_nvfp4_mlp_parent_fused_from_input_fp4,
    cutlass_nvfp4_rmsnorm_quant_mlp_parent_row_chunk,
    fused_add_rms_norm_scaled_fp4_quant,
)


def round_up(x: int, y: int) -> int:
    return (x + y - 1) // y * y


def swizzled_sf_shape(m: int, n: int) -> tuple[int, int]:
    return round_up(m, 128), round_up(n // 16, 4)


def fp8_ones(shape: tuple[int, int], device: torch.device) -> torch.Tensor:
    return torch.ones(shape, device=device, dtype=torch.float32).to(torch.float8_e4m3fn)


def summarize_us(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    p90_idx = min(len(ordered) - 1, math.ceil(len(ordered) * 0.9) - 1)
    return {
        "mean_us": float(statistics.fmean(samples)),
        "median_us": float(statistics.median(samples)),
        "std_us": float(statistics.pstdev(samples)),
        "min_us": float(min(samples)),
        "max_us": float(max(samples)),
        "p90_us": float(ordered[p90_idx]),
    }


def time_cuda_us(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> tuple[dict[str, float], list[float], dict[str, float]]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

    gpu_samples: list[float] = []
    wall_samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        wall_start = time.perf_counter_ns()
        out = fn()
        end.record()
        torch.cuda.synchronize(device)
        wall_end = time.perf_counter_ns()
        gpu_samples.append(float(start.elapsed_time(end)) * 1000.0)
        wall_samples.append((wall_end - wall_start) / 1000.0)
        del out
    return summarize_us(gpu_samples), gpu_samples, summarize_us(wall_samples)


def tensor_diff(a: torch.Tensor, b: torch.Tensor) -> dict[str, object]:
    diff = (a.float() - b.float()).abs()
    return {
        "exact": bool(torch.equal(a, b)),
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "numel": int(diff.numel()),
    }


class Case:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device)
        torch.cuda.set_device(self.device)
        self.dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
        torch.manual_seed(args.seed)

        m = args.m
        hidden = args.hidden
        gate_up_output = args.gate_up_output
        act_cols = gate_up_output // 2
        down_output = args.down_output

        self.input = torch.randn((m, hidden), device=self.device, dtype=self.dtype)
        self.residual = torch.randn((m, hidden), device=self.device, dtype=self.dtype)
        self.rms_weight = torch.randn((hidden,), device=self.device, dtype=self.dtype)
        self.gate_up_input_global_scale_inv = torch.tensor(
            [args.input_global_scale],
            device=self.device,
            dtype=torch.float32,
        )
        self.down_input_global_scale_inv = torch.tensor(
            [args.down_input_global_scale],
            device=self.device,
            dtype=torch.float32,
        )
        self.gate_up_alpha = torch.tensor([args.gate_up_alpha], device=self.device)
        self.down_alpha = torch.tensor([args.down_alpha], device=self.device)

        self.gate_up_weight = torch.randint(
            0,
            256,
            (gate_up_output, hidden // 2),
            device=self.device,
            dtype=torch.uint8,
        )
        self.gate_up_weight_scale = fp8_ones(
            swizzled_sf_shape(gate_up_output, hidden), self.device
        )
        self.down_weight = torch.randint(
            0,
            256,
            (down_output, act_cols // 2),
            device=self.device,
            dtype=torch.uint8,
        )
        self.down_weight_scale = fp8_ones(
            swizzled_sf_shape(down_output, act_cols), self.device
        )
        torch.cuda.synchronize(self.device)

    def current(self) -> torch.Tensor:
        input_fp4, input_sf = fused_add_rms_norm_scaled_fp4_quant(
            self.input,
            self.current_residual,
            self.rms_weight,
            self.gate_up_input_global_scale_inv,
            self.args.eps,
            rms_weight_offset=self.args.rms_weight_offset,
            is_sf_swizzled_layout=True,
        )
        return cutlass_nvfp4_mlp_parent_fused_from_input_fp4(
            input_fp4,
            input_sf,
            self.gate_up_weight,
            self.gate_up_weight_scale,
            self.gate_up_alpha,
            self.down_weight,
            self.down_weight_scale,
            self.down_alpha,
            self.down_input_global_scale_inv,
            self.args.gate_up_output,
            self.args.down_output,
            0,
            0,
            self.dtype is torch.bfloat16,
            None,
        )

    def row_chunk(self, chunk_rows: int) -> torch.Tensor:
        return cutlass_nvfp4_rmsnorm_quant_mlp_parent_row_chunk(
            self.input,
            self.row_chunk_residual,
            self.rms_weight,
            self.gate_up_weight,
            self.gate_up_weight_scale,
            self.gate_up_alpha,
            self.gate_up_input_global_scale_inv,
            self.down_weight,
            self.down_weight_scale,
            self.down_alpha,
            self.down_input_global_scale_inv,
            self.args.eps,
            self.args.rms_weight_offset,
            self.args.gate_up_output,
            self.args.down_output,
            0,
            0,
            self.dtype is torch.bfloat16,
            chunk_rows,
            None,
        )

    def correctness(self, chunk_rows: int) -> dict[str, object]:
        self.current_residual = self.residual.clone()
        out_current = self.current()
        residual_current = self.current_residual

        self.row_chunk_residual = self.residual.clone()
        out_row_chunk = self.row_chunk(chunk_rows)
        residual_row_chunk = self.row_chunk_residual
        torch.cuda.synchronize(self.device)

        return {
            "chunk_rows": chunk_rows,
            "output": tensor_diff(out_current, out_row_chunk),
            "residual": tensor_diff(residual_current, residual_row_chunk),
        }

    def timing_current(self) -> torch.Tensor:
        return self.current()

    def timing_row_chunk(self, chunk_rows: int) -> torch.Tensor:
        return self.row_chunk(chunk_rows)


def run(args: argparse.Namespace) -> dict[str, object]:
    os.environ.pop("VLLM_PROJ4_MLP_PARENT_C1_STORE_MODE", None)
    case = Case(args)

    correctness: dict[str, object] = {}
    timings: dict[str, object] = {}
    samples: dict[str, list[float]] = {}

    first_chunk = args.chunk_rows[0]
    correctness["current_vs_row_chunk_first"] = case.correctness(first_chunk)

    case.current_residual = case.residual.clone()
    current_summary, current_samples, current_wall = time_cuda_us(
        case.timing_current,
        warmup=args.warmup,
        repeats=args.repeats,
        device=case.device,
    )
    timings["current_fused_rmsnorm_quant_then_parent_mlp"] = {
        "gpu": current_summary,
        "wall": current_wall,
    }
    if args.include_samples:
        samples["current_fused_rmsnorm_quant_then_parent_mlp"] = current_samples

    for chunk_rows in args.chunk_rows:
        case.row_chunk_residual = case.residual.clone()
        summary, raw, wall = time_cuda_us(
            lambda chunk_rows=chunk_rows: case.timing_row_chunk(chunk_rows),
            warmup=args.warmup,
            repeats=args.repeats,
            device=case.device,
        )
        key = f"row_chunk_{chunk_rows}"
        timings[key] = {"gpu": summary, "wall": wall}
        if args.include_samples:
            samples[key] = raw

    current_median = timings["current_fused_rmsnorm_quant_then_parent_mlp"]["gpu"][
        "median_us"
    ]
    comparisons = {}
    for chunk_rows in args.chunk_rows:
        key = f"row_chunk_{chunk_rows}"
        candidate = timings[key]["gpu"]["median_us"]
        comparisons[key] = {
            "delta_us": float(candidate - current_median),
            "ratio": float(candidate / current_median),
            "speedup_pct": float((current_median - candidate) / current_median * 100.0),
        }

    return {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "rmsnorm_mlp_parent_row_chunk_probe",
        "device": torch.cuda.get_device_name(case.device),
        "shape": {
            "m": args.m,
            "hidden": args.hidden,
            "gate_up_output": args.gate_up_output,
            "down_output": args.down_output,
            "dtype": args.dtype,
            "eps": args.eps,
            "rms_weight_offset": args.rms_weight_offset,
        },
        "warmup": args.warmup,
        "repeats": args.repeats,
        "chunk_rows": args.chunk_rows,
        "correctness": correctness,
        "timings": timings,
        "comparisons": comparisons,
        "samples": samples if args.include_samples else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1625)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--gate-up-output", type=int, default=34816)
    parser.add_argument("--down-output", type=int, default=5120)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--input-global-scale", type=float, default=1.0)
    parser.add_argument("--down-input-global-scale", type=float, default=1.0)
    parser.add_argument("--gate-up-alpha", type=float, default=1.0)
    parser.add_argument("--down-alpha", type=float, default=1.0)
    parser.add_argument("--rms-weight-offset", action="store_true")
    parser.add_argument("--chunk-rows", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-samples", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


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
