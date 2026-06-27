#!/usr/bin/env python3
"""Probe RMSNorm -> MLP input NVFP4 quant parent-boundary fusion.

gpu-wiki archive note:
    Component probe for the SM120 no-PDL RMSNorm+input-quant boundary. It
    validates and times the fused primitive against separate RMSNorm + quant.
    This is reference/probe code, not a production dispatch wrapper.

The component gate compares the current two-kernel path:

    fused_add_rms_norm(input, residual, weight, eps)
    scaled_fp4_quant(input, input_global_scale)

against the new fused primitive:

    fused_add_rms_norm_scaled_fp4_quant(
        input, residual, weight, input_global_scale, eps)

The fused primitive updates residual and directly writes the FP4 payload/scale
consumed by the gate_up GEMM, so it avoids materializing normalized BF16/FP16
hidden states at the RMSNorm -> MLP boundary.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import torch

import vllm._C  # noqa: F401
import vllm._C_stable_libtorch  # noqa: F401
from vllm._custom_ops import create_fp4_output_tensors


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
    fn: Callable[[], object],
    *,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> tuple[dict[str, float], list[float]]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize(device)
        samples.append(float(start.elapsed_time(end)) * 1000.0)
        del out
    return summarize_us(samples), samples


def int_tensor_diff(a: torch.Tensor, b: torch.Tensor) -> dict[str, object]:
    ai = a.to(torch.int32)
    bi = b.to(torch.int32)
    diff = (ai - bi).abs()
    mismatch_count = int((a != b).sum().item())
    total = int(a.numel())
    return {
        "exact": bool(torch.equal(a, b)),
        "max_abs": int(diff.max().item()) if diff.numel() else 0,
        "mismatch_count": mismatch_count,
        "mismatch_ratio": float(mismatch_count / total) if total else 0.0,
        "numel": total,
    }


class Case:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device)
        torch.cuda.set_device(self.device)
        self.dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
        torch.manual_seed(args.seed)

        m = args.m
        h = args.hidden
        self.x = torch.randn((m, h), device=self.device, dtype=self.dtype)
        self.residual = torch.randn((m, h), device=self.device, dtype=self.dtype)
        self.weight = torch.randn((h,), device=self.device, dtype=self.dtype)
        self.input_scale = torch.tensor(
            [args.input_global_scale], device=self.device, dtype=torch.float32
        )

        self.sep_input = self.x.clone()
        self.sep_residual = self.residual.clone()
        self.sep_payload, self.sep_sf = create_fp4_output_tensors(
            m, h, self.device, True
        )

        self.fused_input = self.x.clone()
        self.fused_residual = self.residual.clone()
        self.fused_payload, self.fused_sf = create_fp4_output_tensors(
            m, h, self.device, True
        )

        self.rms_input = self.x.clone()
        self.rms_residual = self.residual.clone()

        self.quant_input = self.x.clone()
        self.quant_residual = self.residual.clone()
        torch.ops._C.fused_add_rms_norm(
            self.quant_input, self.quant_residual, self.weight, args.eps
        )
        self.quant_payload, self.quant_sf = create_fp4_output_tensors(
            m, h, self.device, True
        )
        torch.cuda.synchronize(self.device)

    def separate_rms_then_quant(self) -> None:
        torch.ops._C.fused_add_rms_norm(
            self.sep_input, self.sep_residual, self.weight, self.args.eps
        )
        torch.ops._C.scaled_fp4_quant.out(
            self.sep_input,
            self.input_scale,
            True,
            output=self.sep_payload,
            output_scale=self.sep_sf,
        )

    def fused_rms_quant(self) -> None:
        torch.ops._C.fused_add_rms_norm_scaled_fp4_quant.out(
            self.fused_input,
            self.fused_residual,
            self.weight,
            self.input_scale,
            float(self.args.eps),
            False,
            True,
            output=self.fused_payload,
            output_scale=self.fused_sf,
        )

    def rms_only(self) -> None:
        torch.ops._C.fused_add_rms_norm(
            self.rms_input, self.rms_residual, self.weight, self.args.eps
        )

    def quant_only(self) -> None:
        torch.ops._C.scaled_fp4_quant.out(
            self.quant_input,
            self.input_scale,
            True,
            output=self.quant_payload,
            output_scale=self.quant_sf,
        )

    def correctness(self) -> dict[str, object]:
        sep_input = self.x.clone()
        sep_residual = self.residual.clone()
        sep_payload, sep_sf = create_fp4_output_tensors(
            self.args.m, self.args.hidden, self.device, True
        )
        torch.ops._C.fused_add_rms_norm(
            sep_input, sep_residual, self.weight, self.args.eps
        )
        torch.ops._C.scaled_fp4_quant.out(
            sep_input,
            self.input_scale,
            True,
            output=sep_payload,
            output_scale=sep_sf,
        )

        fused_residual = self.residual.clone()
        fused_payload, fused_sf = create_fp4_output_tensors(
            self.args.m, self.args.hidden, self.device, True
        )
        torch.ops._C.fused_add_rms_norm_scaled_fp4_quant.out(
            self.x,
            fused_residual,
            self.weight,
            self.input_scale,
            float(self.args.eps),
            False,
            True,
            output=fused_payload,
            output_scale=fused_sf,
        )
        torch.cuda.synchronize(self.device)

        residual_diff = (sep_residual.float() - fused_residual.float()).abs()
        norm_ref = sep_input
        norm_ref_payload, norm_ref_sf = sep_payload, sep_sf
        return {
            "residual_exact": bool(torch.equal(sep_residual, fused_residual)),
            "residual_max_abs": float(residual_diff.max().item()),
            "payload": int_tensor_diff(norm_ref_payload, fused_payload),
            "scale_storage_words": int_tensor_diff(norm_ref_sf, fused_sf),
            "scale_storage_bytes": int_tensor_diff(
                norm_ref_sf.view(torch.uint8), fused_sf.view(torch.uint8)
            ),
            "reference_norm_mean_abs": float(norm_ref.float().abs().mean().item()),
        }


def run(args: argparse.Namespace) -> dict[str, object]:
    if not hasattr(torch.ops._C, "fused_add_rms_norm_scaled_fp4_quant"):
        raise RuntimeError("missing fused_add_rms_norm_scaled_fp4_quant op")

    case = Case(args)
    correctness = case.correctness()

    timing_fns = [
        ("separate_fused_add_rms_norm_then_scaled_fp4_quant",
         case.separate_rms_then_quant),
        ("fused_add_rms_norm_scaled_fp4_quant", case.fused_rms_quant),
        ("rms_only_fused_add_rms_norm", case.rms_only),
        ("quant_only_scaled_fp4_quant", case.quant_only),
    ]
    timings: dict[str, dict[str, float]] = {}
    samples: dict[str, list[float]] = {}
    for name, fn in timing_fns:
        summary, raw = time_cuda_us(
            fn, warmup=args.warmup, repeats=args.repeats, device=case.device
        )
        timings[name] = summary
        samples[name] = raw

    separate = timings[
        "separate_fused_add_rms_norm_then_scaled_fp4_quant"
    ]["median_us"]
    fused = timings["fused_add_rms_norm_scaled_fp4_quant"]["median_us"]
    rms_only = timings["rms_only_fused_add_rms_norm"]["median_us"]
    quant_only = timings["quant_only_scaled_fp4_quant"]["median_us"]
    norm_bytes = args.m * args.hidden * torch.empty((), dtype=case.dtype).element_size()
    payload_bytes = case.fused_payload.numel() * case.fused_payload.element_size()
    sf_storage_bytes = case.fused_sf.numel() * case.fused_sf.element_size()

    return {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "rmsnorm_mlp_parent_quant_probe",
        "device": torch.cuda.get_device_name(case.device),
        "shape": {
            "m": args.m,
            "hidden": args.hidden,
            "dtype": args.dtype,
            "eps": args.eps,
            "input_global_scale": args.input_global_scale,
        },
        "warmup": args.warmup,
        "repeats": args.repeats,
        "correctness": correctness,
        "timings": timings,
        "derived": {
            "fused_vs_separate_delta_us": float(fused - separate),
            "fused_vs_separate_ratio": float(fused / separate),
            "fused_speedup_pct": float((separate - fused) / separate * 100.0),
            "separate_component_sum_us": float(rms_only + quant_only),
            "separate_component_sum_minus_measured_us": float(
                rms_only + quant_only - separate
            ),
            "avoided_normalized_hidden_write_read_bytes": int(norm_bytes * 2),
            "normalized_hidden_tensor_bytes": int(norm_bytes),
            "fp4_payload_output_bytes": int(payload_bytes),
            "swizzled_sf_storage_bytes": int(sf_storage_bytes),
        },
        "samples": samples if args.include_samples else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1625)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--input-global-scale", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=200)
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
