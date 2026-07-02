#!/usr/bin/env python3
# Copyright 2026 Alibaba Group.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Roofline bottleneck analysis plus compute and bandwidth utilization calculator.

Core idea:
  Roofline analysis is performed at tile/block granularity rather than over
  whole-kernel global FLOPs and bytes. A GPU schedules blocks; each block loads
  its tile from HBM, computes, and stores results independently. Different tile
  sizes produce different arithmetic intensity and may change the bottleneck.

Features:
  1. Build a tile-level Roofline model and classify each tile as compute-bound
     or memory-bound.
  2. Select the proper utilization metric from the bottleneck type:
     - Compute-bound: actual TFLOPS / peak TFLOPS
     - Memory-bound: actual bandwidth / bandwidth ceiling

Supported GPUs (built-in specs):
  NVIDIA Hopper: h100, h20, h200
  AMD CDNA3:     mi300x, mi308x
  AMD CDNA4:     mi355x
  Any other GPU (e.g. Blackwell): pass gpu-wiki-sourced peaks via --peak-tflops and
  --peak-bandwidth-tb-s; the tool never fabricates specs it does not have.

Usage:
    python tools/compute_utilization.py         --gpu h20 --dtype bf16         --flops-expr "2*BM*BN*K" --bytes-expr "(BM*K + BN*K + BM*BN)*2"         --time-ms 0.5 --grid-blocks 64

    python tools/compute_utilization.py         --gpu h100 --dtype bf16         --flops-expr "2*BM*BN*K" --bytes-expr "(BM*K + BN*K + BM*BN)*2"         --time-ms 0.05 --grid-blocks 16         --measured-bandwidth-tb-s 2.8

    python tools/compute_utilization.py         --flops 134217728 --bytes 1212416         --time-ms 0.5 --grid-blocks 64         --gpu mi300x --dtype bf16
"""

import argparse
import sys
import os
import shutil
import importlib.util

# ============================================================
# Unified GPU hardware spec table (NVIDIA Hopper + AMD CDNA3 + AMD CDNA4)
# ============================================================
HARDWARE_SPECS = {
    # ── NVIDIA Hopper ──
    "h100": {
        "fp64_tensor": 33.5,
        "fp32_cuda": 67.0,
        "tf32": 494.7,
        "fp16": 989.4,
        "bf16": 989.4,
        "fp8": 1978.9,
        "int8": 1978.9,
        "memory_bandwidth_tb_s": 3.35,
        "num_units": 132,
        "unit_type": "SM",
        "description": "NVIDIA H100 SXM (sm_90, Hopper)",
    },
    "h20": {
        "fp16": 148.0,
        "bf16": 148.0,
        "fp8": 296.0,
        "int8": 296.0,
        "fp32_cuda": 39.6,
        "memory_bandwidth_tb_s": 4.0,
        "num_units": 78,
        "unit_type": "SM",
        "description": "NVIDIA H20 (sm_90, Hopper)",
    },
    "h200": {
        "fp64_tensor": 33.5,
        "fp32_cuda": 67.0,
        "tf32": 494.7,
        "fp16": 989.4,
        "bf16": 989.4,
        "fp8": 1978.9,
        "int8": 1978.9,
        "memory_bandwidth_tb_s": 4.8,
        "num_units": 132,
        "unit_type": "SM",
        "description": "NVIDIA H200 (sm_90, Hopper, HBM3e)",
    },
    # ── AMD CDNA3 ──
    "mi300x": {
        "fp64_vector": 81.7,
        "fp64_matrix": 163.4,
        "fp32": 163.4,
        "tf32": 653.7,
        "fp16": 1307.4,
        "bf16": 1307.4,
        "fp8": 2614.9,
        "int8": 2614.9,
        "memory_bandwidth_tb_s": 5.3,
        "num_units": 304,
        "unit_type": "CU",
        "description": "AMD Instinct MI300X (gfx942, CDNA3)",
    },
    "mi308x": {
        "fp16": 232.0,
        "bf16": 232.0,
        "fp8": 465.0,
        "int8": 465.0,
        "memory_bandwidth_tb_s": 5.3,
        "num_units": 80,
        "unit_type": "CU",
        "description": "AMD Instinct MI308X (gfx942, CDNA3)",
    },
    # ── AMD CDNA4 ──
    "mi355x": {
        "fp64": 78.6,
        "fp32": 157.3,
        "fp16": 5033.2,
        "bf16": 5033.2,
        "fp8": 10066.4,
        "int8": 10066.4,
        "fp6": 20132.6,
        "fp4": 20132.6,
        "memory_bandwidth_tb_s": 8.0,
        "num_units": 256,
        "unit_type": "CU",
        "description": "AMD Instinct MI355X (gfx950, CDNA4, HBM3e)",
    },
}

# Map dtype names to compute-capability keys across NVIDIA and AMD.
# For dtypes with multiple metrics, such as fp64 tensor/vector/matrix,
# prefer the highest-throughput path by default.
# If a GPU has no mapped key, fall back to the dtype name itself.
DTYPE_TO_COMPUTE = {
    "fp64": ["fp64_tensor", "fp64_matrix", "fp64"],      # NVIDIA tensor > AMD matrix > generic
    "fp32": ["fp32_cuda", "fp32"],                        # NVIDIA CUDA cores > generic
    "tf32": ["tf32"],
    "fp16": ["fp16"],
    "bf16": ["bf16"],
    "fp8": ["fp8"],
    "fp6": ["fp6"],
    "fp4": ["fp4"],
    "int8": ["int8"],
}

# Non-compute fields used when listing supported compute types
_META_KEYS = ("memory_bandwidth_tb_s", "num_units", "unit_type", "description")


def _resolve_compute_type(gpu_specs: dict, dtype: str) -> str:
    """Resolve a dtype to the compute key supported by this GPU."""
    candidates = DTYPE_TO_COMPUTE.get(dtype, [dtype])
    for candidate in candidates:
        if candidate in gpu_specs:
            return candidate
    return dtype  # fallback


def get_peak_tflops(gpu: str, dtype: str) -> float:
    """Return peak compute throughput in TFLOPS."""
    gpu = gpu.lower()
    dtype = dtype.lower()

    if gpu not in HARDWARE_SPECS:
        raise ValueError(
            f"Unknown GPU: {gpu}. Supported GPUs: {list(HARDWARE_SPECS.keys())}"
        )

    specs = HARDWARE_SPECS[gpu]
    compute_type = _resolve_compute_type(specs, dtype)

    if compute_type not in specs:
        raise ValueError(
            f"GPU {gpu} does not support {dtype} ({compute_type}). "
            f"Supported compute types: {[key for key in specs if key not in _META_KEYS]}"
        )

    return specs[compute_type]


def get_peak_bandwidth(gpu: str) -> float:
    """Return peak memory bandwidth in TB/s."""
    gpu = gpu.lower()

    if gpu not in HARDWARE_SPECS:
        raise ValueError(
            f"Unknown GPU: {gpu}. Supported GPUs: {list(HARDWARE_SPECS.keys())}"
        )

    specs = HARDWARE_SPECS[gpu]
    if "memory_bandwidth_tb_s" not in specs:
        raise ValueError(
            f"GPU {gpu} has no configured peak memory bandwidth (memory_bandwidth_tb_s). "
            f"Please add it to HARDWARE_SPECS."
        )

    return specs["memory_bandwidth_tb_s"]


def get_num_units(gpu: str) -> int:
    """Return the number of compute units: SM for NVIDIA, CU for AMD."""
    gpu = gpu.lower()

    if gpu not in HARDWARE_SPECS:
        raise ValueError(
            f"Unknown GPU: {gpu}. Supported GPUs: {list(HARDWARE_SPECS.keys())}"
        )

    specs = HARDWARE_SPECS[gpu]
    if "num_units" not in specs:
        raise ValueError(
            f"GPU {gpu} has no configured compute unit count (num_units). "
            f"Please add it to HARDWARE_SPECS."
        )

    return specs["num_units"]


def get_unit_type(gpu: str) -> str:
    """Return the compute-unit type name, SM or CU."""
    gpu = gpu.lower()
    return HARDWARE_SPECS.get(gpu, {}).get("unit_type", "Unit")


def compute_ridge_point(gpu: str, dtype: str) -> float:
    """
    Compute the Roofline ridge point.

    Ridge Point = peak compute (FLOPS) / peak bandwidth (Bytes/s)
    Unit: FLOPs/Byte
    """
    peak_tflops = get_peak_tflops(gpu, dtype)
    peak_bandwidth_tb_s = get_peak_bandwidth(gpu)
    # TFLOPS / (TB/s) = (1e12 FLOPS) / (1e12 Bytes/s) = FLOPs/Byte
    return peak_tflops / peak_bandwidth_tb_s


def roofline_analysis(flops: float, bytes_transferred: float, gpu: str, dtype: str) -> dict:
    """
    Run Roofline bottleneck analysis.

    Returns:
        dict containing arithmetic_intensity, ridge_point, and bottleneck ("compute" | "memory")
    """
    arithmetic_intensity = flops / bytes_transferred
    ridge_point = compute_ridge_point(gpu, dtype)

    if arithmetic_intensity >= ridge_point:
        bottleneck = "compute"
    else:
        bottleneck = "memory"

    return {
        "arithmetic_intensity": arithmetic_intensity,
        "ridge_point": ridge_point,
        "bottleneck": bottleneck,
        "flops": flops,
        "bytes_transferred": bytes_transferred,
    }


def compute_utilization(flops: float, time_ms: float, gpu: str, dtype: str) -> dict:
    """Compute compute-throughput utilization for compute-bound cases."""
    peak_tflops = get_peak_tflops(gpu, dtype)
    time_s = time_ms / 1000.0
    actual_tflops = flops / time_s / 1e12
    utilization = actual_tflops / peak_tflops * 100.0

    return {
        "flops": flops,
        "time_ms": time_ms,
        "actual_tflops": actual_tflops,
        "peak_tflops": peak_tflops,
        "utilization_pct": utilization,
        "gpu": gpu,
        "dtype": dtype,
    }


def compute_bandwidth_utilization(
    bytes_transferred: float,
    time_ms: float,
    gpu: str,
    measured_bandwidth_tb_s: float = None,
) -> dict:
    """
    Compute bandwidth utilization for memory-bound cases.

    GPUs are high-latency, high-bandwidth devices. Small kernels may not
    have enough data movement to fill the memory pipeline, so they may never
    reach theoretical peak bandwidth. If measured_bandwidth_tb_s is provided,
    use it as the denominator; otherwise fall back to hardware peak bandwidth.
    """
    hardware_peak_bandwidth_tb_s = get_peak_bandwidth(gpu)
    time_s = time_ms / 1000.0
    actual_bandwidth_tb_s = bytes_transferred / time_s / 1e12

    if measured_bandwidth_tb_s is not None:
        bandwidth_ceiling_tb_s = measured_bandwidth_tb_s
        ceiling_source = "measured bandwidth ceiling"
    else:
        bandwidth_ceiling_tb_s = hardware_peak_bandwidth_tb_s
        ceiling_source = "hardware theoretical peak"

    utilization = actual_bandwidth_tb_s / bandwidth_ceiling_tb_s * 100.0

    return {
        "bytes_transferred": bytes_transferred,
        "time_ms": time_ms,
        "actual_bandwidth_tb_s": actual_bandwidth_tb_s,
        "bandwidth_ceiling_tb_s": bandwidth_ceiling_tb_s,
        "hardware_peak_bandwidth_tb_s": hardware_peak_bandwidth_tb_s,
        "ceiling_source": ceiling_source,
        "utilization_pct": utilization,
        "gpu": gpu,
    }


def compute_theoretical_ceiling(
    tile_flops: float,
    tile_bytes: float,
    grid_blocks: int,
    num_units: int,
    gpu: str,
    dtype: str,
    measured_bandwidth_tb_s: float = None,
) -> dict:
    """
    Estimate the theoretical performance ceiling for the current configuration.

    Considers:
      - tile-level Roofline bound type
      - SM/CU utilization, based on grid_blocks vs num_units
      - bandwidth ceiling, measured or theoretical

    Principle:
      The GPU schedules blocks to SMs/CUs in waves. Each wave can run at most
      num_units blocks in parallel. The number of waves is
      ceil(grid_blocks / num_units).

      The minimum per-block time depends on the bottleneck:
        - Compute-bound: tile_time_min = tile_flops / peak_compute
        - Memory-bound:  tile_time_min = tile_bytes / bandwidth_ceiling

      The minimum kernel latency is num_waves * tile_time_min.
      The theoretical ceiling is total FLOPs / minimum kernel latency.
    """
    import math

    peak_tflops = get_peak_tflops(gpu, dtype)
    peak_bandwidth_tb_s = get_peak_bandwidth(gpu)
    unit_type = get_unit_type(gpu)

    # Theoretical performance ceiling
    if measured_bandwidth_tb_s is not None:
        bandwidth_ceiling_tb_s = measured_bandwidth_tb_s
        bandwidth_source = "measured bandwidth ceiling"
    else:
        bandwidth_ceiling_tb_s = peak_bandwidth_tb_s
        bandwidth_source = ""

    # Roofline bound classification
    arithmetic_intensity = tile_flops / tile_bytes
    ridge_point = peak_tflops / peak_bandwidth_tb_s

    if arithmetic_intensity >= ridge_point:
        bottleneck = "compute"
        tile_time_min_s = tile_flops / (peak_tflops * 1e12)
    else:
        bottleneck = "memory"
        tile_time_min_s = tile_bytes / (bandwidth_ceiling_tb_s * 1e12)

    # SM/CU scheduling: number of waves
    num_waves = math.ceil(grid_blocks / num_units)
    unit_utilization_pct = min(grid_blocks / num_units, 1.0) * 100.0

    # Minimum theoretical kernel latency
    theoretical_kernel_time_s = num_waves * tile_time_min_s
    theoretical_kernel_time_ms = theoretical_kernel_time_s * 1000.0

    # Theoretical compute ceiling
    total_flops = tile_flops * grid_blocks
    theoretical_tflops = total_flops / theoretical_kernel_time_s / 1e12

    # Theoretical compute ceiling ()
    total_bytes = tile_bytes * grid_blocks
    theoretical_bandwidth_tb_s = total_bytes / theoretical_kernel_time_s / 1e12

    return {
        "bottleneck": bottleneck,
        "tile_flops": tile_flops,
        "tile_bytes": tile_bytes,
        "arithmetic_intensity": arithmetic_intensity,
        "ridge_point": ridge_point,
        "grid_blocks": grid_blocks,
        "num_units": num_units,
        "unit_type": unit_type,
        "num_waves": num_waves,
        "unit_utilization_pct": unit_utilization_pct,
        "tile_time_min_ms": tile_time_min_s * 1000.0,
        "theoretical_kernel_time_ms": theoretical_kernel_time_ms,
        "theoretical_tflops": theoretical_tflops,
        "theoretical_bandwidth_tb_s": theoretical_bandwidth_tb_s,
        "peak_tflops": peak_tflops,
        "bandwidth_ceiling_tb_s": bandwidth_ceiling_tb_s,
        "bandwidth_source": bandwidth_source,
        "total_flops": total_flops,
        "total_bytes": total_bytes,
        "gpu": gpu,
        "dtype": dtype,
    }


def _print_theoretical_ceiling(ceiling: dict):
    """Print the theoretical performance ceiling."""
    desc = HARDWARE_SPECS[ceiling["gpu"].lower()].get("description", "")
    unit_type = ceiling["unit_type"]
    bottleneck_label = (
        "Compute Bound" if ceiling["bottleneck"] == "compute"
        else "Memory Bound"
    )

    print(f"\n{'='*64}")
    print(f"  Theoretical Performance Ceiling")
    print(f"{'='*64}")
    print(f"  GPU              : {ceiling['gpu'].upper()} ({desc})")
    print(f"  dtype          : {ceiling['dtype']}")
    print(f"  Bottleneck       : {bottleneck_label}")
    print(f"  ─────────────────────────────────────────")
    print(f"  Tile AI          : {ceiling['arithmetic_intensity']:.2f} FLOPs/Byte")
    print(f"  Ridge Point      : {ceiling['ridge_point']:.2f} FLOPs/Byte")
    print(f"  ─────────────────────────────────────────")
    print(f"  Grid blocks      : {ceiling['grid_blocks']}")
    print(f"  {unit_type} count       : {ceiling['num_units']}")
    print(f"  Scheduling waves : {ceiling['num_waves']}")
    print(f"  {unit_type} utilization : {ceiling['unit_utilization_pct']:.1f}%")
    print(f"  ─────────────────────────────────────────")
    print(f"  Per-tile min time : {ceiling['tile_time_min_ms']:.6f} ms")
    print(f"  Kernel min time   : {ceiling['theoretical_kernel_time_ms']:.4f} ms")
    print(f"  ─────────────────────────────────────────")
    print(f"  Compute ceiling  : {ceiling['theoretical_tflops']:.2f} TFLOPS"
          f" (peak {ceiling['peak_tflops']:.1f} TFLOPS)")
    print(f"  Bandwidth ceiling: {ceiling['theoretical_bandwidth_tb_s']:.2f} TB/s"
          f" ({ceiling['bandwidth_source']}: {ceiling['bandwidth_ceiling_tb_s']:.1f} TB/s)")
    print(f"{'='*64}")


def _clear_triton_cache():
    cache_dir = os.path.expanduser("~/.triton")
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)


def measure_kernel_time(kernel_file, wrapper_name, setup_name, warmup=25, rep=100):
    """Measure kernel latency in ms. Requires torch and triton."""
    import torch
    import triton

    spec = importlib.util.spec_from_file_location("kernel_module", kernel_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    wrapper_fn = getattr(module, wrapper_name)
    setup_fn = getattr(module, setup_name)

    captured = {}
    original_wrapper = wrapper_fn

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return original_wrapper(*args, **kwargs)

    setattr(module, wrapper_name, spy)
    try:
        setup_fn()
    except Exception:
        pass

    if not captured:
        raise RuntimeError(f"Unable to capture call arguments for {wrapper_name} from {setup_name}")

    wrapper_fn = getattr(module, wrapper_name)
    torch.cuda.synchronize()
    _clear_triton_cache()
    ms, _, _ = triton.testing.do_bench(
        lambda: wrapper_fn(**captured),
        quantiles=[0.5, 0.2, 0.8],
        warmup=warmup,
        rep=rep,
    )
    return ms


def _eval_expr(expr: str, label: str) -> float:
    """Evaluate a numeric Python expression."""
    try:
        return float(eval(expr))
    except Exception as exc:
        print(f"Error: failed to evaluate {label} expression: {exc}")
        sys.exit(1)


def _print_roofline_result(roofline: dict, gpu: str, dtype: str):
    """Print the tile-level Roofline result."""
    desc = HARDWARE_SPECS[gpu.lower()].get("description", "")
    bottleneck_label = (
        "Compute Bound" if roofline["bottleneck"] == "compute"
        else "Memory Bound"
    )

    print(f"\n{'='*64}")
    print(f"  Tile-level Roofline Analysis")
    print(f"{'='*64}")
    print(f"  GPU              : {gpu.upper()} ({desc})")
    print(f"  dtype          : {dtype}")
    print(f"  Tile FLOPs       : {roofline['flops']:.2e}")
    print(f"  Tile Bytes       : {roofline['bytes_transferred']:.2e}")
    print(f"  Tile AI          : {roofline['arithmetic_intensity']:.2f} FLOPs/Byte")
    print(f"  Ridge Point      : {roofline['ridge_point']:.2f} FLOPs/Byte")
    print(f"  ─────────────────────────────────────────")
    print(f"  Bottleneck       : {bottleneck_label}")
    print(f"{'='*64}")


def _print_compute_utilization(result: dict):
    """Print the compute-utilization result (compute-bound case)."""
    desc = HARDWARE_SPECS[result["gpu"].lower()].get("description", "")

    print(f"\n{'='*64}")
    print(f"  Compute Utilization Analysis (Compute Bound)")
    print(f"{'='*64}")
    print(f"  GPU              : {result['gpu'].upper()} ({desc})")
    print(f"  dtype          : {result['dtype']}")
    print(f"  FLOPs            : {result['flops']:.2e}")
    print(f"  latency              : {result['time_ms']:.4f} ms")
    print(f"  Actual compute   : {result['actual_tflops']:.2f} TFLOPS")
    print(f"  Peak compute     : {result['peak_tflops']:.2f} TFLOPS")
    print(f"  Utilization      : {result['utilization_pct']:.1f}%")
    print(f"{'='*64}")


def _print_bandwidth_utilization(result: dict):
    """Print the bandwidth-utilization result (memory-bound case)."""
    desc = HARDWARE_SPECS[result["gpu"].lower()].get("description", "")

    print(f"\n{'='*64}")
    print(f"  Bandwidth Utilization Analysis (Memory Bound)")
    print(f"{'='*64}")
    print(f"  GPU              : {result['gpu'].upper()} ({desc})")
    print(f"  Bytes transferred: {result['bytes_transferred']:.2e}")
    print(f"  latency              : {result['time_ms']:.4f} ms")
    print(f"  Actual bandwidth : {result['actual_bandwidth_tb_s']:.2f} TB/s"
          f"  ({result['actual_bandwidth_tb_s'] * 1000:.1f} GB/s  <- record as bandwidth_gbps)")
    print(f"  Bandwidth ceiling: {result['bandwidth_ceiling_tb_s']:.2f} TB/s ({result['ceiling_source']})")
    if result["ceiling_source"] == "measured bandwidth ceiling":
        print(f"  Hardware peak    : {result['hardware_peak_bandwidth_tb_s']:.2f} TB/s")
    print(f"  Utilization      : {result['utilization_pct']:.1f}%")
    print(f"{'='*64}")


def _print_exit_status(utilization_pct: float, bottleneck: str):
    """Print final status and return an exit code."""
    metric_name = "compute utilization" if bottleneck == "compute" else "bandwidth utilization"

    if utilization_pct >= 90:
        print(f"✅ {metric_name} reached {utilization_pct:.1f}% (>=90%); no further optimization is required")
        return 0
    else:
        print(f"⚠️  {metric_name} is {utilization_pct:.1f}% (<90%); instruction-level profiling is recommended")
        return 2  # exit code 2 means more optimization is recommended


def main():
    gpu_list = ", ".join(HARDWARE_SPECS.keys())
    parser = argparse.ArgumentParser(
        description="Roofline bottleneck analysis plus compute/bandwidth utilization calculation (NVIDIA + AMD)"
    )
    parser.add_argument("kernel", nargs="?", help="Kernel source file")
    parser.add_argument("--gpu", required=True, help=f"GPU model ({gpu_list})")
    parser.add_argument("--dtype", required=True, help="dtype (bf16, fp16, fp8, fp32, ...)")
    parser.add_argument("--wrapper-name", help="wrapper function name")
    parser.add_argument("--setup-name", help="setup function name")

    parser.add_argument("--flops-expr", help="FLOPs expression (Python expression)")
    parser.add_argument("--flops", type=float, help="FLOPs value")

    parser.add_argument("--bytes-expr", help="Bytes transferred expression (Python expression)")
    parser.add_argument("--bytes", type=float, help="Bytes transferred value")

    parser.add_argument("--measured-bandwidth-tb-s", type=float,
                        help="Measured same-size bandwidth ceiling in TB/s. Use a memcpy kernel with the same data volume as a practical memory-bandwidth baseline.")

    parser.add_argument("--peak-tflops", type=float, default=None,
                        help="Peak compute throughput in TFLOPS for --dtype. REQUIRED for a GPU not in the built-in table (e.g. Blackwell); source it from gpu-wiki. Overrides the built-in value when the GPU is known.")
    parser.add_argument("--peak-bandwidth-tb-s", type=float, default=None,
                        help="Peak HBM bandwidth in TB/s. REQUIRED for a GPU not in the built-in table; source it from gpu-wiki. Overrides the built-in value when the GPU is known.")

    parser.add_argument("--grid-blocks", type=int,
                        help="Number of blocks in the grid. If provided, --time-ms is treated as whole-kernel latency and divided by grid-blocks to derive per-tile latency. If omitted, --flops, --bytes, and --time-ms are assumed to be tile-level values.")

    parser.add_argument("--num-units", type=int, default=None,
                        help="Number of GPU compute units, SM for NVIDIA or CU for AMD. If omitted, infer it from --gpu.")

    parser.add_argument("--time-ms", type=float, help="Latency in ms")
    parser.add_argument("--warmup", type=int, default=25, help="number of warmup iterations")
    parser.add_argument("--rep", type=int, default=100, help="number of repetitions")
    parser.add_argument("--list-gpus", action="store_true", help="List supported GPUs and their specs")
    args = parser.parse_args()

    # ──  GPU  ──
    if args.list_gpus:
        print("Supported GPU models:")
        for gpu_name, specs in HARDWARE_SPECS.items():
            print(f"\n  {gpu_name}: {specs.get('description', '')}")
            for key, val in specs.items():
                if key == "memory_bandwidth_tb_s":
                    print(f"    peak memory bandwidth: {val} TB/s")
                elif key == "num_units":
                    unit_type = specs.get("unit_type", "Unit")
                    print(f"    {unit_type} count: {val}")
                elif key not in ("description", "unit_type"):
                    print(f"    {key}: {val} TFLOPS")
        sys.exit(0)

    # ── latency ──
    if args.time_ms is not None:
        time_ms = args.time_ms
    elif args.kernel and args.wrapper_name and args.setup_name:
        print("Measuring kernel latency...")
        time_ms = measure_kernel_time(
            args.kernel, args.wrapper_name, args.setup_name,
            args.warmup, args.rep
        )
        print(f"  latency: {time_ms:.4f} ms")
    else:
        print("Error: provide --time-ms or provide kernel with --wrapper-name and --setup-name")
        sys.exit(1)

    # ──  FLOPs ──
    if args.flops is not None:
        flops = args.flops
    elif args.flops_expr:
        flops = _eval_expr(args.flops_expr, "FLOPs")
    else:
        print("Error: provide --flops or --flops-expr")
        sys.exit(1)

    # ──  Bytes transferred（） ──
    bytes_transferred = None
    if args.bytes is not None:
        bytes_transferred = args.bytes
    elif args.bytes_expr:
        bytes_transferred = _eval_expr(args.bytes_expr, "Bytes")

    # ──  ──
    gpu = args.gpu.lower()
    dtype = args.dtype.lower()
    grid_blocks = args.grid_blocks

    # Architecture-agnostic peaks: for a GPU not in the built-in table (e.g. Blackwell
    # sm_100/sm_103) the caller supplies gpu-wiki-sourced peaks via --peak-tflops /
    # --peak-bandwidth-tb-s. The tool never invents specs it does not have (no fabrication);
    # when the GPU IS known these flags override the built-in value.
    if args.peak_tflops is not None or args.peak_bandwidth_tb_s is not None:
        spec = dict(HARDWARE_SPECS.get(gpu, {}))
        if args.peak_tflops is not None:
            spec[dtype] = args.peak_tflops
        if args.peak_bandwidth_tb_s is not None:
            spec["memory_bandwidth_tb_s"] = args.peak_bandwidth_tb_s
        if args.num_units is not None:
            spec["num_units"] = args.num_units
        spec.setdefault("unit_type", "SM")
        spec.setdefault("description", f"{args.gpu} (peaks provided via CLI; source: gpu-wiki)")
        HARDWARE_SPECS[gpu] = spec
    elif gpu not in HARDWARE_SPECS:
        print(
            f"Error: unknown GPU '{args.gpu}' and no peaks provided. Either pick one of "
            f"{list(HARDWARE_SPECS.keys())}, or pass gpu-wiki-sourced peaks via "
            f"--peak-tflops and --peak-bandwidth-tb-s (recommended for new architectures)."
        )
        sys.exit(1)

    # ──  ──
    num_units = args.num_units
    if num_units is None:
        try:
            num_units = get_num_units(gpu)
        except ValueError:
            num_units = None

    unit_type = get_unit_type(gpu)

    # ──  per-tile latency ──
    if grid_blocks is not None and grid_blocks > 0:
        per_tile_time_ms = time_ms / grid_blocks
        print(f"\n  Tile : kernel latency {time_ms:.4f} ms / {grid_blocks} blocks "
              f"= {per_tile_time_ms:.6f} ms per tile")
    else:
        per_tile_time_ms = time_ms

    measured_bw = args.measured_bandwidth_tb_s

    if bytes_transferred is not None:
        # Analysis inputs Tile  Roofline 
        roofline = roofline_analysis(flops, bytes_transferred, gpu, dtype)
        _print_roofline_result(roofline, gpu, dtype)

        # ── ceiling ──
        if grid_blocks is not None and num_units is not None:
            ceiling = compute_theoretical_ceiling(
                tile_flops=flops,
                tile_bytes=bytes_transferred,
                grid_blocks=grid_blocks,
                num_units=num_units,
                gpu=gpu,
                dtype=dtype,
                measured_bandwidth_tb_s=measured_bw,
            )
            _print_theoretical_ceiling(ceiling)

        if roofline["bottleneck"] == "compute":
            result = compute_utilization(flops, per_tile_time_ms, gpu, dtype)
            _print_compute_utilization(result)
        else:
            result = compute_bandwidth_utilization(
                bytes_transferred, per_tile_time_ms, gpu, measured_bw
            )
            _print_bandwidth_utilization(result)

        exit_code = _print_exit_status(result["utilization_pct"], roofline["bottleneck"])

        # Analysis inputs: , Output
        if roofline["bottleneck"] == "memory":
            compute_result = compute_utilization(flops, per_tile_time_ms, gpu, dtype)
            print(f"\n  Compute (reference) : {compute_result['actual_tflops']:.2f} / "
                  f"{compute_result['peak_tflops']:.2f} TFLOPS = "
                  f"{compute_result['utilization_pct']:.1f}%")

        # Analysis inputs: , Output
        if roofline["bottleneck"] == "compute":
            bw_result = compute_bandwidth_utilization(
                bytes_transferred, per_tile_time_ms, gpu, measured_bw
            )
            print(f"\n  Bandwidth (reference) : {bw_result['actual_bandwidth_tb_s']:.2f} / "
                  f"{bw_result['bandwidth_ceiling_tb_s']:.2f} TB/s = "
                  f"{bw_result['utilization_pct']:.1f}%")

        # Theoretical compute ceiling, Output vs 
        if grid_blocks is not None and num_units is not None:
            actual_kernel_tflops = flops * grid_blocks / (time_ms / 1000.0) / 1e12
            efficiency = actual_kernel_tflops / ceiling["theoretical_tflops"] * 100.0
            print(f"\n  Actual vs ceiling : {actual_kernel_tflops:.2f} / "
                  f"{ceiling['theoretical_tflops']:.2f} TFLOPS = {efficiency:.1f}%")

        sys.exit(exit_code)

    else:
        # No bytes provided: skip Roofline classification; run compute-utilization only.
        print("\n⚠️  No --bytes/--bytes-expr provided; skipping Roofline classification, "
              "running compute-utilization only.")
        result = compute_utilization(flops, per_tile_time_ms, gpu, dtype)
        _print_compute_utilization(result)
        exit_code = _print_exit_status(result["utilization_pct"], "compute")
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
