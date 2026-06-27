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
Measured bandwidth-ceiling tool

Measure the practical bandwidth ceiling for a specified data volume with a Gluon memcpy kernel.
GPUs are high-latency, high-bandwidth architectures; small data sizes cannot saturate HBM bandwidth,
so a same-size memcpy measurement is needed as the baseline for memory-bound bandwidth-utilization evaluation.

Usage:
    # Measure the bandwidth ceiling for a specified byte count
    python tools/measure_bandwidth_ceiling.py --bytes 1e8 --dtype fp32

    # Measure the bandwidth ceiling for a specified tensor shape (compute bytes automatically)
    python tools/measure_bandwidth_ceiling.py --shape 4096,4096 --dtype bf16

    # Adjust benchmark parameters
    python tools/measure_bandwidth_ceiling.py --bytes 1e8 --dtype fp32 \
        --warmup 50 --rep 200

    # Emit an argument that can be passed directly to compute_utilization.py
    python tools/measure_bandwidth_ceiling.py --bytes 1e8 --dtype fp32 --emit-flag
"""

import argparse
import sys
import os
import shutil
import math


def compute_element_size(dtype: str) -> int:
    """Return the byte size for a dtype"""
    dtype_sizes = {
        "fp64": 8, "float64": 8,
        "fp32": 4, "float32": 4,
        "tf32": 4,
        "fp16": 2, "float16": 2,
        "bf16": 2, "bfloat16": 2,
        "fp8": 1, "float8": 1,
        "int8": 1,
    }
    dtype = dtype.lower()
    if dtype not in dtype_sizes:
        raise ValueError(f"Unknown dtype: {dtype}.Supported: {list(dtype_sizes.keys())}")
    return dtype_sizes[dtype]


def _clear_triton_cache():
    cache_dir = os.path.expanduser("~/.triton")
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)


def measure_bandwidth_ceiling(
    total_bytes: float,
    element_size: int,
    xblock: int = 2048,
    num_warps: int = 4,
    warmup: int = 25,
    rep: int = 100,
) -> dict:
    """
     Gluon memcpy kernel bandwidth ceiling for this data volume.

    Principle: Allocate a tensor of total_bytes and run one full copy with a memcpy kernel,
    measure latency, and compute bandwidth = 2 * total_bytes / time (read + write).

    Parameters:
        total_bytes: data volume to measure (bytes)
        element_size: bytes per element
        xblock: elements handled by each program
        num_warps: warps per program
        warmup: number of benchmark warmup iterations
        rep: number of benchmark repetitions

    Returns:
        dict: contains bandwidth_tb_s, time_ms, total_bytes, and related fields
    """
    import torch
    import triton
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl

    # ── Define memcpy kernel ──
    @gluon.jit
    def _memcpy_kernel(
        in_ptr, out_ptr, xnumel,
        XBLOCK: gl.constexpr,
        layout: gl.constexpr,
    ):
        pid = gl.program_id(0)
        start = pid * XBLOCK
        indices = gl.arange(0, XBLOCK, layout=layout)
        offsets = start + indices
        mask = offsets < xnumel
        value = gl.load(in_ptr + offsets, mask=mask)
        gl.store(out_ptr + offsets, value, mask=mask)
    
    # ── Define memcpy kernel ──
    @gluon.jit
    def _memcpy_kernel_amd(
        in_ptr, out_ptr, xnumel,
        XBLOCK: gl.constexpr,
        layout: gl.constexpr,
    ):
        pid = gl.program_id(0)
        start = pid * XBLOCK
        indices = gl.arange(0, XBLOCK, layout=layout)
        offsets = start + indices
        mask = offsets < xnumel
        value = gl.amd.cdna3.buffer_load(in_ptr, offsets=offsets, mask=mask)
        gl.amd.cdna3.buffer_store(value, out_ptr, offsets=offsets, mask=mask)

    # ── Compute element count ──
    xnumel = int(total_bytes / element_size)
    if xnumel < 1:
        raise ValueError(f"Data volume is too small: {total_bytes} bytes / {element_size} B/elem = {xnumel} elements")

    # ── Select torch dtype ──
    torch_dtype_map = {
        1: torch.int8,
        2: torch.float16,
        4: torch.float32,
        8: torch.float64,
    }
    torch_dtype = torch_dtype_map.get(element_size)
    if torch_dtype is None:
        raise ValueError(f"Unsupported element_size: {element_size}")

    # ── Allocate tensors ──
    input_tensor = torch.empty(xnumel, dtype=torch_dtype, device="cuda")
    output_tensor = torch.empty_like(input_tensor)

    # ── Select the best layout (R = size_per_thread) ──
    # R larger values produce wider instructions, but R * 64 * num_warps <= XBLOCK
    max_r = xblock // (64 * num_warps)
    # use the largest power of two, capped at 16 (dwordx4 ceiling)
    r_value = min(max_r, 16)
    r_value = max(r_value, 1)
    # Ensure it is a power of two
    r_value = 1 << int(math.log2(r_value))

    layout = gl.BlockedLayout([r_value], [64], [num_warps], [0])
    grid = (triton.cdiv(xnumel, xblock),)

    is_amd_backend = getattr(torch.version, "hip", None) is not None
    hardware_backend = "amd" if is_amd_backend else "nvidia"
    memcpy_kernel = _memcpy_kernel_amd if is_amd_backend else _memcpy_kernel

    # ── warmup once to ensure compilation is complete ──
    memcpy_kernel[grid](input_tensor, output_tensor, xnumel, xblock, layout, num_warps=num_warps)
    torch.cuda.synchronize()

    # ── benchmark ──
    benchmark_fn = lambda: memcpy_kernel[grid](
        input_tensor, output_tensor, xnumel, xblock, layout, num_warps=num_warps
    )
    _clear_triton_cache()
    time_ms = triton.testing.do_bench(benchmark_fn, warmup=warmup, rep=rep)

    # ── Compute bandwidth ──
    # read + write = 2 × total_bytes
    actual_bytes = 2 * xnumel * element_size
    time_s = time_ms / 1000.0
    bandwidth_tb_s = actual_bytes / time_s / 1e12

    return {
        "bandwidth_tb_s": bandwidth_tb_s,
        "time_ms": time_ms,
        "total_bytes": xnumel * element_size,
        "actual_bytes_rw": actual_bytes,
        "xnumel": xnumel,
        "element_size": element_size,
        "xblock": xblock,
        "num_warps": num_warps,
        "r_value": r_value,
        "hardware_backend": hardware_backend,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Measure the bandwidth ceiling for this data volume with a Gluon memcpy kernel"
    )

    size_group = parser.add_mutually_exclusive_group(required=True)
    size_group.add_argument(
        "--bytes", type=float,
        help="data volume to measure (bytes),  1e8 = 100MB"
    )
    size_group.add_argument(
        "--shape", type=str,
        help="Tensor shape (),  4096,4096.compute bytes automatically"
    )

    parser.add_argument("--dtype", required=True,
                        help="dtype (fp32, bf16, fp16, fp8, int8)")
    parser.add_argument("--xblock", type=int, default=2048,
                        help="elements handled by each program (default: 2048)")
    parser.add_argument("--num-warps", type=int, default=4,
                        help="warps per program (default: 4)")
    parser.add_argument("--warmup", type=int, default=25,
                        help="number of benchmark warmup iterations (default: 25)")
    parser.add_argument("--rep", type=int, default=100,
                        help="number of benchmark repetitions (default: 100)")
    parser.add_argument("--emit-flag", action="store_true",
                        help="Print the --measured-bandwidth-tb-s argument for compute_utilization.py")

    args = parser.parse_args()

    # ── Compute data volume ──
    element_size = compute_element_size(args.dtype)

    if args.bytes is not None:
        total_bytes = args.bytes
    else:
        dims = [int(d.strip()) for d in args.shape.split(",")]
        total_elements = 1
        for dim in dims:
            total_elements *= dim
        total_bytes = total_elements * element_size

    # ── Run measurement ──
    print(f"Measuring {total_bytes / 1e6:.1f} MB ({args.dtype}) bandwidth ceiling for this data volume...")
    print(f"  XBLOCK={args.xblock}, num_warps={args.num_warps}")

    try:
        result = measure_bandwidth_ceiling(
            total_bytes=total_bytes,
            element_size=element_size,
            xblock=args.xblock,
            num_warps=args.num_warps,
            warmup=args.warmup,
            rep=args.rep,
        )
    except Exception as exc:
        print(f"❌ measurement failed: {exc}")
        sys.exit(1)

    # ── Output results ──
    print(f"\n{'='*64}")
    print(f"  Measured Bandwidth Ceiling Result (Gluon memcpy kernel)")
    print(f"{'='*64}")
    print(f"  hardware backend          : {result['hardware_backend']}")
    print(f"  dtype          : {args.dtype}")
    print(f"  data volume            : {result['total_bytes'] / 1e6:.1f} MB ({result['xnumel']:,} elements)")
    print(f"  total read/write volume          : {result['actual_bytes_rw'] / 1e6:.1f} MB (read + write)")
    print(f"  memcpy latency       : {result['time_ms']:.4f} ms")
    print(f"  measured bandwidth ceiling      : {result['bandwidth_tb_s']:.3f} TB/s")
    print(f"  ─────────────────────────────────────────")
    print(f"  kernel parameters       : XBLOCK={result['xblock']}, R={result['r_value']}, "
          f"num_warps={result['num_warps']}")
    print(f"{'='*64}")

    if args.emit_flag:
        print(f"\nFor compute_utilization.py:")
        print(f"  --measured-bandwidth-tb-s {result['bandwidth_tb_s']:.3f}")


if __name__ == "__main__":
    main()
