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
Bandwidth baseline measurement script: use torch.copy_ to measure achievable HBM bandwidth for different data sizes.
Used to evaluate bandwidth utilization of CuteDSL kernels.

Usage:
  # Measure bandwidth for a specified data size
  python bench_bandwidth.py --size-bytes 150994944  # 144 MB

  # Measure a range of data sizes
  python bench_bandwidth.py --sweep

  # Customize the sweep range
  python bench_bandwidth.py --sweep --min-mb 1 --max-mb 1024
"""
import torch
import argparse
import math


def measure_memcpy_bw(num_bytes, warmup=20, iters=100):
    """Measure torch.copy_ read bandwidth for the specified data size in GB/s"""
    num_elements = num_bytes // 2  # BF16 = 2 bytes
    try:
        src = torch.randn(num_elements, device="cuda", dtype=torch.bfloat16)
        dst = torch.empty_like(src)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None

    for _ in range(warmup):
        dst.copy_(src)
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        dst.copy_(src)
    e.record()
    torch.cuda.synchronize()

    elapsed_us = s.elapsed_time(e) / iters * 1000
    read_bw = num_bytes / (elapsed_us * 1e-6) / 1e9

    del src, dst
    torch.cuda.empty_cache()
    return elapsed_us, read_bw


def main():
    parser = argparse.ArgumentParser(description="HBM Bandwidth Baseline Measurement")
    parser.add_argument("--size-bytes", type=int, help="Measure bandwidth at this exact size (bytes)")
    parser.add_argument("--size-mb", type=float, help="Measure bandwidth at this size (MB)")
    parser.add_argument("--sweep", action="store_true", help="Sweep a range of sizes")
    parser.add_argument("--min-mb", type=float, default=1, help="Sweep min size (MB)")
    parser.add_argument("--max-mb", type=float, default=512, help="Sweep max size (MB)")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name()}")
    props = torch.cuda.get_device_properties(0)
    print(f"SMs: {props.multi_processor_count}, Compute: {props.major}.{props.minor}")
    print()

    if args.size_bytes:
        us, bw = measure_memcpy_bw(args.size_bytes, args.warmup, args.iters)
        mb = args.size_bytes / (1024 * 1024)
        if bw:
            print(f"Size: {mb:.1f} MB | Time: {us:.1f} us | Read BW: {bw:.0f} GB/s")
        else:
            print(f"Size: {mb:.1f} MB | OOM")

    elif args.size_mb:
        num_bytes = int(args.size_mb * 1024 * 1024)
        us, bw = measure_memcpy_bw(num_bytes, args.warmup, args.iters)
        if bw:
            print(f"Size: {args.size_mb:.1f} MB | Time: {us:.1f} us | Read BW: {bw:.0f} GB/s")
        else:
            print(f"Size: {args.size_mb:.1f} MB | OOM")

    elif args.sweep:
        print(f"{'Size_MB':>10} | {'Time_us':>10} | {'Read_BW':>12}")
        print("-" * 40)

        # Log-spaced sweep
        min_exp = math.log2(args.min_mb)
        max_exp = math.log2(args.max_mb)
        steps = int((max_exp - min_exp) * 2) + 1  # ~2 points per octave

        for i in range(steps):
            mb = 2 ** (min_exp + i * (max_exp - min_exp) / max(steps - 1, 1))
            num_bytes = int(mb * 1024 * 1024)
            us, bw = measure_memcpy_bw(num_bytes, args.warmup, args.iters)
            if bw:
                print(f"{mb:>10.1f} | {us:>10.1f} | {bw:>10.0f} GB/s")
            else:
                print(f"{mb:>10.1f} | {'OOM':>10} |")
                break

    else:
        # Default: sweep common sizes
        print(f"{'Size_MB':>10} | {'Time_us':>10} | {'Read_BW':>12}")
        print("-" * 40)
        for mb in [1, 2, 4, 9, 18, 36, 72, 144, 288, 576]:
            num_bytes = int(mb * 1024 * 1024)
            us, bw = measure_memcpy_bw(num_bytes, args.warmup, args.iters)
            if bw:
                print(f"{mb:>10} | {us:>10.1f} | {bw:>10.0f} GB/s")
            else:
                print(f"{mb:>10} | {'OOM':>10} |")
                break

    print("\nDone.")


if __name__ == "__main__":
    main()
