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
Kernel latency measurement tool

Measure the execution latency of a single Gluon kernel（p50 median），for compute-utilization calculation and before/after optimization comparison.

Usage:
    python tools/measure_kernel_time.py <kernel.py> \
        --wrapper-name <wrapper_fn> --setup-name <setup_fn>
"""

import argparse
import sys
import os
import shutil
import importlib.util


def import_from_file(file_path, name):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File does not exist: {file_path}")
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    obj = getattr(module, name, None)
    if obj is None:
        raise AttributeError(f"'{name}'  in  {file_path}  was not found in ")
    return obj, module


def capture_wrapper_kwargs(module, wrapper_name, setup_name):
    original_wrapper = getattr(module, wrapper_name, None)
    setup_fn = getattr(module, setup_name, None)

    if original_wrapper is None:
        raise AttributeError(f"'{wrapper_name}'  was not found")
    if setup_fn is None:
        raise AttributeError(f"'{setup_name}'  was not found")

    captured = {}

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return original_wrapper(*args, **kwargs)

    setattr(module, wrapper_name, spy)
    try:
        setup_fn()
    except Exception:
        pass

    if not captured:
        raise RuntimeError(
            f"Unable to capture  {setup_name}  from  {wrapper_name}  call arguments"
        )
    return captured


def _clear_triton_cache():
    cache_dir = os.path.expanduser("~/.triton")
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)


def measure(kernel_file, wrapper_name, setup_name, warmup=25, rep=100):
    import torch
    import triton

    spec = importlib.util.spec_from_file_location("kernel_mod", kernel_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    kwargs = capture_wrapper_kwargs(module, wrapper_name, setup_name)

    wrapper_fn = getattr(module, wrapper_name)
    wrapper_fn(**kwargs)
    torch.cuda.synchronize()

    _clear_triton_cache()
    p50, p20, p80 = triton.testing.do_bench(
        lambda: wrapper_fn(**kwargs),
        quantiles=[0.5, 0.2, 0.8],
        warmup=warmup,
        rep=rep,
    )

    return p50, p20, p80


def main():
    parser = argparse.ArgumentParser(description="Measure Gluon kernel execution latency")
    parser.add_argument("kernel", help="Kernel source file")
    parser.add_argument("--wrapper-name", required=True, help="wrapper function name")
    parser.add_argument("--setup-name", required=True, help="setup function name")
    parser.add_argument("--warmup", type=int, default=25, help="number of warmup iterations (default: 25)")
    parser.add_argument("--rep", type=int, default=100, help="number of repetitions (default: 100)")
    args = parser.parse_args()

    try:
        p50, p20, p80 = measure(
            args.kernel, args.wrapper_name, args.setup_name,
            args.warmup, args.rep
        )
    except Exception as e:
        print(f"❌ measurement failed: {e}")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"  Kernel latency measurement results")
    print(f"{'='*60}")
    print(f"  p20 (fast): {p20:.4f} ms")
    print(f"  p50 (median): {p50:.4f} ms")
    print(f"  p80 (slow): {p80:.4f} ms")
    print(f"{'='*60}")
    print(f"\nFor compute-utilization calculation: --time-ms {p50:.4f}")


if __name__ == "__main__":
    main()
