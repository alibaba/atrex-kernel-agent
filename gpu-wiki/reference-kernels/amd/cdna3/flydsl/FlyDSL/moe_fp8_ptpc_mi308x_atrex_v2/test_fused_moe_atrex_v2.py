# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and profile parity harness for the atrex-open FlyDSL v2 archive."""

import os
import sys
from pathlib import Path

import pytest
import torch
from torch.profiler import ProfilerActivity
from torch.profiler import profile as torch_profile


def _require_aiter_base() -> Path:
    value = os.environ.get("AITER_BASE")
    if not value:
        raise RuntimeError(
            "AITER_BASE must point to an aiter checkout. "
            "Example: export AITER_BASE=/path/to/aiter"
        )
    return Path(value).expanduser()


AITER_BASE = _require_aiter_base()
if (AITER_BASE / "aiter" / "__init__.py").exists():
    aiter_base_str = str(AITER_BASE.resolve())
    if aiter_base_str not in sys.path:
        sys.path.insert(0, aiter_base_str)

from aiter import ActivationType, QuantType, dtypes  # noqa: E402
from aiter.fused_moe import fused_moe as aiter_fused_moe  # noqa: E402
from aiter.fused_moe import fused_topk, torch_moe_stage1, torch_moe_stage2  # noqa: E402
from aiter.ops.quant import get_torch_quant  # noqa: E402
from aiter.ops.shuffle import shuffle_weight  # noqa: E402
from aiter.test_common import checkAllclose  # noqa: E402

from moe_fp8_ptpc_mi308x_atrex_v2.fused_moe_flydsl_fp8 import (  # noqa: E402
    fused_moe_flydsl_fp8_ptpc,
)


M_VALUES = [1, 16, 32, 64, 128, 256, 512]
SHAPE = {
    "name": "task16",
    "E": 512,
    "TOPK": 10,
    "model_dim": 4096,
    "inter_dim": 256,
}
PROFILE_STEPS = [
    "routing",
    "quant",
    "stage1",
    "stage2",
    "fused_1stage",
    "finalize",
    "overhead",
    "other",
]

# Full-pipeline baseline from
# proj011/assets/task_11.../atrex_open_baseline_profile.log.
ATREX_V2_BASELINE_US = {
    1: {"routing": 17.7, "quant": 5.1, "stage1": 11.6, "stage2": 7.9, "overhead": 1.5, "other": 0.0, "kernel_sum": 43.9, "e2e_avg": 544.9, "e2e_min": 528.9},
    16: {"routing": 12.2, "quant": 8.3, "stage1": 91.4, "stage2": 54.6, "overhead": 2.4, "other": 0.0, "kernel_sum": 169.0, "e2e_avg": 601.0, "e2e_min": 575.2},
    32: {"routing": 12.0, "quant": 8.7, "stage1": 120.9, "stage2": 85.3, "overhead": 4.3, "other": 0.0, "kernel_sum": 231.2, "e2e_avg": 587.6, "e2e_min": 563.8},
    64: {"routing": 13.1, "quant": 8.4, "stage1": 180.5, "stage2": 128.6, "overhead": 4.6, "other": 0.0, "kernel_sum": 335.2, "e2e_avg": 663.4, "e2e_min": 620.2},
    128: {"routing": 13.9, "quant": 9.3, "stage1": 244.4, "stage2": 170.6, "overhead": 5.2, "other": 0.0, "kernel_sum": 443.4, "e2e_avg": 735.1, "e2e_min": 714.8},
    256: {"routing": 15.8, "quant": 14.4, "stage1": 275.1, "stage2": 185.5, "overhead": 5.7, "other": 0.0, "kernel_sum": 496.4, "e2e_avg": 796.2, "e2e_min": 782.6},
    512: {"routing": 20.1, "quant": 23.5, "stage1": 334.2, "stage2": 200.0, "overhead": 5.9, "other": 0.0, "kernel_sum": 583.8, "e2e_avg": 938.0, "e2e_min": 925.4},
}

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for fused_moe tests",
)


def flush_cache(size_mb=128, device="cuda:0", dtype=torch.int32, rounds=2):
    n = (size_mb * 1024 * 1024) // torch.tensor([], dtype=dtype).element_size()
    buf = torch.empty(n, device=device, dtype=dtype)
    for _ in range(rounds):
        buf.add_(1)
    torch.cuda.synchronize()
    return buf


def profile_cuda_kernels_ordered(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    with torch_profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        acc_events=True,
    ) as prof:
        for _ in range(iters):
            torch.cuda.synchronize()
            fn()
            torch.cuda.synchronize()

    cuda_events = []
    for evt in prof.events():
        if evt.device_type == torch.autograd.DeviceType.CUDA and evt.device_time > 0:
            cuda_events.append((evt.name, evt.device_time))

    if not cuda_events:
        return []

    kernels_per_iter = len(cuda_events) // iters
    if kernels_per_iter == 0:
        return []

    return [
        cuda_events[i * kernels_per_iter : (i + 1) * kernels_per_iter]
        for i in range(iters)
    ]


def profile_e2e_cuda_events(fn, warmup=5, iters=20, setup_fn=None):
    for _ in range(warmup):
        if setup_fn:
            setup_fn()
        fn()
    torch.cuda.synchronize()

    times_us = []
    for _ in range(iters):
        if setup_fn:
            setup_fn()
            torch.cuda.synchronize()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        fn()
        end_evt.record()
        torch.cuda.synchronize()
        times_us.append(start_evt.elapsed_time(end_evt) * 1000.0)
    return times_us


def _is_flush_cache_kernel(name):
    lower = name.lower()
    return (
        "cudafunctoronself_add<int>" in lower
        or ("vectorized_elementwise_kernel" in lower and "add<int>" in lower)
    )


def _is_overhead_kernel(lower):
    return (
        "memcpy" in lower
        or "fillfunctor" in lower
        or "aten::fill" in lower
        or "fill_kernel" in lower
    )


def classify_kernels_fp8_ptpc(kernel_list):
    steps = {name: 0.0 for name in PROFILE_STEPS}
    generic_gemm_index = 0
    for name, dt in kernel_list:
        lower = name.lower()
        if _is_flush_cache_kernel(name):
            continue
        if _is_overhead_kernel(lower):
            steps["overhead"] += dt
        elif (
            "moesortingkernel" in lower
            or "moe_sorting" in lower
            or "moe sorting" in lower
            or "topk" in lower
            or "routing" in lower
            or "sort" in lower
        ):
            steps["routing"] += dt
        elif (
            "dynamic_per_token_scaled_quant" in lower
            or "smoothquant" in lower
            or "quant" in lower
            or "cast" in lower
        ):
            steps["quant"] += dt
        elif (
            "1stage" in lower
            or "1_stage" in lower
            or "moe_ck1stage" in lower
            or "ck_moe_stage1_stage2" in lower
            or ("fmoe_" in lower and "pertokenfp8" in lower and "stage1" not in lower)
        ):
            steps["fused_1stage"] += dt
        elif (
            "moe_gemm1" in lower
            or "fmoe_stage1" in lower
            or "ck_moe_stage1" in lower
            or "stage1" in lower
            or "gemm1" in lower
        ):
            steps["stage1"] += dt
        elif (
            "moe_gemm2" in lower
            or "moe_ck2stages_gemm2" in lower
            or "ck_moe_stage2" in lower
            or "mulroutedweight" in lower
            or "stage2" in lower
            or "gemm2" in lower
        ):
            steps["stage2"] += dt
        elif "kernel_moe_gemm" in lower or "gridwisemoegemm" in lower:
            if steps["stage1"] > 0 or steps["fused_1stage"] > 0:
                steps["stage2"] += dt
            elif generic_gemm_index == 0:
                steps["stage1"] += dt
            else:
                steps["stage2"] += dt
            generic_gemm_index += 1
        elif "reduce" in lower or "final" in lower:
            steps["finalize"] += dt
        else:
            steps["other"] += dt
    return steps


def _average_steps(per_iter):
    if not per_iter:
        return {name: 0.0 for name in PROFILE_STEPS}
    steps_all = [classify_kernels_fp8_ptpc(kl) for kl in per_iter]
    return {
        step: sum(row[step] for row in steps_all) / len(steps_all)
        for step in PROFILE_STEPS
    }


def _task_style_named_kernel_us(per_iter, *needles):
    totals = {}
    counts = {}
    for kernel_list in per_iter:
        for name, dt in kernel_list:
            lower = name.lower()
            if any(needle in lower for needle in needles):
                totals[name] = totals.get(name, 0.0) + dt
                counts[name] = counts.get(name, 0) + 1
    return sum(totals[name] / counts[name] for name in totals)


def _print_kernel_trace(label, per_iter):
    if not per_iter:
        print(f"{label} kernel trace: <empty>")
        return
    print(f"{label} kernel trace (iter 0, {len(per_iter[0])} kernels):")
    for i, (name, dt) in enumerate(per_iter[0]):
        print(f"  [{i:2d}] {dt:8.1f} us  {name[:100]}")


def _make_fp8_ptpc_env(M, device="cuda:0"):
    dtype = torch.bfloat16
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    E = SHAPE["E"]
    TOPK = SHAPE["TOPK"]
    model_dim = SHAPE["model_dim"]
    inter_dim = SHAPE["inter_dim"]

    hidden_states = torch.empty(
        (M, model_dim), dtype=dtype, device=device
    ).uniform_(-1, 1)
    w1 = torch.empty(
        (E, inter_dim * 2, model_dim), dtype=dtype, device=device
    ).uniform_(-1, 1)
    w2 = torch.empty(
        (E, model_dim, inter_dim), dtype=dtype, device=device
    ).uniform_(-1, 1)
    score = torch.empty((M, E), dtype=dtype, device=device).uniform_(-1, 1)

    topk_weights, topk_ids = fused_topk(hidden_states, score, TOPK, True)
    torch_quant = get_torch_quant(QuantType.per_Token)
    w1_qt, w1_scale = torch_quant(w1, quant_dtype=dtypes.fp8)
    w2_qt, w2_scale = torch_quant(w2, quant_dtype=dtypes.fp8)

    return {
        "hidden_states": hidden_states,
        "w1_qt": w1_qt,
        "w2_qt": w2_qt,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "w1_scale": w1_scale,
        "w2_scale": w2_scale,
        "M": M,
        "dtype": dtype,
        "device": device,
    }


def _make_flydsl_runner(env):
    w1_shuffled = shuffle_weight(env["w1_qt"], layout=(16, 16), use_int4=False)
    w2_shuffled = shuffle_weight(env["w2_qt"], layout=(16, 16), use_int4=False)

    def run():
        return fused_moe_flydsl_fp8_ptpc(
            env["hidden_states"],
            w1_shuffled,
            w2_shuffled,
            env["topk_weights"],
            env["topk_ids"],
            w1_scale=env["w1_scale"],
            w2_scale=env["w2_scale"],
        )

    return run


def _make_aiter_runner(env):
    w1_shuffled = shuffle_weight(env["w1_qt"], layout=(16, 16), use_int4=False)
    w2_shuffled = shuffle_weight(env["w2_qt"], layout=(16, 16), use_int4=False)

    def run():
        return aiter_fused_moe(
            env["hidden_states"],
            w1_shuffled,
            w2_shuffled,
            env["topk_weights"],
            env["topk_ids"],
            activation=ActivationType.Silu,
            quant_type=QuantType.per_Token,
            w1_scale=env["w1_scale"],
            w2_scale=env["w2_scale"],
        )

    return run


def _fused_moe_ref(env):
    torch_quant = get_torch_quant(QuantType.per_Token)
    a1_qt, a1_scale = torch_quant(env["hidden_states"], quant_dtype=dtypes.fp8)
    out1_ref = torch_moe_stage1(
        a1_qt,
        env["w1_qt"],
        env["w2_qt"],
        env["topk_weights"],
        env["topk_ids"],
        dtype=env["dtype"],
        activation=ActivationType.Silu,
        quant_type=QuantType.per_Token,
        a1_scale=a1_scale,
        w1_scale=env["w1_scale"],
        doweight=False,
    )
    a2_qt, a2_scale = torch_quant(out1_ref, quant_dtype=dtypes.fp8)
    a2_qt = a2_qt.view(env["M"], SHAPE["TOPK"], -1)
    return torch_moe_stage2(
        a2_qt,
        env["w1_qt"],
        env["w2_qt"],
        env["topk_weights"],
        env["topk_ids"],
        dtype=env["dtype"],
        quant_type=QuantType.per_Token,
        w2_scale=env["w2_scale"],
        a2_scale=a2_scale,
        doweight=True,
    )


def _profile_runner(label, fn, device, warmup=10, iters=20):
    flush_fn = lambda: flush_cache(128, device=device)

    def profiled_fn():
        flush_cache(128, device=device)
        return fn()

    per_iter = profile_cuda_kernels_ordered(profiled_fn, warmup=warmup, iters=iters)
    _print_kernel_trace(label, per_iter)
    steps = _average_steps(per_iter)
    task_style_steps = {
        "stage1": _task_style_named_kernel_us(per_iter, "moe_gemm1"),
        "stage2": _task_style_named_kernel_us(per_iter, "moe_gemm2"),
    }
    e2e_times = profile_e2e_cuda_events(
        fn, warmup=warmup, iters=iters, setup_fn=flush_fn
    )
    return {
        "steps": steps,
        "task_style_steps": task_style_steps,
        "kernel_sum": sum(steps.values()),
        "e2e_avg": sum(e2e_times) / len(e2e_times),
        "e2e_min": min(e2e_times),
        "per_iter": per_iter,
    }


def _print_profile_table(M, profiles, iters):
    labels = ["aiter", "flydsl_v2"]
    label_names = {"aiter": "AITER", "flydsl_v2": "FlyDSL v2"}
    table_width = 17 + 17 * len(labels)
    print(f"\n{'=' * table_width}")
    print(
        "  FP8 PTPC Comparison "
        f"(shape={SHAPE['name']}, M={M}, E={SHAPE['E']}, "
        f"topk={SHAPE['TOPK']}, model_dim={SHAPE['model_dim']}, "
        f"inter_dim={SHAPE['inter_dim']}, flush_cache 128MB, iters={iters})"
    )
    print(f"{'=' * table_width}")
    header = f"{'Step':>14s}"
    for label in labels:
        header += f" | {label_names[label]:>14s}"
    print(header)
    print("-" * table_width)
    for step in PROFILE_STEPS:
        row = f"{step:>14s}"
        for label in labels:
            row += f" | {profiles[label]['steps'].get(step, 0.0):12.1f}us"
        print(row)
    print("-" * table_width)
    for metric in ("kernel_sum", "e2e_avg", "e2e_min"):
        row = f"{metric.replace('_', ' '):>14s}"
        for label in labels:
            row += f" | {profiles[label][metric]:12.1f}us"
        print(row)
    print("-" * table_width)
    print(
        f"{'speedup(kern)':>14s} | {'1.00x':>14s} | "
        f"{profiles['aiter']['kernel_sum'] / profiles['flydsl_v2']['kernel_sum']:12.2f}x"
    )
    print(
        f"{'speedup(e2e)':>14s} | {'1.00x':>14s} | "
        f"{profiles['aiter']['e2e_avg'] / profiles['flydsl_v2']['e2e_avg']:12.2f}x"
    )


def _profile_kernel_names(profile):
    return [name for kernel_list in profile["per_iter"] for name, _ in kernel_list]


def _assert_v2_trace_and_perf_parity(M, profiles):
    aiter_profile = profiles["aiter"]
    v2_profile = profiles["flydsl_v2"]
    memcpy_names = [
        name for name in _profile_kernel_names(v2_profile) if "memcpy" in name.lower()
    ]
    assert not memcpy_names, (
        "flydsl_v2 profile trace contains unexpected memcpy kernels: "
        + ", ".join(sorted(set(memcpy_names))[:4])
    )
    assert v2_profile["steps"]["other"] == 0.0

    routing_limit = aiter_profile["steps"]["routing"] * 1.15 + 2.0
    assert v2_profile["steps"]["routing"] <= routing_limit, (
        f"flydsl_v2 routing {v2_profile['steps']['routing']:.1f}us exceeds "
        f"AITER-relative limit {routing_limit:.1f}us"
    )

    baseline = ATREX_V2_BASELINE_US[M]
    e2e_limit = baseline["e2e_avg"] * 1.03 + 5.0
    assert v2_profile["e2e_avg"] <= e2e_limit, (
        f"flydsl_v2 e2e_avg {v2_profile['e2e_avg']:.1f}us exceeds "
        f"atrex-open baseline {baseline['e2e_avg']:.1f}us with limit {e2e_limit:.1f}us"
    )
    for stage in ("stage1", "stage2"):
        abs_slack = 6.0 if baseline[stage] < 20.0 else 2.0
        limit = baseline[stage] * 1.03 + abs_slack
        seen = v2_profile["steps"][stage]
        assert seen <= limit, (
            f"flydsl_v2 {stage} {seen:.1f}us exceeds atrex-open baseline "
            f"{baseline[stage]:.1f}us with limit {limit:.1f}us"
        )


@requires_cuda
@pytest.mark.parametrize("M", M_VALUES)
def test_fused_moe_flydsl_v2_correctness(M):
    env = _make_fp8_ptpc_env(M)
    flydsl_out = _make_flydsl_runner(env)()
    ref = _fused_moe_ref(env)
    torch.cuda.synchronize()
    assert not torch.isnan(flydsl_out).any()
    err = checkAllclose(flydsl_out, ref, rtol=1e-02, atol=1e-02)
    assert err <= 0.22, f"checkAllclose failed with error ratio {err:.2%}"


@requires_cuda
@pytest.mark.parametrize("M", M_VALUES)
def test_profile_fp8_ptpc_flydsl_v2_vs_aiter(M):
    device = "cuda:0"
    env = _make_fp8_ptpc_env(M, device=device)
    aiter_fn = _make_aiter_runner(env)
    flydsl_fn = _make_flydsl_runner(env)

    aiter_out = aiter_fn()
    flydsl_out = flydsl_fn()
    torch.cuda.synchronize()
    assert not torch.isnan(aiter_out).any(), "AITER output contains NaN"
    assert not torch.isnan(flydsl_out).any(), "FlyDSL v2 output contains NaN"
    assert aiter_out.shape == flydsl_out.shape
    err = checkAllclose(flydsl_out, aiter_out, rtol=1e-02, atol=1e-02)
    print(f"\nAITER vs FlyDSL v2 checkAllclose error ratio: {err:.2%}")

    warmup, iters = 10, 20
    print(
        f"\n=== Profiling FP8 PTPC: shape={SHAPE['name']}, M={M}, "
        f"E={SHAPE['E']}, topk={SHAPE['TOPK']}, model_dim={SHAPE['model_dim']}, "
        f"inter_dim={SHAPE['inter_dim']}, pipelines=flydsl_v2, "
        f"warmup={warmup}, iters={iters} ==="
    )

    print("\nProfiling AITER...")
    profiles = {"aiter": _profile_runner("AITER", aiter_fn, device, warmup, iters)}
    print("\nProfiling flydsl_v2...")
    profiles["flydsl_v2"] = _profile_runner(
        "flydsl_v2", flydsl_fn, device, warmup, iters
    )

    _print_profile_table(M, profiles, iters)
    assert profiles["aiter"]["per_iter"], "AITER profile produced no CUDA kernels"
    assert profiles["flydsl_v2"]["per_iter"], "FlyDSL v2 profile produced no CUDA kernels"
    assert profiles["aiter"]["e2e_avg"] > 0
    assert profiles["flydsl_v2"]["e2e_avg"] > 0
    assert profiles["flydsl_v2"]["steps"]["stage1"] > 0
    assert profiles["flydsl_v2"]["steps"]["stage2"] > 0
    _assert_v2_trace_and_perf_parity(M, profiles)
