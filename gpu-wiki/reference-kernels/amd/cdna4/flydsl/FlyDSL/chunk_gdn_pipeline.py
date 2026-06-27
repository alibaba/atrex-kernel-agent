#!/usr/bin/env python3
"""FlyDSL chunk-GDN forward pipeline: all 5 kernels wired together."""
from pathlib import Path
import sys

import torch
import time

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

BT = 64
RCP_LN2 = 1.0 / 0.6931471805599453
H_SIZE = 32
Hg_SIZE = 8
DK = 128
DV = 128


def flydsl_pipeline(q, k, v, g, beta, scale):
    """FlyDSL version of the chunk-GDN forward pipeline."""
    from cumsum_kernel import cumsum_fwd
    from kkt_solve_kernel import fused_kkt_solve_fwd
    from recompute_wu_kernel import recompute_w_u_fwd
    from fwd_h_kernel import build_fwd_h
    from fwd_o_kernel import fwd_o_fwd

    B, T, Hg, K = q.shape
    H = v.shape[2]
    V = v.shape[3]
    NT = T // BT

    # 1. cumsum(g) with scale=1/ln2
    g_cumsum = cumsum_fwd(g, scale_val=RCP_LN2)

    # 2. fused_kkt_solve
    A = fused_kkt_solve_fwd(k, g_cumsum, beta)

    # 3. recompute_w_u
    w, u = recompute_w_u_fwd(k, v, beta, A, g_cumsum)

    # 4. fwd_h
    global _fwd_h_fn
    if '_fwd_h_fn' not in globals() or _fwd_h_fn is None:
        _fwd_h_fn = build_fwd_h()
    h = torch.zeros(B, NT, H, K, V, device=k.device, dtype=k.dtype)
    v_new = torch.zeros(B, T, H, V, device=k.device, dtype=k.dtype)
    _fwd_h_fn(k, u, w, v_new, g_cumsum, h, T)

    # 5. fwd_o
    o = fwd_o_fwd(q, k, v_new, h, g_cumsum, scale)

    return o


_fwd_h_fn = None


def bench_fn(fn, warmup=5, repeat=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e6)
    return sum(times) / len(times), min(times)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--T', type=int, default=65536)
    parser.add_argument('--check', action='store_true', help='Run correctness check vs Triton')
    parser.add_argument('--bench', action='store_true', help='Run benchmark')
    args = parser.parse_args()

    B = 1
    T = args.T
    SCALE = DK ** -0.5

    torch.manual_seed(42)
    q = torch.randn(B, T, Hg_SIZE, DK, device='cuda', dtype=torch.bfloat16)
    k = torch.nn.functional.normalize(
        torch.randn(B, T, Hg_SIZE, DK, device='cuda', dtype=torch.bfloat16), p=2, dim=-1)
    v = torch.randn(B, T, H_SIZE, DV, device='cuda', dtype=torch.bfloat16)
    g = torch.nn.functional.logsigmoid(
        torch.randn(B, T, H_SIZE, device='cuda', dtype=torch.bfloat16))
    beta = torch.rand(B, T, H_SIZE, device='cuda', dtype=torch.bfloat16).sigmoid()

    if args.check:
        print("=== Correctness Check ===")
        print("Running Triton pipeline...")
        from bench_standalone import new_pipeline as triton_pipeline
        o_tri, _ = triton_pipeline(q, k, v, g, beta, SCALE)
        torch.cuda.synchronize()

        print("Running FlyDSL pipeline...")
        o_fly = flydsl_pipeline(q, k, v, g, beta, SCALE)
        torch.cuda.synchronize()

        diff = (o_fly.float() - o_tri.float()).abs()
        rel = diff / (o_tri.float().abs() + 1e-8)
        cos = torch.nn.functional.cosine_similarity(
            o_fly.float().flatten(), o_tri.float().flatten(), dim=0)
        print(f"  max_abs_diff:  {diff.max().item():.6e}")
        print(f"  mean_abs_diff: {diff.mean().item():.6e}")
        print(f"  max_rel_err:   {rel.max().item():.6e}")
        print(f"  mean_rel_err:  {rel.mean().item():.6e}")
        print(f"  cosine_sim:    {cos.item():.6f}")
        passed = rel.mean().item() < 1e-2
        print(f"  -> {'PASS' if passed else 'FAIL'}")

    if args.bench:
        print(f"\n=== Benchmark (T={T}) ===")
        # Warmup FlyDSL (compile all kernels)
        print("Compiling FlyDSL kernels...")
        o_fly = flydsl_pipeline(q, k, v, g, beta, SCALE)
        torch.cuda.synchronize()
        print("FlyDSL kernels compiled.")

        avg_fly, min_fly = bench_fn(
            lambda: flydsl_pipeline(q, k, v, g, beta, SCALE))
        print(f"  FlyDSL:  avg={avg_fly:8.0f} us  min={min_fly:8.0f} us")

        # Triton benchmark
        from bench_standalone import new_pipeline as triton_pipeline
        avg_tri, min_tri = bench_fn(
            lambda: triton_pipeline(q, k, v, g, beta, SCALE))
        print(f"  Triton:  avg={avg_tri:8.0f} us  min={min_tri:8.0f} us")
        print(f"  Ratio:   {avg_fly/avg_tri:.2f}x (avg)  {min_fly/min_tri:.2f}x (min)")

    if not args.check and not args.bench:
        print("Use --check for correctness or --bench for benchmark")


if __name__ == '__main__':
    main()
