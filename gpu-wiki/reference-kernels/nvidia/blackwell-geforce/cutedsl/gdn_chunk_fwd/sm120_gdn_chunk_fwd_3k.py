"""
# ============================================================================
# WARNING: TUNED FOR NVIDIA RTX PRO 5000 (SM120, Blackwell GeForce)
# ============================================================================
#
# CuteDSL DeltaNet Gated Delta Rule — Chunk-Level Forward (Precomputed Neumann)
#
# Production V113 archive. 3-kernel no-cache pipeline: K0 (preprocess), K_inv
# (parallel Neumann), and K1 (sequential fused chunk_h + chunk_o + final_state).
# BT=32, BV=16, 128 threads, cp.async 128-bit for K/Q/M/GQK/V staging.
#
# Tuning vs the generic / V31 baseline:
#   - cp.async 128-bit with val_layout=(1,8) for bf16 (128/16=8 elements)
#   - KQ_STRIDE = K_DIM + 8 = 136 for 16-byte SMEM row alignment
#   - q_norm/k_norm in (B,H,T,K) layout for rank-2 slice alignment
#   - t-outer/v-inner state update loop to reduce register hoisting
#   - K2 (chunk_o) fused inline into K1, eliminating ~250MB GMEM intermediates
#   - LDSM-fed MMA operands plus R2S/STSM stores and direct output copy
#   - scaled-vnew state update: store v_new * exp_decay and reuse transposed sK
#   - V113 reuse-B LDSM: reuse the sS B fragment for K*S and Q*S, and reuse the
#     sNK_A B fragment for four final-state update fragments
#   - V113 non-tail fast path uses B LdMatrix num_matrices=2; tail / STATE_SPLIT
#     keeps the safer B4 launch
#   - K0/Kinv/K1 recompute every execution; no preprocess cache, graph replay, or
#     FLA fallback is part of this accepted path
#
# Tail-aware integration note:
#   - Adds tail-aware K0/K_inv/K1 dispatch for T % BT != 0 without padding the
#     host input tensors.
#   - Keeps the original 3-kernel fast path for T % BT == 0.
#
# Performance at B=1,T=6144,H=16,HV=32,K=128,V=128,bf16, directional input,
# output_final_state=True:
#   - strict same-process Atrex/FLA: 0.531-0.533 ms vs FLA 0.804 ms, 1.51x.
#   - nsys one-call split: K0 78.080 us, Kinv 42.720 us, K1 383.456 us.
#   - ncu counter bandwidth: K0 908.50 GB/s DRAM, Kinv 183.60 GB/s DRAM,
#     K1 269.78 GB/s DRAM and 2.87 TB/s L2.
#
# Related docs:
#   - Optimization journey: docs/ref-docs/nvidia/cutedsl/sm120/sm120-gdn-chunk-fwd-bf16-neumann-optimization.md
#   - Pitfalls: docs/pitfalls/nvidia/cutedsl/gdn-chunk-fwd-pitfalls.md
#   - V31 backup: reference-kernels/nvidia/blackwell-geforce/cutedsl/gdn_chunk_fwd/sm120_gdn_chunk_fwd_3k.v31-0p615ms.bak
# ============================================================================
"""
import functools
import hashlib
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("CUTE_DSL_ARCH", "sm_120a")

_CUTLASS_DSL_PACKAGES = Path(
    "/usr/local/lib/python3.10/dist-packages/nvidia_cutlass_dsl/python_packages"
)
if _CUTLASS_DSL_PACKAGES.exists():
    sys.path.insert(0, str(_CUTLASS_DSL_PACKAGES))

import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils
import cutlass.cutlass_dsl.cutlass as _cutlass_dsl
from cutlass.cute.nvgpu import warp, cpasync
from cutlass.cute.runtime import from_dlpack
from cutlass import pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait


@functools.lru_cache(maxsize=1)
def _stable_cutlass_dsl_version_hash():
    return hashlib.sha256(b"gdn-sm120-local-cutlass-dsl")


# Some nvidia-cutlass-dsl builds walk every submodule for their cache key. That
# can import experimental APIs unrelated to this kernel, so keep the cache hash
# stable locally.
_cutlass_dsl.CutlassBaseDSL.get_version = (
    lambda self: _stable_cutlass_dsl_version_hash()
)

BT = 32
K_DIM = 128
BV = 16
NUM_THREADS = 128
PER_K = (BT * K_DIM) // NUM_THREADS   # 32
PER_V = (BT * BV) // NUM_THREADS      # 4
PER_A = (BT * BT) // NUM_THREADS      # 8


# ============================================================================
# Kernel 0: preprocess_kk
# ============================================================================

@cute.kernel
def preprocess_kk_kernel(
    tiled_mma: cute.TiledMma,
    sK_layout: cute.Layout,
    mQ: cute.Tensor, mK: cute.Tensor,
    mQnorm: cute.Tensor, mKnorm: cute.Tensor,
    mKK: cute.Tensor, mQK: cute.Tensor,
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    NT: cutlass.Constexpr[int],
    USE_FASTMATH: cutlass.Constexpr[bool],
):
    tidx, _, _ = cute.arch.thread_idx()
    chunk_idx = cute.arch.block_idx()[0]
    bid_bh = cute.arch.block_idx()[1]
    i_b = bid_bh // H
    i_h = bid_bh % H
    t0 = chunk_idx * BT

    smem = cutlass.utils.SmemAllocator()
    sK = smem.allocate_tensor(cutlass.BFloat16, sK_layout, 16)
    sQ = smem.allocate_tensor(cutlass.BFloat16, sK_layout, 16)
    thr_mma = tiled_mma.get_slice(tidx)

    for i in cutlass.range_constexpr(PER_K):
        idx = i * NUM_THREADS + tidx
        r = idx // K_DIM
        c = idx % K_DIM
        sK[r, c] = mK[i_b, t0 + r, i_h, c]
    cute.arch.barrier()

    if tidx < BT:
        nk = cutlass.Float32(0.0)
        for j in cutlass.range_constexpr(K_DIM):
            val = cutlass.Float32(sK[tidx, j])
            nk = nk + val * val
        if cutlass.const_expr(USE_FASTMATH):
            ik = cute.rsqrt(nk + cutlass.Float32(1e-6), fastmath=True)
        else:
            ik = cute.rsqrt(nk + cutlass.Float32(1e-6))
        for j in cutlass.range_constexpr(K_DIM):
            sK[tidx, j] = (cutlass.Float32(sK[tidx, j]) * ik).to(cutlass.BFloat16)
    cute.arch.barrier()

    acc = cute.make_rmem_tensor(
        thr_mma.partition_shape_C((BT, BT)), cutlass.Float32)
    acc.fill(cutlass.Float32(0.0))
    tCsKA = thr_mma.partition_A(sK)
    tCsKB = thr_mma.partition_B(sK)
    tCrKA = thr_mma.make_fragment_A(tCsKA)
    tCrKB = thr_mma.make_fragment_B(tCsKB)
    for kk in cutlass.range_constexpr(K_DIM // 16):
        cute.autovec_copy(tCsKA[None, None, kk], tCrKA[None, None, kk])
        cute.autovec_copy(tCsKB[None, None, kk], tCrKB[None, None, kk])
        cute.gemm(tiled_mma, acc, tCrKA[None, None, kk], tCrKB[None, None, kk], acc)

    for i in cutlass.range_constexpr(PER_K):
        idx = i * NUM_THREADS + tidx
        r = idx // K_DIM
        c = idx % K_DIM
        mKnorm[i_b, i_h, t0 + r, c] = sK[r, c]

    cC = cute.make_identity_tensor((BT, BT))
    tCcC = thr_mma.partition_C(cC)
    for idx in cutlass.range(cute.size(acc)):
        co = tCcC[idx]
        mKK[i_b, chunk_idx, i_h, co[0], co[1]] = acc[idx].to(mKK.element_type)

    cute.arch.barrier()

    for i in cutlass.range_constexpr(PER_K):
        idx = i * NUM_THREADS + tidx
        r = idx // K_DIM
        c = idx % K_DIM
        sQ[r, c] = mQ[i_b, t0 + r, i_h, c]
    cute.arch.barrier()

    if tidx < BT:
        nq = cutlass.Float32(0.0)
        for j in cutlass.range_constexpr(K_DIM):
            val = cutlass.Float32(sQ[tidx, j])
            nq = nq + val * val
        if cutlass.const_expr(USE_FASTMATH):
            iq = cute.rsqrt(nq + cutlass.Float32(1e-6), fastmath=True)
        else:
            iq = cute.rsqrt(nq + cutlass.Float32(1e-6))
        for j in cutlass.range_constexpr(K_DIM):
            sQ[tidx, j] = (cutlass.Float32(sQ[tidx, j]) * iq).to(cutlass.BFloat16)
    cute.arch.barrier()

    for i in cutlass.range_constexpr(PER_K):
        idx = i * NUM_THREADS + tidx
        r = idx // K_DIM
        c = idx % K_DIM
        mQnorm[i_b, i_h, t0 + r, c] = sQ[r, c]

    acc.fill(cutlass.Float32(0.0))
    tCsQA = thr_mma.partition_A(sQ)
    tCrQA = thr_mma.make_fragment_A(tCsQA)
    for kk in cutlass.range_constexpr(K_DIM // 16):
        cute.autovec_copy(tCsQA[None, None, kk], tCrQA[None, None, kk])
        cute.autovec_copy(tCsKB[None, None, kk], tCrKB[None, None, kk])
        cute.gemm(tiled_mma, acc, tCrQA[None, None, kk], tCrKB[None, None, kk], acc)

    for idx in cutlass.range(cute.size(acc)):
        co = tCcC[idx]
        mQK[i_b, chunk_idx, i_h, co[0], co[1]] = acc[idx].to(mQK.element_type)


@cute.kernel
def preprocess_kk_only_kernel(
    tiled_mma: cute.TiledMma,
    sK_layout: cute.Layout,
    mQ: cute.Tensor, mK: cute.Tensor,
    mQnorm: cute.Tensor, mKnorm: cute.Tensor, mKK: cute.Tensor,
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    NT: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    chunk_idx = cute.arch.block_idx()[0]
    bid_bh = cute.arch.block_idx()[1]
    i_b = bid_bh // H
    i_h = bid_bh % H
    t0 = chunk_idx * BT

    smem = cutlass.utils.SmemAllocator()
    sK = smem.allocate_tensor(cutlass.BFloat16, sK_layout, 16)
    thr_mma = tiled_mma.get_slice(tidx)

    for i in cutlass.range_constexpr(PER_K):
        idx = i * NUM_THREADS + tidx
        r = idx // K_DIM
        c = idx % K_DIM
        sK[r, c] = mK[i_b, t0 + r, i_h, c]
    cute.arch.barrier()

    if tidx < BT:
        nk = cutlass.Float32(0.0)
        for j in cutlass.range_constexpr(K_DIM):
            val = cutlass.Float32(sK[tidx, j])
            nk = nk + val * val
        ik = cute.rsqrt(nk + cutlass.Float32(1e-6))
        for j in cutlass.range_constexpr(K_DIM):
            sK[tidx, j] = (cutlass.Float32(sK[tidx, j]) * ik).to(cutlass.BFloat16)
    cute.arch.barrier()

    acc = cute.make_rmem_tensor(
        thr_mma.partition_shape_C((BT, BT)), cutlass.Float32)
    acc.fill(cutlass.Float32(0.0))
    tCsKA = thr_mma.partition_A(sK)
    tCsKB = thr_mma.partition_B(sK)
    tCrKA = thr_mma.make_fragment_A(tCsKA)
    tCrKB = thr_mma.make_fragment_B(tCsKB)
    for kk in cutlass.range_constexpr(K_DIM // 16):
        cute.autovec_copy(tCsKA[None, None, kk], tCrKA[None, None, kk])
        cute.autovec_copy(tCsKB[None, None, kk], tCrKB[None, None, kk])
        cute.gemm(tiled_mma, acc, tCrKA[None, None, kk], tCrKB[None, None, kk], acc)

    for i in cutlass.range_constexpr(PER_K):
        idx = i * NUM_THREADS + tidx
        r = idx // K_DIM
        c = idx % K_DIM
        mKnorm[i_b, i_h, t0 + r, c] = sK[r, c]

    cC = cute.make_identity_tensor((BT, BT))
    tCcC = thr_mma.partition_C(cC)
    for idx in cutlass.range(cute.size(acc)):
        co = tCcC[idx]
        mKK[i_b, chunk_idx, i_h, co[0], co[1]] = acc[idx].to(mKK.element_type)

    cute.arch.barrier()

    for i in cutlass.range_constexpr(PER_K):
        idx = i * NUM_THREADS + tidx
        r = idx // K_DIM
        c = idx % K_DIM
        sK[r, c] = mQ[i_b, t0 + r, i_h, c]
    cute.arch.barrier()

    if tidx < BT:
        nq = cutlass.Float32(0.0)
        for j in cutlass.range_constexpr(K_DIM):
            val = cutlass.Float32(sK[tidx, j])
            nq = nq + val * val
        iq = cute.rsqrt(nq + cutlass.Float32(1e-6))
        for j in cutlass.range_constexpr(K_DIM):
            sK[tidx, j] = (cutlass.Float32(sK[tidx, j]) * iq).to(cutlass.BFloat16)
    cute.arch.barrier()

    for i in cutlass.range_constexpr(PER_K):
        idx = i * NUM_THREADS + tidx
        r = idx // K_DIM
        c = idx % K_DIM
        mQnorm[i_b, i_h, t0 + r, c] = sK[r, c]


# ============================================================================
# Kernel INV: precompute Neumann inverse.
# A is strictly lower triangular within BT=32, so summing through A^31 gives
# the exact inverse of (I + A) while keeping the parallel K_inv kernel shape.
# ============================================================================

@cute.jit
def _gdn_exp(
    x,
    USE_FASTMATH: cutlass.Constexpr[bool],
):
    if cutlass.const_expr(USE_FASTMATH):
        return cute.exp(x, fastmath=True)
    return cute.exp(x)


@cute.kernel
def precompute_inv_kernel(
    tiled_mma: cute.TiledMma,
    sA_layout: cute.Layout,
    sA_T_layout: cute.Layout,
    sGC_layout: cute.Layout,
    sBeta_layout: cute.Layout,
    mKK: cute.Tensor,
    mQK: cute.Tensor,
    mGate: cute.Tensor,
    mBeta: cute.Tensor,
    mM: cute.Tensor,
    mGQK: cute.Tensor,
    mExpGC_out: cute.Tensor,
    mExpDecay_out: cute.Tensor,
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    NT: cutlass.Constexpr[int],
    H_PER_HV: cutlass.Constexpr[int],
    FOLD_M_BETA: cutlass.Constexpr[bool],
    USE_FASTMATH: cutlass.Constexpr[bool],
):
    tidx, _, _ = cute.arch.thread_idx()
    chunk_idx = cute.arch.block_idx()[0]
    bid_bhv = cute.arch.block_idx()[1]
    i_b = bid_bhv // HV
    i_hv = bid_bhv % HV
    i_h = i_hv // H_PER_HV
    t0 = chunk_idx * BT

    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(cutlass.Float32, sA_layout, 16)
    sGC = smem.allocate_tensor(cutlass.Float32, sGC_layout, 16)
    sBeta = smem.allocate_tensor(cutlass.Float32, sBeta_layout, 16)
    sInv = smem.allocate_tensor(cutlass.Float32, sA_layout, 16)
    sTmp = smem.allocate_tensor(cutlass.Float32, sA_layout, 16)

    thr_mma = tiled_mma.get_slice(tidx)

    for i in cutlass.range_constexpr(PER_A):
        idx = i * NUM_THREADS + tidx
        row = idx // BT
        col = idx % BT
        sA[row, col] = mKK[i_b, chunk_idx, i_h, row, col]

    if tidx < BT:
        sGC[tidx] = cutlass.Float32(mGate[i_b, t0 + tidx, i_hv])
        sBeta[tidx] = cutlass.Float32(mBeta[i_b, t0 + tidx, i_hv])
    cute.arch.barrier()

    if tidx == 0:
        rs = cutlass.Float32(0.0)
        for t in cutlass.range_constexpr(BT):
            rs = rs + sGC[t]
            sGC[t] = rs
    cute.arch.barrier()
    gc_last = sGC[BT - 1]

    for i in cutlass.range_constexpr(PER_A):
        idx = i * NUM_THREADS + tidx
        row = idx // BT
        col = idx % BT
        factor = cutlass.Float32(0.0)
        if row >= col:
            factor = _gdn_exp(sGC[row] - sGC[col], USE_FASTMATH)
            qk_val = cutlass.Float32(mQK[i_b, chunk_idx, i_h, row, col])
            mGQK[i_b, chunk_idx, i_hv, row, col] = (
                qk_val * factor
            ).to(cutlass.BFloat16)
        else:
            mGQK[i_b, chunk_idx, i_hv, row, col] = cutlass.BFloat16(0.0)
        if row > col:
            a_val = cutlass.Float32(sA[row, col]) * sBeta[row] * factor
            sA[row, col] = a_val
        else:
            sA[row, col] = cutlass.Float32(0.0)

    if tidx < BT:
        mExpGC_out[i_b, chunk_idx, i_hv, tidx] = _gdn_exp(
            sGC[tidx], USE_FASTMATH)
        mExpDecay_out[i_b, chunk_idx, i_hv, tidx] = _gdn_exp(
            gc_last - sGC[tidx], USE_FASTMATH)
    cute.arch.barrier()

    for diag_i in cutlass.range_constexpr(16):
        if tidx < 32:
            block = tidx // 16
            col = tidx % 16
            base = block * 16
            row = base + diag_i
            g_col = base + col
            val = cutlass.Float32(0.0)
            if col < diag_i:
                for k_rel in cutlass.range_constexpr(16):
                    if cutlass.const_expr(k_rel < diag_i):
                        val = val - cutlass.Float32(sA[row, base + k_rel]) * sInv[base + k_rel, g_col]
            elif col == diag_i:
                val = cutlass.Float32(1.0)
            sInv[row, g_col] = val
        cute.arch.barrier()

    for i in cutlass.range_constexpr(2):
        idx = i * NUM_THREADS + tidx
        row = idx // 16
        col = idx % 16
        acc = cutlass.Float32(0.0)
        for kk in cutlass.range_constexpr(16):
            acc = acc + cutlass.Float32(sA[16 + row, kk]) * sInv[kk, col]
        sTmp[row, col] = acc
    cute.arch.barrier()

    for i in cutlass.range_constexpr(2):
        idx = i * NUM_THREADS + tidx
        row = idx // 16
        col = idx % 16
        acc = cutlass.Float32(0.0)
        for kk in cutlass.range_constexpr(16):
            acc = acc + sInv[16 + row, 16 + kk] * sTmp[kk, col]
        sInv[16 + row, col] = -acc
    cute.arch.barrier()

    for i in cutlass.range_constexpr(PER_A):
        idx = i * NUM_THREADS + tidx
        row = idx // BT
        col = idx % BT
        val = cutlass.Float32(0.0)
        if row < 16:
            if col < 16:
                val = sInv[row, col]
        else:
            if col < 16:
                val = sInv[row, col]
            else:
                val = sInv[row, col]
        if cutlass.const_expr(FOLD_M_BETA):
            val = val * sBeta[col]
        mM[i_b, chunk_idx, i_hv, row, col] = val.to(cutlass.BFloat16)

@cute.kernel
def preprocess_kk_tail_kernel(
    tiled_mma: cute.TiledMma,
    sK_layout: cute.Layout,
    mQ: cute.Tensor, mK: cute.Tensor,
    mQnorm: cute.Tensor, mKnorm: cute.Tensor,
    mKK: cute.Tensor, mQK: cute.Tensor,
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    NT: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    chunk_idx = cute.arch.block_idx()[0]
    bid_bh = cute.arch.block_idx()[1]
    i_b = bid_bh // H
    i_h = bid_bh % H
    t0 = chunk_idx * BT

    smem = cutlass.utils.SmemAllocator()
    sK = smem.allocate_tensor(cutlass.BFloat16, sK_layout, 16)
    sQ = smem.allocate_tensor(cutlass.BFloat16, sK_layout, 16)
    thr_mma = tiled_mma.get_slice(tidx)

    for i in cutlass.range_constexpr(PER_K):
        idx = i * NUM_THREADS + tidx
        r = idx // K_DIM
        c = idx % K_DIM
        if chunk_idx == NT - 1:
            if t0 + r < T:
                sK[r, c] = mK[i_b, t0 + r, i_h, c]
            else:
                sK[r, c] = cutlass.BFloat16(0.0)
        else:
            sK[r, c] = mK[i_b, t0 + r, i_h, c]
    cute.arch.barrier()

    if tidx < BT:
        nk = cutlass.Float32(0.0)
        for j in cutlass.range_constexpr(K_DIM):
            val = cutlass.Float32(sK[tidx, j])
            nk = nk + val * val
        ik = cute.rsqrt(nk + cutlass.Float32(1e-6))
        for j in cutlass.range_constexpr(K_DIM):
            sK[tidx, j] = (cutlass.Float32(sK[tidx, j]) * ik).to(cutlass.BFloat16)
    cute.arch.barrier()

    acc = cute.make_rmem_tensor(
        thr_mma.partition_shape_C((BT, BT)), cutlass.Float32)
    acc.fill(cutlass.Float32(0.0))
    tCsKA = thr_mma.partition_A(sK)
    tCsKB = thr_mma.partition_B(sK)
    tCrKA = thr_mma.make_fragment_A(tCsKA)
    tCrKB = thr_mma.make_fragment_B(tCsKB)
    for kk in cutlass.range_constexpr(K_DIM // 16):
        cute.autovec_copy(tCsKA[None, None, kk], tCrKA[None, None, kk])
        cute.autovec_copy(tCsKB[None, None, kk], tCrKB[None, None, kk])
        cute.gemm(tiled_mma, acc, tCrKA[None, None, kk], tCrKB[None, None, kk], acc)

    for i in cutlass.range_constexpr(PER_K):
        idx = i * NUM_THREADS + tidx
        r = idx // K_DIM
        c = idx % K_DIM
        mKnorm[i_b, i_h, t0 + r, c] = sK[r, c]

    cC = cute.make_identity_tensor((BT, BT))
    tCcC = thr_mma.partition_C(cC)
    for idx in cutlass.range(cute.size(acc)):
        co = tCcC[idx]
        mKK[i_b, chunk_idx, i_h, co[0], co[1]] = acc[idx].to(mKK.element_type)

    cute.arch.barrier()

    for i in cutlass.range_constexpr(PER_K):
        idx = i * NUM_THREADS + tidx
        r = idx // K_DIM
        c = idx % K_DIM
        if chunk_idx == NT - 1:
            if t0 + r < T:
                sQ[r, c] = mQ[i_b, t0 + r, i_h, c]
            else:
                sQ[r, c] = cutlass.BFloat16(0.0)
        else:
            sQ[r, c] = mQ[i_b, t0 + r, i_h, c]
    cute.arch.barrier()

    if tidx < BT:
        nq = cutlass.Float32(0.0)
        for j in cutlass.range_constexpr(K_DIM):
            val = cutlass.Float32(sQ[tidx, j])
            nq = nq + val * val
        iq = cute.rsqrt(nq + cutlass.Float32(1e-6))
        for j in cutlass.range_constexpr(K_DIM):
            sQ[tidx, j] = (cutlass.Float32(sQ[tidx, j]) * iq).to(cutlass.BFloat16)
    cute.arch.barrier()

    for i in cutlass.range_constexpr(PER_K):
        idx = i * NUM_THREADS + tidx
        r = idx // K_DIM
        c = idx % K_DIM
        mQnorm[i_b, i_h, t0 + r, c] = sQ[r, c]

    acc.fill(cutlass.Float32(0.0))
    tCsQA = thr_mma.partition_A(sQ)
    tCrQA = thr_mma.make_fragment_A(tCsQA)
    for kk in cutlass.range_constexpr(K_DIM // 16):
        cute.autovec_copy(tCsQA[None, None, kk], tCrQA[None, None, kk])
        cute.autovec_copy(tCsKB[None, None, kk], tCrKB[None, None, kk])
        cute.gemm(tiled_mma, acc, tCrQA[None, None, kk], tCrKB[None, None, kk], acc)

    for idx in cutlass.range(cute.size(acc)):
        co = tCcC[idx]
        mQK[i_b, chunk_idx, i_h, co[0], co[1]] = acc[idx].to(mQK.element_type)


@cute.kernel
def precompute_inv_tail_kernel(
    tiled_mma: cute.TiledMma,
    sA_layout: cute.Layout,
    sA_T_layout: cute.Layout,
    sGC_layout: cute.Layout,
    sBeta_layout: cute.Layout,
    mKK: cute.Tensor,
    mQK: cute.Tensor,
    mGate: cute.Tensor,
    mBeta: cute.Tensor,
    mM: cute.Tensor,
    mGQK: cute.Tensor,
    mExpGC_out: cute.Tensor,
    mExpDecay_out: cute.Tensor,
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    NT: cutlass.Constexpr[int],
    H_PER_HV: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    chunk_idx = cute.arch.block_idx()[0]
    bid_bhv = cute.arch.block_idx()[1]
    i_b = bid_bhv // HV
    i_hv = bid_bhv % HV
    i_h = i_hv // H_PER_HV
    t0 = chunk_idx * BT

    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(cutlass.Float32, sA_layout, 16)
    sGC = smem.allocate_tensor(cutlass.Float32, sGC_layout, 16)
    sBeta = smem.allocate_tensor(cutlass.Float32, sBeta_layout, 16)
    sInv = smem.allocate_tensor(cutlass.Float32, sA_layout, 16)
    sTmp = smem.allocate_tensor(cutlass.Float32, sA_layout, 16)

    thr_mma = tiled_mma.get_slice(tidx)

    for i in cutlass.range_constexpr(PER_A):
        idx = i * NUM_THREADS + tidx
        row = idx // BT
        col = idx % BT
        sA[row, col] = mKK[i_b, chunk_idx, i_h, row, col]

    if tidx < BT:
        if chunk_idx == NT - 1:
            if t0 + tidx < T:
                sGC[tidx] = cutlass.Float32(mGate[i_b, t0 + tidx, i_hv])
                sBeta[tidx] = cutlass.Float32(mBeta[i_b, t0 + tidx, i_hv])
            else:
                sGC[tidx] = cutlass.Float32(0.0)
                sBeta[tidx] = cutlass.Float32(0.0)
        else:
            sGC[tidx] = cutlass.Float32(mGate[i_b, t0 + tidx, i_hv])
            sBeta[tidx] = cutlass.Float32(mBeta[i_b, t0 + tidx, i_hv])
    cute.arch.barrier()

    if tidx == 0:
        rs = cutlass.Float32(0.0)
        for t in cutlass.range_constexpr(BT):
            rs = rs + sGC[t]
            sGC[t] = rs
    cute.arch.barrier()
    gc_last = sGC[BT - 1]

    for i in cutlass.range_constexpr(PER_A):
        idx = i * NUM_THREADS + tidx
        row = idx // BT
        col = idx % BT
        if chunk_idx == NT - 1:
            if row > col and t0 + row < T and t0 + col < T:
                a_val = cutlass.Float32(sA[row, col]) * sBeta[row] * cute.exp(sGC[row] - sGC[col])
                sA[row, col] = a_val
            else:
                sA[row, col] = cutlass.Float32(0.0)
        else:
            if row > col:
                a_val = cutlass.Float32(sA[row, col]) * sBeta[row] * cute.exp(sGC[row] - sGC[col])
                sA[row, col] = a_val
            else:
                sA[row, col] = cutlass.Float32(0.0)

    if tidx < BT:
        mExpGC_out[i_b, chunk_idx, i_hv, tidx] = cute.exp(sGC[tidx])
        mExpDecay_out[i_b, chunk_idx, i_hv, tidx] = cute.exp(gc_last - sGC[tidx])
    cute.arch.barrier()

    for diag_i in cutlass.range_constexpr(16):
        if tidx < 32:
            block = tidx // 16
            col = tidx % 16
            base = block * 16
            row = base + diag_i
            g_col = base + col
            val = cutlass.Float32(0.0)
            if col < diag_i:
                for k_rel in cutlass.range_constexpr(16):
                    if cutlass.const_expr(k_rel < diag_i):
                        val = val - cutlass.Float32(sA[row, base + k_rel]) * sInv[base + k_rel, g_col]
            elif col == diag_i:
                val = cutlass.Float32(1.0)
            sInv[row, g_col] = val
        cute.arch.barrier()

    for i in cutlass.range_constexpr(2):
        idx = i * NUM_THREADS + tidx
        row = idx // 16
        col = idx % 16
        acc = cutlass.Float32(0.0)
        for kk in cutlass.range_constexpr(16):
            acc = acc + cutlass.Float32(sA[16 + row, kk]) * sInv[kk, col]
        sTmp[row, col] = acc
    cute.arch.barrier()

    for i in cutlass.range_constexpr(2):
        idx = i * NUM_THREADS + tidx
        row = idx // 16
        col = idx % 16
        acc = cutlass.Float32(0.0)
        for kk in cutlass.range_constexpr(16):
            acc = acc + sInv[16 + row, 16 + kk] * sTmp[kk, col]
        sInv[16 + row, col] = -acc
    cute.arch.barrier()

    for i in cutlass.range_constexpr(PER_A):
        idx = i * NUM_THREADS + tidx
        row = idx // BT
        col = idx % BT
        val = cutlass.Float32(0.0)
        if row < 16:
            if col < 16:
                val = sInv[row, col]
        else:
            if col < 16:
                val = sInv[row, col]
            else:
                val = sInv[row, col]
        mM[i_b, chunk_idx, i_hv, row, col] = val.to(cutlass.BFloat16)

    for i in cutlass.range_constexpr(PER_A):
        idx = i * NUM_THREADS + tidx
        row = idx // BT
        col = idx % BT
        if chunk_idx == NT - 1:
            if row >= col and t0 + row < T and t0 + col < T:
                qk_val = cutlass.Float32(mQK[i_b, chunk_idx, i_h, row, col])
                mGQK[i_b, chunk_idx, i_hv, row, col] = (
                    qk_val * cute.exp(sGC[row] - sGC[col])
                ).to(cutlass.BFloat16)
            else:
                mGQK[i_b, chunk_idx, i_hv, row, col] = cutlass.BFloat16(0.0)
        else:
            if row >= col:
                qk_val = cutlass.Float32(mQK[i_b, chunk_idx, i_h, row, col])
                mGQK[i_b, chunk_idx, i_hv, row, col] = (
                    qk_val * cute.exp(sGC[row] - sGC[col])
                ).to(cutlass.BFloat16)
            else:
                mGQK[i_b, chunk_idx, i_hv, row, col] = cutlass.BFloat16(0.0)


# ============================================================================
# Kernel 1 helpers
# ============================================================================

@cute.jit
def _mma_AS_full(tiled_mma, thr_mma, sA_op, sS, acc):
    tCsAA = thr_mma.partition_A(sA_op)
    tCsSB = thr_mma.partition_B(sS)
    tCrAA = thr_mma.make_fragment_A(tCsAA)
    tCrSB = thr_mma.make_fragment_B(tCsSB)
    for kk in cutlass.range_constexpr(K_DIM // 16):
        cute.autovec_copy(tCsAA[None, None, kk], tCrAA[None, None, kk])
        cute.autovec_copy(tCsSB[None, None, kk], tCrSB[None, None, kk])
        cute.gemm(tiled_mma, acc, tCrAA[None, None, kk], tCrSB[None, None, kk], acc)


@cute.jit
def _mma_full_autovec(
    tiled_mma,
    thr_mma,
    sA_op,
    sB_op,
    acc,
    K_BLOCKS: cutlass.Constexpr[int],
):
    tCsAA = thr_mma.partition_A(sA_op)
    tCsBB = thr_mma.partition_B(sB_op)
    tCrAA = thr_mma.make_fragment_A(tCsAA)
    tCrBB = thr_mma.make_fragment_B(tCsBB)
    for kk in cutlass.range_constexpr(K_BLOCKS):
        cute.autovec_copy(tCsAA[None, None, kk], tCrAA[None, None, kk])
        cute.autovec_copy(tCsBB[None, None, kk], tCrBB[None, None, kk])
        cute.gemm(tiled_mma, acc, tCrAA[None, None, kk], tCrBB[None, None, kk], acc)


@cute.jit
def _mma_full_ldsm(
    tiled_mma,
    smem_tiled_copy_A,
    smem_tiled_copy_B,
    thr_mma,
    tidx,
    sA_op,
    sB_op,
    acc,
    K_BLOCKS: cutlass.Constexpr[int],
):
    tCsAA = thr_mma.partition_A(sA_op)
    tCsBB = thr_mma.partition_B(sB_op)
    tCrAA = thr_mma.make_fragment_A(tCsAA)
    tCrBB = thr_mma.make_fragment_B(tCsBB)

    thr_copy_A = smem_tiled_copy_A.get_slice(tidx)
    thr_copy_B = smem_tiled_copy_B.get_slice(tidx)
    tCsAA_copy = thr_copy_A.partition_S(sA_op)
    tCsBB_copy = thr_copy_B.partition_S(sB_op)
    tCrAA_copy = thr_copy_A.retile(tCrAA)
    tCrBB_copy = thr_copy_B.retile(tCrBB)

    for kk in cutlass.range_constexpr(K_BLOCKS):
        tCrAA.fill(cutlass.BFloat16(0.0))
        tCrBB.fill(cutlass.BFloat16(0.0))
        tCrAA_copy.fill(cutlass.BFloat16(0.0))
        tCrBB_copy.fill(cutlass.BFloat16(0.0))
        cute.copy(
            smem_tiled_copy_A,
            tCsAA_copy[None, None, kk],
            tCrAA_copy[None, None, kk],
        )
        cute.copy(
            smem_tiled_copy_B,
            tCsBB_copy[None, None, kk],
            tCrBB_copy[None, None, kk],
        )
        cute.gemm(tiled_mma, acc, tCrAA[None, None, kk], tCrBB[None, None, kk], acc)


@cute.jit
def _mma_state4_ldsm_reuse_b(
    tiled_mma,
    smem_tiled_copy_A,
    smem_tiled_copy_B,
    thr_mma,
    tidx,
    sA0_op,
    sA1_op,
    sA2_op,
    sA3_op,
    sB_op,
    acc0,
    acc1,
    acc2,
    acc3,
    K_BLOCKS: cutlass.Constexpr[int],
):
    tCsA0 = thr_mma.partition_A(sA0_op)
    tCsA1 = thr_mma.partition_A(sA1_op)
    tCsA2 = thr_mma.partition_A(sA2_op)
    tCsA3 = thr_mma.partition_A(sA3_op)
    tCsBB = thr_mma.partition_B(sB_op)
    tCrAA = thr_mma.make_fragment_A(tCsA0)
    tCrBB = thr_mma.make_fragment_B(tCsBB)

    thr_copy_A = smem_tiled_copy_A.get_slice(tidx)
    thr_copy_B = smem_tiled_copy_B.get_slice(tidx)
    tCsA0_copy = thr_copy_A.partition_S(sA0_op)
    tCsA1_copy = thr_copy_A.partition_S(sA1_op)
    tCsA2_copy = thr_copy_A.partition_S(sA2_op)
    tCsA3_copy = thr_copy_A.partition_S(sA3_op)
    tCsBB_copy = thr_copy_B.partition_S(sB_op)
    tCrAA_copy = thr_copy_A.retile(tCrAA)
    tCrBB_copy = thr_copy_B.retile(tCrBB)

    for kk in cutlass.range_constexpr(K_BLOCKS):
        cute.copy(
            smem_tiled_copy_B,
            tCsBB_copy[None, None, kk],
            tCrBB_copy[None, None, kk],
        )

        cute.copy(
            smem_tiled_copy_A,
            tCsA0_copy[None, None, kk],
            tCrAA_copy[None, None, kk],
        )
        cute.gemm(tiled_mma, acc0, tCrAA[None, None, kk], tCrBB[None, None, kk], acc0)

        cute.copy(
            smem_tiled_copy_A,
            tCsA1_copy[None, None, kk],
            tCrAA_copy[None, None, kk],
        )
        cute.gemm(tiled_mma, acc1, tCrAA[None, None, kk], tCrBB[None, None, kk], acc1)

        cute.copy(
            smem_tiled_copy_A,
            tCsA2_copy[None, None, kk],
            tCrAA_copy[None, None, kk],
        )
        cute.gemm(tiled_mma, acc2, tCrAA[None, None, kk], tCrBB[None, None, kk], acc2)

        cute.copy(
            smem_tiled_copy_A,
            tCsA3_copy[None, None, kk],
            tCrAA_copy[None, None, kk],
        )
        cute.gemm(tiled_mma, acc3, tCrAA[None, None, kk], tCrBB[None, None, kk], acc3)


@cute.jit
def _mma_kq_s_reuse_b_ldsm(
    tiled_mma,
    smem_tiled_copy_A,
    smem_tiled_copy_B,
    thr_mma,
    tidx,
    sK_op,
    sQ_op,
    sB_op,
    sExpGC,
    tCcC_bv,
    acc_k,
    acc_q,
    K_BLOCKS: cutlass.Constexpr[int],
):
    tCsKA = thr_mma.partition_A(sK_op)
    tCsQA = thr_mma.partition_A(sQ_op)
    tCsBB = thr_mma.partition_B(sB_op)
    tCrAA = thr_mma.make_fragment_A(tCsKA)
    tCrBB = thr_mma.make_fragment_B(tCsBB)

    thr_copy_A = smem_tiled_copy_A.get_slice(tidx)
    thr_copy_B = smem_tiled_copy_B.get_slice(tidx)
    tCsKA_copy = thr_copy_A.partition_S(sK_op)
    tCsQA_copy = thr_copy_A.partition_S(sQ_op)
    tCsBB_copy = thr_copy_B.partition_S(sB_op)
    tCrAA_copy = thr_copy_A.retile(tCrAA)
    tCrBB_copy = thr_copy_B.retile(tCrBB)

    for kk in cutlass.range_constexpr(K_BLOCKS):
        cute.copy(
            smem_tiled_copy_B,
            tCsBB_copy[None, None, kk],
            tCrBB_copy[None, None, kk],
        )

        cute.copy(
            smem_tiled_copy_A,
            tCsKA_copy[None, None, kk],
            tCrAA_copy[None, None, kk],
        )
        cute.gemm(tiled_mma, acc_k, tCrAA[None, None, kk], tCrBB[None, None, kk], acc_k)

        cute.copy(
            smem_tiled_copy_A,
            tCsQA_copy[None, None, kk],
            tCrAA_copy[None, None, kk],
        )
        cute.gemm(tiled_mma, acc_q, tCrAA[None, None, kk], tCrBB[None, None, kk], acc_q)
    for idx in cutlass.range(cute.size(acc_q)):
        co = tCcC_bv[idx]
        acc_q[idx] = acc_q[idx] * sExpGC[co[0]]


@cute.jit
def _store_state_frag_r2s(
    tiled_copy_state_r2s,
    tidx,
    state_frag,
    sS_tile,
):
    thr_copy = tiled_copy_state_r2s.get_slice(tidx)
    tRS_sS = thr_copy.partition_D(sS_tile)
    tRS_rState = tiled_copy_state_r2s.retile(state_frag)
    tRS_rState_bf16 = cute.make_rmem_tensor(cute.shape(tRS_rState), cutlass.BFloat16)
    for idx in cutlass.range(cute.size(tRS_rState_bf16)):
        tRS_rState_bf16[idx] = tRS_rState[idx].to(cutlass.BFloat16)
    cute.copy(tiled_copy_state_r2s, tRS_rState_bf16, tRS_sS)


@cute.jit
def _store_state_frag_lo_r2s(
    tiled_copy_state_r2s,
    tidx,
    state_frag,
    sS_lo_tile,
):
    thr_copy = tiled_copy_state_r2s.get_slice(tidx)
    tRS_sS_lo = thr_copy.partition_D(sS_lo_tile)
    tRS_rState = tiled_copy_state_r2s.retile(state_frag)
    tRS_rState_lo = cute.make_rmem_tensor(cute.shape(tRS_rState), cutlass.BFloat16)
    for idx in cutlass.range(cute.size(tRS_rState_lo)):
        hi = tRS_rState[idx].to(cutlass.BFloat16)
        tRS_rState_lo[idx] = (tRS_rState[idx] - cutlass.Float32(hi)).to(cutlass.BFloat16)
    cute.copy(tiled_copy_state_r2s, tRS_rState_lo, tRS_sS_lo)


@cute.jit
def _add_qs_state_lo_rows_scalar(
    acc_qS,
    tCcC_bv,
    sQ,
    sS_lo,
):
    for idx in cutlass.range(cute.size(acc_qS)):
        co = tCcC_bv[idx]
        row = co[0]
        v_col = co[1]
        if row >= 16 and row < 18:
            corr = cutlass.Float32(0.0)
            for k_col in cutlass.range_constexpr(32):
                corr = corr + cutlass.Float32(sQ[row, k_col]) * cutlass.Float32(sS_lo[v_col, k_col])
            acc_qS[idx] = acc_qS[idx] + corr


@cute.jit
def _store_acc_o_gmem_vector(
    tiled_copy_o_gmem,
    tidx,
    acc_o,
    gO_tile,
    scale: cutlass.Constexpr[float],
):
    thr_copy = tiled_copy_o_gmem.get_slice(tidx)
    tOgO = thr_copy.partition_D(gO_tile)
    tOrO = tiled_copy_o_gmem.retile(acc_o)
    tOrO_bf16 = cute.make_rmem_tensor(cute.shape(tOrO), gO_tile.element_type)
    for idx in cutlass.range(cute.size(tOrO_bf16)):
        tOrO_bf16[idx] = (scale * tOrO[idx]).to(gO_tile.element_type)
    cute.copy(tiled_copy_o_gmem, tOrO_bf16, tOgO)


@cute.jit
def _store_kdecay_scratch_r2s(
    tiled_copy_kdecay_r2s,
    tidx,
    sK,
    sExpDecay,
    sScratch_tile,
    K_BASE: cutlass.Constexpr[int],
    T_BASE: cutlass.Constexpr[int],
):
    thr_copy = tiled_copy_kdecay_r2s.get_slice(tidx)
    tRS_sScratch = thr_copy.partition_D(sScratch_tile)
    cScratch = cute.make_identity_tensor((BT, BV))
    tRS_cScratch = thr_copy.partition_S(cScratch)
    tRS_rScratch = cute.make_rmem_tensor(cute.shape(tRS_sScratch), cutlass.BFloat16)
    for idx in cutlass.range(cute.size(tRS_rScratch)):
        co = tRS_cScratch[idx]
        row = co[0]
        col = co[1]
        t = T_BASE + col
        k_col = K_BASE + row
        tRS_rScratch[idx] = (
            cutlass.Float32(sK[t, k_col]) * sExpDecay[t]
        ).to(cutlass.BFloat16)
    cute.copy(tiled_copy_kdecay_r2s, tRS_rScratch, tRS_sScratch)


@cute.jit
def _store_rhs_snk_r2s(
    tiled_copy_rhs_r2s,
    tidx,
    acc_kS,
    sV,
    sBeta,
    sExpGC,
    sNK_tile,
):
    thr_copy = tiled_copy_rhs_r2s.get_slice(tidx)
    tRS_sNK = thr_copy.partition_D(sNK_tile)
    tRS_acc = tiled_copy_rhs_r2s.retile(acc_kS)
    cNK = cute.make_identity_tensor((BT, BV))
    tRS_cNK = thr_copy.partition_S(cNK)
    tRS_rNK = cute.make_rmem_tensor(cute.shape(tRS_acc), cutlass.BFloat16)
    for idx in cutlass.range(cute.size(tRS_rNK)):
        co = tRS_cNK[idx]
        row = co[0]
        v_col = co[1]
        rhs = sBeta[row] * (
            cutlass.Float32(sV[row, v_col]) - sExpGC[row] * tRS_acc[idx]
        )
        tRS_rNK[idx] = rhs.to(cutlass.BFloat16)
    cute.copy(tiled_copy_rhs_r2s, tRS_rNK, tRS_sNK)


@cute.jit
def _store_rhs_snk_r2s_nobeta(
    tiled_copy_rhs_r2s,
    tidx,
    acc_kS,
    sV,
    sExpGC,
    sNK_tile,
):
    thr_copy = tiled_copy_rhs_r2s.get_slice(tidx)
    tRS_sNK = thr_copy.partition_D(sNK_tile)
    tRS_acc = tiled_copy_rhs_r2s.retile(acc_kS)
    cNK = cute.make_identity_tensor((BT, BV))
    tRS_cNK = thr_copy.partition_S(cNK)
    tRS_rNK = cute.make_rmem_tensor(cute.shape(tRS_acc), cutlass.BFloat16)
    for idx in cutlass.range(cute.size(tRS_rNK)):
        co = tRS_cNK[idx]
        row = co[0]
        v_col = co[1]
        rhs = cutlass.Float32(sV[row, v_col]) - sExpGC[row] * tRS_acc[idx]
        tRS_rNK[idx] = rhs.to(cutlass.BFloat16)
    cute.copy(tiled_copy_rhs_r2s, tRS_rNK, tRS_sNK)


@cute.jit
def _store_rhs_snk_r2s_direct_v(
    tiled_copy_rhs_r2s,
    tidx,
    acc_kS,
    gV_tile,
    sBeta,
    sExpGC,
    sNK_tile,
):
    thr_copy = tiled_copy_rhs_r2s.get_slice(tidx)
    tRS_sNK = thr_copy.partition_D(sNK_tile)
    tRS_acc = tiled_copy_rhs_r2s.retile(acc_kS)
    cNK = cute.make_identity_tensor((BT, BV))
    tRS_cNK = thr_copy.partition_S(cNK)
    tRS_rNK = cute.make_rmem_tensor(cute.shape(tRS_acc), cutlass.BFloat16)
    for idx in cutlass.range(cute.size(tRS_rNK)):
        co = tRS_cNK[idx]
        row = co[0]
        v_col = co[1]
        rhs = sBeta[row] * (
            cutlass.Float32(gV_tile[row, v_col]) - sExpGC[row] * tRS_acc[idx]
        )
        tRS_rNK[idx] = rhs.to(cutlass.BFloat16)
    cute.copy(tiled_copy_rhs_r2s, tRS_rNK, tRS_sNK)


@cute.jit
def _store_scaled_vnew_snk_r2s(
    tiled_copy_state_r2s,
    tidx,
    acc_vnew,
    sExpDecay,
    sNK_tile,
):
    thr_copy = tiled_copy_state_r2s.get_slice(tidx)
    tRS_sNK = thr_copy.partition_D(sNK_tile)
    tRS_acc = tiled_copy_state_r2s.retile(acc_vnew)
    cNK = cute.make_identity_tensor((BT, BV))
    tRS_cNK = thr_copy.partition_S(cNK)
    tRS_rNK = cute.make_rmem_tensor(cute.shape(tRS_acc), cutlass.BFloat16)
    for idx in cutlass.range(cute.size(tRS_rNK)):
        co = tRS_cNK[idx]
        row = co[0]
        tRS_rNK[idx] = (tRS_acc[idx] * sExpDecay[row]).to(cutlass.BFloat16)
    cute.copy(tiled_copy_state_r2s, tRS_rNK, tRS_sNK)


@cute.jit
def _mma_qk_to_sA_gated(tiled_mma, thr_mma, sQ, sK, sA, sExpGC):
    acc = cute.make_rmem_tensor(
        thr_mma.partition_shape_C((BT, BT)), cutlass.Float32)
    acc.fill(cutlass.Float32(0.0))
    tCsQA = thr_mma.partition_A(sQ)
    tCsKB = thr_mma.partition_B(sK)
    tCrQA = thr_mma.make_fragment_A(tCsQA)
    tCrKB = thr_mma.make_fragment_B(tCsKB)
    for kk in cutlass.range_constexpr(K_DIM // 16):
        cute.autovec_copy(tCsQA[None, None, kk], tCrQA[None, None, kk])
        cute.autovec_copy(tCsKB[None, None, kk], tCrKB[None, None, kk])
        cute.gemm(tiled_mma, acc, tCrQA[None, None, kk], tCrKB[None, None, kk], acc)
    cC = cute.make_identity_tensor((BT, BT))
    tCcC = thr_mma.partition_C(cC)
    for idx in cutlass.range(cute.size(acc)):
        co = tCcC[idx]
        row = co[0]
        col = co[1]
        if row >= col:
            a_val = acc[idx] * sExpGC[row] / sExpGC[col]
            sA[row, col] = a_val.to(cutlass.BFloat16)
        else:
            sA[row, col] = cutlass.BFloat16(0.0)


# ============================================================================
# Kernel 1: fused_chunk_h with precomputed M + inline chunk_o
# ============================================================================

@cute.kernel
def fused_chunk_h_kernel(
    tiled_mma: cute.TiledMma,
    tiled_copy_kq: cute.TiledCopy,
    sK_layout: cute.Layout,
    sV_layout: cute.Layout, sA_layout: cute.Layout,
    sGQK_layout: cute.Layout,
    sBeta_layout: cute.Layout,
    sS_layout: cute.Layout, sNK_layout: cute.Layout,
    sExpGC_layout: cute.Layout, sExpDecay_layout: cute.Layout,
    mKnorm: cute.Tensor, mQnorm: cute.Tensor,
    mV: cute.Tensor, mBeta: cute.Tensor,
    mM: cute.Tensor, mGQK: cute.Tensor,
    mExpGC_in: cute.Tensor, mExpDecay_in: cute.Tensor,
    mO: cute.Tensor,
    scale: cutlass.Constexpr[float],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int], HV: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int], NT: cutlass.Constexpr[int],
    H_PER_HV: cutlass.Constexpr[int],
    BV_TILE: cutlass.Constexpr[int],
    PER_V_TILE: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    bid_v = cute.arch.block_idx()[0]
    bid_bh = cute.arch.block_idx()[1]
    i_b = bid_bh // HV
    i_hv = bid_bh % HV
    i_h = i_hv // H_PER_HV
    v_off = bid_v * BV_TILE

    smem = cutlass.utils.SmemAllocator()
    sK = smem.allocate_tensor(cutlass.BFloat16, sK_layout, 16)
    sQ = smem.allocate_tensor(cutlass.BFloat16, sK_layout, 16)
    sV = smem.allocate_tensor(cutlass.BFloat16, sV_layout, 16)
    sA = smem.allocate_tensor(cutlass.BFloat16, sA_layout, 16)
    sGQK = smem.allocate_tensor(cutlass.BFloat16, sGQK_layout, 16)
    sA2 = smem.allocate_tensor(cutlass.BFloat16, sA_layout, 16)
    sBeta = smem.allocate_tensor(cutlass.Float32, sBeta_layout, 16)
    sS = smem.allocate_tensor(cutlass.BFloat16, sS_layout, 16)
    sNK_A = smem.allocate_tensor(cutlass.BFloat16, sNK_layout, 16)
    sExpGC = smem.allocate_tensor(cutlass.Float32, sExpGC_layout, 16)
    sExpDecay = smem.allocate_tensor(cutlass.Float32, sExpDecay_layout, 16)

    state = cute.make_rmem_tensor(cute.make_layout((BV_TILE,)), cutlass.Float32)
    state.fill(cutlass.Float32(0.0))

    thr_mma = tiled_mma.get_slice(tidx)

    cC_bv = cute.make_identity_tensor((BT, BV_TILE))
    tCcC_bv = thr_mma.partition_C(cC_bv)

    gK_full = mKnorm[(i_b, i_h, None, None)]
    gQ_full = mQnorm[(i_b, i_h, None, None)]
    thr_cp = tiled_copy_kq.get_slice(tidx)
    thr_sK_cp = thr_cp.partition_D(sK)
    thr_sQ_cp = thr_cp.partition_D(sQ)

    for chunk_idx in cutlass.range(NT):
        t0 = chunk_idx * BT

        # ==================== cp.async K, Q (non-blocking) ====================
        gK_chunk = cute.local_tile(gK_full, (BT, K_DIM), (chunk_idx, 0))
        gQ_chunk = cute.local_tile(gQ_full, (BT, K_DIM), (chunk_idx, 0))
        thr_gK = thr_cp.partition_S(gK_chunk)
        thr_gQ = thr_cp.partition_S(gQ_chunk)
        cute.copy(tiled_copy_kq, thr_gK, thr_sK_cp)
        cute.copy(tiled_copy_kq, thr_gQ, thr_sQ_cp)
        cute.arch.cp_async_commit_group()

        # ==================== Overlap: load rest while cp.async flies =========
        for i in cutlass.range_constexpr(PER_A):
            idx = i * NUM_THREADS + tidx
            row = idx // BT
            col = idx % BT
            sA[row, col] = mM[i_b, chunk_idx, i_hv, row, col]
            sGQK[row, col] = mGQK[i_b, chunk_idx, i_hv, row, col]

        for i in cutlass.range_constexpr(PER_V_TILE):
            idx = i * NUM_THREADS + tidx
            r = idx // BV_TILE
            c = idx % BV_TILE
            sV[r, c] = mV[i_b, t0 + r, i_hv, v_off + c]

        if tidx < BT:
            sExpGC[tidx] = mExpGC_in[i_b, chunk_idx, i_hv, tidx]
            sExpDecay[tidx] = mExpDecay_in[i_b, chunk_idx, i_hv, tidx]
            sBeta[tidx] = cutlass.Float32(mBeta[i_b, t0 + tidx, i_hv])

        for v in cutlass.range_constexpr(BV_TILE):
            sS[v, tidx] = state[v].to(cutlass.BFloat16)

        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        # ==================== kS + qS (full-width MMA) ====================
        acc_kS = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        acc_kS.fill(cutlass.Float32(0.0))
        acc_qS = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        acc_qS.fill(cutlass.Float32(0.0))

        _mma_AS_full(tiled_mma, thr_mma, sK, sS, acc_kS)
        _mma_AS_full(tiled_mma, thr_mma, sQ, sS, acc_qS)

        # ==================== RHS + TRANSPOSE ====================
        for idx in cutlass.range(cute.size(acc_kS)):
            co = tCcC_bv[idx]
            t = co[0]
            v = co[1]
            v_val = cutlass.Float32(sV[t, v])
            kS_val = acc_kS[idx]
            rhs = sBeta[t] * (v_val - sExpGC[t] * kS_val)
            sNK_A[v, t] = rhs.to(cutlass.BFloat16)
            acc_qS[idx] = acc_qS[idx] * sExpGC[t]
        cute.arch.barrier()

        # ==================== v_new = M @ RHS (single MMA) ====================
        acc_vnew = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        acc_vnew.fill(cutlass.Float32(0.0))

        tCsMA = thr_mma.partition_A(sA)
        tCsNKB = thr_mma.partition_B(sNK_A)
        tCrMA = thr_mma.make_fragment_A(tCsMA)
        tCrNKB = thr_mma.make_fragment_B(tCsNKB)
        for kk in cutlass.range_constexpr(BT // 16):
            cute.autovec_copy(tCsMA[None, None, kk], tCrMA[None, None, kk])
            cute.autovec_copy(tCsNKB[None, None, kk], tCrNKB[None, None, kk])
            cute.gemm(tiled_mma, acc_vnew, tCrMA[None, None, kk], tCrNKB[None, None, kk], acc_vnew)

        for idx in cutlass.range(cute.size(acc_vnew)):
            co = tCcC_bv[idx]
            sV[co[0], co[1]] = acc_vnew[idx].to(cutlass.BFloat16)
            sNK_A[co[1], co[0]] = acc_vnew[idx].to(cutlass.BFloat16)
        cute.arch.barrier()

        # ==================== INLINE CHUNK_O ====================
        tCsAA = thr_mma.partition_A(sGQK)
        tCsNKB2 = thr_mma.partition_B(sNK_A)
        tCrAA = thr_mma.make_fragment_A(tCsAA)
        tCrNKB2 = thr_mma.make_fragment_B(tCsNKB2)
        for kk in cutlass.range_constexpr(BT // 16):
            cute.autovec_copy(tCsAA[None, None, kk], tCrAA[None, None, kk])
            cute.autovec_copy(tCsNKB2[None, None, kk], tCrNKB2[None, None, kk])
            cute.gemm(tiled_mma, acc_qS, tCrAA[None, None, kk], tCrNKB2[None, None, kk], acc_qS)

        for idx in cutlass.range(cute.size(acc_qS)):
            co = tCcC_bv[idx]
            o_val = scale * acc_qS[idx]
            mO[i_b, t0 + co[0], i_hv, v_off + co[1]] = o_val.to(mO.element_type)

        # ==================== STATE UPDATE ====================
        if tidx < K_DIM:
            phi = sExpGC[BT - 1]
            for v in cutlass.range_constexpr(BV_TILE):
                state[v] = phi * state[v]
            for t in cutlass.range_constexpr(BT):
                kd = cutlass.Float32(sK[t, tidx]) * sExpDecay[t]
                for v in cutlass.range_constexpr(BV_TILE):
                    state[v] = state[v] + kd * cutlass.Float32(sV[t, v])

        cute.arch.barrier()


@cute.kernel
def fused_chunk_h_mgqk_cpasync_probe_kernel(
    tiled_mma: cute.TiledMma,
    tiled_copy_kq: cute.TiledCopy,
    tiled_copy_mgqk: cute.TiledCopy,
    sK_layout: cute.Layout,
    sV_layout: cute.Layout, sA_layout: cute.Layout,
    sGQK_layout: cute.Layout,
    sBeta_layout: cute.Layout,
    sS_layout: cute.Layout, sNK_layout: cute.Layout,
    sExpGC_layout: cute.Layout, sExpDecay_layout: cute.Layout,
    mKnorm: cute.Tensor, mQnorm: cute.Tensor,
    mV: cute.Tensor, mBeta: cute.Tensor,
    mM: cute.Tensor, mGQK: cute.Tensor,
    mExpGC_in: cute.Tensor, mExpDecay_in: cute.Tensor,
    mO: cute.Tensor,
    scale: cutlass.Constexpr[float],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int], HV: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int], NT: cutlass.Constexpr[int],
    H_PER_HV: cutlass.Constexpr[int],
    BV_TILE: cutlass.Constexpr[int],
    PER_V_TILE: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    bid_v = cute.arch.block_idx()[0]
    bid_bh = cute.arch.block_idx()[1]
    i_b = bid_bh // HV
    i_hv = bid_bh % HV
    i_h = i_hv // H_PER_HV
    v_off = bid_v * BV_TILE

    smem = cutlass.utils.SmemAllocator()
    sK = smem.allocate_tensor(cutlass.BFloat16, sK_layout, 16)
    sQ = smem.allocate_tensor(cutlass.BFloat16, sK_layout, 16)
    sV = smem.allocate_tensor(cutlass.BFloat16, sV_layout, 16)
    sA = smem.allocate_tensor(cutlass.BFloat16, sA_layout, 16)
    sGQK = smem.allocate_tensor(cutlass.BFloat16, sGQK_layout, 16)
    sA2 = smem.allocate_tensor(cutlass.BFloat16, sA_layout, 16)
    sBeta = smem.allocate_tensor(cutlass.Float32, sBeta_layout, 16)
    sS = smem.allocate_tensor(cutlass.BFloat16, sS_layout, 16)
    sNK_A = smem.allocate_tensor(cutlass.BFloat16, sNK_layout, 16)
    sExpGC = smem.allocate_tensor(cutlass.Float32, sExpGC_layout, 16)
    sExpDecay = smem.allocate_tensor(cutlass.Float32, sExpDecay_layout, 16)

    state = cute.make_rmem_tensor(cute.make_layout((BV_TILE,)), cutlass.Float32)
    state.fill(cutlass.Float32(0.0))

    thr_mma = tiled_mma.get_slice(tidx)

    cC_bv = cute.make_identity_tensor((BT, BV_TILE))
    tCcC_bv = thr_mma.partition_C(cC_bv)

    gK_full = mKnorm[(i_b, i_h, None, None)]
    gQ_full = mQnorm[(i_b, i_h, None, None)]
    thr_cp = tiled_copy_kq.get_slice(tidx)
    thr_sK_cp = thr_cp.partition_D(sK)
    thr_sQ_cp = thr_cp.partition_D(sQ)

    thr_cp_mgqk = tiled_copy_mgqk.get_slice(tidx)
    thr_sA_cp = thr_cp_mgqk.partition_D(sA)
    thr_sGQK_cp = thr_cp_mgqk.partition_D(sGQK)

    for chunk_idx in cutlass.range(NT):
        t0 = chunk_idx * BT

        # ==================== cp.async K/Q/M/GQK ====================
        gK_chunk = cute.local_tile(gK_full, (BT, K_DIM), (chunk_idx, 0))
        gQ_chunk = cute.local_tile(gQ_full, (BT, K_DIM), (chunk_idx, 0))
        thr_gK = thr_cp.partition_S(gK_chunk)
        thr_gQ = thr_cp.partition_S(gQ_chunk)
        cute.copy(tiled_copy_kq, thr_gK, thr_sK_cp)
        cute.copy(tiled_copy_kq, thr_gQ, thr_sQ_cp)

        gM_chunk = mM[(i_b, chunk_idx, i_hv, None, None)]
        gGQK_chunk = mGQK[(i_b, chunk_idx, i_hv, None, None)]
        thr_gM = thr_cp_mgqk.partition_S(gM_chunk)
        thr_gGQK = thr_cp_mgqk.partition_S(gGQK_chunk)
        cute.copy(tiled_copy_mgqk, thr_gM, thr_sA_cp)
        cute.copy(tiled_copy_mgqk, thr_gGQK, thr_sGQK_cp)
        cute.arch.cp_async_commit_group()

        # ==================== Overlap scalar/V loads while cp.async flies =====
        for i in cutlass.range_constexpr(PER_V_TILE):
            idx = i * NUM_THREADS + tidx
            r = idx // BV_TILE
            c = idx % BV_TILE
            sV[r, c] = mV[i_b, t0 + r, i_hv, v_off + c]

        if tidx < BT:
            sExpGC[tidx] = mExpGC_in[i_b, chunk_idx, i_hv, tidx]
            sExpDecay[tidx] = mExpDecay_in[i_b, chunk_idx, i_hv, tidx]
            sBeta[tidx] = cutlass.Float32(mBeta[i_b, t0 + tidx, i_hv])

        for v in cutlass.range_constexpr(BV_TILE):
            sS[v, tidx] = state[v].to(cutlass.BFloat16)

        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        # ==================== kS + qS (full-width MMA) ====================
        acc_kS = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        acc_kS.fill(cutlass.Float32(0.0))
        acc_qS = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        acc_qS.fill(cutlass.Float32(0.0))

        _mma_AS_full(tiled_mma, thr_mma, sK, sS, acc_kS)
        _mma_AS_full(tiled_mma, thr_mma, sQ, sS, acc_qS)

        # ==================== RHS + TRANSPOSE ====================
        for idx in cutlass.range(cute.size(acc_kS)):
            co = tCcC_bv[idx]
            t = co[0]
            v = co[1]
            v_val = cutlass.Float32(sV[t, v])
            kS_val = acc_kS[idx]
            rhs = sBeta[t] * (v_val - sExpGC[t] * kS_val)
            sNK_A[v, t] = rhs.to(cutlass.BFloat16)
            acc_qS[idx] = acc_qS[idx] * sExpGC[t]
        cute.arch.barrier()

        # ==================== v_new = M @ RHS (single MMA) ====================
        acc_vnew = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        acc_vnew.fill(cutlass.Float32(0.0))

        tCsMA = thr_mma.partition_A(sA)
        tCsNKB = thr_mma.partition_B(sNK_A)
        tCrMA = thr_mma.make_fragment_A(tCsMA)
        tCrNKB = thr_mma.make_fragment_B(tCsNKB)
        for kk in cutlass.range_constexpr(BT // 16):
            cute.autovec_copy(tCsMA[None, None, kk], tCrMA[None, None, kk])
            cute.autovec_copy(tCsNKB[None, None, kk], tCrNKB[None, None, kk])
            cute.gemm(tiled_mma, acc_vnew, tCrMA[None, None, kk], tCrNKB[None, None, kk], acc_vnew)

        for idx in cutlass.range(cute.size(acc_vnew)):
            co = tCcC_bv[idx]
            sV[co[0], co[1]] = acc_vnew[idx].to(cutlass.BFloat16)
            sNK_A[co[1], co[0]] = acc_vnew[idx].to(cutlass.BFloat16)
        cute.arch.barrier()

        # ==================== INLINE CHUNK_O ====================
        tCsAA = thr_mma.partition_A(sGQK)
        tCsNKB2 = thr_mma.partition_B(sNK_A)
        tCrAA = thr_mma.make_fragment_A(tCsAA)
        tCrNKB2 = thr_mma.make_fragment_B(tCsNKB2)
        for kk in cutlass.range_constexpr(BT // 16):
            cute.autovec_copy(tCsAA[None, None, kk], tCrAA[None, None, kk])
            cute.autovec_copy(tCsNKB2[None, None, kk], tCrNKB2[None, None, kk])
            cute.gemm(tiled_mma, acc_qS, tCrAA[None, None, kk], tCrNKB2[None, None, kk], acc_qS)

        for idx in cutlass.range(cute.size(acc_qS)):
            co = tCcC_bv[idx]
            o_val = scale * acc_qS[idx]
            mO[i_b, t0 + co[0], i_hv, v_off + co[1]] = o_val.to(mO.element_type)

        # ==================== STATE UPDATE ====================
        if tidx < K_DIM:
            phi = sExpGC[BT - 1]
            for v in cutlass.range_constexpr(BV_TILE):
                state[v] = phi * state[v]
            for t in cutlass.range_constexpr(BT):
                kd = cutlass.Float32(sK[t, tidx]) * sExpDecay[t]
                for v in cutlass.range_constexpr(BV_TILE):
                    state[v] = state[v] + kd * cutlass.Float32(sV[t, v])

        cute.arch.barrier()


@cute.kernel
def fused_chunk_h_mgqk_v_cpasync_probe_kernel(
    tiled_mma: cute.TiledMma,
    tiled_copy_kq: cute.TiledCopy,
    tiled_copy_mgqk: cute.TiledCopy,
    tiled_copy_v: cute.TiledCopy,
    sK_layout: cute.Layout,
    sV_layout: cute.Layout, sA_layout: cute.Layout,
    sGQK_layout: cute.Layout,
    sBeta_layout: cute.Layout,
    sS_layout: cute.Layout, sNK_layout: cute.Layout,
    sExpGC_layout: cute.Layout, sExpDecay_layout: cute.Layout,
    mKnorm: cute.Tensor, mQnorm: cute.Tensor,
    mV: cute.Tensor, mBeta: cute.Tensor,
    mM: cute.Tensor, mGQK: cute.Tensor,
    mExpGC_in: cute.Tensor, mExpDecay_in: cute.Tensor,
    mO: cute.Tensor,
    scale: cutlass.Constexpr[float],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int], HV: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int], NT: cutlass.Constexpr[int],
    H_PER_HV: cutlass.Constexpr[int],
    BV_TILE: cutlass.Constexpr[int],
    PER_V_TILE: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    bid_v = cute.arch.block_idx()[0]
    bid_bh = cute.arch.block_idx()[1]
    i_b = bid_bh // HV
    i_hv = bid_bh % HV
    i_h = i_hv // H_PER_HV
    v_off = bid_v * BV_TILE

    smem = cutlass.utils.SmemAllocator()
    sK = smem.allocate_tensor(cutlass.BFloat16, sK_layout, 16)
    sQ = smem.allocate_tensor(cutlass.BFloat16, sK_layout, 16)
    sV = smem.allocate_tensor(cutlass.BFloat16, sV_layout, 16)
    sA = smem.allocate_tensor(cutlass.BFloat16, sA_layout, 16)
    sGQK = smem.allocate_tensor(cutlass.BFloat16, sGQK_layout, 16)
    sBeta = smem.allocate_tensor(cutlass.Float32, sBeta_layout, 16)
    sS = smem.allocate_tensor(cutlass.BFloat16, sS_layout, 16)
    sNK_A = smem.allocate_tensor(cutlass.BFloat16, sNK_layout, 16)
    sExpGC = smem.allocate_tensor(cutlass.Float32, sExpGC_layout, 16)
    sExpDecay = smem.allocate_tensor(cutlass.Float32, sExpDecay_layout, 16)

    state = cute.make_rmem_tensor(cute.make_layout((BV_TILE,)), cutlass.Float32)
    state.fill(cutlass.Float32(0.0))

    thr_mma = tiled_mma.get_slice(tidx)

    cC_bv = cute.make_identity_tensor((BT, BV_TILE))
    tCcC_bv = thr_mma.partition_C(cC_bv)

    gK_full = mKnorm[(i_b, i_h, None, None)]
    gQ_full = mQnorm[(i_b, i_h, None, None)]
    gV_full = mV[(i_b, None, i_hv, None)]
    thr_cp = tiled_copy_kq.get_slice(tidx)
    thr_sK_cp = thr_cp.partition_D(sK)
    thr_sQ_cp = thr_cp.partition_D(sQ)

    thr_cp_mgqk = tiled_copy_mgqk.get_slice(tidx)
    thr_sA_cp = thr_cp_mgqk.partition_D(sA)
    thr_sGQK_cp = thr_cp_mgqk.partition_D(sGQK)

    for chunk_idx in cutlass.range(NT):
        t0 = chunk_idx * BT

        # ==================== cp.async K/Q/M/GQK/V ====================
        gK_chunk = cute.local_tile(gK_full, (BT, K_DIM), (chunk_idx, 0))
        gQ_chunk = cute.local_tile(gQ_full, (BT, K_DIM), (chunk_idx, 0))
        thr_gK = thr_cp.partition_S(gK_chunk)
        thr_gQ = thr_cp.partition_S(gQ_chunk)
        cute.copy(tiled_copy_kq, thr_gK, thr_sK_cp)
        cute.copy(tiled_copy_kq, thr_gQ, thr_sQ_cp)

        gM_chunk = mM[(i_b, chunk_idx, i_hv, None, None)]
        gGQK_chunk = mGQK[(i_b, chunk_idx, i_hv, None, None)]
        thr_gM = thr_cp_mgqk.partition_S(gM_chunk)
        thr_gGQK = thr_cp_mgqk.partition_S(gGQK_chunk)
        cute.copy(tiled_copy_mgqk, thr_gM, thr_sA_cp)
        cute.copy(tiled_copy_mgqk, thr_gGQK, thr_sGQK_cp)

        if tidx < 64:
            gV_chunk = cute.local_tile(gV_full, (BT, BV_TILE), (chunk_idx, bid_v))
            thr_cp_v = tiled_copy_v.get_slice(tidx)
            thr_sV_cp = thr_cp_v.partition_D(sV)
            thr_gV = thr_cp_v.partition_S(gV_chunk)
            cute.copy(tiled_copy_v, thr_gV, thr_sV_cp)
        cute.arch.cp_async_commit_group()

        # ==================== Overlap scalar loads while cp.async flies =====
        if tidx < BT:
            sExpGC[tidx] = mExpGC_in[i_b, chunk_idx, i_hv, tidx]
            sExpDecay[tidx] = mExpDecay_in[i_b, chunk_idx, i_hv, tidx]
            sBeta[tidx] = cutlass.Float32(mBeta[i_b, t0 + tidx, i_hv])

        for v in cutlass.range_constexpr(BV_TILE):
            sS[v, tidx] = state[v].to(cutlass.BFloat16)

        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        # ==================== kS + qS (full-width MMA) ====================
        acc_kS = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        acc_kS.fill(cutlass.Float32(0.0))
        acc_qS = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        acc_qS.fill(cutlass.Float32(0.0))

        _mma_AS_full(tiled_mma, thr_mma, sK, sS, acc_kS)
        _mma_AS_full(tiled_mma, thr_mma, sQ, sS, acc_qS)

        # ==================== RHS + TRANSPOSE ====================
        for idx in cutlass.range(cute.size(acc_kS)):
            co = tCcC_bv[idx]
            t = co[0]
            v = co[1]
            v_val = cutlass.Float32(sV[t, v])
            kS_val = acc_kS[idx]
            rhs = sBeta[t] * (v_val - sExpGC[t] * kS_val)
            sNK_A[v, t] = rhs.to(cutlass.BFloat16)
            acc_qS[idx] = acc_qS[idx] * sExpGC[t]
        cute.arch.barrier()

        # ==================== v_new = M @ RHS (single MMA) ====================
        acc_vnew = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        acc_vnew.fill(cutlass.Float32(0.0))

        tCsMA = thr_mma.partition_A(sA)
        tCsNKB = thr_mma.partition_B(sNK_A)
        tCrMA = thr_mma.make_fragment_A(tCsMA)
        tCrNKB = thr_mma.make_fragment_B(tCsNKB)
        for kk in cutlass.range_constexpr(BT // 16):
            cute.autovec_copy(tCsMA[None, None, kk], tCrMA[None, None, kk])
            cute.autovec_copy(tCsNKB[None, None, kk], tCrNKB[None, None, kk])
            cute.gemm(tiled_mma, acc_vnew, tCrMA[None, None, kk], tCrNKB[None, None, kk], acc_vnew)

        for idx in cutlass.range(cute.size(acc_vnew)):
            co = tCcC_bv[idx]
            sV[co[0], co[1]] = acc_vnew[idx].to(cutlass.BFloat16)
            sNK_A[co[1], co[0]] = acc_vnew[idx].to(cutlass.BFloat16)
        cute.arch.barrier()

        # ==================== INLINE CHUNK_O ====================
        tCsAA = thr_mma.partition_A(sGQK)
        tCsNKB2 = thr_mma.partition_B(sNK_A)
        tCrAA = thr_mma.make_fragment_A(tCsAA)
        tCrNKB2 = thr_mma.make_fragment_B(tCsNKB2)
        for kk in cutlass.range_constexpr(BT // 16):
            cute.autovec_copy(tCsAA[None, None, kk], tCrAA[None, None, kk])
            cute.autovec_copy(tCsNKB2[None, None, kk], tCrNKB2[None, None, kk])
            cute.gemm(tiled_mma, acc_qS, tCrAA[None, None, kk], tCrNKB2[None, None, kk], acc_qS)

        for idx in cutlass.range(cute.size(acc_qS)):
            co = tCcC_bv[idx]
            o_val = scale * acc_qS[idx]
            mO[i_b, t0 + co[0], i_hv, v_off + co[1]] = o_val.to(mO.element_type)

        # ==================== STATE UPDATE ====================
        if tidx < K_DIM:
            phi = sExpGC[BT - 1]
            for v in cutlass.range_constexpr(BV_TILE):
                state[v] = phi * state[v]
            for t in cutlass.range_constexpr(BT):
                kd = cutlass.Float32(sK[t, tidx]) * sExpDecay[t]
                for v in cutlass.range_constexpr(BV_TILE):
                    state[v] = state[v] + kd * cutlass.Float32(sV[t, v])

        cute.arch.barrier()


@cute.kernel
def fused_chunk_h_mgqk_v_fp32state_cpasync_probe_kernel(
    tiled_mma: cute.TiledMma,
    tiled_copy_kq: cute.TiledCopy,
    tiled_copy_mgqk: cute.TiledCopy,
    tiled_copy_v: cute.TiledCopy,
    sK_layout: cute.Layout,
    sV_layout: cute.Layout, sVState_layout: cute.Layout, sA_layout: cute.Layout,
    sGQK_layout: cute.Layout,
    sBeta_layout: cute.Layout,
    sS_layout: cute.Layout, sNK_layout: cute.Layout,
    sExpGC_layout: cute.Layout, sExpDecay_layout: cute.Layout,
    mKnorm: cute.Tensor, mQnorm: cute.Tensor,
    mV: cute.Tensor, mBeta: cute.Tensor,
    mM: cute.Tensor, mGQK: cute.Tensor,
    mExpGC_in: cute.Tensor, mExpDecay_in: cute.Tensor,
    mO: cute.Tensor,
    scale: cutlass.Constexpr[float],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int], HV: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int], NT: cutlass.Constexpr[int],
    H_PER_HV: cutlass.Constexpr[int],
    BV_TILE: cutlass.Constexpr[int],
    PER_V_TILE: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    bid_v = cute.arch.block_idx()[0]
    bid_bh = cute.arch.block_idx()[1]
    i_b = bid_bh // HV
    i_hv = bid_bh % HV
    i_h = i_hv // H_PER_HV
    v_off = bid_v * BV_TILE

    smem = cutlass.utils.SmemAllocator()
    sK = smem.allocate_tensor(cutlass.BFloat16, sK_layout, 16)
    sQ = smem.allocate_tensor(cutlass.BFloat16, sK_layout, 16)
    sV = smem.allocate_tensor(cutlass.BFloat16, sV_layout, 16)
    sVState = smem.allocate_tensor(cutlass.Float32, sVState_layout, 16)
    sA = smem.allocate_tensor(cutlass.BFloat16, sA_layout, 16)
    sGQK = smem.allocate_tensor(cutlass.BFloat16, sGQK_layout, 16)
    sBeta = smem.allocate_tensor(cutlass.Float32, sBeta_layout, 16)
    sS = smem.allocate_tensor(cutlass.BFloat16, sS_layout, 16)
    sNK_A = smem.allocate_tensor(cutlass.BFloat16, sNK_layout, 16)
    sExpGC = smem.allocate_tensor(cutlass.Float32, sExpGC_layout, 16)
    sExpDecay = smem.allocate_tensor(cutlass.Float32, sExpDecay_layout, 16)

    state = cute.make_rmem_tensor(cute.make_layout((BV_TILE,)), cutlass.Float32)
    state.fill(cutlass.Float32(0.0))

    thr_mma = tiled_mma.get_slice(tidx)

    cC_bv = cute.make_identity_tensor((BT, BV_TILE))
    tCcC_bv = thr_mma.partition_C(cC_bv)

    gK_full = mKnorm[(i_b, i_h, None, None)]
    gQ_full = mQnorm[(i_b, i_h, None, None)]
    gV_full = mV[(i_b, None, i_hv, None)]
    thr_cp = tiled_copy_kq.get_slice(tidx)
    thr_sK_cp = thr_cp.partition_D(sK)
    thr_sQ_cp = thr_cp.partition_D(sQ)

    thr_cp_mgqk = tiled_copy_mgqk.get_slice(tidx)
    thr_sA_cp = thr_cp_mgqk.partition_D(sA)
    thr_sGQK_cp = thr_cp_mgqk.partition_D(sGQK)

    for chunk_idx in cutlass.range(NT):
        t0 = chunk_idx * BT

        # ==================== cp.async K/Q/M/GQK/V ====================
        gK_chunk = cute.local_tile(gK_full, (BT, K_DIM), (chunk_idx, 0))
        gQ_chunk = cute.local_tile(gQ_full, (BT, K_DIM), (chunk_idx, 0))
        thr_gK = thr_cp.partition_S(gK_chunk)
        thr_gQ = thr_cp.partition_S(gQ_chunk)
        cute.copy(tiled_copy_kq, thr_gK, thr_sK_cp)
        cute.copy(tiled_copy_kq, thr_gQ, thr_sQ_cp)

        gM_chunk = mM[(i_b, chunk_idx, i_hv, None, None)]
        gGQK_chunk = mGQK[(i_b, chunk_idx, i_hv, None, None)]
        thr_gM = thr_cp_mgqk.partition_S(gM_chunk)
        thr_gGQK = thr_cp_mgqk.partition_S(gGQK_chunk)
        cute.copy(tiled_copy_mgqk, thr_gM, thr_sA_cp)
        cute.copy(tiled_copy_mgqk, thr_gGQK, thr_sGQK_cp)

        if tidx < 64:
            gV_chunk = cute.local_tile(gV_full, (BT, BV_TILE), (chunk_idx, bid_v))
            thr_cp_v = tiled_copy_v.get_slice(tidx)
            thr_sV_cp = thr_cp_v.partition_D(sV)
            thr_gV = thr_cp_v.partition_S(gV_chunk)
            cute.copy(tiled_copy_v, thr_gV, thr_sV_cp)
        cute.arch.cp_async_commit_group()

        if tidx < BT:
            sExpGC[tidx] = mExpGC_in[i_b, chunk_idx, i_hv, tidx]
            sExpDecay[tidx] = mExpDecay_in[i_b, chunk_idx, i_hv, tidx]
            sBeta[tidx] = cutlass.Float32(mBeta[i_b, t0 + tidx, i_hv])

        for v in cutlass.range_constexpr(BV_TILE):
            sS[v, tidx] = state[v].to(cutlass.BFloat16)

        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        # ==================== kS + qS (full-width MMA) ====================
        acc_kS = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        acc_kS.fill(cutlass.Float32(0.0))
        acc_qS = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        acc_qS.fill(cutlass.Float32(0.0))

        _mma_AS_full(tiled_mma, thr_mma, sK, sS, acc_kS)
        _mma_AS_full(tiled_mma, thr_mma, sQ, sS, acc_qS)

        # ==================== RHS + TRANSPOSE ====================
        for idx in cutlass.range(cute.size(acc_kS)):
            co = tCcC_bv[idx]
            t = co[0]
            v = co[1]
            v_val = cutlass.Float32(sV[t, v])
            kS_val = acc_kS[idx]
            rhs = sBeta[t] * (v_val - sExpGC[t] * kS_val)
            sNK_A[v, t] = rhs.to(cutlass.BFloat16)
            acc_qS[idx] = acc_qS[idx] * sExpGC[t]
        cute.arch.barrier()

        # ==================== v_new = M @ RHS (single MMA) ====================
        acc_vnew = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        acc_vnew.fill(cutlass.Float32(0.0))

        tCsMA = thr_mma.partition_A(sA)
        tCsNKB = thr_mma.partition_B(sNK_A)
        tCrMA = thr_mma.make_fragment_A(tCsMA)
        tCrNKB = thr_mma.make_fragment_B(tCsNKB)
        for kk in cutlass.range_constexpr(BT // 16):
            cute.autovec_copy(tCsMA[None, None, kk], tCrMA[None, None, kk])
            cute.autovec_copy(tCsNKB[None, None, kk], tCrNKB[None, None, kk])
            cute.gemm(tiled_mma, acc_vnew, tCrMA[None, None, kk], tCrNKB[None, None, kk], acc_vnew)

        for idx in cutlass.range(cute.size(acc_vnew)):
            co = tCcC_bv[idx]
            vnew = acc_vnew[idx].to(cutlass.BFloat16)
            sV[co[0], co[1]] = vnew
            sVState[co[0], co[1]] = acc_vnew[idx]
            sNK_A[co[1], co[0]] = vnew
        cute.arch.barrier()

        # ==================== INLINE CHUNK_O ====================
        tCsAA = thr_mma.partition_A(sGQK)
        tCsNKB2 = thr_mma.partition_B(sNK_A)
        tCrAA = thr_mma.make_fragment_A(tCsAA)
        tCrNKB2 = thr_mma.make_fragment_B(tCsNKB2)
        for kk in cutlass.range_constexpr(BT // 16):
            cute.autovec_copy(tCsAA[None, None, kk], tCrAA[None, None, kk])
            cute.autovec_copy(tCsNKB2[None, None, kk], tCrNKB2[None, None, kk])
            cute.gemm(tiled_mma, acc_qS, tCrAA[None, None, kk], tCrNKB2[None, None, kk], acc_qS)

        for idx in cutlass.range(cute.size(acc_qS)):
            co = tCcC_bv[idx]
            o_val = scale * acc_qS[idx]
            mO[i_b, t0 + co[0], i_hv, v_off + co[1]] = o_val.to(mO.element_type)

        # ==================== STATE UPDATE ====================
        if tidx < K_DIM:
            phi = sExpGC[BT - 1]
            for v in cutlass.range_constexpr(BV_TILE):
                state[v] = phi * state[v]
            for t in cutlass.range_constexpr(BT):
                kd = cutlass.Float32(sK[t, tidx]) * sExpDecay[t]
                for v in cutlass.range_constexpr(BV_TILE):
                    state[v] = state[v] + kd * sVState[t, v]

        cute.arch.barrier()


@cute.kernel
def fused_chunk_h_mgqk_v_ldsm_probe_kernel(
    tiled_mma: cute.TiledMma,
    smem_tiled_copy_A: cute.TiledCopy,
    smem_tiled_copy_A_trans: cute.TiledCopy,
    smem_tiled_copy_B: cute.TiledCopy,
    tiled_copy_state_r2s: cute.TiledCopy,
    tiled_copy_o_gmem: cute.TiledCopy,
    tiled_copy_kdecay_r2s: cute.TiledCopy,
    tiled_copy_kq: cute.TiledCopy,
    tiled_copy_mgqk: cute.TiledCopy,
    tiled_copy_v: cute.TiledCopy,
    sK_layout: cute.Layout,
    sV_layout: cute.Layout, sA_layout: cute.Layout,
    sGQK_layout: cute.Layout,
    sBeta_layout: cute.Layout,
    sS_layout: cute.Layout, sNK_layout: cute.Layout,
    sExpGC_layout: cute.Layout, sExpDecay_layout: cute.Layout,
    mKnorm: cute.Tensor, mQnorm: cute.Tensor,
    mV: cute.Tensor, mBeta: cute.Tensor,
    mM: cute.Tensor, mGQK: cute.Tensor,
    mExpGC_in: cute.Tensor, mExpDecay_in: cute.Tensor,
    mO: cute.Tensor,
    scale: cutlass.Constexpr[float],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int], HV: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int], NT: cutlass.Constexpr[int],
    H_PER_HV: cutlass.Constexpr[int],
    BV_TILE: cutlass.Constexpr[int],
    PER_V_TILE: cutlass.Constexpr[int],
    USE_STATE_FRAG: cutlass.Constexpr[bool],
    USE_DUAL_STATE_BUFFER: cutlass.Constexpr[bool],
    USE_DEDICATED_STATE_SCRATCH: cutlass.Constexpr[bool],
    USE_STATE_R2S: cutlass.Constexpr[bool],
    USE_VECTOR_O_GMEM: cutlass.Constexpr[bool],
    USE_KDECAY_R2S: cutlass.Constexpr[bool],
    USE_RHS_VNEW_R2S: cutlass.Constexpr[bool],
    USE_SKIP_SV_VNEW: cutlass.Constexpr[bool],
    USE_DIRECT_V_RHS: cutlass.Constexpr[bool],
    USE_SCALED_VNEW_STATE: cutlass.Constexpr[bool],
    USE_AUTOVEC_MMA: cutlass.Constexpr[bool],
    USE_AUTOVEC_STATE_MMA: cutlass.Constexpr[bool],
    USE_AUTOVEC_KQS_MMA: cutlass.Constexpr[bool],
    USE_AUTOVEC_KS_MMA: cutlass.Constexpr[bool],
    USE_AUTOVEC_QS_MMA: cutlass.Constexpr[bool],
    USE_AUTOVEC_VNEW_MMA: cutlass.Constexpr[bool],
    USE_AUTOVEC_CHUNKO_MMA: cutlass.Constexpr[bool],
    USE_SCALAR_SCALED_VNEW_STORE: cutlass.Constexpr[bool],
    USE_SCALAR_VNEW_STORE: cutlass.Constexpr[bool],
    USE_SCALAR_RHS_STORE: cutlass.Constexpr[bool],
):
    tidx, _, _ = cute.arch.thread_idx()
    bid_v = cute.arch.block_idx()[0]
    bid_bh = cute.arch.block_idx()[1]
    i_b = bid_bh // HV
    i_hv = bid_bh % HV
    i_h = i_hv // H_PER_HV
    v_off = bid_v * BV_TILE

    smem = cutlass.utils.SmemAllocator()
    sK = smem.allocate_tensor(cutlass.BFloat16, sK_layout, 16)
    sQ = smem.allocate_tensor(cutlass.BFloat16, sK_layout, 16)
    sV = smem.allocate_tensor(cutlass.BFloat16, sV_layout, 16)
    sA = smem.allocate_tensor(cutlass.BFloat16, sA_layout, 16)
    sGQK = smem.allocate_tensor(cutlass.BFloat16, sGQK_layout, 16)
    if cutlass.const_expr(USE_DEDICATED_STATE_SCRATCH):
        sA2 = smem.allocate_tensor(cutlass.BFloat16, sA_layout, 16)
    sBeta = smem.allocate_tensor(cutlass.Float32, sBeta_layout, 16)
    sS = smem.allocate_tensor(cutlass.BFloat16, sS_layout, 16)
    sNK_A = smem.allocate_tensor(cutlass.BFloat16, sNK_layout, 16)
    sExpGC = smem.allocate_tensor(cutlass.Float32, sExpGC_layout, 16)
    sExpDecay = smem.allocate_tensor(cutlass.Float32, sExpDecay_layout, 16)

    thr_mma = tiled_mma.get_slice(tidx)

    cC_bv = cute.make_identity_tensor((BT, BV_TILE))
    tCcC_bv = thr_mma.partition_C(cC_bv)

    if cutlass.const_expr(USE_STATE_FRAG):
        state0 = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        state1 = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        state2 = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        state3 = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        state0.fill(cutlass.Float32(0.0))
        state1.fill(cutlass.Float32(0.0))
        state2.fill(cutlass.Float32(0.0))
        state3.fill(cutlass.Float32(0.0))
    else:
        state = cute.make_rmem_tensor(cute.make_layout((BV_TILE,)), cutlass.Float32)
        state.fill(cutlass.Float32(0.0))

    gK_full = mKnorm[(i_b, i_h, None, None)]
    gQ_full = mQnorm[(i_b, i_h, None, None)]
    gV_full = mV[(i_b, None, i_hv, None)]
    gO_full = mO[(i_b, None, i_hv, None)]
    thr_cp = tiled_copy_kq.get_slice(tidx)
    thr_sK_cp = thr_cp.partition_D(sK)
    thr_sQ_cp = thr_cp.partition_D(sQ)

    thr_cp_mgqk = tiled_copy_mgqk.get_slice(tidx)
    thr_sA_cp = thr_cp_mgqk.partition_D(sA)
    thr_sGQK_cp = thr_cp_mgqk.partition_D(sGQK)

    for chunk_idx in cutlass.range(NT):
        t0 = chunk_idx * BT

        # ==================== cp.async K/Q/M/GQK/V ====================
        gK_chunk = cute.local_tile(gK_full, (BT, K_DIM), (chunk_idx, 0))
        gQ_chunk = cute.local_tile(gQ_full, (BT, K_DIM), (chunk_idx, 0))
        thr_gK = thr_cp.partition_S(gK_chunk)
        thr_gQ = thr_cp.partition_S(gQ_chunk)
        cute.copy(tiled_copy_kq, thr_gK, thr_sK_cp)
        cute.copy(tiled_copy_kq, thr_gQ, thr_sQ_cp)

        gM_chunk = mM[(i_b, chunk_idx, i_hv, None, None)]
        gGQK_chunk = mGQK[(i_b, chunk_idx, i_hv, None, None)]
        thr_gM = thr_cp_mgqk.partition_S(gM_chunk)
        thr_gGQK = thr_cp_mgqk.partition_S(gGQK_chunk)
        cute.copy(tiled_copy_mgqk, thr_gM, thr_sA_cp)
        cute.copy(tiled_copy_mgqk, thr_gGQK, thr_sGQK_cp)

        gV_chunk = cute.local_tile(gV_full, (BT, BV_TILE), (chunk_idx, bid_v))
        if cutlass.const_expr(not USE_DIRECT_V_RHS):
            if tidx < 64:
                thr_cp_v = tiled_copy_v.get_slice(tidx)
                thr_sV_cp = thr_cp_v.partition_D(sV)
                thr_gV = thr_cp_v.partition_S(gV_chunk)
                cute.copy(tiled_copy_v, thr_gV, thr_sV_cp)
        cute.arch.cp_async_commit_group()

        if tidx < BT:
            sExpGC[tidx] = mExpGC_in[i_b, chunk_idx, i_hv, tidx]
            sExpDecay[tidx] = mExpDecay_in[i_b, chunk_idx, i_hv, tidx]
            sBeta[tidx] = cutlass.Float32(mBeta[i_b, t0 + tidx, i_hv])

        if cutlass.const_expr(USE_STATE_FRAG):
            if cutlass.const_expr(USE_STATE_R2S):
                sS_state_layout = cute.make_layout(
                    (BT, BV_TILE), stride=(1, K_DIM + 8)
                )
                sS0 = cute.make_tensor(sS.iterator.align(16), sS_state_layout)
                sS1 = cute.make_tensor((sS.iterator + BT).align(16), sS_state_layout)
                sS2 = cute.make_tensor((sS.iterator + 2 * BT).align(16), sS_state_layout)
                sS3 = cute.make_tensor((sS.iterator + 3 * BT).align(16), sS_state_layout)
                _store_state_frag_r2s(tiled_copy_state_r2s, tidx, state0, sS0)
                _store_state_frag_r2s(tiled_copy_state_r2s, tidx, state1, sS1)
                _store_state_frag_r2s(tiled_copy_state_r2s, tidx, state2, sS2)
                _store_state_frag_r2s(tiled_copy_state_r2s, tidx, state3, sS3)
            else:
                for idx in cutlass.range(cute.size(state0)):
                    co = tCcC_bv[idx]
                    row = co[0]
                    v_col = co[1]
                    sS[v_col, row] = state0[idx].to(cutlass.BFloat16)
                    sS[v_col, BT + row] = state1[idx].to(cutlass.BFloat16)
                    sS[v_col, 2 * BT + row] = state2[idx].to(cutlass.BFloat16)
                    sS[v_col, 3 * BT + row] = state3[idx].to(cutlass.BFloat16)
        else:
            for v in cutlass.range_constexpr(BV_TILE):
                sS[v, tidx] = state[v].to(cutlass.BFloat16)

        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        # ==================== kS + qS (LDSM-fed MMA) ====================
        acc_kS = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        acc_kS.fill(cutlass.Float32(0.0))
        acc_qS = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        acc_qS.fill(cutlass.Float32(0.0))

        if cutlass.const_expr(USE_AUTOVEC_MMA or USE_AUTOVEC_KQS_MMA or USE_AUTOVEC_KS_MMA):
            _mma_full_autovec(
                tiled_mma, thr_mma, sK, sS, acc_kS, K_DIM // 16,
            )
        else:
            _mma_full_ldsm(
                tiled_mma, smem_tiled_copy_A, smem_tiled_copy_B,
                thr_mma, tidx, sK, sS, acc_kS, K_DIM // 16,
            )
        if cutlass.const_expr(USE_AUTOVEC_MMA or USE_AUTOVEC_KQS_MMA or USE_AUTOVEC_QS_MMA):
            _mma_full_autovec(
                tiled_mma, thr_mma, sQ, sS, acc_qS, K_DIM // 16,
            )
        else:
            _mma_full_ldsm(
                tiled_mma, smem_tiled_copy_A, smem_tiled_copy_B,
                thr_mma, tidx, sQ, sS, acc_qS, K_DIM // 16,
            )

        # ==================== RHS + TRANSPOSE ====================
        if cutlass.const_expr(USE_RHS_VNEW_R2S):
            sNK_rhs_layout = cute.make_layout(
                (BT, BV_TILE), stride=(1, BT + 8)
            )
            sNK_rhs = cute.make_tensor(sNK_A.iterator.align(16), sNK_rhs_layout)
            if cutlass.const_expr(USE_DIRECT_V_RHS):
                _store_rhs_snk_r2s_direct_v(
                    tiled_copy_state_r2s, tidx, acc_kS,
                    gV_chunk, sBeta, sExpGC, sNK_rhs,
                )
            elif cutlass.const_expr(USE_SCALAR_RHS_STORE):
                for idx in cutlass.range(cute.size(acc_kS)):
                    co = tCcC_bv[idx]
                    row = co[0]
                    v_col = co[1]
                    rhs = sBeta[row] * (
                        cutlass.Float32(sV[row, v_col]) - sExpGC[row] * acc_kS[idx]
                    )
                    sNK_A[v_col, row] = rhs.to(cutlass.BFloat16)
            else:
                _store_rhs_snk_r2s(
                    tiled_copy_state_r2s, tidx, acc_kS,
                    sV, sBeta, sExpGC, sNK_rhs,
                )
            for idx in cutlass.range(cute.size(acc_qS)):
                co = tCcC_bv[idx]
                acc_qS[idx] = acc_qS[idx] * sExpGC[co[0]]
        else:
            for idx in cutlass.range(cute.size(acc_kS)):
                co = tCcC_bv[idx]
                t = co[0]
                v = co[1]
                v_val = cutlass.Float32(sV[t, v])
                kS_val = acc_kS[idx]
                rhs = sBeta[t] * (v_val - sExpGC[t] * kS_val)
                sNK_A[v, t] = rhs.to(cutlass.BFloat16)
                acc_qS[idx] = acc_qS[idx] * sExpGC[t]
        cute.arch.barrier()

        # ==================== v_new = M @ RHS (LDSM-fed MMA) ====================
        acc_vnew = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        acc_vnew.fill(cutlass.Float32(0.0))

        if cutlass.const_expr(USE_AUTOVEC_MMA or USE_AUTOVEC_VNEW_MMA):
            _mma_full_autovec(
                tiled_mma, thr_mma, sA, sNK_A, acc_vnew, BT // 16,
            )
        else:
            _mma_full_ldsm(
                tiled_mma, smem_tiled_copy_A, smem_tiled_copy_B,
                thr_mma, tidx, sA, sNK_A, acc_vnew, BT // 16,
            )

        if cutlass.const_expr(USE_RHS_VNEW_R2S):
            sV_vnew_layout = cute.make_layout(
                (BT, BV_TILE), stride=(BV_TILE + 8, 1)
            )
            sNK_vnew_layout = cute.make_layout(
                (BT, BV_TILE), stride=(1, BT + 8)
            )
            sNK_vnew = cute.make_tensor(sNK_A.iterator.align(16), sNK_vnew_layout)
            if cutlass.const_expr(not USE_SKIP_SV_VNEW):
                sV_vnew = cute.make_tensor(sV.iterator.align(16), sV_vnew_layout)
                _store_state_frag_r2s(
                    tiled_copy_kdecay_r2s, tidx, acc_vnew, sV_vnew
                )
            if cutlass.const_expr(USE_SCALAR_VNEW_STORE):
                for idx in cutlass.range(cute.size(acc_vnew)):
                    co = tCcC_bv[idx]
                    sNK_A[co[1], co[0]] = acc_vnew[idx].to(cutlass.BFloat16)
            else:
                _store_state_frag_r2s(
                    tiled_copy_state_r2s, tidx, acc_vnew, sNK_vnew
                )
        else:
            for idx in cutlass.range(cute.size(acc_vnew)):
                co = tCcC_bv[idx]
                vnew = acc_vnew[idx].to(cutlass.BFloat16)
                sV[co[0], co[1]] = vnew
                sNK_A[co[1], co[0]] = vnew
        cute.arch.barrier()

        # ==================== INLINE CHUNK_O (LDSM-fed MMA) ====================
        if cutlass.const_expr(USE_AUTOVEC_MMA or USE_AUTOVEC_CHUNKO_MMA):
            _mma_full_autovec(
                tiled_mma, thr_mma, sGQK, sNK_A, acc_qS, BT // 16,
            )
        else:
            _mma_full_ldsm(
                tiled_mma, smem_tiled_copy_A, smem_tiled_copy_B,
                thr_mma, tidx, sGQK, sNK_A, acc_qS, BT // 16,
            )

        if cutlass.const_expr(USE_VECTOR_O_GMEM):
            gO_chunk = cute.local_tile(gO_full, (BT, BV_TILE), (chunk_idx, bid_v))
            _store_acc_o_gmem_vector(
                tiled_copy_o_gmem, tidx, acc_qS, gO_chunk, scale
            )
        else:
            for idx in cutlass.range(cute.size(acc_qS)):
                co = tCcC_bv[idx]
                o_val = scale * acc_qS[idx]
                mO[i_b, t0 + co[0], i_hv, v_off + co[1]] = o_val.to(mO.element_type)

        if cutlass.const_expr(USE_SCALED_VNEW_STATE):
            cute.arch.barrier()
            if cutlass.const_expr(USE_SCALAR_SCALED_VNEW_STORE):
                for idx in cutlass.range(cute.size(acc_vnew)):
                    co = tCcC_bv[idx]
                    row = co[0]
                    v_col = co[1]
                    sNK_A[v_col, row] = (
                        acc_vnew[idx] * sExpDecay[row]
                    ).to(cutlass.BFloat16)
            else:
                sNK_scaled_layout = cute.make_layout(
                (BT, BV_TILE), stride=(1, BT + 8)
                )
                sNK_scaled = cute.make_tensor(sNK_A.iterator.align(16), sNK_scaled_layout)
                _store_scaled_vnew_snk_r2s(
                    tiled_copy_state_r2s, tidx, acc_vnew, sExpDecay, sNK_scaled
                )

        # ==================== STATE UPDATE ====================
        if cutlass.const_expr(USE_STATE_FRAG):
            phi = sExpGC[BT - 1]
            for idx in cutlass.range(cute.size(state0)):
                state0[idx] = phi * state0[idx]
                state1[idx] = phi * state1[idx]
                state2[idx] = phi * state2[idx]
                state3[idx] = phi * state3[idx]

            if cutlass.const_expr(USE_DEDICATED_STATE_SCRATCH):
                if cutlass.const_expr(USE_SCALED_VNEW_STATE):
                    sK_state_layout = cute.make_layout(
                        (BT, BT), stride=(1, K_DIM + 8)
                    )
                    sK_t0 = cute.make_tensor(sK.iterator.align(16), sK_state_layout)
                    sK_t1 = cute.make_tensor((sK.iterator + BT).align(16), sK_state_layout)
                    sK_t2 = cute.make_tensor((sK.iterator + 2 * BT).align(16), sK_state_layout)
                    sK_t3 = cute.make_tensor((sK.iterator + 3 * BT).align(16), sK_state_layout)
                    cute.arch.barrier()
                    if cutlass.const_expr(USE_AUTOVEC_STATE_MMA):
                        _mma_full_autovec(
                            tiled_mma, thr_mma, sK_t0, sNK_A, state0, BT // 16,
                        )
                        _mma_full_autovec(
                            tiled_mma, thr_mma, sK_t1, sNK_A, state1, BT // 16,
                        )
                        _mma_full_autovec(
                            tiled_mma, thr_mma, sK_t2, sNK_A, state2, BT // 16,
                        )
                        _mma_full_autovec(
                            tiled_mma, thr_mma, sK_t3, sNK_A, state3, BT // 16,
                        )
                    else:
                        _mma_full_ldsm(
                            tiled_mma, smem_tiled_copy_A_trans, smem_tiled_copy_B,
                            thr_mma, tidx, sK_t0, sNK_A, state0, BT // 16,
                        )
                        _mma_full_ldsm(
                            tiled_mma, smem_tiled_copy_A_trans, smem_tiled_copy_B,
                            thr_mma, tidx, sK_t1, sNK_A, state1, BT // 16,
                        )
                        _mma_full_ldsm(
                            tiled_mma, smem_tiled_copy_A_trans, smem_tiled_copy_B,
                            thr_mma, tidx, sK_t2, sNK_A, state2, BT // 16,
                        )
                        _mma_full_ldsm(
                            tiled_mma, smem_tiled_copy_A_trans, smem_tiled_copy_B,
                            thr_mma, tidx, sK_t3, sNK_A, state3, BT // 16,
                        )
                else:
                    if cutlass.const_expr(USE_KDECAY_R2S):
                        sScratch_layout = cute.make_layout(
                            (BT, BV_TILE), stride=(BT + 8, 1)
                        )
                        sA_t0 = cute.make_tensor(sA.iterator.align(16), sScratch_layout)
                        sA_t1 = cute.make_tensor((sA.iterator + BV_TILE).align(16), sScratch_layout)
                        sA2_t0 = cute.make_tensor(sA2.iterator.align(16), sScratch_layout)
                        sA2_t1 = cute.make_tensor((sA2.iterator + BV_TILE).align(16), sScratch_layout)
                        _store_kdecay_scratch_r2s(
                            tiled_copy_kdecay_r2s, tidx, sK, sExpDecay, sA_t0, 0, 0
                        )
                        _store_kdecay_scratch_r2s(
                            tiled_copy_kdecay_r2s, tidx, sK, sExpDecay, sA_t1, 0, BV_TILE
                        )
                        _store_kdecay_scratch_r2s(
                            tiled_copy_kdecay_r2s, tidx, sK, sExpDecay, sA2_t0, BT, 0
                        )
                        _store_kdecay_scratch_r2s(
                            tiled_copy_kdecay_r2s, tidx, sK, sExpDecay, sA2_t1, BT, BV_TILE
                        )
                    else:
                        for i in cutlass.range_constexpr(PER_A):
                            idx = i * NUM_THREADS + tidx
                            row = idx // BT
                            t = idx % BT
                            sA[row, t] = (
                                cutlass.Float32(sK[t, row]) * sExpDecay[t]
                            ).to(cutlass.BFloat16)
                            sA2[row, t] = (
                                cutlass.Float32(sK[t, BT + row]) * sExpDecay[t]
                            ).to(cutlass.BFloat16)
                    cute.arch.barrier()
                    _mma_full_ldsm(
                        tiled_mma, smem_tiled_copy_A, smem_tiled_copy_B,
                        thr_mma, tidx, sA, sNK_A, state0, BT // 16,
                    )
                    _mma_full_ldsm(
                        tiled_mma, smem_tiled_copy_A, smem_tiled_copy_B,
                        thr_mma, tidx, sA2, sNK_A, state1, BT // 16,
                    )
                    cute.arch.barrier()

                    if cutlass.const_expr(USE_KDECAY_R2S):
                        sScratch_layout = cute.make_layout(
                            (BT, BV_TILE), stride=(BT + 8, 1)
                        )
                        sA_t0 = cute.make_tensor(sA.iterator.align(16), sScratch_layout)
                        sA_t1 = cute.make_tensor((sA.iterator + BV_TILE).align(16), sScratch_layout)
                        sA2_t0 = cute.make_tensor(sA2.iterator.align(16), sScratch_layout)
                        sA2_t1 = cute.make_tensor((sA2.iterator + BV_TILE).align(16), sScratch_layout)
                        _store_kdecay_scratch_r2s(
                            tiled_copy_kdecay_r2s, tidx, sK, sExpDecay, sA_t0, 2 * BT, 0
                        )
                        _store_kdecay_scratch_r2s(
                            tiled_copy_kdecay_r2s, tidx, sK, sExpDecay, sA_t1, 2 * BT, BV_TILE
                        )
                        _store_kdecay_scratch_r2s(
                            tiled_copy_kdecay_r2s, tidx, sK, sExpDecay, sA2_t0, 3 * BT, 0
                        )
                        _store_kdecay_scratch_r2s(
                            tiled_copy_kdecay_r2s, tidx, sK, sExpDecay, sA2_t1, 3 * BT, BV_TILE
                        )
                    else:
                        for i in cutlass.range_constexpr(PER_A):
                            idx = i * NUM_THREADS + tidx
                            row = idx // BT
                            t = idx % BT
                            sA[row, t] = (
                                cutlass.Float32(sK[t, 2 * BT + row]) * sExpDecay[t]
                            ).to(cutlass.BFloat16)
                            sA2[row, t] = (
                                cutlass.Float32(sK[t, 3 * BT + row]) * sExpDecay[t]
                            ).to(cutlass.BFloat16)
                    cute.arch.barrier()
                    _mma_full_ldsm(
                        tiled_mma, smem_tiled_copy_A, smem_tiled_copy_B,
                        thr_mma, tidx, sA, sNK_A, state2, BT // 16,
                    )
                    _mma_full_ldsm(
                        tiled_mma, smem_tiled_copy_A, smem_tiled_copy_B,
                        thr_mma, tidx, sA2, sNK_A, state3, BT // 16,
                    )
            elif cutlass.const_expr(USE_DUAL_STATE_BUFFER):
                # sGQK was the chunk-O A operand immediately above. Synchronize
                # before reusing it as scratch for two state-update tiles.
                cute.arch.barrier()
                for i in cutlass.range_constexpr(PER_A):
                    idx = i * NUM_THREADS + tidx
                    row = idx // BT
                    t = idx % BT
                    sA[row, t] = (
                        cutlass.Float32(sK[t, row]) * sExpDecay[t]
                    ).to(cutlass.BFloat16)
                    sGQK[row, t] = (
                        cutlass.Float32(sK[t, BT + row]) * sExpDecay[t]
                    ).to(cutlass.BFloat16)
                cute.arch.barrier()
                _mma_full_ldsm(
                    tiled_mma, smem_tiled_copy_A, smem_tiled_copy_B,
                    thr_mma, tidx, sA, sNK_A, state0, BT // 16,
                )
                _mma_full_ldsm(
                    tiled_mma, smem_tiled_copy_A, smem_tiled_copy_B,
                    thr_mma, tidx, sGQK, sNK_A, state1, BT // 16,
                )
                cute.arch.barrier()

                for i in cutlass.range_constexpr(PER_A):
                    idx = i * NUM_THREADS + tidx
                    row = idx // BT
                    t = idx % BT
                    sA[row, t] = (
                        cutlass.Float32(sK[t, 2 * BT + row]) * sExpDecay[t]
                    ).to(cutlass.BFloat16)
                    sGQK[row, t] = (
                        cutlass.Float32(sK[t, 3 * BT + row]) * sExpDecay[t]
                    ).to(cutlass.BFloat16)
                cute.arch.barrier()
                _mma_full_ldsm(
                    tiled_mma, smem_tiled_copy_A, smem_tiled_copy_B,
                    thr_mma, tidx, sA, sNK_A, state2, BT // 16,
                )
                _mma_full_ldsm(
                    tiled_mma, smem_tiled_copy_A, smem_tiled_copy_B,
                    thr_mma, tidx, sGQK, sNK_A, state3, BT // 16,
                )
            else:
                for i in cutlass.range_constexpr(PER_A):
                    idx = i * NUM_THREADS + tidx
                    row = idx // BT
                    t = idx % BT
                    sA[row, t] = (
                        cutlass.Float32(sK[t, row]) * sExpDecay[t]
                    ).to(cutlass.BFloat16)
                cute.arch.barrier()
                _mma_full_ldsm(
                    tiled_mma, smem_tiled_copy_A, smem_tiled_copy_B,
                    thr_mma, tidx, sA, sNK_A, state0, BT // 16,
                )
                cute.arch.barrier()

                for i in cutlass.range_constexpr(PER_A):
                    idx = i * NUM_THREADS + tidx
                    row = idx // BT
                    t = idx % BT
                    sA[row, t] = (
                        cutlass.Float32(sK[t, BT + row]) * sExpDecay[t]
                    ).to(cutlass.BFloat16)
                cute.arch.barrier()
                _mma_full_ldsm(
                    tiled_mma, smem_tiled_copy_A, smem_tiled_copy_B,
                    thr_mma, tidx, sA, sNK_A, state1, BT // 16,
                )
                cute.arch.barrier()

                for i in cutlass.range_constexpr(PER_A):
                    idx = i * NUM_THREADS + tidx
                    row = idx // BT
                    t = idx % BT
                    sA[row, t] = (
                        cutlass.Float32(sK[t, 2 * BT + row]) * sExpDecay[t]
                    ).to(cutlass.BFloat16)
                cute.arch.barrier()
                _mma_full_ldsm(
                    tiled_mma, smem_tiled_copy_A, smem_tiled_copy_B,
                    thr_mma, tidx, sA, sNK_A, state2, BT // 16,
                )
                cute.arch.barrier()

                for i in cutlass.range_constexpr(PER_A):
                    idx = i * NUM_THREADS + tidx
                    row = idx // BT
                    t = idx % BT
                    sA[row, t] = (
                        cutlass.Float32(sK[t, 3 * BT + row]) * sExpDecay[t]
                    ).to(cutlass.BFloat16)
                cute.arch.barrier()
                _mma_full_ldsm(
                    tiled_mma, smem_tiled_copy_A, smem_tiled_copy_B,
                    thr_mma, tidx, sA, sNK_A, state3, BT // 16,
                )
        else:
            phi = sExpGC[BT - 1]
            for v in cutlass.range_constexpr(BV_TILE):
                state[v] = phi * state[v]
            for t in cutlass.range_constexpr(BT):
                kd = cutlass.Float32(sK[t, tidx]) * sExpDecay[t]
                for v in cutlass.range_constexpr(BV_TILE):
                    state[v] = state[v] + kd * cutlass.Float32(sV[t, v])

        cute.arch.barrier()


@cute.kernel
def fused_chunk_h_mgqk_v_ldsm_state_mma_probe_kernel(
    tiled_mma: cute.TiledMma,
    smem_tiled_copy_A: cute.TiledCopy,
    smem_tiled_copy_B: cute.TiledCopy,
    tiled_copy_kq: cute.TiledCopy,
    tiled_copy_mgqk: cute.TiledCopy,
    tiled_copy_v: cute.TiledCopy,
    sK_layout: cute.Layout,
    sV_layout: cute.Layout, sA_layout: cute.Layout,
    sGQK_layout: cute.Layout,
    sBeta_layout: cute.Layout,
    sS_layout: cute.Layout, sNK_layout: cute.Layout,
    sState_layout: cute.Layout,
    sExpGC_layout: cute.Layout, sExpDecay_layout: cute.Layout,
    mKnorm: cute.Tensor, mQnorm: cute.Tensor,
    mV: cute.Tensor, mBeta: cute.Tensor,
    mM: cute.Tensor, mGQK: cute.Tensor,
    mExpGC_in: cute.Tensor, mExpDecay_in: cute.Tensor,
    mO: cute.Tensor,
    scale: cutlass.Constexpr[float],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int], HV: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int], NT: cutlass.Constexpr[int],
    H_PER_HV: cutlass.Constexpr[int],
    BV_TILE: cutlass.Constexpr[int],
    PER_V_TILE: cutlass.Constexpr[int],
    PER_STATE_TILE: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    bid_v = cute.arch.block_idx()[0]
    bid_bh = cute.arch.block_idx()[1]
    i_b = bid_bh // HV
    i_hv = bid_bh % HV
    i_h = i_hv // H_PER_HV
    v_off = bid_v * BV_TILE

    smem = cutlass.utils.SmemAllocator()
    sK = smem.allocate_tensor(cutlass.BFloat16, sK_layout, 16)
    sQ = smem.allocate_tensor(cutlass.BFloat16, sK_layout, 16)
    sV = smem.allocate_tensor(cutlass.BFloat16, sV_layout, 16)
    sA = smem.allocate_tensor(cutlass.BFloat16, sA_layout, 16)
    sGQK = smem.allocate_tensor(cutlass.BFloat16, sGQK_layout, 16)
    sBeta = smem.allocate_tensor(cutlass.Float32, sBeta_layout, 16)
    sS = smem.allocate_tensor(cutlass.BFloat16, sS_layout, 16)
    sNK_A = smem.allocate_tensor(cutlass.BFloat16, sNK_layout, 16)
    sState = smem.allocate_tensor(cutlass.Float32, sState_layout, 16)
    sExpGC = smem.allocate_tensor(cutlass.Float32, sExpGC_layout, 16)
    sExpDecay = smem.allocate_tensor(cutlass.Float32, sExpDecay_layout, 16)

    thr_mma = tiled_mma.get_slice(tidx)

    cC_bv = cute.make_identity_tensor((BT, BV_TILE))
    tCcC_bv = thr_mma.partition_C(cC_bv)

    for i in cutlass.range_constexpr(PER_STATE_TILE):
        idx = i * NUM_THREADS + tidx
        k_row = idx // BV_TILE
        v_col = idx % BV_TILE
        sState[k_row, v_col] = cutlass.Float32(0.0)
    cute.arch.barrier()

    gK_full = mKnorm[(i_b, i_h, None, None)]
    gQ_full = mQnorm[(i_b, i_h, None, None)]
    gV_full = mV[(i_b, None, i_hv, None)]
    thr_cp = tiled_copy_kq.get_slice(tidx)
    thr_sK_cp = thr_cp.partition_D(sK)
    thr_sQ_cp = thr_cp.partition_D(sQ)

    thr_cp_mgqk = tiled_copy_mgqk.get_slice(tidx)
    thr_sA_cp = thr_cp_mgqk.partition_D(sA)
    thr_sGQK_cp = thr_cp_mgqk.partition_D(sGQK)

    for chunk_idx in cutlass.range(NT):
        t0 = chunk_idx * BT

        # ==================== cp.async K/Q/M/GQK/V ====================
        gK_chunk = cute.local_tile(gK_full, (BT, K_DIM), (chunk_idx, 0))
        gQ_chunk = cute.local_tile(gQ_full, (BT, K_DIM), (chunk_idx, 0))
        thr_gK = thr_cp.partition_S(gK_chunk)
        thr_gQ = thr_cp.partition_S(gQ_chunk)
        cute.copy(tiled_copy_kq, thr_gK, thr_sK_cp)
        cute.copy(tiled_copy_kq, thr_gQ, thr_sQ_cp)

        gM_chunk = mM[(i_b, chunk_idx, i_hv, None, None)]
        gGQK_chunk = mGQK[(i_b, chunk_idx, i_hv, None, None)]
        thr_gM = thr_cp_mgqk.partition_S(gM_chunk)
        thr_gGQK = thr_cp_mgqk.partition_S(gGQK_chunk)
        cute.copy(tiled_copy_mgqk, thr_gM, thr_sA_cp)
        cute.copy(tiled_copy_mgqk, thr_gGQK, thr_sGQK_cp)

        if tidx < 64:
            gV_chunk = cute.local_tile(gV_full, (BT, BV_TILE), (chunk_idx, bid_v))
            thr_cp_v = tiled_copy_v.get_slice(tidx)
            thr_sV_cp = thr_cp_v.partition_D(sV)
            thr_gV = thr_cp_v.partition_S(gV_chunk)
            cute.copy(tiled_copy_v, thr_gV, thr_sV_cp)
        cute.arch.cp_async_commit_group()

        if tidx < BT:
            sExpGC[tidx] = mExpGC_in[i_b, chunk_idx, i_hv, tidx]
            sExpDecay[tidx] = mExpDecay_in[i_b, chunk_idx, i_hv, tidx]
            sBeta[tidx] = cutlass.Float32(mBeta[i_b, t0 + tidx, i_hv])

        for i in cutlass.range_constexpr(PER_STATE_TILE):
            idx = i * NUM_THREADS + tidx
            k_row = idx // BV_TILE
            v_col = idx % BV_TILE
            sS[v_col, k_row] = sState[k_row, v_col].to(cutlass.BFloat16)

        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        # ==================== kS + qS (LDSM-fed MMA) ====================
        acc_kS = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        acc_kS.fill(cutlass.Float32(0.0))
        acc_qS = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        acc_qS.fill(cutlass.Float32(0.0))

        _mma_full_ldsm(
            tiled_mma, smem_tiled_copy_A, smem_tiled_copy_B,
            thr_mma, tidx, sK, sS, acc_kS, K_DIM // 16,
        )
        _mma_full_autovec(
            tiled_mma, thr_mma, sQ, sS, acc_qS, K_DIM // 16,
        )
        if cutlass.const_expr(STATE_SPLIT):
            if chunk_idx >= STATE_SPLIT_START_CHUNK:
                _add_qs_state_lo_rows_scalar(acc_qS, tCcC_bv, sQ, sS_lo)

        # ==================== RHS + TRANSPOSE ====================
        for idx in cutlass.range(cute.size(acc_kS)):
            co = tCcC_bv[idx]
            t = co[0]
            v = co[1]
            v_val = cutlass.Float32(sV[t, v])
            kS_val = acc_kS[idx]
            rhs = sBeta[t] * (v_val - sExpGC[t] * kS_val)
            sNK_A[v, t] = rhs.to(cutlass.BFloat16)
            acc_qS[idx] = acc_qS[idx] * sExpGC[t]
        cute.arch.barrier()

        # ==================== v_new = M @ RHS (LDSM-fed MMA) ====================
        acc_vnew = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        acc_vnew.fill(cutlass.Float32(0.0))

        _mma_full_autovec(
            tiled_mma, thr_mma, sA, sNK_A, acc_vnew, BT // 16,
        )

        for idx in cutlass.range(cute.size(acc_vnew)):
            co = tCcC_bv[idx]
            vnew = acc_vnew[idx].to(cutlass.BFloat16)
            sV[co[0], co[1]] = vnew
            sNK_A[co[1], co[0]] = vnew
        cute.arch.barrier()

        # ==================== INLINE CHUNK_O (LDSM-fed MMA) ====================
        _mma_full_autovec(
            tiled_mma, thr_mma, sGQK, sNK_A, acc_qS, BT // 16,
        )

        for idx in cutlass.range(cute.size(acc_qS)):
            co = tCcC_bv[idx]
            o_val = scale * acc_qS[idx]
            mO[i_b, t0 + co[0], i_hv, v_off + co[1]] = o_val.to(mO.element_type)

        # ==================== STATE UPDATE BY HMMA ====================
        phi = sExpGC[BT - 1]
        for k_block in cutlass.range_constexpr(K_DIM // BT):
            for i in cutlass.range_constexpr(PER_A):
                idx = i * NUM_THREADS + tidx
                row = idx // BT
                t = idx % BT
                sA[row, t] = (
                    cutlass.Float32(sK[t, k_block * BT + row]) * sExpDecay[t]
                ).to(cutlass.BFloat16)
            cute.arch.barrier()

            acc_state = cute.make_rmem_tensor(
                thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
            acc_state.fill(cutlass.Float32(0.0))
            _mma_full_ldsm(
                tiled_mma, smem_tiled_copy_A, smem_tiled_copy_B,
                thr_mma, tidx, sA, sNK_A, acc_state, BT // 16,
            )

            for idx in cutlass.range(cute.size(acc_state)):
                co = tCcC_bv[idx]
                k_row = k_block * BT + co[0]
                v_col = co[1]
                sState[k_row, v_col] = phi * sState[k_row, v_col] + acc_state[idx]
            cute.arch.barrier()


@cute.kernel
def chunk_h_store_vnew_cpasync_probe_kernel(
    tiled_mma: cute.TiledMma,
    tiled_copy_k: cute.TiledCopy,
    tiled_copy_m: cute.TiledCopy,
    tiled_copy_v: cute.TiledCopy,
    sK_layout: cute.Layout,
    sV_layout: cute.Layout,
    sA_layout: cute.Layout,
    sBeta_layout: cute.Layout,
    sS_layout: cute.Layout,
    sNK_layout: cute.Layout,
    sExpGC_layout: cute.Layout,
    sExpDecay_layout: cute.Layout,
    mKnorm: cute.Tensor,
    mV: cute.Tensor,
    mBeta: cute.Tensor,
    mM: cute.Tensor,
    mExpGC_in: cute.Tensor,
    mExpDecay_in: cute.Tensor,
    mH: cute.Tensor,
    mVNew: cute.Tensor,
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int], HV: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int], NT: cutlass.Constexpr[int],
    H_PER_HV: cutlass.Constexpr[int],
    BV_TILE: cutlass.Constexpr[int],
    PER_V_TILE: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    bid_v = cute.arch.block_idx()[0]
    bid_bh = cute.arch.block_idx()[1]
    i_b = bid_bh // HV
    i_hv = bid_bh % HV
    i_h = i_hv // H_PER_HV
    v_off = bid_v * BV_TILE

    smem = cutlass.utils.SmemAllocator()
    sK = smem.allocate_tensor(cutlass.BFloat16, sK_layout, 16)
    sV = smem.allocate_tensor(cutlass.BFloat16, sV_layout, 16)
    sA = smem.allocate_tensor(cutlass.BFloat16, sA_layout, 16)
    sBeta = smem.allocate_tensor(cutlass.Float32, sBeta_layout, 16)
    sS = smem.allocate_tensor(cutlass.BFloat16, sS_layout, 16)
    sNK_A = smem.allocate_tensor(cutlass.BFloat16, sNK_layout, 16)
    sExpGC = smem.allocate_tensor(cutlass.Float32, sExpGC_layout, 16)
    sExpDecay = smem.allocate_tensor(cutlass.Float32, sExpDecay_layout, 16)

    state = cute.make_rmem_tensor(cute.make_layout((BV_TILE,)), cutlass.Float32)
    state.fill(cutlass.Float32(0.0))

    thr_mma = tiled_mma.get_slice(tidx)
    cC_bv = cute.make_identity_tensor((BT, BV_TILE))
    tCcC_bv = thr_mma.partition_C(cC_bv)

    gK_full = mKnorm[(i_b, i_h, None, None)]
    gV_full = mV[(i_b, None, i_hv, None)]

    thr_cp_k = tiled_copy_k.get_slice(tidx)
    thr_sK_cp = thr_cp_k.partition_D(sK)

    thr_cp_m = tiled_copy_m.get_slice(tidx)
    thr_sA_cp = thr_cp_m.partition_D(sA)

    for chunk_idx in cutlass.range(NT):
        t0 = chunk_idx * BT

        gK_chunk = cute.local_tile(gK_full, (BT, K_DIM), (chunk_idx, 0))
        thr_gK = thr_cp_k.partition_S(gK_chunk)
        cute.copy(tiled_copy_k, thr_gK, thr_sK_cp)

        gM_chunk = mM[(i_b, chunk_idx, i_hv, None, None)]
        thr_gM = thr_cp_m.partition_S(gM_chunk)
        cute.copy(tiled_copy_m, thr_gM, thr_sA_cp)

        if tidx < 64:
            gV_chunk = cute.local_tile(gV_full, (BT, BV_TILE), (chunk_idx, bid_v))
            thr_cp_v = tiled_copy_v.get_slice(tidx)
            thr_sV_cp = thr_cp_v.partition_D(sV)
            thr_gV = thr_cp_v.partition_S(gV_chunk)
            cute.copy(tiled_copy_v, thr_gV, thr_sV_cp)
        cute.arch.cp_async_commit_group()

        if tidx < BT:
            sExpGC[tidx] = mExpGC_in[i_b, chunk_idx, i_hv, tidx]
            sExpDecay[tidx] = mExpDecay_in[i_b, chunk_idx, i_hv, tidx]
            sBeta[tidx] = cutlass.Float32(mBeta[i_b, t0 + tidx, i_hv])

        for v in cutlass.range_constexpr(BV_TILE):
            h_val = state[v].to(cutlass.BFloat16)
            sS[v, tidx] = h_val
            mH[i_b, chunk_idx, i_hv, v_off + v, tidx] = h_val

        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        acc_kS = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        acc_kS.fill(cutlass.Float32(0.0))
        _mma_AS_full(tiled_mma, thr_mma, sK, sS, acc_kS)

        for idx in cutlass.range(cute.size(acc_kS)):
            co = tCcC_bv[idx]
            t = co[0]
            v = co[1]
            v_val = cutlass.Float32(sV[t, v])
            rhs = sBeta[t] * (v_val - sExpGC[t] * acc_kS[idx])
            sNK_A[v, t] = rhs.to(cutlass.BFloat16)
        cute.arch.barrier()

        acc_vnew = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        acc_vnew.fill(cutlass.Float32(0.0))

        tCsMA = thr_mma.partition_A(sA)
        tCsNKB = thr_mma.partition_B(sNK_A)
        tCrMA = thr_mma.make_fragment_A(tCsMA)
        tCrNKB = thr_mma.make_fragment_B(tCsNKB)
        for kk in cutlass.range_constexpr(BT // 16):
            cute.autovec_copy(tCsMA[None, None, kk], tCrMA[None, None, kk])
            cute.autovec_copy(tCsNKB[None, None, kk], tCrNKB[None, None, kk])
            cute.gemm(tiled_mma, acc_vnew, tCrMA[None, None, kk], tCrNKB[None, None, kk], acc_vnew)

        for idx in cutlass.range(cute.size(acc_vnew)):
            co = tCcC_bv[idx]
            vnew = acc_vnew[idx].to(cutlass.BFloat16)
            sV[co[0], co[1]] = vnew
            mVNew[i_b, chunk_idx, i_hv, v_off + co[1], co[0]] = vnew
        cute.arch.barrier()

        if tidx < K_DIM:
            phi = sExpGC[BT - 1]
            for v in cutlass.range_constexpr(BV_TILE):
                state[v] = phi * state[v]
            for t in cutlass.range_constexpr(BT):
                kd = cutlass.Float32(sK[t, tidx]) * sExpDecay[t]
                for v in cutlass.range_constexpr(BV_TILE):
                    state[v] = state[v] + kd * cutlass.Float32(sV[t, v])

        cute.arch.barrier()


@cute.kernel
def chunk_o_split_cpasync_probe_kernel(
    tiled_mma: cute.TiledMma,
    tiled_copy_q: cute.TiledCopy,
    tiled_copy_h: cute.TiledCopy,
    tiled_copy_a: cute.TiledCopy,
    tiled_copy_v: cute.TiledCopy,
    sQ_layout: cute.Layout,
    sH_layout: cute.Layout,
    sA_layout: cute.Layout,
    sV_layout: cute.Layout,
    sExpGC_layout: cute.Layout,
    mQnorm: cute.Tensor,
    mH: cute.Tensor,
    mGQK: cute.Tensor,
    mVNew: cute.Tensor,
    mExpGC_in: cute.Tensor,
    mO: cute.Tensor,
    scale: cutlass.Constexpr[float],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int], HV: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int], NT: cutlass.Constexpr[int],
    H_PER_HV: cutlass.Constexpr[int],
    BV_TILE: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    bid_v = cute.arch.block_idx()[0]
    chunk_idx = cute.arch.block_idx()[1]
    bid_bh = cute.arch.block_idx()[2]
    i_b = bid_bh // HV
    i_hv = bid_bh % HV
    i_h = i_hv // H_PER_HV
    v_off = bid_v * BV_TILE
    t0 = chunk_idx * BT

    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(cutlass.BFloat16, sQ_layout, 16)
    sH = smem.allocate_tensor(cutlass.BFloat16, sH_layout, 16)
    sA = smem.allocate_tensor(cutlass.BFloat16, sA_layout, 16)
    sV = smem.allocate_tensor(cutlass.BFloat16, sV_layout, 16)
    sExpGC = smem.allocate_tensor(cutlass.Float32, sExpGC_layout, 16)

    thr_mma = tiled_mma.get_slice(tidx)
    cC_bv = cute.make_identity_tensor((BT, BV_TILE))
    tCcC_bv = thr_mma.partition_C(cC_bv)

    gQ_full = mQnorm[(i_b, i_h, None, None)]
    gQ_chunk = cute.local_tile(gQ_full, (BT, K_DIM), (chunk_idx, 0))
    thr_cp_q = tiled_copy_q.get_slice(tidx)
    thr_gQ = thr_cp_q.partition_S(gQ_chunk)
    thr_sQ = thr_cp_q.partition_D(sQ)
    cute.copy(tiled_copy_q, thr_gQ, thr_sQ)

    gA_chunk = mGQK[(i_b, chunk_idx, i_hv, None, None)]
    thr_cp_a = tiled_copy_a.get_slice(tidx)
    thr_gA = thr_cp_a.partition_S(gA_chunk)
    thr_sA = thr_cp_a.partition_D(sA)
    cute.copy(tiled_copy_a, thr_gA, thr_sA)

    cute.arch.cp_async_commit_group()

    for i in cutlass.range_constexpr((BV_TILE * K_DIM) // NUM_THREADS):
        idx = i * NUM_THREADS + tidx
        row = idx // K_DIM
        col = idx % K_DIM
        sH[row, col] = mH[i_b, chunk_idx, i_hv, v_off + row, col]

    for i in cutlass.range_constexpr((BV_TILE * BT) // NUM_THREADS):
        idx = i * NUM_THREADS + tidx
        row = idx // BT
        col = idx % BT
        sV[row, col] = mVNew[i_b, chunk_idx, i_hv, v_off + row, col]

    if tidx < BT:
        sExpGC[tidx] = mExpGC_in[i_b, chunk_idx, i_hv, tidx]

    cute.arch.cp_async_wait_group(0)
    cute.arch.barrier()

    acc_o = cute.make_rmem_tensor(
        thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
    acc_o.fill(cutlass.Float32(0.0))
    _mma_AS_full(tiled_mma, thr_mma, sQ, sH, acc_o)

    for idx in cutlass.range(cute.size(acc_o)):
        co = tCcC_bv[idx]
        acc_o[idx] = acc_o[idx] * sExpGC[co[0]]

    tCsAA = thr_mma.partition_A(sA)
    tCsVB = thr_mma.partition_B(sV)
    tCrAA = thr_mma.make_fragment_A(tCsAA)
    tCrVB = thr_mma.make_fragment_B(tCsVB)
    for kk in cutlass.range_constexpr(BT // 16):
        cute.autovec_copy(tCsAA[None, None, kk], tCrAA[None, None, kk])
        cute.autovec_copy(tCsVB[None, None, kk], tCrVB[None, None, kk])
        cute.gemm(tiled_mma, acc_o, tCrAA[None, None, kk], tCrVB[None, None, kk], acc_o)

    for idx in cutlass.range(cute.size(acc_o)):
        co = tCcC_bv[idx]
        o_val = scale * acc_o[idx]
        mO[i_b, t0 + co[0], i_hv, v_off + co[1]] = o_val.to(mO.element_type)


@cute.kernel
def fused_chunk_h_m_cpasync_probe_kernel(
    tiled_mma: cute.TiledMma,
    tiled_copy_kq: cute.TiledCopy,
    tiled_copy_m: cute.TiledCopy,
    sK_layout: cute.Layout,
    sV_layout: cute.Layout, sA_layout: cute.Layout,
    sGQK_layout: cute.Layout,
    sBeta_layout: cute.Layout,
    sS_layout: cute.Layout, sNK_layout: cute.Layout,
    sExpGC_layout: cute.Layout, sExpDecay_layout: cute.Layout,
    mKnorm: cute.Tensor, mQnorm: cute.Tensor,
    mV: cute.Tensor, mBeta: cute.Tensor,
    mM: cute.Tensor, mGQK: cute.Tensor,
    mExpGC_in: cute.Tensor, mExpDecay_in: cute.Tensor,
    mO: cute.Tensor,
    scale: cutlass.Constexpr[float],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int], HV: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int], NT: cutlass.Constexpr[int],
    H_PER_HV: cutlass.Constexpr[int],
    BV_TILE: cutlass.Constexpr[int],
    PER_V_TILE: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    bid_v = cute.arch.block_idx()[0]
    bid_bh = cute.arch.block_idx()[1]
    i_b = bid_bh // HV
    i_hv = bid_bh % HV
    i_h = i_hv // H_PER_HV
    v_off = bid_v * BV_TILE

    smem = cutlass.utils.SmemAllocator()
    sK = smem.allocate_tensor(cutlass.BFloat16, sK_layout, 16)
    sQ = smem.allocate_tensor(cutlass.BFloat16, sK_layout, 16)
    sV = smem.allocate_tensor(cutlass.BFloat16, sV_layout, 16)
    sA = smem.allocate_tensor(cutlass.BFloat16, sA_layout, 16)
    sGQK = smem.allocate_tensor(cutlass.BFloat16, sGQK_layout, 16)
    sBeta = smem.allocate_tensor(cutlass.Float32, sBeta_layout, 16)
    sS = smem.allocate_tensor(cutlass.BFloat16, sS_layout, 16)
    sNK_A = smem.allocate_tensor(cutlass.BFloat16, sNK_layout, 16)
    sExpGC = smem.allocate_tensor(cutlass.Float32, sExpGC_layout, 16)
    sExpDecay = smem.allocate_tensor(cutlass.Float32, sExpDecay_layout, 16)

    state = cute.make_rmem_tensor(cute.make_layout((BV_TILE,)), cutlass.Float32)
    state.fill(cutlass.Float32(0.0))

    thr_mma = tiled_mma.get_slice(tidx)

    cC_bv = cute.make_identity_tensor((BT, BV_TILE))
    tCcC_bv = thr_mma.partition_C(cC_bv)

    gK_full = mKnorm[(i_b, i_h, None, None)]
    gQ_full = mQnorm[(i_b, i_h, None, None)]
    thr_cp = tiled_copy_kq.get_slice(tidx)
    thr_sK_cp = thr_cp.partition_D(sK)
    thr_sQ_cp = thr_cp.partition_D(sQ)

    thr_cp_m = tiled_copy_m.get_slice(tidx)
    thr_sA_cp = thr_cp_m.partition_D(sA)

    for chunk_idx in cutlass.range(NT):
        t0 = chunk_idx * BT

        # ==================== cp.async K/Q/M ====================
        gK_chunk = cute.local_tile(gK_full, (BT, K_DIM), (chunk_idx, 0))
        gQ_chunk = cute.local_tile(gQ_full, (BT, K_DIM), (chunk_idx, 0))
        thr_gK = thr_cp.partition_S(gK_chunk)
        thr_gQ = thr_cp.partition_S(gQ_chunk)
        cute.copy(tiled_copy_kq, thr_gK, thr_sK_cp)
        cute.copy(tiled_copy_kq, thr_gQ, thr_sQ_cp)

        gM_chunk = mM[(i_b, chunk_idx, i_hv, None, None)]
        thr_gM = thr_cp_m.partition_S(gM_chunk)
        cute.copy(tiled_copy_m, thr_gM, thr_sA_cp)
        cute.arch.cp_async_commit_group()

        # ==================== Overlap scalar/GQK/V loads while cp.async flies =
        for i in cutlass.range_constexpr(PER_A):
            idx = i * NUM_THREADS + tidx
            row = idx // BT
            col = idx % BT
            sGQK[row, col] = mGQK[i_b, chunk_idx, i_hv, row, col]

        for i in cutlass.range_constexpr(PER_V_TILE):
            idx = i * NUM_THREADS + tidx
            r = idx // BV_TILE
            c = idx % BV_TILE
            sV[r, c] = mV[i_b, t0 + r, i_hv, v_off + c]

        if tidx < BT:
            sExpGC[tidx] = mExpGC_in[i_b, chunk_idx, i_hv, tidx]
            sExpDecay[tidx] = mExpDecay_in[i_b, chunk_idx, i_hv, tidx]
            sBeta[tidx] = cutlass.Float32(mBeta[i_b, t0 + tidx, i_hv])

        for v in cutlass.range_constexpr(BV_TILE):
            sS[v, tidx] = state[v].to(cutlass.BFloat16)

        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        # ==================== kS + qS (full-width MMA) ====================
        acc_kS = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        acc_kS.fill(cutlass.Float32(0.0))
        acc_qS = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        acc_qS.fill(cutlass.Float32(0.0))

        _mma_AS_full(tiled_mma, thr_mma, sK, sS, acc_kS)
        _mma_AS_full(tiled_mma, thr_mma, sQ, sS, acc_qS)

        # ==================== RHS + TRANSPOSE ====================
        for idx in cutlass.range(cute.size(acc_kS)):
            co = tCcC_bv[idx]
            t = co[0]
            v = co[1]
            v_val = cutlass.Float32(sV[t, v])
            kS_val = acc_kS[idx]
            rhs = sBeta[t] * (v_val - sExpGC[t] * kS_val)
            sNK_A[v, t] = rhs.to(cutlass.BFloat16)
            acc_qS[idx] = acc_qS[idx] * sExpGC[t]
        cute.arch.barrier()

        # ==================== v_new = M @ RHS (single MMA) ====================
        acc_vnew = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        acc_vnew.fill(cutlass.Float32(0.0))

        tCsMA = thr_mma.partition_A(sA)
        tCsNKB = thr_mma.partition_B(sNK_A)
        tCrMA = thr_mma.make_fragment_A(tCsMA)
        tCrNKB = thr_mma.make_fragment_B(tCsNKB)
        for kk in cutlass.range_constexpr(BT // 16):
            cute.autovec_copy(tCsMA[None, None, kk], tCrMA[None, None, kk])
            cute.autovec_copy(tCsNKB[None, None, kk], tCrNKB[None, None, kk])
            cute.gemm(tiled_mma, acc_vnew, tCrMA[None, None, kk], tCrNKB[None, None, kk], acc_vnew)

        for idx in cutlass.range(cute.size(acc_vnew)):
            co = tCcC_bv[idx]
            sV[co[0], co[1]] = acc_vnew[idx].to(cutlass.BFloat16)
            sNK_A[co[1], co[0]] = acc_vnew[idx].to(cutlass.BFloat16)
        cute.arch.barrier()

        # ==================== INLINE CHUNK_O ====================
        tCsAA = thr_mma.partition_A(sGQK)
        tCsNKB2 = thr_mma.partition_B(sNK_A)
        tCrAA = thr_mma.make_fragment_A(tCsAA)
        tCrNKB2 = thr_mma.make_fragment_B(tCsNKB2)
        for kk in cutlass.range_constexpr(BT // 16):
            cute.autovec_copy(tCsAA[None, None, kk], tCrAA[None, None, kk])
            cute.autovec_copy(tCsNKB2[None, None, kk], tCrNKB2[None, None, kk])
            cute.gemm(tiled_mma, acc_qS, tCrAA[None, None, kk], tCrNKB2[None, None, kk], acc_qS)

        for idx in cutlass.range(cute.size(acc_qS)):
            co = tCcC_bv[idx]
            o_val = scale * acc_qS[idx]
            mO[i_b, t0 + co[0], i_hv, v_off + co[1]] = o_val.to(mO.element_type)

        # ==================== STATE UPDATE ====================
        if tidx < K_DIM:
            phi = sExpGC[BT - 1]
            for v in cutlass.range_constexpr(BV_TILE):
                state[v] = phi * state[v]
            for t in cutlass.range_constexpr(BT):
                kd = cutlass.Float32(sK[t, tidx]) * sExpDecay[t]
                for v in cutlass.range_constexpr(BV_TILE):
                    state[v] = state[v] + kd * cutlass.Float32(sV[t, v])

        cute.arch.barrier()


@cute.kernel
def fused_chunk_h_gqk_cpasync_probe_kernel(
    tiled_mma: cute.TiledMma,
    tiled_copy_kq: cute.TiledCopy,
    tiled_copy_gqk: cute.TiledCopy,
    sK_layout: cute.Layout,
    sV_layout: cute.Layout, sA_layout: cute.Layout,
    sGQK_layout: cute.Layout,
    sBeta_layout: cute.Layout,
    sS_layout: cute.Layout, sNK_layout: cute.Layout,
    sExpGC_layout: cute.Layout, sExpDecay_layout: cute.Layout,
    mKnorm: cute.Tensor, mQnorm: cute.Tensor,
    mV: cute.Tensor, mBeta: cute.Tensor,
    mM: cute.Tensor, mGQK: cute.Tensor,
    mExpGC_in: cute.Tensor, mExpDecay_in: cute.Tensor,
    mO: cute.Tensor,
    scale: cutlass.Constexpr[float],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int], HV: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int], NT: cutlass.Constexpr[int],
    H_PER_HV: cutlass.Constexpr[int],
    BV_TILE: cutlass.Constexpr[int],
    PER_V_TILE: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    bid_v = cute.arch.block_idx()[0]
    bid_bh = cute.arch.block_idx()[1]
    i_b = bid_bh // HV
    i_hv = bid_bh % HV
    i_h = i_hv // H_PER_HV
    v_off = bid_v * BV_TILE

    smem = cutlass.utils.SmemAllocator()
    sK = smem.allocate_tensor(cutlass.BFloat16, sK_layout, 16)
    sQ = smem.allocate_tensor(cutlass.BFloat16, sK_layout, 16)
    sV = smem.allocate_tensor(cutlass.BFloat16, sV_layout, 16)
    sA = smem.allocate_tensor(cutlass.BFloat16, sA_layout, 16)
    sGQK = smem.allocate_tensor(cutlass.BFloat16, sGQK_layout, 16)
    sBeta = smem.allocate_tensor(cutlass.Float32, sBeta_layout, 16)
    sS = smem.allocate_tensor(cutlass.BFloat16, sS_layout, 16)
    sNK_A = smem.allocate_tensor(cutlass.BFloat16, sNK_layout, 16)
    sExpGC = smem.allocate_tensor(cutlass.Float32, sExpGC_layout, 16)
    sExpDecay = smem.allocate_tensor(cutlass.Float32, sExpDecay_layout, 16)

    state = cute.make_rmem_tensor(cute.make_layout((BV_TILE,)), cutlass.Float32)
    state.fill(cutlass.Float32(0.0))

    thr_mma = tiled_mma.get_slice(tidx)

    cC_bv = cute.make_identity_tensor((BT, BV_TILE))
    tCcC_bv = thr_mma.partition_C(cC_bv)

    gK_full = mKnorm[(i_b, i_h, None, None)]
    gQ_full = mQnorm[(i_b, i_h, None, None)]
    thr_cp = tiled_copy_kq.get_slice(tidx)
    thr_sK_cp = thr_cp.partition_D(sK)
    thr_sQ_cp = thr_cp.partition_D(sQ)

    thr_cp_gqk = tiled_copy_gqk.get_slice(tidx)
    thr_sGQK_cp = thr_cp_gqk.partition_D(sGQK)

    for chunk_idx in cutlass.range(NT):
        t0 = chunk_idx * BT

        # ==================== cp.async K/Q/GQK ====================
        gK_chunk = cute.local_tile(gK_full, (BT, K_DIM), (chunk_idx, 0))
        gQ_chunk = cute.local_tile(gQ_full, (BT, K_DIM), (chunk_idx, 0))
        thr_gK = thr_cp.partition_S(gK_chunk)
        thr_gQ = thr_cp.partition_S(gQ_chunk)
        cute.copy(tiled_copy_kq, thr_gK, thr_sK_cp)
        cute.copy(tiled_copy_kq, thr_gQ, thr_sQ_cp)

        gGQK_chunk = mGQK[(i_b, chunk_idx, i_hv, None, None)]
        thr_gGQK = thr_cp_gqk.partition_S(gGQK_chunk)
        cute.copy(tiled_copy_gqk, thr_gGQK, thr_sGQK_cp)
        cute.arch.cp_async_commit_group()

        # ==================== Overlap scalar/M/V loads while cp.async flies ==
        for i in cutlass.range_constexpr(PER_A):
            idx = i * NUM_THREADS + tidx
            row = idx // BT
            col = idx % BT
            sA[row, col] = mM[i_b, chunk_idx, i_hv, row, col]

        for i in cutlass.range_constexpr(PER_V_TILE):
            idx = i * NUM_THREADS + tidx
            r = idx // BV_TILE
            c = idx % BV_TILE
            sV[r, c] = mV[i_b, t0 + r, i_hv, v_off + c]

        if tidx < BT:
            sExpGC[tidx] = mExpGC_in[i_b, chunk_idx, i_hv, tidx]
            sExpDecay[tidx] = mExpDecay_in[i_b, chunk_idx, i_hv, tidx]
            sBeta[tidx] = cutlass.Float32(mBeta[i_b, t0 + tidx, i_hv])

        for v in cutlass.range_constexpr(BV_TILE):
            sS[v, tidx] = state[v].to(cutlass.BFloat16)

        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        # ==================== kS + qS (full-width MMA) ====================
        acc_kS = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        acc_kS.fill(cutlass.Float32(0.0))
        acc_qS = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        acc_qS.fill(cutlass.Float32(0.0))

        _mma_AS_full(tiled_mma, thr_mma, sK, sS, acc_kS)
        _mma_AS_full(tiled_mma, thr_mma, sQ, sS, acc_qS)

        # ==================== RHS + TRANSPOSE ====================
        for idx in cutlass.range(cute.size(acc_kS)):
            co = tCcC_bv[idx]
            t = co[0]
            v = co[1]
            v_val = cutlass.Float32(sV[t, v])
            kS_val = acc_kS[idx]
            rhs = sBeta[t] * (v_val - sExpGC[t] * kS_val)
            sNK_A[v, t] = rhs.to(cutlass.BFloat16)
            acc_qS[idx] = acc_qS[idx] * sExpGC[t]
        cute.arch.barrier()

        # ==================== v_new = M @ RHS (single MMA) ====================
        acc_vnew = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        acc_vnew.fill(cutlass.Float32(0.0))

        tCsMA = thr_mma.partition_A(sA)
        tCsNKB = thr_mma.partition_B(sNK_A)
        tCrMA = thr_mma.make_fragment_A(tCsMA)
        tCrNKB = thr_mma.make_fragment_B(tCsNKB)
        for kk in cutlass.range_constexpr(BT // 16):
            cute.autovec_copy(tCsMA[None, None, kk], tCrMA[None, None, kk])
            cute.autovec_copy(tCsNKB[None, None, kk], tCrNKB[None, None, kk])
            cute.gemm(tiled_mma, acc_vnew, tCrMA[None, None, kk], tCrNKB[None, None, kk], acc_vnew)

        for idx in cutlass.range(cute.size(acc_vnew)):
            co = tCcC_bv[idx]
            sV[co[0], co[1]] = acc_vnew[idx].to(cutlass.BFloat16)
            sNK_A[co[1], co[0]] = acc_vnew[idx].to(cutlass.BFloat16)
        cute.arch.barrier()

        # ==================== INLINE CHUNK_O ====================
        tCsAA = thr_mma.partition_A(sGQK)
        tCsNKB2 = thr_mma.partition_B(sNK_A)
        tCrAA = thr_mma.make_fragment_A(tCsAA)
        tCrNKB2 = thr_mma.make_fragment_B(tCsNKB2)
        for kk in cutlass.range_constexpr(BT // 16):
            cute.autovec_copy(tCsAA[None, None, kk], tCrAA[None, None, kk])
            cute.autovec_copy(tCsNKB2[None, None, kk], tCrNKB2[None, None, kk])
            cute.gemm(tiled_mma, acc_qS, tCrAA[None, None, kk], tCrNKB2[None, None, kk], acc_qS)

        for idx in cutlass.range(cute.size(acc_qS)):
            co = tCcC_bv[idx]
            o_val = scale * acc_qS[idx]
            mO[i_b, t0 + co[0], i_hv, v_off + co[1]] = o_val.to(mO.element_type)

        # ==================== STATE UPDATE ====================
        if tidx < K_DIM:
            phi = sExpGC[BT - 1]
            for v in cutlass.range_constexpr(BV_TILE):
                state[v] = phi * state[v]
            for t in cutlass.range_constexpr(BT):
                kd = cutlass.Float32(sK[t, tidx]) * sExpDecay[t]
                for v in cutlass.range_constexpr(BV_TILE):
                    state[v] = state[v] + kd * cutlass.Float32(sV[t, v])

        cute.arch.barrier()


@cute.kernel
def fused_chunk_h_pairv_reuse_probe_kernel(
    tiled_mma: cute.TiledMma,
    tiled_copy_kq: cute.TiledCopy,
    sK_layout: cute.Layout,
    sV_layout: cute.Layout, sA_layout: cute.Layout,
    sGQK_layout: cute.Layout,
    sBeta_layout: cute.Layout,
    sS_layout: cute.Layout, sNK_layout: cute.Layout,
    sExpGC_layout: cute.Layout, sExpDecay_layout: cute.Layout,
    mKnorm: cute.Tensor, mQnorm: cute.Tensor,
    mV: cute.Tensor, mBeta: cute.Tensor,
    mM: cute.Tensor, mGQK: cute.Tensor,
    mExpGC_in: cute.Tensor, mExpDecay_in: cute.Tensor,
    mO: cute.Tensor,
    scale: cutlass.Constexpr[float],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int], HV: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int], NT: cutlass.Constexpr[int],
    H_PER_HV: cutlass.Constexpr[int],
    BV_TILE: cutlass.Constexpr[int],
    PER_V_TILE: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    bid_v_pair = cute.arch.block_idx()[0]
    bid_bh = cute.arch.block_idx()[1]
    i_b = bid_bh // HV
    i_hv = bid_bh % HV
    i_h = i_hv // H_PER_HV
    v_pair_off = bid_v_pair * (BV_TILE * 2)

    smem = cutlass.utils.SmemAllocator()
    sK = smem.allocate_tensor(cutlass.BFloat16, sK_layout, 16)
    sQ = smem.allocate_tensor(cutlass.BFloat16, sK_layout, 16)
    sV = smem.allocate_tensor(cutlass.BFloat16, sV_layout, 16)
    sA = smem.allocate_tensor(cutlass.BFloat16, sA_layout, 16)
    sGQK = smem.allocate_tensor(cutlass.BFloat16, sGQK_layout, 16)
    sBeta = smem.allocate_tensor(cutlass.Float32, sBeta_layout, 16)
    sS = smem.allocate_tensor(cutlass.BFloat16, sS_layout, 16)
    sNK_A = smem.allocate_tensor(cutlass.BFloat16, sNK_layout, 16)
    sExpGC = smem.allocate_tensor(cutlass.Float32, sExpGC_layout, 16)
    sExpDecay = smem.allocate_tensor(cutlass.Float32, sExpDecay_layout, 16)

    state0 = cute.make_rmem_tensor(cute.make_layout((BV_TILE,)), cutlass.Float32)
    state1 = cute.make_rmem_tensor(cute.make_layout((BV_TILE,)), cutlass.Float32)
    state0.fill(cutlass.Float32(0.0))
    state1.fill(cutlass.Float32(0.0))

    thr_mma = tiled_mma.get_slice(tidx)

    cC_bv = cute.make_identity_tensor((BT, BV_TILE))
    tCcC_bv = thr_mma.partition_C(cC_bv)

    gK_full = mKnorm[(i_b, i_h, None, None)]
    gQ_full = mQnorm[(i_b, i_h, None, None)]
    thr_cp = tiled_copy_kq.get_slice(tidx)
    thr_sK_cp = thr_cp.partition_D(sK)
    thr_sQ_cp = thr_cp.partition_D(sQ)

    for chunk_idx in cutlass.range(NT):
        t0 = chunk_idx * BT

        # Shared across both BV16 subtiles in this CTA.
        gK_chunk = cute.local_tile(gK_full, (BT, K_DIM), (chunk_idx, 0))
        gQ_chunk = cute.local_tile(gQ_full, (BT, K_DIM), (chunk_idx, 0))
        thr_gK = thr_cp.partition_S(gK_chunk)
        thr_gQ = thr_cp.partition_S(gQ_chunk)
        cute.copy(tiled_copy_kq, thr_gK, thr_sK_cp)
        cute.copy(tiled_copy_kq, thr_gQ, thr_sQ_cp)
        cute.arch.cp_async_commit_group()

        for i in cutlass.range_constexpr(PER_A):
            idx = i * NUM_THREADS + tidx
            row = idx // BT
            col = idx % BT
            sA[row, col] = mM[i_b, chunk_idx, i_hv, row, col]
            sGQK[row, col] = mGQK[i_b, chunk_idx, i_hv, row, col]

        if tidx < BT:
            sExpGC[tidx] = mExpGC_in[i_b, chunk_idx, i_hv, tidx]
            sExpDecay[tidx] = mExpDecay_in[i_b, chunk_idx, i_hv, tidx]
            sBeta[tidx] = cutlass.Float32(mBeta[i_b, t0 + tidx, i_hv])

        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        for subtile in cutlass.range_constexpr(2):
            v_off = v_pair_off + subtile * BV_TILE

            for i in cutlass.range_constexpr(PER_V_TILE):
                idx = i * NUM_THREADS + tidx
                r = idx // BV_TILE
                c = idx % BV_TILE
                sV[r, c] = mV[i_b, t0 + r, i_hv, v_off + c]

            if subtile == 0:
                for v in cutlass.range_constexpr(BV_TILE):
                    sS[v, tidx] = state0[v].to(cutlass.BFloat16)
            else:
                for v in cutlass.range_constexpr(BV_TILE):
                    sS[v, tidx] = state1[v].to(cutlass.BFloat16)
            cute.arch.barrier()

            acc_kS = cute.make_rmem_tensor(
                thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
            acc_kS.fill(cutlass.Float32(0.0))
            acc_qS = cute.make_rmem_tensor(
                thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
            acc_qS.fill(cutlass.Float32(0.0))

            _mma_AS_full(tiled_mma, thr_mma, sK, sS, acc_kS)
            _mma_AS_full(tiled_mma, thr_mma, sQ, sS, acc_qS)

            for idx in cutlass.range(cute.size(acc_kS)):
                co = tCcC_bv[idx]
                t = co[0]
                v = co[1]
                v_val = cutlass.Float32(sV[t, v])
                kS_val = acc_kS[idx]
                rhs = sBeta[t] * (v_val - sExpGC[t] * kS_val)
                sNK_A[v, t] = rhs.to(cutlass.BFloat16)
                acc_qS[idx] = acc_qS[idx] * sExpGC[t]
            cute.arch.barrier()

            acc_vnew = cute.make_rmem_tensor(
                thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
            acc_vnew.fill(cutlass.Float32(0.0))

            tCsMA = thr_mma.partition_A(sA)
            tCsNKB = thr_mma.partition_B(sNK_A)
            tCrMA = thr_mma.make_fragment_A(tCsMA)
            tCrNKB = thr_mma.make_fragment_B(tCsNKB)
            for kk in cutlass.range_constexpr(BT // 16):
                cute.autovec_copy(tCsMA[None, None, kk], tCrMA[None, None, kk])
                cute.autovec_copy(tCsNKB[None, None, kk], tCrNKB[None, None, kk])
                cute.gemm(tiled_mma, acc_vnew, tCrMA[None, None, kk], tCrNKB[None, None, kk], acc_vnew)

            for idx in cutlass.range(cute.size(acc_vnew)):
                co = tCcC_bv[idx]
                sV[co[0], co[1]] = acc_vnew[idx].to(cutlass.BFloat16)
                sNK_A[co[1], co[0]] = acc_vnew[idx].to(cutlass.BFloat16)
            cute.arch.barrier()

            tCsAA = thr_mma.partition_A(sGQK)
            tCsNKB2 = thr_mma.partition_B(sNK_A)
            tCrAA = thr_mma.make_fragment_A(tCsAA)
            tCrNKB2 = thr_mma.make_fragment_B(tCsNKB2)
            for kk in cutlass.range_constexpr(BT // 16):
                cute.autovec_copy(tCsAA[None, None, kk], tCrAA[None, None, kk])
                cute.autovec_copy(tCsNKB2[None, None, kk], tCrNKB2[None, None, kk])
                cute.gemm(tiled_mma, acc_qS, tCrAA[None, None, kk], tCrNKB2[None, None, kk], acc_qS)

            for idx in cutlass.range(cute.size(acc_qS)):
                co = tCcC_bv[idx]
                o_val = scale * acc_qS[idx]
                mO[i_b, t0 + co[0], i_hv, v_off + co[1]] = o_val.to(mO.element_type)

            if tidx < K_DIM:
                phi = sExpGC[BT - 1]
                if subtile == 0:
                    for v in cutlass.range_constexpr(BV_TILE):
                        state0[v] = phi * state0[v]
                    for t in cutlass.range_constexpr(BT):
                        kd = cutlass.Float32(sK[t, tidx]) * sExpDecay[t]
                        for v in cutlass.range_constexpr(BV_TILE):
                            state0[v] = state0[v] + kd * cutlass.Float32(sV[t, v])
                else:
                    for v in cutlass.range_constexpr(BV_TILE):
                        state1[v] = phi * state1[v]
                    for t in cutlass.range_constexpr(BT):
                        kd = cutlass.Float32(sK[t, tidx]) * sExpDecay[t]
                        for v in cutlass.range_constexpr(BV_TILE):
                            state1[v] = state1[v] + kd * cutlass.Float32(sV[t, v])
            cute.arch.barrier()


@cute.kernel
def fused_chunk_h_bf16_state_probe_kernel(
    tiled_mma: cute.TiledMma,
    tiled_copy_kq: cute.TiledCopy,
    sK_layout: cute.Layout,
    sV_layout: cute.Layout, sA_layout: cute.Layout,
    sGQK_layout: cute.Layout,
    sBeta_layout: cute.Layout,
    sS_layout: cute.Layout, sNK_layout: cute.Layout,
    sExpGC_layout: cute.Layout, sExpDecay_layout: cute.Layout,
    mKnorm: cute.Tensor, mQnorm: cute.Tensor,
    mV: cute.Tensor, mBeta: cute.Tensor,
    mM: cute.Tensor, mGQK: cute.Tensor,
    mExpGC_in: cute.Tensor, mExpDecay_in: cute.Tensor,
    mO: cute.Tensor,
    scale: cutlass.Constexpr[float],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int], HV: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int], NT: cutlass.Constexpr[int],
    H_PER_HV: cutlass.Constexpr[int],
    BV_TILE: cutlass.Constexpr[int],
    PER_V_TILE: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    bid_v = cute.arch.block_idx()[0]
    bid_bh = cute.arch.block_idx()[1]
    i_b = bid_bh // HV
    i_hv = bid_bh % HV
    i_h = i_hv // H_PER_HV
    v_off = bid_v * BV_TILE

    smem = cutlass.utils.SmemAllocator()
    sK = smem.allocate_tensor(cutlass.BFloat16, sK_layout, 16)
    sQ = smem.allocate_tensor(cutlass.BFloat16, sK_layout, 16)
    sV = smem.allocate_tensor(cutlass.BFloat16, sV_layout, 16)
    sA = smem.allocate_tensor(cutlass.BFloat16, sA_layout, 16)
    sGQK = smem.allocate_tensor(cutlass.BFloat16, sGQK_layout, 16)
    sBeta = smem.allocate_tensor(cutlass.Float32, sBeta_layout, 16)
    sS = smem.allocate_tensor(cutlass.BFloat16, sS_layout, 16)
    sNK_A = smem.allocate_tensor(cutlass.BFloat16, sNK_layout, 16)
    sExpGC = smem.allocate_tensor(cutlass.Float32, sExpGC_layout, 16)
    sExpDecay = smem.allocate_tensor(cutlass.Float32, sExpDecay_layout, 16)

    state = cute.make_rmem_tensor(cute.make_layout((BV_TILE,)), cutlass.BFloat16)
    state.fill(cutlass.BFloat16(0.0))

    thr_mma = tiled_mma.get_slice(tidx)

    cC_bv = cute.make_identity_tensor((BT, BV_TILE))
    tCcC_bv = thr_mma.partition_C(cC_bv)

    gK_full = mKnorm[(i_b, i_h, None, None)]
    gQ_full = mQnorm[(i_b, i_h, None, None)]
    thr_cp = tiled_copy_kq.get_slice(tidx)
    thr_sK_cp = thr_cp.partition_D(sK)
    thr_sQ_cp = thr_cp.partition_D(sQ)

    for chunk_idx in cutlass.range(NT):
        t0 = chunk_idx * BT

        # ==================== cp.async K, Q (non-blocking) ====================
        gK_chunk = cute.local_tile(gK_full, (BT, K_DIM), (chunk_idx, 0))
        gQ_chunk = cute.local_tile(gQ_full, (BT, K_DIM), (chunk_idx, 0))
        thr_gK = thr_cp.partition_S(gK_chunk)
        thr_gQ = thr_cp.partition_S(gQ_chunk)
        cute.copy(tiled_copy_kq, thr_gK, thr_sK_cp)
        cute.copy(tiled_copy_kq, thr_gQ, thr_sQ_cp)
        cute.arch.cp_async_commit_group()

        # ==================== Overlap: load rest while cp.async flies =========
        for i in cutlass.range_constexpr(PER_A):
            idx = i * NUM_THREADS + tidx
            row = idx // BT
            col = idx % BT
            sA[row, col] = mM[i_b, chunk_idx, i_hv, row, col]
            sGQK[row, col] = mGQK[i_b, chunk_idx, i_hv, row, col]

        for i in cutlass.range_constexpr(PER_V_TILE):
            idx = i * NUM_THREADS + tidx
            r = idx // BV_TILE
            c = idx % BV_TILE
            sV[r, c] = mV[i_b, t0 + r, i_hv, v_off + c]

        if tidx < BT:
            sExpGC[tidx] = mExpGC_in[i_b, chunk_idx, i_hv, tidx]
            sExpDecay[tidx] = mExpDecay_in[i_b, chunk_idx, i_hv, tidx]
            sBeta[tidx] = cutlass.Float32(mBeta[i_b, t0 + tidx, i_hv])

        for v in cutlass.range_constexpr(BV_TILE):
            sS[v, tidx] = state[v]

        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        # ==================== kS + qS (full-width MMA) ====================
        acc_kS = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        acc_kS.fill(cutlass.Float32(0.0))
        acc_qS = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        acc_qS.fill(cutlass.Float32(0.0))

        _mma_AS_full(tiled_mma, thr_mma, sK, sS, acc_kS)
        _mma_AS_full(tiled_mma, thr_mma, sQ, sS, acc_qS)

        # ==================== RHS + TRANSPOSE ====================
        for idx in cutlass.range(cute.size(acc_kS)):
            co = tCcC_bv[idx]
            t = co[0]
            v = co[1]
            v_val = cutlass.Float32(sV[t, v])
            kS_val = acc_kS[idx]
            rhs = sBeta[t] * (v_val - sExpGC[t] * kS_val)
            sNK_A[v, t] = rhs.to(cutlass.BFloat16)
            acc_qS[idx] = acc_qS[idx] * sExpGC[t]
        cute.arch.barrier()

        # ==================== v_new = M @ RHS (single MMA) ====================
        acc_vnew = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        acc_vnew.fill(cutlass.Float32(0.0))

        tCsMA = thr_mma.partition_A(sA)
        tCsNKB = thr_mma.partition_B(sNK_A)
        tCrMA = thr_mma.make_fragment_A(tCsMA)
        tCrNKB = thr_mma.make_fragment_B(tCsNKB)
        for kk in cutlass.range_constexpr(BT // 16):
            cute.autovec_copy(tCsMA[None, None, kk], tCrMA[None, None, kk])
            cute.autovec_copy(tCsNKB[None, None, kk], tCrNKB[None, None, kk])
            cute.gemm(tiled_mma, acc_vnew, tCrMA[None, None, kk], tCrNKB[None, None, kk], acc_vnew)

        for idx in cutlass.range(cute.size(acc_vnew)):
            co = tCcC_bv[idx]
            sV[co[0], co[1]] = acc_vnew[idx].to(cutlass.BFloat16)
            sNK_A[co[1], co[0]] = acc_vnew[idx].to(cutlass.BFloat16)
        cute.arch.barrier()

        # ==================== INLINE CHUNK_O ====================
        tCsAA = thr_mma.partition_A(sGQK)
        tCsNKB2 = thr_mma.partition_B(sNK_A)
        tCrAA = thr_mma.make_fragment_A(tCsAA)
        tCrNKB2 = thr_mma.make_fragment_B(tCsNKB2)
        for kk in cutlass.range_constexpr(BT // 16):
            cute.autovec_copy(tCsAA[None, None, kk], tCrAA[None, None, kk])
            cute.autovec_copy(tCsNKB2[None, None, kk], tCrNKB2[None, None, kk])
            cute.gemm(tiled_mma, acc_qS, tCrAA[None, None, kk], tCrNKB2[None, None, kk], acc_qS)

        for idx in cutlass.range(cute.size(acc_qS)):
            co = tCcC_bv[idx]
            o_val = scale * acc_qS[idx]
            mO[i_b, t0 + co[0], i_hv, v_off + co[1]] = o_val.to(mO.element_type)

        # ==================== STATE UPDATE ====================
        if tidx < K_DIM:
            phi = sExpGC[BT - 1]
            for v in cutlass.range_constexpr(BV_TILE):
                state[v] = (phi * cutlass.Float32(state[v])).to(cutlass.BFloat16)
            for t in cutlass.range_constexpr(BT):
                kd = cutlass.Float32(sK[t, tidx]) * sExpDecay[t]
                for v in cutlass.range_constexpr(BV_TILE):
                    next_state = cutlass.Float32(state[v]) + kd * cutlass.Float32(sV[t, v])
                    state[v] = next_state.to(cutlass.BFloat16)

        cute.arch.barrier()


@cute.kernel
def fused_chunk_h_state_mma_probe_kernel(
    tiled_mma: cute.TiledMma,
    tiled_copy_kq: cute.TiledCopy,
    sK_layout: cute.Layout,
    sV_layout: cute.Layout, sA_layout: cute.Layout,
    sGQK_layout: cute.Layout,
    sBeta_layout: cute.Layout,
    sS_layout: cute.Layout, sNK_layout: cute.Layout,
    sState_layout: cute.Layout,
    sExpGC_layout: cute.Layout, sExpDecay_layout: cute.Layout,
    mKnorm: cute.Tensor, mQnorm: cute.Tensor,
    mV: cute.Tensor, mBeta: cute.Tensor,
    mM: cute.Tensor, mGQK: cute.Tensor,
    mExpGC_in: cute.Tensor, mExpDecay_in: cute.Tensor,
    mO: cute.Tensor,
    scale: cutlass.Constexpr[float],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int], HV: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int], NT: cutlass.Constexpr[int],
    H_PER_HV: cutlass.Constexpr[int],
    BV_TILE: cutlass.Constexpr[int],
    PER_V_TILE: cutlass.Constexpr[int],
    PER_STATE_TILE: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    bid_v = cute.arch.block_idx()[0]
    bid_bh = cute.arch.block_idx()[1]
    i_b = bid_bh // HV
    i_hv = bid_bh % HV
    i_h = i_hv // H_PER_HV
    v_off = bid_v * BV_TILE

    smem = cutlass.utils.SmemAllocator()
    sK = smem.allocate_tensor(cutlass.BFloat16, sK_layout, 16)
    sQ = smem.allocate_tensor(cutlass.BFloat16, sK_layout, 16)
    sV = smem.allocate_tensor(cutlass.BFloat16, sV_layout, 16)
    sA = smem.allocate_tensor(cutlass.BFloat16, sA_layout, 16)
    sGQK = smem.allocate_tensor(cutlass.BFloat16, sGQK_layout, 16)
    sBeta = smem.allocate_tensor(cutlass.Float32, sBeta_layout, 16)
    sS = smem.allocate_tensor(cutlass.BFloat16, sS_layout, 16)
    sNK_A = smem.allocate_tensor(cutlass.BFloat16, sNK_layout, 16)
    sState = smem.allocate_tensor(cutlass.Float32, sState_layout, 16)
    sExpGC = smem.allocate_tensor(cutlass.Float32, sExpGC_layout, 16)
    sExpDecay = smem.allocate_tensor(cutlass.Float32, sExpDecay_layout, 16)

    thr_mma = tiled_mma.get_slice(tidx)

    cC_bv = cute.make_identity_tensor((BT, BV_TILE))
    tCcC_bv = thr_mma.partition_C(cC_bv)

    for i in cutlass.range_constexpr(PER_STATE_TILE):
        idx = i * NUM_THREADS + tidx
        k_row = idx // BV_TILE
        v_col = idx % BV_TILE
        sState[k_row, v_col] = cutlass.Float32(0.0)
    cute.arch.barrier()

    gK_full = mKnorm[(i_b, i_h, None, None)]
    gQ_full = mQnorm[(i_b, i_h, None, None)]
    thr_cp = tiled_copy_kq.get_slice(tidx)
    thr_sK_cp = thr_cp.partition_D(sK)
    thr_sQ_cp = thr_cp.partition_D(sQ)

    for chunk_idx in cutlass.range(NT):
        t0 = chunk_idx * BT

        gK_chunk = cute.local_tile(gK_full, (BT, K_DIM), (chunk_idx, 0))
        gQ_chunk = cute.local_tile(gQ_full, (BT, K_DIM), (chunk_idx, 0))
        thr_gK = thr_cp.partition_S(gK_chunk)
        thr_gQ = thr_cp.partition_S(gQ_chunk)
        cute.copy(tiled_copy_kq, thr_gK, thr_sK_cp)
        cute.copy(tiled_copy_kq, thr_gQ, thr_sQ_cp)
        cute.arch.cp_async_commit_group()

        for i in cutlass.range_constexpr(PER_A):
            idx = i * NUM_THREADS + tidx
            row = idx // BT
            col = idx % BT
            sA[row, col] = mM[i_b, chunk_idx, i_hv, row, col]
            sGQK[row, col] = mGQK[i_b, chunk_idx, i_hv, row, col]

        for i in cutlass.range_constexpr(PER_V_TILE):
            idx = i * NUM_THREADS + tidx
            r = idx // BV_TILE
            c = idx % BV_TILE
            sV[r, c] = mV[i_b, t0 + r, i_hv, v_off + c]

        if tidx < BT:
            sExpGC[tidx] = mExpGC_in[i_b, chunk_idx, i_hv, tidx]
            sExpDecay[tidx] = mExpDecay_in[i_b, chunk_idx, i_hv, tidx]
            sBeta[tidx] = cutlass.Float32(mBeta[i_b, t0 + tidx, i_hv])

        for i in cutlass.range_constexpr(PER_STATE_TILE):
            idx = i * NUM_THREADS + tidx
            k_row = idx // BV_TILE
            v_col = idx % BV_TILE
            sS[v_col, k_row] = sState[k_row, v_col].to(cutlass.BFloat16)

        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        acc_kS = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        acc_kS.fill(cutlass.Float32(0.0))
        acc_qS = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        acc_qS.fill(cutlass.Float32(0.0))

        _mma_AS_full(tiled_mma, thr_mma, sK, sS, acc_kS)
        _mma_AS_full(tiled_mma, thr_mma, sQ, sS, acc_qS)

        for idx in cutlass.range(cute.size(acc_kS)):
            co = tCcC_bv[idx]
            t = co[0]
            v = co[1]
            v_val = cutlass.Float32(sV[t, v])
            kS_val = acc_kS[idx]
            rhs = sBeta[t] * (v_val - sExpGC[t] * kS_val)
            sNK_A[v, t] = rhs.to(cutlass.BFloat16)
            acc_qS[idx] = acc_qS[idx] * sExpGC[t]
        cute.arch.barrier()

        acc_vnew = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        acc_vnew.fill(cutlass.Float32(0.0))

        tCsMA = thr_mma.partition_A(sA)
        tCsNKB = thr_mma.partition_B(sNK_A)
        tCrMA = thr_mma.make_fragment_A(tCsMA)
        tCrNKB = thr_mma.make_fragment_B(tCsNKB)
        for kk in cutlass.range_constexpr(BT // 16):
            cute.autovec_copy(tCsMA[None, None, kk], tCrMA[None, None, kk])
            cute.autovec_copy(tCsNKB[None, None, kk], tCrNKB[None, None, kk])
            cute.gemm(tiled_mma, acc_vnew, tCrMA[None, None, kk], tCrNKB[None, None, kk], acc_vnew)

        for idx in cutlass.range(cute.size(acc_vnew)):
            co = tCcC_bv[idx]
            sV[co[0], co[1]] = acc_vnew[idx].to(cutlass.BFloat16)
            sNK_A[co[1], co[0]] = acc_vnew[idx].to(cutlass.BFloat16)
        cute.arch.barrier()

        tCsAA = thr_mma.partition_A(sGQK)
        tCsNKB2 = thr_mma.partition_B(sNK_A)
        tCrAA = thr_mma.make_fragment_A(tCsAA)
        tCrNKB2 = thr_mma.make_fragment_B(tCsNKB2)
        for kk in cutlass.range_constexpr(BT // 16):
            cute.autovec_copy(tCsAA[None, None, kk], tCrAA[None, None, kk])
            cute.autovec_copy(tCsNKB2[None, None, kk], tCrNKB2[None, None, kk])
            cute.gemm(tiled_mma, acc_qS, tCrAA[None, None, kk], tCrNKB2[None, None, kk], acc_qS)

        for idx in cutlass.range(cute.size(acc_qS)):
            co = tCcC_bv[idx]
            o_val = scale * acc_qS[idx]
            mO[i_b, t0 + co[0], i_hv, v_off + co[1]] = o_val.to(mO.element_type)

        phi = sExpGC[BT - 1]
        for k_block in cutlass.range_constexpr(K_DIM // BT):
            for i in cutlass.range_constexpr(PER_A):
                idx = i * NUM_THREADS + tidx
                row = idx // BT
                t = idx % BT
                sA[row, t] = (
                    cutlass.Float32(sK[t, k_block * BT + row]) * sExpDecay[t]
                ).to(cutlass.BFloat16)
            cute.arch.barrier()

            acc_state = cute.make_rmem_tensor(
                thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
            acc_state.fill(cutlass.Float32(0.0))
            tCsKTA = thr_mma.partition_A(sA)
            tCsVB = thr_mma.partition_B(sNK_A)
            tCrKTA = thr_mma.make_fragment_A(tCsKTA)
            tCrVB = thr_mma.make_fragment_B(tCsVB)
            for kk in cutlass.range_constexpr(BT // 16):
                cute.autovec_copy(tCsKTA[None, None, kk], tCrKTA[None, None, kk])
                cute.autovec_copy(tCsVB[None, None, kk], tCrVB[None, None, kk])
                cute.gemm(tiled_mma, acc_state, tCrKTA[None, None, kk], tCrVB[None, None, kk], acc_state)

            for idx in cutlass.range(cute.size(acc_state)):
                co = tCcC_bv[idx]
                k_row = k_block * BT + co[0]
                v_col = co[1]
                sState[k_row, v_col] = phi * sState[k_row, v_col] + acc_state[idx]
            cute.arch.barrier()


# ============================================================================
# Kernel 1 Mega: inline K_inv + fused_chunk_h + inline chunk_o
# ============================================================================

@cute.kernel
def fused_chunk_h_megakernel(
    tiled_mma: cute.TiledMma,
    tiled_copy_kq: cute.TiledCopy,
    sK_layout: cute.Layout,
    sV_layout: cute.Layout, sA_layout: cute.Layout, sA_T_layout: cute.Layout,
    sBeta_layout: cute.Layout,
    sS_layout: cute.Layout, sNK_layout: cute.Layout,
    sExpGC_layout: cute.Layout, sExpDecay_layout: cute.Layout,
    mKK: cute.Tensor, mKnorm: cute.Tensor, mQnorm: cute.Tensor,
    mV: cute.Tensor, mGate: cute.Tensor, mBeta: cute.Tensor,
    mO: cute.Tensor,
    scale: cutlass.Constexpr[float],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int], HV: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int], NT: cutlass.Constexpr[int],
    H_PER_HV: cutlass.Constexpr[int],
    BV_TILE: cutlass.Constexpr[int],
    PER_V_TILE: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    bid_v = cute.arch.block_idx()[0]
    bid_bh = cute.arch.block_idx()[1]
    i_b = bid_bh // HV
    i_hv = bid_bh % HV
    i_h = i_hv // H_PER_HV
    v_off = bid_v * BV_TILE

    smem = cutlass.utils.SmemAllocator()
    sK = smem.allocate_tensor(cutlass.BFloat16, sK_layout, 16)
    sQ = smem.allocate_tensor(cutlass.BFloat16, sK_layout, 16)
    sV = smem.allocate_tensor(cutlass.BFloat16, sV_layout, 16)
    sA = smem.allocate_tensor(cutlass.BFloat16, sA_layout, 16)
    sA_T = smem.allocate_tensor(cutlass.BFloat16, sA_T_layout, 16)
    sBeta = smem.allocate_tensor(cutlass.Float32, sBeta_layout, 16)
    sS = smem.allocate_tensor(cutlass.BFloat16, sS_layout, 16)
    sNK_A = smem.allocate_tensor(cutlass.BFloat16, sNK_layout, 16)
    sExpGC = smem.allocate_tensor(cutlass.Float32, sExpGC_layout, 16)
    sExpDecay = smem.allocate_tensor(cutlass.Float32, sExpDecay_layout, 16)

    state = cute.make_rmem_tensor(cute.make_layout((BV_TILE,)), cutlass.Float32)
    state.fill(cutlass.Float32(0.0))

    thr_mma = tiled_mma.get_slice(tidx)

    cC_bt = cute.make_identity_tensor((BT, BT))
    tCcC_bt = thr_mma.partition_C(cC_bt)
    cC_bv = cute.make_identity_tensor((BT, BV_TILE))
    tCcC_bv = thr_mma.partition_C(cC_bv)

    gK_full = mKnorm[(i_b, i_h, None, None)]
    gQ_full = mQnorm[(i_b, i_h, None, None)]
    thr_cp = tiled_copy_kq.get_slice(tidx)
    thr_sK_cp = thr_cp.partition_D(sK)
    thr_sQ_cp = thr_cp.partition_D(sQ)

    for chunk_idx in cutlass.range(NT):
        t0 = chunk_idx * BT

        # Start K/Q copy first. The inline inverse computation below is long
        # enough to hide the cp.async latency without TMA/warp-specialization.
        gK_chunk = cute.local_tile(gK_full, (BT, K_DIM), (chunk_idx, 0))
        gQ_chunk = cute.local_tile(gQ_full, (BT, K_DIM), (chunk_idx, 0))
        thr_gK = thr_cp.partition_S(gK_chunk)
        thr_gQ = thr_cp.partition_S(gQ_chunk)
        cute.copy(tiled_copy_kq, thr_gK, thr_sK_cp)
        cute.copy(tiled_copy_kq, thr_gQ, thr_sQ_cp)
        cute.arch.cp_async_commit_group()

        # ==================== Inline K_inv: M = I - A + A^2 - A^3 =========
        for i in cutlass.range_constexpr(PER_A):
            idx = i * NUM_THREADS + tidx
            row = idx // BT
            col = idx % BT
            sA[row, col] = mKK[i_b, chunk_idx, i_h, row, col]

        if tidx < BT:
            sExpGC[tidx] = cutlass.Float32(mGate[i_b, t0 + tidx, i_hv])
            sBeta[tidx] = cutlass.Float32(mBeta[i_b, t0 + tidx, i_hv])
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        if tidx == 0:
            rs = cutlass.Float32(0.0)
            for t in cutlass.range_constexpr(BT):
                rs = rs + sExpGC[t]
                sExpGC[t] = rs
        cute.arch.barrier()

        gc_last = sExpGC[BT - 1]
        for i in cutlass.range_constexpr(PER_A):
            idx = i * NUM_THREADS + tidx
            row = idx // BT
            col = idx % BT
            if row > col:
                a_val = cutlass.Float32(sA[row, col]) * sBeta[row] * cute.exp(sExpGC[row] - sExpGC[col])
                sA[row, col] = a_val.to(cutlass.BFloat16)
                sA_T[col, row] = a_val.to(cutlass.BFloat16)
            else:
                sA[row, col] = cutlass.BFloat16(0.0)
                sA_T[col, row] = cutlass.BFloat16(0.0)

        if tidx < BT:
            gc_t = sExpGC[tidx]
            sExpDecay[tidx] = cute.exp(gc_last - gc_t)
        cute.arch.barrier()

        acc_A2 = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BT)), cutlass.Float32)
        acc_A2.fill(cutlass.Float32(0.0))
        tCsAA = thr_mma.partition_A(sA)
        tCsATB = thr_mma.partition_B(sA_T)
        tCrAA = thr_mma.make_fragment_A(tCsAA)
        tCrATB = thr_mma.make_fragment_B(tCsATB)
        for kk in cutlass.range_constexpr(BT // 16):
            cute.autovec_copy(tCsAA[None, None, kk], tCrAA[None, None, kk])
            cute.autovec_copy(tCsATB[None, None, kk], tCrATB[None, None, kk])
            cute.gemm(tiled_mma, acc_A2, tCrAA[None, None, kk], tCrATB[None, None, kk], acc_A2)

        for idx in cutlass.range(cute.size(acc_A2)):
            co = tCcC_bt[idx]
            sA[co[0], co[1]] = acc_A2[idx].to(cutlass.BFloat16)
        cute.arch.barrier()

        acc_A3 = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BT)), cutlass.Float32)
        acc_A3.fill(cutlass.Float32(0.0))
        for kk in cutlass.range_constexpr(BT // 16):
            cute.autovec_copy(tCsAA[None, None, kk], tCrAA[None, None, kk])
            cute.autovec_copy(tCsATB[None, None, kk], tCrATB[None, None, kk])
            cute.gemm(tiled_mma, acc_A3, tCrAA[None, None, kk], tCrATB[None, None, kk], acc_A3)

        for idx in cutlass.range(cute.size(acc_A2)):
            co = tCcC_bt[idx]
            row = co[0]
            col = co[1]
            i_val = cutlass.Float32(1.0) if row == col else cutlass.Float32(0.0)
            a_val = cutlass.Float32(sA_T[col, row])
            m_val = i_val - a_val + acc_A2[idx] - acc_A3[idx]
            sA[row, col] = m_val.to(cutlass.BFloat16)
        cute.arch.barrier()

        if tidx < BT:
            sExpGC[tidx] = cute.exp(sExpGC[tidx])
        cute.arch.barrier()

        # ==================== Load V and current state tile =================
        for i in cutlass.range_constexpr(PER_V_TILE):
            idx = i * NUM_THREADS + tidx
            r = idx // BV_TILE
            c = idx % BV_TILE
            sV[r, c] = mV[i_b, t0 + r, i_hv, v_off + c]

        for v in cutlass.range_constexpr(BV_TILE):
            sS[v, tidx] = state[v].to(cutlass.BFloat16)

        cute.arch.barrier()

        # ==================== kS + qS (full-width MMA) ====================
        acc_kS = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        acc_kS.fill(cutlass.Float32(0.0))
        acc_qS = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        acc_qS.fill(cutlass.Float32(0.0))

        _mma_AS_full(tiled_mma, thr_mma, sK, sS, acc_kS)
        _mma_AS_full(tiled_mma, thr_mma, sQ, sS, acc_qS)

        # ==================== RHS + TRANSPOSE ====================
        for idx in cutlass.range(cute.size(acc_kS)):
            co = tCcC_bv[idx]
            t = co[0]
            v = co[1]
            v_val = cutlass.Float32(sV[t, v])
            kS_val = acc_kS[idx]
            rhs = sBeta[t] * (v_val - sExpGC[t] * kS_val)
            sNK_A[v, t] = rhs.to(cutlass.BFloat16)
            acc_qS[idx] = acc_qS[idx] * sExpGC[t]
        cute.arch.barrier()

        # ==================== v_new = M @ RHS ====================
        acc_vnew = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        acc_vnew.fill(cutlass.Float32(0.0))

        tCsMA = thr_mma.partition_A(sA)
        tCsNKB = thr_mma.partition_B(sNK_A)
        tCrMA = thr_mma.make_fragment_A(tCsMA)
        tCrNKB = thr_mma.make_fragment_B(tCsNKB)
        for kk in cutlass.range_constexpr(BT // 16):
            cute.autovec_copy(tCsMA[None, None, kk], tCrMA[None, None, kk])
            cute.autovec_copy(tCsNKB[None, None, kk], tCrNKB[None, None, kk])
            cute.gemm(tiled_mma, acc_vnew, tCrMA[None, None, kk], tCrNKB[None, None, kk], acc_vnew)

        for idx in cutlass.range(cute.size(acc_vnew)):
            co = tCcC_bv[idx]
            sV[co[0], co[1]] = acc_vnew[idx].to(cutlass.BFloat16)
            sNK_A[co[1], co[0]] = acc_vnew[idx].to(cutlass.BFloat16)

        # ==================== INLINE CHUNK_O ====================
        _mma_qk_to_sA_gated(tiled_mma, thr_mma, sQ, sK, sA, sExpGC)
        cute.arch.barrier()

        tCsAA2 = thr_mma.partition_A(sA)
        tCsNKB2 = thr_mma.partition_B(sNK_A)
        tCrAA2 = thr_mma.make_fragment_A(tCsAA2)
        tCrNKB2 = thr_mma.make_fragment_B(tCsNKB2)
        for kk in cutlass.range_constexpr(BT // 16):
            cute.autovec_copy(tCsAA2[None, None, kk], tCrAA2[None, None, kk])
            cute.autovec_copy(tCsNKB2[None, None, kk], tCrNKB2[None, None, kk])
            cute.gemm(tiled_mma, acc_qS, tCrAA2[None, None, kk], tCrNKB2[None, None, kk], acc_qS)

        for idx in cutlass.range(cute.size(acc_qS)):
            co = tCcC_bv[idx]
            o_val = scale * acc_qS[idx]
            mO[i_b, t0 + co[0], i_hv, v_off + co[1]] = o_val.to(mO.element_type)

        # ==================== STATE UPDATE ====================
        if tidx < K_DIM:
            phi = sExpGC[BT - 1]
            for v in cutlass.range_constexpr(BV_TILE):
                state[v] = phi * state[v]
            for t in cutlass.range_constexpr(BT):
                kd = cutlass.Float32(sK[t, tidx]) * sExpDecay[t]
                for v in cutlass.range_constexpr(BV_TILE):
                    state[v] = state[v] + kd * cutlass.Float32(sV[t, v])

        cute.arch.barrier()


@cute.kernel
def fused_chunk_h_megakernel_tma(
    tiled_mma: cute.TiledMma,
    tma_atom_k: cute.CopyAtom, gK_tma: cute.Tensor,
    tma_atom_q: cute.CopyAtom, gQ_tma: cute.Tensor,
    sK_layout: cute.Layout, sK_tma_layout: cute.Layout,
    sV_layout: cute.Layout, sA_layout: cute.Layout, sA_T_layout: cute.Layout,
    sBeta_layout: cute.Layout,
    sS_layout: cute.Layout, sNK_layout: cute.Layout,
    sExpGC_layout: cute.Layout, sExpDecay_layout: cute.Layout,
    mKK: cute.Tensor,
    mV: cute.Tensor, mGate: cute.Tensor, mBeta: cute.Tensor,
    mO: cute.Tensor,
    scale: cutlass.Constexpr[float],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int], HV: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int], NT: cutlass.Constexpr[int],
    H_PER_HV: cutlass.Constexpr[int],
    BV_TILE: cutlass.Constexpr[int],
    PER_V_TILE: cutlass.Constexpr[int],
    TMA_TX_BYTES: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    bid_v = cute.arch.block_idx()[0]
    bid_bh = cute.arch.block_idx()[1]
    i_b = bid_bh // HV
    i_hv = bid_bh % HV
    i_h = i_hv // H_PER_HV
    v_off = bid_v * BV_TILE

    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    is_producer = warp_idx == cutlass.Int32(0)
    is_consumer = warp_idx >= cutlass.Int32(1)

    smem = cutlass.utils.SmemAllocator()
    mbar_storage = smem.allocate_array(cutlass.Int64, 2)
    sK = smem.allocate_tensor(cutlass.BFloat16, sK_layout, 1024)
    sQ = smem.allocate_tensor(cutlass.BFloat16, sK_layout, 1024)
    sV = smem.allocate_tensor(cutlass.BFloat16, sV_layout, 16)
    sA = smem.allocate_tensor(cutlass.BFloat16, sA_layout, 16)
    sA_T = smem.allocate_tensor(cutlass.BFloat16, sA_T_layout, 16)
    sBeta = smem.allocate_tensor(cutlass.Float32, sBeta_layout, 16)
    sS = smem.allocate_tensor(cutlass.BFloat16, sS_layout, 16)
    sNK_A = smem.allocate_tensor(cutlass.BFloat16, sNK_layout, 16)
    sExpGC = smem.allocate_tensor(cutlass.Float32, sExpGC_layout, 16)
    sExpDecay = smem.allocate_tensor(cutlass.Float32, sExpDecay_layout, 16)

    sK_for_tma = cute.make_tensor(sK.iterator, sK_tma_layout)
    sQ_for_tma = cute.make_tensor(sQ.iterator, sK_tma_layout)
    tKsK, tKgK = cpasync.tma_partition(
        tma_atom_k, 0, cute.make_layout(1),
        cute.group_modes(sK_for_tma, 0, 2),
        cute.group_modes(cute.local_tile(gK_tma, (BT, K_DIM), (None, 0)), 0, 2),
    )
    tQsQ, tQgQ = cpasync.tma_partition(
        tma_atom_q, 0, cute.make_layout(1),
        cute.group_modes(sQ_for_tma, 0, 2),
        cute.group_modes(cute.local_tile(gQ_tma, (BT, K_DIM), (None, 0)), 0, 2),
    )

    load_pipeline = pipeline.PipelineTmaAsync.create(
        barrier_storage=mbar_storage,
        num_stages=1,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 3),
        tx_count=TMA_TX_BYTES,
        cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
    )
    pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)
    pipeline_init_wait(cluster_shape_mn=(1, 1))

    prod_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
    cons_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)

    state = cute.make_rmem_tensor(cute.make_layout((BV_TILE,)), cutlass.Float32)
    state.fill(cutlass.Float32(0.0))

    thr_mma = tiled_mma.get_slice(tidx)

    cC_bt = cute.make_identity_tensor((BT, BT))
    tCcC_bt = thr_mma.partition_C(cC_bt)
    cC_bv = cute.make_identity_tensor((BT, BV_TILE))
    tCcC_bv = thr_mma.partition_C(cC_bv)

    for chunk_idx in cutlass.range(NT):
        t0 = chunk_idx * BT
        linear_chunk = (i_b * H + i_h) * NT + chunk_idx

        if is_producer:
            load_pipeline.producer_acquire(prod_state)
            bar = load_pipeline.producer_get_barrier(prod_state)
            cute.copy(tma_atom_k, tKgK[(None, linear_chunk)], tKsK[(None, 0)], tma_bar_ptr=bar)
            cute.copy(tma_atom_q, tQgQ[(None, linear_chunk)], tQsQ[(None, 0)], tma_bar_ptr=bar)
            load_pipeline.producer_commit(prod_state)
            prod_state.advance()

        if is_consumer:
            load_pipeline.consumer_wait(cons_state)
        cute.arch.barrier()

        # ==================== Inline K_inv: M = I - A + A^2 - A^3 =========
        for i in cutlass.range_constexpr(PER_A):
            idx = i * NUM_THREADS + tidx
            row = idx // BT
            col = idx % BT
            sA[row, col] = mKK[i_b, chunk_idx, i_h, row, col]

        if tidx < BT:
            sExpGC[tidx] = cutlass.Float32(mGate[i_b, t0 + tidx, i_hv])
            sBeta[tidx] = cutlass.Float32(mBeta[i_b, t0 + tidx, i_hv])
        cute.arch.barrier()

        if tidx == 0:
            rs = cutlass.Float32(0.0)
            for t in cutlass.range_constexpr(BT):
                rs = rs + sExpGC[t]
                sExpGC[t] = rs
        cute.arch.barrier()

        gc_last = sExpGC[BT - 1]
        for i in cutlass.range_constexpr(PER_A):
            idx = i * NUM_THREADS + tidx
            row = idx // BT
            col = idx % BT
            if row > col:
                a_val = cutlass.Float32(sA[row, col]) * sBeta[row] * cute.exp(sExpGC[row] - sExpGC[col])
                sA[row, col] = a_val.to(cutlass.BFloat16)
                sA_T[col, row] = a_val.to(cutlass.BFloat16)
            else:
                sA[row, col] = cutlass.BFloat16(0.0)
                sA_T[col, row] = cutlass.BFloat16(0.0)

        if tidx < BT:
            gc_t = sExpGC[tidx]
            sExpDecay[tidx] = cute.exp(gc_last - gc_t)
        cute.arch.barrier()

        acc_A2 = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BT)), cutlass.Float32)
        acc_A2.fill(cutlass.Float32(0.0))
        tCsAA = thr_mma.partition_A(sA)
        tCsATB = thr_mma.partition_B(sA_T)
        tCrAA = thr_mma.make_fragment_A(tCsAA)
        tCrATB = thr_mma.make_fragment_B(tCsATB)
        for kk in cutlass.range_constexpr(BT // 16):
            cute.autovec_copy(tCsAA[None, None, kk], tCrAA[None, None, kk])
            cute.autovec_copy(tCsATB[None, None, kk], tCrATB[None, None, kk])
            cute.gemm(tiled_mma, acc_A2, tCrAA[None, None, kk], tCrATB[None, None, kk], acc_A2)

        for idx in cutlass.range(cute.size(acc_A2)):
            co = tCcC_bt[idx]
            sA[co[0], co[1]] = acc_A2[idx].to(cutlass.BFloat16)
        cute.arch.barrier()

        acc_A3 = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BT)), cutlass.Float32)
        acc_A3.fill(cutlass.Float32(0.0))
        for kk in cutlass.range_constexpr(BT // 16):
            cute.autovec_copy(tCsAA[None, None, kk], tCrAA[None, None, kk])
            cute.autovec_copy(tCsATB[None, None, kk], tCrATB[None, None, kk])
            cute.gemm(tiled_mma, acc_A3, tCrAA[None, None, kk], tCrATB[None, None, kk], acc_A3)

        for idx in cutlass.range(cute.size(acc_A2)):
            co = tCcC_bt[idx]
            row = co[0]
            col = co[1]
            i_val = cutlass.Float32(1.0) if row == col else cutlass.Float32(0.0)
            a_val = cutlass.Float32(sA_T[col, row])
            m_val = i_val - a_val + acc_A2[idx] - acc_A3[idx]
            sA[row, col] = m_val.to(cutlass.BFloat16)
        cute.arch.barrier()

        if tidx < BT:
            sExpGC[tidx] = cute.exp(sExpGC[tidx])
        cute.arch.barrier()

        # ==================== Load V and current state tile =================
        for i in cutlass.range_constexpr(PER_V_TILE):
            idx = i * NUM_THREADS + tidx
            r = idx // BV_TILE
            c = idx % BV_TILE
            sV[r, c] = mV[i_b, t0 + r, i_hv, v_off + c]

        for v in cutlass.range_constexpr(BV_TILE):
            sS[v, tidx] = state[v].to(cutlass.BFloat16)

        cute.arch.barrier()

        # ==================== kS + qS (full-width MMA) ====================
        acc_kS = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        acc_kS.fill(cutlass.Float32(0.0))
        acc_qS = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        acc_qS.fill(cutlass.Float32(0.0))

        _mma_AS_full(tiled_mma, thr_mma, sK, sS, acc_kS)
        _mma_AS_full(tiled_mma, thr_mma, sQ, sS, acc_qS)

        # ==================== RHS + TRANSPOSE ====================
        for idx in cutlass.range(cute.size(acc_kS)):
            co = tCcC_bv[idx]
            t = co[0]
            v = co[1]
            v_val = cutlass.Float32(sV[t, v])
            kS_val = acc_kS[idx]
            rhs = sBeta[t] * (v_val - sExpGC[t] * kS_val)
            sNK_A[v, t] = rhs.to(cutlass.BFloat16)
            acc_qS[idx] = acc_qS[idx] * sExpGC[t]
        cute.arch.barrier()

        # ==================== v_new = M @ RHS ====================
        acc_vnew = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        acc_vnew.fill(cutlass.Float32(0.0))

        tCsMA = thr_mma.partition_A(sA)
        tCsNKB = thr_mma.partition_B(sNK_A)
        tCrMA = thr_mma.make_fragment_A(tCsMA)
        tCrNKB = thr_mma.make_fragment_B(tCsNKB)
        for kk in cutlass.range_constexpr(BT // 16):
            cute.autovec_copy(tCsMA[None, None, kk], tCrMA[None, None, kk])
            cute.autovec_copy(tCsNKB[None, None, kk], tCrNKB[None, None, kk])
            cute.gemm(tiled_mma, acc_vnew, tCrMA[None, None, kk], tCrNKB[None, None, kk], acc_vnew)

        for idx in cutlass.range(cute.size(acc_vnew)):
            co = tCcC_bv[idx]
            sV[co[0], co[1]] = acc_vnew[idx].to(cutlass.BFloat16)
            sNK_A[co[1], co[0]] = acc_vnew[idx].to(cutlass.BFloat16)

        # ==================== INLINE CHUNK_O ====================
        _mma_qk_to_sA_gated(tiled_mma, thr_mma, sQ, sK, sA, sExpGC)
        cute.arch.barrier()

        tCsAA2 = thr_mma.partition_A(sA)
        tCsNKB2 = thr_mma.partition_B(sNK_A)
        tCrAA2 = thr_mma.make_fragment_A(tCsAA2)
        tCrNKB2 = thr_mma.make_fragment_B(tCsNKB2)
        for kk in cutlass.range_constexpr(BT // 16):
            cute.autovec_copy(tCsAA2[None, None, kk], tCrAA2[None, None, kk])
            cute.autovec_copy(tCsNKB2[None, None, kk], tCrNKB2[None, None, kk])
            cute.gemm(tiled_mma, acc_qS, tCrAA2[None, None, kk], tCrNKB2[None, None, kk], acc_qS)

        for idx in cutlass.range(cute.size(acc_qS)):
            co = tCcC_bv[idx]
            o_val = scale * acc_qS[idx]
            mO[i_b, t0 + co[0], i_hv, v_off + co[1]] = o_val.to(mO.element_type)

        # ==================== STATE UPDATE ====================
        if tidx < K_DIM:
            phi = sExpGC[BT - 1]
            for v in cutlass.range_constexpr(BV_TILE):
                state[v] = phi * state[v]
            for t in cutlass.range_constexpr(BT):
                kd = cutlass.Float32(sK[t, tidx]) * sExpDecay[t]
                for v in cutlass.range_constexpr(BV_TILE):
                    state[v] = state[v] + kd * cutlass.Float32(sV[t, v])

        if is_consumer:
            load_pipeline.consumer_release(cons_state)
            cons_state.advance()
        cute.arch.barrier()

    if is_producer:
        load_pipeline.producer_tail(prod_state)


# ============================================================================
# Launch wrappers
# ============================================================================

@cute.jit
def launch_kernel0(
    mQ, mK, mQnorm, mKnorm, mKK, mQK,
    B: cutlass.Int32,
    T: cutlass.Constexpr[int], H: cutlass.Constexpr[int],
    NT: cutlass.Constexpr[int],
    USE_FASTMATH: cutlass.Constexpr[bool] = True,
):
    mma_op = warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
    tC = cute.make_layout((2, 2, 1))
    perm = (2 * 16, 2 * 8 * 1, 1 * 16)
    tiled_mma = cute.make_tiled_mma(mma_op, tC, permutation_mnk=perm)

    sK_l = cute.make_layout((BT, K_DIM), stride=(K_DIM + 8, 1))
    smem = 2 * BT * (K_DIM + 8) * 2

    preprocess_kk_kernel(
        tiled_mma, sK_l,
        mQ, mK, mQnorm, mKnorm, mKK, mQK,
        T, H, NT, USE_FASTMATH,
    ).launch(
        grid=(NT, B * H, 1),
        block=(NUM_THREADS, 1, 1),
        smem=smem,
    )


@cute.jit
def launch_kernel0_tail(
    mQ, mK, mQnorm, mKnorm, mKK, mQK,
    B: cutlass.Int32,
    T: cutlass.Constexpr[int], H: cutlass.Constexpr[int],
    NT: cutlass.Constexpr[int],
):
    mma_op = warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
    tC = cute.make_layout((2, 2, 1))
    perm = (2 * 16, 2 * 8 * 1, 1 * 16)
    tiled_mma = cute.make_tiled_mma(mma_op, tC, permutation_mnk=perm)

    sK_l = cute.make_layout((BT, K_DIM), stride=(K_DIM + 2, 1))
    smem = 2 * BT * (K_DIM + 2) * 2

    preprocess_kk_tail_kernel(
        tiled_mma, sK_l,
        mQ, mK, mQnorm, mKnorm, mKK, mQK,
        T, H, NT,
    ).launch(
        grid=(NT, B * H, 1),
        block=(NUM_THREADS, 1, 1),
        smem=smem,
    )


@cute.jit
def launch_kernel0_kk_only(
    mQ, mK, mQnorm, mKnorm, mKK,
    B: cutlass.Int32,
    T: cutlass.Constexpr[int], H: cutlass.Constexpr[int],
    NT: cutlass.Constexpr[int],
):
    mma_op = warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
    tC = cute.make_layout((2, 2, 1))
    perm = (2 * 16, 2 * 8 * 1, 1 * 16)
    tiled_mma = cute.make_tiled_mma(mma_op, tC, permutation_mnk=perm)

    sK_l = cute.make_layout((BT, K_DIM), stride=(K_DIM + 2, 1))
    smem = BT * (K_DIM + 2) * 2

    preprocess_kk_only_kernel(
        tiled_mma, sK_l,
        mQ, mK, mQnorm, mKnorm, mKK,
        T, H, NT,
    ).launch(
        grid=(NT, B * H, 1),
        block=(NUM_THREADS, 1, 1),
        smem=smem,
    )


@cute.jit
def launch_kernel_inv(
    mKK, mQK, mGate, mBeta,
    mM, mGQK, mExpGC_out, mExpDecay_out,
    B: cutlass.Int32,
    T: cutlass.Constexpr[int], H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int], NT: cutlass.Constexpr[int],
    H_PER_HV: cutlass.Constexpr[int],
    USE_FASTMATH: cutlass.Constexpr[bool] = True,
):
    mma_op = warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
    tC = cute.make_layout((2, 2, 1))
    perm = (2 * 16, 2 * 8 * 1, 1 * 16)
    tiled_mma = cute.make_tiled_mma(mma_op, tC, permutation_mnk=perm)

    sA_l = cute.make_layout((BT, BT), stride=(BT + 2, 1))
    sA_T_l = cute.make_layout((BT, BT), stride=(BT + 2, 1))
    sGC_l = cute.make_layout((BT,), stride=(1,))
    sBeta_l = cute.make_layout((BT,), stride=(1,))

    smem = (
        BT * (BT + 2) * 4 +     # sA
        BT * 4 +                  # sGC
        BT * 4 +                  # sBeta
        BT * (BT + 2) * 4 +       # sInv
        BT * (BT + 2) * 4         # sTmp
    )

    precompute_inv_kernel(
        tiled_mma,
        sA_l, sA_T_l, sGC_l, sBeta_l,
        mKK, mQK, mGate, mBeta,
        mM, mGQK, mExpGC_out, mExpDecay_out,
        T, H, HV, NT, H_PER_HV, False, USE_FASTMATH,
    ).launch(
        grid=(NT, B * HV, 1),
        block=(NUM_THREADS, 1, 1),
        smem=smem,
    )


@cute.jit
def launch_kernel_inv_betafold(
    mKK, mQK, mGate, mBeta,
    mM, mGQK, mExpGC_out, mExpDecay_out,
    B: cutlass.Int32,
    T: cutlass.Constexpr[int], H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int], NT: cutlass.Constexpr[int],
    H_PER_HV: cutlass.Constexpr[int],
    USE_FASTMATH: cutlass.Constexpr[bool] = True,
):
    mma_op = warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
    tC = cute.make_layout((2, 2, 1))
    perm = (2 * 16, 2 * 8 * 1, 1 * 16)
    tiled_mma = cute.make_tiled_mma(mma_op, tC, permutation_mnk=perm)

    sA_l = cute.make_layout((BT, BT), stride=(BT, 1))
    sA_T_l = cute.make_layout((BT, BT), stride=(BT, 1))
    sGC_l = cute.make_layout((BT,), stride=(1,))
    sBeta_l = cute.make_layout((BT,), stride=(1,))

    smem = (
        BT * BT * 4 +
        BT * 4 +
        BT * 4 +
        BT * BT * 4 +
        BT * BT * 4
    )

    precompute_inv_kernel(
        tiled_mma,
        sA_l, sA_T_l, sGC_l, sBeta_l,
        mKK, mQK, mGate, mBeta,
        mM, mGQK, mExpGC_out, mExpDecay_out,
        T, H, HV, NT, H_PER_HV, True, USE_FASTMATH,
    ).launch(
        grid=(NT, B * HV, 1),
        block=(NUM_THREADS, 1, 1),
        smem=smem,
    )


@cute.jit
def launch_kernel_inv_tail(
    mKK, mQK, mGate, mBeta,
    mM, mGQK, mExpGC_out, mExpDecay_out,
    B: cutlass.Int32,
    T: cutlass.Constexpr[int], H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int], NT: cutlass.Constexpr[int],
    H_PER_HV: cutlass.Constexpr[int],
):
    mma_op = warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
    tC = cute.make_layout((2, 2, 1))
    perm = (2 * 16, 2 * 8 * 1, 1 * 16)
    tiled_mma = cute.make_tiled_mma(mma_op, tC, permutation_mnk=perm)

    sA_l = cute.make_layout((BT, BT), stride=(BT + 2, 1))
    sA_T_l = cute.make_layout((BT, BT), stride=(BT + 2, 1))
    sGC_l = cute.make_layout((BT,), stride=(1,))
    sBeta_l = cute.make_layout((BT,), stride=(1,))

    smem = (
        BT * (BT + 2) * 4 +
        BT * 4 +
        BT * 4 +
        BT * (BT + 2) * 4 +
        BT * (BT + 2) * 4
    )

    precompute_inv_tail_kernel(
        tiled_mma,
        sA_l, sA_T_l, sGC_l, sBeta_l,
        mKK, mQK, mGate, mBeta,
        mM, mGQK, mExpGC_out, mExpDecay_out,
        T, H, HV, NT, H_PER_HV,
    ).launch(
        grid=(NT, B * HV, 1),
        block=(NUM_THREADS, 1, 1),
        smem=smem,
    )


@cute.jit
def launch_kernel1(
    mKnorm, mQnorm, mV, mBeta,
    mM, mGQK, mExpGC_in, mExpDecay_in,
    mO,
    scale: cutlass.Constexpr[float],
    B: cutlass.Int32,
    T: cutlass.Constexpr[int], H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int], V: cutlass.Constexpr[int],
    NT: cutlass.Constexpr[int], H_PER_HV: cutlass.Constexpr[int],
    BV_TILE: cutlass.Constexpr[int], PER_V_TILE: cutlass.Constexpr[int],
):
    mma_op = warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
    tC = cute.make_layout((2, 2, 1))
    perm = (2 * 16, 2 * 8 * 1, 1 * 16)
    tiled_mma = cute.make_tiled_mma(mma_op, tC, permutation_mnk=perm)

    cp_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.BFloat16,
        num_bits_per_copy=32,
    )
    thr_layout_kq = cute.make_layout((32, 4), stride=(4, 1))
    val_layout_kq = cute.make_layout((1, 8), stride=(8, 1))
    tiled_copy_kq = cute.make_tiled_copy_tv(cp_atom, thr_layout_kq, val_layout_kq)

    KQ_STRIDE = K_DIM + 8
    sK_l = cute.make_layout((BT, K_DIM), stride=(KQ_STRIDE, 1))
    sV_l = cute.make_layout((BT, BV_TILE), stride=(BV_TILE + 2, 1))
    sA_l = cute.make_layout((BT, BT), stride=(BT + 2, 1))
    sGQK_l = cute.make_layout((BT, BT), stride=(BT + 2, 1))
    sBeta_l = cute.make_layout((BT,), stride=(1,))
    sS_l = cute.make_layout((BV_TILE, K_DIM), stride=(K_DIM + 2, 1))
    sNK_l = cute.make_layout((BV_TILE, BT), stride=(BT + 2, 1))
    sExpGC_l = cute.make_layout((BT,), stride=(1,))
    sExpDecay_l = cute.make_layout((BT,), stride=(1,))

    smem = (
        BT * KQ_STRIDE * 2 * 2 +       # sK + sQ (same layout, 16B-aligned rows)
        BT * (BV_TILE + 2) * 2 +        # sV
        BT * (BT + 2) * 2 +             # sA
        BT * (BT + 2) * 2 +             # sGQK
        BT * 4 +                         # sBeta
        BV_TILE * (K_DIM + 2) * 2 +     # sS
        BV_TILE * (BT + 2) * 2 +        # sNK_A
        BT * 4 +                         # sExpGC
        BT * 4                           # sExpDecay
    )

    fused_chunk_h_kernel(
        tiled_mma,
        tiled_copy_kq,
        sK_l, sV_l, sA_l, sGQK_l,
        sBeta_l,
        sS_l, sNK_l,
        sExpGC_l, sExpDecay_l,
        mKnorm, mQnorm, mV, mBeta,
        mM, mGQK, mExpGC_in, mExpDecay_in,
        mO,
        scale, T, H, HV, V, NT, H_PER_HV, BV_TILE, PER_V_TILE,
    ).launch(
        grid=(V // BV_TILE, B * HV, 1),
        block=(NUM_THREADS, 1, 1),
        smem=smem,
    )


@cute.jit
def launch_kernel1_mgqk_cpasync_probe(
    mKnorm, mQnorm, mV, mBeta,
    mM, mGQK, mExpGC_in, mExpDecay_in,
    mO,
    scale: cutlass.Constexpr[float],
    B: cutlass.Int32,
    T: cutlass.Constexpr[int], H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int], V: cutlass.Constexpr[int],
    NT: cutlass.Constexpr[int], H_PER_HV: cutlass.Constexpr[int],
    BV_TILE: cutlass.Constexpr[int], PER_V_TILE: cutlass.Constexpr[int],
):
    mma_op = warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
    tC = cute.make_layout((2, 2, 1))
    perm = (2 * 16, 2 * 8 * 1, 1 * 16)
    tiled_mma = cute.make_tiled_mma(mma_op, tC, permutation_mnk=perm)

    cp_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.STREAMING),
        cutlass.BFloat16,
        num_bits_per_copy=128,
    )
    thr_layout_kq = cute.make_layout((32, 4), stride=(4, 1))
    val_layout_kq = cute.make_layout((1, 8), stride=(8, 1))
    tiled_copy_kq = cute.make_tiled_copy_tv(cp_atom, thr_layout_kq, val_layout_kq)

    thr_layout_mgqk = cute.make_layout((32, 4), stride=(4, 1))
    val_layout_mgqk = cute.make_layout((1, 8), stride=(8, 1))
    tiled_copy_mgqk = cute.make_tiled_copy_tv(cp_atom, thr_layout_mgqk, val_layout_mgqk)

    KQ_STRIDE = K_DIM + 8
    A_STRIDE = BT + 8
    sK_l = cute.make_layout((BT, K_DIM), stride=(KQ_STRIDE, 1))
    sV_l = cute.make_layout((BT, BV_TILE), stride=(BV_TILE + 2, 1))
    sA_l = cute.make_layout((BT, BT), stride=(A_STRIDE, 1))
    sGQK_l = cute.make_layout((BT, BT), stride=(A_STRIDE, 1))
    sBeta_l = cute.make_layout((BT,), stride=(1,))
    sS_l = cute.make_layout((BV_TILE, K_DIM), stride=(K_DIM + 2, 1))
    sNK_l = cute.make_layout((BV_TILE, BT), stride=(BT + 2, 1))
    sExpGC_l = cute.make_layout((BT,), stride=(1,))
    sExpDecay_l = cute.make_layout((BT,), stride=(1,))

    smem = (
        BT * KQ_STRIDE * 2 * 2 +       # sK + sQ
        BT * (BV_TILE + 2) * 2 +        # sV
        BT * A_STRIDE * 2 +             # sA, 16B-aligned rows for cp.async
        BT * A_STRIDE * 2 +             # sGQK, 16B-aligned rows for cp.async
        BT * 4 +                         # sBeta
        BV_TILE * (K_DIM + 2) * 2 +     # sS
        BV_TILE * (BT + 2) * 2 +        # sNK_A
        BT * 4 +                         # sExpGC
        BT * 4                           # sExpDecay
    )

    fused_chunk_h_mgqk_cpasync_probe_kernel(
        tiled_mma,
        tiled_copy_kq,
        tiled_copy_mgqk,
        sK_l, sV_l, sA_l, sGQK_l,
        sBeta_l,
        sS_l, sNK_l,
        sExpGC_l, sExpDecay_l,
        mKnorm, mQnorm, mV, mBeta,
        mM, mGQK, mExpGC_in, mExpDecay_in,
        mO,
        scale, T, H, HV, V, NT, H_PER_HV, BV_TILE, PER_V_TILE,
    ).launch(
        grid=(V // BV_TILE, B * HV, 1),
        block=(NUM_THREADS, 1, 1),
        smem=smem,
    )


@cute.jit
def launch_kernel1_mgqk_v_cpasync_probe(
    mKnorm, mQnorm, mV, mBeta,
    mM, mGQK, mExpGC_in, mExpDecay_in,
    mO,
    scale: cutlass.Constexpr[float],
    B: cutlass.Int32,
    T: cutlass.Constexpr[int], H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int], V: cutlass.Constexpr[int],
    NT: cutlass.Constexpr[int], H_PER_HV: cutlass.Constexpr[int],
    BV_TILE: cutlass.Constexpr[int], PER_V_TILE: cutlass.Constexpr[int],
):
    mma_op = warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
    tC = cute.make_layout((2, 2, 1))
    perm = (2 * 16, 2 * 8 * 1, 1 * 16)
    tiled_mma = cute.make_tiled_mma(mma_op, tC, permutation_mnk=perm)

    cp_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.BFloat16,
        num_bits_per_copy=128,
    )
    thr_layout_kq = cute.make_layout((32, 4), stride=(4, 1))
    val_layout_kq = cute.make_layout((1, 8), stride=(8, 1))
    tiled_copy_kq = cute.make_tiled_copy_tv(cp_atom, thr_layout_kq, val_layout_kq)

    thr_layout_mgqk = cute.make_layout((32, 4), stride=(4, 1))
    val_layout_mgqk = cute.make_layout((1, 8), stride=(8, 1))
    tiled_copy_mgqk = cute.make_tiled_copy_tv(cp_atom, thr_layout_mgqk, val_layout_mgqk)

    cp_atom_v = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.BFloat16,
        num_bits_per_copy=128,
    )
    thr_layout_v = cute.make_layout((32, 2), stride=(2, 1))
    val_layout_v = cute.make_layout((1, 8), stride=(8, 1))
    tiled_copy_v = cute.make_tiled_copy_tv(cp_atom_v, thr_layout_v, val_layout_v)

    KQ_STRIDE = K_DIM + 8
    A_STRIDE = BT + 8
    V_STRIDE = BV_TILE + 8
    sK_l = cute.make_layout((BT, K_DIM), stride=(KQ_STRIDE, 1))
    sV_l = cute.make_layout((BT, BV_TILE), stride=(V_STRIDE, 1))
    sA_l = cute.make_layout((BT, BT), stride=(A_STRIDE, 1))
    sGQK_l = cute.make_layout((BT, BT), stride=(A_STRIDE, 1))
    sBeta_l = cute.make_layout((BT,), stride=(1,))
    sS_l = cute.make_layout((BV_TILE, K_DIM), stride=(K_DIM + 2, 1))
    sNK_l = cute.make_layout((BV_TILE, BT), stride=(BT + 2, 1))
    sExpGC_l = cute.make_layout((BT,), stride=(1,))
    sExpDecay_l = cute.make_layout((BT,), stride=(1,))

    smem = (
        BT * KQ_STRIDE * 2 * 2 +       # sK + sQ
        BT * V_STRIDE * 2 +             # sV, aligned for vector cp.async
        BT * A_STRIDE * 2 +             # sA, 16B-aligned rows for cp.async
        BT * A_STRIDE * 2 +             # sGQK, 16B-aligned rows for cp.async
        BT * 4 +                         # sBeta
        BV_TILE * (K_DIM + 2) * 2 +     # sS
        BV_TILE * (BT + 2) * 2 +        # sNK_A
        BT * 4 +                         # sExpGC
        BT * 4                           # sExpDecay
    )

    fused_chunk_h_mgqk_v_cpasync_probe_kernel(
        tiled_mma,
        tiled_copy_kq,
        tiled_copy_mgqk,
        tiled_copy_v,
        sK_l, sV_l, sA_l, sGQK_l,
        sBeta_l,
        sS_l, sNK_l,
        sExpGC_l, sExpDecay_l,
        mKnorm, mQnorm, mV, mBeta,
        mM, mGQK, mExpGC_in, mExpDecay_in,
        mO,
        scale, T, H, HV, V, NT, H_PER_HV, BV_TILE, PER_V_TILE,
    ).launch(
        grid=(V // BV_TILE, B * HV, 1),
        block=(NUM_THREADS, 1, 1),
        smem=smem,
    )


@cute.jit
def launch_kernel1_mgqk_v_fp32state_cpasync_probe(
    mKnorm, mQnorm, mV, mBeta,
    mM, mGQK, mExpGC_in, mExpDecay_in,
    mO,
    scale: cutlass.Constexpr[float],
    B: cutlass.Int32,
    T: cutlass.Constexpr[int], H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int], V: cutlass.Constexpr[int],
    NT: cutlass.Constexpr[int], H_PER_HV: cutlass.Constexpr[int],
    BV_TILE: cutlass.Constexpr[int], PER_V_TILE: cutlass.Constexpr[int],
):
    mma_op = warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
    tC = cute.make_layout((2, 2, 1))
    perm = (2 * 16, 2 * 8 * 1, 1 * 16)
    tiled_mma = cute.make_tiled_mma(mma_op, tC, permutation_mnk=perm)

    cp_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.BFloat16,
        num_bits_per_copy=128,
    )
    thr_layout_kq = cute.make_layout((32, 4), stride=(4, 1))
    val_layout_kq = cute.make_layout((1, 8), stride=(8, 1))
    tiled_copy_kq = cute.make_tiled_copy_tv(cp_atom, thr_layout_kq, val_layout_kq)

    thr_layout_mgqk = cute.make_layout((32, 4), stride=(4, 1))
    val_layout_mgqk = cute.make_layout((1, 8), stride=(8, 1))
    tiled_copy_mgqk = cute.make_tiled_copy_tv(cp_atom, thr_layout_mgqk, val_layout_mgqk)

    cp_atom_v = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.BFloat16,
        num_bits_per_copy=128,
    )
    thr_layout_v = cute.make_layout((32, 2), stride=(2, 1))
    val_layout_v = cute.make_layout((1, 8), stride=(8, 1))
    tiled_copy_v = cute.make_tiled_copy_tv(cp_atom_v, thr_layout_v, val_layout_v)

    KQ_STRIDE = K_DIM + 8
    A_STRIDE = BT + 8
    V_STRIDE = BV_TILE + 8
    V_STATE_STRIDE = BV_TILE
    sK_l = cute.make_layout((BT, K_DIM), stride=(KQ_STRIDE, 1))
    sV_l = cute.make_layout((BT, BV_TILE), stride=(V_STRIDE, 1))
    sVState_l = cute.make_layout((BT, BV_TILE), stride=(V_STATE_STRIDE, 1))
    sA_l = cute.make_layout((BT, BT), stride=(A_STRIDE, 1))
    sGQK_l = cute.make_layout((BT, BT), stride=(A_STRIDE, 1))
    sBeta_l = cute.make_layout((BT,), stride=(1,))
    sS_l = cute.make_layout((BV_TILE, K_DIM), stride=(K_DIM + 2, 1))
    sNK_l = cute.make_layout((BV_TILE, BT), stride=(BT + 2, 1))
    sExpGC_l = cute.make_layout((BT,), stride=(1,))
    sExpDecay_l = cute.make_layout((BT,), stride=(1,))

    smem = (
        BT * KQ_STRIDE * 2 * 2 +        # sK + sQ
        BT * V_STRIDE * 2 +             # sV, aligned for vector cp.async
        BT * V_STATE_STRIDE * 4 +       # sVState fp32 for recurrent update
        BT * A_STRIDE * 2 +             # sA, 16B-aligned rows for cp.async
        BT * A_STRIDE * 2 +             # sGQK, 16B-aligned rows for cp.async
        BT * 4 +                        # sBeta
        BV_TILE * (K_DIM + 2) * 2 +     # sS
        BV_TILE * (BT + 2) * 2 +        # sNK_A
        BT * 4 +                        # sExpGC
        BT * 4                          # sExpDecay
    )

    fused_chunk_h_mgqk_v_fp32state_cpasync_probe_kernel(
        tiled_mma,
        tiled_copy_kq,
        tiled_copy_mgqk,
        tiled_copy_v,
        sK_l, sV_l, sVState_l, sA_l, sGQK_l,
        sBeta_l,
        sS_l, sNK_l,
        sExpGC_l, sExpDecay_l,
        mKnorm, mQnorm, mV, mBeta,
        mM, mGQK, mExpGC_in, mExpDecay_in,
        mO,
        scale, T, H, HV, V, NT, H_PER_HV, BV_TILE, PER_V_TILE,
    ).launch(
        grid=(V // BV_TILE, B * HV, 1),
        block=(NUM_THREADS, 1, 1),
        smem=smem,
    )


@cute.jit
def launch_kernel1_mgqk_v_ldsm_probe(
    mKnorm, mQnorm, mV, mBeta,
    mM, mGQK, mExpGC_in, mExpDecay_in,
    mO,
    scale: cutlass.Constexpr[float],
    B: cutlass.Int32,
    T: cutlass.Constexpr[int], H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int], V: cutlass.Constexpr[int],
    NT: cutlass.Constexpr[int], H_PER_HV: cutlass.Constexpr[int],
    BV_TILE: cutlass.Constexpr[int], PER_V_TILE: cutlass.Constexpr[int],
    USE_STATE_FRAG: cutlass.Constexpr[bool],
    USE_DUAL_STATE_BUFFER: cutlass.Constexpr[bool],
    USE_DEDICATED_STATE_SCRATCH: cutlass.Constexpr[bool],
    USE_STATE_R2S: cutlass.Constexpr[bool],
    USE_VECTOR_O_GMEM: cutlass.Constexpr[bool],
    USE_KDECAY_R2S: cutlass.Constexpr[bool],
    USE_RHS_VNEW_R2S: cutlass.Constexpr[bool],
    USE_SKIP_SV_VNEW: cutlass.Constexpr[bool],
    USE_DIRECT_V_RHS: cutlass.Constexpr[bool],
    USE_SCALED_VNEW_STATE: cutlass.Constexpr[bool],
    USE_AUTOVEC_MMA: cutlass.Constexpr[bool],
    USE_AUTOVEC_STATE_MMA: cutlass.Constexpr[bool],
    USE_AUTOVEC_KQS_MMA: cutlass.Constexpr[bool],
    USE_AUTOVEC_KS_MMA: cutlass.Constexpr[bool],
    USE_AUTOVEC_QS_MMA: cutlass.Constexpr[bool],
    USE_AUTOVEC_VNEW_MMA: cutlass.Constexpr[bool],
    USE_AUTOVEC_CHUNKO_MMA: cutlass.Constexpr[bool],
    USE_SCALAR_SCALED_VNEW_STORE: cutlass.Constexpr[bool],
    USE_SCALAR_VNEW_STORE: cutlass.Constexpr[bool],
    USE_SCALAR_RHS_STORE: cutlass.Constexpr[bool],
):
    mma_op = warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
    tC = cute.make_layout((2, 2, 1))
    perm = (2 * 16, 2 * 8 * 1, 1 * 16)
    tiled_mma = cute.make_tiled_mma(mma_op, tC, permutation_mnk=perm)

    smem_copy_atom_A = cute.make_copy_atom(
        warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
        cutlass.BFloat16,
    )
    smem_copy_atom_A_trans = cute.make_copy_atom(
        warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
        cutlass.BFloat16,
    )
    smem_copy_atom_B = cute.make_copy_atom(
        warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
        cutlass.BFloat16,
    )
    smem_tiled_copy_A = cute.make_tiled_copy_A(smem_copy_atom_A, tiled_mma)
    smem_tiled_copy_A_trans = cute.make_tiled_copy_A(smem_copy_atom_A_trans, tiled_mma)
    smem_tiled_copy_B = cute.make_tiled_copy_B(smem_copy_atom_B, tiled_mma)

    copy_atom_state_r2s = cute.make_copy_atom(
        warp.StMatrix8x8x16bOp(transpose=True, num_matrices=2),
        cutlass.BFloat16,
    )
    tiled_copy_state_C = cute.make_tiled_copy_C_atom(copy_atom_state_r2s, tiled_mma)
    tiled_copy_state_r2s = cute.make_tiled_copy_S(
        copy_atom_state_r2s,
        tiled_copy_state_C,
    )

    copy_atom_o_gmem = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        cutlass.BFloat16,
        num_bits_per_copy=128,
    )
    tiled_copy_o_gmem = cute.make_tiled_copy_C(copy_atom_o_gmem, tiled_mma)

    copy_atom_kdecay_r2s = cute.make_copy_atom(
        warp.StMatrix8x8x16bOp(transpose=False, num_matrices=2),
        cutlass.BFloat16,
    )
    tiled_copy_kdecay_C = cute.make_tiled_copy_C_atom(copy_atom_kdecay_r2s, tiled_mma)
    tiled_copy_kdecay_r2s = cute.make_tiled_copy_S(
        copy_atom_kdecay_r2s,
        tiled_copy_kdecay_C,
    )

    cp_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.BFloat16,
        num_bits_per_copy=128,
    )
    thr_layout_kq = cute.make_layout((32, 4), stride=(4, 1))
    val_layout_kq = cute.make_layout((1, 8), stride=(8, 1))
    tiled_copy_kq = cute.make_tiled_copy_tv(cp_atom, thr_layout_kq, val_layout_kq)

    thr_layout_mgqk = cute.make_layout((32, 4), stride=(4, 1))
    val_layout_mgqk = cute.make_layout((1, 8), stride=(8, 1))
    tiled_copy_mgqk = cute.make_tiled_copy_tv(cp_atom, thr_layout_mgqk, val_layout_mgqk)

    cp_atom_v = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.BFloat16,
        num_bits_per_copy=128,
    )
    thr_layout_v = cute.make_layout((32, 2), stride=(2, 1))
    val_layout_v = cute.make_layout((1, 8), stride=(8, 1))
    tiled_copy_v = cute.make_tiled_copy_tv(cp_atom_v, thr_layout_v, val_layout_v)

    KQ_STRIDE = K_DIM + 8
    A_STRIDE = BT + 8
    V_STRIDE = BV_TILE + 8
    S_STRIDE = K_DIM
    NK_STRIDE = BT
    sK_l = cute.make_layout((BT, K_DIM), stride=(KQ_STRIDE, 1))
    sV_l = cute.make_layout((BT, BV_TILE), stride=(V_STRIDE, 1))
    sA_l = cute.make_layout((BT, BT), stride=(A_STRIDE, 1))
    sGQK_l = cute.make_layout((BT, BT), stride=(A_STRIDE, 1))
    sBeta_l = cute.make_layout((BT,), stride=(1,))
    sS_l = cute.make_layout((BV_TILE, K_DIM), stride=(S_STRIDE, 1))
    sNK_l = cute.make_layout((BV_TILE, BT), stride=(NK_STRIDE, 1))
    sExpGC_l = cute.make_layout((BT,), stride=(1,))
    sExpDecay_l = cute.make_layout((BT,), stride=(1,))

    smem_base = (
        BT * KQ_STRIDE * 2 * 2 +       # sK + sQ
        BT * V_STRIDE * 2 +             # sV, aligned for vector cp.async
        BT * A_STRIDE * 2 +             # sA, 16B-aligned rows for cp.async
        BT * A_STRIDE * 2 +             # sGQK, 16B-aligned rows for cp.async
        BT * 4 +                        # sBeta
        BV_TILE * S_STRIDE * 2 +        # sS, 16B-aligned rows for ldmatrix
        BV_TILE * NK_STRIDE * 2 +       # sNK_A, 16B-aligned rows for ldmatrix
        BT * 4 +                        # sExpGC
        BT * 4                          # sExpDecay
    )
    if cutlass.const_expr(USE_DEDICATED_STATE_SCRATCH):
        smem = smem_base + BT * A_STRIDE * 2
    else:
        smem = smem_base

    fused_chunk_h_mgqk_v_ldsm_probe_kernel(
        tiled_mma,
        smem_tiled_copy_A,
        smem_tiled_copy_A_trans,
        smem_tiled_copy_B,
        tiled_copy_state_r2s,
        tiled_copy_o_gmem,
        tiled_copy_kdecay_r2s,
        tiled_copy_kq,
        tiled_copy_mgqk,
        tiled_copy_v,
        sK_l, sV_l, sA_l, sGQK_l,
        sBeta_l,
        sS_l, sNK_l,
        sExpGC_l, sExpDecay_l,
        mKnorm, mQnorm, mV, mBeta,
        mM, mGQK, mExpGC_in, mExpDecay_in,
        mO,
        scale, T, H, HV, V, NT, H_PER_HV, BV_TILE, PER_V_TILE,
        USE_STATE_FRAG, USE_DUAL_STATE_BUFFER, USE_DEDICATED_STATE_SCRATCH,
        USE_STATE_R2S, USE_VECTOR_O_GMEM, USE_KDECAY_R2S,
        USE_RHS_VNEW_R2S, USE_SKIP_SV_VNEW, USE_DIRECT_V_RHS,
        USE_SCALED_VNEW_STATE, USE_AUTOVEC_MMA, USE_AUTOVEC_STATE_MMA,
        USE_AUTOVEC_KQS_MMA, USE_AUTOVEC_KS_MMA, USE_AUTOVEC_QS_MMA,
        USE_AUTOVEC_VNEW_MMA, USE_AUTOVEC_CHUNKO_MMA,
        USE_SCALAR_SCALED_VNEW_STORE, USE_SCALAR_VNEW_STORE,
        USE_SCALAR_RHS_STORE,
    ).launch(
        grid=(V // BV_TILE, B * HV, 1),
        block=(NUM_THREADS, 1, 1),
        smem=smem,
    )


@cute.jit
def launch_kernel1_mgqk_v_ldsm_state_mma_probe(
    mKnorm, mQnorm, mV, mBeta,
    mM, mGQK, mExpGC_in, mExpDecay_in,
    mO,
    scale: cutlass.Constexpr[float],
    B: cutlass.Int32,
    T: cutlass.Constexpr[int], H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int], V: cutlass.Constexpr[int],
    NT: cutlass.Constexpr[int], H_PER_HV: cutlass.Constexpr[int],
    BV_TILE: cutlass.Constexpr[int], PER_V_TILE: cutlass.Constexpr[int],
    PER_STATE_TILE: cutlass.Constexpr[int],
):
    mma_op = warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
    tC = cute.make_layout((2, 2, 1))
    perm = (2 * 16, 2 * 8 * 1, 1 * 16)
    tiled_mma = cute.make_tiled_mma(mma_op, tC, permutation_mnk=perm)

    smem_copy_atom_A = cute.make_copy_atom(
        warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
        cutlass.BFloat16,
    )
    smem_copy_atom_B = cute.make_copy_atom(
        warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
        cutlass.BFloat16,
    )
    smem_tiled_copy_A = cute.make_tiled_copy_A(smem_copy_atom_A, tiled_mma)
    smem_tiled_copy_B = cute.make_tiled_copy_B(smem_copy_atom_B, tiled_mma)

    cp_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.BFloat16,
        num_bits_per_copy=128,
    )
    thr_layout_kq = cute.make_layout((32, 4), stride=(4, 1))
    val_layout_kq = cute.make_layout((1, 8), stride=(8, 1))
    tiled_copy_kq = cute.make_tiled_copy_tv(cp_atom, thr_layout_kq, val_layout_kq)

    thr_layout_mgqk = cute.make_layout((32, 4), stride=(4, 1))
    val_layout_mgqk = cute.make_layout((1, 8), stride=(8, 1))
    tiled_copy_mgqk = cute.make_tiled_copy_tv(cp_atom, thr_layout_mgqk, val_layout_mgqk)

    cp_atom_v = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.BFloat16,
        num_bits_per_copy=128,
    )
    thr_layout_v = cute.make_layout((32, 2), stride=(2, 1))
    val_layout_v = cute.make_layout((1, 8), stride=(8, 1))
    tiled_copy_v = cute.make_tiled_copy_tv(cp_atom_v, thr_layout_v, val_layout_v)

    KQ_STRIDE = K_DIM + 8
    A_STRIDE = BT + 8
    V_STRIDE = BV_TILE + 8
    S_STRIDE = K_DIM + 8
    NK_STRIDE = BT + 8
    STATE_STRIDE = BV_TILE
    sK_l = cute.make_layout((BT, K_DIM), stride=(KQ_STRIDE, 1))
    sV_l = cute.make_layout((BT, BV_TILE), stride=(V_STRIDE, 1))
    sA_l = cute.make_layout((BT, BT), stride=(A_STRIDE, 1))
    sGQK_l = cute.make_layout((BT, BT), stride=(A_STRIDE, 1))
    sBeta_l = cute.make_layout((BT,), stride=(1,))
    sS_l = cute.make_layout((BV_TILE, K_DIM), stride=(S_STRIDE, 1))
    sNK_l = cute.make_layout((BV_TILE, BT), stride=(NK_STRIDE, 1))
    sState_l = cute.make_layout((K_DIM, BV_TILE), stride=(STATE_STRIDE, 1))
    sExpGC_l = cute.make_layout((BT,), stride=(1,))
    sExpDecay_l = cute.make_layout((BT,), stride=(1,))

    smem = (
        BT * KQ_STRIDE * 2 * 2 +       # sK + sQ
        BT * V_STRIDE * 2 +             # sV, aligned for vector cp.async
        BT * A_STRIDE * 2 +             # sA, 16B-aligned rows for cp.async/ldsm
        BT * A_STRIDE * 2 +             # sGQK, 16B-aligned rows for cp.async/ldsm
        BT * 4 +                        # sBeta
        BV_TILE * S_STRIDE * 2 +        # sS, 16B-aligned rows for ldmatrix
        BV_TILE * NK_STRIDE * 2 +       # sNK_A, 16B-aligned rows for ldmatrix
        K_DIM * STATE_STRIDE * 4 +      # fp32 recurrent state in shared memory
        BT * 4 +                        # sExpGC
        BT * 4                          # sExpDecay
    )

    fused_chunk_h_mgqk_v_ldsm_state_mma_probe_kernel(
        tiled_mma,
        smem_tiled_copy_A,
        smem_tiled_copy_B,
        tiled_copy_kq,
        tiled_copy_mgqk,
        tiled_copy_v,
        sK_l, sV_l, sA_l, sGQK_l,
        sBeta_l,
        sS_l, sNK_l, sState_l,
        sExpGC_l, sExpDecay_l,
        mKnorm, mQnorm, mV, mBeta,
        mM, mGQK, mExpGC_in, mExpDecay_in,
        mO,
        scale, T, H, HV, V, NT, H_PER_HV, BV_TILE, PER_V_TILE,
        PER_STATE_TILE,
    ).launch(
        grid=(V // BV_TILE, B * HV, 1),
        block=(NUM_THREADS, 1, 1),
        smem=smem,
    )


@cute.jit
def launch_kernel1_split_h_cpasync_probe(
    mKnorm, mV, mBeta,
    mM, mExpGC_in, mExpDecay_in,
    mH, mVNew,
    B: cutlass.Int32,
    T: cutlass.Constexpr[int], H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int], V: cutlass.Constexpr[int],
    NT: cutlass.Constexpr[int], H_PER_HV: cutlass.Constexpr[int],
    BV_TILE: cutlass.Constexpr[int], PER_V_TILE: cutlass.Constexpr[int],
):
    mma_op = warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
    tC = cute.make_layout((2, 2, 1))
    perm = (2 * 16, 2 * 8 * 1, 1 * 16)
    tiled_mma = cute.make_tiled_mma(mma_op, tC, permutation_mnk=perm)

    cp_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.BFloat16,
        num_bits_per_copy=128,
    )
    thr_layout_k = cute.make_layout((32, 4), stride=(4, 1))
    val_layout_k = cute.make_layout((1, 8), stride=(8, 1))
    tiled_copy_k = cute.make_tiled_copy_tv(cp_atom, thr_layout_k, val_layout_k)

    thr_layout_m = cute.make_layout((32, 4), stride=(4, 1))
    val_layout_m = cute.make_layout((1, 8), stride=(8, 1))
    tiled_copy_m = cute.make_tiled_copy_tv(cp_atom, thr_layout_m, val_layout_m)

    thr_layout_v = cute.make_layout((32, 2), stride=(2, 1))
    val_layout_v = cute.make_layout((1, 8), stride=(8, 1))
    tiled_copy_v = cute.make_tiled_copy_tv(cp_atom, thr_layout_v, val_layout_v)

    KQ_STRIDE = K_DIM + 8
    A_STRIDE = BT + 8
    V_STRIDE = BV_TILE + 8
    sK_l = cute.make_layout((BT, K_DIM), stride=(KQ_STRIDE, 1))
    sV_l = cute.make_layout((BT, BV_TILE), stride=(V_STRIDE, 1))
    sA_l = cute.make_layout((BT, BT), stride=(A_STRIDE, 1))
    sBeta_l = cute.make_layout((BT,), stride=(1,))
    sS_l = cute.make_layout((BV_TILE, K_DIM), stride=(K_DIM + 2, 1))
    sNK_l = cute.make_layout((BV_TILE, BT), stride=(BT + 2, 1))
    sExpGC_l = cute.make_layout((BT,), stride=(1,))
    sExpDecay_l = cute.make_layout((BT,), stride=(1,))

    smem = (
        BT * KQ_STRIDE * 2 +             # sK
        BT * V_STRIDE * 2 +              # sV
        BT * A_STRIDE * 2 +              # sA
        BT * 4 +                         # sBeta
        BV_TILE * (K_DIM + 2) * 2 +      # sS
        BV_TILE * (BT + 2) * 2 +         # sNK_A
        BT * 4 +                         # sExpGC
        BT * 4                           # sExpDecay
    )

    chunk_h_store_vnew_cpasync_probe_kernel(
        tiled_mma,
        tiled_copy_k,
        tiled_copy_m,
        tiled_copy_v,
        sK_l, sV_l, sA_l,
        sBeta_l,
        sS_l, sNK_l,
        sExpGC_l, sExpDecay_l,
        mKnorm, mV, mBeta,
        mM, mExpGC_in, mExpDecay_in,
        mH, mVNew,
        T, H, HV, V, NT, H_PER_HV, BV_TILE, PER_V_TILE,
    ).launch(
        grid=(V // BV_TILE, B * HV, 1),
        block=(NUM_THREADS, 1, 1),
        smem=smem,
    )


@cute.jit
def launch_kernel2_split_o_cpasync_probe(
    mQnorm, mH, mGQK, mVNew, mExpGC_in, mO,
    scale: cutlass.Constexpr[float],
    B: cutlass.Int32,
    T: cutlass.Constexpr[int], H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int], V: cutlass.Constexpr[int],
    NT: cutlass.Constexpr[int], H_PER_HV: cutlass.Constexpr[int],
    BV_TILE: cutlass.Constexpr[int],
):
    mma_op = warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
    tC = cute.make_layout((2, 2, 1))
    perm = (2 * 16, 2 * 8 * 1, 1 * 16)
    tiled_mma = cute.make_tiled_mma(mma_op, tC, permutation_mnk=perm)

    cp_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.BFloat16,
        num_bits_per_copy=128,
    )
    thr_layout_q = cute.make_layout((32, 4), stride=(4, 1))
    val_layout_q = cute.make_layout((1, 8), stride=(8, 1))
    tiled_copy_q = cute.make_tiled_copy_tv(cp_atom, thr_layout_q, val_layout_q)

    thr_layout_h = cute.make_layout((32, 2), stride=(2, 1))
    val_layout_h = cute.make_layout((1, 8), stride=(8, 1))
    tiled_copy_h = cute.make_tiled_copy_tv(cp_atom, thr_layout_h, val_layout_h)

    thr_layout_a = cute.make_layout((32, 4), stride=(4, 1))
    val_layout_a = cute.make_layout((1, 8), stride=(8, 1))
    tiled_copy_a = cute.make_tiled_copy_tv(cp_atom, thr_layout_a, val_layout_a)

    thr_layout_v = cute.make_layout((16, 4), stride=(4, 1))
    val_layout_v = cute.make_layout((1, 8), stride=(8, 1))
    tiled_copy_v = cute.make_tiled_copy_tv(cp_atom, thr_layout_v, val_layout_v)

    KQ_STRIDE = K_DIM + 8
    A_STRIDE = BT + 8
    V_STRIDE = BV_TILE + 8
    sQ_l = cute.make_layout((BT, K_DIM), stride=(KQ_STRIDE, 1))
    sH_l = cute.make_layout((BV_TILE, K_DIM), stride=(KQ_STRIDE, 1))
    sA_l = cute.make_layout((BT, BT), stride=(A_STRIDE, 1))
    sV_l = cute.make_layout((BV_TILE, BT), stride=(A_STRIDE, 1))
    sExpGC_l = cute.make_layout((BT,), stride=(1,))

    smem = (
        BT * KQ_STRIDE * 2 +             # sQ
        BV_TILE * KQ_STRIDE * 2 +        # sH
        BT * A_STRIDE * 2 +              # sA
        BV_TILE * A_STRIDE * 2 +         # sV
        BT * 4                           # sExpGC
    )

    chunk_o_split_cpasync_probe_kernel(
        tiled_mma,
        tiled_copy_q,
        tiled_copy_h,
        tiled_copy_a,
        tiled_copy_v,
        sQ_l, sH_l, sA_l, sV_l, sExpGC_l,
        mQnorm, mH, mGQK, mVNew, mExpGC_in, mO,
        scale, T, H, HV, V, NT, H_PER_HV, BV_TILE,
    ).launch(
        grid=(V // BV_TILE, NT, B * HV),
        block=(NUM_THREADS, 1, 1),
        smem=smem,
    )


@cute.jit
def launch_kernel1_aligned_scalar_probe(
    mKnorm, mQnorm, mV, mBeta,
    mM, mGQK, mExpGC_in, mExpDecay_in,
    mO,
    scale: cutlass.Constexpr[float],
    B: cutlass.Int32,
    T: cutlass.Constexpr[int], H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int], V: cutlass.Constexpr[int],
    NT: cutlass.Constexpr[int], H_PER_HV: cutlass.Constexpr[int],
    BV_TILE: cutlass.Constexpr[int], PER_V_TILE: cutlass.Constexpr[int],
):
    mma_op = warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
    tC = cute.make_layout((2, 2, 1))
    perm = (2 * 16, 2 * 8 * 1, 1 * 16)
    tiled_mma = cute.make_tiled_mma(mma_op, tC, permutation_mnk=perm)

    cp_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.BFloat16,
        num_bits_per_copy=128,
    )
    thr_layout_kq = cute.make_layout((32, 4), stride=(4, 1))
    val_layout_kq = cute.make_layout((1, 8), stride=(8, 1))
    tiled_copy_kq = cute.make_tiled_copy_tv(cp_atom, thr_layout_kq, val_layout_kq)

    KQ_STRIDE = K_DIM + 8
    A_STRIDE = BT + 8
    sK_l = cute.make_layout((BT, K_DIM), stride=(KQ_STRIDE, 1))
    sV_l = cute.make_layout((BT, BV_TILE), stride=(BV_TILE + 2, 1))
    sA_l = cute.make_layout((BT, BT), stride=(A_STRIDE, 1))
    sGQK_l = cute.make_layout((BT, BT), stride=(A_STRIDE, 1))
    sBeta_l = cute.make_layout((BT,), stride=(1,))
    sS_l = cute.make_layout((BV_TILE, K_DIM), stride=(K_DIM + 2, 1))
    sNK_l = cute.make_layout((BV_TILE, BT), stride=(BT + 2, 1))
    sExpGC_l = cute.make_layout((BT,), stride=(1,))
    sExpDecay_l = cute.make_layout((BT,), stride=(1,))

    smem = (
        BT * KQ_STRIDE * 2 * 2 +       # sK + sQ
        BT * (BV_TILE + 2) * 2 +        # sV
        BT * A_STRIDE * 2 +             # sA with V15 alignment-only stride
        BT * A_STRIDE * 2 +             # sGQK with V15 alignment-only stride
        BT * 4 +                         # sBeta
        BV_TILE * (K_DIM + 2) * 2 +     # sS
        BV_TILE * (BT + 2) * 2 +        # sNK_A
        BT * 4 +                         # sExpGC
        BT * 4                           # sExpDecay
    )

    fused_chunk_h_kernel(
        tiled_mma,
        tiled_copy_kq,
        sK_l, sV_l, sA_l, sGQK_l,
        sBeta_l,
        sS_l, sNK_l,
        sExpGC_l, sExpDecay_l,
        mKnorm, mQnorm, mV, mBeta,
        mM, mGQK, mExpGC_in, mExpDecay_in,
        mO,
        scale, T, H, HV, V, NT, H_PER_HV, BV_TILE, PER_V_TILE,
    ).launch(
        grid=(V // BV_TILE, B * HV, 1),
        block=(NUM_THREADS, 1, 1),
        smem=smem,
    )


@cute.jit
def launch_kernel1_m_cpasync_probe(
    mKnorm, mQnorm, mV, mBeta,
    mM, mGQK, mExpGC_in, mExpDecay_in,
    mO,
    scale: cutlass.Constexpr[float],
    B: cutlass.Int32,
    T: cutlass.Constexpr[int], H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int], V: cutlass.Constexpr[int],
    NT: cutlass.Constexpr[int], H_PER_HV: cutlass.Constexpr[int],
    BV_TILE: cutlass.Constexpr[int], PER_V_TILE: cutlass.Constexpr[int],
):
    mma_op = warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
    tC = cute.make_layout((2, 2, 1))
    perm = (2 * 16, 2 * 8 * 1, 1 * 16)
    tiled_mma = cute.make_tiled_mma(mma_op, tC, permutation_mnk=perm)

    cp_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.BFloat16,
        num_bits_per_copy=128,
    )
    thr_layout_kq = cute.make_layout((32, 4), stride=(4, 1))
    val_layout_kq = cute.make_layout((1, 8), stride=(8, 1))
    tiled_copy_kq = cute.make_tiled_copy_tv(cp_atom, thr_layout_kq, val_layout_kq)

    thr_layout_m = cute.make_layout((32, 4), stride=(4, 1))
    val_layout_m = cute.make_layout((1, 8), stride=(8, 1))
    tiled_copy_m = cute.make_tiled_copy_tv(cp_atom, thr_layout_m, val_layout_m)

    KQ_STRIDE = K_DIM + 8
    A_STRIDE = BT + 8
    sK_l = cute.make_layout((BT, K_DIM), stride=(KQ_STRIDE, 1))
    sV_l = cute.make_layout((BT, BV_TILE), stride=(BV_TILE + 2, 1))
    sA_l = cute.make_layout((BT, BT), stride=(A_STRIDE, 1))
    sGQK_l = cute.make_layout((BT, BT), stride=(BT + 2, 1))
    sBeta_l = cute.make_layout((BT,), stride=(1,))
    sS_l = cute.make_layout((BV_TILE, K_DIM), stride=(K_DIM + 2, 1))
    sNK_l = cute.make_layout((BV_TILE, BT), stride=(BT + 2, 1))
    sExpGC_l = cute.make_layout((BT,), stride=(1,))
    sExpDecay_l = cute.make_layout((BT,), stride=(1,))

    smem = (
        BT * KQ_STRIDE * 2 * 2 +       # sK + sQ
        BT * (BV_TILE + 2) * 2 +        # sV
        BT * A_STRIDE * 2 +             # sA, 16B-aligned rows for cp.async
        BT * (BT + 2) * 2 +             # sGQK, original scalar-load stride
        BT * 4 +                         # sBeta
        BV_TILE * (K_DIM + 2) * 2 +     # sS
        BV_TILE * (BT + 2) * 2 +        # sNK_A
        BT * 4 +                         # sExpGC
        BT * 4                           # sExpDecay
    )

    fused_chunk_h_m_cpasync_probe_kernel(
        tiled_mma,
        tiled_copy_kq,
        tiled_copy_m,
        sK_l, sV_l, sA_l, sGQK_l,
        sBeta_l,
        sS_l, sNK_l,
        sExpGC_l, sExpDecay_l,
        mKnorm, mQnorm, mV, mBeta,
        mM, mGQK, mExpGC_in, mExpDecay_in,
        mO,
        scale, T, H, HV, V, NT, H_PER_HV, BV_TILE, PER_V_TILE,
    ).launch(
        grid=(V // BV_TILE, B * HV, 1),
        block=(NUM_THREADS, 1, 1),
        smem=smem,
    )


@cute.jit
def launch_kernel1_gqk_cpasync_probe(
    mKnorm, mQnorm, mV, mBeta,
    mM, mGQK, mExpGC_in, mExpDecay_in,
    mO,
    scale: cutlass.Constexpr[float],
    B: cutlass.Int32,
    T: cutlass.Constexpr[int], H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int], V: cutlass.Constexpr[int],
    NT: cutlass.Constexpr[int], H_PER_HV: cutlass.Constexpr[int],
    BV_TILE: cutlass.Constexpr[int], PER_V_TILE: cutlass.Constexpr[int],
):
    mma_op = warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
    tC = cute.make_layout((2, 2, 1))
    perm = (2 * 16, 2 * 8 * 1, 1 * 16)
    tiled_mma = cute.make_tiled_mma(mma_op, tC, permutation_mnk=perm)

    cp_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.BFloat16,
        num_bits_per_copy=128,
    )
    thr_layout_kq = cute.make_layout((32, 4), stride=(4, 1))
    val_layout_kq = cute.make_layout((1, 8), stride=(8, 1))
    tiled_copy_kq = cute.make_tiled_copy_tv(cp_atom, thr_layout_kq, val_layout_kq)

    thr_layout_gqk = cute.make_layout((32, 4), stride=(4, 1))
    val_layout_gqk = cute.make_layout((1, 8), stride=(8, 1))
    tiled_copy_gqk = cute.make_tiled_copy_tv(cp_atom, thr_layout_gqk, val_layout_gqk)

    KQ_STRIDE = K_DIM + 8
    GQK_STRIDE = BT + 8
    sK_l = cute.make_layout((BT, K_DIM), stride=(KQ_STRIDE, 1))
    sV_l = cute.make_layout((BT, BV_TILE), stride=(BV_TILE + 2, 1))
    sA_l = cute.make_layout((BT, BT), stride=(BT + 2, 1))
    sGQK_l = cute.make_layout((BT, BT), stride=(GQK_STRIDE, 1))
    sBeta_l = cute.make_layout((BT,), stride=(1,))
    sS_l = cute.make_layout((BV_TILE, K_DIM), stride=(K_DIM + 2, 1))
    sNK_l = cute.make_layout((BV_TILE, BT), stride=(BT + 2, 1))
    sExpGC_l = cute.make_layout((BT,), stride=(1,))
    sExpDecay_l = cute.make_layout((BT,), stride=(1,))

    smem = (
        BT * KQ_STRIDE * 2 * 2 +       # sK + sQ
        BT * (BV_TILE + 2) * 2 +        # sV
        BT * (BT + 2) * 2 +             # sA, original scalar-load stride
        BT * GQK_STRIDE * 2 +           # sGQK, 16B-aligned rows for cp.async
        BT * 4 +                         # sBeta
        BV_TILE * (K_DIM + 2) * 2 +     # sS
        BV_TILE * (BT + 2) * 2 +        # sNK_A
        BT * 4 +                         # sExpGC
        BT * 4                           # sExpDecay
    )

    fused_chunk_h_gqk_cpasync_probe_kernel(
        tiled_mma,
        tiled_copy_kq,
        tiled_copy_gqk,
        sK_l, sV_l, sA_l, sGQK_l,
        sBeta_l,
        sS_l, sNK_l,
        sExpGC_l, sExpDecay_l,
        mKnorm, mQnorm, mV, mBeta,
        mM, mGQK, mExpGC_in, mExpDecay_in,
        mO,
        scale, T, H, HV, V, NT, H_PER_HV, BV_TILE, PER_V_TILE,
    ).launch(
        grid=(V // BV_TILE, B * HV, 1),
        block=(NUM_THREADS, 1, 1),
        smem=smem,
    )


@cute.jit
def launch_kernel1_bv8_probe(
    mKnorm, mQnorm, mV, mBeta,
    mM, mGQK, mExpGC_in, mExpDecay_in,
    mO,
    scale: cutlass.Constexpr[float],
    B: cutlass.Int32,
    T: cutlass.Constexpr[int], H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int], V: cutlass.Constexpr[int],
    NT: cutlass.Constexpr[int], H_PER_HV: cutlass.Constexpr[int],
    BV_TILE: cutlass.Constexpr[int], PER_V_TILE: cutlass.Constexpr[int],
):
    mma_op = warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
    tC = cute.make_layout((2, 1, 1))
    perm = (2 * 16, 1 * 8 * 1, 1 * 16)
    tiled_mma = cute.make_tiled_mma(mma_op, tC, permutation_mnk=perm)

    cp_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.BFloat16,
        num_bits_per_copy=128,
    )
    thr_layout_kq = cute.make_layout((32, 4), stride=(4, 1))
    val_layout_kq = cute.make_layout((1, 8), stride=(8, 1))
    tiled_copy_kq = cute.make_tiled_copy_tv(cp_atom, thr_layout_kq, val_layout_kq)

    KQ_STRIDE = K_DIM + 8
    sK_l = cute.make_layout((BT, K_DIM), stride=(KQ_STRIDE, 1))
    sV_l = cute.make_layout((BT, BV_TILE), stride=(BV_TILE + 2, 1))
    sA_l = cute.make_layout((BT, BT), stride=(BT + 2, 1))
    sGQK_l = cute.make_layout((BT, BT), stride=(BT + 2, 1))
    sBeta_l = cute.make_layout((BT,), stride=(1,))
    sS_l = cute.make_layout((BV_TILE, K_DIM), stride=(K_DIM + 2, 1))
    sNK_l = cute.make_layout((BV_TILE, BT), stride=(BT + 2, 1))
    sExpGC_l = cute.make_layout((BT,), stride=(1,))
    sExpDecay_l = cute.make_layout((BT,), stride=(1,))

    smem = (
        BT * KQ_STRIDE * 2 * 2 +       # sK + sQ (same layout, 16B-aligned rows)
        BT * (BV_TILE + 2) * 2 +        # sV
        BT * (BT + 2) * 2 +             # sA
        BT * (BT + 2) * 2 +             # sGQK
        BT * 4 +                         # sBeta
        BV_TILE * (K_DIM + 2) * 2 +     # sS
        BV_TILE * (BT + 2) * 2 +        # sNK_A
        BT * 4 +                         # sExpGC
        BT * 4                           # sExpDecay
    )

    fused_chunk_h_kernel(
        tiled_mma,
        tiled_copy_kq,
        sK_l, sV_l, sA_l, sGQK_l,
        sBeta_l,
        sS_l, sNK_l,
        sExpGC_l, sExpDecay_l,
        mKnorm, mQnorm, mV, mBeta,
        mM, mGQK, mExpGC_in, mExpDecay_in,
        mO,
        scale, T, H, HV, V, NT, H_PER_HV, BV_TILE, PER_V_TILE,
    ).launch(
        grid=(V // BV_TILE, B * HV, 1),
        block=(NUM_THREADS, 1, 1),
        smem=smem,
    )


@cute.jit
def launch_kernel1_pairv_reuse_probe(
    mKnorm, mQnorm, mV, mBeta,
    mM, mGQK, mExpGC_in, mExpDecay_in,
    mO,
    scale: cutlass.Constexpr[float],
    B: cutlass.Int32,
    T: cutlass.Constexpr[int], H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int], V: cutlass.Constexpr[int],
    NT: cutlass.Constexpr[int], H_PER_HV: cutlass.Constexpr[int],
    BV_TILE: cutlass.Constexpr[int], PER_V_TILE: cutlass.Constexpr[int],
):
    mma_op = warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
    tC = cute.make_layout((2, 2, 1))
    perm = (2 * 16, 2 * 8 * 1, 1 * 16)
    tiled_mma = cute.make_tiled_mma(mma_op, tC, permutation_mnk=perm)

    cp_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.BFloat16,
        num_bits_per_copy=128,
    )
    thr_layout_kq = cute.make_layout((32, 4), stride=(4, 1))
    val_layout_kq = cute.make_layout((1, 8), stride=(8, 1))
    tiled_copy_kq = cute.make_tiled_copy_tv(cp_atom, thr_layout_kq, val_layout_kq)

    KQ_STRIDE = K_DIM + 8
    sK_l = cute.make_layout((BT, K_DIM), stride=(KQ_STRIDE, 1))
    sV_l = cute.make_layout((BT, BV_TILE), stride=(BV_TILE + 2, 1))
    sA_l = cute.make_layout((BT, BT), stride=(BT + 2, 1))
    sGQK_l = cute.make_layout((BT, BT), stride=(BT + 2, 1))
    sBeta_l = cute.make_layout((BT,), stride=(1,))
    sS_l = cute.make_layout((BV_TILE, K_DIM), stride=(K_DIM + 2, 1))
    sNK_l = cute.make_layout((BV_TILE, BT), stride=(BT + 2, 1))
    sExpGC_l = cute.make_layout((BT,), stride=(1,))
    sExpDecay_l = cute.make_layout((BT,), stride=(1,))

    smem = (
        BT * KQ_STRIDE * 2 * 2 +       # sK + sQ (same layout, 16B-aligned rows)
        BT * (BV_TILE + 2) * 2 +        # sV
        BT * (BT + 2) * 2 +             # sA
        BT * (BT + 2) * 2 +             # sGQK
        BT * 4 +                         # sBeta
        BV_TILE * (K_DIM + 2) * 2 +     # sS
        BV_TILE * (BT + 2) * 2 +        # sNK_A
        BT * 4 +                         # sExpGC
        BT * 4                           # sExpDecay
    )

    fused_chunk_h_pairv_reuse_probe_kernel(
        tiled_mma,
        tiled_copy_kq,
        sK_l, sV_l, sA_l, sGQK_l,
        sBeta_l,
        sS_l, sNK_l,
        sExpGC_l, sExpDecay_l,
        mKnorm, mQnorm, mV, mBeta,
        mM, mGQK, mExpGC_in, mExpDecay_in,
        mO,
        scale, T, H, HV, V, NT, H_PER_HV, BV_TILE, PER_V_TILE,
    ).launch(
        grid=(V // (BV_TILE * 2), B * HV, 1),
        block=(NUM_THREADS, 1, 1),
        smem=smem,
    )


@cute.jit
def launch_kernel1_bf16_state_probe(
    mKnorm, mQnorm, mV, mBeta,
    mM, mGQK, mExpGC_in, mExpDecay_in,
    mO,
    scale: cutlass.Constexpr[float],
    B: cutlass.Int32,
    T: cutlass.Constexpr[int], H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int], V: cutlass.Constexpr[int],
    NT: cutlass.Constexpr[int], H_PER_HV: cutlass.Constexpr[int],
    BV_TILE: cutlass.Constexpr[int], PER_V_TILE: cutlass.Constexpr[int],
):
    mma_op = warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
    tC = cute.make_layout((2, 2, 1))
    perm = (2 * 16, 2 * 8 * 1, 1 * 16)
    tiled_mma = cute.make_tiled_mma(mma_op, tC, permutation_mnk=perm)

    cp_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.BFloat16,
        num_bits_per_copy=128,
    )
    thr_layout_kq = cute.make_layout((32, 4), stride=(4, 1))
    val_layout_kq = cute.make_layout((1, 8), stride=(8, 1))
    tiled_copy_kq = cute.make_tiled_copy_tv(cp_atom, thr_layout_kq, val_layout_kq)

    KQ_STRIDE = K_DIM + 8
    sK_l = cute.make_layout((BT, K_DIM), stride=(KQ_STRIDE, 1))
    sV_l = cute.make_layout((BT, BV_TILE), stride=(BV_TILE + 2, 1))
    sA_l = cute.make_layout((BT, BT), stride=(BT + 2, 1))
    sGQK_l = cute.make_layout((BT, BT), stride=(BT + 2, 1))
    sBeta_l = cute.make_layout((BT,), stride=(1,))
    sS_l = cute.make_layout((BV_TILE, K_DIM), stride=(K_DIM + 2, 1))
    sNK_l = cute.make_layout((BV_TILE, BT), stride=(BT + 2, 1))
    sExpGC_l = cute.make_layout((BT,), stride=(1,))
    sExpDecay_l = cute.make_layout((BT,), stride=(1,))

    smem = (
        BT * KQ_STRIDE * 2 * 2 +       # sK + sQ (same layout, 16B-aligned rows)
        BT * (BV_TILE + 2) * 2 +        # sV
        BT * (BT + 2) * 2 +             # sA
        BT * (BT + 2) * 2 +             # sGQK
        BT * 4 +                         # sBeta
        BV_TILE * (K_DIM + 2) * 2 +     # sS
        BV_TILE * (BT + 2) * 2 +        # sNK_A
        BT * 4 +                         # sExpGC
        BT * 4                           # sExpDecay
    )

    fused_chunk_h_bf16_state_probe_kernel(
        tiled_mma,
        tiled_copy_kq,
        sK_l, sV_l, sA_l, sGQK_l,
        sBeta_l,
        sS_l, sNK_l,
        sExpGC_l, sExpDecay_l,
        mKnorm, mQnorm, mV, mBeta,
        mM, mGQK, mExpGC_in, mExpDecay_in,
        mO,
        scale, T, H, HV, V, NT, H_PER_HV, BV_TILE, PER_V_TILE,
    ).launch(
        grid=(V // BV_TILE, B * HV, 1),
        block=(NUM_THREADS, 1, 1),
        smem=smem,
    )


@cute.jit
def launch_kernel1_state_mma_probe(
    mKnorm, mQnorm, mV, mBeta,
    mM, mGQK, mExpGC_in, mExpDecay_in,
    mO,
    scale: cutlass.Constexpr[float],
    B: cutlass.Int32,
    T: cutlass.Constexpr[int], H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int], V: cutlass.Constexpr[int],
    NT: cutlass.Constexpr[int], H_PER_HV: cutlass.Constexpr[int],
    BV_TILE: cutlass.Constexpr[int], PER_V_TILE: cutlass.Constexpr[int],
    PER_STATE_TILE: cutlass.Constexpr[int],
):
    mma_op = warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
    tC = cute.make_layout((2, 2, 1))
    perm = (2 * 16, 2 * 8 * 1, 1 * 16)
    tiled_mma = cute.make_tiled_mma(mma_op, tC, permutation_mnk=perm)

    cp_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.BFloat16,
        num_bits_per_copy=128,
    )
    thr_layout_kq = cute.make_layout((32, 4), stride=(4, 1))
    val_layout_kq = cute.make_layout((1, 8), stride=(8, 1))
    tiled_copy_kq = cute.make_tiled_copy_tv(cp_atom, thr_layout_kq, val_layout_kq)

    KQ_STRIDE = K_DIM + 8
    sK_l = cute.make_layout((BT, K_DIM), stride=(KQ_STRIDE, 1))
    sV_l = cute.make_layout((BT, BV_TILE), stride=(BV_TILE + 2, 1))
    sA_l = cute.make_layout((BT, BT), stride=(BT + 2, 1))
    sGQK_l = cute.make_layout((BT, BT), stride=(BT + 2, 1))
    sBeta_l = cute.make_layout((BT,), stride=(1,))
    sS_l = cute.make_layout((BV_TILE, K_DIM), stride=(K_DIM + 2, 1))
    sNK_l = cute.make_layout((BV_TILE, BT), stride=(BT + 2, 1))
    sState_l = cute.make_layout((K_DIM, BV_TILE), stride=(BV_TILE + 2, 1))
    sExpGC_l = cute.make_layout((BT,), stride=(1,))
    sExpDecay_l = cute.make_layout((BT,), stride=(1,))

    smem = (
        BT * KQ_STRIDE * 2 * 2 +       # sK + sQ
        BT * (BV_TILE + 2) * 2 +        # sV
        BT * (BT + 2) * 2 +             # sA / state-update A tile
        BT * (BT + 2) * 2 +             # sGQK
        BT * 4 +                         # sBeta
        BV_TILE * (K_DIM + 2) * 2 +     # sS
        BV_TILE * (BT + 2) * 2 +        # sNK_A
        K_DIM * (BV_TILE + 2) * 4 +     # sState fp32
        BT * 4 +                         # sExpGC
        BT * 4                           # sExpDecay
    )

    fused_chunk_h_state_mma_probe_kernel(
        tiled_mma,
        tiled_copy_kq,
        sK_l, sV_l, sA_l, sGQK_l,
        sBeta_l,
        sS_l, sNK_l,
        sState_l,
        sExpGC_l, sExpDecay_l,
        mKnorm, mQnorm, mV, mBeta,
        mM, mGQK, mExpGC_in, mExpDecay_in,
        mO,
        scale, T, H, HV, V, NT, H_PER_HV,
        BV_TILE, PER_V_TILE, PER_STATE_TILE,
    ).launch(
        grid=(V // BV_TILE, B * HV, 1),
        block=(NUM_THREADS, 1, 1),
        smem=smem,
    )


@cute.jit
def launch_kernel1_mega(
    mKK, mKnorm, mQnorm, mV, mGate, mBeta,
    mO,
    scale: cutlass.Constexpr[float],
    B: cutlass.Int32,
    T: cutlass.Constexpr[int], H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int], V: cutlass.Constexpr[int],
    NT: cutlass.Constexpr[int], H_PER_HV: cutlass.Constexpr[int],
    BV_TILE: cutlass.Constexpr[int], PER_V_TILE: cutlass.Constexpr[int],
):
    mma_op = warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
    tC = cute.make_layout((2, 2, 1))
    perm = (2 * 16, 2 * 8 * 1, 1 * 16)
    tiled_mma = cute.make_tiled_mma(mma_op, tC, permutation_mnk=perm)

    cp_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.BFloat16,
        num_bits_per_copy=128,
    )
    thr_layout_kq = cute.make_layout((32, 4), stride=(4, 1))
    val_layout_kq = cute.make_layout((1, 8), stride=(8, 1))
    tiled_copy_kq = cute.make_tiled_copy_tv(cp_atom, thr_layout_kq, val_layout_kq)

    KQ_STRIDE = K_DIM + 8
    sK_l = cute.make_layout((BT, K_DIM), stride=(KQ_STRIDE, 1))
    sV_l = cute.make_layout((BT, BV_TILE), stride=(BV_TILE + 2, 1))
    sA_l = cute.make_layout((BT, BT), stride=(BT + 2, 1))
    sA_T_l = cute.make_layout((BT, BT), stride=(BT + 2, 1))
    sBeta_l = cute.make_layout((BT,), stride=(1,))
    sS_l = cute.make_layout((BV_TILE, K_DIM), stride=(K_DIM + 2, 1))
    sNK_l = cute.make_layout((BV_TILE, BT), stride=(BT + 2, 1))
    sExpGC_l = cute.make_layout((BT,), stride=(1,))
    sExpDecay_l = cute.make_layout((BT,), stride=(1,))

    smem = (
        BT * KQ_STRIDE * 2 * 2 +       # sK + sQ
        BT * (BV_TILE + 2) * 2 +        # sV
        BT * (BT + 2) * 2 +             # sA
        BT * (BT + 2) * 2 +             # sA_T
        BT * 4 +                         # sBeta
        BV_TILE * (K_DIM + 2) * 2 +     # sS
        BV_TILE * (BT + 2) * 2 +        # sNK_A
        BT * 4 +                         # sExpGC
        BT * 4                           # sExpDecay
    )

    fused_chunk_h_megakernel(
        tiled_mma,
        tiled_copy_kq,
        sK_l, sV_l, sA_l, sA_T_l,
        sBeta_l,
        sS_l, sNK_l,
        sExpGC_l, sExpDecay_l,
        mKK, mKnorm, mQnorm, mV, mGate, mBeta,
        mO,
        scale, T, H, HV, V, NT, H_PER_HV, BV_TILE, PER_V_TILE,
    ).launch(
        grid=(V // BV_TILE, B * HV, 1),
        block=(NUM_THREADS, 1, 1),
        smem=smem,
    )


@cute.jit
def launch_kernel1_mega_tma(
    mKK, mKnorm, mQnorm, mV, mGate, mBeta,
    mO,
    scale: cutlass.Constexpr[float],
    B: cutlass.Int32,
    T: cutlass.Constexpr[int], H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int], V: cutlass.Constexpr[int],
    NT: cutlass.Constexpr[int], H_PER_HV: cutlass.Constexpr[int],
    BV_TILE: cutlass.Constexpr[int], PER_V_TILE: cutlass.Constexpr[int],
):
    mma_op = warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
    tC = cute.make_layout((2, 2, 1))
    perm = (2 * 16, 2 * 8 * 1, 1 * 16)
    tiled_mma = cute.make_tiled_mma(mma_op, tC, permutation_mnk=perm)

    KQ_STRIDE = K_DIM + 8
    sK_l = cute.make_layout((BT, K_DIM), stride=(KQ_STRIDE, 1))
    sK_tma_l = cute.make_layout(
        (BT, K_DIM, 1),
        stride=(KQ_STRIDE, 1, BT * KQ_STRIDE),
    )
    sV_l = cute.make_layout((BT, BV_TILE), stride=(BV_TILE + 2, 1))
    sA_l = cute.make_layout((BT, BT), stride=(BT + 2, 1))
    sA_T_l = cute.make_layout((BT, BT), stride=(BT + 2, 1))
    sBeta_l = cute.make_layout((BT,), stride=(1,))
    sS_l = cute.make_layout((BV_TILE, K_DIM), stride=(K_DIM + 2, 1))
    sNK_l = cute.make_layout((BV_TILE, BT), stride=(BT + 2, 1))
    sExpGC_l = cute.make_layout((BT,), stride=(1,))
    sExpDecay_l = cute.make_layout((BT,), stride=(1,))

    rows = B * cutlass.Int32(H * T)
    gK_tma = cute.make_tensor(
        mKnorm.iterator,
        cute.make_layout((rows, K_DIM), stride=(K_DIM, 1)),
    )
    gQ_tma = cute.make_tensor(
        mQnorm.iterator,
        cute.make_layout((rows, K_DIM), stride=(K_DIM, 1)),
    )
    tma_atom_k, tma_tensor_k = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        gK_tma,
        sK_l,
        (BT, K_DIM),
    )
    tma_atom_q, tma_tensor_q = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        gQ_tma,
        sK_l,
        (BT, K_DIM),
    )

    smem = (
        2 * 8 +                            # one full + one empty mbarrier
        BT * KQ_STRIDE * 2 * 2 +          # sK + sQ
        BT * (BV_TILE + 2) * 2 +          # sV
        BT * (BT + 2) * 2 +               # sA
        BT * (BT + 2) * 2 +               # sA_T
        BT * 4 +                          # sBeta
        BV_TILE * (K_DIM + 2) * 2 +       # sS
        BV_TILE * (BT + 2) * 2 +          # sNK_A
        BT * 4 +                          # sExpGC
        BT * 4 +                          # sExpDecay
        8192                              # allocator alignment headroom
    )
    tma_tx_bytes = BT * K_DIM * 2 * 2

    fused_chunk_h_megakernel_tma(
        tiled_mma,
        tma_atom_k, tma_tensor_k,
        tma_atom_q, tma_tensor_q,
        sK_l, sK_tma_l,
        sV_l, sA_l, sA_T_l,
        sBeta_l,
        sS_l, sNK_l,
        sExpGC_l, sExpDecay_l,
        mKK, mV, mGate, mBeta,
        mO,
        scale, T, H, HV, V, NT, H_PER_HV, BV_TILE, PER_V_TILE,
        tma_tx_bytes,
    ).launch(
        grid=(V // BV_TILE, B * HV, 1),
        block=(NUM_THREADS, 1, 1),
        smem=smem,
        cluster=(1, 1, 1),
    )


# ============================================================================
# Python entry point
# ============================================================================

_compiled_cache: dict = {}


def _run_3kernel_tiled(q, k, v, g, beta, scale, bv_tile: int):
    B, T, H, K = q.shape
    _, _, HV, V = v.shape
    NT = T // BT
    H_PER_HV = HV // H
    if V % bv_tile != 0:
        raise ValueError(f"V={V} must be divisible by bv_tile={bv_tile}")
    per_v_tile = (BT * bv_tile) // NUM_THREADS

    qc = q.contiguous()
    kc = k.contiguous()
    vc = v.contiguous()
    gc_ = g.contiguous()
    betac = beta.contiguous()

    q_norm = torch.empty(B, H, T, K, dtype=qc.dtype, device=qc.device)
    k_norm = torch.empty(B, H, T, K, dtype=kc.dtype, device=kc.device)
    kk = torch.empty(B, NT, H, BT, BT, dtype=torch.bfloat16, device=q.device)
    qk = torch.empty(B, NT, H, BT, BT, dtype=torch.bfloat16, device=q.device)
    neumann_m = torch.empty(B, NT, HV, BT, BT, dtype=torch.bfloat16, device=q.device)
    gated_qk = torch.empty(B, NT, HV, BT, BT, dtype=torch.bfloat16, device=q.device)
    exp_gc = torch.empty(B, NT, HV, BT, dtype=torch.float32, device=q.device)
    exp_decay = torch.empty(B, NT, HV, BT, dtype=torch.float32, device=q.device)
    o = torch.empty(B, T, HV, V, dtype=v.dtype, device=v.device)

    mQ = from_dlpack(qc, assumed_align=16)
    mK = from_dlpack(kc, assumed_align=16)
    mV = from_dlpack(vc, assumed_align=16)
    mG = from_dlpack(gc_, assumed_align=16)
    mBeta = from_dlpack(betac, assumed_align=16)
    mQnorm = from_dlpack(q_norm, assumed_align=16)
    mKnorm = from_dlpack(k_norm, assumed_align=16)
    mKK = from_dlpack(kk, assumed_align=16)
    mQK = from_dlpack(qk, assumed_align=16)
    mM = from_dlpack(neumann_m, assumed_align=16)
    mGQK = from_dlpack(gated_qk, assumed_align=16)
    mExpGC = from_dlpack(exp_gc, assumed_align=16)
    mExpDecay = from_dlpack(exp_decay, assumed_align=16)
    mO = from_dlpack(o, assumed_align=16)

    key0 = ("pinv-k0-qk-bt32", B, T, H, K, q.dtype)
    compiled0 = _compiled_cache.get(key0)
    if compiled0 is None:
        compiled0 = cute.compile(
            launch_kernel0,
            mQ, mK, mQnorm, mKnorm, mKK, mQK,
            cutlass.Int32(B), T, H, NT,
        )
        _compiled_cache[key0] = compiled0
    compiled0(mQ, mK, mQnorm, mKnorm, mKK, mQK, cutlass.Int32(B))

    key_inv = ("pinv-kinv-gqk-bt32", B, T, H, HV, q.dtype)
    compiled_inv = _compiled_cache.get(key_inv)
    if compiled_inv is None:
        compiled_inv = cute.compile(
            launch_kernel_inv,
            mKK, mQK, mG, mBeta,
            mM, mGQK, mExpGC, mExpDecay,
            cutlass.Int32(B),
            T, H, HV, NT, H_PER_HV,
        )
        _compiled_cache[key_inv] = compiled_inv
    compiled_inv(
        mKK, mQK, mG, mBeta,
        mM, mGQK, mExpGC, mExpDecay,
        cutlass.Int32(B),
    )

    key1 = (
        "pinv-k1-bt32-vtile-gqk-cpasync-v1",
        bv_tile,
        B, T, H, HV, K, V, q.dtype, float(scale),
    )
    compiled1 = _compiled_cache.get(key1)
    if compiled1 is None:
        kernel1 = launch_kernel1_bv8_probe if bv_tile == 8 else launch_kernel1
        compiled1 = cute.compile(
            kernel1,
            mKnorm, mQnorm, mV, mBeta,
            mM, mGQK, mExpGC, mExpDecay,
            mO,
            float(scale), cutlass.Int32(B),
            T, H, HV, V, NT, H_PER_HV, bv_tile, per_v_tile,
        )
        _compiled_cache[key1] = compiled1
    compiled1(
        mKnorm, mQnorm, mV, mBeta,
        mM, mGQK, mExpGC, mExpDecay,
        mO,
        cutlass.Int32(B),
    )

    return o


def run_3kernel_legacy(q, k, v, g, beta, scale):
    return _run_3kernel_tiled(q, k, v, g, beta, scale, BV)


def run_3kernel(q, k, v, g, beta, scale):
    return run_3kernel_mgqk_v_ldsm_state_frag_r2s_vec_o_kdecay_rhs_skipsv_scaledv_probe(
        q, k, v, g, beta, scale
    )


def run_3kernel_bv32(q, k, v, g, beta, scale):
    """Precomputed-inverse CuTeDSL path with K1 widened to BV=32."""
    return _run_3kernel_tiled(q, k, v, g, beta, scale, 32)


def run_3kernel_bv8(q, k, v, g, beta, scale):
    """V13 diagnostic probe with K1 narrowed to BV=8."""
    return _run_3kernel_tiled(q, k, v, g, beta, scale, 8)


def _run_3kernel_mgqk_cpasync_probe_tiled(q, k, v, g, beta, scale, bv_tile: int):
    B, T, H, K = q.shape
    _, _, HV, V = v.shape
    NT = T // BT
    H_PER_HV = HV // H
    if V % bv_tile != 0:
        raise ValueError(f"V={V} must be divisible by bv_tile={bv_tile}")
    per_v_tile = (BT * bv_tile) // NUM_THREADS

    qc = q.contiguous()
    kc = k.contiguous()
    vc = v.contiguous()
    gc_ = g.contiguous()
    betac = beta.contiguous()

    q_norm = torch.empty(B, H, T, K, dtype=qc.dtype, device=qc.device)
    k_norm = torch.empty(B, H, T, K, dtype=kc.dtype, device=kc.device)
    kk = torch.empty(B, NT, H, BT, BT, dtype=torch.bfloat16, device=q.device)
    qk = torch.empty(B, NT, H, BT, BT, dtype=torch.bfloat16, device=q.device)
    neumann_m = torch.empty(B, NT, HV, BT, BT, dtype=torch.bfloat16, device=q.device)
    gated_qk = torch.empty(B, NT, HV, BT, BT, dtype=torch.bfloat16, device=q.device)
    exp_gc = torch.empty(B, NT, HV, BT, dtype=torch.float32, device=q.device)
    exp_decay = torch.empty(B, NT, HV, BT, dtype=torch.float32, device=q.device)
    o = torch.empty(B, T, HV, V, dtype=v.dtype, device=v.device)

    mQ = from_dlpack(qc, assumed_align=16)
    mK = from_dlpack(kc, assumed_align=16)
    mV = from_dlpack(vc, assumed_align=16)
    mG = from_dlpack(gc_, assumed_align=16)
    mBeta = from_dlpack(betac, assumed_align=16)
    mQnorm = from_dlpack(q_norm, assumed_align=16)
    mKnorm = from_dlpack(k_norm, assumed_align=16)
    mKK = from_dlpack(kk, assumed_align=16)
    mQK = from_dlpack(qk, assumed_align=16)
    mM = from_dlpack(neumann_m, assumed_align=16)
    mGQK = from_dlpack(gated_qk, assumed_align=16)
    mExpGC = from_dlpack(exp_gc, assumed_align=16)
    mExpDecay = from_dlpack(exp_decay, assumed_align=16)
    mO = from_dlpack(o, assumed_align=16)

    key0 = ("pinv-k0-qk-bt32", B, T, H, K, q.dtype)
    compiled0 = _compiled_cache.get(key0)
    if compiled0 is None:
        compiled0 = cute.compile(
            launch_kernel0,
            mQ, mK, mQnorm, mKnorm, mKK, mQK,
            cutlass.Int32(B), T, H, NT,
        )
        _compiled_cache[key0] = compiled0
    compiled0(mQ, mK, mQnorm, mKnorm, mKK, mQK, cutlass.Int32(B))

    key_inv = ("pinv-kinv-gqk-bt32", B, T, H, HV, q.dtype)
    compiled_inv = _compiled_cache.get(key_inv)
    if compiled_inv is None:
        compiled_inv = cute.compile(
            launch_kernel_inv,
            mKK, mQK, mG, mBeta,
            mM, mGQK, mExpGC, mExpDecay,
            cutlass.Int32(B),
            T, H, HV, NT, H_PER_HV,
        )
        _compiled_cache[key_inv] = compiled_inv
    compiled_inv(
        mKK, mQK, mG, mBeta,
        mM, mGQK, mExpGC, mExpDecay,
        cutlass.Int32(B),
    )

    key1 = (
        "pinv-k1-bt32-vtile-mgqk-cpasync-probe-v1",
        bv_tile,
        B, T, H, HV, K, V, q.dtype, float(scale),
    )
    compiled1 = _compiled_cache.get(key1)
    if compiled1 is None:
        compiled1 = cute.compile(
            launch_kernel1_mgqk_cpasync_probe,
            mKnorm, mQnorm, mV, mBeta,
            mM, mGQK, mExpGC, mExpDecay,
            mO,
            float(scale), cutlass.Int32(B),
            T, H, HV, V, NT, H_PER_HV, bv_tile, per_v_tile,
        )
        _compiled_cache[key1] = compiled1
    compiled1(
        mKnorm, mQnorm, mV, mBeta,
        mM, mGQK, mExpGC, mExpDecay,
        mO,
        cutlass.Int32(B),
    )

    return o


def run_3kernel_mgqk_cpasync_probe(q, k, v, g, beta, scale):
    """V14 opt-in probe that stages M/GQK with cp.async in K1."""
    return _run_3kernel_mgqk_cpasync_probe_tiled(q, k, v, g, beta, scale, BV)


def _run_3kernel_mgqk_v_cpasync_probe_tiled(q, k, v, g, beta, scale, bv_tile: int):
    B, T, H, K = q.shape
    _, _, HV, V = v.shape
    NT = T // BT
    H_PER_HV = HV // H
    if V % bv_tile != 0:
        raise ValueError(f"V={V} must be divisible by bv_tile={bv_tile}")
    per_v_tile = (BT * bv_tile) // NUM_THREADS

    qc = q.contiguous()
    kc = k.contiguous()
    vc = v.contiguous()
    gc_ = g.contiguous()
    betac = beta.contiguous()

    q_norm = torch.empty(B, H, T, K, dtype=qc.dtype, device=qc.device)
    k_norm = torch.empty(B, H, T, K, dtype=kc.dtype, device=kc.device)
    kk = torch.empty(B, NT, H, BT, BT, dtype=torch.bfloat16, device=q.device)
    qk = torch.empty(B, NT, H, BT, BT, dtype=torch.bfloat16, device=q.device)
    neumann_m = torch.empty(B, NT, HV, BT, BT, dtype=torch.bfloat16, device=q.device)
    gated_qk = torch.empty(B, NT, HV, BT, BT, dtype=torch.bfloat16, device=q.device)
    exp_gc = torch.empty(B, NT, HV, BT, dtype=torch.float32, device=q.device)
    exp_decay = torch.empty(B, NT, HV, BT, dtype=torch.float32, device=q.device)
    o = torch.empty(B, T, HV, V, dtype=v.dtype, device=v.device)

    mQ = from_dlpack(qc, assumed_align=16)
    mK = from_dlpack(kc, assumed_align=16)
    mV = from_dlpack(vc, assumed_align=16)
    mG = from_dlpack(gc_, assumed_align=16)
    mBeta = from_dlpack(betac, assumed_align=16)
    mQnorm = from_dlpack(q_norm, assumed_align=16)
    mKnorm = from_dlpack(k_norm, assumed_align=16)
    mKK = from_dlpack(kk, assumed_align=16)
    mQK = from_dlpack(qk, assumed_align=16)
    mM = from_dlpack(neumann_m, assumed_align=16)
    mGQK = from_dlpack(gated_qk, assumed_align=16)
    mExpGC = from_dlpack(exp_gc, assumed_align=16)
    mExpDecay = from_dlpack(exp_decay, assumed_align=16)
    mO = from_dlpack(o, assumed_align=16)

    key0 = ("pinv-k0-qk-bt32", B, T, H, K, q.dtype)
    compiled0 = _compiled_cache.get(key0)
    if compiled0 is None:
        compiled0 = cute.compile(
            launch_kernel0,
            mQ, mK, mQnorm, mKnorm, mKK, mQK,
            cutlass.Int32(B), T, H, NT,
        )
        _compiled_cache[key0] = compiled0
    compiled0(mQ, mK, mQnorm, mKnorm, mKK, mQK, cutlass.Int32(B))

    key_inv = ("pinv-kinv-gqk-bt32", B, T, H, HV, q.dtype)
    compiled_inv = _compiled_cache.get(key_inv)
    if compiled_inv is None:
        compiled_inv = cute.compile(
            launch_kernel_inv,
            mKK, mQK, mG, mBeta,
            mM, mGQK, mExpGC, mExpDecay,
            cutlass.Int32(B),
            T, H, HV, NT, H_PER_HV,
        )
        _compiled_cache[key_inv] = compiled_inv
    compiled_inv(
        mKK, mQK, mG, mBeta,
        mM, mGQK, mExpGC, mExpDecay,
        cutlass.Int32(B),
    )

    key1 = (
        "pinv-k1-bt32-vtile-mgqk-v-cpasync-probe-v1",
        bv_tile,
        B, T, H, HV, K, V, q.dtype, float(scale),
    )
    compiled1 = _compiled_cache.get(key1)
    if compiled1 is None:
        compiled1 = cute.compile(
            launch_kernel1_mgqk_v_cpasync_probe,
            mKnorm, mQnorm, mV, mBeta,
            mM, mGQK, mExpGC, mExpDecay,
            mO,
            float(scale), cutlass.Int32(B),
            T, H, HV, V, NT, H_PER_HV, bv_tile, per_v_tile,
        )
        _compiled_cache[key1] = compiled1
    compiled1(
        mKnorm, mQnorm, mV, mBeta,
        mM, mGQK, mExpGC, mExpDecay,
        mO,
        cutlass.Int32(B),
    )

    return o


def run_3kernel_mgqk_v_cpasync_probe(q, k, v, g, beta, scale):
    """V17 opt-in probe that stages M/GQK and the initial V tile with cp.async."""
    return _run_3kernel_mgqk_v_cpasync_probe_tiled(q, k, v, g, beta, scale, BV)


def _run_3kernel_mgqk_v_fp32state_cpasync_probe_tiled(q, k, v, g, beta, scale, bv_tile: int):
    B, T, H, K = q.shape
    _, _, HV, V = v.shape
    NT = T // BT
    H_PER_HV = HV // H
    if V % bv_tile != 0:
        raise ValueError(f"V={V} must be divisible by bv_tile={bv_tile}")
    per_v_tile = (BT * bv_tile) // NUM_THREADS

    qc = q.contiguous()
    kc = k.contiguous()
    vc = v.contiguous()
    gc_ = g.contiguous()
    betac = beta.contiguous()

    q_norm = torch.empty(B, H, T, K, dtype=qc.dtype, device=qc.device)
    k_norm = torch.empty(B, H, T, K, dtype=kc.dtype, device=kc.device)
    kk = torch.empty(B, NT, H, BT, BT, dtype=torch.bfloat16, device=q.device)
    qk = torch.empty(B, NT, H, BT, BT, dtype=torch.bfloat16, device=q.device)
    neumann_m = torch.empty(B, NT, HV, BT, BT, dtype=torch.bfloat16, device=q.device)
    gated_qk = torch.empty(B, NT, HV, BT, BT, dtype=torch.bfloat16, device=q.device)
    exp_gc = torch.empty(B, NT, HV, BT, dtype=torch.float32, device=q.device)
    exp_decay = torch.empty(B, NT, HV, BT, dtype=torch.float32, device=q.device)
    o = torch.empty(B, T, HV, V, dtype=v.dtype, device=v.device)

    mQ = from_dlpack(qc, assumed_align=16)
    mK = from_dlpack(kc, assumed_align=16)
    mV = from_dlpack(vc, assumed_align=16)
    mG = from_dlpack(gc_, assumed_align=16)
    mBeta = from_dlpack(betac, assumed_align=16)
    mQnorm = from_dlpack(q_norm, assumed_align=16)
    mKnorm = from_dlpack(k_norm, assumed_align=16)
    mKK = from_dlpack(kk, assumed_align=16)
    mQK = from_dlpack(qk, assumed_align=16)
    mM = from_dlpack(neumann_m, assumed_align=16)
    mGQK = from_dlpack(gated_qk, assumed_align=16)
    mExpGC = from_dlpack(exp_gc, assumed_align=16)
    mExpDecay = from_dlpack(exp_decay, assumed_align=16)
    mO = from_dlpack(o, assumed_align=16)

    key0 = ("pinv-k0-qk-bt32", B, T, H, K, q.dtype)
    compiled0 = _compiled_cache.get(key0)
    if compiled0 is None:
        compiled0 = cute.compile(
            launch_kernel0,
            mQ, mK, mQnorm, mKnorm, mKK, mQK,
            cutlass.Int32(B), T, H, NT,
        )
        _compiled_cache[key0] = compiled0
    compiled0(mQ, mK, mQnorm, mKnorm, mKK, mQK, cutlass.Int32(B))

    key_inv = ("pinv-kinv-gqk-bt32", B, T, H, HV, q.dtype)
    compiled_inv = _compiled_cache.get(key_inv)
    if compiled_inv is None:
        compiled_inv = cute.compile(
            launch_kernel_inv,
            mKK, mQK, mG, mBeta,
            mM, mGQK, mExpGC, mExpDecay,
            cutlass.Int32(B),
            T, H, HV, NT, H_PER_HV,
        )
        _compiled_cache[key_inv] = compiled_inv
    compiled_inv(
        mKK, mQK, mG, mBeta,
        mM, mGQK, mExpGC, mExpDecay,
        cutlass.Int32(B),
    )

    key1 = (
        "pinv-k1-bt32-vtile-mgqk-v-fp32state-cpasync-probe-v1",
        bv_tile,
        B, T, H, HV, K, V, q.dtype, float(scale),
    )
    compiled1 = _compiled_cache.get(key1)
    if compiled1 is None:
        compiled1 = cute.compile(
            launch_kernel1_mgqk_v_fp32state_cpasync_probe,
            mKnorm, mQnorm, mV, mBeta,
            mM, mGQK, mExpGC, mExpDecay,
            mO,
            float(scale), cutlass.Int32(B),
            T, H, HV, V, NT, H_PER_HV, bv_tile, per_v_tile,
        )
        _compiled_cache[key1] = compiled1
    compiled1(
        mKnorm, mQnorm, mV, mBeta,
        mM, mGQK, mExpGC, mExpDecay,
        mO,
        cutlass.Int32(B),
    )

    return o


def run_3kernel_mgqk_v_fp32state_cpasync_probe(q, k, v, g, beta, scale):
    """V18 opt-in probe: V17 plus fp32 shared v_new for the recurrent update."""
    return _run_3kernel_mgqk_v_fp32state_cpasync_probe_tiled(q, k, v, g, beta, scale, BV)


def _run_3kernel_mgqk_v_ldsm_probe_tiled(
    q, k, v, g, beta, scale, bv_tile: int,
    state_frag: bool = False,
    dual_state_buffer: bool = False,
    dedicated_state_scratch: bool = False,
    state_r2s: bool = False,
    vector_o_gmem: bool = False,
    kdecay_r2s: bool = False,
    rhs_vnew_r2s: bool = False,
    skip_sv_vnew: bool = False,
    direct_v_rhs: bool = False,
    scaled_vnew_state: bool = False,
    autovec_mma: bool = False,
    autovec_state_mma: bool = False,
    autovec_kqs_mma: bool = False,
    autovec_ks_mma: bool = False,
    autovec_qs_mma: bool = False,
    autovec_vnew_mma: bool = False,
    autovec_chunko_mma: bool = False,
    scalar_scaled_vnew_store: bool = False,
    scalar_vnew_store: bool = False,
    scalar_rhs_store: bool = False,
    output_fp32: bool = False,
):
    B, T, H, K = q.shape
    _, _, HV, V = v.shape
    NT = T // BT
    H_PER_HV = HV // H
    if V % bv_tile != 0:
        raise ValueError(f"V={V} must be divisible by bv_tile={bv_tile}")
    per_v_tile = (BT * bv_tile) // NUM_THREADS

    qc = q.contiguous()
    kc = k.contiguous()
    vc = v.contiguous()
    gc_ = g.contiguous()
    betac = beta.contiguous()

    q_norm = torch.empty(B, H, T, K, dtype=qc.dtype, device=qc.device)
    k_norm = torch.empty(B, H, T, K, dtype=kc.dtype, device=kc.device)
    kk = torch.empty(B, NT, H, BT, BT, dtype=torch.bfloat16, device=q.device)
    qk = torch.empty(B, NT, H, BT, BT, dtype=torch.bfloat16, device=q.device)
    neumann_m = torch.empty(B, NT, HV, BT, BT, dtype=torch.bfloat16, device=q.device)
    gated_qk = torch.empty(B, NT, HV, BT, BT, dtype=torch.bfloat16, device=q.device)
    exp_gc = torch.empty(B, NT, HV, BT, dtype=torch.float32, device=q.device)
    exp_decay = torch.empty(B, NT, HV, BT, dtype=torch.float32, device=q.device)
    o_dtype = torch.float32 if output_fp32 else v.dtype
    o = torch.empty(B, T, HV, V, dtype=o_dtype, device=v.device)

    mQ = from_dlpack(qc, assumed_align=16)
    mK = from_dlpack(kc, assumed_align=16)
    mV = from_dlpack(vc, assumed_align=16)
    mG = from_dlpack(gc_, assumed_align=16)
    mBeta = from_dlpack(betac, assumed_align=16)
    mQnorm = from_dlpack(q_norm, assumed_align=16)
    mKnorm = from_dlpack(k_norm, assumed_align=16)
    mKK = from_dlpack(kk, assumed_align=16)
    mQK = from_dlpack(qk, assumed_align=16)
    mM = from_dlpack(neumann_m, assumed_align=16)
    mGQK = from_dlpack(gated_qk, assumed_align=16)
    mExpGC = from_dlpack(exp_gc, assumed_align=16)
    mExpDecay = from_dlpack(exp_decay, assumed_align=16)
    mO = from_dlpack(o, assumed_align=16)

    key0 = ("pinv-k0-qk-bt32", B, T, H, K, q.dtype)
    compiled0 = _compiled_cache.get(key0)
    if compiled0 is None:
        compiled0 = cute.compile(
            launch_kernel0,
            mQ, mK, mQnorm, mKnorm, mKK, mQK,
            cutlass.Int32(B), T, H, NT,
        )
        _compiled_cache[key0] = compiled0
    compiled0(mQ, mK, mQnorm, mKnorm, mKK, mQK, cutlass.Int32(B))

    key_inv = ("pinv-kinv-gqk-bt32", B, T, H, HV, q.dtype)
    compiled_inv = _compiled_cache.get(key_inv)
    if compiled_inv is None:
        compiled_inv = cute.compile(
            launch_kernel_inv,
            mKK, mQK, mG, mBeta,
            mM, mGQK, mExpGC, mExpDecay,
            cutlass.Int32(B),
            T, H, HV, NT, H_PER_HV,
        )
        _compiled_cache[key_inv] = compiled_inv
    compiled_inv(
        mKK, mQK, mG, mBeta,
        mM, mGQK, mExpGC, mExpDecay,
        cutlass.Int32(B),
    )

    key1 = (
        "pinv-k1-bt32-vtile-mgqk-v-ldsm-probe-v2",
        bv_tile,
        B, T, H, HV, K, V, q.dtype, o_dtype, float(scale),
        state_frag, dual_state_buffer, dedicated_state_scratch,
        state_r2s, vector_o_gmem, kdecay_r2s,
        rhs_vnew_r2s, skip_sv_vnew, direct_v_rhs,
        scaled_vnew_state, autovec_mma, autovec_state_mma,
        autovec_kqs_mma, autovec_ks_mma, autovec_qs_mma,
        autovec_vnew_mma, autovec_chunko_mma,
        scalar_scaled_vnew_store, scalar_vnew_store,
        scalar_rhs_store,
    )
    compiled1 = _compiled_cache.get(key1)
    if compiled1 is None:
        compiled1 = cute.compile(
            launch_kernel1_mgqk_v_ldsm_probe,
            mKnorm, mQnorm, mV, mBeta,
            mM, mGQK, mExpGC, mExpDecay,
            mO,
            float(scale), cutlass.Int32(B),
            T, H, HV, V, NT, H_PER_HV, bv_tile, per_v_tile,
            state_frag, dual_state_buffer, dedicated_state_scratch,
            state_r2s, vector_o_gmem, kdecay_r2s,
            rhs_vnew_r2s, skip_sv_vnew, direct_v_rhs,
            scaled_vnew_state, autovec_mma, autovec_state_mma,
            autovec_kqs_mma, autovec_ks_mma, autovec_qs_mma,
            autovec_vnew_mma, autovec_chunko_mma,
            scalar_scaled_vnew_store, scalar_vnew_store,
            scalar_rhs_store,
        )
        _compiled_cache[key1] = compiled1
    compiled1(
        mKnorm, mQnorm, mV, mBeta,
        mM, mGQK, mExpGC, mExpDecay,
        mO,
        cutlass.Int32(B),
    )

    return o


def run_3kernel_mgqk_v_ldsm_probe(q, k, v, g, beta, scale):
    """V19 opt-in probe: V17 plus LDSM-fed MMA shared-to-register copies."""
    return _run_3kernel_mgqk_v_ldsm_probe_tiled(q, k, v, g, beta, scale, BV)


def run_3kernel_mgqk_v_ldsm_state_frag_probe(q, k, v, g, beta, scale):
    """V22 opt-in probe: V19 LDSM path with register-resident state fragments."""
    return _run_3kernel_mgqk_v_ldsm_probe_tiled(
        q, k, v, g, beta, scale, BV, state_frag=True
    )


def run_3kernel_mgqk_v_ldsm_state_frag_vec_o_probe(q, k, v, g, beta, scale):
    """V23 opt-in probe: V22 plus two-at-a-time state updates using sGQK scratch."""
    return _run_3kernel_mgqk_v_ldsm_probe_tiled(
        q, k, v, g, beta, scale, BV, state_frag=True, dual_state_buffer=True
    )


def run_3kernel_mgqk_v_ldsm_state_frag_scratch_probe(q, k, v, g, beta, scale):
    """V24 opt-in probe: V23 scheduling with a dedicated second state scratch tile."""
    return _run_3kernel_mgqk_v_ldsm_probe_tiled(
        q, k, v, g, beta, scale, BV,
        state_frag=True, dedicated_state_scratch=True,
    )


def run_3kernel_mgqk_v_ldsm_state_frag_r2s_probe(q, k, v, g, beta, scale):
    """V25 opt-in probe: V24 plus StMatrix state-fragment spill to sS."""
    return _run_3kernel_mgqk_v_ldsm_probe_tiled(
        q, k, v, g, beta, scale, BV,
        state_frag=True,
        dedicated_state_scratch=True,
        state_r2s=True,
    )


def run_3kernel_mgqk_v_ldsm_state_frag_r2s_vec_o_probe(q, k, v, g, beta, scale):
    """V26 opt-in probe: V25 plus direct vectorized output GMEM store."""
    return _run_3kernel_mgqk_v_ldsm_probe_tiled(
        q, k, v, g, beta, scale, BV,
        state_frag=True,
        dedicated_state_scratch=True,
        state_r2s=True,
        vector_o_gmem=True,
    )


def run_3kernel_mgqk_v_ldsm_state_frag_r2s_vec_o_kdecay_probe(q, k, v, g, beta, scale):
    """V27 opt-in probe: V26 plus R2S K-decay scratch stores."""
    return _run_3kernel_mgqk_v_ldsm_probe_tiled(
        q, k, v, g, beta, scale, BV,
        state_frag=True,
        dedicated_state_scratch=True,
        state_r2s=True,
        vector_o_gmem=True,
        kdecay_r2s=True,
    )


def run_3kernel_mgqk_v_ldsm_state_frag_r2s_vec_o_kdecay_rhs_probe(q, k, v, g, beta, scale):
    """V28 opt-in probe: V27 plus R2S for RHS and v_new shared stores."""
    return _run_3kernel_mgqk_v_ldsm_probe_tiled(
        q, k, v, g, beta, scale, BV,
        state_frag=True,
        dedicated_state_scratch=True,
        state_r2s=True,
        vector_o_gmem=True,
        kdecay_r2s=True,
        rhs_vnew_r2s=True,
    )


def run_3kernel_mgqk_v_ldsm_state_frag_r2s_vec_o_kdecay_rhs_skipsv_probe(q, k, v, g, beta, scale):
    """V29 opt-in probe: V28 but skip the dead v_new -> sV write."""
    return _run_3kernel_mgqk_v_ldsm_probe_tiled(
        q, k, v, g, beta, scale, BV,
        state_frag=True,
        dedicated_state_scratch=True,
        state_r2s=True,
        vector_o_gmem=True,
        kdecay_r2s=True,
        rhs_vnew_r2s=True,
        skip_sv_vnew=True,
    )


def run_3kernel_mgqk_v_ldsm_state_frag_r2s_vec_o_kdecay_rhs_skipsv_directv_probe(q, k, v, g, beta, scale):
    """V30 opt-in probe: V29 plus direct global V loads for RHS."""
    return _run_3kernel_mgqk_v_ldsm_probe_tiled(
        q, k, v, g, beta, scale, BV,
        state_frag=True,
        dedicated_state_scratch=True,
        state_r2s=True,
        vector_o_gmem=True,
        kdecay_r2s=True,
        rhs_vnew_r2s=True,
        skip_sv_vnew=True,
        direct_v_rhs=True,
    )


def run_3kernel_mgqk_v_ldsm_state_frag_r2s_vec_o_kdecay_rhs_skipsv_scaledv_probe(q, k, v, g, beta, scale):
    """Directional-safe V31: keep LDSM for k/state, autovec qS/v_new/chunk-O."""
    return _run_3kernel_mgqk_v_ldsm_probe_tiled(
        q, k, v, g, beta, scale, BV,
        state_frag=True,
        dedicated_state_scratch=True,
        state_r2s=True,
        vector_o_gmem=True,
        kdecay_r2s=True,
        rhs_vnew_r2s=True,
        skip_sv_vnew=True,
        scaled_vnew_state=True,
        autovec_qs_mma=True,
        autovec_vnew_mma=True,
        autovec_chunko_mma=True,
    )


def run_3kernel_mgqk_v_ldsm_state_frag_r2s_vec_o_kdecay_rhs_skipsv_scaledv_scalarstore_probe(q, k, v, g, beta, scale):
    """Directional diagnostic: V31 with scalar scaled-vnew state bridge."""
    return _run_3kernel_mgqk_v_ldsm_probe_tiled(
        q, k, v, g, beta, scale, BV,
        state_frag=True,
        dedicated_state_scratch=True,
        state_r2s=True,
        vector_o_gmem=True,
        kdecay_r2s=True,
        rhs_vnew_r2s=True,
        skip_sv_vnew=True,
        scaled_vnew_state=True,
        scalar_scaled_vnew_store=True,
    )


def run_3kernel_mgqk_v_ldsm_state_frag_r2s_vec_o_kdecay_rhs_skipsv_scaledv_fp32o_probe(q, k, v, g, beta, scale):
    """Directional diagnostic: V31 writing O as fp32."""
    return _run_3kernel_mgqk_v_ldsm_probe_tiled(
        q, k, v, g, beta, scale, BV,
        state_frag=True,
        dedicated_state_scratch=True,
        state_r2s=True,
        vector_o_gmem=True,
        kdecay_r2s=True,
        rhs_vnew_r2s=True,
        skip_sv_vnew=True,
        scaled_vnew_state=True,
        output_fp32=True,
    )


def run_3kernel_mgqk_v_autovec_state_frag_r2s_vec_o_kdecay_rhs_skipsv_scaledv_probe(q, k, v, g, beta, scale):
    """Directional diagnostic: V31 algebra with autovec MMA feeds instead of LDSM."""
    return _run_3kernel_mgqk_v_ldsm_probe_tiled(
        q, k, v, g, beta, scale, BV,
        state_frag=True,
        dedicated_state_scratch=True,
        state_r2s=True,
        vector_o_gmem=True,
        kdecay_r2s=True,
        rhs_vnew_r2s=True,
        skip_sv_vnew=True,
        scaled_vnew_state=True,
        autovec_mma=True,
        autovec_state_mma=True,
    )


def run_3kernel_mgqk_v_autovec_main_state_frag_r2s_vec_o_kdecay_rhs_skipsv_scaledv_probe(q, k, v, g, beta, scale):
    """Directional diagnostic: V31 with autovec main/output MMAs only."""
    return _run_3kernel_mgqk_v_ldsm_probe_tiled(
        q, k, v, g, beta, scale, BV,
        state_frag=True,
        dedicated_state_scratch=True,
        state_r2s=True,
        vector_o_gmem=True,
        kdecay_r2s=True,
        rhs_vnew_r2s=True,
        skip_sv_vnew=True,
        scaled_vnew_state=True,
        autovec_mma=True,
    )


def run_3kernel_mgqk_v_autovec_state_state_frag_r2s_vec_o_kdecay_rhs_skipsv_scaledv_probe(q, k, v, g, beta, scale):
    """Directional diagnostic: V31 with autovec state-update MMAs only."""
    return _run_3kernel_mgqk_v_ldsm_probe_tiled(
        q, k, v, g, beta, scale, BV,
        state_frag=True,
        dedicated_state_scratch=True,
        state_r2s=True,
        vector_o_gmem=True,
        kdecay_r2s=True,
        rhs_vnew_r2s=True,
        skip_sv_vnew=True,
        scaled_vnew_state=True,
        autovec_state_mma=True,
    )


def _run_3kernel_mgqk_v_ldsm_state_mma_probe_tiled(q, k, v, g, beta, scale, bv_tile: int):
    B, T, H, K = q.shape
    _, _, HV, V = v.shape
    NT = T // BT
    H_PER_HV = HV // H
    if V % bv_tile != 0:
        raise ValueError(f"V={V} must be divisible by bv_tile={bv_tile}")
    per_v_tile = (BT * bv_tile) // NUM_THREADS
    per_state_tile = (K_DIM * bv_tile) // NUM_THREADS

    qc = q.contiguous()
    kc = k.contiguous()
    vc = v.contiguous()
    gc_ = g.contiguous()
    betac = beta.contiguous()

    q_norm = torch.empty(B, H, T, K, dtype=qc.dtype, device=qc.device)
    k_norm = torch.empty(B, H, T, K, dtype=kc.dtype, device=kc.device)
    kk = torch.empty(B, NT, H, BT, BT, dtype=torch.bfloat16, device=q.device)
    qk = torch.empty(B, NT, H, BT, BT, dtype=torch.bfloat16, device=q.device)
    neumann_m = torch.empty(B, NT, HV, BT, BT, dtype=torch.bfloat16, device=q.device)
    gated_qk = torch.empty(B, NT, HV, BT, BT, dtype=torch.bfloat16, device=q.device)
    exp_gc = torch.empty(B, NT, HV, BT, dtype=torch.float32, device=q.device)
    exp_decay = torch.empty(B, NT, HV, BT, dtype=torch.float32, device=q.device)
    o = torch.empty(B, T, HV, V, dtype=v.dtype, device=v.device)

    mQ = from_dlpack(qc, assumed_align=16)
    mK = from_dlpack(kc, assumed_align=16)
    mV = from_dlpack(vc, assumed_align=16)
    mG = from_dlpack(gc_, assumed_align=16)
    mBeta = from_dlpack(betac, assumed_align=16)
    mQnorm = from_dlpack(q_norm, assumed_align=16)
    mKnorm = from_dlpack(k_norm, assumed_align=16)
    mKK = from_dlpack(kk, assumed_align=16)
    mQK = from_dlpack(qk, assumed_align=16)
    mM = from_dlpack(neumann_m, assumed_align=16)
    mGQK = from_dlpack(gated_qk, assumed_align=16)
    mExpGC = from_dlpack(exp_gc, assumed_align=16)
    mExpDecay = from_dlpack(exp_decay, assumed_align=16)
    mO = from_dlpack(o, assumed_align=16)

    key0 = ("pinv-k0-qk-bt32", B, T, H, K, q.dtype)
    compiled0 = _compiled_cache.get(key0)
    if compiled0 is None:
        compiled0 = cute.compile(
            launch_kernel0,
            mQ, mK, mQnorm, mKnorm, mKK, mQK,
            cutlass.Int32(B), T, H, NT,
        )
        _compiled_cache[key0] = compiled0
    compiled0(mQ, mK, mQnorm, mKnorm, mKK, mQK, cutlass.Int32(B))

    key_inv = ("pinv-kinv-gqk-bt32", B, T, H, HV, q.dtype)
    compiled_inv = _compiled_cache.get(key_inv)
    if compiled_inv is None:
        compiled_inv = cute.compile(
            launch_kernel_inv,
            mKK, mQK, mG, mBeta,
            mM, mGQK, mExpGC, mExpDecay,
            cutlass.Int32(B),
            T, H, HV, NT, H_PER_HV,
        )
        _compiled_cache[key_inv] = compiled_inv
    compiled_inv(
        mKK, mQK, mG, mBeta,
        mM, mGQK, mExpGC, mExpDecay,
        cutlass.Int32(B),
    )

    key1 = (
        "pinv-k1-bt32-vtile-mgqk-v-ldsm-state-mma-probe-v1",
        bv_tile,
        B, T, H, HV, K, V, q.dtype, float(scale),
    )
    compiled1 = _compiled_cache.get(key1)
    if compiled1 is None:
        compiled1 = cute.compile(
            launch_kernel1_mgqk_v_ldsm_state_mma_probe,
            mKnorm, mQnorm, mV, mBeta,
            mM, mGQK, mExpGC, mExpDecay,
            mO,
            float(scale), cutlass.Int32(B),
            T, H, HV, V, NT, H_PER_HV,
            bv_tile, per_v_tile, per_state_tile,
        )
        _compiled_cache[key1] = compiled1
    compiled1(
        mKnorm, mQnorm, mV, mBeta,
        mM, mGQK, mExpGC, mExpDecay,
        mO,
        cutlass.Int32(B),
    )

    return o


def run_3kernel_mgqk_v_ldsm_state_mma_probe(q, k, v, g, beta, scale):
    """V21 opt-in probe: V19 LDSM path with recurrent state update via HMMA."""
    return _run_3kernel_mgqk_v_ldsm_state_mma_probe_tiled(q, k, v, g, beta, scale, BV)


def _run_4kernel_split_o_cpasync_probe_tiled(q, k, v, g, beta, scale, bv_tile: int):
    B, T, H, K = q.shape
    _, _, HV, V = v.shape
    NT = T // BT
    H_PER_HV = HV // H
    if V % bv_tile != 0:
        raise ValueError(f"V={V} must be divisible by bv_tile={bv_tile}")
    per_v_tile = (BT * bv_tile) // NUM_THREADS

    qc = q.contiguous()
    kc = k.contiguous()
    vc = v.contiguous()
    gc_ = g.contiguous()
    betac = beta.contiguous()

    q_norm = torch.empty(B, H, T, K, dtype=qc.dtype, device=qc.device)
    k_norm = torch.empty(B, H, T, K, dtype=kc.dtype, device=kc.device)
    kk = torch.empty(B, NT, H, BT, BT, dtype=torch.bfloat16, device=q.device)
    qk = torch.empty(B, NT, H, BT, BT, dtype=torch.bfloat16, device=q.device)
    neumann_m = torch.empty(B, NT, HV, BT, BT, dtype=torch.bfloat16, device=q.device)
    gated_qk = torch.empty(B, NT, HV, BT, BT, dtype=torch.bfloat16, device=q.device)
    exp_gc = torch.empty(B, NT, HV, BT, dtype=torch.float32, device=q.device)
    exp_decay = torch.empty(B, NT, HV, BT, dtype=torch.float32, device=q.device)
    h_state = torch.empty(B, NT, HV, V, K, dtype=torch.bfloat16, device=q.device)
    v_new_t = torch.empty(B, NT, HV, V, BT, dtype=torch.bfloat16, device=q.device)
    o = torch.empty(B, T, HV, V, dtype=v.dtype, device=v.device)

    mQ = from_dlpack(qc, assumed_align=16)
    mK = from_dlpack(kc, assumed_align=16)
    mV = from_dlpack(vc, assumed_align=16)
    mG = from_dlpack(gc_, assumed_align=16)
    mBeta = from_dlpack(betac, assumed_align=16)
    mQnorm = from_dlpack(q_norm, assumed_align=16)
    mKnorm = from_dlpack(k_norm, assumed_align=16)
    mKK = from_dlpack(kk, assumed_align=16)
    mQK = from_dlpack(qk, assumed_align=16)
    mM = from_dlpack(neumann_m, assumed_align=16)
    mGQK = from_dlpack(gated_qk, assumed_align=16)
    mExpGC = from_dlpack(exp_gc, assumed_align=16)
    mExpDecay = from_dlpack(exp_decay, assumed_align=16)
    mH = from_dlpack(h_state, assumed_align=16)
    mVNew = from_dlpack(v_new_t, assumed_align=16)
    mO = from_dlpack(o, assumed_align=16)

    key0 = ("pinv-k0-qk-bt32", B, T, H, K, q.dtype)
    compiled0 = _compiled_cache.get(key0)
    if compiled0 is None:
        compiled0 = cute.compile(
            launch_kernel0,
            mQ, mK, mQnorm, mKnorm, mKK, mQK,
            cutlass.Int32(B), T, H, NT,
        )
        _compiled_cache[key0] = compiled0
    compiled0(mQ, mK, mQnorm, mKnorm, mKK, mQK, cutlass.Int32(B))

    key_inv = ("pinv-kinv-gqk-bt32", B, T, H, HV, q.dtype)
    compiled_inv = _compiled_cache.get(key_inv)
    if compiled_inv is None:
        compiled_inv = cute.compile(
            launch_kernel_inv,
            mKK, mQK, mG, mBeta,
            mM, mGQK, mExpGC, mExpDecay,
            cutlass.Int32(B),
            T, H, HV, NT, H_PER_HV,
        )
        _compiled_cache[key_inv] = compiled_inv
    compiled_inv(
        mKK, mQK, mG, mBeta,
        mM, mGQK, mExpGC, mExpDecay,
        cutlass.Int32(B),
    )

    key_h = (
        "pinv-k1-bt32-vtile-split-h-cpasync-probe-v1",
        bv_tile,
        B, T, H, HV, K, V, q.dtype,
    )
    compiled_h = _compiled_cache.get(key_h)
    if compiled_h is None:
        compiled_h = cute.compile(
            launch_kernel1_split_h_cpasync_probe,
            mKnorm, mV, mBeta,
            mM, mExpGC, mExpDecay,
            mH, mVNew,
            cutlass.Int32(B),
            T, H, HV, V, NT, H_PER_HV, bv_tile, per_v_tile,
        )
        _compiled_cache[key_h] = compiled_h
    compiled_h(
        mKnorm, mV, mBeta,
        mM, mExpGC, mExpDecay,
        mH, mVNew,
        cutlass.Int32(B),
    )

    key_o = (
        "pinv-k2-bt32-vtile-split-o-cpasync-probe-v1",
        bv_tile,
        B, T, H, HV, K, V, q.dtype, float(scale),
    )
    compiled_o = _compiled_cache.get(key_o)
    if compiled_o is None:
        compiled_o = cute.compile(
            launch_kernel2_split_o_cpasync_probe,
            mQnorm, mH, mGQK, mVNew, mExpGC, mO,
            float(scale), cutlass.Int32(B),
            T, H, HV, V, NT, H_PER_HV, bv_tile,
        )
        _compiled_cache[key_o] = compiled_o
    compiled_o(
        mQnorm, mH, mGQK, mVNew, mExpGC, mO,
        cutlass.Int32(B),
    )

    return o


def run_4kernel_split_o_cpasync_probe(q, k, v, g, beta, scale):
    """V17 opt-in probe: CuTeDSL FLA-style split H/v_new plus chunk-parallel O."""
    return _run_4kernel_split_o_cpasync_probe_tiled(q, k, v, g, beta, scale, BV)


def _run_3kernel_aligned_scalar_probe_tiled(q, k, v, g, beta, scale, bv_tile: int):
    B, T, H, K = q.shape
    _, _, HV, V = v.shape
    NT = T // BT
    H_PER_HV = HV // H
    if V % bv_tile != 0:
        raise ValueError(f"V={V} must be divisible by bv_tile={bv_tile}")
    per_v_tile = (BT * bv_tile) // NUM_THREADS

    qc = q.contiguous()
    kc = k.contiguous()
    vc = v.contiguous()
    gc_ = g.contiguous()
    betac = beta.contiguous()

    q_norm = torch.empty(B, H, T, K, dtype=qc.dtype, device=qc.device)
    k_norm = torch.empty(B, H, T, K, dtype=kc.dtype, device=kc.device)
    kk = torch.empty(B, NT, H, BT, BT, dtype=torch.bfloat16, device=q.device)
    qk = torch.empty(B, NT, H, BT, BT, dtype=torch.bfloat16, device=q.device)
    neumann_m = torch.empty(B, NT, HV, BT, BT, dtype=torch.bfloat16, device=q.device)
    gated_qk = torch.empty(B, NT, HV, BT, BT, dtype=torch.bfloat16, device=q.device)
    exp_gc = torch.empty(B, NT, HV, BT, dtype=torch.float32, device=q.device)
    exp_decay = torch.empty(B, NT, HV, BT, dtype=torch.float32, device=q.device)
    o = torch.empty(B, T, HV, V, dtype=v.dtype, device=v.device)

    mQ = from_dlpack(qc, assumed_align=16)
    mK = from_dlpack(kc, assumed_align=16)
    mV = from_dlpack(vc, assumed_align=16)
    mG = from_dlpack(gc_, assumed_align=16)
    mBeta = from_dlpack(betac, assumed_align=16)
    mQnorm = from_dlpack(q_norm, assumed_align=16)
    mKnorm = from_dlpack(k_norm, assumed_align=16)
    mKK = from_dlpack(kk, assumed_align=16)
    mQK = from_dlpack(qk, assumed_align=16)
    mM = from_dlpack(neumann_m, assumed_align=16)
    mGQK = from_dlpack(gated_qk, assumed_align=16)
    mExpGC = from_dlpack(exp_gc, assumed_align=16)
    mExpDecay = from_dlpack(exp_decay, assumed_align=16)
    mO = from_dlpack(o, assumed_align=16)

    key0 = ("pinv-k0-qk-bt32", B, T, H, K, q.dtype)
    compiled0 = _compiled_cache.get(key0)
    if compiled0 is None:
        compiled0 = cute.compile(
            launch_kernel0,
            mQ, mK, mQnorm, mKnorm, mKK, mQK,
            cutlass.Int32(B), T, H, NT,
        )
        _compiled_cache[key0] = compiled0
    compiled0(mQ, mK, mQnorm, mKnorm, mKK, mQK, cutlass.Int32(B))

    key_inv = ("pinv-kinv-gqk-bt32", B, T, H, HV, q.dtype)
    compiled_inv = _compiled_cache.get(key_inv)
    if compiled_inv is None:
        compiled_inv = cute.compile(
            launch_kernel_inv,
            mKK, mQK, mG, mBeta,
            mM, mGQK, mExpGC, mExpDecay,
            cutlass.Int32(B),
            T, H, HV, NT, H_PER_HV,
        )
        _compiled_cache[key_inv] = compiled_inv
    compiled_inv(
        mKK, mQK, mG, mBeta,
        mM, mGQK, mExpGC, mExpDecay,
        cutlass.Int32(B),
    )

    key1 = (
        "pinv-k1-bt32-vtile-aligned-scalar-probe-v1",
        bv_tile,
        B, T, H, HV, K, V, q.dtype, float(scale),
    )
    compiled1 = _compiled_cache.get(key1)
    if compiled1 is None:
        compiled1 = cute.compile(
            launch_kernel1_aligned_scalar_probe,
            mKnorm, mQnorm, mV, mBeta,
            mM, mGQK, mExpGC, mExpDecay,
            mO,
            float(scale), cutlass.Int32(B),
            T, H, HV, V, NT, H_PER_HV, bv_tile, per_v_tile,
        )
        _compiled_cache[key1] = compiled1
    compiled1(
        mKnorm, mQnorm, mV, mBeta,
        mM, mGQK, mExpGC, mExpDecay,
        mO,
        cutlass.Int32(B),
    )

    return o


def run_3kernel_aligned_scalar_probe(q, k, v, g, beta, scale):
    """V15 opt-in probe: scalar M/GQK loads with BT+8 shared strides."""
    return _run_3kernel_aligned_scalar_probe_tiled(q, k, v, g, beta, scale, BV)


def _run_3kernel_m_cpasync_probe_tiled(q, k, v, g, beta, scale, bv_tile: int):
    B, T, H, K = q.shape
    _, _, HV, V = v.shape
    NT = T // BT
    H_PER_HV = HV // H
    if V % bv_tile != 0:
        raise ValueError(f"V={V} must be divisible by bv_tile={bv_tile}")
    per_v_tile = (BT * bv_tile) // NUM_THREADS

    qc = q.contiguous()
    kc = k.contiguous()
    vc = v.contiguous()
    gc_ = g.contiguous()
    betac = beta.contiguous()

    q_norm = torch.empty(B, H, T, K, dtype=qc.dtype, device=qc.device)
    k_norm = torch.empty(B, H, T, K, dtype=kc.dtype, device=kc.device)
    kk = torch.empty(B, NT, H, BT, BT, dtype=torch.bfloat16, device=q.device)
    qk = torch.empty(B, NT, H, BT, BT, dtype=torch.bfloat16, device=q.device)
    neumann_m = torch.empty(B, NT, HV, BT, BT, dtype=torch.bfloat16, device=q.device)
    gated_qk = torch.empty(B, NT, HV, BT, BT, dtype=torch.bfloat16, device=q.device)
    exp_gc = torch.empty(B, NT, HV, BT, dtype=torch.float32, device=q.device)
    exp_decay = torch.empty(B, NT, HV, BT, dtype=torch.float32, device=q.device)
    o = torch.empty(B, T, HV, V, dtype=v.dtype, device=v.device)

    mQ = from_dlpack(qc, assumed_align=16)
    mK = from_dlpack(kc, assumed_align=16)
    mV = from_dlpack(vc, assumed_align=16)
    mG = from_dlpack(gc_, assumed_align=16)
    mBeta = from_dlpack(betac, assumed_align=16)
    mQnorm = from_dlpack(q_norm, assumed_align=16)
    mKnorm = from_dlpack(k_norm, assumed_align=16)
    mKK = from_dlpack(kk, assumed_align=16)
    mQK = from_dlpack(qk, assumed_align=16)
    mM = from_dlpack(neumann_m, assumed_align=16)
    mGQK = from_dlpack(gated_qk, assumed_align=16)
    mExpGC = from_dlpack(exp_gc, assumed_align=16)
    mExpDecay = from_dlpack(exp_decay, assumed_align=16)
    mO = from_dlpack(o, assumed_align=16)

    key0 = ("pinv-k0-qk-bt32", B, T, H, K, q.dtype)
    compiled0 = _compiled_cache.get(key0)
    if compiled0 is None:
        compiled0 = cute.compile(
            launch_kernel0,
            mQ, mK, mQnorm, mKnorm, mKK, mQK,
            cutlass.Int32(B), T, H, NT,
        )
        _compiled_cache[key0] = compiled0
    compiled0(mQ, mK, mQnorm, mKnorm, mKK, mQK, cutlass.Int32(B))

    key_inv = ("pinv-kinv-gqk-bt32", B, T, H, HV, q.dtype)
    compiled_inv = _compiled_cache.get(key_inv)
    if compiled_inv is None:
        compiled_inv = cute.compile(
            launch_kernel_inv,
            mKK, mQK, mG, mBeta,
            mM, mGQK, mExpGC, mExpDecay,
            cutlass.Int32(B),
            T, H, HV, NT, H_PER_HV,
        )
        _compiled_cache[key_inv] = compiled_inv
    compiled_inv(
        mKK, mQK, mG, mBeta,
        mM, mGQK, mExpGC, mExpDecay,
        cutlass.Int32(B),
    )

    key1 = (
        "pinv-k1-bt32-vtile-m-cpasync-probe-v1",
        bv_tile,
        B, T, H, HV, K, V, q.dtype, float(scale),
    )
    compiled1 = _compiled_cache.get(key1)
    if compiled1 is None:
        compiled1 = cute.compile(
            launch_kernel1_m_cpasync_probe,
            mKnorm, mQnorm, mV, mBeta,
            mM, mGQK, mExpGC, mExpDecay,
            mO,
            float(scale), cutlass.Int32(B),
            T, H, HV, V, NT, H_PER_HV, bv_tile, per_v_tile,
        )
        _compiled_cache[key1] = compiled1
    compiled1(
        mKnorm, mQnorm, mV, mBeta,
        mM, mGQK, mExpGC, mExpDecay,
        mO,
        cutlass.Int32(B),
    )

    return o


def run_3kernel_m_cpasync_probe(q, k, v, g, beta, scale):
    """V15 opt-in probe: cp.async M loads with scalar GQK loads."""
    return _run_3kernel_m_cpasync_probe_tiled(q, k, v, g, beta, scale, BV)


def _run_3kernel_gqk_cpasync_probe_tiled(q, k, v, g, beta, scale, bv_tile: int):
    B, T, H, K = q.shape
    _, _, HV, V = v.shape
    NT = T // BT
    H_PER_HV = HV // H
    if V % bv_tile != 0:
        raise ValueError(f"V={V} must be divisible by bv_tile={bv_tile}")
    per_v_tile = (BT * bv_tile) // NUM_THREADS

    qc = q.contiguous()
    kc = k.contiguous()
    vc = v.contiguous()
    gc_ = g.contiguous()
    betac = beta.contiguous()

    q_norm = torch.empty(B, H, T, K, dtype=qc.dtype, device=qc.device)
    k_norm = torch.empty(B, H, T, K, dtype=kc.dtype, device=kc.device)
    kk = torch.empty(B, NT, H, BT, BT, dtype=torch.bfloat16, device=q.device)
    qk = torch.empty(B, NT, H, BT, BT, dtype=torch.bfloat16, device=q.device)
    neumann_m = torch.empty(B, NT, HV, BT, BT, dtype=torch.bfloat16, device=q.device)
    gated_qk = torch.empty(B, NT, HV, BT, BT, dtype=torch.bfloat16, device=q.device)
    exp_gc = torch.empty(B, NT, HV, BT, dtype=torch.float32, device=q.device)
    exp_decay = torch.empty(B, NT, HV, BT, dtype=torch.float32, device=q.device)
    o = torch.empty(B, T, HV, V, dtype=v.dtype, device=v.device)

    mQ = from_dlpack(qc, assumed_align=16)
    mK = from_dlpack(kc, assumed_align=16)
    mV = from_dlpack(vc, assumed_align=16)
    mG = from_dlpack(gc_, assumed_align=16)
    mBeta = from_dlpack(betac, assumed_align=16)
    mQnorm = from_dlpack(q_norm, assumed_align=16)
    mKnorm = from_dlpack(k_norm, assumed_align=16)
    mKK = from_dlpack(kk, assumed_align=16)
    mQK = from_dlpack(qk, assumed_align=16)
    mM = from_dlpack(neumann_m, assumed_align=16)
    mGQK = from_dlpack(gated_qk, assumed_align=16)
    mExpGC = from_dlpack(exp_gc, assumed_align=16)
    mExpDecay = from_dlpack(exp_decay, assumed_align=16)
    mO = from_dlpack(o, assumed_align=16)

    key0 = ("pinv-k0-qk-bt32", B, T, H, K, q.dtype)
    compiled0 = _compiled_cache.get(key0)
    if compiled0 is None:
        compiled0 = cute.compile(
            launch_kernel0,
            mQ, mK, mQnorm, mKnorm, mKK, mQK,
            cutlass.Int32(B), T, H, NT,
        )
        _compiled_cache[key0] = compiled0
    compiled0(mQ, mK, mQnorm, mKnorm, mKK, mQK, cutlass.Int32(B))

    key_inv = ("pinv-kinv-gqk-bt32", B, T, H, HV, q.dtype)
    compiled_inv = _compiled_cache.get(key_inv)
    if compiled_inv is None:
        compiled_inv = cute.compile(
            launch_kernel_inv,
            mKK, mQK, mG, mBeta,
            mM, mGQK, mExpGC, mExpDecay,
            cutlass.Int32(B),
            T, H, HV, NT, H_PER_HV,
        )
        _compiled_cache[key_inv] = compiled_inv
    compiled_inv(
        mKK, mQK, mG, mBeta,
        mM, mGQK, mExpGC, mExpDecay,
        cutlass.Int32(B),
    )

    key1 = (
        "pinv-k1-bt32-vtile-gqk-cpasync-probe-v1",
        bv_tile,
        B, T, H, HV, K, V, q.dtype, float(scale),
    )
    compiled1 = _compiled_cache.get(key1)
    if compiled1 is None:
        compiled1 = cute.compile(
            launch_kernel1_gqk_cpasync_probe,
            mKnorm, mQnorm, mV, mBeta,
            mM, mGQK, mExpGC, mExpDecay,
            mO,
            float(scale), cutlass.Int32(B),
            T, H, HV, V, NT, H_PER_HV, bv_tile, per_v_tile,
        )
        _compiled_cache[key1] = compiled1
    compiled1(
        mKnorm, mQnorm, mV, mBeta,
        mM, mGQK, mExpGC, mExpDecay,
        mO,
        cutlass.Int32(B),
    )

    return o


def run_3kernel_gqk_cpasync_probe(q, k, v, g, beta, scale):
    """V16 opt-in probe: cp.async GQK loads with scalar M loads."""
    return _run_3kernel_gqk_cpasync_probe_tiled(q, k, v, g, beta, scale, BV)


def _run_3kernel_pairv_reuse_probe_tiled(q, k, v, g, beta, scale, bv_tile: int):
    B, T, H, K = q.shape
    _, _, HV, V = v.shape
    NT = T // BT
    H_PER_HV = HV // H
    if V % (2 * bv_tile) != 0:
        raise ValueError(f"V={V} must be divisible by 2*bv_tile={2 * bv_tile}")
    per_v_tile = (BT * bv_tile) // NUM_THREADS

    qc = q.contiguous()
    kc = k.contiguous()
    vc = v.contiguous()
    gc_ = g.contiguous()
    betac = beta.contiguous()

    q_norm = torch.empty(B, H, T, K, dtype=qc.dtype, device=qc.device)
    k_norm = torch.empty(B, H, T, K, dtype=kc.dtype, device=kc.device)
    kk = torch.empty(B, NT, H, BT, BT, dtype=torch.bfloat16, device=q.device)
    qk = torch.empty(B, NT, H, BT, BT, dtype=torch.bfloat16, device=q.device)
    neumann_m = torch.empty(B, NT, HV, BT, BT, dtype=torch.bfloat16, device=q.device)
    gated_qk = torch.empty(B, NT, HV, BT, BT, dtype=torch.bfloat16, device=q.device)
    exp_gc = torch.empty(B, NT, HV, BT, dtype=torch.float32, device=q.device)
    exp_decay = torch.empty(B, NT, HV, BT, dtype=torch.float32, device=q.device)
    o = torch.empty(B, T, HV, V, dtype=v.dtype, device=v.device)

    mQ = from_dlpack(qc, assumed_align=16)
    mK = from_dlpack(kc, assumed_align=16)
    mV = from_dlpack(vc, assumed_align=16)
    mG = from_dlpack(gc_, assumed_align=16)
    mBeta = from_dlpack(betac, assumed_align=16)
    mQnorm = from_dlpack(q_norm, assumed_align=16)
    mKnorm = from_dlpack(k_norm, assumed_align=16)
    mKK = from_dlpack(kk, assumed_align=16)
    mQK = from_dlpack(qk, assumed_align=16)
    mM = from_dlpack(neumann_m, assumed_align=16)
    mGQK = from_dlpack(gated_qk, assumed_align=16)
    mExpGC = from_dlpack(exp_gc, assumed_align=16)
    mExpDecay = from_dlpack(exp_decay, assumed_align=16)
    mO = from_dlpack(o, assumed_align=16)

    key0 = ("pinv-k0-qk-bt32", B, T, H, K, q.dtype)
    compiled0 = _compiled_cache.get(key0)
    if compiled0 is None:
        compiled0 = cute.compile(
            launch_kernel0,
            mQ, mK, mQnorm, mKnorm, mKK, mQK,
            cutlass.Int32(B), T, H, NT,
        )
        _compiled_cache[key0] = compiled0
    compiled0(mQ, mK, mQnorm, mKnorm, mKK, mQK, cutlass.Int32(B))

    key_inv = ("pinv-kinv-gqk-bt32", B, T, H, HV, q.dtype)
    compiled_inv = _compiled_cache.get(key_inv)
    if compiled_inv is None:
        compiled_inv = cute.compile(
            launch_kernel_inv,
            mKK, mQK, mG, mBeta,
            mM, mGQK, mExpGC, mExpDecay,
            cutlass.Int32(B),
            T, H, HV, NT, H_PER_HV,
        )
        _compiled_cache[key_inv] = compiled_inv
    compiled_inv(
        mKK, mQK, mG, mBeta,
        mM, mGQK, mExpGC, mExpDecay,
        cutlass.Int32(B),
    )

    key1 = (
        "pinv-k1-bt32-pairv-reuse-probe-v1",
        bv_tile,
        B, T, H, HV, K, V, q.dtype, float(scale),
    )
    compiled1 = _compiled_cache.get(key1)
    if compiled1 is None:
        compiled1 = cute.compile(
            launch_kernel1_pairv_reuse_probe,
            mKnorm, mQnorm, mV, mBeta,
            mM, mGQK, mExpGC, mExpDecay,
            mO,
            float(scale), cutlass.Int32(B),
            T, H, HV, V, NT, H_PER_HV, bv_tile, per_v_tile,
        )
        _compiled_cache[key1] = compiled1
    compiled1(
        mKnorm, mQnorm, mV, mBeta,
        mM, mGQK, mExpGC, mExpDecay,
        mO,
        cutlass.Int32(B),
    )

    return o


def run_3kernel_pairv_reuse_probe(q, k, v, g, beta, scale):
    """V12 isolated probe: one CTA serially computes two BV16 V subtiles."""
    return _run_3kernel_pairv_reuse_probe_tiled(q, k, v, g, beta, scale, BV)


def _run_3kernel_bf16_state_probe_tiled(q, k, v, g, beta, scale, bv_tile: int):
    B, T, H, K = q.shape
    _, _, HV, V = v.shape
    NT = T // BT
    H_PER_HV = HV // H
    if V % bv_tile != 0:
        raise ValueError(f"V={V} must be divisible by bv_tile={bv_tile}")
    per_v_tile = (BT * bv_tile) // NUM_THREADS

    qc = q.contiguous()
    kc = k.contiguous()
    vc = v.contiguous()
    gc_ = g.contiguous()
    betac = beta.contiguous()

    q_norm = torch.empty(B, H, T, K, dtype=qc.dtype, device=qc.device)
    k_norm = torch.empty(B, H, T, K, dtype=kc.dtype, device=kc.device)
    kk = torch.empty(B, NT, H, BT, BT, dtype=torch.bfloat16, device=q.device)
    qk = torch.empty(B, NT, H, BT, BT, dtype=torch.bfloat16, device=q.device)
    neumann_m = torch.empty(B, NT, HV, BT, BT, dtype=torch.bfloat16, device=q.device)
    gated_qk = torch.empty(B, NT, HV, BT, BT, dtype=torch.bfloat16, device=q.device)
    exp_gc = torch.empty(B, NT, HV, BT, dtype=torch.float32, device=q.device)
    exp_decay = torch.empty(B, NT, HV, BT, dtype=torch.float32, device=q.device)
    o = torch.empty(B, T, HV, V, dtype=v.dtype, device=v.device)

    mQ = from_dlpack(qc, assumed_align=16)
    mK = from_dlpack(kc, assumed_align=16)
    mV = from_dlpack(vc, assumed_align=16)
    mG = from_dlpack(gc_, assumed_align=16)
    mBeta = from_dlpack(betac, assumed_align=16)
    mQnorm = from_dlpack(q_norm, assumed_align=16)
    mKnorm = from_dlpack(k_norm, assumed_align=16)
    mKK = from_dlpack(kk, assumed_align=16)
    mQK = from_dlpack(qk, assumed_align=16)
    mM = from_dlpack(neumann_m, assumed_align=16)
    mGQK = from_dlpack(gated_qk, assumed_align=16)
    mExpGC = from_dlpack(exp_gc, assumed_align=16)
    mExpDecay = from_dlpack(exp_decay, assumed_align=16)
    mO = from_dlpack(o, assumed_align=16)

    key0 = ("pinv-k0-qk-bt32", B, T, H, K, q.dtype)
    compiled0 = _compiled_cache.get(key0)
    if compiled0 is None:
        compiled0 = cute.compile(
            launch_kernel0,
            mQ, mK, mQnorm, mKnorm, mKK, mQK,
            cutlass.Int32(B), T, H, NT,
        )
        _compiled_cache[key0] = compiled0
    compiled0(mQ, mK, mQnorm, mKnorm, mKK, mQK, cutlass.Int32(B))

    key_inv = ("pinv-kinv-gqk-bt32", B, T, H, HV, q.dtype)
    compiled_inv = _compiled_cache.get(key_inv)
    if compiled_inv is None:
        compiled_inv = cute.compile(
            launch_kernel_inv,
            mKK, mQK, mG, mBeta,
            mM, mGQK, mExpGC, mExpDecay,
            cutlass.Int32(B),
            T, H, HV, NT, H_PER_HV,
        )
        _compiled_cache[key_inv] = compiled_inv
    compiled_inv(
        mKK, mQK, mG, mBeta,
        mM, mGQK, mExpGC, mExpDecay,
        cutlass.Int32(B),
    )

    key1 = (
        "pinv-k1-bt32-vtile-bf16-state-probe-v1",
        bv_tile,
        B, T, H, HV, K, V, q.dtype, float(scale),
    )
    compiled1 = _compiled_cache.get(key1)
    if compiled1 is None:
        compiled1 = cute.compile(
            launch_kernel1_bf16_state_probe,
            mKnorm, mQnorm, mV, mBeta,
            mM, mGQK, mExpGC, mExpDecay,
            mO,
            float(scale), cutlass.Int32(B),
            T, H, HV, V, NT, H_PER_HV, bv_tile, per_v_tile,
        )
        _compiled_cache[key1] = compiled1
    compiled1(
        mKnorm, mQnorm, mV, mBeta,
        mM, mGQK, mExpGC, mExpDecay,
        mO,
        cutlass.Int32(B),
    )

    return o


def run_3kernel_bf16_state_probe(q, k, v, g, beta, scale):
    """V11 isolated probe: V8 pipeline with bf16 recurrent state registers."""
    return _run_3kernel_bf16_state_probe_tiled(q, k, v, g, beta, scale, BV)


def _run_3kernel_state_mma_probe_tiled(q, k, v, g, beta, scale, bv_tile: int):
    B, T, H, K = q.shape
    _, _, HV, V = v.shape
    NT = T // BT
    H_PER_HV = HV // H
    if V % bv_tile != 0:
        raise ValueError(f"V={V} must be divisible by bv_tile={bv_tile}")
    per_v_tile = (BT * bv_tile) // NUM_THREADS
    per_state_tile = (K_DIM * bv_tile) // NUM_THREADS

    qc = q.contiguous()
    kc = k.contiguous()
    vc = v.contiguous()
    gc_ = g.contiguous()
    betac = beta.contiguous()

    q_norm = torch.empty(B, H, T, K, dtype=qc.dtype, device=qc.device)
    k_norm = torch.empty(B, H, T, K, dtype=kc.dtype, device=kc.device)
    kk = torch.empty(B, NT, H, BT, BT, dtype=torch.bfloat16, device=q.device)
    qk = torch.empty(B, NT, H, BT, BT, dtype=torch.bfloat16, device=q.device)
    neumann_m = torch.empty(B, NT, HV, BT, BT, dtype=torch.bfloat16, device=q.device)
    gated_qk = torch.empty(B, NT, HV, BT, BT, dtype=torch.bfloat16, device=q.device)
    exp_gc = torch.empty(B, NT, HV, BT, dtype=torch.float32, device=q.device)
    exp_decay = torch.empty(B, NT, HV, BT, dtype=torch.float32, device=q.device)
    o = torch.empty(B, T, HV, V, dtype=v.dtype, device=v.device)

    mQ = from_dlpack(qc, assumed_align=16)
    mK = from_dlpack(kc, assumed_align=16)
    mV = from_dlpack(vc, assumed_align=16)
    mG = from_dlpack(gc_, assumed_align=16)
    mBeta = from_dlpack(betac, assumed_align=16)
    mQnorm = from_dlpack(q_norm, assumed_align=16)
    mKnorm = from_dlpack(k_norm, assumed_align=16)
    mKK = from_dlpack(kk, assumed_align=16)
    mQK = from_dlpack(qk, assumed_align=16)
    mM = from_dlpack(neumann_m, assumed_align=16)
    mGQK = from_dlpack(gated_qk, assumed_align=16)
    mExpGC = from_dlpack(exp_gc, assumed_align=16)
    mExpDecay = from_dlpack(exp_decay, assumed_align=16)
    mO = from_dlpack(o, assumed_align=16)

    key0 = ("pinv-k0-qk-bt32", B, T, H, K, q.dtype)
    compiled0 = _compiled_cache.get(key0)
    if compiled0 is None:
        compiled0 = cute.compile(
            launch_kernel0,
            mQ, mK, mQnorm, mKnorm, mKK, mQK,
            cutlass.Int32(B), T, H, NT,
        )
        _compiled_cache[key0] = compiled0
    compiled0(mQ, mK, mQnorm, mKnorm, mKK, mQK, cutlass.Int32(B))

    key_inv = ("pinv-kinv-gqk-bt32", B, T, H, HV, q.dtype)
    compiled_inv = _compiled_cache.get(key_inv)
    if compiled_inv is None:
        compiled_inv = cute.compile(
            launch_kernel_inv,
            mKK, mQK, mG, mBeta,
            mM, mGQK, mExpGC, mExpDecay,
            cutlass.Int32(B),
            T, H, HV, NT, H_PER_HV,
        )
        _compiled_cache[key_inv] = compiled_inv
    compiled_inv(
        mKK, mQK, mG, mBeta,
        mM, mGQK, mExpGC, mExpDecay,
        cutlass.Int32(B),
    )

    key1 = (
        "pinv-k1-bt32-vtile-state-mma-probe-v1",
        bv_tile,
        B, T, H, HV, K, V, q.dtype, float(scale),
    )
    compiled1 = _compiled_cache.get(key1)
    if compiled1 is None:
        compiled1 = cute.compile(
            launch_kernel1_state_mma_probe,
            mKnorm, mQnorm, mV, mBeta,
            mM, mGQK, mExpGC, mExpDecay,
            mO,
            float(scale), cutlass.Int32(B),
            T, H, HV, V, NT, H_PER_HV,
            bv_tile, per_v_tile, per_state_tile,
        )
        _compiled_cache[key1] = compiled1
    compiled1(
        mKnorm, mQnorm, mV, mBeta,
        mM, mGQK, mExpGC, mExpDecay,
        mO,
        cutlass.Int32(B),
    )

    return o


def run_3kernel_state_mma_probe(q, k, v, g, beta, scale):
    """V10 isolated splice probe: V8 pipeline with K1 state update via MMA."""
    return _run_3kernel_state_mma_probe_tiled(q, k, v, g, beta, scale, BV)


def _run_megakernel_tiled(q, k, v, g, beta, scale, bv_tile: int):
    """Two-kernel CuTeDSL prototype: K0 preprocess + K1 inline inverse/h/o.

    `bv_tile` is a compile-time V tile size. Larger tiles reduce repeated inverse
    work but increase register pressure and reduce CTA parallelism.
    """
    B, T, H, K = q.shape
    _, _, HV, V = v.shape
    NT = T // BT
    H_PER_HV = HV // H
    if V % bv_tile != 0:
        raise ValueError(f"V={V} must be divisible by bv_tile={bv_tile}")
    per_v_tile = (BT * bv_tile) // NUM_THREADS

    qc = q.contiguous()
    kc = k.contiguous()
    vc = v.contiguous()
    gc_ = g.contiguous()
    betac = beta.contiguous()

    q_norm = torch.empty(B, H, T, K, dtype=qc.dtype, device=qc.device)
    k_norm = torch.empty(B, H, T, K, dtype=kc.dtype, device=kc.device)
    kk = torch.empty(B, NT, H, BT, BT, dtype=torch.bfloat16, device=q.device)
    o = torch.empty(B, T, HV, V, dtype=v.dtype, device=v.device)

    mQ = from_dlpack(qc, assumed_align=16)
    mK = from_dlpack(kc, assumed_align=16)
    mV = from_dlpack(vc, assumed_align=16)
    mG = from_dlpack(gc_, assumed_align=16)
    mBeta = from_dlpack(betac, assumed_align=16)
    mQnorm = from_dlpack(q_norm, assumed_align=16)
    mKnorm = from_dlpack(k_norm, assumed_align=16)
    mKK = from_dlpack(kk, assumed_align=16)
    mO = from_dlpack(o, assumed_align=16)

    key0 = ("pinv-k0-kk-only-bt32", B, T, H, K, q.dtype)
    compiled0 = _compiled_cache.get(key0)
    if compiled0 is None:
        compiled0 = cute.compile(
            launch_kernel0_kk_only,
            mQ, mK, mQnorm, mKnorm, mKK,
            cutlass.Int32(B), T, H, NT,
        )
        _compiled_cache[key0] = compiled0
    compiled0(mQ, mK, mQnorm, mKnorm, mKK, cutlass.Int32(B))

    key1 = (
        "mega-k1-inline-inv-bt32-vtile-v2",
        bv_tile,
        B, T, H, HV, K, V, q.dtype, float(scale),
    )
    compiled1 = _compiled_cache.get(key1)
    if compiled1 is None:
        compiled1 = cute.compile(
            launch_kernel1_mega,
            mKK, mKnorm, mQnorm, mV, mG, mBeta,
            mO,
            float(scale), cutlass.Int32(B),
            T, H, HV, V, NT, H_PER_HV, bv_tile, per_v_tile,
        )
        _compiled_cache[key1] = compiled1
    compiled1(
        mKK, mKnorm, mQnorm, mV, mG, mBeta,
        mO,
        cutlass.Int32(B),
    )

    return o


def _run_megakernel_tma_tiled(q, k, v, g, beta, scale, bv_tile: int):
    """Two-kernel CuTeDSL mega probe with TMA K/Q G2S in K1."""
    B, T, H, K = q.shape
    _, _, HV, V = v.shape
    NT = T // BT
    H_PER_HV = HV // H
    if V % bv_tile != 0:
        raise ValueError(f"V={V} must be divisible by bv_tile={bv_tile}")
    per_v_tile = (BT * bv_tile) // NUM_THREADS

    qc = q.contiguous()
    kc = k.contiguous()
    vc = v.contiguous()
    gc_ = g.contiguous()
    betac = beta.contiguous()

    q_norm = torch.empty(B, H, T, K, dtype=qc.dtype, device=qc.device)
    k_norm = torch.empty(B, H, T, K, dtype=kc.dtype, device=kc.device)
    kk = torch.empty(B, NT, H, BT, BT, dtype=torch.bfloat16, device=q.device)
    o = torch.empty(B, T, HV, V, dtype=v.dtype, device=v.device)

    mQ = from_dlpack(qc, assumed_align=16)
    mK = from_dlpack(kc, assumed_align=16)
    mV = from_dlpack(vc, assumed_align=16)
    mG = from_dlpack(gc_, assumed_align=16)
    mBeta = from_dlpack(betac, assumed_align=16)
    mQnorm = from_dlpack(q_norm, assumed_align=16)
    mKnorm = from_dlpack(k_norm, assumed_align=16)
    mKK = from_dlpack(kk, assumed_align=16)
    mO = from_dlpack(o, assumed_align=16)

    key0 = ("pinv-k0-kk-only-bt32", B, T, H, K, q.dtype)
    compiled0 = _compiled_cache.get(key0)
    if compiled0 is None:
        compiled0 = cute.compile(
            launch_kernel0_kk_only,
            mQ, mK, mQnorm, mKnorm, mKK,
            cutlass.Int32(B), T, H, NT,
        )
        _compiled_cache[key0] = compiled0
    compiled0(mQ, mK, mQnorm, mKnorm, mKK, cutlass.Int32(B))

    key1 = (
        "mega-k1-tma-inline-inv-bt32-vtile-v1",
        bv_tile,
        B, T, H, HV, K, V, q.dtype, float(scale),
    )
    compiled1 = _compiled_cache.get(key1)
    if compiled1 is None:
        compiled1 = cute.compile(
            launch_kernel1_mega_tma,
            mKK, mKnorm, mQnorm, mV, mG, mBeta,
            mO,
            float(scale), cutlass.Int32(B),
            T, H, HV, V, NT, H_PER_HV, bv_tile, per_v_tile,
        )
        _compiled_cache[key1] = compiled1
    compiled1(
        mKK, mKnorm, mQnorm, mV, mG, mBeta,
        mO,
        cutlass.Int32(B),
    )

    return o


def run_megakernel(q, k, v, g, beta, scale):
    """Compatibility entry for the original BV=16 sm120 megakernel probe."""
    return _run_megakernel_tiled(q, k, v, g, beta, scale, BV)


def run_megakernel_bv32(q, k, v, g, beta, scale):
    """BV=32 sm120 megakernel probe to reduce duplicate inline inverse work."""
    return _run_megakernel_tiled(q, k, v, g, beta, scale, 32)


def run_megakernel_tma(q, k, v, g, beta, scale):
    """BV=16 sm120 megakernel probe using TMA for K/Q loads."""
    return _run_megakernel_tma_tiled(q, k, v, g, beta, scale, BV)


# ============================================================================
# Atrex public wrapper and tail-aware final-state path
# ============================================================================

_run_3kernel_no_tail = run_3kernel


def _is_supported_fast_path(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool,
    cu_seqlens: torch.Tensor | None,
    cp_context,
    transpose_state_layout: bool,
    kwargs: dict,
) -> bool:
    if q.device.type != "cuda" or k.device != q.device or v.device != q.device:
        return False
    if q.dtype is not torch.bfloat16 or k.dtype is not torch.bfloat16 or v.dtype is not torch.bfloat16:
        return False
    if g.dtype is not torch.bfloat16 or beta.dtype is not torch.bfloat16:
        return False
    if q.shape != k.shape or q.ndim != 4 or v.ndim != 4:
        return False
    b, t, h, k_dim = q.shape
    bv, tv, hv, v_dim = v.shape
    if (b, tv, h, hv, k_dim, v_dim) != (1, t, 16, 32, 128, 128):
        return False
    if g.shape != (1, t, hv) or beta.shape != (1, t, hv):
        return False
    if initial_state is not None or cp_context is not None or transpose_state_layout:
        return False
    if not use_qk_l2norm_in_kernel:
        return False
    if kwargs.get("use_gate_in_kernel", False):
        return False
    if cu_seqlens is not None and (cu_seqlens.numel() != 2):
        return False
    return True


@cute.kernel
def _fused_chunk_h_v31_final_state_kernel(
    tiled_mma: cute.TiledMma,
    smem_tiled_copy_A: cute.TiledCopy,
    smem_tiled_copy_A_trans: cute.TiledCopy,
    smem_tiled_copy_B: cute.TiledCopy,
    tiled_copy_state_r2s: cute.TiledCopy,
    tiled_copy_o_gmem: cute.TiledCopy,
    tiled_copy_kq: cute.TiledCopy,
    tiled_copy_mgqk: cute.TiledCopy,
    tiled_copy_v: cute.TiledCopy,
    sK_layout: cute.Layout,
    sV_layout: cute.Layout,
    sA_layout: cute.Layout,
    sGQK_layout: cute.Layout,
    sBeta_layout: cute.Layout,
    sS_layout: cute.Layout,
    sNK_layout: cute.Layout,
    sExpGC_layout: cute.Layout,
    sExpDecay_layout: cute.Layout,
    mKnorm: cute.Tensor,
    mQnorm: cute.Tensor,
    mV: cute.Tensor,
    mBeta: cute.Tensor,
    mM: cute.Tensor,
    mGQK: cute.Tensor,
    mExpGC_in: cute.Tensor,
    mExpDecay_in: cute.Tensor,
    mO: cute.Tensor,
    mFinalState: cute.Tensor,
    scale: cutlass.Constexpr[float],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    NT: cutlass.Constexpr[int],
    H_PER_HV: cutlass.Constexpr[int],
    BV_TILE: cutlass.Constexpr[int],
    STATE_SPLIT: cutlass.Constexpr[bool],
    STATE_SPLIT_START_CHUNK: cutlass.Constexpr[int],
    STATE_PAD_ELEMS: cutlass.Constexpr[int],
    M_FOLDED_BETA: cutlass.Constexpr[bool],
):
    tidx, _, _ = cute.arch.thread_idx()
    bid_v = cute.arch.block_idx()[0]
    bid_bh = cute.arch.block_idx()[1]
    i_b = bid_bh // HV
    i_hv = bid_bh % HV
    i_h = i_hv // H_PER_HV
    v_off = bid_v * BV_TILE

    smem = cutlass.utils.SmemAllocator()
    sK = smem.allocate_tensor(cutlass.BFloat16, sK_layout, 16)
    sQ = smem.allocate_tensor(cutlass.BFloat16, sK_layout, 16)
    sV = smem.allocate_tensor(cutlass.BFloat16, sV_layout, 16)
    sA = smem.allocate_tensor(cutlass.BFloat16, sA_layout, 16)
    sGQK = smem.allocate_tensor(cutlass.BFloat16, sGQK_layout, 16)
    if cutlass.const_expr(not M_FOLDED_BETA):
        sBeta = smem.allocate_tensor(cutlass.Float32, sBeta_layout, 16)
    sS = smem.allocate_tensor(cutlass.BFloat16, sS_layout, 16)
    if cutlass.const_expr(STATE_PAD_ELEMS > 0):
        sPad_layout = cute.make_layout((STATE_PAD_ELEMS,), stride=(1,))
        sPad = smem.allocate_tensor(cutlass.BFloat16, sPad_layout, 16)
    sNK_A = smem.allocate_tensor(cutlass.BFloat16, sNK_layout, 16)
    sExpGC = smem.allocate_tensor(cutlass.Float32, sExpGC_layout, 16)
    sExpDecay = smem.allocate_tensor(cutlass.Float32, sExpDecay_layout, 16)

    thr_mma = tiled_mma.get_slice(tidx)
    cC_bv = cute.make_identity_tensor((BT, BV_TILE))
    tCcC_bv = thr_mma.partition_C(cC_bv)

    state0 = cute.make_rmem_tensor(
        thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
    state1 = cute.make_rmem_tensor(
        thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
    state2 = cute.make_rmem_tensor(
        thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
    state3 = cute.make_rmem_tensor(
        thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
    state0.fill(cutlass.Float32(0.0))
    state1.fill(cutlass.Float32(0.0))
    state2.fill(cutlass.Float32(0.0))
    state3.fill(cutlass.Float32(0.0))

    gK_full = mKnorm[(i_b, i_h, None, None)]
    gQ_full = mQnorm[(i_b, i_h, None, None)]
    gV_full = mV[(i_b, None, i_hv, None)]
    gO_full = mO[(i_b, None, i_hv, None)]

    thr_cp = tiled_copy_kq.get_slice(tidx)
    thr_sK_cp = thr_cp.partition_D(sK)
    thr_sQ_cp = thr_cp.partition_D(sQ)

    thr_cp_mgqk = tiled_copy_mgqk.get_slice(tidx)
    thr_sA_cp = thr_cp_mgqk.partition_D(sA)
    thr_sGQK_cp = thr_cp_mgqk.partition_D(sGQK)

    for chunk_idx in cutlass.range(NT):
        t0 = chunk_idx * BT

        gK_chunk = cute.local_tile(gK_full, (BT, K_DIM), (chunk_idx, 0))
        gQ_chunk = cute.local_tile(gQ_full, (BT, K_DIM), (chunk_idx, 0))
        thr_gK = thr_cp.partition_S(gK_chunk)
        thr_gQ = thr_cp.partition_S(gQ_chunk)
        cute.copy(tiled_copy_kq, thr_gK, thr_sK_cp)
        cute.copy(tiled_copy_kq, thr_gQ, thr_sQ_cp)

        gM_chunk = mM[(i_b, chunk_idx, i_hv, None, None)]
        gGQK_chunk = mGQK[(i_b, chunk_idx, i_hv, None, None)]
        thr_gM = thr_cp_mgqk.partition_S(gM_chunk)
        thr_gGQK = thr_cp_mgqk.partition_S(gGQK_chunk)
        cute.copy(tiled_copy_mgqk, thr_gM, thr_sA_cp)
        cute.copy(tiled_copy_mgqk, thr_gGQK, thr_sGQK_cp)

        if cutlass.const_expr(T % BT != 0):
            if chunk_idx == NT - 1:
                for i in cutlass.range_constexpr(PER_V):
                    idx = i * NUM_THREADS + tidx
                    row = idx // BV_TILE
                    col = idx % BV_TILE
                    if t0 + row < T:
                        sV[row, col] = mV[i_b, t0 + row, i_hv, v_off + col]
                    else:
                        sV[row, col] = cutlass.BFloat16(0.0)
            else:
                gV_chunk = cute.local_tile(gV_full, (BT, BV_TILE), (chunk_idx, bid_v))
                if tidx < 64:
                    thr_cp_v = tiled_copy_v.get_slice(tidx)
                    thr_sV_cp = thr_cp_v.partition_D(sV)
                    thr_gV = thr_cp_v.partition_S(gV_chunk)
                    cute.copy(tiled_copy_v, thr_gV, thr_sV_cp)
        else:
            gV_chunk = cute.local_tile(gV_full, (BT, BV_TILE), (chunk_idx, bid_v))
            if tidx < 64:
                thr_cp_v = tiled_copy_v.get_slice(tidx)
                thr_sV_cp = thr_cp_v.partition_D(sV)
                thr_gV = thr_cp_v.partition_S(gV_chunk)
                cute.copy(tiled_copy_v, thr_gV, thr_sV_cp)
        cute.arch.cp_async_commit_group()

        if tidx < BT:
            sExpGC[tidx] = mExpGC_in[i_b, chunk_idx, i_hv, tidx]
            sExpDecay[tidx] = mExpDecay_in[i_b, chunk_idx, i_hv, tidx]
            if cutlass.const_expr(not M_FOLDED_BETA):
                if t0 + tidx < T:
                    sBeta[tidx] = cutlass.Float32(mBeta[i_b, t0 + tidx, i_hv])
                else:
                    sBeta[tidx] = cutlass.Float32(0.0)

        sS_state_layout = cute.make_layout((BT, BV_TILE), stride=(1, K_DIM + 8))
        sS0 = cute.make_tensor(sS.iterator.align(16), sS_state_layout)
        sS1 = cute.make_tensor((sS.iterator + BT).align(16), sS_state_layout)
        sS2 = cute.make_tensor((sS.iterator + 2 * BT).align(16), sS_state_layout)
        sS3 = cute.make_tensor((sS.iterator + 3 * BT).align(16), sS_state_layout)
        _store_state_frag_r2s(tiled_copy_state_r2s, tidx, state0, sS0)
        _store_state_frag_r2s(tiled_copy_state_r2s, tidx, state1, sS1)
        _store_state_frag_r2s(tiled_copy_state_r2s, tidx, state2, sS2)
        _store_state_frag_r2s(tiled_copy_state_r2s, tidx, state3, sS3)

        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        acc_kS = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        acc_kS.fill(cutlass.Float32(0.0))
        acc_qS = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        acc_qS.fill(cutlass.Float32(0.0))

        if cutlass.const_expr(not STATE_SPLIT):
            _mma_kq_s_reuse_b_ldsm(
                tiled_mma, smem_tiled_copy_A, smem_tiled_copy_B,
                thr_mma, tidx, sK, sQ, sS, sExpGC, tCcC_bv,
                acc_kS, acc_qS, K_DIM // 16,
            )
        else:
            _mma_full_ldsm(
                tiled_mma, smem_tiled_copy_A, smem_tiled_copy_B,
                thr_mma, tidx, sK, sS, acc_kS, K_DIM // 16,
            )
            sStateLo_layout = cute.make_layout((BT, BV_TILE), stride=(1, K_DIM + 8))
            sStateLo = cute.make_tensor(sK.iterator.align(16), sStateLo_layout)
            _store_state_frag_lo_r2s(
                tiled_copy_state_r2s, tidx, state0, sStateLo
            )
            cute.arch.barrier()
            _mma_full_autovec(
                tiled_mma, thr_mma, sQ, sS, acc_qS, K_DIM // 16,
            )

        sNK_rhs_layout = cute.make_layout((BT, BV_TILE), stride=(1, BT + 8))
        sNK_rhs = cute.make_tensor(sNK_A.iterator.align(16), sNK_rhs_layout)
        if cutlass.const_expr(M_FOLDED_BETA):
            _store_rhs_snk_r2s_nobeta(
                tiled_copy_state_r2s, tidx, acc_kS,
                sV, sExpGC, sNK_rhs,
            )
        else:
            _store_rhs_snk_r2s(
                tiled_copy_state_r2s, tidx, acc_kS,
                sV, sBeta, sExpGC, sNK_rhs,
            )
        if cutlass.const_expr(STATE_SPLIT):
            for idx in cutlass.range(cute.size(acc_qS)):
                co = tCcC_bv[idx]
                acc_qS[idx] = acc_qS[idx] * sExpGC[co[0]]
        cute.arch.barrier()

        acc_vnew = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((BT, BV_TILE)), cutlass.Float32)
        acc_vnew.fill(cutlass.Float32(0.0))

        _mma_full_autovec(
            tiled_mma, thr_mma, sA, sNK_A, acc_vnew, BT // 16,
        )

        sNK_vnew_layout = cute.make_layout((BT, BV_TILE), stride=(1, BT + 8))
        sNK_vnew = cute.make_tensor(sNK_A.iterator.align(16), sNK_vnew_layout)
        _store_state_frag_r2s(
            tiled_copy_state_r2s, tidx, acc_vnew, sNK_vnew,
        )
        cute.arch.barrier()

        _mma_full_autovec(
            tiled_mma, thr_mma, sGQK, sNK_A, acc_qS, BT // 16,
        )

        if cutlass.const_expr(T % BT != 0):
            if chunk_idx == NT - 1:
                for idx in cutlass.range(cute.size(acc_qS)):
                    co = tCcC_bv[idx]
                    if t0 + co[0] < T:
                        o_val = scale * acc_qS[idx]
                        mO[i_b, t0 + co[0], i_hv, v_off + co[1]] = o_val.to(mO.element_type)
            else:
                gO_chunk = cute.local_tile(gO_full, (BT, BV_TILE), (chunk_idx, bid_v))
                _store_acc_o_gmem_vector(tiled_copy_o_gmem, tidx, acc_qS, gO_chunk, scale)
        else:
            gO_chunk = cute.local_tile(gO_full, (BT, BV_TILE), (chunk_idx, bid_v))
            _store_acc_o_gmem_vector(tiled_copy_o_gmem, tidx, acc_qS, gO_chunk, scale)
        cute.arch.barrier()
        sNK_scaled_layout = cute.make_layout((BT, BV_TILE), stride=(1, BT + 8))
        sNK_scaled = cute.make_tensor(sNK_A.iterator.align(16), sNK_scaled_layout)
        _store_scaled_vnew_snk_r2s(
            tiled_copy_state_r2s, tidx, acc_vnew, sExpDecay, sNK_scaled,
        )
        if cutlass.const_expr(STATE_SPLIT):
            for i in cutlass.range_constexpr(4):
                restore_idx = i * NUM_THREADS + tidx
                row = restore_idx // 32
                k_col = restore_idx % 32
                sK[row, k_col] = mKnorm[i_b, i_h, t0 + row, k_col]

        phi = sExpGC[BT - 1]
        for idx in cutlass.range(cute.size(state0)):
            state0[idx] = phi * state0[idx]
            state1[idx] = phi * state1[idx]
            state2[idx] = phi * state2[idx]
            state3[idx] = phi * state3[idx]

        sK_state_layout = cute.make_layout((BT, BT), stride=(1, K_DIM + 8))
        sK_t0 = cute.make_tensor(sK.iterator.align(16), sK_state_layout)
        sK_t1 = cute.make_tensor((sK.iterator + BT).align(16), sK_state_layout)
        sK_t2 = cute.make_tensor((sK.iterator + 2 * BT).align(16), sK_state_layout)
        sK_t3 = cute.make_tensor((sK.iterator + 3 * BT).align(16), sK_state_layout)
        cute.arch.barrier()
        if cutlass.const_expr(not STATE_SPLIT):
            _mma_state4_ldsm_reuse_b(
                tiled_mma, smem_tiled_copy_A_trans, smem_tiled_copy_B,
                thr_mma, tidx, sK_t0, sK_t1, sK_t2, sK_t3, sNK_A,
                state0, state1, state2, state3, BT // 16,
            )
        else:
            _mma_full_ldsm(
                tiled_mma, smem_tiled_copy_A_trans, smem_tiled_copy_B,
                thr_mma, tidx, sK_t0, sNK_A, state0, BT // 16,
            )
            _mma_full_ldsm(
                tiled_mma, smem_tiled_copy_A_trans, smem_tiled_copy_B,
                thr_mma, tidx, sK_t1, sNK_A, state1, BT // 16,
            )
            _mma_full_ldsm(
                tiled_mma, smem_tiled_copy_A_trans, smem_tiled_copy_B,
                thr_mma, tidx, sK_t2, sNK_A, state2, BT // 16,
            )
            _mma_full_ldsm(
                tiled_mma, smem_tiled_copy_A_trans, smem_tiled_copy_B,
                thr_mma, tidx, sK_t3, sNK_A, state3, BT // 16,
            )

    for idx in cutlass.range(cute.size(state0)):
        co = tCcC_bv[idx]
        k_row = co[0]
        v_col = v_off + co[1]
        mFinalState[i_b, i_hv, k_row, v_col] = state0[idx].to(mFinalState.element_type)
        mFinalState[i_b, i_hv, BT + k_row, v_col] = state1[idx].to(mFinalState.element_type)
        mFinalState[i_b, i_hv, 2 * BT + k_row, v_col] = state2[idx].to(mFinalState.element_type)
        mFinalState[i_b, i_hv, 3 * BT + k_row, v_col] = state3[idx].to(mFinalState.element_type)


@cute.jit
def _launch_kernel1_v31_final_state(
    mKnorm,
    mQnorm,
    mV,
    mBeta,
    mM,
    mGQK,
    mExpGC_in,
    mExpDecay_in,
    mO,
    mFinalState,
    scale: cutlass.Constexpr[float],
    B: cutlass.Int32,
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    NT: cutlass.Constexpr[int],
    H_PER_HV: cutlass.Constexpr[int],
    BV_TILE: cutlass.Constexpr[int],
    STATE_SPLIT: cutlass.Constexpr[bool],
    STATE_SPLIT_START_CHUNK: cutlass.Constexpr[int],
    STATE_PAD_ELEMS: cutlass.Constexpr[int],
    M_FOLDED_BETA: cutlass.Constexpr[bool],
):
    mma_op = warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
    tC = cute.make_layout((2, 2, 1))
    perm = (2 * 16, 2 * 8 * 1, 1 * 16)
    tiled_mma = cute.make_tiled_mma(mma_op, tC, permutation_mnk=perm)

    smem_copy_atom_A = cute.make_copy_atom(
        warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
        cutlass.BFloat16,
    )
    smem_copy_atom_A_trans = cute.make_copy_atom(
        warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
        cutlass.BFloat16,
    )
    smem_copy_atom_B = cute.make_copy_atom(
        warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
        cutlass.BFloat16,
    )
    smem_tiled_copy_A = cute.make_tiled_copy_A(smem_copy_atom_A, tiled_mma)
    smem_tiled_copy_A_trans = cute.make_tiled_copy_A(smem_copy_atom_A_trans, tiled_mma)
    smem_tiled_copy_B = cute.make_tiled_copy_B(smem_copy_atom_B, tiled_mma)

    copy_atom_state_r2s = cute.make_copy_atom(
        warp.StMatrix8x8x16bOp(transpose=True, num_matrices=2),
        cutlass.BFloat16,
    )
    tiled_copy_state_C = cute.make_tiled_copy_C_atom(copy_atom_state_r2s, tiled_mma)
    tiled_copy_state_r2s = cute.make_tiled_copy_S(copy_atom_state_r2s, tiled_copy_state_C)

    copy_atom_o_gmem = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        cutlass.BFloat16,
        num_bits_per_copy=32,
    )
    tiled_copy_o_gmem = cute.make_tiled_copy_C(copy_atom_o_gmem, tiled_mma)
    cp_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.BFloat16,
        num_bits_per_copy=128,
    )
    thr_layout_kq = cute.make_layout((32, 4), stride=(4, 1))
    val_layout_kq = cute.make_layout((1, 8), stride=(8, 1))
    tiled_copy_kq = cute.make_tiled_copy_tv(cp_atom, thr_layout_kq, val_layout_kq)

    thr_layout_mgqk = cute.make_layout((32, 4), stride=(4, 1))
    val_layout_mgqk = cute.make_layout((1, 8), stride=(8, 1))
    tiled_copy_mgqk = cute.make_tiled_copy_tv(cp_atom, thr_layout_mgqk, val_layout_mgqk)

    cp_atom_v = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.BFloat16,
        num_bits_per_copy=128,
    )
    thr_layout_v = cute.make_layout((32, 2), stride=(2, 1))
    val_layout_v = cute.make_layout((1, 8), stride=(8, 1))
    tiled_copy_v = cute.make_tiled_copy_tv(cp_atom_v, thr_layout_v, val_layout_v)

    KQ_STRIDE = K_DIM + 8
    if cutlass.const_expr(STATE_SPLIT):
        A_STRIDE = BT + 8
        GQK_STRIDE = BT + 8
    else:
        A_STRIDE = BT + 16
        GQK_STRIDE = BT + 16
    V_STRIDE = BV_TILE
    S_STRIDE = K_DIM + 8
    NK_STRIDE = BT + 8
    sK_l = cute.make_layout((BT, K_DIM), stride=(KQ_STRIDE, 1))
    sV_l = cute.make_layout((BT, BV_TILE), stride=(V_STRIDE, 1))
    sA_l = cute.make_layout((BT, BT), stride=(A_STRIDE, 1))
    sGQK_l = cute.make_layout((BT, BT), stride=(GQK_STRIDE, 1))
    sBeta_l = cute.make_layout((BT,), stride=(1,))
    sS_l = cute.make_layout((BV_TILE, K_DIM), stride=(S_STRIDE, 1))
    sNK_l = cute.make_layout((BV_TILE, BT), stride=(NK_STRIDE, 1))
    sExpGC_l = cute.make_layout((BT,), stride=(1,))
    sExpDecay_l = cute.make_layout((BT,), stride=(1,))

    beta_smem = 0 if M_FOLDED_BETA else BT * 4
    smem = (
        BT * KQ_STRIDE * 2 * 2 +
        BT * V_STRIDE * 2 +
        BT * A_STRIDE * 2 +
        BT * GQK_STRIDE * 2 +
        beta_smem +
        BV_TILE * S_STRIDE * 2 +
        STATE_PAD_ELEMS * 2 +
        BV_TILE * NK_STRIDE * 2 +
        BT * 4 +
        BT * 4
    )

    _fused_chunk_h_v31_final_state_kernel(
        tiled_mma,
        smem_tiled_copy_A,
        smem_tiled_copy_A_trans,
        smem_tiled_copy_B,
        tiled_copy_state_r2s,
        tiled_copy_o_gmem,
        tiled_copy_kq,
        tiled_copy_mgqk,
        tiled_copy_v,
        sK_l,
        sV_l,
        sA_l,
        sGQK_l,
        sBeta_l,
        sS_l,
        sNK_l,
        sExpGC_l,
        sExpDecay_l,
        mKnorm,
        mQnorm,
        mV,
        mBeta,
        mM,
        mGQK,
        mExpGC_in,
        mExpDecay_in,
        mO,
        mFinalState,
        scale,
        T,
        H,
        HV,
        V,
        NT,
        H_PER_HV,
        BV_TILE,
        STATE_SPLIT,
        STATE_SPLIT_START_CHUNK,
        STATE_PAD_ELEMS,
        M_FOLDED_BETA,
    ).launch(
        grid=(V // BV_TILE, B * HV, 1),
        block=(NUM_THREADS, 1, 1),
        smem=smem,
    )


@cute.jit
def _launch_kernel1_v31_final_state_bx2(
    mKnorm,
    mQnorm,
    mV,
    mBeta,
    mM,
    mGQK,
    mExpGC_in,
    mExpDecay_in,
    mO,
    mFinalState,
    scale: cutlass.Constexpr[float],
    B: cutlass.Int32,
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    NT: cutlass.Constexpr[int],
    H_PER_HV: cutlass.Constexpr[int],
    BV_TILE: cutlass.Constexpr[int],
    STATE_SPLIT: cutlass.Constexpr[bool],
    STATE_SPLIT_START_CHUNK: cutlass.Constexpr[int],
    STATE_PAD_ELEMS: cutlass.Constexpr[int],
    M_FOLDED_BETA: cutlass.Constexpr[bool],
):
    mma_op = warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
    tC = cute.make_layout((2, 2, 1))
    perm = (2 * 16, 2 * 8 * 1, 1 * 16)
    tiled_mma = cute.make_tiled_mma(mma_op, tC, permutation_mnk=perm)

    smem_copy_atom_A = cute.make_copy_atom(
        warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
        cutlass.BFloat16,
    )
    smem_copy_atom_A_trans = cute.make_copy_atom(
        warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
        cutlass.BFloat16,
    )
    smem_copy_atom_B = cute.make_copy_atom(
        warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=2),
        cutlass.BFloat16,
    )
    smem_tiled_copy_A = cute.make_tiled_copy_A(smem_copy_atom_A, tiled_mma)
    smem_tiled_copy_A_trans = cute.make_tiled_copy_A(smem_copy_atom_A_trans, tiled_mma)
    smem_tiled_copy_B = cute.make_tiled_copy_B(smem_copy_atom_B, tiled_mma)

    copy_atom_state_r2s = cute.make_copy_atom(
        warp.StMatrix8x8x16bOp(transpose=True, num_matrices=2),
        cutlass.BFloat16,
    )
    tiled_copy_state_C = cute.make_tiled_copy_C_atom(copy_atom_state_r2s, tiled_mma)
    tiled_copy_state_r2s = cute.make_tiled_copy_S(copy_atom_state_r2s, tiled_copy_state_C)

    copy_atom_o_gmem = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        cutlass.BFloat16,
        num_bits_per_copy=32,
    )
    tiled_copy_o_gmem = cute.make_tiled_copy_C(copy_atom_o_gmem, tiled_mma)
    cp_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.BFloat16,
        num_bits_per_copy=128,
    )
    thr_layout_kq = cute.make_layout((32, 4), stride=(4, 1))
    val_layout_kq = cute.make_layout((1, 8), stride=(8, 1))
    tiled_copy_kq = cute.make_tiled_copy_tv(cp_atom, thr_layout_kq, val_layout_kq)

    thr_layout_mgqk = cute.make_layout((32, 4), stride=(4, 1))
    val_layout_mgqk = cute.make_layout((1, 8), stride=(8, 1))
    tiled_copy_mgqk = cute.make_tiled_copy_tv(cp_atom, thr_layout_mgqk, val_layout_mgqk)

    cp_atom_v = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.BFloat16,
        num_bits_per_copy=128,
    )
    thr_layout_v = cute.make_layout((32, 2), stride=(2, 1))
    val_layout_v = cute.make_layout((1, 8), stride=(8, 1))
    tiled_copy_v = cute.make_tiled_copy_tv(cp_atom_v, thr_layout_v, val_layout_v)

    KQ_STRIDE = K_DIM + 8
    if cutlass.const_expr(STATE_SPLIT):
        A_STRIDE = BT + 8
        GQK_STRIDE = BT + 8
    else:
        A_STRIDE = BT + 16
        GQK_STRIDE = BT + 16
    V_STRIDE = BV_TILE
    S_STRIDE = K_DIM + 8
    NK_STRIDE = BT + 8
    sK_l = cute.make_layout((BT, K_DIM), stride=(KQ_STRIDE, 1))
    sV_l = cute.make_layout((BT, BV_TILE), stride=(V_STRIDE, 1))
    sA_l = cute.make_layout((BT, BT), stride=(A_STRIDE, 1))
    sGQK_l = cute.make_layout((BT, BT), stride=(GQK_STRIDE, 1))
    sBeta_l = cute.make_layout((BT,), stride=(1,))
    sS_l = cute.make_layout((BV_TILE, K_DIM), stride=(S_STRIDE, 1))
    sNK_l = cute.make_layout((BV_TILE, BT), stride=(NK_STRIDE, 1))
    sExpGC_l = cute.make_layout((BT,), stride=(1,))
    sExpDecay_l = cute.make_layout((BT,), stride=(1,))

    beta_smem = 0 if M_FOLDED_BETA else BT * 4
    smem = (
        BT * KQ_STRIDE * 2 * 2 +
        BT * V_STRIDE * 2 +
        BT * A_STRIDE * 2 +
        BT * GQK_STRIDE * 2 +
        beta_smem +
        BV_TILE * S_STRIDE * 2 +
        STATE_PAD_ELEMS * 2 +
        BV_TILE * NK_STRIDE * 2 +
        BT * 4 +
        BT * 4
    )

    _fused_chunk_h_v31_final_state_kernel(
        tiled_mma,
        smem_tiled_copy_A,
        smem_tiled_copy_A_trans,
        smem_tiled_copy_B,
        tiled_copy_state_r2s,
        tiled_copy_o_gmem,
        tiled_copy_kq,
        tiled_copy_mgqk,
        tiled_copy_v,
        sK_l,
        sV_l,
        sA_l,
        sGQK_l,
        sBeta_l,
        sS_l,
        sNK_l,
        sExpGC_l,
        sExpDecay_l,
        mKnorm,
        mQnorm,
        mV,
        mBeta,
        mM,
        mGQK,
        mExpGC_in,
        mExpDecay_in,
        mO,
        mFinalState,
        scale,
        T,
        H,
        HV,
        V,
        NT,
        H_PER_HV,
        BV_TILE,
        STATE_SPLIT,
        STATE_SPLIT_START_CHUNK,
        STATE_PAD_ELEMS,
        M_FOLDED_BETA,
    ).launch(
        grid=(V // BV_TILE, B * HV, 1),
        block=(NUM_THREADS, 1, 1),
        smem=smem,
    )


def _use_split_state_output(t: int) -> bool:
    return t >= 32768


def _split_state_start_chunk(t: int) -> int:
    if not _use_split_state_output(t):
        return 0
    return 0


def _state_layout_pad_elems(t: int) -> int:
    return 0


def _run_3kernel_v31_final_state(q, k, v, g, beta, scale):
    b, t, h, k_dim = q.shape
    _, _, hv, v_dim = v.shape
    nt = t // BT
    h_per_hv = hv // h

    qc = q.contiguous()
    kc = k.contiguous()
    vc = v.contiguous()
    gc = g.contiguous()
    betac = beta.contiguous()

    q_norm = torch.empty(b, h, t, k_dim, dtype=qc.dtype, device=qc.device)
    k_norm = torch.empty(b, h, t, k_dim, dtype=kc.dtype, device=kc.device)
    kk = torch.empty(b, nt, h, BT, BT, dtype=torch.bfloat16, device=q.device)
    qk = torch.empty(b, nt, h, BT, BT, dtype=torch.bfloat16, device=q.device)
    neumann_m = torch.empty(b, nt, hv, BT, BT, dtype=torch.bfloat16, device=q.device)
    gated_qk = torch.empty(b, nt, hv, BT, BT, dtype=torch.bfloat16, device=q.device)
    exp_gc = torch.empty(b, nt, hv, BT, dtype=torch.float32, device=q.device)
    exp_decay = torch.empty(b, nt, hv, BT, dtype=torch.float32, device=q.device)
    o = torch.empty(b, t, hv, v_dim, dtype=v.dtype, device=v.device)
    final_state = torch.empty(b, hv, k_dim, v_dim, dtype=torch.float32, device=v.device)
    state_split = _use_split_state_output(t)
    state_split_start_chunk = _split_state_start_chunk(t)
    state_pad_elems = _state_layout_pad_elems(t)
    use_fastmath = True

    mQ = from_dlpack(qc, assumed_align=16)
    mK = from_dlpack(kc, assumed_align=16)
    mV = from_dlpack(vc, assumed_align=16)
    mG = from_dlpack(gc, assumed_align=16)
    mBeta = from_dlpack(betac, assumed_align=16)
    mQnorm = from_dlpack(q_norm, assumed_align=16)
    mKnorm = from_dlpack(k_norm, assumed_align=16)
    mKK = from_dlpack(kk, assumed_align=16)
    mQK = from_dlpack(qk, assumed_align=16)
    mM = from_dlpack(neumann_m, assumed_align=16)
    mGQK = from_dlpack(gated_qk, assumed_align=16)
    mExpGC = from_dlpack(exp_gc, assumed_align=16)
    mExpDecay = from_dlpack(exp_decay, assumed_align=16)
    mO = from_dlpack(o, assumed_align=16)
    mFinalState = from_dlpack(final_state, assumed_align=16)

    key0 = (
        "pinv-k0-qk-bt32-kqstride136-v1",
        b, t, h, k_dim, q.dtype, use_fastmath,
    )
    compiled0 = _compiled_cache.get(key0)
    if compiled0 is None:
        compiled0 = cute.compile(
            launch_kernel0,
            mQ, mK, mQnorm, mKnorm, mKK, mQK,
            cutlass.Int32(b), t, h, nt, use_fastmath,
        )
        _compiled_cache[key0] = compiled0
    compiled0(mQ, mK, mQnorm, mKnorm, mKK, mQK, cutlass.Int32(b))

    key_inv = (
        "pinv-kinv-gqk-bt32-betafold-gqk-early-astride32-v1",
        b, t, h, hv, q.dtype, use_fastmath,
    )
    compiled_inv = _compiled_cache.get(key_inv)
    if compiled_inv is None:
        compiled_inv = cute.compile(
            launch_kernel_inv_betafold,
            mKK, mQK, mG, mBeta,
            mM, mGQK, mExpGC, mExpDecay,
            cutlass.Int32(b),
            t, h, hv, nt, h_per_hv, use_fastmath,
        )
        _compiled_cache[key_inv] = compiled_inv
    compiled_inv(
        mKK, mQK, mG, mBeta,
        mM, mGQK, mExpGC, mExpDecay,
        cutlass.Int32(b),
    )

    key1 = (
        "kernelpy-v112-k1-kq-scale-fused-bx2-launch-v1",
        b, t, h, hv, k_dim, v_dim, q.dtype, float(scale), state_split,
        state_split_start_chunk, state_pad_elems,
    )
    compiled1 = _compiled_cache.get(key1)
    if compiled1 is None:
        compiled1 = cute.compile(
            _launch_kernel1_v31_final_state_bx2,
            mKnorm, mQnorm, mV, mBeta,
            mM, mGQK, mExpGC, mExpDecay,
            mO, mFinalState,
            float(scale), cutlass.Int32(b),
            t, h, hv, v_dim, nt, h_per_hv, BV,
            state_split, state_split_start_chunk, state_pad_elems, True,
        )
        _compiled_cache[key1] = compiled1
    compiled1(
        mKnorm, mQnorm, mV, mBeta,
        mM, mGQK, mExpGC, mExpDecay,
        mO, mFinalState,
        cutlass.Int32(b),
    )

    return o, final_state


def _run_3kernel_v31_final_state_tail(q, k, v, g, beta, scale):
    b, t, h, k_dim = q.shape
    _, _, hv, v_dim = v.shape
    t_pad = ((t + BT - 1) // BT) * BT
    nt = t_pad // BT
    h_per_hv = hv // h

    qc = q.contiguous()
    kc = k.contiguous()
    vc = v.contiguous()
    gc = g.contiguous()
    betac = beta.contiguous()

    q_norm = torch.empty(b, h, t_pad, k_dim, dtype=qc.dtype, device=qc.device)
    k_norm = torch.empty(b, h, t_pad, k_dim, dtype=kc.dtype, device=kc.device)
    kk = torch.empty(b, nt, h, BT, BT, dtype=torch.bfloat16, device=q.device)
    qk = torch.empty(b, nt, h, BT, BT, dtype=torch.bfloat16, device=q.device)
    neumann_m = torch.empty(b, nt, hv, BT, BT, dtype=torch.bfloat16, device=q.device)
    gated_qk = torch.empty(b, nt, hv, BT, BT, dtype=torch.bfloat16, device=q.device)
    exp_gc = torch.empty(b, nt, hv, BT, dtype=torch.float32, device=q.device)
    exp_decay = torch.empty(b, nt, hv, BT, dtype=torch.float32, device=q.device)
    o = torch.empty(b, t, hv, v_dim, dtype=v.dtype, device=v.device)
    final_state = torch.empty(b, hv, k_dim, v_dim, dtype=torch.float32, device=v.device)
    state_split = _use_split_state_output(t)
    state_split_start_chunk = _split_state_start_chunk(t)
    state_pad_elems = _state_layout_pad_elems(t)

    mQ = from_dlpack(qc, assumed_align=16)
    mK = from_dlpack(kc, assumed_align=16)
    mV = from_dlpack(vc, assumed_align=16)
    mG = from_dlpack(gc, assumed_align=16)
    mBeta = from_dlpack(betac, assumed_align=16)
    mQnorm = from_dlpack(q_norm, assumed_align=16)
    mKnorm = from_dlpack(k_norm, assumed_align=16)
    mKK = from_dlpack(kk, assumed_align=16)
    mQK = from_dlpack(qk, assumed_align=16)
    mM = from_dlpack(neumann_m, assumed_align=16)
    mGQK = from_dlpack(gated_qk, assumed_align=16)
    mExpGC = from_dlpack(exp_gc, assumed_align=16)
    mExpDecay = from_dlpack(exp_decay, assumed_align=16)
    mO = from_dlpack(o, assumed_align=16)
    mFinalState = from_dlpack(final_state, assumed_align=16)

    key0 = ("tail-k0-qk-bt32", b, t, t_pad, h, k_dim, q.dtype)
    compiled0 = _compiled_cache.get(key0)
    if compiled0 is None:
        compiled0 = cute.compile(
            launch_kernel0_tail,
            mQ, mK, mQnorm, mKnorm, mKK, mQK,
            cutlass.Int32(b), t, h, nt,
        )
        _compiled_cache[key0] = compiled0
    compiled0(mQ, mK, mQnorm, mKnorm, mKK, mQK, cutlass.Int32(b))

    key_inv = ("tail-kinv-gqk-bt32", b, t, t_pad, h, hv, q.dtype)
    compiled_inv = _compiled_cache.get(key_inv)
    if compiled_inv is None:
        compiled_inv = cute.compile(
            launch_kernel_inv_tail,
            mKK, mQK, mG, mBeta,
            mM, mGQK, mExpGC, mExpDecay,
            cutlass.Int32(b),
            t, h, hv, nt, h_per_hv,
        )
        _compiled_cache[key_inv] = compiled_inv
    compiled_inv(
        mKK, mQK, mG, mBeta,
        mM, mGQK, mExpGC, mExpDecay,
        cutlass.Int32(b),
    )

    key1 = (
        "kernelpy-k1-v31-final-state-tail-v27",
        b, t, t_pad, h, hv, k_dim, v_dim, q.dtype, float(scale), state_split,
        state_split_start_chunk, state_pad_elems,
    )
    compiled1 = _compiled_cache.get(key1)
    if compiled1 is None:
        compiled1 = cute.compile(
            _launch_kernel1_v31_final_state,
            mKnorm, mQnorm, mV, mBeta,
            mM, mGQK, mExpGC, mExpDecay,
            mO, mFinalState,
            float(scale), cutlass.Int32(b),
            t, h, hv, v_dim, nt, h_per_hv, BV,
            state_split, state_split_start_chunk, state_pad_elems, False,
        )
        _compiled_cache[key1] = compiled1
    compiled1(
        mKnorm, mQnorm, mV, mBeta,
        mM, mGQK, mExpGC, mExpDecay,
        mO, mFinalState,
        cutlass.Int32(b),
    )

    return o, final_state


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
    cp_context=None,
    transpose_state_layout: bool = False,
    **kwargs,
):
    if scale is None:
        scale = 1.0 / math.sqrt(k.shape[-1])

    if not _is_supported_fast_path(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        cu_seqlens,
        cp_context,
        transpose_state_layout,
        kwargs,
    ):
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk_gated_delta_rule

        return fla_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            cp_context=cp_context,
            transpose_state_layout=transpose_state_layout,
            **kwargs,
        )

    t = q.shape[1]
    if t % BT != 0:
        o, final_state = _run_3kernel_v31_final_state_tail(q, k, v, g, beta, scale)
        if output_final_state:
            return o, final_state
        return o, None

    if output_final_state:
        o, final_state = _run_3kernel_v31_final_state(q, k, v, g, beta, scale)
        return o, final_state
    o = _run_3kernel_no_tail(q, k, v, g, beta, scale)
    return o, None


chunk_gdn = chunk_gated_delta_rule


def initialize_3kernel(
    B: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    T: int,
    scale: float,
    output_final_state: bool = True,
) -> None:
    """Compile and warm the selected GDN forward specialization.

    This is intended for model initialization. It uses disposable zero tensors
    with the target static shape so later forward calls hit the CuTeDSL cache
    and only launch the forward kernels.
    """
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = torch.bfloat16
    q = torch.zeros(B, T, H, K, device=device, dtype=dtype)
    k = torch.zeros(B, T, H, K, device=device, dtype=dtype)
    v = torch.zeros(B, T, HV, V, device=device, dtype=dtype)
    g = torch.zeros(B, T, HV, device=device, dtype=dtype)
    beta = torch.zeros(B, T, HV, device=device, dtype=dtype)

    with torch.inference_mode():
        chunk_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            scale=float(scale),
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=True,
        )
    torch.cuda.synchronize(device)


# Backward-compatible src API. The previous implementation exposed an eager
# compile context; kernels compile lazily per shape through the CuTeDSL cache.
def compile_3kernel(B: int, H: int, HV: int, K: int, V: int, bv_tile: int = BV) -> dict:
    if B != 1:
        raise ValueError(f"only B=1 is supported, got B={B}")
    if K != K_DIM:
        raise ValueError(f"only K={K_DIM} is supported, got K={K}")
    if V % bv_tile != 0:
        raise ValueError(f"V={V} must be divisible by bv_tile={bv_tile}")
    if HV % H != 0:
        raise ValueError(f"HV must be divisible by H, got HV={HV}, H={H}")
    return {
        "B": int(B),
        "H": int(H),
        "HV": int(HV),
        "K": int(K),
        "V": int(V),
        "BT": BT,
        "BV": int(bv_tile),
    }


def run_3kernel_precompiled(compiled: dict, q, k, v, g, beta, scale, return_final_state: bool = False):
    expected = (compiled["B"], compiled["H"], compiled["HV"], compiled["K"], compiled["V"])
    B, _, H, K = q.shape
    _, _, HV, V = v.shape
    actual = (B, H, HV, K, V)
    if actual != expected:
        raise ValueError(f"shape config mismatch: expected B/H/HV/K/V={expected}, got {actual}")
    result = chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        scale=scale,
        output_final_state=return_final_state,
        use_qk_l2norm_in_kernel=True,
    )
    return result if return_final_state else result[0]


def run_3kernel(q, k, v, g, beta, scale, compiled=None):
    if compiled is not None:
        return run_3kernel_precompiled(compiled, q, k, v, g, beta, scale, False)
    o, _ = chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        scale=scale,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    return o


def run_3kernel_with_final_state(q, k, v, g, beta, scale, compiled=None):
    if compiled is not None:
        return run_3kernel_precompiled(compiled, q, k, v, g, beta, scale, True)
    return chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        scale=scale,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
