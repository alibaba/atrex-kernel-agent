"""
================================================================================
⚠  TUNED FOR `sm_120` (NVIDIA RTX PRO 5000 / 4000 Blackwell, GeForce Blackwell)
================================================================================

CuteDSL Gated DeltaNet (GDN) `fused_recurrent_gated_delta_rule_fwd` decode kernel,
T=1, fp32 state, bf16 q/k/v, with L2-norm + scalar gate + delta rule fused.

Final winning version (V13/18): cp.async + LoadCacheMode.GLOBAL + assumed_align=16.
Wall-clock matches FLA Triton at B>=32 (B=64: 246us vs FLA 247us).
Memory throughput 1.04 TB/s = 100.8% of D2D memcpy ceiling on Pro5000.

Differences vs hopper / sm_90 GDN reference (e.g. flashinfer/gdn_decode_*.py):
  - sm_120 has NO tcgen05/TMEM — must use Hopper TMA + Ampere warp ALU
  - state stored as fp32 (not bf16) — matches FLA fp32 final_state contract
  - cp.async cache_mode=GLOBAL is REQUIRED to avoid L2 false-saturation
  - per-thread V slice = 4 fp32 (16B aligned) — V_PER_WARP=4, NUM_WARPS=2

⚠ Critical setup:
  - `from_dlpack(h0, assumed_align=16)` else cp.async cp_size=128b fails align check
  - thread layout `(32,2)/(4,4)` — each thread takes 4K rows × 4V cols (16B vec)
  - SMEM staging: cp.async → SMEM → register read for compute

References:
  - Optimization journey:
    docs/nvidia/blackwell-geforce/ref-docs/cutedsl/sm120-gdn-decode-fp32state-bf16qkv-optimization.md
  - Pitfalls (18 versions iterated, 9+ traps documented):
    docs/nvidia/blackwell-geforce/pitfalls/cutedsl/gdn-decode-pitfalls.md
  - Oracle: fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule (PyTorch reference in reference.py)
  - Hopper bf16-state variant (different chip + dtype):
    reference-kernels/nvidia/hopper/cutedsl/flashinfer/gdn_decode_*.py
"""
from __future__ import annotations
import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack

BK_C = 128
BV_C = 8
K_PER_T_C = 4
V_PER_WARP = 4
NUM_WARPS = 2
NUM_THREADS = NUM_WARPS * 32  # 64


@cute.kernel
def gdn_fwd_T1_v13_kernel(
    tiled_copy_h0: cute.TiledCopy,
    smem_layout_h0: cute.Layout,
    mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor,
    mG: cute.Tensor, mBeta: cute.Tensor,
    mH0: cute.Tensor, mO: cute.Tensor, mHt: cute.Tensor,
    scale: cutlass.Constexpr[float],
    H_per_HV: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    bid_v = cute.arch.block_idx()[0]
    bid_nh = cute.arch.block_idx()[1]
    n  = bid_nh // HV
    hv = bid_nh % HV
    h  = hv // H_per_HV
    lane_id = tidx % 32
    warp_id = tidx // 32

    smem = cutlass.utils.SmemAllocator()
    sH0 = smem.allocate_tensor(cutlass.Float32, smem_layout_h0, 16)

    v_tile_id = bid_v * NUM_WARPS + warp_id
    v_global_base = v_tile_id * V_PER_WARP

    # --- Issue cp.async G2S with GLOBAL cache mode ---
    gH0_full = mH0[(n, hv, None, None)]  # rank-2 (K, V)
    gH0 = cute.local_tile(gH0_full, (BK_C, BV_C), (0, bid_v))
    thr_copy = tiled_copy_h0.get_slice(tidx)
    thr_gH0 = thr_copy.partition_S(gH0)
    thr_sH0 = thr_copy.partition_D(sH0)
    cute.copy(tiled_copy_h0, thr_gH0, thr_sH0)
    cute.arch.cp_async_commit_group()

    # --- Concurrently: load q,k,v + L2 norm (latency hiding) ---
    r_q_bf16 = cute.make_rmem_tensor(cute.make_layout((K_PER_T_C,), stride=(1,)), cutlass.BFloat16)
    r_k_bf16 = cute.make_rmem_tensor(cute.make_layout((K_PER_T_C,), stride=(1,)), cutlass.BFloat16)
    cute.autovec_copy(cute.local_tile(mQ, (1, 1, K_PER_T_C), (n, h, lane_id)), r_q_bf16)
    cute.autovec_copy(cute.local_tile(mK, (1, 1, K_PER_T_C), (n, h, lane_id)), r_k_bf16)
    r_q = cute.make_rmem_tensor(cute.make_layout((K_PER_T_C,), stride=(1,)), cutlass.Float32)
    r_k = cute.make_rmem_tensor(cute.make_layout((K_PER_T_C,), stride=(1,)), cutlass.Float32)
    sum_q = cutlass.Float32(0.0); sum_k = cutlass.Float32(0.0)
    for ki in cutlass.range_constexpr(K_PER_T_C):
        r_q[ki] = cutlass.Float32(r_q_bf16[ki])
        r_k[ki] = cutlass.Float32(r_k_bf16[ki])
        sum_q = sum_q + r_q[ki] * r_q[ki]
        sum_k = sum_k + r_k[ki] * r_k[ki]
    for offset in [16, 8, 4, 2, 1]:
        sum_q = sum_q + cute.arch.shuffle_sync_bfly(sum_q, offset=offset, mask=-1, mask_and_clamp=31)
        sum_k = sum_k + cute.arch.shuffle_sync_bfly(sum_k, offset=offset, mask=-1, mask_and_clamp=31)
    inv_q = cute.rsqrt(sum_q + cutlass.Float32(1e-6))
    inv_k = cute.rsqrt(sum_k + cutlass.Float32(1e-6))
    for ki in cutlass.range_constexpr(K_PER_T_C):
        r_q[ki] = r_q[ki] * inv_q * scale
        r_k[ki] = r_k[ki] * inv_k

    r_v_bf16 = cute.make_rmem_tensor(cute.make_layout((V_PER_WARP,), stride=(1,)), cutlass.BFloat16)
    cute.autovec_copy(cute.local_tile(mV, (1, 1, V_PER_WARP), (n, hv, v_tile_id)), r_v_bf16)

    eg = cute.exp(mG[n, hv])
    beta_val = mBeta[n, hv]

    # --- Wait for cp.async ---
    cute.arch.cp_async_wait_group(0)
    cute.arch.barrier()

    # --- Read state from SMEM (cheap), apply gate ---
    state = cute.make_rmem_tensor(
        cute.make_layout((K_PER_T_C, V_PER_WARP), stride=(V_PER_WARP, 1)),
        cutlass.Float32,
    )
    k_off = lane_id * K_PER_T_C
    v_smem_base = warp_id * V_PER_WARP
    for ki in cutlass.range_constexpr(K_PER_T_C):
        for vi in cutlass.range_constexpr(V_PER_WARP):
            state[ki, vi] = sH0[k_off + ki, v_smem_base + vi] * eg

    r_buf = cute.make_rmem_tensor(cute.make_layout((V_PER_WARP,), stride=(1,)), cutlass.Float32)
    for vi in cutlass.range_constexpr(V_PER_WARP):
        partial = cutlass.Float32(0.0)
        for ki in cutlass.range_constexpr(K_PER_T_C):
            partial = partial + state[ki, vi] * r_k[ki]
        for offset in [16, 8, 4, 2, 1]:
            partial = partial + cute.arch.shuffle_sync_bfly(partial, offset=offset, mask=-1, mask_and_clamp=31)
        r_buf[vi] = beta_val * (cutlass.Float32(r_v_bf16[vi]) - partial)

    for ki in cutlass.range_constexpr(K_PER_T_C):
        for vi in cutlass.range_constexpr(V_PER_WARP):
            state[ki, vi] = state[ki, vi] + r_k[ki] * r_buf[vi]

    for vi in cutlass.range_constexpr(V_PER_WARP):
        partial = cutlass.Float32(0.0)
        for ki in cutlass.range_constexpr(K_PER_T_C):
            partial = partial + state[ki, vi] * r_q[ki]
        for offset in [16, 8, 4, 2, 1]:
            partial = partial + cute.arch.shuffle_sync_bfly(partial, offset=offset, mask=-1, mask_and_clamp=31)
        r_buf[vi] = partial

    if lane_id == 0:
        for vi in cutlass.range_constexpr(V_PER_WARP):
            mO[n, hv, v_global_base + vi] = r_buf[vi].to(mO.element_type)

    # --- ht store (regular autovec, no cache hint) ---
    ht_thr_tile = cute.local_tile(
        mHt, (1, 1, K_PER_T_C, V_PER_WARP),
        (n, hv, lane_id, v_tile_id),
    )
    cute.autovec_copy(state, ht_thr_tile)


@cute.jit
def gdn_fwd_T1_v13_launch(
    mQ, mK, mV, mG, mBeta, mH0, mO, mHt,
    scale: cutlass.Constexpr[float],
    H_per_HV: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    N: cutlass.Int32, NV: cutlass.Int32,
):
    smem_layout_h0 = cute.make_layout((BK_C, BV_C), stride=(BV_C, 1))
    # cp.async with LoadCacheMode.GLOBAL, vec=128b (16B = 4 fp32)
    cp_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.Float32,
        num_bits_per_copy=128,
    )
    # Layout: 32 K-threads × 2 V-threads = 64 (= NUM_THREADS)
    # Each thread takes 4 K rows × 4 V cols = 16 elements (4 vec=4 loads internally)
    # Total CTA: 32*2*4*4 = 1024 = full BK*BV ✓
    thr_layout = cute.make_layout((32, 2), stride=(2, 1))
    val_layout = cute.make_layout((4, 4), stride=(4, 1))
    tiled_copy = cute.make_tiled_copy_tv(cp_atom, thr_layout, val_layout)

    smem_size = cute.size_in_bytes(cutlass.Float32, smem_layout_h0)
    gdn_fwd_T1_v13_kernel(
        tiled_copy, smem_layout_h0,
        mQ, mK, mV, mG, mBeta, mH0, mO, mHt,
        scale, H_per_HV, HV,
    ).launch(
        grid=(NV, N * HV, 1),
        block=(NUM_THREADS, 1, 1),
        smem=smem_size,
    )


_compiled_cache: dict = {}


def run_gdn_fwd_T1_v13(q, k, v, g, beta, h0, scale):
    assert q.shape[1] == 1
    B, T, H, K = q.shape
    _, _, HV, V = v.shape
    NV = V // BV_C
    H_per_HV = HV // H

    q2 = q.squeeze(1).contiguous()
    k2 = k.squeeze(1).contiguous()
    v2 = v.squeeze(1).contiguous()
    g2 = g.squeeze(1).contiguous()
    beta2 = beta.squeeze(1).contiguous()
    o = torch.empty(B, HV, V, dtype=v.dtype, device=v.device)
    ht = torch.empty(B, HV, K, V, dtype=torch.float32, device=v.device)

    # H0/Ht: declare 16-byte alignment (PyTorch CUDA tensors are 256-byte aligned)
    mQ, mK, mV = from_dlpack(q2), from_dlpack(k2), from_dlpack(v2)
    mG, mBeta = from_dlpack(g2), from_dlpack(beta2)
    mH0 = from_dlpack(h0.contiguous(), assumed_align=16)
    mO = from_dlpack(o)
    mHt = from_dlpack(ht, assumed_align=16)

    key = (B, H, HV, K, V, q.dtype, v.dtype, float(scale))
    compiled = _compiled_cache.get(key)
    if compiled is None:
        compiled = cute.compile(
            gdn_fwd_T1_v13_launch,
            mQ, mK, mV, mG, mBeta, mH0, mO, mHt,
            float(scale), H_per_HV, HV,
            cutlass.Int32(B), cutlass.Int32(NV),
        )
        _compiled_cache[key] = compiled
    compiled(
        mQ, mK, mV, mG, mBeta, mH0, mO, mHt,
        float(scale), H_per_HV, HV,
        cutlass.Int32(B), cutlass.Int32(NV),
    )
    return o.unsqueeze(1), ht
