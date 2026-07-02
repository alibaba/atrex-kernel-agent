# FlashAttention-4


**Last updated**: 2026-06-30

## Overview

FlashAttention-4 is the Blackwell-native evolution of the FlashAttention family, designed to exploit SM100 architectural features that break the Hopper-era bottleneck: tensor core throughput doubles on Blackwell, but SFU (Special Function Unit) count and shared memory bandwidth remain unchanged. FA4 addresses this asymmetry through software-emulated exponentials, ping-pong scheduling across TMEM, and 2-CTA cooperative backward passes.

Written entirely in CuTe DSL (Python), FA4 compiles 20-30x faster than equivalent CUTLASS C++ template code while matching or exceeding cuDNN performance.

## Key Techniques

### Ping-Pong Scheduling

Two 128-token query tiles are assigned to each CTA. While one tile's matmul runs on the tensor cores, the other tile's softmax rescaling runs on dedicated warpgroups accessing TMEM. This hides the softmax latency behind MMA compute.

```python
# CuTe DSL: Ping-pong tile scheduling (simplified)
# Two query tiles per CTA, alternating MMA and softmax phases

@cute.kernel
def flash_attention_4_fwd(Q, K, V, O, L):
    # Each CTA processes 2 query tiles of 128 tokens
    TILE_Q = 128
    NUM_TILES = 2

    # Phase A: tile_0 does MMA(Q0, K), tile_1 does softmax rescale
    # Phase B: tile_1 does MMA(Q1, K), tile_0 does softmax rescale

    for kv_block in range(num_kv_blocks):
        # Ping: MMA on tile 0, softmax on tile 1
        with warpgroup(mma_wg):
            S0 = cute.mma(Q_tile[0], K_block)  # tcgen05.mma -> TMEM
        with warpgroup(softmax_wg):
            O_tile[1], L_tile[1] = rescale_and_accumulate(
                O_tile[1], L_tile[1], S1, V_prev
            )

        # Pong: MMA on tile 1, softmax on tile 0
        with warpgroup(mma_wg):
            S1 = cute.mma(Q_tile[1], K_block)
        with warpgroup(softmax_wg):
            O_tile[0], L_tile[0] = rescale_and_accumulate(
                O_tile[0], L_tile[0], S0, V_block
            )
```

### Software-Emulated Exponential (Cody-Waite)

The SFU `ex2` instruction is the bottleneck on Blackwell -- its throughput does not scale with the doubled tensor core rate. FA4 replaces it with a software exponential distributed across FMA units using Cody-Waite range reduction and Horner polynomial evaluation.

```python
# Software exp2 via Cody-Waite range reduction + Horner polynomial
# Distributes across FMA units instead of using scarce SFU hardware

def software_exp2(x):
    """
    Compute 2^x using FMA units instead of SFU ex2.
    Cody-Waite range reduction splits x into integer + fraction.
    Horner polynomial approximates 2^frac.
    """
    # Range reduction: x = n + f, where n is integer, f in [-0.5, 0.5]
    n = round(x)
    f = x - n  # Cody-Waite: use extended precision subtraction

    # Horner polynomial for 2^f on [-0.5, 0.5]
    # Coefficients chosen for bf16 precision target
    p = f * (C5 + f * (C4 + f * (C3 + f * (C2 + f * C1))))
    p = 1.0 + p

    # Reconstruct: 2^x = 2^n * 2^f
    return ldexp(p, n)  # Integer exponent via bit manipulation
```

This gives roughly 4x throughput improvement over the hardware SFU path by utilizing FMA units that would otherwise be idle during softmax phases.

### Conditional Softmax Rescaling

Standard FlashAttention rescales the output accumulator every KV block. FA4 only rescales when the running maximum changes significantly (large jump), reducing non-matmul operations.

```python
# Only rescale when max changes substantially
def conditional_rescale(O_acc, lse_old, lse_new, threshold=2.0):
    diff = lse_new - lse_old
    if abs(diff) > threshold:
        # Full rescale: O_acc *= exp(lse_old - lse_new)
        scale = software_exp2((lse_old - lse_new) * LOG2E)
        O_acc = O_acc * scale
    # Otherwise: skip rescale, accumulate normally
    return O_acc
```

### 2-CTA Backward Pass

The backward pass spans two paired CTAs in a cluster, sharing TMEM across both SMs. This halves shared memory traffic for the dQ/dK/dV gradient computation.

```python
# 2-CTA backward: paired CTAs share TMEM via 2-SM cooperative mode
@cute.kernel
def flash_attention_4_bwd(Q, K, V, O, dO, dQ, dK, dV):
    # Two CTAs cooperate: CTA_0 and CTA_1 in same cluster
    # tcgen05.mma shape: m256 x n256 x k16 (2-SM cooperative)

    cta_id = cute.cluster_rank()  # 0 or 1

    for q_block in range(num_q_blocks):
        # Both CTAs load shared KV block via TMA
        K_smem = tma_load(K, kv_offset)
        V_smem = tma_load(V, kv_offset)

        # CTA_0: compute dK contribution
        # CTA_1: compute dV contribution
        if cta_id == 0:
            # dK += dS^T @ Q  (accumulated in TMEM)
            dS = compute_dS(O, dO, S)
            cute.mma(dS.T, Q_tile, accumulator=dK_tmem)
        else:
            # dV += S^T @ dO  (accumulated in TMEM)
            cute.mma(S.T, dO_tile, accumulator=dV_tmem)
```

## Performance

| Configuration | GPU | Dtype | TFLOPS | Utilization | vs cuDNN | vs Triton |
|---------------|-----|-------|--------|-------------|----------|-----------|
| seqlen=8192, headdim=128 | B200 | BF16 | 1605 | 71% | 1.1-1.3x | 2.1-2.7x |

The 71% MMA utilization represents the state of the art for attention kernels on Blackwell. The remaining 29% is consumed by softmax, rescaling, and memory transfers.

## Implementation Notes

- Written entirely in CuTe DSL (Python), not C++ templates
- 20-30x faster compilation than equivalent CUTLASS C++ code
- Uses `SM100_MMA_SS` atoms for tcgen05 MMA from shared memory
- TMEM locality via `TMEM` locale in CuTe layout
- TMA bulk loads for Q, K, V tiles into shared memory

## When to Use

- Standard multi-head attention on Blackwell with sequence lengths >= 1024
- Both forward and backward passes
- BF16 precision (FP8 support planned)

## Caveats

- SM100 only -- no fallback to SM90
- Requires CuTe DSL toolchain (CUTLASS 4.5.0 + Python frontend)
- Ping-pong scheduling most effective for headdim=128; smaller headdims may not fully overlap

## Related

- [FA4 Kernel Architecture (roles / barriers / TMEM)](../articles/flash-attention-4-kernel-architecture.md) — end-to-end decomposition of the warp roles, barrier graph, and shared-TMEM layout
- [FA4 Warp Specialization & 2-CTA](../articles/flash-attention-4-warp-specialization-2cta.md)
- [FA4 Source Code Analysis — Part 1](../articles/flash-attention-4-source-code-analysis-part1.md)

## Sources

- [FlashAttention-4 paper](https://arxiv.org/abs/2603.05451)
- [Tri Dao's blog](https://tridao.me/blog/2026/flash4/)
