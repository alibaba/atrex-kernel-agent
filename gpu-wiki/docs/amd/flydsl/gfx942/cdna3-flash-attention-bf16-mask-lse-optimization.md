# FlyDSL Flash Attention Forward (bf16, mask+LSE) on MI308X — V8-V10

Applicability: backend: flydsl; hardware: amd; topic: reference


**Last updated**: 2026-06-30

Continuation of the [V0-V7 mask optimization journey](cdna3-flash-attention-bf16-mask-optimization.md)
(50.6 TFLOPS → **71.8 TFLOPS**, +42%).

## Target hardware

- AMD Instinct MI308X (CDNA3, gfx942)
- 80 CUs, ~1.4 GHz sclk, bf16 MFMA peak 206 TFLOPS
- HBM3: 5.3 TB/s bandwidth
- LDS: 64 KB per CU, VGPR: 512 per SIMD (128 per wave at occupancy 4)
- Specs sourced from `<gpu-wiki>/docs/hardware/mi308x_spec.md`

## Workload

- B=1024, H=8, S=316, D=64, bf16
- Arbitrary per-batch binary mask (packed to u32 bitmask)
- **NEW**: Synchronous LSE (log-sum-exp) output for training backward pass
- FLOPs: 0.2094 T (4×B×H×S²×D)
- 70 TFLOPS target → 2.99 ms budget

## Baseline (V7)

V7 kernel (bit-packed u32 mask): **4.14 ms @ 50.6 TFLOPS**
- vgpr_count=123, LDS=17,152 bytes, occupancy=3 waves/SIMD
- K and V occupy separate LDS regions (8,448 + 8,704 = 17,152 bytes)
- Bottleneck: s_nop hazard padding ~26%, VALU overhead from bit extraction

## Kernel resource footprint (final, V10)

| Resource | V7 | V10 | Change |
|----------|-----|------|--------|
| VGPRs | 123 | 128 | +5 (controlled by waves_per_eu=4) |
| LDS | 17,152 B | 8,704 B | -49% (SHARE_KV_LDS) |
| Occupancy | 3 waves/SIMD | 4 waves/SIMD | +33% |
| flat_work_group_size | 256 | 256 | same |

## Optimization journey

### V8 — Shared K/V LDS + LSE output

**What**: K and V never overlap in time within the inner loop (K consumed by GEMM1 before V is loaded for GEMM2). Sharing their LDS region halves the footprint from 17,152→8,704 bytes. Added `gpu.barrier()` after GEMM1 to ensure all waves finish reading K before V load overwrites the same LDS address. Also added LSE output tensor and `seq_stride_o` parameter for separate output stride.

**Why**: At occupancy=3, LDS was the binding constraint (17,152 × 3 = 51,456 > 49,152 limit for 4 WGs). Reducing to 8,704 bytes removes the LDS bottleneck (8,704 × 4 = 34,816 < 64 KB).

**Result**: LDS reduced to 8,704 bytes but VGPRs increased from 123→130 (compiler used freed budget for registers). Occupancy stayed at 3. Performance: **64.9 TFLOPS** (V7 was 50.6 — but the gain is from baseline improvements + LSE, not from occupancy yet).

**Critical constraint**: `FLYDSL_FLASH_ATTN_FUNC_K_INTERBLOCK` must be `0`. K interblock prefetch loads next iteration's K into the shared LDS while current V is still stored there, causing silent correctness corruption (rel_err 5% instead of 1.4%). See pitfall trap 46.

### V9 — v_pk_fma_f32 packed softmax

**What**: Replaced 32 scalar `math.fma(score, scale_log2e, neg_max)` calls with 16 packed `v_pk_fma_f32` inline asm instructions. Each v_pk_fma_f32 computes two FMAs in one VALU cycle on CDNA3.

**Why**: Softmax scaling is pure VALU work that competes with MFMA for pipeline slots. Halving the VALU count frees cycles for mask extraction and exp2.

**Result**: ~2% improvement. Note: attempted extending pk_fma to pk_add for sum accumulation, but v_pk_add_f32 chained with pk_fma produced systematically wrong results (25.7% error). Only pk_fma alone is reliable. See pitfall trap 47.

### V10 — waves_per_eu=4 occupancy boost

**What**: Set `waves_per_eu=4` in the kernel build, which tells the AMDGPU compiler to target ≤128 VGPRs per wave (down from the 130 the compiler chose in V8). This is the final piece needed to realize occupancy=4.

**Why**: V8 reduced LDS enough for 4 WGs/CU, but the compiler expanded VGPRs to 130 (needs ≤128 for 4 waves/SIMD with 512 VGPRs total). `waves_per_eu` is a compiler hint that constrains register allocation.

**Result**: VGPRs=128, occupancy=4 waves/SIMD. **2.99 ms @ 70.0 TFLOPS** (E2E with host overhead). With BHSD native layout avoiding transpose: **2.92 ms @ 71.8 TFLOPS**.

### BHSD native layout (host-side optimization)

**What**: Added `layout="BHSD"` parameter to the kernel builder. When set, the kernel reads Q/K/V directly in BHSD layout and writes output with a separate `seq_stride_o` parameter, eliminating the need for BHSD→BSHD pad+transpose (3 fused torch.compile kernels) and BSHD→BHSD unpad+transpose (1 fused kernel).

**Result**: ~0.3 ms saved on host side for B=1024 H=8 S=316 D=64. Combined with kernel improvements: **2.92 ms → 71.8 TFLOPS** E2E.

## Failed attempts

| Attempt | Expected | Actual | Why |
|---------|----------|--------|-----|
| v_pk_add_f32 for sum | -16 VALU | 25.7% error, NaN | MLIR lowering issue with chained v2f32 inline asm (trap 47) |
| Remove barriers (GEMM1/mask/max) | Reduce stalls | 64.9→59.8 TFLOPS (-8%) | Barriers keep MFMAs back-to-back; removal fragments pipeline (trap 48) |
| BLOCK_M=64 | Better tail handling | 50.2 TFLOPS | Occupancy 3→2 (more VGPRs/thread), doubled KV iterations (trap 49) |
| l_final clamping in epilogue | Avoid div-by-zero | NaN output | Combined with double-masking in softmax exp2, clamping triggers NaN (trap 51) |
| Double-masking in exp2 | Extra safety | NaN output | BFI mask already applies penalty; HAS_MASK conditional applies it again (trap 51) |

## Final perf vs baseline

| Version | Time (ms) | TFLOPS | vs V7 | Occupancy |
|---------|-----------|--------|-------|-----------|
| V7 (bit-packed mask) | 4.14 | 50.6 | 1.00x | 3 |
| V8 (SHARE_KV_LDS + LSE) | 3.23 | 64.9 | 1.28x | 3 |
| V10 (waves_per_eu=4) | 2.99 | 70.0 | 1.38x | 4 |
| V10 + BHSD layout | 2.92 | 71.8 | 1.42x | 4 |
| PyTorch SDPA | 8.39 | 25.0 | — | — |

## Remaining bottlenecks

Based on rocprofv3 ISA analysis of the V10 compiled kernel:

1. **s_nop hazard padding**: ~20% of total cycles. VCC serialization from BFI mask extraction (v_bfe_i32 → v_cmp → v_cndmask) forces 2-cycle stalls per position. Partially hidden by MFMA drain.
2. **Small problem size**: S=316 padded to 320 gives only 5 KV iterations and 3 Q tiles. Low tile count limits CU utilization across SEs.
3. **MFMA utilization ceiling**: At occupancy=4, MFMA pipeline is well-fed but mask VALU competes for issue slots during MFMA latency bubbles.

## What would close the remaining gap

1. **BFI mask instruction**: Replace AND+CMP+CNDMASK (3 VALU + VCC) with a single bit-field-insert that directly produces the -1e6 penalty, eliminating VCC serialization.
2. **Larger problem sizes**: S≥512 would give more KV iterations and better amortization of per-tile overhead.
3. **CK V3–style ISA scheduling**: `CoreLoopScheduler` or equivalent cycle-level manual scheduling could recover 5-10% by eliminating unnecessary s_nop padding.

## Sustained recipe (do these, in this order)

1. Start from the bit-packed mask kernel (V7 baseline)
2. Enable SHARE_KV_LDS: set K and V LDS base to offset 0, add barrier after GEMM1
3. **Disable K_INTERBLOCK** (`FLYDSL_FLASH_ATTN_FUNC_K_INTERBLOCK=0`) — non-negotiable for correctness
4. Add LSE output tensor and `seq_stride_o` parameter for flexible output stride
5. Replace scalar softmax FMAs with v_pk_fma_f32 (16 packed ops for 32 positions)
6. Set `waves_per_eu=4` to force ≤128 VGPRs and achieve occupancy=4
7. Add BHSD native layout to avoid transpose overhead
8. Verify correctness: rel_err < 0.02 for output, rel_err ≈ 0 for LSE

## Related docs

- [V0-V7 mask optimization journey](cdna3-flash-attention-bf16-mask-optimization.md)
- [Causal+GQA optimization journey](cdna3-flash-attention-bf16-gqa-optimization.md)
- [Pitfalls](../pitfalls/flash-attn-pitfalls.md) (traps 46-51: mask+LSE V8-V10)
- [Reference kernel](../../../../reference-kernels/amd/cdna3/flydsl/FlyDSL/flash_attn_func_mask_mi308x.py) (V10 final)
- V7 backup: `flash_attn_func_mask_mi308x.py.v7.bak`
