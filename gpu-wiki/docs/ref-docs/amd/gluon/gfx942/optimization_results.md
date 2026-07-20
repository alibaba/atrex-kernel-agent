---
pattern: fused_attention
type: results
pitfalls: [24-31, 38-40]
---

# Flash Attention Optimization Results Summary

## Causal Attention Optimization Results (fp16, MI308X, BLOCK_M=128, BLOCK_N=64, DIM=64)

| Configuration | Triton (TFLOPS) | Gluon-v1 (TFLOPS) | Gluon-v2 (TFLOPS) | v2/Triton | v2/v1 Improvement |
|------|----------------|-------------------|-------------------|-----------|-----------|
| B=2 H=16 S=1024 | 30.33 | 42.34 | **47.96** | **158%** | +13% |
| B=2 H=16 S=2048 | 38.08 | 53.40 | **57.05** | **150%** | +7% |
| B=4 H=16 S=2048 | 41.05 | 58.74 | **66.79** | **163%** | +14% |
| B=2 H=32 S=4096 | 56.22 | 78.88 | **89.85** | **160%** | +14% |

## Non-Causal Attention Optimization Results (fp16, MI308X, BLOCK_M=128, BLOCK_N=64, DIM=64)

| Configuration | Triton (TFLOPS) | Gluon-v1 (TFLOPS) | Gluon-v2 (TFLOPS) | v2/Triton | v2/v1 Improvement |
|------|----------------|-------------------|-------------------|-----------|-----------|
| B=2 H=16 S=1024 | 96.80 | 97.34 | **100.78** | **104%** | +4% |
| B=4 H=16 S=1024 | 109.81 | 115.11 | **127.00** | **116%** | +10% |
| B=4 H=16 S=2048 | 133.91 | 135.34 | **151.85** | **113%** | +12% |
| B=2 H=32 S=4096 | 146.90 | 141.45 | **159.17** | **108%** | +13% |
| B=4 H=32 S=4096 | 147.94 | 142.31 | **161.32** | **109%** | +13% |
| B=2 H=32 S=8192 | 151.08 | 144.18 | **162.67** | **108%** | +13% |

## v1 → v2 Key Change List

| Change | v1 | v2 | Performance Impact |
|--------|----|----|---------|
| Full/Masked blocks separation | ❌ None | ✅ Yes | +20% |
| V path | ❌ Via smem | ✅ Via convert_layout | +12% |
| K smem type | ❌ Persistent smem (`depth=1`) | ✅ On-the-fly smem (`value=`) | +5% |
| Zigzag remap | ❌ None | ✅ start_m × batch-head zigzag | +15% |
| tl.assume hint | ❌ None | ✅ 16 stride hints | +3% |
| Division optimization | ❌ Division | ✅ Reciprocal multiplication | +4% |
| Pointer arithmetic | ❌ Offset accumulation | ✅ Pointer arithmetic | +2% |

## Key Conclusions

1. **Attention optimization is fundamentally different from GEMM**:
   - GEMM optimization focuses on compute-memory overlap (warp_pipeline_stage, double buffering)
   - Attention optimization focuses on reducing non-MFMA compute overhead (softmax, masking, branch elimination)

2. **Load balancing is critical for causal attention**:
   - Zigzag remap can bring ~15-20% performance improvement
   - start_m × batch-head zigzag is superior to simple start_m-only zigzag

3. **Not routing V through smem is an important optimization**:
   - V is used only onceega, no reuse is needed
   - Routing through smem increases LDS pressure and bank conflicts
   - Directly using `convert_layout` to skip smem can yield ~10-15% performance improvement

4. **Full/Masked block separation is critical for causal attention**:
   - Eliminates branch overhead
   - Can yield ~20% performance improvement

5. **Small optimizations also add up**:
   - tl.assume, 1/d_i reciprocal multiplication, pointer arithmetic, and other small optimizations can cumulatively yield ~10% performance improvement

## Applicable Pitfall Experiences

| Number | Title | Key Points | Source |
|------|------|---------|------|
| 24 | P→smem to eliminate ds_bpermute is actually slower | 4 ds_bpermute < 128 ds_read_u16. P matrix using convert_layout is generally more efficient than going through smem | Appendix E |
| 25 | V's shared_layout order must match the dot_op read pattern | shared layout order must match the K-dimension requirements of the dot operand, not the load layout | Appendix E |
| 26 | V does not use persistent smem; use convert_layout to reduce LDS pressure | K (transposed) goes through smem, V (non-transposed) uses convert_layout. Reduces LDS/VGPR pressure, +10-13% | Appendix E |
| 27 | Full blocks / Masked blocks separation is the key optimization for non-causal scenarios | Separate full/masked iterations into two constexpr branches. Flips non-causal from trailing by 5% to leading by 8-13% | Appendix E |
| 28 | `tl.assume` compiler hints are effective for attention kernels | Add 16 stride assumes. Zero-cost optimization, +2-3% | Appendix E |
| 29 | `1/d_i` reciprocal multiplication replaces `acc/d_i` division | Compute reciprocal first then multiply, Newton-Raphson friendly. ~1% improvement | Appendix E |
| 30 | K uses on-the-fly smem (value=) instead of persistent smem (depth=1) | Cleaner code, compatible with sub-function patterns. Performance is comparable or better | Appendix E |
| 31 | Pointer arithmetic vs offset accumulation | Incorporate batch/head offset into base pointer, only compute block-level offset inside the loop | Appendix E |
| 38 | start_m-only Zigzag — effective for large S, ineffective for small S | Zigzag only on the start_m dimension, completely ineffective when M-blocks ≤ NUM_SES | Appendix G |
| 39 | start_m × batch-head Zigzag — effective for all S | Incorporate the batch-head dimension into the permutation, total blocks far exceeds SEs. +65% improvement at S=2048 | Appendix G |
| 40 | Non-Causal must maintain bh-first ordering | Non-causal uses bh-first to maintain L2 cache locality. start_m-first ordering causes a 3-4% performance regression | Appendix G |

## Stopping Conditions

Use the general stopping conditions from optimization-guide.md §1.8:

1. **Precision Verification**:
   - Compare with PyTorch's `F.scaled_dot_product_attention`, error < 1e-3 (fp16/bf16)
   - Verify correctness of causal mask
   - Verify correctness of GQA/MQA (if applicable)

2. **Performance Verification**:
   - Achieve performance at or above the Triton baseline
   - Verify performance stability across different sequence lengths

3. **Assembly Verification**:
   - Check whether the number of `ds_bpermute` is reasonable (unavoidable, but should not be excessive)
   - Check VGPR usage to ensure no scratch spill
   - Check instruction width of buffer_load/buffer_store (dwordx2 or dwordx4)

4. **Functional Verification**:
   - Verify correctness of causal/non-causal modes
   - Verify correctness across different head dims (16/32/64/128)
   - Verify correctness across different sequence lengths (short sequences, long sequences)
