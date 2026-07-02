---
pattern: small_gemm
type: conclusion
---

# Key Conclusions


**Last updated**: 2026-06-30

## 1. Correct Priority for Small Matrix Optimization

```
 tile size ≫ BLOCK_K > ISA optimization
```

The benefit of reducing tile size (2.5×) far exceeds the sum of all other optimizations.

## 2. CU Utilization Improvement Is the Root Cause of Performance Doubling

Going from 1 block to 32 blocks is a 32× increase in parallelism. ISA-level optimizations can only improve the efficiency of a single CU and cannot solve the problem of idle CUs.

## 3. Small Matrices Do Not Have the Problem of Tiles Being Too Small

Within the small matrix range of 128×64×256 ~ 256×128×512, performance is **strictly monotonically increasing** with tile size reduction:

```
128×256 (14.5µs) → 64×128 (8.5µs) → 64×64 (7.6µs) → 32×64 (6.9µs) → 32×32 (6.7µs)
```

No "regression due to tiles being too small" occurred. Reasons:
1. The benefit of improved CU utilization far outweighs the loss from reduced data reuse rate
2. Small matrix data is inherently small (128×64 = 16KB), and can be covered by L1/L2 cache even without reuse
3. Kernel launch overhead is a fixed cost; having more blocks does not increase launch overhead

## 4. Gluon Can Significantly Outperform Triton on Small Matrices

Final results (32×32×32 tile, 2 warps):
- vs Triton: **2.6-2.85× faster**
- vs Converter baseline: **3.3× faster**

---

## Full Process Final Results

### Round 1: ISA-Level Optimization (Fixed 128×256 Tile)

**Test matrices**: M=256, K=128, N=512 (largest test case)

| Variant | Key Change | Time (ms) | vs Baseline | vs Triton | Conclusion |
|------|---------|----------|---------|-----------|------|
| Baseline (Converter) | BLOCK_K=16, 4w, persistent smem, other=0.0 | 0.0224 | — | 1.087× slower | Initial version |
| v2: warp_pipeline_stage | Added WPS pipeline | 0.0255 | ❌ -13.8% | — | Too few iterations, overhead > benefit |
| v3: no other + assume | Removed other=0.0, added tl.assume | 0.0223 | +0.4% | 1.084× slower | Slight improvement |
| v4: 8-warp | num_warps=8, warps_per_cta=[1,8] | 0.0256 | ❌ -14.3% | — | Grid too small, no occupancy benefit |
| v5: XCD remap | MI308X 4-XCD remapping | 0.0225 | ≈ 0% | — | 4 blocks, no XCD imbalance |
| v6: value= pattern | allocate_shared_memory(value=...) | 0.0252 | ❌ -12.5% | — | Allocation per iteration adds overhead |
| **v7: BLOCK_K=32** ✅ | **BLOCK_K 16→32 + no other + assume** | **0.0186** | **+17.0%** | **0.903× faster** | **Round 1 best** |
| v8: BLOCK_K=64 | BLOCK_K 32→64 | 0.0192 | +14.3% | 0.93× faster | Increased register pressure |

### Round 2: Tile Size Search (BLOCK_K=32 Fixed)

| Tile (M×N) | Warps | Grid (128×64×256) | Grid (256×128×512) | tc1 (µs) | tc3 (µs) | vs Triton | vs v7 |
|-----------|-------|-------------------|-------------------|----------|----------|-----------|-------|
| 128×256 (v7) | 4 | 1 | 4 | 14.5 | 18.6 | 0.83-0.90 | Baseline |
| 64×128 | 4 | 4 | 16 | 8.5 | 10.6 | 0.48-0.51 | 1.7× faster |
| 64×64 | 4 | 8 | 32 | 7.6 | 8.7 | 0.42-0.44 | 1.9× faster |
| 32×64 | 4 | 16 | 64 | 6.9 | 8.0 | 0.37-0.40 | 2.1× faster |
| **32×32** ✅ | **2** | **32** | **128** | **6.7** | **7.2** | **0.35-0.39** | **2.2× faster** |

### Full Process Final Results (32×32×32 Tile, 2 Warps)

| Test Case | M×K×N | Triton (µs) | Gluon (µs) | Ratio | Notes |
|-----------|-------|------------|----------------|-------|------|
| tc1 | 128×64×256 | 18.6 | 6.7 | 0.36 | Small matrix |
| tc2 | 128×128×256 | 35.7 | 12.8 | 0.36 | Medium matrix |
| tc3 | 256×128×512 | 70.9 | 7.2 | 0.10 | Large matrix (grid saturated) |

---

## Stopping Criteria

Stop optimization when the following conditions are met:

1. **grid_blocks > 2× CU count** (e.g., MI308X: > 160 blocks)
2. **ISA already optimized** (common optimizations such as pitfalls 36ф, 37 have been applied)

At this point, CU utilization is no longer the bottleneck, and further reducing tile size will slow things down due to decreased data reuse rate.


## Related

- [Changelog for Preview 0.1.4](CHANGELOG.md)
- [AMD MI308X (gfx942) GEMM Optimization Techniques Reference](ck_gemm_optimization_reference.md)
- [ISA Optimization Detailed Checklist](common_optimizations.md)
- [Stopping Conditions](final_config_template.md)
- [Gluon AMD gfx942 (CDNA3 / MI300) API & Performance Optimization Guide](gluon-amd-gfx942-optimization.md)
- [CUTLASS GEMM Optimization Strategy](../../../nvidia/common/cutedsl/cutlass-gemm-optimization.md)
- [Community GEMM Optimization Practical Summary](../../../generic/gemm-optimization-guide.md)
- [Composable Kernel (CK) Architecture Overview](../../common/ck-architecture-overview.md)
- [Document Relationship Diagram](../../../RELATIONS.md)
