pattern: small_gemm
type: optimization
priority: ⭐⭐⭐
performance_gain: +250%
Applicable scenario: Small matrix GEMM
---

# Optimization Strategies (Sorted by Priority)


**Last updated**: 2026-06-30

## Priority Table

| Priority | Optimization | Gain | Description |
|--------|--------|------|------|
| ⭐⭐⭐ | **Reduce tile size** | **2.5×** | Improves CU utilization, highest priority |
| ⭐⭐ | **Double BLOCK_K** | **+17%** | Reduces loop overhead, extremely cost-effective |
| ⭐ | Remove `other=0.0` | +0.4-3.5% | Eliminates redundant `v_cndmask_b32` instructions |
| ⭐ | `tl.assume` compiler hints | +1-2% | Zero-cost optimization, always do |
| ❌ | `warp_pipeline_stage` | **-14%** | Loop count too small, overhead > benefit |
| ❌ | `num_warps=8` | **-14%** | Grid too small, no occupancy benefit |
| ❌ | `value=` pattern | **-13%** | Per-iteration allocation adds overhead |

## Anti-Patterns (What NOT to Do)

- ❌ **Do NOT use `warp_pipeline_stage` on small matrices** — harmful when loop count < 4
- ❌ **Do NOT increase `num_warps` to 8** — ineffective when grid blocks are far fewer than the number of CUs
- ❌ **Do NOT use `value=` pattern** — per-iteration allocation adds overhead
- ❌ **Best practices for large matrices may be harmful for small matrices** — e.g., warp_pipeline_stage, num_warps=8

---

## Tile Size Search Results

### Search Process

Systematically search small tile configurations, create a temporary Triton script for each tile to extract TTGIR, obtain the correct layout, build a Gluon kernel, validate precision, and benchmark each one individually.

**Search results** (fp16, MI308X 80 CUs):

| Tile (M×N×K) | Warps | Grid (128×64×256) | Grid (256×128×512) | tc1 (µs) | tc3 (µs) | vs Triton |
|-------------|-------|-------------------|-------------------|----------|----------|-----------|
| 128×256×32 | 4 | 1 | 4 | 14.5 | 18.6 | 0.83-0.90 |
| 64×128×32 | 4 | 4 | 16 | 8.5 | 10.6 | 0.48-0.51 |
| 64×64×32 | 4 | 8 | 32 | 7.6 | 8.7 | 0.42-0.44 |
| 32×64×32 | 2 | 16 | 64 | 6.9 | 8.0 | 0.37-0.40 |
| **32×32×32** | **2** | **32** | **128** | **6.7** | **7.2** | **0.35-0.39** |

### Performance Comparison

| Version | Tile | Latency (µs) | vs Baseline | vs Triton |
|------|------|----------|---------|-----------|
| Baseline (Converter) | 128×256×16 | 22.4 | — | 1.087× slower |
| BLOCK_K=32 | 128×256×32 | 14.5 | +54% | 0.83× faster |
| **Final Optimal** | **32×32×32** | **6.7** | **+234%** | **2.6-2.85× faster** |

**Conclusion**: The benefit of reducing tile size (2.5×) far exceeds doubling BLOCK_K (1.17×) and all ISA optimizations combined.

---

## Small Tile Layouts Must Be Extracted from TTGIR

### Correct Approach

When reducing the tile from 128×256 to 32×32, all layout parameters change — warps_per_cta, threads_per_warp, size_per_thread, shared layout swizzle parameters, etc.

**Never manually derive layouts for small tiles. TTGIR extraction is the only reliable method.**

### TTGIR Extraction Results Comparison Across Tiles

| Tile | blocked_a spt | blocked_b spt | mma wpc | shared_b swizzle | num_warps |
|------|-------------|-------------|---------|-----------------|-----------|
| 128×256 | [1,16] | [4,8] | [1,4] | (4,4,4,[0,1]) | 4 |
| 64×128 | [1,8] | [2,8] | [1,4] | (4,2,8,[0,1]) | 4 |
| 64×64 | [1,8] | [1,8] | [2,2] | (1,1,1,[1,0]) | 4 |
| 32×64 | [1,8] | [2,8] | [1,2] | (4,2,8,[0,1]) | 2 |
| 32×32 | [1,8] | [1,8] | [2,1] | (1,1,1,[1,0]) | 2 |

**Key Observations**:
1. **warps_per_cta varies with tile size** — 128×256 uses [1,4], 32×32 uses [2,1]; they are not interchangeable
2. **shared_b swizzle differs significantly** — 64×64 and 32×32 use (1,1,1) with no swizzle, while others have swizzle
3. **num_warps adapts automatically** — small tiles use 2 warps, large tiles use 4 warps
4. **MFMA instr_shape remains [32,32,8]** — even the 32×32 tile uses the same MFMA shape


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
