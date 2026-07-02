pattern: small_gemm
type: overview
Applicable Scenario: Small Matrix GEMM
Pitfalls: [32-37, 41-43]
---

# Small Matrix GEMM Optimization Guide

> This document is a specialized scenario of `matmul`. For general GEMM optimization, see `docs/ref-docs/amd/gluon/gfx942/warp_pipeline_stage.md` and the general ISA checklist in `common_optimizations.md`.

**Last updated**: 2026-06-30

> For applicable pitfall experiences, see the optimization point files in this folder (pitfalls 32-37, 41-43).

---

## Pattern Characteristics

| Characteristic | Description |
|------|------|
| **Core Computation** | `C = A × B` (matrix dimension ≤ 512 or grid blocks < 10% of CU count) |
| **Main Loop Structure** | Iterates along the K dimension with very few iterations (1-8 times) |
| **Inter-Iteration Dependencies** | None |
| **ASM Signature** | Contains `v_mfma` instructions; very few grid blocks |

**Identification Criteria**:
- Matrix dimensions M, N ≤ 512
- `grid_blocks / num_CUs < 10%` (e.g., MI308X with 80 CUs, grid < 8 blocks)
- Main loop iterates along the K dimension with ≤ 8 iterations
- Contains v_mfma instructions

---

## Bottleneck Characteristics: CU Utilization Insufficiency

### The Primary Bottleneck is CU Utilization, Not Instruction Efficiency

The primary bottleneck for small matrix GEMM is **CU utilization insufficiency**, not instruction efficiency on a single CU. This is the fundamental difference compared to large matrix GEMM.

**Criteria**:
```
CU Utilization = grid_blocks / num_CUs
```

When `grid_blocks / num_CUs < 10%`:
- 90%+ of CUs are idle
- ISA optimization can only improve instruction efficiency on a single CU and contributes nothing to CU utilization
- **Reducing tile size through partitioning strategy is the only effective optimization direction**

**Example** (MI308X, 80 CUs):

| Matrix Size | Tile (M×N) | Grid Blocks | CU Utilization | Bottleneck Type |
|---------|-----------|-------------|-----------|---------|
| 128×64 | 128×256 | 0.25 (1 block) | 0.3% | **CU Utilization** |
| 128×64 | 32×32 | 8 blocks | 10% | **CU Utilization** |
| 256×128 | 32×32 | 32 blocks | 40% | Mixed |
| 4096×4096 | 128×256 | 512 blocks | 640% | **Instruction Efficiency** |

**Key Rule**: When `grid_blocks / num_CUs < 10%`, **the benefit of all ISA optimizations combined is far less than the benefit of reducing tile size**.

---

## Applicable Pitfall Experiences

The following pitfall experiences are distributed across the optimization point files in this folder:
- Optimization strategy-related pitfalls (32-37, 41-43) → `optimization_strategy.md`
- Key conclusions and stopping conditions → `key_conclusions.md`

Summary:

| # | Title | Key Point |
|---|------|---------|
| 32 | Doubling BLOCK_SIZE_K is the first step | Double the K dimension, halve the loop count, 17% improvement. Extremely high ROI (minimal changes, low risk, high gain) |
| 33 | BlockedLayout must be recalculated entirely | After increasing BLOCK_SIZE_K, spt/tpw/wpc must all be recalculated. `size_per_thread[K] × threads_per_warp[K] × warps_per_cta[K] = BLOCK_SIZE_K` |
| 34 | warp_pipeline_stage is harmful in low-iteration scenarios | When loop count < 4, ping-pong cannot be unrolled at all and purely adds scheduling overhead. 14% performance degradation |
| 35 | Increasing num_warps is ineffective in low-grid scenarios | When grid blocks are far fewer than CU count, the GPU is not fully loaded, so adding warps does not help. 14% performance degradation |
| 36 | Removing other=0.0 from buffer_load is effective for all matrix sizes | Eliminates redundant `v_cndmask_b32` instructions, improving performance by 0.4-3.5%. Zero risk, always do |
| 37 | tl.assume compiler hints are also effective for small matrices | Helps the compiler use unsigned operations Facts for address calculation, improving performance by 1-2%. Zero cost, always do |
| 41 | Reducing tile size yields the largest single optimization gain for small matrix GEMM | 2.5× improvement, far exceeding BLOCK_K doubling (1.17×) and all ISA optimizations combined |
| 42 | Small tile layouts must be extracted from TTGIR, not manually derived | All layout parameters change; TTGIR extraction is the only reliable method |
| 43 | For small matrices, the benefit of reducing tile size increases monotonically; no need to worry about "too small" | Performance increases strictly monotonically with tile size reduction. The counter-effect of tiles being too small only needs to be considered when grid blocks > 2× CU count |


## Related

- [Changelog for Preview 0.1.4](CHANGELOG.md)
- [AMD MI308X (gfx942) GEMM Optimization Techniques Reference](ck_gemm_optimization_reference.md)
- [ISA Optimization Detailed Checklist](common_optimizations.md)
- [Stopping Conditions](final_config_template.md)
- [Gluon AMD gfx942 (CDNA3 / MI300) API & Performance Optimization Guide](gluon-amd-gfx942-optimization.md)
- [CUTLASS GEMM Optimization Strategy](../../../nvidia/common/cutedsl/cutlass-gemm-optimization.md)
- [Community GEMM Optimization Practical Summary](../../../generic/gemm-optimization-guide.md)
- [Software Pipeline Depth Optimization](../../../nvidia/common/software-pipeline-depth-optimization.md)
- [Document Relationship Diagram](../../../RELATIONS.md)
