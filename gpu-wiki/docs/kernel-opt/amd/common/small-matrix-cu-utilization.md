# Small Matrix / Low CU Utilization Optimization

For scenarios with small matrices (grid size far smaller than the number of CUs), maximize actual throughput by adjusting tiling strategies and loop parameters. Applicable to all CDNA3/CDNA4 DSLs (Gluon/FlyDSL/Triton).

---

## When to Use

When Roofline analysis (see [Roofline Analysis Methodology](roofline-analysis-methodology.md)) shows **compute utilization < 5%** and the cause is **matrix dimensions too small, resulting in grid blocks far fewer than GPU CU count**.

**Core Insight**: In small matrix scenarios, **CU utilization is the primary bottleneck**, taking priority over all ISA-level optimizations. When the grid has only 1-4 blocks, 90%+ of CUs are idle, and the benefits of optimizations like warp_pipeline_stage and bank conflict elimination are negligible or even negative.

## CU Utilization Quick Reference

```
grid_blocks = cdiv(M, BLOCK_SIZE_M) × cdiv(N, BLOCK_SIZE_N)
CU_utilization = grid_blocks / num_CUs

MI308X: 80 CUs, MI300X: 304 CUs, MI355X: 256 CUs
```

| grid_blocks / CUs | CU Utilization | Optimization Direction |
|-------------------|----------|---------|
| ≥ 50% | Normal | Proceed with ISA-level optimization |
| 10%-50% | Low | Consider reducing tile size + ISA optimization |
| < 10% | **Severely Insufficient** | **Must prioritize tiling strategy adjustment (this document)** |

---

## Optimization Checklist

### 1. Reduce BLOCK_SIZE_M / BLOCK_SIZE_N to Increase Grid Parallelism (Highest Priority ⭐⭐)

**This is the single most impactful optimization for small matrix scenarios**, delivering measured improvements of **2.5-2.85×** (far exceeding the sum of all other optimizations).

When grid blocks are extremely few (< 10% CU), reducing tile size significantly increases the number of parallel blocks, directly improving CU utilization.

**Measured Data** (fp16 GEMM, MI308X 80 CUs):

| Tile (M×N×K) | Warps | Grid (128×64×256) | Grid (256×128×512) | tc1 (µs) | tc3 (µs) | vs Triton |
|-------------|-------|-------------------|-------------------|----------|----------|-----------|
| 128×256×32 | 4 | 1×1=**1** | 2×2=**4** | 14.5 | 18.6 | 0.83-0.90 |
| 64×128×32 | 4 | 2×2=4 | 4×4=16 | 8.5 | 10.6 | 0.48-0.51 |
| 64×64×32 | 4 | 2×4=8 | 4×8=32 | 7.6 | 8.7 | 0.42-0.44 |
| 32×64×32 | 2 | 4×4=16 | 8×8=64 | 6.9 | 8.0 | 0.37-0.40 |
| **32×32×32** | **2** | **4×8=32** | **8×16=128** | **6.7** | **7.2** | **0.35-0.39** |

**Key Findings**:
1. **Performance is approximately monotonically increasing with grid size** — more blocks = higher CU utilization = faster
2. **Smaller tile means lower per-CU efficiency, but the CU utilization improvement far outweighs the efficiency loss** — 32×32 per-tile compute efficiency is much lower than 128×256, but the 32× grid growth completely overwhelms the efficiency loss
3. **num_warps must be adjusted with tile size** — 32×32/32×64 tile uses 2 warps, 64×64/64×128 uses 4 warps
4. **MFMA instr_shape is always [32,32,8]** — even 32×32 tile uses the same MFMA shape, only warps_per_cta differs

### 2. Increase BLOCK_SIZE_K to Reduce Loop Overhead

For small K dimensions (K < 256), **loop control overhead is proportionally high**. Increasing BLOCK_SIZE_K can significantly reduce iteration count.

| BLOCK_SIZE_K | K=64 Iterations | K=128 Iterations | Notes |
|-------------|------------|-------------|------|
| 16 | 4 | 8 | Default, high overhead ratio |
| **32** | **2** | **4** | **Recommended**, loop overhead halved |
| 64 | 1 | 2 | Max, only for very small K |

**Modifying K-dimension tile size requires simultaneous changes to**:
- A/B matrix load layout (K dimension must cover new tile K size)
- Shared memory allocation size
- Constraint: per-thread load amount × thread count × warp count = tile K size
- Target: per-thread contiguous load amount in K dimension × element byte size ≥ 16 (achieve dwordx4 width)

> Specific layout parameter adjustment methods vary by DSL; refer to each DSL's dedicated documentation.

### 3. ISA-Level Micro-Optimizations (After Tiling Strategy is Fixed)

After tiling strategy optimization is complete, layer on the following low-cost ISA optimizations:

1. **Eliminate boundary condition padding branch instructions**: Removing default padding value parameters for loads can eliminate `v_cndmask_b32` instructions (effective for all sizes)
2. **Provide positive stride assumptions to the compiler**: Helps the compiler optimize address computation (zero cost)
3. **Scalar offset stepping**: Use scalar to compute offset increments in loops, avoiding recomputation each time

> Specific implementation details vary by DSL; refer to each DSL's dedicated documentation.

---

## Optimizations to Avoid for Small Matrices (Prevent Negative Optimization)

The following optimizations have proven ineffective or harmful on small matrices (under large tile configurations):

| Optimization | Small Matrix Result | Reason |
|------|----------|------|
| **Software Pipelining** | ❌ -14% | Too few loop iterations (2-7 times), pipeline fill/drain overhead > overlap benefit |
| **Increasing Warp Count** | ❌ -14% | When grid is extremely small, increasing warps cannot improve CU occupancy |
| **XCD Remapping** | ≈ 0% | Grid too small, no load imbalance across XCDs |
| **Dynamic Shared Memory Allocation** | ❌ -13% | Allocating shared memory per iteration increases overhead in low-iteration scenarios |> **Note**: The above negative optimization conclusions are based on **large tile + small matrix** scenarios. When tile size is reduced Chick grid increases, some optimizations (such as XCD remapping) may become effective and require re-evaluation.

---

## Related Documents

- **Prerequisites**: [Roofline Analysis Methodology](roofline-analysis-methodology.md) — Tile AI, CU utilization pre-check
- **Hardware Specifications**: [Hardware Specification Comparison](../../../hardware-specs/hardware-comparison-cdna3-cdna4.md) — CU count, peak compute
- **Occupancy**: [Occupancy Optimization](occupancy-optimization.md) — VGPR and occupancy relationship
