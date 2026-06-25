# AMD GPU Roofline Analysis Methodology

Tile-level Roofline Model construction, bottleneck identification, tile size selection, CU utilization pre-check, and theoretical performance upper bound estimation. Applicable to all DSLs (Gluon/FlyDSL/Triton) for CDNA3/CDNA4.

---

## 1. Building a Tile-Level Roofline Model

**Key Principle**: Roofline analysis should be performed at the **tile (block)** level, not using the global FLOPs/Bytes of the entire kernel. The GPU's scheduling unit is the block; each block independently loads tile data from HBM, performs computation, and stores results. Different tile sizes lead to different Arithmetic Intensity, which may in turn change the bottleneck type.

### Computing Tile-Level Arithmetic Intensity (AI)

```
Tile AI = Tile_FLOPs / Tile_Bytes (unit: FLOPs/Byte)
```

| Item | Computation Method | Description |
|------|---------|------|
| **Tile_FLOPs** | Floating-point operations for a single tile | GEMM tile: `2×BM×BN×K`; element-wise tile: `BM×BN` per op |
| **Tile_Bytes** | Total bytes read/written from HBM for a single tile | Input tile bytes + Output tile bytes (accounting for data type width) |

**GEMM tile example** (tile size = BM×BN, main loop iterates along the K dimension):

```
Tile_FLOPs = 2 × BM × BN × K
Tile_Bytes = (BM × K + BN × K + BM × BN) × element_size
Tile_AI    = 2 × BM × BN × K / ((BM × K + BN × K + BM × BN) × element_size)
```

**Impact of different tile sizes on AI** (GEMM, K=4096, bf16=2B, MI300X):

| BM | BN | Tile_FLOPs | Tile_Bytes | Tile AI | vs Ridge Point (247) |
|----|-----|-----------|-----------|---------|---------------------|
| 256 | 256 | 537M | 2.26MB | **237** | ≈ Ridge Point (on the edge) |
| 128 | 128 | 134M | 1.16MB | **115** | < Ridge Point → **Memory Bound** |
| 64 | 64 | 33.6M | 0.58MB | **57** | < Ridge Point → **Memory Bound** |
| 256 | 128 | 268M | 1.64MB | **163** | < Ridge Point → **Memory Bound** |

> **Key Insight**: For the same GEMM kernel, reducing the tile size from 256×256 to 128×128 changes the bottleneck type from Compute Bound to Memory Bound. This is why tile-level AI must be used instead of global AI to determine the bottleneck.

### Roofline Ridge Point

```
Ridge Point = peak (FLOPS) / peakbandwidth (Bytes/s) (unit: FLOPs/Byte)
```

| GPU | Precision | Peak Compute (TFLOPS) | Peak Bandwidth (TB/s) | Ridge Point (FLOPs/Byte) |
|-----|------|-------------------|-----------------|--------------------------|
| MI300X | FP16/BF16 | 1,307.4 | 5.3 | **247** |
| MI300X | FP32 | 163.4 | 5.3 | **30.8** |
| MI300X | FP64 | 81.7 | 5.3 | **15.4** |
| MI308X | FP16/BF16 | ~232 | ~5.3* | **~43.8** |
| MI355X | FP16/BF16 | — | — | **~629** |
> The Ridge Point of MI355X is much higher than that of MI300X; the same tile configuration is more likely to be memory-bound on MI355X. See  for details.

### Determining the Bottleneck

```
If Tile AI ≥ Ridge Point:
    → Compute Bound
    → Evaluation metric: Compute utilization
    → Optimization focus: Improve MFMA instruction throughput, eliminate stalls, increase compute-memory overlap

If Tile AI < Ridge Point:
    → Memory Bound
    → Evaluation metric: Bandwidth utilization
    → Optimization focus: Reduce memory access, increase cache hit rate, increase data reuse, optimize load/store width
```

> **Rule of Thumb**: Large tile (BM≥256, BN≥256) GEMM is typically Compute Bound; small tile, element-wise, reduction, and attention score operators are typically Memory Bound.

---

## 2. Tile Size Selection Guidance

The core value of tile-level Roofline analysis lies in **guiding tile size selection**. However, tile size cannot be chosen based solely on AI; the following multiple constraints must be comprehensively considered:

| Constraint | Impact | Judgment Method |
|---------|------|---------|
| **Tile AI vs Ridge Point** | Determines the bottleneck type. Larger tile → Higher AI → More likely Compute Bound | Compute using the formula above |
| **CU Utilization** | Larger tile → Fewer grid blocks → More idle CUs. Requires `grid_blocks ≥ num_CUs` | `grid_blocks = cdiv(M,BM) × cdiv(N,BN)`, see §3 for details |
| **Register Pressure** | Larger tile → More VGPR required per block → Potential spilling to scratch (catastrophic performance degradation) | Check assembly for `buffer_store` to scratch |
| **LDS Capacity** | Larger tile → Higher LDS usage → May exceed limits | `LDS = (BM×BK + BK×BN) × element_size`; CDNA3: 64 KB, CDNA4: 160 KB |
| **Occupancy** | Register and LDS usage affect the number of wavefronts that can simultaneously reside on each CU → Affects latency hiding ability | Determined by the compiler, adjustable via `num_warps` |**Tile Size Selection Decision Process**:

```
1. Start with the maximum feasible tile size (e.g., 256×256)
2. Check CU utilization:
   - grid_blocks < num_CUs → Tile too large, reduce tile size
3. Check hardware resources:
   - VGPR spilling (scratch) → Tile too large, reduce tile size
   - LDS exceeds limit → Tile too large, reduce tile size
4. Compute Tile AI:
   - AI ≥ Ridge Point → Compute Bound, current tile size feasible
   - AI < Ridge Point → Memory Bound, consider increasing tile size (if CU utilization allows)
5. Find balance between CU utilization and Tile AI
```

> **Experience**: The optimal tile size is usually neither the largest nor the smallest, but rather the point where **AI is as close as possible to or exceeds the Ridge Point under the precondition of sufficient CU utilization (≥ 1 wave/CU)**. For small matrices, CU utilization is often the main bottleneck; in this case, prioritize ensuring CU utilization, even if tile AI is lower.

---

## 3. CU Utilization Pre-check

Before entering ISA-level optimization, **you must check the ratio of grid blocks to the number of CUs**. CU utilization is directly correlated with tile size — the larger the tile, the fewer the grid blocks, and the more idle CUs:

```
grid_blocks = cdiv(M, BLOCK_SIZE_M) × cdiv(N, BLOCK_SIZE_N)
CU_ratio = grid_blocks / num_CUs   (MI308X: 80, MI300X: 304, MI355X: 256)
```

**Trade-off between CU utilization and tile size** (M=N=512, MI308X 80 CUs):

| BM | BN | grid_blocks | CU_ratio | Tile AI (K=4096, bf16) | Overall Assessment |
|----|-----|------------|----------|----------------------|---------|
| 256 | 256 | 4 | **5%** | 237 | ❌ Severely insufficient CUs, tile too large |
| 128 | 128 | 16 | **20%** | 115 | ⚠️ Low CUs, but acceptable |
| 64 | 64 | 64 | **80%** | 57 | ✅ Sufficient CUs, but Memory Bound |
| 64 | 128 | 32 | **40%** | 79 | ✅ Good balance point |

**Decision Rules**:

| grid_blocks / CUs | CU Utilization | Optimization Direction |
|-------------------|----------|---------|
| ≥ 50% | Normal | Proceed to ISA-level optimization |
| 10%-50% | Low | Consider reducing tile + ISA optimization |
| < 10% | **Severely insufficient** | **Must prioritize adjusting tiling strategy**, see [Small Matrix / Low CU Utilization Optimization](small-matrix-cu-utilization.md) |

> **Experience**: This is the most common "directional error" — after doing extensive profiling and ISA optimization on small matrices, only to discover that 95% of CUs are idle, rendering all the prior work wasted. Calculate CU_ratio first, and combine it with Tile Size selection guidance to find the balance point between CU utilization and Tile AI.

---

## 4. Utilization Assessment

### Compute Bound → Compute Utilization

```
Compute Utilization (%) = Actual Compute (TFLOPS) / Peak Compute (TFLOPS) × 100%
```

| GPU | Precision | Peak Compute |
|-----|------|---------|
| MI300X | FP16/BF16 | 1,307.4 TFLOPS |
| MI300X | FP8/INT8 | 2,614.9 TFLOPS |
| MI300X | FP32 | 163.4 TFLOPS |
| MI308X | FP16/BF16 | ~232 TFLOPS |

### Memory Bound → Bandwidth Utilization

```
Bandwidth Utilization (%) = Actual Bandwidth (TB/s) / Bandwidth Upper Bound (TB/s) × 100%
```

**Choosing the bandwidth upper bound**: GPUs are high-latency, high-bandwidth architectures. The data volume of a small-sized kernel is insufficient to saturate the memory pipelineorer, and **cannot reach the theoretical peak bandwidth at all**. Therefore, the denominator for bandwidth utilization should be chosen based on data volume:

| Data Volume Level | Bandwidth Upper Bound | How to Obtain |
|-----------|---------|---------|
| **Large data volume** (sufficient to saturate HBM) | Hardware theoretical peak bandwidth | Check  |
| **Small data volume** (insufficient to saturate HBM) | **Measured bandwidth upper bound for the same data volume** | Measure using a memcpy kernel |

> **Experience**: When data volume is < ~100MB, the measured bandwidth is usually far below the theoretical peak (possibly only 50-80%). Using the theoretical peak as the denominator at this point will make bandwidth utilization appear very low, but the kernel may actually be approaching the bandwidth limit for that data volume.

---

## 5. Theoretical Performance Upper Bound Assessment

After determining the bottleneck type and tile size, **you must assess the theoretical performance upper bound achievable by the current operator under the current configuration**. This upper bound integrates CU utilization, tiling strategy, and bandwidth upper bound, and is the key basis for judging "how much optimization headroom remains."

### Calculation Method

```
1. Determine per-tile theoretical minimum time under bottleneck type:
   - Compute Bound: tile_time_min = Tile_FLOPs / Peak Compute
   - Memory Bound:  tile_time_min = Tile_Bytes / Bandwidth Upper Bound (measured or theoretical)

2. Calculate CU scheduling rounds:
   num_waves = ceil(grid_blocks / num_CUs)

3. Theoretical minimum kernel time:
   theoretical_time = num_waves × tile_time_min

4. Theoretical performance upper bound:
   theoretical_TFLOPS = (Tile_FLOPs × grid_blocks) / theoretical_time
```

**Example** (GEMM M=N=512, K=4096, BM=BN=64, bf16, MI308X 80 CUs):

```
Tile_FLOPs = 2 × 64 × 64 × 4096 = 33.6M
Tile_Bytes = (64×4096 + 64×4096 + 64×64) × 2 = 1.05MB
Tile AI    = 33.6M / 1.05MB = 32 → Memory Bound (< 43.8)

grid_blocks = 8 × 8 = 64
num_waves   = ceil(64 / 80) = 1

tile_time_min = 1.05MB / 3.2 TB/s = 0.328 μs (usebandwidthupper bound)
theoretical_time = 1 × 0.328 μs = 0.328 μs
theoretical_TFLOPS = 33.6M × 64 / 0.328 μs = 6.55 TFLOPS

→ Theoretical performance upper bound 6.55 TFLOPS (2.8% of peak 232 TFLOPS)
→ Indicates the operator cannot efficiently utilize hardware under this configuration; optimization space is limited
``` > **Key Insight**: When the theoretical performance ceiling is far below the hardware peak, the problem lies not in ISA-level optimization but at the **operator configuration level** (tile size, matrix dimensions, data reuse). In this case, prioritize algorithmic optimizations (e.g., kernel fusion, reordering computation) rather than instruction-level optimizations.

---

## Related Documents

- **Hardware Specifications**:  — Ridge Point, peak performance, bandwidth for each architecture
- **Occupancy**: [Occupancy Optimization](occupancy-optimization.md) — Relationship between VGPR and occupancy
- **Small Matrix Optimization**: [Small Matrix/Low CU Utilization Optimization](small-matrix-cu-utilization.md) — Targeted strategies when CU utilization < 10%
- **General Theory**: [GPU Execution Model](../../../ref-docs/generic/gpu-execution-model.md), [GPU Memory Hierarchy](../../../ref-docs/generic/gpu-memory-hierarchy.md)
