# Standard GEMM / Batched GEMM Optimization Guide

> This document is a **base pattern (leaf node)** with no sub-pattern dependencies.
> For the general ISA optimization checklist, see `common_optimizations.md`.
> For applicable pitfall experiences, see entries marked `` and `[GEMM]` in `pitfalls.md` (Pitfalls 1-5).

---

## Pattern Characteristics

| Characteristic | Description |
|------|------|
| **Core Computation** | `C = A × B` (or batched `C[b] = A[b] × B[b]`) |
| **Main Loop Structure** | Iterates along the K dimension, each iteration loads a tile of A/B to smemמוand executes wgmma |
| **Inter-Iteration Dependency** | **None**. Each iteration independently accumulates into the accumulator; no output[i] → input[i+1] |
| **SASS Indicators** | Contains `WGMMA` instructions; contains `LDGSTS` (async_copy) or `LDG`+`STS` |

**Identification Criteria**:
- Presence of wgmma instructions
- Main loop iterates along the K dimension with **no cross-iteration data dependency** (distinguishes from Recurrent pattern)
- Output tile `[BM, BN]`, accumulator updated each iteration

---

## Bottleneck Characteristics

Standard GEMM is typically accurately characterized by the **Roofline Model**, with the bottleneck type depending on Tile AI vs Ridge Point.

### Tile-Level Arithmetic Intensity

```
Tile_FLOPs = 2 × BM × BN × K
Tile_Bytes = (BM × K + BN × K + BM × BN) × element_size
Tile_AI    = Tile_FLOPs / Tile_Bytes
```

**Impact of different tile sizes on bottleneck type** (K=4096, bf16=2B):

| BM | BN | Tile_FLOPs | Tile_Bytes | Tile AI | H100 (Ridge=295) | H20 (Ridge=37) |
|----|-----|-----------|-----------|---------|-------------------|-----------------|
| 256 | 256 | 537M | 2.26MB | **237** | Memory Bound | **Compute Bound** |
| 128 | 128 | 134M | 1.16MB | **115** | Memory Bound | **Compute Bound** |
| 64 | 64 | 33.6M | 0.58MB | **57** | Memory Bound | **Compute Bound** |

> **Key Insight**: H20's Ridge Point is extremely low (≈37 BF16), so most GEMMs are Compute Bound. H100's Ridge Point is very high (≈295 BF16), making small to medium tile GEMMs Memory Bound instead.

---

## Tile Size Selection Guidance

Tile size selection should not focus solely on AI; the following multiple constraints must be considered together:

| Constraint | Impact | Evaluation Method |
|---------|------|---------|
| **Tile AI vs Ridge Point** | Determines bottleneck type. Larger tiles → higher AI → more likely Compute Bound | Calculate using the formula above |
| **SM Utilization** | Larger tiles → fewer grid blocks → more idle SMs. Requires `grid_blocks ≥ num_SMs` | `grid_blocks = cdiv(M,BM) × cdiv(N,BN)`, see optimization-guide.md §1.4 for details |
| **Register Pressure** | Larger tiles → more accumulator registers → potential spilling to local memory (disastrous performance degradation) | Check `local_memory_overhead` in ncu or `STL`/`LDL` instructions in SASS |
| **Shared Memory Capacity** | Larger tiles → more smem usage → may exceed limits | Hopper: up to 228 KB/block (requires `cudaFuncSetAttribute` configuration), default 48 KB |
| **Occupancy** | Register and smem usage affect the number of thread blocks that can simultaneously reside on each SM → affects latency hiding capability | `Achieved Occupancy` in ncu report |

**Decision Flow**:

```
1. row tile size ( 256×256)
2. check SM utilization:
 - grid_blocks < num_SMs -> tile , tile
3. check:
 - register (local memory) -> tile , tile
 - Shared memory -> tile , tile
4. compute Tile AI:
 - AI ≥ Ridge Point -> Compute Bound, current tile size row
 - AI < Ridge Point -> Memory Bound, tile(if SM utilization)
5. SM utilization Tile AI
```

> **Rule of Thumb**: The optimal tile size is usually neither the largest nor the smallest, but rather the point where **AI approaches or exceeds the Ridge Point as closely as possible while maintaining sufficient SM utilization (≥ 1 block/SM)**. For small matrices, SM utilization is often the primary bottleneck; in such cases, prioritize ensuring SM utilization even if tile AI is lower.

---

## Optimization Strategy Priority

Source: `common_optimizations.md` Appendix A "Large Tile GEMM" row.

### Large Tile GEMM (grid_blocks ≥ num_SMs)

| Priority | Optimization | Description |
|--------|--------|------|
| ⭐⭐⭐ | §3.0 Coalesced Memory Access Pre-Check | Required for all kernels; directional errors can cause several-fold performance losses |
| ⭐⭐⭐ | §3.1 Coalesced Access + Wide Load | First-order benefit when Memory Bound |
| ⭐⭐ | §3.3 Eliminate Scratch/Spill | First-order benefit when Compute Bound |
| ⭐⭐ | §3.4 async_copy Pipelining | Second-order benefit when Memory Bound |
| ⭐⭐ | §3.5 wgmma Correctness | Prerequisite for correctness; moderate performance impact |
| ⭐ | §3.2 Bank Conflicts | Third-order fine-tuning |
| — | §3.6 SM Utilization Tuning | SMs already sufficient; not applicable |### Small Matrix GEMM (grid_blocks < num_SMs)

| Priority | Optimization | Description |
|----------|--------------|-------------|
| ⭐⭐⭐ | §3.0 Coalesced access pre-check | Required |
| ⭐⭐⭐ | §3.6 Tile size tuning | **Highest priority**, SM utilization is the primary bottleneck |
| ⭐ | §3.1 Coalesced access + wide load | Optional |
| ⭐ | §3.3 Eliminate scratch/spill | Optional |
| — | §3.4 async_copy pipeline | Loop too short, not applicable |

---

## Stopping Conditions

Use the general stopping conditions from optimization-guide.md §1.8. The applicable checklist covers all items in §3.0-3.6 of `common_optimizations.md`.

---

## Applicable Pitfall Experience

See `pitfalls.md` for details. Summary below:

| # | Title | Key Points |
|---|-------|------------|
| 1 | Compiler is sensitive to code structure | Manual code reordering is usually a negative optimization. Only make mathematically equivalent parameter changes |
| 2 | Tile dimension tuning has a sweet spot | Bigger is not always better. Change only one dimension at a time and must benchmark |
| 3 | swizzle_byte_width constraints | `swizzle_byte_width ≤ TILE_DIM × element_bytes` |
| 4 | BlockedLayout must be fully recalculated | After modifying tile dimensions, spt/tpw/wpc must all be recalculated |
| 5 | Benchmark variance is high | CUDA events + 200 samples + P50, ≥5% is considered a valid improvement |

## Related Documents

- **Prerequisites**: [Hopper Hardware Specs](../../../../hardware-specs/hardware_specs_hopper.md) + [Common Optimization Checklist](common_optimizations.md)
- **Cross-Architecture Reference**: [CDNA4 GEMM Optimization](../../../amd/gluon/gfx950/matmul.md) | [CDNA3 GEMM Optimization](../../../../kernel-opt/amd/gluon/gfx942/pattern_overview.md) (5 files)
- **wgmma Details**: [PTX MMA Instruction Evolution](../../common/nvidia-ptx-mma-instructions.md) — wgmma instruction specification
- **CuTeDSL**: [CuTeDSL SM90-Specific Features](../../cutedsl/sm90/hopper-cutedsl-sm90.md) — WGMMA usage in CuTeDSL
- **Converter Reference**: [Hopper Matrix Multiply Conversion](../../../../converter/nvidia/hopper.md)
