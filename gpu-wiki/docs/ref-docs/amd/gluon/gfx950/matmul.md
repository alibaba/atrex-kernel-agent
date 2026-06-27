---
pattern: matmul
sub_patterns: []
pitfalls: [1, 2, 3, 4, 5]
---

# Standard GEMM / Batched GEMM Optimization Guide

> This document is a **leaf pattern**, with no sub-pattern dependencies.
> For the general ISA optimization checklist, see `common_optimizations.md`.
> For applicable pitfalls, see entries marked as `` and `[GEMM]` in `pitfalls.md`.

---

## Pattern Characteristics

| Characteristic | Description |
|------|------|
| **Core Computation** | `C = A × B` (or batched `C[b] = A[b] × B[b]`) |
| **Main Loop Structure** | Iterates along the K dimension, loading a tile of A/B into smem each iteration and executing mfma/mfma_scaled |
| **Inter-Iteration Dependencies** | **None**. Each iteration independently accumulates into the accumulator; no output[i] → input[i+1] dependency |
| **ISA Flags** | Contains `v_mfma_*` instructions; contains `buffer_load_to_shared` or `buffer_load` + `ds_write` |

**Identification Conditions**:
- Contains mfma/mfma_scaled instructions
- Main loop iterates along K dimension with **no cross-iteration data dependencies** (distinguishes from Recurrent patterns)
- Output tile `[BM, BN]`, accumulator is updated each iteration

---

## Bottleneck Characteristics

Standard GEMM is typically accurately modeled by the **Roofline Model**, with the bottleneck type determined by Tile AI vs. Ridge Point.

### Tile-Level Arithmetic Intensity

```
Tile_FLOPs = 2 × BM × BN × K
Tile_Bytes = (BM × K + BN × K + BM × BN) × element_size
Tile_AI    = Tile_FLOPs / Tile_Bytes
```

**Impact of different tile sizes on bottleneck type** (K=4096, bf16=2B):

| BM | BN | Tile_FLOPs | Tile_Bytes | Tile AI | MI355X (Ridge≈245) |
|----|-----|-----------|-----------|---------|-------------------|
| 256 | 256 | 537M | 2.26MB | **237** | ≈ Ridge Point (boundary) |
| 128 | 128 | 134M | 1.16MB | **115** | Memory Bound |
| 64 | 64 | 33.6M | 0.58MB | **57** | Memory Bound |

---

## Tile Size Selection Guidance

Tile size cannot be chosen based solely on AI; the following constraints must be considered comprehensively:

| Constraint | Impact | Evaluation Method |
|---------|------|---------|
| **Tile AI vs. Ridge Point** | Determines the bottleneck type. Larger tile → higher AI → more likely Compute Bound | Calculate using the formula above |
| **CU Utilization** | Larger tile → fewer grid blocks → more idle CUs. Requires `grid_blocks ≥ num_CUs` | `grid_blocks = cdiv(M,BM) × cdiv(N,BN)`, see optimization-guide.md §1.4 for details |
| **Register Pressure** | Larger tile → more accumulator registers → potential spill to scratch (catastrophic performance degradation) | Check assembly for `buffer_store` to scratch |
| **LDS Capacity** | Larger tile → higher LDS usage → may exceed the 160KB/block limit | `LDS = (BM×BK + BK×BN) × element_size` |
| **Occupancy** | Register and LDS usage affect the number of wavefronts that can simultaneously reside per CU → impacts latency hiding capability | Determined by the compiler, adjustable via `--num-warps` |

**Decision Flow**:

```
1. row tile size ( 256×256)
2. check CU utilization:
 - grid_blocks < num_CUs -> tile , tile
3. check:
 - VGPR (scratch) -> tile , tile
 - LDS -> tile , tile
4. compute Tile AI:
 - AI ≥ Ridge Point -> Compute Bound, current tile size row
 - AI < Ridge Point -> Memory Bound, tile(if CU utilization)
5. CU utilization Tile AI
```

> **Rule of Thumb**: The optimal tile size is usually neither the largest nor the smallest, but rather the one where **AI is as close to or above the Ridge Point as possible, provided CU utilization is sufficient (≥ 1 wave/CU)**. For small matrices, CU utilization is often the primary bottleneck; in such cases, CU utilization should be prioritized even if tile AI is lower.

---

## Optimization Strategy Priorities

Source: `common_optimizations.md` Appendix A, "Large-tile GEMM" row.

### Large-Tile GEMM (grid_blocks ≥ num_CUs)

| Priority | Optimization Item | Description |
|--------|------|------|
| ⭐⭐⭐ | §3.0 Coalesced Access Pre-Check | Mandatory for all kernels; directional errors can cause several-fold performance loss |
| ⭐⭐⭐ | §3.1 Coalesced Access + Wide Load | First-order benefit when Memory Bound |
| ⭐⭐ | §3.3 Eliminate Scratch/Spill | First-order benefit when Compute Bound |
| ⭐⭐ | §3.5 warp_pipeline_stage | Critical GEMM optimization for compute-memory overlap |
| ⭐ | §3.2 Bank Conflicts | Third-order fine-tuning |
| ⭐ | §3.4 async_copy Pipeline | Second-order benefit when Memory Bound |
| — | §3.6 CU Utilization Tuning | CUs are already sufficient; not applicable |

### Small-Tile GEMM (grid_blocks < num_CUs)

| Priority | Optimization Item | Description |
|--------|------|------|
| ⭐⭐⭐ | §3.6 Reduce Tile Size to Increase Parallelism | Highest priority; measured 2.5–2.85× improvement |
| ⭐⭐ | §3.6 Increase BLOCK_SIZE_K | Reduces loop overhead |
| ⭐ | §3.0–3.3 Basic ISA Optimizations | Low-cost micro-optimizations |
| ❌ | warp_pipeline_stage | Ineffective or negative optimization for small matrix scenarios |
| ❌ | XCD Remapping | Grid is too small; no load imbalance |## CDNA4-Specific Considerations

### gfx950 Disables In-Thread Transpose

The Triton compiler disables the in-thread transpose optimization for gfx950. Ensure data has the correct layout for MFMA consumption **before** storing it into shared memory.

### gfx950 Forces kpack=1

The gfx950 compiler enforces a hard constraint of `kpack == 1`. Use `k_width` in `DotOperandLayout` instead of relying on kpack bundling.

### Ping-Pong Scheduling Activates Only with async_copy

CDNA4's ping-pong scheduling optimization is only activated by the compiler when `async_copy.buffer_load_to_shared` is used. Using regular `buffer_load` will not trigger this optimization.

---

## Typical Optimization Paths

### Path 1: Large-Tile Compute-Bound GEMM

```
§3.0 order check -> §3.3 scratch -> §3.5 warp_pipeline_stage -> §3.1 load
```

### Path 2: Small-Tile Memory-Bound GEMM

```
§3.6 tile -> §3.6 BLOCK_SIZE_K -> §3.0 order check -> §3.1 load
```

### Path 3: MFMA_scaled FP4 GEMM

```
§3.0 order check -> §3.5 warp_pipeline_stage -> §3.1 dwordx4 load (fp4 )
```

---

## References

- General Optimizations: `common_optimizations.md`
- Lessons Learned: `pitfalls.md` (tagged [GEMM])
- Hardware Specs: `hardware_specs.md`

## Related Documents

- **Prerequisites**: [CDNA4 Hardware Specs](../../../../hardware-specs/hardware_specs_mi355x.md) + [General Optimization Checklist](common_optimizations.md)
- **Cross-Architecture Comparison**: [Hopper GEMM Optimization](../../../nvidia/gluon/sm90/matmul.md) | [CDNA3 GEMM Optimization](../../../../kernel-opt/amd/gluon/gfx942/pattern_overview.md) (5 files)
- **FP8 Deep Dive**: [CDNA4 FP8 GEMM Optimization Practice](../../common/gfx950/cdna4-fp8-gemm-optimization.md) — CDNA4 vs CDNA3 FP8 Hardware Comparison
- **Converter Reference**: [CDNA4 Matrix Multiplication Conversion](../../../../converter/amd/cdna4/matrix_multiply.md)
