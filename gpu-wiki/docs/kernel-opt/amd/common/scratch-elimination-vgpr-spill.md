# Eliminating Scratch Operations (Register Spills)

Eliminating buffer_load/buffer_store to scratch space (i.e., VGPR spilling to VRAM) is one of the most severe performance issues on AMD GPUs. This applies to all DSLs (Gluon/FlyDSL/Triton/CK) on CDNA3/CDNA4.

---

## Background

GPU kernels have a limited number of registers (VGPRs). When the compiler cannot fit all live variables into VGPRs, it "spills" some data to scratch space (essentially a temporary buffer in VRAM). Scratch access latency is comparable to global memory (hundreds of cycles), **far slower than the zero-latency access of VGPRs**, with catastrophic performance impact.

---

## VGPR Limits

| Architecture | Max VGPRs per Wave | Total VGPRs per CU | Notes |
|------|-------------------|-------------------|----|
| CDNA3 (gfx942) | 512 | 512 × max_waves | architectural limit |
| CDNA4 (gfx950) | 512 | 512 × max_waves | architectural limit |

> **Key Point**: More VGPR usage → lower occupancy → fewer concurrent waves → harder to hide latency. Even without spilling, excessive VGPR usage reduces performance. See [Occupancy Optimization](occupancy-optimization.md) for details.

---

## Diagnostic Methods

### 1. Assembly Inspection

Search for `buffer_load` / `buffer_store` targeting scratch space in the assembly:

```
# scratch :
buffer_store_dword v*, off, s[...], 0 offen  ; spill: VGPR → scratch
buffer_load_dword v*, off, s[...], 0 offen   ; reload: scratch → VGPR
```

**Distinguishing scratch from global**: Scratch operations typically use the `off` addressing mode and target a scratch buffer descriptor, whereas global load/store uses `buffer_load/store` + a data buffer descriptor.

### 2. Compiler Output

Most compilers report VGPR usage and spill counts:

| Information | Meaning |
|------|------|
| `vgpr_count: 256` | 256 VGPRs used |
| `scratch_size: 0` | No spills (ideal) |
| `scratch_size: 1024` | 1024 bytes spilled (serious issue) |
| `spilled_vgprs: 8` | 8 VGPRs spilled |

### 3. Performance Counters

| Counter | Description |
|--------|------|
| `SQ_WAVES` | Wave count |
| `SQ_INSTS_VALU` / `SQ_INSTS_SMEM` | Instruction count |
| `SPI_RA_VGPR_SGPR_FULL_CSN` | VGPR allocation failure count (high values indicate VGPR pressure) |

---

## Common Causes and Fixes

### 1. Tile Size Too Large

**Problem**: Larger tile sizes require more accumulator registers.

**Fix**: Reduce the tile's M / N / K dimensions.

**GEMM Accumulator VGPR Estimation**:

```
acc_vgprs = (tile_M / mfma_M) × (tile_N / mfma_N) × mfma_output_vgprs
```

| Tile Size | MFMA Shape | Warps | Per-warp Acc VGPRs | Notes |
|-----------|-----------|-------|-------------------|------|
| 256×256 | 32×32×8 | 4 (2×2) | 512 | **Limit reached! No room for pipeline** |
| 256×256 | 32×32×8 | 8 (2×4) | 256 | Sufficient for pipeline |
| 128×128 | 32×32×8 | 4 (2×2) | 128 | Comfortable |
| 64×64 | 32×32×8 | 4 (2×2) | 32 | Very comfortable |

### 2. Too Many Simultaneously Live Variables in Loops

**Problem**: A large number of intermediate results are simultaneously live inside the loop body, exceeding VGPR capacity.

**Fixes**:
- **Reorder code to reduce variable liveness range** — tighten the scope of intermediate variables.
- **Fuse multiple element-wise operations** into a single expression.
- **Split loops** — break one large loop into multiple smaller loops, each using fewer VGPRs.

### 3. Pipeline Depth Too Deep

**Problem**: Deep software pipeline stages mean more data must be simultaneously resident (prefetch buffers).

**Fixes**:
- Reduce the number of pipeline stages.
- Trade-off: shallower pipeline reduces compute-memory overlap, so balance spill elimination against instruction scheduling efficiency.

---

## Increasing Warp Count as an Alternative

When tile size cannot be reduced (constrained by the upper-level algorithm), **increasing the number of warps per block** is an effective alternative, because accumulators are distributed across more warps:

```
per_warp_acc_vgprs = total_acc_elements / num_warps × element_vgprs
```

| Warps/Block | Per-warp Acc VGPRs (256×256 tile) | Pipeline Headroom |
|-------------|-----------------------------------|--------------|
| 4 | 512 | 0 (spill inevitable) |
| 8 | 256 | 256 VGPRs (sufficient) |

> **Note**: Increasing warp count requires corresponding adjustments to the load layout configuration.

---

## Related Documentation

- **Occupancy**: [Occupancy Optimization](occupancy-optimization.md) — the relationship between VGPRs and occupancy
- **Instruction Width**: [Coalesced Access and Instruction Width](coalesced-access-load-store-width.md) — increasing size_per_thread can exacerbate VGPR pressure
- **Hardware Specs**: [Hardware Comparison](../../../hardware-specs/hardware-comparison-cdna3-cdna4.md) — total VGPR counts per architecture
- **Small Matrix Optimization**: [Small Matrix / Low CU Utilization Optimization](small-matrix-cu-utilization.md) — reducing tile size to eliminate spills
