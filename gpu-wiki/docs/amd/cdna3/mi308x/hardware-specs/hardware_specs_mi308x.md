# AMD MI308X (gfx942 / CDNA3) Hardware Compute Specifications

**Last Updated**: 2026-05-25

---

## Quick Lookup Table

Usage: Find the peak TFLOPS based on the kernel's **primary computation type** for compute utilization calculations.

```
utilization = actual TFLOPS / peak TFLOPS × 100%
```

---

## AMD Instinct MI308X (gfx942 / CDNA3)

> MI308X is a 4-XCD variant of the MI300 series with 80 CUs. Same ISA (gfx942) as MI300X but fewer compute units.

> **Evidence status**: this repository does not currently link a public AMD MI308X product specification. The product-level values below are repository reference values and must be verified against the deployed board (`rocminfo`, profiler/device properties, and measured bandwidth) before they are used as hard limits.

| Precision | Peak TFLOPS | Notes |
|------|-------------|------|
| **FP64 (Vector)** | 21.5 | 80 CUs × 64 lanes × 2100 MHz |
| **FP64 (Matrix)** | 43.0 | 2× FP64 vector |
| **FP32 (Vector)** | 43.0 | General-purpose computation |
| **TF32 (Matrix)** | 172.0 | Low-precision matmul training |
| **FP16 / BF16 (Matrix)** | 206 | Repository reference peak; verify on deployed MI308X |
| **FP8 / INT8 (Matrix)** | 412 | 2× BF16 ratio per CDNA3 architecture |

### Memory Specifications

| Parameter | Value |
|------|------|
| VRAM | 128 GB HBM3 |
| Memory Bandwidth | 5.3 TB/s |
| TDP | 500 W |

### Compute Units

| Parameter | Value |
|------|------|
| XCD Count | 4 |
| Compute Units | 80 (20 per XCD) |
| Stream Processors | 5,120 |
| MFMA Units | 320 (4 per CU) |
| Peak Clock | 2,100 MHz |
| Total VGPRs per CU | 512 × 4 SIMD = 2048 32-bit VGPRs |
| Max VGPRs per Wave | 512 |
| LDS per CU | 64 KB |
| L2 Cache | 16 MB (4 MB per XCD) |
| L3 Cache (Infinity Cache) | 128 MB |

---

## CDNA3 Architecture Key Parameters (gfx942)

These parameters are shared with MI300X and influence optimization decisions:

### Execution Units

| Parameter | Value | Impact |
|------|------|------|
| Wavefront size | 64 threads | 64 threads per wave |
| SIMD count / CU | 4 | 4 SIMD units per CU |
| VGPR / SIMD | 512 (arch) | Register spill threshold |
| SGPR / wave | 108 (max usable) | Scalar registers |
| Max waves / SIMD | Depends on VGPR usage | Occupancy upper limit |

### Memory Hierarchy

| Level | Size | Bandwidth/Latency | Notes |
|------|------|----------|------|
| VGPR (Registers) | 512 × 32-bit / wave | Fastest | Optimal when no spills |
| LDS (Shared Memory) | 64 KB / CU | ~High throughput | Bank conflicts reduce speed |
| L1 Cache (TCP) | 32 KB / CU | Medium | Vector cache |
| L2 Cache (TCC) | 16 MB (total) | Medium | 4 MB per XCD |
| L3 (Infinity Cache) | 128 MB | Medium | Shared across XCDs |
| HBM3 (Global Memory) | 128 GB | 5.3 TB/s | Slowest, masked by pipelining |

### LDS Specifications

| Parameter | Value |
|------|------|
| LDS Size per CU | 64 KB |
| LDS Banks | 32 |
| LDS Bank Width | 4 bytes |
| LDS Read Bandwidth | 128 bytes/clock |
| LDS Write Bandwidth | 64 bytes/clock |

### MFMA Instruction Specifications

| Instruction Shape | Input Type | Output Type | Execution Cycles |
|----------|---------|---------|-------------------|
| 16×16×4 | FP32 | FP32 | 32 |
| 32×32×2 | FP32 | FP32 | 64 |
| 16×16×16 | FP16/BF16 | FP32 | 16 |
| 32×32×8 | FP16/BF16 | FP32 | 32 |
| 16×16×32 | FP8 | FP32 | 16 |
| 32×32×16 | FP8 | FP32 | 32 |

> **Note**: FP8 format on gfx942 uses **FNUZ** (non-standard), not OCP standard. This differs from gfx950 (CDNA4) which uses OCP FP8.

---

## Roofline Analysis Aid

### Compute Arithmetic Intensity

```
AI = FLOPs / Bytes_transferred
```

### Roofline Ridge Points

| Precision | Peak TFLOPS | Bandwidth | Ridge Point (FLOPs/Byte) |
|------|-------------|-----------|--------------------------|
| FP16/BF16 | 206 | 5.3 TB/s | 206 / 5.3 ≈ **38.9** |
| FP8 | 412 | 5.3 TB/s | 412 / 5.3 ≈ **77.7** |
| FP32 | 43.0 | 5.3 TB/s | 43.0 / 5.3 ≈ **8.1** |

> **Repository-reference comparison vs MI300X**: using the unverified 206 TFLOPS and 5.3 TB/s MI308X reference values gives a ridge point of ~39 versus MI300X's ~247. This suggests more MFMA kernels may be compute-bound on MI308X, but the conclusion must be re-evaluated with deployed-board properties and measurements.

### Identifying Bottlenecks

```
if AI < Ridge Point:
    → Memory Bound: optimize data movement, caching, prefetch
else:
    → Compute Bound: optimize MFMA utilization, reduce stalls, increase occupancy
```

---

## MI308X vs MI300X Quick Comparison

| Parameter | MI300X | MI308X | Ratio |
|------|--------|--------|-------|
| XCDs | 8 | 4 | 0.5× |
| CUs | 304 | 80 | 0.26× |
| BF16 Peak TFLOPS | 1,307.4 | 206 | 0.16× |
| HBM Bandwidth | 5.3 TB/s | 5.3 TB/s | 1.0× |
| VRAM | 192 GB | 128 GB | 0.67× |
| BF16 Ridge Point | ~247 | ~39 | 0.16× |
| LDS / CU | 64 KB | 64 KB | 1.0× |

> **Optimization Implication**: The same kernel that is memory-bound on MI300X (AI < 247) may be compute-bound on MI308X (AI > 39). Always recalculate the roofline when porting between the two.

---

## How to Select Peak Compute Power

1. **Determine the primary computation type**: What computation accounts for the largest share in the kernel?
   - MFMA-intensive → Use Matrix TFLOPS (206 TFLOPS BF16)
   - Element-wise-intensive → Use Vector TFLOPS (43.0 TFLOPS FP32)
2. **Determine the data precision**: Which data type is used?
   - BF16 MFMA → 206 TFLOPS
   - FP32 element-wise → 43.0 TFLOPS
3. **Mixed computation**: Evaluate the MFMA mainloop and element-wise/epilogue phases separately; use profiling to determine which phase limits the shape.

---

## Related Documents

- **Same ISA, Larger Config**: [MI300X Hardware Specs](../../mi300x/hardware-specs/hardware_specs_mi300x.md) — Full 8-XCD MI300X specifications
- **Architecture Comparison**: [CDNA3 vs CDNA4 Hardware Comparison](../../../common/hardware-specs/hardware-comparison-cdna3-cdna4.md)
- **Cross-Architecture**: [CDNA4 MI355X Specs](../../../cdna4/hardware-specs/hardware_specs_mi355x.md) | [Hopper Specs](../../../../nvidia/hopper/hardware-specs/hardware_specs_hopper.md)
- **Downstream Consumers**: [ISA Optimization Checklist](../../ref-docs/gluon/common_optimizations.md) — Roofline calculations require the peak TFLOPS from this document
