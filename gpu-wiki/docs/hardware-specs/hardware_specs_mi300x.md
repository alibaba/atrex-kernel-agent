# AMD GPU Hardware Compute Power Specification Table

**Last Updated**: 2026-03-06

---

## Quick Lookup Table

Usage: Find the peak TFLOPS based on the kernel's **primary computation type** and **target GPU** for compute utilization calculations.

```
utilization = actual TFLOPS / peak TFLOPS × 100%
```

---

## AMD Instinct MI300X (gfx942 / CDNA3)

| Precision | Peak TFLOPS | With Sparsity | Use Case |
|------|-------------|---------|---------|
| **FP64 (Vector)** | 81.7 | — | Scientific computing |
| **FP64 (Matrix)** | 163.4 | — | FP64 matrix multiply |
| **FP32 (Vector)** | 163.4 | — | General-purpose computation |
| **TF32 (Matrix)** | 653.7 | 1,307.4 | Training (low-precision matmul) |
| **FP16 / BF16 (Matrix)** | 1,307.4 | 2,614.9 | Inference / Training |
| **FP8 (Matrix)** | 2,614.9 | 5,229.8 | Inference |
| **INT8 (Matrix)** | 2,614.9 | 5,229.8 | Quantized inference |

### Memory Specifications

| Parameter | Value |
|------|------|
| VRAM | 192 GB HBM3 |
| Memory Bandwidth | 5.3 TB/s |
| TDP | 750 W |

### Compute Units

| Parameter | Value |
|------|------|
| Compute Units | 304 |
| Stream Processors | 19,456 |
| Matrix Cores (MFMA units) | 1,216 |
| Peak Clock | 2,100 MHz |
| Total VGPRs per CU | 512 × 4 SIMD = 2048 32-bit VGPRs |
| Max VGPRs per Wave | 512 |
| LDS per CU | 64 KB |

---

## AMD Instinct MI308X (gfx942 / CDNA3)

> Note: The MI308X is a variant of the MI300 series. The values below are official/measured reference values.

| Precision | Peak TFLOPS | Notes |
|------|-------------|------|
| **FP16 / BF16 (Matrix)** | 206 | MI308X official peak compute power |
| **FP8 / INT8 (Matrix)** | 412 | Based on CDNA3 architecture 2× BF16 ratio |

### Memory Specifications

| Parameter | Value |
|------|------|
| Memory Type | HBM3 |
| Memory Bandwidth | 5.3 TB/s |

---

## CDNA3 Architecture Key Parameters (gfx942)

These parameters influence optimization decisions:

### Execution Units

| Parameter | Value | Impact |
|------|------|------|
| Wavefront size | 64 threads | 64 threads per wave |
| SIMD count / CU | 4 | 4 SIMD units |
| VGPR / SIMD | 512 (arch), actual depends on occupancy | Register spill threshold |
| SGPR / wave | 108 (max usable) | Scalar registers |
| Max waves / SIMD | Depends on VGPR usage | Occupancy upper limit |

### Memory Hierarchy

| Level | Size | Bandwidth/Latency | Notes |
|------|------|----------|------|
| VGPR (Registers) | 512 × 32-bit / wave | Fastest | Optimal when no spills |
| LDS (Shared Memory) | 64 KB / CU | ~High throughput | Bank conflicts reduce speed |
| L1 Cache (TCP) | 32 KB / CU | Medium | Vector cache |
| L2 Cache (TCC) | 256 MB (total) | Medium | Shared cache |
| HBM3 (Global Memory) | 192 GB | 5.3 TB/s | Slowest, masked by pipelining |

### MFMA Instruction Specifications

| Instruction Shape | Input Type | Output Type | Execution Cycles per Instruction |
|----------|---------|---------|-------------------|
| 16×16×4 | FP32 | FP32 | 32 |
| 32×32×2 | FP32 | FP32 | 64 |
| 16×16×16 | FP16/BF16 | FP32 | 16 |
| 32×32×8 | FP16/BF16 | FP32 | 32 |
| 16×16×32 | FP8 | FP32 | 16 |
| 32×32×16 | FP8 | FP32 | 32 |

---

## Roofline Analysis Aid

### Computing Arithmetic Intensity

```
AI = FLOPs / Bytes_transferred
```

### Identifying Bottlenecks

```
if AI < (peak TFLOPS / peakbandwidth TB/s):
 -> Memory Bound (bottleneck)
 -> optimization: decrease, high cache , increasedata
otherwise:
 -> Compute Bound (computebottleneck)
 -> optimization: high, stall, increase MFMA
```

**Roofline knee points for MI300X**:
- FP16/BF16: 1,307.4 / 5.3 ≈ **247 FLOPs/Byte**
- FP32: 163.4 / 5.3 ≈ **30.8 FLOPs/Byte**
- FP64: 81.7 / 5.3 ≈ **15.4 FLOPs/Byte**

Most MFMA kernels have AI > 247, placing them in the Compute Bound region. The optimization focus should be on improving compute instruction utilization efficiency.

---

## How to Select Peak Compute Power

1. **Determine the primary computation type**: What computation accounts for the largest share in the kernel?
   - MFMA-intensive → Use Matrix TFLOPS
   - Element-wise-intensive → Use Vector TFLOPS
2. **Determine the data precision**: Which data type is used?
   - BF16 MFMA → FP16/BF16 Matrix = 1,307.4 TFLOPS (MI300X)
   - FP32 element-wise → FP32 Vector = 163.4 TFLOPS (MI300X)
3. **Mixed computation**: If a kernel has both MFMA and element-wise operations, evaluate using the MFMA compute power (MFMA is usually the performance-determining factor)## Related Documents

- **Cross-Architecture Comparison**:  | [Hopper Hardware Specs](hardware_specs_hopper.md)
- **Downstream Consumers**: [ISA Optimization Checklist](../ref-docs/amd/gluon/gfx942/common_optimizations.md) — Roofline calculations require the peak TFLOPS from this document
- **⚠️ Difference Note**: Ridge point ~247 (BF16), significantly different from CDNA4 (~629) and H20 (~37). The optimization direction for the same kernel may be entirely different
