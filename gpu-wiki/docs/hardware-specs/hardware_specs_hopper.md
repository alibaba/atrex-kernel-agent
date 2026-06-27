# NVIDIA Hopper GPU Hardware Compute Specifications Table

**Last Updated**: 2026-03-18

---

## Quick Reference Table

Usage: Look up peak TFLOPS based on the kernel's **primary compute type** and **target GPU**, for use in compute utilization calculations.

```
utilization = actual TFLOPS / peak TFLOPS × 100%
```

---

## NVIDIA H100 SXM (sm_90, Hopper)

| Precision | Peak TFLOPS | With Sparsity | Use Case |
|------|-------------|---------|---------|
| **FP64 (Tensor Core)** | 33.5 | — | Scientific computing |
| **FP32 (Tensor Core / TF32)** | 494.7 | 989.4 | TF32 matrix multiply |
| **FP32 (CUDA Core)** | 67.0 | — | General-purpose computing |
| **FP16 / BF16 (Tensor Core)** | 989.4 | 1,978.9 | Training / Inference |
| **FP8 (Tensor Core)** | 1,978.9 | 3,957.8 | Inference |
| **INT8 (Tensor Core)** | 1,978.9 | 3,957.8 | Quantized inference |

### Memory Specifications

| Parameter | Value |
|------|------|
| VRAM | 80 GB HBM3 |
| Memory Bandwidth | 3.35 TB/s |
| TDP | 700 W |

### Compute Units

| Parameter | Value |
|------|------|
| Streaming Multiprocessors (SMs) | 132 |
| CUDA Cores | 16,896 |
| Tensor Cores (4th gen) | 528 |
| Peak Clock | 1,830 MHz |
| Registers per SM | 65,536 × 32-bit |
| Max Registers per Thread | 255 |
| Shared Memory per SM | Up to 228 KB (configurable) |
| L2 Cache | 50 MB |

---

## NVIDIA H20 (sm_90, Hopper)

> Note: H20 is a lower-compute, higher-bandwidth variant of the H100, designed for the Chinese market.

| Precision | Peak TFLOPS | Notes |
|------|-------------|------|
| **FP16 / BF16 (Tensor Core)** | ~148 | 78 SMs |
| **FP8 (Tensor Core)** | ~296 | 78 SMs |
| **FP32 (CUDA Core)** | ~39.6 | 78 SMs |

### Memory Specifications

| Parameter | Value |
|------|------|
| VRAM | 96 GB HBM3 |
| Memory Bandwidth | 4.0 TB/s |
| TDP | 400 W |

### Compute Units

| Parameter | Value |
|------|------|
| Streaming Multiprocessors (SMs) | 78 |
| CUDA Cores | 9,984 |
| Tensor Cores (4th gen) | 312 |
| Registers per SM | 65,536 × 32-bit |
| Max Registers per Thread | 255 |
| Shared Memory per SM | Up to 228 KB (configurable) |

---

## NVIDIA H200 (sm_90, Hopper)

> The H200 shares the same GPU die as the H100, but features larger and faster HBM3e.

| Precision | Peak TFLOPS | Notes |
|------|-------------|------|
| **FP16 / BF16 (Tensor Core)** | 989.4 | Same as H100 |
| **FP8 (Tensor Core)** | 1,978.9 | Same as H100 |
| **FP32 (CUDA Core)** | 67.0 | Same as H100 |

### Memory Specifications

| Parameter | Value |
|------|------|
| VRAM | 141 GB HBM3e |
| Memory Bandwidth | 4.8 TB/s |
| TDP | 700 W |

### Compute Units

| Parameter | Value |
|------|------|
| Streaming Multiprocessors (SMs) | 132 |
| All other parameters same as H100 | — |

---

## Hopper Architecture Key Parameters (sm_90)

These parameters influence optimization decisions:

### Execution Units

| Parameter | Value | Impact |
|------|------|------|
| Warp Size | **32** threads | 32 threads per warp (AMD uses 64) |
| Warp Groups | 4 warps = 1 warp group | wgmma executes at warp group granularity |
| Max Threads per SM | 2,048 | Maximum concurrent threads |
| Max Blocks per SM | 32 | Maximum concurrent blocks |
| Max Warps per SM | 64 | Maximum concurrent warps |
| Max Registers per Thread | 255 | Register spill threshold |

### Memory Hierarchy

| Level | Size | Bandwidth / Latency | Notes |
|------|------|----------|------|
| Registers | 255 × 32-bit per thread, 65536 per SM | Fastest | Optimal when no spill |
| Shared Memory | Up to 228 KB per SM (configurable) | ~High throughput | Bank conflicts reduce speed |
| L1 Cache | Shared with Shared Memory | Medium | Automatically managed |
| L2 Cache | 50 MB (H100) | Medium | Shared across all SMs |
| HBM3/HBM3e | 80–141 GB | 3.35–4.8 TB/s | Slowest, hidden via pipelining |

### Shared Memory Configuration

Hopper's shared memory and L1 cache share 256 KB per SM:

| Configuration | Shared Memory | L1 Cache |
|------|---------------|----------|
| Default | 48 KB | 208 KB |
| Medium | 100 KB | 156 KB |
| Maximum | **228 KB** | 28 KB |Configuration is required via `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, size)`.

> **Key Point**: Hopper's smem capacity (228 KB) is much larger than AMD CDNA3 (64 KB), making double/multi-buffered pipelines easier to implement.

### Tensor Core Instruction Specs (sm_90 wgmma)

| Instruction Shape (M×N×K) | Input Type | Output Type | Notes |
|------------------|---------|---------|------|
| 64×N×16 | FP16/BF16 | FP32 | N=8,16,24,...,256 |
| 64×N×32 | FP8 (e4m3/e5m2) | FP32 | N=8,16,24,...,256 |
| 64×N×8 | TF32 | FP32 | N=8,16,24,...,256 |

> **Key Difference**: wgmma operands are read directly from shared memory (via `NVMMASharedLayout`), without needing to first load them into registers. This is completely different from AMD MFMA, which requires `DotOperandLayout` to load into registers.

---

## Roofline Analysis Aid

### Compute Arithmetic Intensity

```
AI = FLOPs / Bytes_transferred
```

### Identify Bottleneck

```
if AI < (peak TFLOPS / peakbandwidth TB/s):
 -> Memory Bound (bottleneck)
 -> optimization: decrease, high cache , increasedata
otherwise:
 -> Compute Bound (computebottleneck)
 -> optimization: high, stall, increase wgmma
```

**Roofline Ridge Points for Each GPU**:

| GPU | Precision | Ridge Point (FLOPs/Byte) |
|-----|------|--------------------------|
| H100 SXM | FP16/BF16 | 989.4 / 3.35 ≈ **295** |
| H100 SXM | FP8 | 1978.9 / 3.35 ≈ **591** |
| H100 SXM | FP32 | 67.0 / 3.35 ≈ **20** |
| H20 | FP16/BF16 | 148.0 / 4.0 ≈ **37** |
| H20 | FP8 | 296.0 / 4.0 ≈ **74** |
| H200 | FP16/BF16 | 989.4 / 4.8 ≈ **206** |
| H200 | FP8 | 1978.9 / 4.8 ≈ **412** |

> **H20's Ridge Point is extremely low** (≈37 for BF16). This means most GEMMs on H20 (with AI > 37) are Compute Bound, so the optimization focus is on improving wgmma instruction throughput. In contrast, H100's Ridge Point is very high (≈295), so small-to-medium tile GEMMs may be Memory Bound.

---

## How to Choose Peak Compute

1. **Determine the main compute type**: What computation dominates the kernel?
   - wgmma/Tensor Core intensive → Use Tensor Core TFLOPS
   - Element-wise intensive → Use CUDA Core TFLOPS
2. **Determine data precision**: Which data type is used?
   - BF16 wgmma → FP16/BF16 Tensor Core = 989.4 TFLOPS (H100) / 148 TFLOPS (H20)
   - FP32 element-wise → FP32 CUDA Core = 67.0 TFLOPS (H100) / ~39.6 TFLOPS (H20)
3. **Mixed computation**: If the kernel has both wgmma and element-wise operations, evaluate using wgmma compute (wgmma is typically the performance-determining factor)

## Related Documents

- **Cross-Architecture Comparison**: [CDNA3 Hardware Specs](hardware_specs_mi300x.md) | [CDNA4 Hardware Specs](hardware_specs_mi355x.md)
- **General Spec Tables**: [NVIDIA Compute Capability Reference Table](../kernel-opt/nvidia/common/nvidia-compute-capabilities.md)
- **Downstream Consumption**: [Hopper ISA Optimization Checklist](../ref-docs/nvidia/gluon/sm90/common_optimizations.md)
- **⚠️ Difference to Note**: H20 ridge point ~37 vs H100 ~295, the same kernel may be compute-bound on H20 but memory-bound on H100
