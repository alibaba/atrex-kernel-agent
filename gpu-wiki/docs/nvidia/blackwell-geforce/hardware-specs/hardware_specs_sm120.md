# NVIDIA Blackwell GeForce / RTX PRO Hardware Compute Specification Table

**Last Updated**: 2026-05-12

---

## Quick Reference Table

Usage: Look up peak TFLOPS/TOPS based on the kernel's **primary compute type** and **target GPU** for compute utilization calculations.

```
utilization = actual TFLOPS / peak TFLOPS × 100%
```

> This document covers the Blackwell GeForce / RTX PRO series for `sm_120` / `sm_120a` (Compute Capability 12.0). RTX PRO 6000 specifications are derived from the published NVIDIA architecture whitepaper; RTX PRO 5000 specifications are compiled from NVIDIA product pages/datasheets combined with in-repo measurement records.

---

## NVIDIA RTX PRO 6000 Blackwell Workstation Edition (GB202 / sm_120)

| Precision | Peak TFLOPS / TOPS | With Sparsity | Use Case |
|------|-------------------|---------|---------|
| **FP64 (CUDA Core)** | 1.97 | — | Program correctness / compatibility; not suitable for high-throughput FP64 (officially 1/64 of FP32) |
| **FP32 (CUDA Core)** | 126.0 | — | General-purpose compute, element-wise, reduction |
| **TF32 (Tensor Core)** | 251.9 | 503.8 | Training / inference matrix multiply |
| **FP16 / BF16 (Tensor Core)** | 503.8 | 1,007.6 | Training / inference |
| **FP8 (Tensor Core, FP32 Accumulate)** | 1,007.6 | 2,015.2 | Inference / low-precision training |
| **FP4 (Tensor Core, FP32 Accumulate)** | 2,015.2 TOPS | 4,030.4 TOPS | NVFP4 / MXFP4 quantized inference |
| **INT8 (Tensor Core)** | 1,007.6 TOPS | 2,015.2 TOPS | Quantized inference |

### Memory Specifications

| Parameter | Value |
|------|------|
| VRAM | 96 GB GDDR7 ECC |
| Memory Interface | 512-bit |
| Memory Data Rate | 28 Gbps |
| Memory Bandwidth | 1.792 TB/s |
| L2 Cache | 128 MB |
| TGP | 600 W |
| PCIe | Gen 5 |

### Compute Units

| Parameter | Value |
|------|------|
| Graphics Processing Clusters (GPCs) | 12 |
| Texture Processing Clusters (TPCs) | 94 (product-enabled value) |
| Streaming Multiprocessors (SMs) | 188 (product-enabled value; GB202 full chip has 192) |
| CUDA Cores | 24,064 |
| Tensor Cores (5th gen) | 752 |
| RT Cores (4th gen) | 188 |
| Texture Units | 752 |
| ROPs | 192 |
| GPU Boost Clock | 2,617 MHz |
| CUDA Cores per SM | 128 |
| Tensor Cores per SM | 4 |
| RT Cores per SM | 1 |
| Register File per SM | 256 KB |
| L1 Data Cache / Shared Memory per SM | 128 KB physical pool |
| Total L1 Data Cache / Shared Memory | 24,064 KB |
| Total Register File | 48,128 KB |

---

## NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (GB202 / sm_120)

> The Max-Q edition uses the same GB202 product configuration as the Workstation Edition; the main differences are power consumption and boost clock, resulting in lower peak compute. Memory capacity and bandwidth remain the same.

| Precision | Peak TFLOPS / TOPS | With Sparsity | Use Case |
|------|-------------------|---------|---------|
| **FP64 (CUDA Core)** | 1.71 | — | Program correctness / compatibility (officially 1/64 of FP32) |
| **FP32 (CUDA Core)** | 109.7 | — | General-purpose compute, element-wise, reduction |
| **TF32 (Tensor Core)** | 219.5 | 438.9 | Training / inference matrix multiply |
| **FP16 / BF16 (Tensor Core)** | 438.9 | 877.9 | Training / inference |
| **FP8 (Tensor Core, FP32 Accumulate)** | 877.9 | 1,755.7 | Inference / low-precision training |
| **FP4 (Tensor Core, FP32 Accumulate)** | 1,755.7 TOPS | 3,511.4 TOPS | NVFP4 / MXFP4 quantized inference |
| **INT8 (Tensor Core)** | 877.9 TOPS | 1,755.7 TOPS | Quantized inference |

### Memory Specifications

| Parameter | Value |
|------|------|
| VRAM | 96 GB GDDR7 ECC |
| Memory Interface | 512-bit |
| Memory Data Rate | 28 Gbps |
| Memory Bandwidth | 1.792 TB/s |
| L2 Cache | 128 MB |
| TGP | 300 W |
| PCIe | Gen 5 |

### Compute Units

| Parameter | Value |
|------|------|
| GPCs | 12 |
| TPCs | 94 |
| SMs | 188 |
| CUDA Cores | 24,064 |
| Tensor Cores (5th gen) | 752 |
| RT Cores (4th gen) | 188 |
| Texture Units | 752 |
| ROPs | 192 |
| GPU Boost Clock | 2,280 MHz |
| Register File per SM | 256 KB |
| L1 Data Cache / Shared Memory per SM | 128 KB physical pool |

## NVIDIA RTX PRO 5000 Blackwell (GB202 / sm_120a)

> The RTX PRO 5000 Blackwell is the most common target card for `sm_120` kernel optimization in this repository. NVIDIA's official datasheet explicitly lists 14,080 CUDA Cores, 48/72 GB GDDR7 ECC, 512-bit, 1,344 GB/s, 300 W, 65 TFLOPS Single-Precision Performance, 196 RT TFLOPS, and 2,064 AI TOPS. The table below uses the official `2,064 AI TOPS` (FP4 with sparsity) as the Tensor peak anchor point and expands based on Blackwell Tensor Core throughput ratios `FP4:FP8:FP16/BF16:TF32 = 8:4:2:1`.

| Precision | Peak TFLOPS / TOPS | With Sparsity | Use Case |
|------|-------------------|---------|---------|
| **FP64 (CUDA Core)** | 1.02 | — | Program correctness / compatibility (derived as 1/64 of official FP32) |
| **FP32 (CUDA Core)** | 65.0 | — | General-purpose computation, element-wise, reduction |
| **TF32 (Tensor Core)** | 129.0 | 258.0 | Training / inference matrix multiply |
| **FP16 / BF16 (Tensor Core)** | 258.0 | 516.0 | Training / inference |
| **FP8 (Tensor Core, FP32 Accumulate)** | 516.0 | 1,032.0 | Inference / low-precision training |
| **FP4 (Tensor Core, FP32 Accumulate)** | 1,032.0 TOPS | 2,064.0 TOPS | NVFP4 / MXFP4 quantized inference |
| **INT8 (Tensor Core)** | 516.0 TOPS | 1,032.0 TOPS | Quantized inference |

### Memory Specifications

| Parameter | Value |
|------|------|
| VRAM | 48 GB or 72 GB GDDR7 ECC |
| Memory Interface | 512-bit |
| Memory Bandwidth | 1.344 TB/s (datasheet: 1,344 GB/s) |
| Repository-measured D2D memcpy ceiling | 1.032-1.099 TB/s (from existing `sm_120` optimization records) |
| TGP | 300 W |
| PCIe | Gen 5.0 x16 |

### Compute Units

| Parameter | Value |
|------|------|
| Streaming Multiprocessors (SMs) | 110 |
| CUDA Cores | 14,080 |
| Tensor Cores (5th gen) | 440 |
| RT Cores (4th gen) | 110 |
| CUDA Cores / SM | 128 |
| Tensor Cores / SM | 4 |
| RT Cores / SM | 1 |
| FP32 Single-Precision Performance | 65.0 TFLOPS |
| RT Core Performance | 196 TFLOPS |
| Register File per SM | 256 KB |
| L1 Data Cache / Shared Memory per SM | 128 KB physical pool |
| Available Dynamic Shared Memory Limit | 99 KB / block (from existing `sm_120` kernel design and measurement records) |

---

## Blackwell GeForce / RTX PRO Architecture Key Parameters (sm_120)

These parameters influence optimization decisions:

### Execution Units

| Parameter | Value | Impact |
|------|------|------|
| Compute Capability | 12.0 (`sm_120` / `sm_120a`) | Blackwell client / GeForce lineup |
| Warp size | 32 threads | 32 threads per warp |
| CUDA Cores per SM | 128 | Basis for FP32/INT32/element-wise throughput |
| Tensor Cores per SM | 4 (5th gen) | Basis for FP16/BF16/TF32/FP8/FP4 matrix throughput |
| RT Cores per SM | 1 (4th gen) | Ray tracing / neural rendering |
| Register File per SM | 256 KB | Core constraint for register pressure and occupancy |
| Max Registers per Thread | 255 | Register spill threshold |
| L1 / Shared Memory per SM | 128 KB physical pool | L1 and shared memory share physical SRAM |
| Dynamic Shared Memory Limit | 99 KB / block | Direct constraint for large-tile / multi-stage pipelines |

### Key Differences from SM100 Data Center Blackwell

| Feature | SM100 / SM103 (B200 / B300) | SM120 / SM120a (RTX PRO 5000) | Optimization Implication |
|------|------|------|------|
| Tensor Core Instruction Route | `tcgen05.mma` / UMMA | `mma.sync` / warp-level MMA | Do not copy SM100 UMMA kernels directly |
| TMEM | Supported | Not supported | Accumulators stay in registers |
| TMA | Supported | Supported | Usable for bulk global-to-shared memory transfers |
| `cp.async` | Supported | Supported | Suitable for hand-rolled multi-stage G2S pipelines |
| FP4 / FP6 | Supported | Supported | SM120 supports NVFP4 / MXFP4 block-scaled MMA |

> **Key Reminder**: `sm_120` is client Blackwell. Repository measurement experience shows it lacks the `tcgen05` / TMEM / UMMA fast path. Optimizations should be architected around "TMA + `cp.async` memory access pipeline + warp-level MMA / warp shuffle computation."

### Memory Hierarchy

| Level | Size | Bandwidth/Latency | Notes |
|------|------|----------|------|
| Registers | 256 KB / SM, up to 255 regs/thread | Fastest | High register pressure rapidly reduces occupancy |
| Shared Memory | 128 KB L1/SMEM physical pool, dynamic cap 99 KB/block | High throughput | Bank conflicts and shared loads affect L1/TEX stat interpretation |
| L1/TEX Cache | Shares physical SRAM with Shared Memory | Medium | `sm_120` Some NCU L1/TEX hit rates are polluted by shared loads |
| L2 Cache | 128 MB（RTX PRO 6000） | Medium | Large L2 benefits tile locality / rasterization locality |
| GDDR7 | 48-96 GB | 1.344-1.792 TB/s | Lower bandwidth compared to HBM; streaming kernels are typically bandwidth-bound |

### Tensor Core / MMA Programming Tips

| Type | SM120 Path | Notes |
|------|----------|------|
| FP16 / BF16 GEMM | `mma.sync.aligned.m16n8k16` | warp-level MMA path |
| FP8 GEMM | block-scaled / mixed-input MMA | Pay attention to scale layout and shared memory unpack |
| FP4 GEMM | NVFP4 / MXFP4 block-scaled MMA | Requires handling e2m1 packed data with e4m3/e8m0 scale factors |
| Attention / FA | Dedicated SM120 fast path | H100/B200 WGMMA/UMMA fast path is not applicable to SM120 |

---

## Roofline Analysis Assistance

### Computing Arithmetic Intensity

```
AI = FLOPs / Bytes_transferred
```

### Identifying Bottlenecks

```
if AI < (peak_TFLOPS / peak_bandwidth_TBps):
 -> Memory Bound
 -> optimize memory traffic, coalescing, reuse, and overlap
otherwise:
 -> Compute Bound
 -> optimize Tensor Core or CUDA Core utilization, stalls, and occupancy
```

**Typical Ridge Points**:

| GPU | Precision | Ridge Point (FLOPs/Byte) |
|-----|------|--------------------------|
| RTX PRO 6000 Workstation | FP16/BF16 Tensor | 503.8 / 1.792 ≈ **281** |
| RTX PRO 6000 Workstation | FP8 Tensor | 1007.6 / 1.792 ≈ **562** |
| RTX PRO 6000 Workstation | FP4 Tensor | 2015.2 / 1.792 ≈ **1125** |
| RTX PRO 6000 Workstation | FP32 CUDA | 126.0 / 1.792 ≈ **70** |
| RTX PRO 5000 | FP16/BF16 Tensor | 258.0 / 1.344 ≈ **192** |
| RTX PRO 5000 | FP8 Tensor | 516.0 / 1.344 ≈ **384** |
| RTX PRO 5000 | FP4 Tensor | 1032.0 / 1.344 ≈ **768** |
| RTX PRO 5000 | FP32 CUDA | 65.0 / 1.344 ≈ **48** |

> **Optimization Implication**: SM120's GDDR7 bandwidth is lower than datacenter HBM bandwidth. Streaming, epilogue, and quantization kernels often become bandwidth-bound; GEMM and attention still require shape-specific roofline and instruction-path verification.

---

## How to Choose Peak Compute

1. **Identify the primary compute type**: What computation dominates the kernel?
   - Tensor Core / MMA intensive → use FP16/BF16/FP8/FP4 Tensor Core TFLOPS/TOPS
   - Element-wise / reduction intensive → use FP32 CUDA Core TFLOPS
   - Memory-bound transport / quant epilogue intensive → prioritize bandwidth roofline, not Tensor Core peak
2. **Identify the target GPU**:
   - RTX PRO 6000 Workstation → use 126.0 FP32, 503.8 BF16, 1,007.6 FP8, 2,015.2 FP4, 1.792 TB/s
   - RTX PRO 6000 Max-Q → use 109.7 FP32, 438.9 BF16, 877.9 FP8, 1,755.7 FP4, 1.792 TB/s
   - RTX PRO 5000 → use 65.0 FP32, 258.0 BF16, 516.0 FP8, 1,032.0 FP4, 1.344 TB/s
3. **Mixed compute**: If the kernel has both Tensor Core computation and element-wise epilogue, compute separate rooflines for the mainloop and epilogue; do not use a single peak to mask bottlenecks.
4. **Source priority**: RTX PRO 6000 uses official peaks from Appendix A of the local NVIDIA architecture whitepaper; RTX PRO 5000 uses NVIDIA's official datasheet figures: 65 TFLOPS Single-Precision Performance, 196 RT TFLOPS, 2,064 AI TOPS, 48/72 GB GDDR7 ECC, 512-bit, 1,344 GB/s, and 300 W.

## Related Documents

- **Official RTX PRO 5000 Product Page**: [NVIDIA RTX PRO 5000 Blackwell](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-5000/)
- **Official RTX PRO 5000 Datasheet**: [NVIDIA RTX PRO 5000 Blackwell Datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/professional-desktop-gpus/rtx-pro-5000-blackwell/workstation-datasheet-blackwell-rtx-pro-5000-gtc25-spring-nvidia-3658700.pdf)
- **PDF Conversion Draft**: `NVIDIA RTX Blackwell PRO GPU Architecture v1.0`
- **Cross-Architecture Reference**: [Hopper Hardware Specs](../../hopper/hardware-specs/hardware_specs_hopper.md) | [CDNA3 Hardware Specs](../../../amd/cdna3/mi300x/hardware-specs/hardware_specs_mi300x.md)
- **SM120 CuTeDSL Index**: [SM120 CuTeDSL](../ref-docs/cutedsl/README.md)
- **Downstream Experience**: [SM120 GDN Decode Optimization Record](../ref-docs/cutedsl/sm120-gdn-decode-fp32state-bf16qkv-optimization.md) | [SM120 NVFP4 FA Epilogue Optimization Record](../ref-docs/cutedsl/sm120-fused-fa-epilogue-nvfp4-bf16-optimization.md)
- **⚠️ Difference Note**: SM120 is a client Blackwell and does not have SM100's `tcgen05` / TMEM path; the same CUTLASS/CuTeDSL kernel on SM100 and SM120 must use a mainloop appropriate for each architecture's Tensor Core instruction path.
