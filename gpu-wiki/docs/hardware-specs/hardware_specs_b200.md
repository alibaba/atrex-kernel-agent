# NVIDIA B200 GPU Hardware Compute Specification Table

**Last Updated**: 2026-06

---

## Quick Reference Table

Usage: Look up peak TFLOPS/TOPS based on the kernel's **primary compute type** and **target GPU** for compute utilization calculations.

```
utilization = actual TFLOPS / peak TFLOPS × 100%
```

> This document covers the NVIDIA B200 data center GPU based on the Blackwell architecture (GB200) with `sm_100` (Compute Capability 10.0). Specifications are compiled from NVIDIA official product pages, DGX B200 datasheets, and the Blackwell architecture whitepaper.

---

## NVIDIA B200 Data Center GPU (GB200 / sm_100)

| Precision | Peak TFLOPS / TOPS | With 2:4 Sparsity | Use Case |
|------|-------------------|---------|---------|
| **FP64 (CUDA Core)** | 37.0 | — | Scientific computing, double-precision HPC |
| **FP32 (CUDA Core)** | 37.5 | — | General-purpose compute, element-wise, reduction |
| **TF32 (Tensor Core)** | 1,100.0 | 2,200.0 | Training / inference matrix multiply |
| **FP16 / BF16 (Tensor Core)** | 2,250.0 | 4,500.0 | Training / inference |
| **FP8 (Tensor Core, FP32 Accumulate)** | 4,500.0 | 9,000.0 | Inference / low-precision training |
| **FP4 (Tensor Core, FP32 Accumulate)** | 9,000.0 TOPS | 18,000.0 TOPS | NVFP4 / MXFP4 quantized inference |
| **INT8 (Tensor Core)** | 4,500.0 TOPS | 9,000.0 TOPS | Quantized inference |

### Memory Specifications

| Parameter | Value |
|------|------|
| VRAM | 180 GB HBM3e (DGX B200: 1,440 GB total / 8 GPUs) |
| Memory Bandwidth | 8.0 TB/s |
| L2 Cache | 126 MB |
| TDP | 1,000 W (max 1,200 W) |
| Form Factor | SXM6 |
| NVLink 5 | 18 links × 100 GB/s = 1.8 TB/s (bidirectional) |
| PCIe | Gen 5.0 |

### Compute Units

| Parameter | Value |
|------|------|
| Graphics Processing Clusters (GPCs) | 8 |
| Streaming Multiprocessors (SMs) | 148 (dual-die, 74 SMs per die; each die has 80 physical SMs, 74 enabled for yield optimization) |
| CUDA Cores | 18,944 |
| Tensor Cores (5th gen) | 592 |
| RT Cores (4th gen) | 148 |
| Texture Units | 592 |
| GPU Boost Clock | 2,100 MHz |
| Base Clock | 1,200 MHz |
| CUDA Cores per SM | 128 |
| Tensor Cores per SM | 4 (5th gen) |
| RT Cores per SM | 1 |
| Register File per SM | 256 KB |
| L1 Data Cache / Shared Memory per SM | 128 KB physical pool |
| Total Register File | 37,888 KB |
| Total L1 Data Cache / Shared Memory | 18,944 KB |

---

## Blackwell Data Center Architecture Key Parameters (sm_100)

These parameters influence optimization decisions:

### Execution Units

| Parameter | Value | Impact |
|------|------|------|
| Compute Capability | 10.0 (`sm_100`) | Blackwell data center lineup |
| Warp size | 32 threads | 32 threads per warp |
| CUDA Cores per SM | 128 | Basis for FP32/INT32/element-wise throughput |
| Tensor Cores per SM | 4 (5th gen) | Basis for FP16/BF16/TF32/FP8/FP4 matrix throughput |
| RT Cores per SM | 1 (4th gen) | Ray tracing acceleration |
| Register File per SM | 256 KB | Core constraint for register pressure and occupancy |
| Max Registers per Thread | 255 | Register spill threshold |
| L1 / Shared Memory per SM | 128 KB physical pool | L1 and shared memory share physical SRAM |
| Warp Schedulers per SM | 4 | Supports concurrent warp execution |

### Key Differences from SM120 Client Blackwell

| Feature | SM100 (B200 Data Center) | SM120 (RTX PRO / GeForce) | Optimization Implication |
|------|------|------|------|
| Tensor Core Instruction Route | `tcgen05.mma` / UMMA | `mma.sync` / warp-level MMA | SM100 uses UMMA fast path, not portable to SM120 |
| TMEM (Tensor Memory) | Supported | Not supported | SM100 accumulators can reside in TMEM |
| TMA (Tensor Memory Accelerator) | Supported | Supported | Bulk global-to-shared memory transfers |
| `cp.async` | Supported | Supported | Multi-stage G2S pipelines |
| FP4 / FP6 | Supported | Supported | Both support NVFP4 block-scaled MMA |
| Memory Type | HBM3e (8.0 TB/s) | GDDR7 (1.3-1.8 TB/s) | SM100 significantly higher bandwidth ceiling |

> **Key Reminder**: `sm_100` is data center Blackwell with full `tcgen05` / TMEM / UMMA support. Optimizations should leverage the "TMA + UMMA + TMEM accumulator" fast path for maximum throughput.

### Memory Hierarchy

| Level | Size | Bandwidth/Latency | Notes |
|------|------|----------|------|
| Registers | 256 KB / SM, up to 255 regs/thread | Fastest | High register pressure rapidly reduces occupancy |
| Shared Memory | 128 KB L1/SMEM physical pool | High throughput | Bank conflicts impact performance |
| L1/TEX Cache | Shares physical SRAM with Shared Memory | Medium | Automatic caching of global loads |
| L2 Cache | 126 MB | Medium | Very large L2 benefits working set locality |
| HBM3e | 180 GB | 8.0 TB/s | Extremely high bandwidth; still a bottleneck for pure streaming kernels |

### Tensor Core / MMA Programming Tips

| Type | SM100 Path | Notes |
|------|----------|------|
| FP16 / BF16 GEMM | `tcgen05.mma` (UMMA) | Warp-group level MMA with TMEM accumulators |
| TF32 GEMM | `tcgen05.mma` (UMMA) | TF32 path with FP32 accumulate |
| FP8 GEMM | Block-scaled UMMA | Handle scale factors and mixed-input types |
| FP4 GEMM | NVFP4 / MXFP4 block-scaled UMMA | e2m1 packed data with e4m3/e8m0 scale factors |
| Attention / FA | Dedicated SM100 UMMA fast path | Leverage TMEM for persistent accumulators |

---

## Roofline Analysis Assistance

### Computing Arithmetic Intensity

```
AI = FLOPs / Bytes_transferred
```

### Identifying Bottlenecks

```
if AI < (peak TFLOPS / peak_bandwidth TB/s):
  -> Memory Bound (bandwidth bottleneck)
  -> optimization: improve data reuse, tiling, L2 locality, reduce memory traffic
otherwise:
  -> Compute Bound (compute bottleneck)
  -> optimization: maximize Tensor Core utilization, reduce stalls, tune occupancy
```

**Typical Ridge Points**:

| GPU | Precision | Ridge Point (FLOPs/Byte) |
|-----|------|--------------------------|
| B200 | FP16/BF16 Tensor | 2,250 / 8.0 ≈ **281** |
| B200 | FP8 Tensor | 4,500 / 8.0 ≈ **563** |
| B200 | FP4 Tensor | 9,000 / 8.0 ≈ **1,125** |
| B200 | FP32 CUDA | 37.5 / 8.0 ≈ **4.7** |
| B200 | TF32 Tensor | 1,100 / 8.0 ≈ **138** |

> **Optimization Implication**: With 8.0 TB/s HBM3e bandwidth, the B200 has an exceptionally high memory ceiling. Most GEMM/attention kernels will be compute-bound; however, pure streaming / element-wise / quantization epilogue kernels can still hit the memory wall. The very low FP32 CUDA ridge point (4.7) means even simple element-wise kernels with modest reuse quickly become compute-bound.

---

## How to Choose Peak Compute

1. **Identify the primary compute type**: What computation dominates the kernel?
   - Tensor Core / MMA intensive → use FP16/BF16/FP8/FP4 Tensor Core TFLOPS/TOPS
   - Element-wise / reduction intensive → use FP32 CUDA Core TFLOPS
   - Memory-bound transport / quant epilogue intensive → prioritize bandwidth roofline, not Tensor Core peak
2. **Identify the target GPU**:
   - B200 → use 37.5 FP32, 2,250 BF16, 4,500 FP8, 9,000 FP4, 8.0 TB/s
3. **Mixed compute**: If the kernel has both Tensor Core computation and element-wise epilogue, compute separate rooflines for the mainloop and epilogue; do not use a single peak to mask bottlenecks.
4. **Source priority**: Official NVIDIA DGX B200 specifications and Blackwell architecture whitepaper.

## Related Documents

- **Cross-Architecture Reference**: [Hopper Hardware Specs](hardware_specs_hopper.md) | [Blackwell GeForce/RTX PRO Specs](hardware_specs_sm120.md) | [B300 Hardware Specs](hardware_specs_b300.md)
- **Cross-Vendor Reference**: [MI300X Hardware Specs](hardware_specs_mi300x.md) | [MI308X Hardware Specs](hardware_specs_mi308x.md) | [MI355X Hardware Specs](hardware_specs_mi355x.md)
- **Blackwell Tuning Guide**: [NVIDIA CUDA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)
- **Official Product Page**: [NVIDIA DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/)
- **Official System Guide**: [DGX B200 introduction](https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html) — 8 GPUs provide 1,440 GB total GPU memory
- **⚠️ Architecture Note**: B200 (SM100) uses the `tcgen05` / TMEM / UMMA path which is NOT available on SM120 client Blackwell GPUs. Kernel code designed for B200's UMMA path must be adapted when targeting RTX PRO / GeForce Blackwell.
