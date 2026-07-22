# NVIDIA CUDA Toolkit Now Supports Blackwell Architecture

Overview of CUDA Toolkit 12.8 as the first release with full Blackwell support across the developer suite, including profiling tools, libraries, and compilers.

---

## 1. Overview

CUDA Toolkit 12.8 is the **first** version to support NVIDIA Blackwell architecture across the entire developer suite (performance tools, profilers, libraries, compilers). Key coverage areas:

- Blackwell architecture support
- CUDA Graphs conditional node enhancements
- Blackwell CUTLASS kernels for LLMs
- Nsight developer tools updates
- Math libraries / compiler updates
- Accelerated Python

---

## 2. Blackwell Key Capabilities

- **20.8 billion transistors**, over 2.5x Hopper
- Second-generation Transformer Engine + custom Tensor Cores (accelerating LLM/MoE training and inference)
- Hardware decompression: LZ4 / Snappy / Deflate
- NVLink and NVLink Switches for accelerated inter-GPU communication in trillion/multi-trillion parameter models

---

## 3. CUDA Graphs: Runtime Kernel Selection with 2x Speedup

CUDA 12.8 adds two new conditional node types: **IF/ELSE composite nodes** and **SWITCH nodes**. Previously, kernel selection required CPU fallback; now it can run entirely on GPU.

| Impact | Benefit |
| --- | --- |
| Inference (DeepSeek-R1 and similar reasoning models) | Runtime kernel selection +2x, improved token generation rate |
| Training | Reduced CPU involvement → sustained Tensor Core throughput → higher MFU |

---

## 4. CUTLASS 3.8 for Blackwell

- GEMM achieves **98%** of peak performance
- **Grouped GEMM (common in MoE inference)** FP4 delivers **5x** improvement over H200 FP16 (for DeepSeek-V3, R1 MoE)
- New data types: MX narrow precision + NVIDIA FP4

---

## 5. Nsight Compute 2025.1 (First Official Blackwell Support)

- **Tensor Memory visualization in Memory Chart**
- Tensor Core performance data surfaced
- Range Profiling major improvements:
  - Source-level metrics within ranges (executed instructions + memory accesses)
  - Range-guided analysis rule evaluation
- New kernel stack size reporting and custom tooltips
- **Compute Sanitizer adds Python call stack support**
- **PTXAS `-g-tmem-access-check` flag**: Blackwell Tensor Core MMA guardrails that catch common errors (accessing unallocated TMEM / invalid address / invalid allocator)

---

## 6. Math Libraries

- **cuBLAS**: MX 4-bit/8-bit mixed-precision matrix multiplication (CC >= 10.0); preliminary CUDA-in-Graphics (CIG) support for Windows x64 + Ampere/Blackwell GeForce GPUs
- cuSOLVER: New `zsytrf/zsytrs` (pivotless complex symmetric direct solver)
- nvJPEG / NPP support for Tegra / DRIVE Thor

---

## 7. cudaStreamGetDevice and Compiler Updates

- New `cudaStreamGetDevice` simplifies multi-device applications
- GCC 14 host compiler support
- Blackwell default advanced optimizer based on LLVM 18
- `nvdisasm` supports JSON-format SASS disassembly

---

## 8. Accelerated Python (Beta)

- `cuda.core` early prototype + CUDA bindings moved to `cuda.bindings`
- CCCL parallel/cooperative algorithm Python prototypes, enabling thread-level parallelism expression in pure Python
- CuPy Blackwell patch validated for full release

---

## 9. Architecture Lifecycle Changes

Maxwell / Pascal / Volta are now considered **feature-complete**. The next major version will remove offline compilation support from compilers (nvcc / nvrtc / nvjitlink). Production applications can continue running within a 3-year LTS support window via LTS drivers.
