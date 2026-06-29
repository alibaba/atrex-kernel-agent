# Blackwell Matrix Multiplication Part 1: Fundamentals

Introduction to GPU architecture fundamentals, Tensor Core evolution from Ampere through Blackwell, and a baseline 4-line matrix multiplication in Mojo achieving 5 TFLOPS on B200.

---

## 1. Series Overview

This series demonstrates how to:
- Write a high-performance GPU kernel on Blackwell that matches NVIDIA cuBLAS
- Leverage Mojo's features to simplify kernel code

Previous work exists for optimizing Ampere and Hopper GPUs, but no blueprint existed for NVIDIA Blackwell optimization.

Series structure: Part 1 (Fundamentals) → Part 2 (Hardware Instructions) → Part 3 (85% SOTA) → Part 4 (Beyond SOTA).

---

## 2. Why Matmul Matters

All large language models (Llama / Qwen / Deepseek / Claude / ChatGPT / Gemini) are built on matrix multiplication (MLP / Attention / MoE).

> Llama 8B running on 2× B200 with FP8 spends over 83% of model execution time on some form of matrix multiplication. Even a 10% improvement in matmul performance yields approximately 8% end-to-end speedup.

---

## 3. GPU Architecture from a Hardware Architect's Perspective

- A GPU consists of multiple Streaming Multiprocessors (SMs), a shared L2 cache, and global memory
- Threads are grouped into CTAs (Cooperative Thread Arrays); each CTA is scheduled to one SM
- Within an SM, threads access: registers (per-thread private), shared memory (CTA-visible)
- Since Hopper, multiple CTAs can form a cluster; CTAs within a cluster can access each other's shared memory

### 3.1 Tensor Core

Each SM has 4 Tensor Cores. A single MMA instruction computes a matrix multiply-accumulate (e.g., 64×128 tile).

- **Hopper**: WGMMA maximum shape **64×256×16**, requires 4 warps (one warpgroup), results stored in registers
- **Blackwell**: tcgen05 extends to **256×256×16** (across 2 SMs), introduces Tensor Memory to reduce register usage

### 3.2 Thread Block Clusters

On Hopper and Blackwell, programmers can group multiple CTAs into a cluster:
- Cluster guarantees all CTAs are scheduled to physically connected SMs within the same GPC
- CTAs within a cluster can bypass global memory and directly access each other's shared memory (DSMEM)
- Execution hierarchy: grid → cluster → CTA → warp → thread

---

## 4. Three-Generation Comparison

| Metric | A100 | H100 | H200 | B100 | B200 |
|--------|------|------|------|------|------|
| Peak memory bandwidth | 1.0× | 1.6× | 2.4× | 3.9× | 3.9× |
| NVLink bandwidth | 1.0× | 1.5× | 1.5× | 3.0× | 3.0× |
| Peak BF16 TFLOPS | 1.0× | 3.2× | 3.2× | 5.6× | 7.2× |
| Peak FP8 TFLOPS | — | 1.0× | 1.0× | 1.8× | 2.3× |

### 4.1 Ampere Specifications

108 SM / 4 TC per SM / 80 GB HBM at 2.0 TB/s / 192 KB SMEM+L1 per SM / 65,536 registers per SM / 40 MB L2 / async cp.async

### 4.2 Hopper Specifications

132 SM / 4 TC per SM / 80 GB HBM at 3.35 TB/s / 256 KB SMEM+L1 per SM / 65,536 registers per SM / 50 MB L2 / TMA engine

### 4.3 Blackwell Specifications

**148 SM** / 4 TC per SM / 7.672 TB/s HBM / **228 KB** SMEM+L1 per SM / 65,536 registers per SM / **192 MB L2** / 5th-gen TC / **256 KB Tensor Memory**

---

## 5. Optimization Paradigm Evolution

### 5.1 Pre-Ampere

Memory operations block compute: load → wait → compute → wait → store. Requires double-buffering + multiple CTAs per SM for overlap.

### 5.2 Ampere (cp.async)

- ✅ Input loads overlap with compute
- ❌ CTA launch overhead remains

### 5.3 Hopper (TMA + WGMMA + Persistent Kernel)

- **TMA**: Dedicated hardware unit for asynchronous multi-dimensional tensor tile transfers
- **WGMMA**: Asynchronous MMA that overlaps not just with memory but also with core compute
- **Persistent kernel**: CTAs remain resident on SMs, processing multiple tiles without returning to host
- ✅ Reduces CTA launch overhead; enables cross-tile data transfer overlap
- ❌ WGMMA consumers occupy many registers, causing contention between Tensor Core and ALU

### 5.4 Blackwell (tcgen05 + TMEM)

tcgen05 instructions + Tensor Memory: MMA results stored in dedicated TMEM hardware, breaking WGMMA's register dependency.

**Three-stage pipeline** (tile N+1 load / tile N compute / tile N-1 store, all concurrent):
1. **Load**: TMA engine fetches inputs
2. **Compute**: MMA computes and writes to TMEM
3. **Store**: TMEM → global memory

- ✅ Pipelined store stage
- ❌ Tensor Memory supports only a very limited instruction set

---

## 6. Four-Line Matmul Implementation

- Create a 2D thread grid covering the entire C matrix
- Each thread computes one element of C based on its thread ID

```mojo
fn matmul(A: LayoutTensor[...], B: LayoutTensor[...], C: LayoutTensor[...]):
    let row = global_idx.y
    let col = global_idx.x
    if row < M and col < N:
        var acc: Float32 = 0.0
        for k in range(K):
            acc += Float32(A[row, k]) * Float32(B[k, col])
        C[row, col] = BFloat16(acc)
```

---

## 7. Data Type Selection

- **BF16**: 1-bit sign + 8-bit exponent + 7-bit mantissa; max positive value 3.39×10³⁸ (FP16 only 6.55×10⁴)
- **FP32 accumulation**: BF16 accumulating many values causes rounding error to grow rapidly; Tensor Cores also accumulate BF16 inputs in FP32

---

## 8. Performance

The 4-line Mojo implementation achieves **5 TFLOPS** (baseline). cuBLAS SOTA: **1763 TFLOPS**. B200 peak: **2250 TFLOPS**. The remaining 3 parts show how to close this gap.
