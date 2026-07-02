# Tensor Core from Volta to Blackwell

A detailed technical walkthrough of NVIDIA Tensor Core architecture evolution from Volta through Blackwell, covering PTX programming models, each generation's MMA instructions, memory hierarchy changes, and the progression toward fully asynchronous execution.


**Last updated**: 2026-06-30

---

## 1. Performance Engineering First Principles

### 1.1 Amdahl's Law

For a fixed-size problem, Amdahl's Law defines the maximum speedup achievable through parallelization. Speedup is bounded by the serial fraction: maximum speedup = 1 / ((1-p) + p/S), where p is the parallelizable fraction and S is the parallel speedup factor.

### 1.2 Strong vs. Weak Scaling

- **Strong scaling**: More resources applied to the same problem (latency reduction)
- **Weak scaling**: Problem and resources grow proportionally (throughput maintenance)

Strong scaling benefits all problem sizes; weak scaling only guarantees throughput when problem size grows with resources.

### 1.3 Data Movement is the Cardinal Sin

Data movement is fundamentally slower than computation: modern DRAM access latency is tens of nanoseconds, while transistor switching achieves sub-nanosecond speeds. Since the 2000s, compute speed improvements have outpaced memory speed improvements, creating the "Memory Wall."

---

## 2. PTX Programming Model

Parallel Thread Execution (PTX) is a virtual ISA abstracting GPU generations. PTX programs describe kernel functions executed by GPU threads on CUDA Cores. Threads organize as grids of Cooperative Thread Arrays (CTAs/thread blocks), synchronized via `__syncthreads()`.

Memory state spaces:
- **Registers**: Per-thread private
- **Shared Memory**: Per-CTA shared
- **Global Memory**: Accessible by all threads

### 2.1 PTX Machine Model

GPUs are built around Streaming Multiprocessor (SM) arrays. Each SM maps threads to scalar cores (CUDA Cores) and manages 32-thread warps. Instruction dispatch selects a warp and issues instructions — this is SIMT (Single-Instruction, Multiple-Thread) execution.

SIMT vs. SIMD:
- **SIMD**: Programmer explicitly specifies vector width (e.g., AVX2 256-bit = 8×FP32)
- **SIMT**: Programmer writes scalar per-thread code; hardware transparently parallelizes 32 threads per warp

### 2.2 Streaming Assembler (SASS)

SASS is the architecture-specific ISA over which PTX is virtualized. Unfortunately, NVIDIA does not fully document SASS to protect competitive ISA details.

---

## 3. Volta: 1st Generation Tensor Core

### 3.1 Why NVIDIA Added Tensor Cores

GPU instruction dispatch consumes ~30 pJ, while a basic floating-point operation (HFMA) costs only ~1.5 pJ — a 20x overhead ratio. For matrix multiplication with massive FMA operations, this energy inefficiency is severe.

Solution: Complex instructions that amortize dispatch overhead. NVIDIA designed the Half-precision Matrix Multiply-Accumulate (HMMA) instruction — executed by dedicated Tensor Core hardware, first appearing in the 2017 Tesla V100.

Notable: Volta Tensor Cores were added in the final months before tape-out, demonstrating NVIDIA's rapid architecture adaptation.

### 3.2 MMA Instruction Overview

MMA computes D = A × B + C (A: M×K, B: K×N, C/D: M×N).

Execution flow:
1. Load A, B, C from shared memory to thread registers (each thread holds matrix fragments)
2. Execute MMA instruction — Tensor Core reads from registers, computes, writes results back
3. Store results from registers to shared memory

Multiple threads collaborate at each step, requiring synchronization.

### 3.3 Volta MMA Details

Tesla V100: 8 Tensor Cores per SM (grouped in pairs, 4 groups). Each Tensor Core completes 4×4×4 matrix multiply per clock = 1024 FLOPs/SM/cycle.

PTX `mma` instruction executes 8×8×4 matrix multiply, requiring an 8-thread quadpair. Warp (32 threads) is divided into 4 quadpairs (QP0: T0–T3 + T16–T19, etc.).

This distribution ensures each QP's two threadgroups access different register file banks simultaneously, avoiding bank conflicts and enabling full-bandwidth data delivery to Tensor Cores.

Data types: FP16 input + FP32 accumulation (mixed-precision training).

---

## 4. Turing: 2nd Generation Tensor Core

Enhanced Volta design with INT8 and INT4 precision support. Introduced warp-level synchronous MMA and Deep Learning Super Sampling (DLSS).

---

## 5. Ampere: 3rd Generation Tensor Core

### 5.1 Asynchronous Data Copy

Ampere introduced `cp.async` — direct asynchronous GMEM→SMEM copy bypassing registers. Previously (Volta), threads loaded GMEM→registers→SMEM, creating severe register pressure since MMA instructions already consume many registers.

`cp.async` directly fetches from DRAM to SMEM (optionally via L1 cache), freeing registers for MMA. PTX instruction: `cp.async`; SASS: `LDGSTS`.

### 5.2 Warp-Level Synchronous MMA

Ampere: 4 Tensor Cores per SM, 512 FLOPs/TC/cycle = 2048 dense FLOPs/SM/cycle (2x Volta). MMA now requires a **full warp (32 threads)** — simplifying layout and reducing register pressure.

Key additions:
- `ldmatrix`: Warp-level vectorized load matching Tensor Core layout (reduces address generation registers)
- **BF16** support: Same 8-bit exponent as FP32 with 7-bit mantissa; eliminates need for loss scaling

---

## 6. Hopper: 4th Generation Tensor Core

### 6.1 Thread Block Cluster

New hierarchy between CTA and grid: clusters correspond to SMs within a GPC. Cluster CTAs are co-scheduled; their shared memories form Distributed Shared Memory (DSMEM) accessible via dedicated SM-to-SM network without L2 cache traversal.

### 6.2 Tensor Memory Accelerator (TMA)

Dedicated hardware for large-scale async GMEM↔SMEM transfers. Single thread initiates TMA copy; TMA handles address generation, bounds checking, and multi-dimensional tensor addressing.

TMA supports **multicast**: single load distributes data to multiple SMs' shared memory within a cluster — reducing L2 and HBM traffic.

Caveat: TMA has higher latency for small requests due to address generation overhead. Best for 16-byte-aligned large blocks.

### 6.3 Warpgroup-Level Async MMA (WGMMA)

4 warps (128 threads) form a warpgroup, collaboratively executing one MMA. Supports shapes m64nNk16 (N: 8–256 in steps of 8). Compiled to SASS `GMMA`/`HGMMA` instructions.

Key innovation: Hopper Tensor Cores **load operand B directly from SMEM** (not registers), saving register space. Operand A can be in registers or SMEM; output D remains in warpgroup registers.

Data types: FP8 (E4M3/E5M2) with FP32 accumulation — though actual accumulation uses 22-bit fixed-point (limited dynamic range), requiring periodic CUDA Core accumulation (see DeepSeek-V3 Section 3.3.2).

---

## 7. Blackwell: 5th Generation Tensor Core

### 7.1 Tensor Memory (TMEM)

New 256 KB per-SM memory (128 lanes × 512 columns × 4 bytes). Access restricted: each warp accesses specific lanes; full warpgroup needed for complete coverage. Requires explicit programmer management (alloc/dealloc/data movement).

### 7.2 CTA Pair

Two CTAs whose cluster ranks differ only in the LSB form a CTA Pair, mapped to a TPC (2 SMs). Enables **shared input operands** between the pair, reducing SMEM capacity and bandwidth requirements.

### 7.3 5th-Gen MMA (tcgen05.mma)

**Completely eliminates register-stored matrices**: operands in SMEM and TMEM only.
- No complex register data layouts
- Freed registers for epilogue and other work
- **Single-thread semantics** — one thread dispatches MMA (warp no longer involved in dispatch)

**MMA.2SM variant**: Two SMs cooperate. Leader CTA's single thread dispatches. Layout A doubles M dimension; each SM loads different A/D halves; B is split and shared via DSMEM.

Supports **weight stationary** mode with collector buffer for operand reuse (convolution). Also supports microscaling formats (MXFP8/6/4) and NVIDIA's NVFP4.

### 7.4 Structured Sparsity

Ampere introduced 2:4 sparsity (2 zeros per 4 elements). At instruction level, this achieves 2x speedup. However, practical GEMM kernels on Hopper fall short of 2x due to: difficulty maintaining accuracy with structured pruning, insufficient cuSPARSELt optimization, and TDP limitations.

Blackwell introduces **pair-wise 4:8 sparsity** for NVFP4: every 8 elements split into 4 pairs, exactly 2 pairs must be non-zero. Despite appearing more relaxed than 2:4, the pair constraint offers no additional freedom for accuracy-preserving pruning.

---

## 8. Architectural Trends

### 8.1 Tensor Core Size Increases

NVIDIA aggressively scales individual Tensor Core size over core count. Rationale: matrix multiply has O(n³) compute vs. O(n²) data movement — arithmetic intensity grows linearly with scale. Larger cores exploit this better.

Trade-offs:
- More cores → **tile quantization** (uneven work distribution)
- Larger cores → **wave quantization** (underutilization on small trailing batches)

MMA shape growth: Volta 8×8×4 (8 threads) → Ampere 16×8×16 (32 threads) → Hopper m64×N×16 (128 threads) → Blackwell m256×N×16 (single thread dispatch).

### 8.2 Memory Size Increases

Shared memory grows each generation while register files stay constant — Tensor Core throughput doubling needs deeper staging buffers. Global memory latency increases (not decreases), requiring more buffered data.

Blackwell's SMEM doesn't grow beyond Hopper because 2-SM cooperation effectively doubles per-MMA SMEM capacity. TMEM introduction adds a new layer closer to Tensor Cores with higher energy efficiency.

Matrix D always resides in TMEM (highest access frequency: 2Kt reads + 2Kt writes per tile vs. single access for A/B).

### 8.3 Asynchrony of MMA Instructions

MMA evolved from synchronous to fully async at SASS level to overlap with LDSM instructions. In Ampere, the "1 LDSM + 2 HMMA" sequence created pipeline bubbles due to hardware interlocks. Hopper introduced async commit/fence for HGMMA. Blackwell achieves full asynchrony with explicit async TMEM operations.

### 8.4 Data Type Precision Reduction

Each generation adds lower precision: 16-bit → 8-bit → 4-bit. Deep learning tolerates low precision (especially inference). Lower precision: better energy efficiency, smaller die area, higher throughput.

Trade-off: FP64 removed in newer architectures; INT4 deprecated from Hopper; INT8 throughput reduced on Blackwell Ultra. This reflects delayed adoption cycles — Turing supported INT4, but practical INT4 LLM quantization methods emerged 4 years later.

### 8.5 Programming Model Evolution

Transition from high-occupancy (multi-CTA per SM) to **single-CTA occupancy** — NVIDIA bets on strong scaling for matrix multiplication. Single-CTA occupancy improves performance across all problem sizes (strong scaling), unlike multi-CTA which only helps as problems grow (weak scaling).

**Asynchronous execution** circumvents Amdahl's Law: overlapping data loads with MMA eliminates non-MMA bottlenecks. Evolution: Ampere (cp.async + arrive/wait), Hopper (TMA + async wgmma + improved barriers), Blackwell (fully async tcgen05 with mbarrier-based completion).

Software pipelining remains the key pattern across all architectures — staging GMEM→SMEM loads, SMEM→RF loads, and MMA execution in overlapping pipeline stages.


## Related

- [Async Global-to-Shared Memory Copy (CC 8.0+)](async-global-to-shared-copy.md)
- [FlashAttention 1–4: GPU Generational Evolution](flash-attention-1-to-4-gpu-evolution.md)
- [FlashInfer: Efficient and Customizable Attention Engine for LLM Inference](flashinfer-efficient-attention-engine.md)
- [GPU Architecture Deep Dive](gpu-architecture-deep-dive.md)
- [Memory-Bound Kernel Optimization: Hierarchical Reduction](hierarchical-reduction-memory-bound.md)
- [PTX Programming Model and Basics](ptx/ptx-programming-model.md)
- [PTX Core Instruction Set](ptx/ptx-instruction-set.md)
- [Composable Kernel (CK) Architecture Overview](../../amd/common/ck-architecture-overview.md)
