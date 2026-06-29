# Blackwell Matrix Multiplication Part 2: Hardware Feature Optimization

Loop tiling, TMA loads, tcgen05 MMA with Tensor Memory, swizzle patterns for bank-conflict-free shared memory access, and TMA store — building from 5 TFLOPS to 293 TFLOPS on B200.

---

## 1. Performance Roadmap

Focus on a specific matmul shape: A is M×K, B is K×N (transposed), C is M×N, with M=N=K=4096. The naive 4-line kernel requires two global memory loads and one store per FMA. Optimization is about exploiting the GPU memory hierarchy to avoid or hide these accesses.

---

## 2. Kernel 1: Loop Tiling

Loop tiling loads small blocks (tiles) into faster memory (shared memory), allowing the processor to execute all necessary computation on that block before returning to slow main memory.

- Partition A into BM×BK tiles, B into BN×BK tiles (BM×BN×BK = 64×64×64)
- Each tile pair fits in shared memory: 2 × 64×64 × 2B = 16 KB (well under 228 KB available)
- Iterate K/BK times, accumulating partial results
- Write final output block to global memory once

---

## 3. Kernel 2: TMA and Tensor Core

### 3.1 Loading Tiles to Shared Memory via TMA

The Tensor Memory Accelerator (TMA) is a dedicated hardware unit introduced in Hopper for asynchronous data transfer between global and shared memory. A tensor map (128 bytes) encodes shape, stride, and global memory address.

Key mechanism:
- A single thread (`elect_one_thread`) initiates the async copy
- A memory barrier (`tma_mbar`) tracks completion via phase-flipping protocol
- When transferred bytes reach the expected count, the barrier phase flips and threads proceed

### 3.2 Core Matrices

Tensor Cores view data as groups of 8×16-byte blocks — the **core matrix** (8×8 elements for BF16). The TMA actually loads 64×8 sub-tiles (one column of 8 core matrices at a time), requiring 8 reads to fill a 64×64 block. The Mojo library's `async_copy` abstracts this complexity.

### 3.3 Issuing MMA Instructions

Blackwell's 5th-generation Tensor Core (`tcgen05`) brings three improvements:
1. Single-SM shape up to **128×256×16** (2× Hopper's 64×256×16)
2. **2-SM tcgen05.mma** supports up to 256×256×16
3. **Tensor Memory (TMEM)** stores results, eliminating register file pressure

TMEM: 256 KB on-chip memory, 128 channels × 512 columns × 4 bytes/element. Allocation granularity: 32 columns minimum.

The `tcgen05.mma` instruction is asynchronous (like TMA), guarded by memory barriers. Configuration is encoded in descriptors:
- **Instruction descriptor (idesc)**: Shape, dtype, layout (constant across iterations)
- **Shared memory descriptors (adesc, bdesc)**: SMEM layout/address (changes per K-slice iteration)

### 3.4 TMEM → Registers

The only way to move data out of TMEM is through registers via `tcgen05_ld`. Each warp reads 16 channels; the full warpgroup (4 warps) covers all 64 channels. Each thread loads 256 bits per iteration (8 FP32 elements), repeated BN/8 times to extract the complete tile.

### 3.5 Registers → Global Memory

Each thread block handles a 64×64 output tile. Mojo's `LayoutTensor` provides per-thread views via `.tile()` and `.distribute[Layout.row_major(8, 4)]`, creating precise partitions matching the `tcgen05.ld.16x256b` element-to-thread mapping.

### 3.6 Shared Memory Setup

The SMEM stack contains: input tiles, memory barriers, and TMEM allocation space. Offsets are computed from the dynamic shared memory base address.

**Performance: 155.0 TFLOPS** — 28× improvement over naive, but only 8.7% of cuBLAS.

---

## 4. Kernel 3: Swizzle

### 4.1 Bank Conflicts

Shared memory consists of 32 consecutive 4-byte-wide banks. Each bank serves one request per cycle. When two threads access the same bank at different addresses, a **bank conflict** occurs — execution serializes across multiple cycles.

For the 128-byte canonical layout (BM×BK with BK=64), the first 8 rows of a core matrix all map to banks 0-3, causing an 8-way bank conflict.

### 4.2 Swizzle Solution

Swizzle uses bitwise XOR (`^`) to redistribute data across banks. The pattern `Swizzle<3, 4, 3>` (128-byte swizzle) works as:
- First 3: 2³ = 8 rows (core matrix height)
- Middle 4: 2⁴ = 16 bytes (core matrix width = 8 elements × 2B)
- Last 3: 2³ = 8 groups of 16B spanning all 32 banks (128B)

The XOR pattern ensures no two elements within the same core matrix fall in the same bank.

### 4.3 Implementation

Code changes are minimal — swizzle support is built into the LayoutTensor and instruction APIs:

```
# Only need to tell TMA and tcgen05.mma which swizzle mode to use
```

**Performance: 288.3 TFLOPS** — 87% improvement. Bank conflicts were nearly halving performance. Now at 16.4% of cuBLAS.

---

## 5. Kernel 4: Pack Output in SMEM + TMA Store

### 5.1 Problem

Previous kernel writes 4-byte stores (two BF16 values) — too granular given Blackwell's 32-byte maximum store width.

### 5.2 Solution: stmatrix + TMA Store

1. Load TMEM → registers (FP32)
2. Convert FP32 → BF16 in registers
3. Use `stmatrix` instruction to store core matrices (8×16B) into shared memory with swizzle (128B swizzle for BN×2B=128B)
4. Issue TMA store: async copy from SMEM → GMEM

TMA store differences from TMA load:
- Multiple threads can issue TMA stores in parallel
- Uses `commit_group()` / `wait_group[N]()` for completion tracking
- `fence_async_view_proxy` ensures prior SMEM writes are visible to TMA

For BN=128, two TMA stores are issued by two threads to maximize parallelism.

**Performance: 293.6 TFLOPS** — marginal improvement (0.7% slower due to remaining global memory bottleneck). The real value of TMA store is enabling future pipelining and operation overlap.

NCU analysis confirms both Kernel 3 and 4 have very low compute and memory throughput utilization — the kernel is fundamentally memory-bound at this stage.

---

## 6. Appendix: Descriptors

The shared memory descriptor encodes:
- **LBO (Leading dimension Byte Offset)**: Bytes between adjacent core matrices along K (e.g., BM×16B = 1024B)
- **SBO (Stride dimension Byte Offset)**: Bytes between adjacent core matrices along M/N (e.g., 8×16B = 128B)

The instruction descriptor (idesc) is 32-bit, encoding sparsity, dtype, transpose, and other flags.

## 7. Appendix: Swizzle Math

Given `Swizzle(bits, base, shift)`: extracts bits at positions [base+shift, base+shift+bits) as a mask, XORs with bits at positions [base, base+bits). For the 128B swizzle (bits=3, base=4, shift=3), it extracts address bits 7-9 and XORs with bits 4-6, producing the conflict-free access pattern.
