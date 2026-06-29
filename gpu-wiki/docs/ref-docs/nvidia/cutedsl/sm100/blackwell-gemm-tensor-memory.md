# CUTLASS Tutorial: Blackwell GEMM with Tensor Memory

A detailed walkthrough of Blackwell's fifth-generation Tensor Core MMA instructions (`tcgen05.mma`) and the dedicated Tensor Memory (TMEM) they operate on, based on the Colfax Research CUTLASS tutorial series.

---

## 1. Introduction

NVIDIA Blackwell architecture introduces features that significantly change the implementation of GEMM kernels. This series explores these new capabilities and demonstrates how to write CUTLASS GEMM kernels leveraging them:

- **Part 1 (this document):** Fifth-generation Tensor Core MMA instructions and Tensor Memory.
- **Part 2:** Using clusters, including TMA multicast and the Blackwell CTA-pair concept.
- **Part 3:** Low-precision MMA with sub-byte data types and native block-scaling support.

The goal is to explain how to update a Hopper GEMM kernel to run on Blackwell, or write a Blackwell GEMM kernel from scratch.

Note: Consumer Blackwell (compute capability 12.0) differs from datacenter Blackwell (compute capability 10.0) in several ways — notably, consumer parts lack Tensor Memory. This series covers datacenter Blackwell only.

## 2. Blackwell MMA Overview

Hopper's WGMMA instructions (`wgmma.mma_async` in PTX) are deprecated on Blackwell. The replacement is `tcgen05.mma`, referred to as **UMMA** in CUTLASS.

Like WGMMA, UMMA is an asynchronous instruction computing either D = A*B + D or D = A*B.

Key differences from WGMMA:

- Supports low-precision data types including FP4 and FP6, with higher throughput at all precisions
- Native block-scaling support
- Dedicated **Tensor Memory** for UMMA accumulation
- Two adjacent CTAs within an SM cluster (a **CTA pair**) can cooperatively execute UMMA across two SMs
- Unlike WGMMA, **UMMA is launched by a single thread** — even with two CTAs, only one thread in one CTA issues the instruction

## 3. Tensor Memory

**Tensor Memory (TMEM)** is on-chip memory dedicated to Tensor Core operations. Its primary purpose is to replace registers for fifth-generation Tensor Core usage.

UMMA expects the following input sources:

- Operand A: TMEM or SMEM
- Operand B: SMEM only
- Accumulator: TMEM only

This means UMMA does not require registers for data, reducing register pressure. Combined with single-thread launch, MMA is further decoupled from the CTA's main execution flow.

**TMEM is 256 KB per SM**, organized two-dimensionally with 512 columns and 128 rows (channels), each cell being 32 bits. This inherent 2D structure is reflected in the 32-bit address: bits 31-16 encode the channel ID, bits 15-0 encode the column.

TMEM is dynamically allocated using `tcgen05.alloc`. Allocation granularity is by column — allocating one column allocates all channels in that column. The number of columns must be a power of two and at least 32. Explicit deallocation via `tcgen05.dealloc` is required. Both instructions must be called from a single warp, and the same warp should perform allocation and deallocation.

The `tcgen05.alloc` instruction stores the 32-bit base address of the allocated region into a specified shared memory location. This base address is then set as the offset for the UMMA accumulator tensor.

Data typically enters TMEM through UMMA operations and is explicitly moved out to registers via `tcgen05.ld` for post-processing. Manual loading into TMEM is possible via `tcgen05.cp` (from SMEM) or `tcgen05.st` (from registers), but explicit load/store access patterns are restricted: **each warp in a warp group can only access 32 channels** (warp 0 → channels 0-31, warp 1 → channels 32-63, etc.).

## 4. tcgen05.mma Instruction

For dense FP16 GEMM with FP32 accumulation (`.kind::f16`), the single-CTA MMA shapes are:

- **64 x N x 16** (N is a multiple of 8)
- **128 x N x 16** (N is a multiple of 16)

In both cases, N is at most 256.

The largest UMMA atom **128 x 256 x 16 is twice the largest WGMMA atom**. Its accumulator occupies exactly half of TMEM, allowing multiple UMMA atoms to be pipelined without performance loss.

Operand descriptors `a-desc` and `b-desc` are shared memory descriptors similar to those used by WGMMA — 64-bit values packing address, layout, and swizzle information for matrices stored in SMEM.

In addition to matrix descriptors, `tcgen05.mma` requires an instruction descriptor (`idesc`): a 32-bit metadata value encoding data type and sparsity information. Two bits control transpose and/or negation of A and B. The `enable-input-d` parameter switches between clearing the accumulator (D = A*B) and preserving it (D = A*B + D).

## 5. tcgen05.ld Instruction

Three memory movement instructions exist under tcgen05: `ld`, `st`, and `cp`. The `ld` instruction copies data from TMEM to RMEM (registers).

`tcgen05.ld` is a **warp-level instruction** — all threads in the warp must execute the same instruction and synchronize within the warp, similar to the earlier `ldmatrix` instruction.

It supports various data movement shapes expressed as `{channels}x{bits}`; a common example is `32x32b` (32 channels, 32-bit per channel within a single warp). The `.num` component indicates repetition count along the column dimension. In a single instruction, a warp can load at most `lanes * bits * num <= 128 kb` (16 KB), corresponding to 128 32-bit registers per thread.

Each warp can only access 32 of the 128 TMEM channels.

## 6. CUTLASS UMMA Interface

The CuTe interface consists of `MMA_Atom` (PTX instruction wrapper) and `MMA_Traits` (CuTe layouts and metadata).

`SM100_MMA_F16BF16_SS` is the atom used in the first CuTe Blackwell code example. Much of the information maps directly to `tcgen05.mma` concepts: SMEM descriptors for A and B, TMEM layout for D, and the instruction descriptor.

**The ThrID layout has changed significantly.** Previously, ThrID mapped logical indices of threads cooperatively executing an MMA to physical thread IDs. For warp-level MMA it was `Layout<_32>`; for WGMMA, `Layout<_128>`. Here it is `Layout<_1>`. Since the instruction is single-threaded, all thread layouts are repurposed as layouts of CTAs cooperatively executing the MMA.

Atom naming convention:

- **SM100_MMA**: UMMA instruction for SM100
- **F16BF16**: Accepted input types for A and B (fp16/bf16)
- **SS**: Both A and B in SMEM (TS = A in TMEM, B in SMEM)

## 7. CUTLASS Example: Simple UMMA

The example is organized into five parts. The recurring theme: **partitioning is done across CTAs, not across threads.**

### 7.1 GMEM Tiling and Slicing

A `tiled_mma` object is created, then tiler dimensions are chosen based on it. The factor of 4 in MMA_K means each GMEM-to-SMEM copy corresponds to 4 UMMA calls.

Since "thread layouts" are repurposed as "peer CTA layouts," the `tiled_mma` is sliced by CTA peer ID rather than thread ID. The sliced MMA is called `cta_mma` (analogous to `thr_mma` on Hopper).

Each CTA pair consists of a pair of adjacent CTAs in the cluster. The `cluster_layout_vmnk` creates a `cluster_shape` aware of CTA pairs; `AtomThrID` is 1 or 2 depending on whether the UMMA atom uses CTA pairs.

### 7.2 SMEM Layouts and Swizzle

For operand A, the target tensor `tCsA` is organized as shape `(MmaA, NumMma_M, NumMma_K) = ((_128,_16),_1,_4)`.

CUTLASS utility functions create the required shapes. The layout should be swizzled — `Layout_K_SW128_Atom<TypeA>` is a 128-byte-wide swizzle pattern for K-major A. Swizzle width is determined by the tile size in the contiguous dimension.

SharedStorage also holds a 32-bit address for the TMEM base address. The example uses auto-vectorized `cute::cooperative_copy` for GMEM to SMEM transfer.

### 7.3 Input and Output Descriptors

UMMA accepts its first input from SMEM or TMEM; the second input must be in SMEM; the accumulator must be in TMEM.

Descriptors are created using `cta_mma`'s `make_fragment` method. Each MMA atom gets one descriptor, tiled by `(NumMma_M, NumMma_K) = (_1, _4)`.

The accumulator tensor is a TMEM-backed tensor whose layout uses stride 65536 (`1 << 16`). This reflects TMEM's 32-bit addressing: upper 16 bits = channel, lower 16 bits = column.

### 7.4 GEMM Loop and Synchronization

UMMA is asynchronous, requiring explicit synchronization via mbarrier abstractions.

The gemm call and loop structure resemble the Hopper example. The key difference: **only one warp issues UMMA**. Calling `cute::gemm` from a single thread would cause deadlock.

`UMMA::ScaleOut::Zero` instructs UMMA to overwrite TMEM. After the first k_block iteration, it is set to `UMMA::ScaleOut::One` to accumulate results.

### 7.5 Copying Out of TMEM

After all MMAs complete, the `tcgen05.ld` instruction copies accumulator results from TMEM to registers. CUTLASS abstracts this as a copy atom using `SM100_TMEM_LOAD_32dp32b1x`.

Unlike CTA-level operations, **this returns to warp/thread-level operation** — data must reach registers for the epilogue. `make_tmem_copy` is hardcoded to use 4 warps (1 warp group). Based on warp index modulo 4, specific TMEM regions are accessible only to the corresponding warp.

## 8. TMEM Allocation and Deallocation

The CuTe helper class `cute::TMEM::Allocator1Sm` provides an interface for `tcgen05.alloc` and `tcgen05.dealloc`.

A single warp performs allocation by passing the column count and a pointer to a 32-bit shared memory location; the allocate method stores the 32-bit address of the allocated TMEM region. Although this MMA instruction only requires 256 columns, the example allocates all 512 columns for simplicity.

The `release_allocation_lock` method wraps `tcgen05.relinquish_alloc_permit`, guaranteeing that the CTA will not perform further TMEM allocations and allowing future CTAs to queue for the same SM.

For TMEM debugging, nvcc provides the `--g-tensor-memory-access-check` flag. When enabled, the kernel reports errors at runtime for any uninitialized or out-of-bounds TMEM access.

## 9. Conclusion

The main concepts and overall structure of CUTLASS GEMM kernels have not changed with Blackwell. The two primary differences are:

1. UMMA atoms operate at CTA level rather than thread level, requiring updates to TiledMMA structures and the synchronization model (e.g., a single thread asynchronously issues UMMA).
2. UMMA accumulates into Tensor Memory, which must be manually managed. A specialized TiledCopy moves the accumulator from TMEM to registers.

This example handles single-SM UMMA with a trivial `<1,1,1>` cluster shape. Cluster-level cooperation is an essential part of Blackwell kernels and is covered in Part 2 (TMA multicast and 2-SM UMMA).
