# tcgen05.mma -- Blackwell MMA Instruction


**Last updated**: 2026-06-30

## Overview

`tcgen05.mma` is the Blackwell (SM100/SM100a) matrix-multiply-accumulate instruction that replaces Hopper's `wgmma.mma_async`. The name stands for **Tensor Core Generation 05**. NVIDIA also refers to the higher-level abstraction as **UMMA** (Unified Matrix Multiply-Accumulate) in the CUTLASS framework.

Key differences from `wgmma`:

| Property | wgmma (SM90) | tcgen05.mma (SM100) |
|---|---|---|
| Issuing scope | Warpgroup (4 warps, 128 threads) | Single thread |
| Accumulator storage | Registers (high pressure) | Tensor Memory (TMEM, 256KB/SM) |
| Operand A source | Registers or SMEM | Shared memory only |
| Operand B source | Shared memory | Shared memory only |
| Matrix load | ldmatrix to registers | Direct from SMEM (no ldmatrix) |
| Synchronization | Warpgroup-scoped barriers | Fully async, fence-based |

## Instruction Variants

tcgen05.mma has 7 variants organized by precision and scaling mode:

| Variant | A Type | B Type | Accumulator | Scale | MMA Shape (1SM) | Notes |
|---|---|---|---|---|---|---|
| `tcgen05.mma.kind::f16` | FP16/BF16 | FP16/BF16 | FP32 | None | m128n256k16 | Standard half-precision |
| `tcgen05.mma.kind::tf32` | TF32 | TF32 | FP32 | None | m128n256k8 | Single-precision approximation |
| `tcgen05.mma.kind::f8f6f4` | FP8/FP6/FP4 | FP8/FP6/FP4 | FP32 | Block (UE8M0) | m128n256k32 | Narrow precision with native block scaling |
| `tcgen05.mma.kind::i8` | INT8 | INT8 | INT32 | None | m128n256k32 | Integer quantized inference |
| `tcgen05.mma.kind::mxf8` | MXFP8 | MXFP8 | FP32 | MX (E8M0) | m128n256k32 | Microscaling FP8 |
| `tcgen05.mma.kind::mxf4` | MXFP4 | MXFP4 | FP32 | MX (E8M0) | m128n256k64 | Microscaling FP4 |
| `tcgen05.mma.kind::mxf4nvf4` | NVFP4 | MXFP4 | FP32 | Mixed | m128n256k64 | Mixed NVFP4/MXFP4 |

## MMA Shapes: 1-SM vs 2-SM

### 1-SM (Single CTA) Shapes

In single-SM mode, a single CTA owns the full MMA operation:

- **BF16/FP16**: M=128, N=256, K=16
- **TF32**: M=128, N=256, K=8
- **FP8/FP6/FP4**: M=128, N=256, K=32
- **MXFP4/NVFP4**: M=128, N=256, K=64

### 2-SM (Cooperative) Shapes

In two-SM cooperative mode, two CTAs share a single MMA across paired SMs:

- **BF16/FP16**: M=256, N=256, K=16
- **TF32**: M=256, N=256, K=8
- **FP8/FP6/FP4**: M=256, N=256, K=32
- **MXFP4/NVFP4**: M=256, N=256, K=64

The M dimension doubles because each SM contributes 128 rows from its own TMEM partition.

## Single-Thread Issuance Model

Unlike `wgmma` which required coordinated issuance from a warpgroup (4 warps, 128 threads), `tcgen05.mma` is issued by a **single thread**. This is a fundamental architectural simplification:

1. **No warpgroup synchronization overhead** -- one elected thread (typically lane 0 of warp 0) issues the MMA.
2. **Fully asynchronous** -- the instruction returns immediately; the hardware pipeline executes the MMA in the background.
3. **Fence-based completion** -- the producer must insert explicit fences before reading results from TMEM.

```cuda
// Single-thread MMA issuance pattern
__device__ void issue_mma(uint32_t tmem_addr, uint64_t smem_desc_a, uint64_t smem_desc_b) {
    // Only one thread issues the MMA
    if (threadIdx.x == 0) {
        asm volatile(
            "tcgen05.mma.cta_group::1.kind::f16 "
            "[%0], %1, %2, %3, 0;"
            :
            : "r"(tmem_addr),
              "l"(smem_desc_a),
              "l"(smem_desc_b),
              "r"(0)  // scale descriptor (unused for f16)
        );
    }
}
```

## PTX Examples

### Basic BF16 MMA (1-SM)

```ptx
// Issue a 128x256x16 BF16 MMA
// Operand A: shared memory descriptor
// Operand B: shared memory descriptor
// Accumulator: TMEM address
tcgen05.mma.cta_group::1.kind::f16 [tmem_addr], desc_a, desc_b, idesc, 0;
```

### 2-SM Cooperative BF16 MMA

```ptx
// Issue a 256x256x16 BF16 MMA across two paired CTAs
// cta_group::2 indicates cooperative mode
tcgen05.mma.cta_group::2.kind::f16 [tmem_addr], desc_a, desc_b, idesc, 0;
```

### FP8 with Block Scaling

```ptx
// FP8 with native UE8M0 block scaling
// scale_desc encodes the per-block scale factors
tcgen05.mma.cta_group::1.kind::f8f6f4 [tmem_addr], desc_a, desc_b, idesc, scale_desc;
```

### CUDA Inline PTX for BF16 MMA with Accumulation

```cuda
__device__ void mma_bf16_128x256x16(
    uint32_t tmem_addr,
    uint64_t desc_a,
    uint64_t desc_b
) {
    if (threadIdx.x == 0) {
        // First MMA: zero-initialize accumulator
        asm volatile(
            "tcgen05.mma.cta_group::1.kind::f16 "
            "[%0], %1, %2, %3, 0;"
            :
            : "r"(tmem_addr), "l"(desc_a), "l"(desc_b), "r"(0)
        );
    }
}

__device__ void mma_bf16_accumulate(
    uint32_t tmem_addr,
    uint64_t desc_a,
    uint64_t desc_b
) {
    if (threadIdx.x == 0) {
        // Subsequent MMAs: accumulate into existing TMEM
        // The enable_accumulate flag controls whether to add to or overwrite TMEM
        asm volatile(
            "tcgen05.mma.cta_group::1.kind::f16 "
            "[%0], %1, %2, %3, 1;"  // last arg 1 = accumulate
            :
            : "r"(tmem_addr), "l"(desc_a), "l"(desc_b), "r"(0)
        );
    }
}
```

## Critical Fences

Fences are **mandatory** for correctness. The hardware does not implicitly synchronize between MMA and TMEM reads/writes.

### tcgen05.mma.fence

Insert before reading MMA results from TMEM:

```cuda
// Wait for all outstanding tcgen05.mma operations to complete
__device__ void fence_before_tmem_read() {
    asm volatile("tcgen05.mma.fence::before_thread_sync;");
    __syncthreads();  // Ensure all threads see the fence
}
```

### tcgen05.mma.fence variants

```ptx
// Fence before reading TMEM (most common)
tcgen05.mma.fence::before_thread_sync;

// Fence at CTA scope for cooperative operations
tcgen05.mma.fence::before_cluster_sync;
```

### Correct Mainloop Fence Pattern

```cuda
__device__ void gemm_mainloop(/* params */) {
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        // 1. Wait for operand data to arrive in SMEM
        wait_barrier(k_tile % NUM_STAGES);

        // 2. Issue MMA (single thread)
        if (threadIdx.x == 0) {
            asm volatile(
                "tcgen05.mma.cta_group::1.kind::f16 "
                "[%0], %1, %2, %3, 1;"
                :
                : "r"(tmem_addr),
                  "l"(make_smem_desc(smem_a, k_tile)),
                  "l"(make_smem_desc(smem_b, k_tile)),
                  "r"(0)
            );
        }

        // 3. Release SMEM buffer for next TMA load
        if (threadIdx.x == 0) {
            arrive_barrier((k_tile + 1) % NUM_STAGES);
        }
    }

    // 4. CRITICAL: Fence before reading accumulator from TMEM
    asm volatile("tcgen05.mma.fence::before_thread_sync;");
    __syncthreads();

    // 5. Now safe to read results from TMEM
    read_tmem_accumulator(tmem_addr, output);
}
```

## Comparison with wgmma

### Programming Model Shift

```cuda
// ---- HOPPER (SM90): wgmma ----
// Requires warpgroup-scoped execution
// All 128 threads in a warpgroup participate
__device__ void hopper_mma() {
    // Load A matrix into registers via ldmatrix
    uint32_t a_frag[4];
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                 "{%0,%1,%2,%3}, [%4];"
                 : "=r"(a_frag[0]), "=r"(a_frag[1]),
                   "=r"(a_frag[2]), "=r"(a_frag[3])
                 : "r"(smem_addr));

    // Issue wgmma -- warpgroup scope, register accumulator
    asm volatile("wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
                 "{%0, %1, ...}, "  // register accumulators (128+ registers!)
                 "{%N, %N+1, ...}, " // A operand in registers
                 "desc_b, ...;"
                 : "+f"(acc[0]), "+f"(acc[1]), ...
                 : ...);

    // Commit and wait
    asm volatile("wgmma.commit_group.sync.aligned;");
    asm volatile("wgmma.wait_group.sync.aligned 0;");
    // Accumulators now in registers -- high register pressure
}

// ---- BLACKWELL (SM100): tcgen05 ----
// Single-thread issuance, TMEM accumulator
__device__ void blackwell_mma() {
    // No ldmatrix needed -- reads directly from SMEM
    // No register allocation for accumulators

    // Single thread issues MMA
    if (threadIdx.x == 0) {
        asm volatile(
            "tcgen05.mma.cta_group::1.kind::f16 "
            "[%0], %1, %2, %3, 1;"
            :
            : "r"(tmem_addr),   // accumulator in TMEM (not registers!)
              "l"(desc_a),       // A from SMEM descriptor
              "l"(desc_b),       // B from SMEM descriptor
              "r"(0)
        );
    }

    // Fence + sync before reading results
    asm volatile("tcgen05.mma.fence::before_thread_sync;");
    __syncthreads();
    // Read from TMEM -- registers free for other work
}
```

### Performance Implications

- **Register pressure**: Hopper wgmma uses 128+ registers for accumulators in a large GEMM tile. Blackwell stores accumulators in TMEM, freeing those registers for data movement and epilogue logic.
- **Occupancy**: Lower register pressure enables higher CTA occupancy or larger tile sizes without spilling.
- **Warp specialization**: With tcgen05, a single MMA-producer warp can feed the tensor cores while other warps handle TMA loads, epilogue, or softmax -- a natural fit for FlashAttention-style kernels.

## SMEM Layout Requirements

tcgen05.mma requires **128-byte swizzled** shared memory layouts for both operands. Non-swizzled or 64-byte swizzled layouts will produce incorrect results silently.

```cuda
// Shared memory descriptor construction for tcgen05
// The descriptor encodes: base address, stride, swizzle mode, dimensions
__device__ uint64_t make_smem_desc(void* smem_ptr, int stride_bytes) {
    uint64_t desc = 0;
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));

    // Encode base address (bits 0-13)
    desc |= (uint64_t)(addr >> 4);
    // Encode leading dimension stride (bits 16-29)
    desc |= (uint64_t)((stride_bytes >> 4) & 0x3FFF) << 16;
    // Encode 128-byte swizzle mode (bits 62-63) -- MANDATORY for tcgen05
    desc |= (uint64_t)(3) << 62;  // 3 = 128-byte swizzle

    return desc;
}
```

## Performance Progression (from tcgen05-for-dummies)

| Optimization Stage | Throughput (TFLOPS) | % of cuBLAS |
|---|---|---|
| Naive tcgen05.mma | 255 | 17% |
| + 128B swizzled SMEM | 695 | 46% |
| + TMA pipelining | 940 | 62% |
| + Persistent kernel + CLC | 1476 | 98% |
| cuBLAS reference | 1507 | 100% |


## Related

- [Two-SM Cooperative MMA](2sm-cooperative.md)
- [Cluster Launch Control (CLC)](clc.md)
- [CUDA Memory Hierarchy for Kernel Optimization](cuda-memory-hierarchy.md)
- [GPU Architecture Resource Model](gpu-architecture-resource-model.md)
- [mbarrier](mbarrier.md)
- [CUTLASS/CuTe Core Concepts and Layout Algebra](../../common/cutedsl/cutlass-cute-fundamentals.md)
- [Composable Kernel (CK) Architecture Overview](../../../amd/common/ck-architecture-overview.md)
