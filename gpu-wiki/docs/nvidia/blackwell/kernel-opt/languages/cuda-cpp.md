# CUDA C++ for Blackwell Kernels

## Overview

Plain CUDA C++ with inline PTX is used for hand-optimized Blackwell kernels. The tcgen05 tutorial achieved 98% of cuBLAS performance using this approach.

## tcgen05 via Inline PTX

```cuda
// Allocate TMEM
__device__ void tmem_alloc(uint32_t* addr, uint32_t num_cols) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned %0, %1;"
        : "=r"(*addr) : "r"(num_cols)
    );
}

// Issue MMA (single thread, typically warp 1 lane 0)
// idesc_c/idesc_d: immediate descriptors for accumulator C and output D
__device__ void tcgen05_mma(uint32_t tmem_addr,
                             uint64_t desc_a, uint64_t desc_b,
                             uint32_t idesc_c, uint32_t idesc_d) {
    asm volatile(
        "tcgen05.mma.cta_group::1.kind::f16"
        " %0, %1, %2, %3, %4;"
        :: "r"(tmem_addr), "l"(desc_a), "l"(desc_b),
           "r"(idesc_c), "r"(idesc_d)
    );
}

// Load TMEM to registers
__device__ void tmem_load(float* dst, uint32_t tmem_addr, int cols) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x1 {%0}, [%1];"
        : "=r"(*dst) : "r"(tmem_addr)
    );
}

// Deallocate TMEM (MUST do before kernel exit)
__device__ void tmem_dealloc(uint32_t addr, uint32_t num_cols) {
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned %0, %1;"
        :: "r"(addr), "r"(num_cols)
    );
}
```

## mbarrier Synchronization

```cuda
// TMA-MMA synchronization via mbarrier
// expected_bytes: total bytes the TMA will deliver to this stage
__device__ void mbarrier_arrive(uint64_t* mbar, uint32_t expected_bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(mbar)),
           "r"(expected_bytes)
    );
}

__device__ void mbarrier_wait(uint64_t* mbar, int phase) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "WAIT_LOOP:\n"
        "  mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
        "  @!p bra WAIT_LOOP;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(mbar)),
           "r"(phase)
    );
}
```

## Warp Role Dispatch

```cuda
__global__ void blackwell_gemm_kernel(...) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (warp_id == 0 && lane_id == 0) {
        // TMA producer: issue cp.async.bulk.tensor
        tma_producer_loop(...);
    } else if (warp_id == 1 && lane_id == 0) {
        // MMA consumer: issue tcgen05.mma
        mma_consumer_loop(...);
    } else if (warp_id >= 2) {
        // Epilogue: read TMEM, write to global
        epilogue_loop(...);
    }
}
```

## Related
- [ptx-sm100](ptx-sm100.md) -- PTX instruction reference
- `tcgen05-tutorial` -- Step-by-step tcgen05 guide
