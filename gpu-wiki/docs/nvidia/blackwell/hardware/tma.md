# Tensor Memory Accelerator (TMA)


**Last updated**: 2026-06-30

## Overview

The Tensor Memory Accelerator (TMA) is a hardware unit that performs **asynchronous bulk data transfers** between global memory and shared memory. First introduced on Hopper (SM90), TMA carries forward to Blackwell (SM100) with stricter requirements for tcgen05 compatibility.

TMA offloads data movement from CUDA cores entirely -- a single thread issues the transfer, and the TMA hardware engine handles the multi-dimensional copy, address calculation, out-of-bounds clamping, and format conversion.

## Key Properties

| Property | Detail |
|---|---|
| Transfer direction | GMEM <-> SMEM (bidirectional) |
| Dimensionality | 1D to 5D tensor copies |
| Max transfer size | Up to 256 bytes per element, tiles up to 128x256 |
| Swizzle modes | None, 32B, 64B, 128B (128B required for tcgen05) |
| Format conversion | FP32<->BF16, FP32<->FP16 during transfer |
| Multicast | Single GMEM tile -> multiple CTAs in a cluster |
| Synchronization | mbarrier-based (arrive/wait) |
| Thread requirement | Single thread issues the operation |

## TMA Descriptor

TMA operations are driven by a **descriptor** that encodes the tensor layout, addressing, and transfer parameters. The descriptor is created on the host and passed to the kernel.

### Host-Side Descriptor Creation

```cuda
#include <cuda.h>

// Create a 2D TMA descriptor for a row-major FP16 matrix
CUtensorMap create_tma_descriptor_2d(
    const half* global_ptr,
    int M, int N,           // Global tensor dimensions
    int tile_m, int tile_n, // Tile dimensions for each transfer
    int swizzle_bytes       // Swizzle mode: 0, 32, 64, 128
) {
    CUtensorMap tensor_map;

    // Tensor dimensions (outermost to innermost)
    uint64_t global_dims[2] = {(uint64_t)N, (uint64_t)M};
    uint64_t global_strides[1] = {(uint64_t)(N * sizeof(half))};

    // Tile box dimensions
    uint32_t box_dims[2] = {(uint32_t)tile_n, (uint32_t)tile_m};

    // Element strides (1 = contiguous)
    uint32_t elem_strides[2] = {1, 1};

    CUtensorMapSwizzle swizzle;
    switch (swizzle_bytes) {
        case 0:   swizzle = CU_TENSOR_MAP_SWIZZLE_NONE; break;
        case 32:  swizzle = CU_TENSOR_MAP_SWIZZLE_32B; break;
        case 64:  swizzle = CU_TENSOR_MAP_SWIZZLE_64B; break;
        case 128: swizzle = CU_TENSOR_MAP_SWIZZLE_128B; break;
    }

    cuTensorMapEncodeTiled(
        &tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,  // Element type
        2,                                  // Dimensionality
        (void*)global_ptr,                 // Global pointer
        global_dims,                       // Tensor dimensions
        global_strides,                    // Byte strides (exclude innermost)
        box_dims,                          // Tile/box dimensions
        elem_strides,                      // Element strides
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    return tensor_map;
}
```

### Blackwell Requirement: 128-Byte Swizzle

On Blackwell, `tcgen05.mma` requires operands in **128-byte swizzled** SMEM layout. If TMA loads data without 128B swizzling, the MMA will produce incorrect results.

```cuda
// CORRECT for Blackwell tcgen05:
CUtensorMap desc = create_tma_descriptor_2d(ptr, M, N, 128, 64, 128);
//                                                     swizzle=128 ^^^

// WRONG for tcgen05 (will silently produce garbage):
CUtensorMap desc = create_tma_descriptor_2d(ptr, M, N, 128, 64, 0);
//                                                     swizzle=0 ^^^
```

## Asynchronous Copy Operations

### GMEM to SMEM (Load)

```cuda
// TMA load: single thread issues, hardware executes asynchronously
__device__ void tma_load_tile(
    const CUtensorMap* desc,
    void* smem_ptr,
    uint64_t* mbar_ptr,  // mbarrier for synchronization
    int coord_x, int coord_y
) {
    if (threadIdx.x == 0) {
        // Set expected bytes on the mbarrier
        uint32_t expected_bytes = TILE_M * TILE_N * sizeof(half);
        asm volatile(
            "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
            :
            : "r"((uint32_t)__cvta_generic_to_shared(mbar_ptr)),
              "r"(expected_bytes)
        );

        // Issue TMA copy
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
            "[%0], [%1, {%2, %3}], [%4];"
            :
            : "r"((uint32_t)__cvta_generic_to_shared(smem_ptr)),
              "l"(desc),
              "r"(coord_x), "r"(coord_y),
              "r"((uint32_t)__cvta_generic_to_shared(mbar_ptr))
        );
    }
}
```

### SMEM to GMEM (Store)

```cuda
// TMA store: write a tile from shared memory back to global memory
__device__ void tma_store_tile(
    const CUtensorMap* desc,
    const void* smem_ptr,
    int coord_x, int coord_y
) {
    if (threadIdx.x == 0) {
        asm volatile(
            "cp.async.bulk.tensor.2d.global.shared::cta "
            "[%0, {%1, %2}], [%3];"
            :
            : "l"(desc),
              "r"(coord_x), "r"(coord_y),
              "r"((uint32_t)__cvta_generic_to_shared(smem_ptr))
        );

        // Commit the store
        asm volatile("cp.async.bulk.commit_group;");
    }
}
```

## mbarrier Synchronization

TMA uses mbarriers (memory barriers) for producer-consumer synchronization. The pattern is:

1. **Producer** (TMA): arrives at the barrier when the transfer completes, decrementing the expected transaction count.
2. **Consumer** (compute warps): waits on the barrier before reading the loaded data.

### Pipeline Stage Pattern

```cuda
// Multi-stage pipeline with TMA + mbarrier
__device__ void pipelined_mainloop(
    const CUtensorMap* desc_a,
    const CUtensorMap* desc_b,
    void* smem_a_stages[NUM_STAGES],
    void* smem_b_stages[NUM_STAGES],
    uint64_t* mbar[NUM_STAGES],
    int num_k_tiles
) {
    // Prologue: fill the first NUM_STAGES-1 stages
    for (int s = 0; s < NUM_STAGES - 1 && s < num_k_tiles; ++s) {
        tma_load_tile(desc_a, smem_a_stages[s], mbar[s], 0, s);
        tma_load_tile(desc_b, smem_b_stages[s], mbar[s], s, 0);
    }

    // Mainloop
    for (int k = 0; k < num_k_tiles; ++k) {
        int stage = k % NUM_STAGES;

        // Wait for data to arrive in this stage
        mbarrier_wait(mbar[stage]);

        // Issue MMA using this stage's SMEM buffers
        if (threadIdx.x == 0) {
            asm volatile(
                "tcgen05.mma.cta_group::1.kind::f16 "
                "[%0], %1, %2, %3, 1;"
                :
                : "r"(tmem_acc),
                  "l"(make_desc(smem_a_stages[stage])),
                  "l"(make_desc(smem_b_stages[stage])),
                  "r"(0)
            );
        }

        // Prefetch next stage
        int next_k = k + NUM_STAGES - 1;
        if (next_k < num_k_tiles) {
            int next_stage = next_k % NUM_STAGES;
            tma_load_tile(desc_a, smem_a_stages[next_stage],
                         mbar[next_stage], 0, next_k);
            tma_load_tile(desc_b, smem_b_stages[next_stage],
                         mbar[next_stage], next_k, 0);
        }
    }
}
```

### mbarrier Operations

```cuda
// Initialize an mbarrier
__device__ void mbarrier_init(uint64_t* mbar, int arrive_count) {
    if (threadIdx.x == 0) {
        asm volatile(
            "mbarrier.init.shared.b64 [%0], %1;"
            :
            : "r"((uint32_t)__cvta_generic_to_shared(mbar)),
              "r"(arrive_count)
        );
    }
}

// Wait for an mbarrier to complete (phase-based)
__device__ void mbarrier_wait(uint64_t* mbar, int phase) {
    uint32_t mbar_addr = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "WAIT_LOOP:\n"
        "mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
        "@!p bra WAIT_LOOP;\n"
        "}\n"
        :
        : "r"(mbar_addr), "r"(phase)
    );
}
```

## TMA Multicast

TMA multicast sends a single GMEM tile to **multiple CTAs within a cluster** simultaneously. This is critical for GEMM where the B operand is shared across M-axis tiles.

```cuda
// Multicast TMA: load B tile to all CTAs in the cluster
__device__ void tma_multicast_load(
    const CUtensorMap* desc,
    void* smem_ptr,
    uint64_t* mbar_ptr,
    int coord_x, int coord_y,
    uint16_t multicast_mask  // bitmask: which CTAs in cluster receive the data
) {
    if (threadIdx.x == 0) {
        uint32_t expected_bytes = TILE_K * TILE_N * sizeof(half);
        asm volatile(
            "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
            :
            : "r"((uint32_t)__cvta_generic_to_shared(mbar_ptr)),
              "r"(expected_bytes)
        );

        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster "
            "[%0], [%1, {%2, %3}], [%4], %5;"
            :
            : "r"((uint32_t)__cvta_generic_to_shared(smem_ptr)),
              "l"(desc),
              "r"(coord_x), "r"(coord_y),
              "r"((uint32_t)__cvta_generic_to_shared(mbar_ptr)),
              "h"(multicast_mask)
        );
    }
}
```

### Multicast in GEMM

```
Cluster: 2 CTAs (CTA0 and CTA1) each computing different M-tiles of the same N column

CTA0: computes C[0:128, 0:256]   -- needs A[0:128, :] and B[:, 0:256]
CTA1: computes C[128:256, 0:256] -- needs A[128:256, :] and B[:, 0:256]

B[:, 0:256] is SHARED -- multicast it once from GMEM to both CTAs
A tiles are UNIQUE -- each CTA loads its own A tile

Result: B bandwidth is halved (1 GMEM read serves 2 CTAs)
```

## Blackwell-Specific Enhancements

### 128-Byte Swizzle for tcgen05

All TMA loads feeding `tcgen05.mma` must use 128-byte swizzling. The swizzle pattern rearranges bytes within each 128-byte line to match the tensor core's internal data layout:

```
Without swizzle (linear):
  Row 0: bytes [0, 1, 2, ..., 127]
  Row 1: bytes [128, 129, ..., 255]

With 128B swizzle:
  Row 0: bytes [0, 1, ..., 127]     (unchanged)
  Row 1: bytes [128, 129, ..., 255] XOR pattern applied
  Row 2: bytes [256, ...] XOR pattern applied differently
  ...
```

The swizzle eliminates bank conflicts when the tensor core reads operand tiles from SMEM.

### TMA + TMEM Integration

On Blackwell, data flows through a characteristic pipeline:

```
GMEM --[TMA]--> SMEM --[tcgen05.mma]--> TMEM --[tcgen05.ld]--> Registers --[st.global]--> GMEM
                  ^                        |
                  |                        v
                  +---- (epilogue) --------+
                        (bias, activation, etc.)
```

## Performance Considerations

| Tip | Detail |
|---|---|
| Maximize TMA utilization | Keep the TMA unit busy with back-to-back loads across pipeline stages |
| Use multicast for shared operands | Reduces GMEM bandwidth by cluster_size x for shared tiles |
| Always use 128B swizzle on Blackwell | Non-128B swizzle produces incorrect tcgen05 results |
| Prefer 2D TMA over manual addressing | TMA handles out-of-bounds clamping, padding, and strided access |
| Pipeline depth | 3-5 stages typically optimal; more stages increase SMEM usage |

## CuTe-DSL Example

```python
# CuTe-DSL TMA copy setup for Blackwell GEMM
from cute import *

# Define TMA copy atom for operand A (BF16, 128x64 tile)
tma_a = make_tma_copy(
    SM100_TMA_LOAD_2D,
    tensor_a,                      # global tensor
    smem_layout_a,                 # shared memory layout
    tile_shape=(128, 64),          # tile dimensions
    swizzle=Swizzle(7, 0, 4),     # 128-byte swizzle
    multicast_mask=None            # no multicast for A
)

# Define TMA copy atom for operand B with multicast
tma_b = make_tma_copy(
    SM100_TMA_LOAD_2D_MULTICAST,
    tensor_b,
    smem_layout_b,
    tile_shape=(64, 256),
    swizzle=Swizzle(7, 0, 4),     # 128-byte swizzle
    multicast_mask=cluster_mask    # multicast B to all CTAs in cluster
)
```


## Related

- [Two-SM Cooperative MMA](2sm-cooperative.md)
- [Cluster Launch Control (CLC)](clc.md)
- [CUDA Memory Hierarchy for Kernel Optimization](cuda-memory-hierarchy.md)
- [GPU Architecture Resource Model](gpu-architecture-resource-model.md)
- [mbarrier](mbarrier.md)
- [Composable Kernel (CK) Architecture Overview](../../../amd/common/ck-architecture-overview.md)
