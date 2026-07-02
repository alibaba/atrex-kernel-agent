# Warp Specialization on Blackwell


**Last updated**: 2026-06-30

## Overview

Warp specialization assigns distinct functional roles to warps within a CTA, allowing each warp to focus on a single pipeline stage (data loading, MMA computation, or epilogue writeback). On Blackwell (SM100), the 16-warp CTA structure replaces Hopper's 4-warp warpgroup model. Because tcgen05.mma is a single-thread instruction that operates on TMEM rather than registers, only one warp needs to issue MMA operations, freeing the remaining warps for producer and consumer roles.

## Blackwell 16-Warp Kernel Structure

The canonical Blackwell GEMM kernel uses 16 warps (512 threads) per CTA with the following role assignment:

| Warp ID | Role | Responsibility |
|---------|------|----------------|
| 0 | TMA Producer | Issues TMA bulk-copy from global to shared memory, signals mbarrier |
| 1 | MMA Consumer | Issues tcgen05.mma on SMEM operands, writes results to TMEM |
| 2-15 | Epilogue | Reads TMEM accumulator, applies scale/bias/activation, writes to global memory |

This contrasts with Hopper where a warpgroup (4 warps, 128 threads) collectively issues wgmma.mma_async, and all threads in the warpgroup participate in the MMA. On Blackwell, the MMA warp dispatches the instruction from a single thread while the hardware handles the data movement internally.

## Comparison with Hopper Warpgroup Model

| Aspect | Hopper (SM90) | Blackwell (SM100) |
|--------|---------------|-------------------|
| MMA granularity | 4-warp warpgroup (128 threads) | Single thread in 1 warp |
| MMA output destination | Registers (shared across warpgroup) | TMEM (256KB, CTA-visible) |
| Producer warps | Separate warp(s) for TMA loads | Warp 0 dedicated to TMA |
| Epilogue execution | Same warpgroup or separate warps | 14 dedicated warps (2-15) |
| Synchronization | warpgroup barriers, arrive/wait | mbarrier pairs (producer/consumer) |
| Register pressure | High (accumulators in registers) | Low (accumulators in TMEM) |

## Warp Role Assignment

The kernel entry point assigns each warp its role based on `threadIdx.x`:

```cuda
// Blackwell 16-warp specialized GEMM kernel skeleton
// 16 warps = 512 threads per CTA
__global__ void __launch_bounds__(512)
blackwell_gemm_warp_specialized(
    const __grid_constant__ GemmParams params)
{
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    // Shared memory for A/B tiles and mbarrier objects
    extern __shared__ char smem[];
    half* smem_A = reinterpret_cast<half*>(smem);
    half* smem_B = reinterpret_cast<half*>(smem + SMEM_A_SIZE);

    // mbarrier pairs: TMA hardware signals "data ready", MMA signals "buffer free"
    __shared__ uint64_t mbar_data_ready[NUM_STAGES];
    __shared__ uint64_t mbar_buffer_free[NUM_STAGES];
    // MMA→epilogue handoff barrier
    __shared__ uint64_t mbar_acc_complete;
    // Phase tracking: mbarriers alternate parity on each reuse cycle
    int phase_data[NUM_STAGES];
    int phase_free[NUM_STAGES];

    if (warp_id == 0) {
        if (lane_id == 0) {
            for (int s = 0; s < NUM_STAGES; s++) {
                // TMA expects arrive.expect_tx → hardware completes
                mbarrier_init(&mbar_data_ready[s], 1);
                mbarrier_init(&mbar_buffer_free[s], 1);
            }
            mbarrier_init(&mbar_acc_complete, 1);
        }
    }
    // Initialize phase counters (all start at 0)
    for (int s = 0; s < NUM_STAGES; s++) {
        phase_data[s] = 0;
        phase_free[s] = 0;
    }
    __syncthreads();

    if (warp_id == 0) {
        // === TMA PRODUCER WARP ===
        for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
            int stage = k_tile % NUM_STAGES;

            // Wait for consumer to release this buffer (with phase tracking)
            if (k_tile >= NUM_STAGES) {
                mbarrier_wait_parity(&mbar_buffer_free[stage], phase_free[stage]);
                phase_free[stage] ^= 1;  // flip parity for next reuse
            }

            // Set expected TX bytes, then issue TMA. TMA hardware will
            // signal mbar_data_ready upon transfer completion.
            // Do NOT manually arrive -- that races with the async transfer.
            if (lane_id == 0) {
                uint32_t tx_bytes = TILE_A_BYTES + TILE_B_BYTES;
                mbarrier_arrive_expect_tx(&mbar_data_ready[stage], tx_bytes);
                tma_copy_async(smem_A + stage * TILE_A_SIZE,
                               &params.A[k_tile * TILE_K], TILE_A_SIZE,
                               &mbar_data_ready[stage]);
                tma_copy_async(smem_B + stage * TILE_B_SIZE,
                               &params.B[k_tile * TILE_K], TILE_B_SIZE,
                               &mbar_data_ready[stage]);
                // TMA hardware arrives on mbar_data_ready when transfer completes
            }
        }

    } else if (warp_id == 1) {
        // === MMA CONSUMER WARP ===
        for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
            int stage = k_tile % NUM_STAGES;

            // Wait for TMA to complete this stage (with phase tracking)
            mbarrier_wait_parity(&mbar_data_ready[stage], phase_data[stage]);
            phase_data[stage] ^= 1;

            // Critical fence: ensure TMA data visible before MMA reads SMEM
            tcgen05_fence_after_thread_sync();

            if (lane_id == 0) {
                tcgen05_mma(smem_A + stage * TILE_A_SIZE,
                            smem_B + stage * TILE_B_SIZE);
            }
            __syncwarp();

            // Signal buffer is free for reuse
            if (lane_id == 0) {
                mbarrier_arrive(&mbar_buffer_free[stage]);
            }
        }

        // Signal epilogue warps that accumulation is complete
        if (lane_id == 0) {
            mbarrier_arrive(&mbar_acc_complete);
        }

    } else {
        // === EPILOGUE WARPS (2-15) ===
        // Wait for MMA completion via dedicated mbarrier (not __syncthreads,
        // which would deadlock since producer/MMA warps don't reach it)
        mbarrier_wait(&mbar_acc_complete);

        // Each epilogue warp handles a partition of the TMEM output.
        // Use ceiling division to cover tail rows when TILE_M % 14 != 0.
        constexpr int NUM_EPI_WARPS = 14;  // warps 2-15
        int epi_warp = warp_id - 2;  // 0..13
        int rows_per_warp = (TILE_M + NUM_EPI_WARPS - 1) / NUM_EPI_WARPS;
        int my_row_start = epi_warp * rows_per_warp;
        int my_row_end = min(my_row_start + rows_per_warp, TILE_M);

        for (int r = my_row_start; r < my_row_end; r++) {
            for (int c = lane_id; c < TILE_N; c += 32) {
                // Read accumulator from TMEM
                float acc = tmem_load(r, c);
                // Apply epilogue: scale + bias + activation
                float result = epilogue_op(acc, params.scale, params.bias[c]);
                // Write to global memory
                params.C[r * params.N + c] = __float2half(result);
            }
        }
    }
}
```

## mbarrier Synchronization Pattern

The producer-consumer synchronization uses mbarrier pairs. Each pipeline stage has two barriers:

1. **data_ready**: Producer (Warp 0) arrives after TMA completes. Consumer (Warp 1) waits before issuing MMA.
2. **buffer_free**: Consumer (Warp 1) arrives after MMA consumes the data. Producer (Warp 0) waits before overwriting the buffer.

At the PTX level, the mbarrier operations map to:

```ptx
// Producer: signal data is ready in stage %stage
mbarrier.arrive.shared.b64  %dummy, [%mbar_data_ready + %stage_offset];

// Consumer: wait for data to be ready
mbarrier.try_wait.parity.shared.b64  %pred, [%mbar_data_ready + %stage_offset], %phase;

// Consumer: signal buffer is consumed
mbarrier.arrive.shared.b64  %dummy, [%mbar_buffer_free + %stage_offset];

// Producer: wait for buffer to be free
mbarrier.try_wait.parity.shared.b64  %pred, [%mbar_buffer_free + %stage_offset], %phase;
```

## CUTLASS SM100 Warp Specialization

In CUTLASS 4.5.0, the SM100 GEMM collective (`CollectiveMma_1SM`) implements this pattern with CuTe abstractions:

```cuda
// CUTLASS SM100 warp role dispatch (simplified from CollectiveMma)
// Template parameter WarpCount = cute::Shape<1, 1, 14>
// Warp 0 = producer, Warp 1 = math, Warps 2-15 = epilogue

template <class TiledMma, class SmemLayout>
struct CollectiveMma_1SM {
    static constexpr int NumProducerWarps = 1;
    static constexpr int NumMathWarps = 1;
    static constexpr int NumEpilogueWarps = 14;

    CUTLASS_DEVICE void operator()(
        Params const& params,
        char* smem_buf,
        TiledMma& tiled_mma)
    {
        int warp_idx = cutlass::canonical_warp_idx_sync();

        if (warp_idx == 0) {
            producer_warp(params, smem_buf);
        } else if (warp_idx == 1) {
            math_warp(params, smem_buf, tiled_mma);
        } else {
            epilogue_warp(params, smem_buf);
        }
    }
};
```

## When to Use

- **Always on Blackwell GEMMs**: Warp specialization is the standard pattern for SM100 tensor core kernels. The tcgen05 instruction model assumes single-thread dispatch with TMEM output.
- **Attention kernels**: FlashAttention-4 extends this to ping-pong scheduling with 2 query tile groups and dedicated softmax warps.
- **Any kernel with producer-consumer pipeline**: When TMA loads and MMA compute can overlap, warp specialization provides the cleanest decomposition.

## Caveats

- The 14 epilogue warps may be underutilized for simple epilogues (e.g., pure store). Complex epilogues (scale, bias, activation, quantization) benefit more.
- The single MMA warp means the kernel cannot overlap multiple independent MMA streams within a CTA. Use 2-SM cooperative mode for larger tiles instead.
- mbarrier initialization must happen before any warp tries to wait; use `__syncthreads()` after init if needed.


## Related

- [Cache Policy Differentiation](cache-policy.md)
- [Chunk-Based Parallelism](chunk-parallelism.md)
- [CUDA GEMM Optimization Ladder](cuda-gemm-optimization-ladder.md)
- [Double/Multi-Buffering Patterns](double-buffering.md)
- [Epilogue Fusion](epilogue-fusion.md)
- [Warp Specialization Design Principles](../../common/warp-specialization-design-principles.md)
- [Software Pipeline Depth Optimization](../../common/software-pipeline-depth-optimization.md)
- [Composable Kernel (CK) Architecture Overview](../../../amd/common/ck-architecture-overview.md)
