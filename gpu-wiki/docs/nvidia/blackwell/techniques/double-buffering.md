# Double/Multi-Buffering Patterns


**Last updated**: 2026-06-30

## Overview

Double-buffering (and multi-buffering) allocates two or more copies of a data buffer so that one copy can be written while another is read. On Blackwell, this pattern applies at two distinct levels: (1) TMEM double-buffering for overlapping MMA accumulation with epilogue readout, and (2) SMEM multi-stage buffering for overlapping TMA loads with MMA consumption. Both levels operate simultaneously in a well-optimized kernel.

## TMEM Double-Buffering

Tensor Memory (TMEM) on Blackwell has 128 rows x 512 columns of 32-bit elements (256 KB per SM). For a standard GEMM with TILE_M=128, TILE_N=256, the accumulator requires 128 x 256 = 32,768 elements, which fits in half the TMEM column space (256 out of 512 columns). The other 256 columns serve as the second buffer.

```cuda
// TMEM double-buffering: ping-pong between two 128x256 accumulator regions
//
// TMEM physical layout (per SM):
// +------ 256 cols ------+------ 256 cols ------+
// |                      |                      |
// |   Buffer A (active)  |   Buffer B (drain)   |  128 rows
// |   MMA writes here    |   Epilogue reads     |
// |                      |                      |
// +----------------------+----------------------+
//
// After tile completes, roles swap:
// Buffer A becomes "drain", Buffer B becomes "active"

#define TMEM_BUF_A_OFFSET  0
#define TMEM_BUF_B_OFFSET  256
#define TMEM_COLS_PER_BUF  256

__device__ void tmem_double_buffer_mainloop(
    const GemmParams& params,
    int num_output_tiles)
{
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // Synchronization between MMA warp and epilogue warps
    __shared__ uint64_t mbar_acc_ready[2];   // MMA done, epilogue can read
    __shared__ uint64_t mbar_acc_drained[2]; // Epilogue done, MMA can reuse

    if (threadIdx.x == 0) {
        mbarrier_init(&mbar_acc_ready[0], 1);
        mbarrier_init(&mbar_acc_ready[1], 1);
        mbarrier_init(&mbar_acc_drained[0], 1);
        mbarrier_init(&mbar_acc_drained[1], 1);
    }
    __syncthreads();

    for (int tile = 0; tile < num_output_tiles; tile++) {
        int buf = tile % 2;
        int tmem_offset = buf ? TMEM_BUF_B_OFFSET : TMEM_BUF_A_OFFSET;

        if (warp_id == 1) {
            // === MMA WARP ===
            // Wait for epilogue to finish draining this buffer
            if (tile >= 2) {
                mbarrier_wait(&mbar_acc_drained[buf]);
            }

            // Clear accumulator region before new tile
            if (lane_id == 0) {
                tmem_clear(tmem_offset, TMEM_COLS_PER_BUF);
            }
            __syncwarp();

            // Accumulate K-tiles into this TMEM buffer
            for (int k = 0; k < num_k_tiles; k++) {
                // (TMA load sync omitted -- see pipeline-stages)
                if (lane_id == 0) {
                    tcgen05_mma_accumulate(tmem_offset);
                }
                __syncwarp();
            }

            // Signal epilogue that accumulator is ready
            if (lane_id == 0) {
                mbarrier_arrive(&mbar_acc_ready[buf]);
            }

        } else if (warp_id >= 2) {
            // === EPILOGUE WARPS ===
            // Wait for MMA to finish accumulating
            mbarrier_wait(&mbar_acc_ready[buf]);

            // Read TMEM and store to global memory
            int rows_per_warp = TILE_M / 14;
            int my_row = (warp_id - 2) * rows_per_warp;
            for (int r = my_row; r < my_row + rows_per_warp; r++) {
                for (int c = lane_id; c < TMEM_COLS_PER_BUF; c += 32) {
                    float val = tmem_load_f32(r, tmem_offset + c);
                    // Apply epilogue and store...
                    params.C_ptr[r * params.N + c] = __float2half(val);
                }
            }

            // All epilogue warps must finish reading TMEM before the MMA
            // warp can overwrite this half-buffer. Each epilogue warp arrives
            // on a shared mbarrier; the mbarrier is initialized with
            // arrival_count = NUM_EPILOGUE_WARPS (e.g. 14 for warps 2-15).
            // The MMA warp waits on this mbarrier before reusing the buffer.
            if (lane_id == 0) {
                mbarrier_arrive(&mbar_acc_drained[buf]);
            }
            // mbar_acc_drained[buf] fires only after ALL epilogue warps arrive
        }
    }
}
```

## SMEM Multi-Stage Buffering

Shared memory multi-buffering provides multiple copies of the A and B input tiles so that TMA loads can overlap with MMA consumption:

```cuda
// SMEM multi-stage buffer allocation
// 3 stages, each holding one A tile and one B tile
//
// Memory layout:
// +--------+--------+--------+
// | Stage0 | Stage1 | Stage2 |
// | A0 B0  | A1 B1  | A2 B2  |
// +--------+--------+--------+
//
// Pipeline state at steady state:
// Stage 0: TMA loading (k+2)
// Stage 1: Ready, waiting for MMA
// Stage 2: MMA consuming (k)

constexpr int SMEM_STAGES = 3;
constexpr int TILE_A_BYTES = TILE_M * TILE_K * sizeof(half);  // e.g. 16 KB
constexpr int TILE_B_BYTES = TILE_K * TILE_N * sizeof(half);  // e.g. 32 KB
constexpr int STAGE_BYTES  = TILE_A_BYTES + TILE_B_BYTES;     // e.g. 48 KB
// Total SMEM for operands: 3 * 48 KB = 144 KB

struct SmemBuffers {
    half A[SMEM_STAGES][TILE_M][TILE_K];
    half B[SMEM_STAGES][TILE_K][TILE_N];
    uint64_t mbar_full[SMEM_STAGES];   // TMA done -> MMA can consume
    uint64_t mbar_empty[SMEM_STAGES];  // MMA done -> TMA can refill
};
```

## Combined TMEM + SMEM Double-Buffering

A fully optimized Blackwell GEMM uses both levels simultaneously:

```cuda
// Combined double-buffering: SMEM (3-stage) + TMEM (2-buffer)
//
// Outer loop: output tiles (TMEM double-buffered)
//   Inner loop: K-tiles (SMEM 3-stage pipelined)
//
// Timeline for 2 output tiles, 6 K-tiles each:
//
// TMEM buf:     |---- A (tile 0) ----|---- B (tile 1) ----|
// MMA warp:     | k0  k1  k2  k3  k4  k5 | k0  k1  k2 ...
// TMA warp:     |k2 k3 k4 k5  -  -  k2 k3 k4 k5 ...
// Epilogue:     | idle                | drain A  | drain B |
// SMEM stage:   |0  1  2  0  1  2    |0  1  2  0  1  2   |
//
// Key insight: the epilogue of tile 0 overlaps with the MMA of tile 1.
// This is only possible because they use different TMEM buffers.

__device__ void full_double_buffered_gemm(const GemmParams& params) {
    int warp_id = threadIdx.x / 32;

    for (int out_tile = 0; out_tile < num_output_tiles; out_tile++) {
        int tmem_buf = out_tile % 2;

        if (warp_id == 0) {
            // TMA producer: fill SMEM stages for this output tile's K-loop
            tma_producer_loop(params, out_tile);
        } else if (warp_id == 1) {
            // MMA: consume SMEM stages, accumulate into TMEM[tmem_buf]
            mma_consumer_loop(params, tmem_buf);
        } else {
            // Epilogue: drain TMEM[1-tmem_buf] from previous tile
            if (out_tile > 0) {
                epilogue_drain(params, out_tile - 1, 1 - tmem_buf);
            }
        }

        // Lightweight sync point between tiles
        // (mbarrier-based, not __syncthreads)
    }

    // Final epilogue for last tile
    if (warp_id >= 2) {
        epilogue_drain(params, num_output_tiles - 1,
                       (num_output_tiles - 1) % 2);
    }
}
```

## Comparison with Hopper

On Hopper (SM90), there is no TMEM. The accumulator lives in registers, and double-buffering the accumulator requires explicit register management:

```cuda
// Hopper: accumulator double-buffering uses register arrays
// Each warpgroup maintains two register-based accumulators
// This doubles register pressure and reduces occupancy

// Hopper approach (register pressure is the primary constraint):
float acc_buf0[REG_TILE_M][REG_TILE_N];  // First accumulator
float acc_buf1[REG_TILE_M][REG_TILE_N];  // Second accumulator
// Total: 2 * REG_TILE_M * REG_TILE_N * 4 bytes per thread
// For a 64x256 tile with 128-thread warpgroup:
//   each thread holds 2 * (64/4) * (256/128) * 4 = 2 * 16 * 2 * 4 = 256 bytes
//   = 64 registers just for accumulators

// Blackwell approach: TMEM holds both buffers with zero register cost
// MMA warp uses ~0 registers for accumulators
// All 256 KB of TMEM is dedicated accumulator space
```

| Aspect | Hopper (Registers) | Blackwell (TMEM) |
|--------|-------------------|------------------|
| Accumulator storage | Thread-local registers | CTA-wide TMEM |
| Double-buffer cost | 2x register usage | Zero register cost |
| Typical occupancy impact | Reduces from 2 to 1 CTA/SM | No impact |
| Epilogue access | Direct (already in registers) | TMEM load required |
| Max accumulator size | ~16K elements (register limited) | 128x512 = 65K elements |

## When to Use

- **TMEM double-buffering**: Always use on Blackwell when the epilogue takes more than trivial time. The only cost is the TMEM space, which is plentiful.
- **SMEM multi-stage buffering**: Always use for GEMM/attention mainloops. 3 stages is the default; increase to 4-5 only if the K-loop is long and memory latency is high.
- **Combined**: The standard approach for production Blackwell GEMM kernels. Both CUTLASS and CuTe-DSL kernels use this pattern.

## Caveats

- TMEM double-buffering requires that the accumulator fits in half the TMEM columns (256 out of 512). For very wide tiles (TILE_N > 256 with FP32 accumulators), the tile must be split or a different buffering strategy used.
- SMEM multi-stage buffering is constrained by the 228 KB SMEM limit. With 3 stages of large tiles, there may not be enough SMEM left for epilogue scratch space.
- The mbarrier synchronization between TMEM buffers adds a few cycles of overhead per tile. For kernels with very few K-iterations per tile, this overhead is proportionally larger.


## Related

- [Cache Policy Differentiation](cache-policy.md)
- [Chunk-Based Parallelism](chunk-parallelism.md)
- [CUDA GEMM Optimization Ladder](cuda-gemm-optimization-ladder.md)
- [Epilogue Fusion](epilogue-fusion.md)
- [Fine-Grained FP8/FP4 Quantization](fine-grained-quantization.md)
- [Composable Kernel (CK) Architecture Overview](../../../amd/common/ck-architecture-overview.md)
