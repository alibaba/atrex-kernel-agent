# Epilogue Fusion

## Overview

Epilogue fusion overlaps the post-MMA operations (scaling, bias addition, activation functions, quantization, store to global memory) with ongoing MMA computation. On Blackwell, the accumulator lives in TMEM rather than registers, enabling dedicated epilogue warps (typically warps 2-15) to read TMEM concurrently while the MMA warp (warp 1) continues accumulating the next tile. This overlap is achieved by double-buffering the TMEM accumulator: the MMA warp writes to one half while the epilogue warps read from the other half.

## TMEM-to-Register Epilogue Path

On Blackwell, the MMA result resides in Tensor Memory (TMEM). Epilogue warps must copy relevant portions of TMEM into registers before applying element-wise operations and writing to global memory:

```cuda
// Epilogue warp: read TMEM accumulator, apply fused operations, store
// This runs on warps 2-15 while warp 1 continues MMA on next tile
__device__ void epilogue_warp_fn(
    int warp_id,
    int tile_m, int tile_n,
    float scale, const float* bias,
    half* C, int ldc)
{
    int lane_id = threadIdx.x % 32;

    // Each epilogue warp handles a stripe of the output tile
    // 14 warps, TILE_M = 128 -> ~9 rows per warp
    int rows_per_warp = (TILE_M + 13) / 14;
    int row_start = (warp_id - 2) * rows_per_warp;
    int row_end   = min(row_start + rows_per_warp, TILE_M);

    for (int r = row_start; r < row_end; r++) {
        int global_row = tile_m * TILE_M + r;

        for (int c = lane_id; c < TILE_N; c += 32) {
            // Step 1: Load accumulator from TMEM into register
            float acc = tmem_load_f32(r, c);

            // Step 2: Fused epilogue operations (all in registers)
            // Scale
            acc *= scale;
            // Bias
            acc += bias[tile_n * TILE_N + c];
            // ReLU activation
            acc = fmaxf(acc, 0.0f);

            // Step 3: Store to global memory
            int global_col = tile_n * TILE_N + c;
            C[global_row * ldc + global_col] = __float2half(acc);
        }
    }
}
```

## Overlapping MMA with Epilogue via Double-Buffering

The key to epilogue fusion on Blackwell is TMEM double-buffering. The 512-column TMEM space is split into two halves (columns 0-255 and 256-511). While the MMA warp accumulates into one half, the epilogue warps drain the other:

```cuda
// TMEM double-buffering for MMA-epilogue overlap
//
// TMEM layout: 128 rows x 512 columns (32-bit elements)
// Buffer A: columns [0, 255]    -- 128 x 256 accumulator
// Buffer B: columns [256, 511]  -- 128 x 256 accumulator
//
// Timeline:
// Tile 0: MMA -> buffer A     | epilogue idle (no prior result)
// Tile 1: MMA -> buffer B     | epilogue reads buffer A
// Tile 2: MMA -> buffer A     | epilogue reads buffer B
// Tile 3: MMA -> buffer B     | epilogue reads buffer A
//    ... ping-pong continues ...

__device__ void mma_epilogue_overlap(
    const GemmParams& params,
    int num_tiles)
{
    int warp_id = threadIdx.x / 32;

    __shared__ uint64_t mbar_mma_done[2];    // One per TMEM buffer half
    __shared__ uint64_t mbar_epi_done[2];    // Epilogue completion signals

    if (threadIdx.x == 0) {
        for (int i = 0; i < 2; i++) {
            mbarrier_init(&mbar_mma_done[i], 1);
            mbarrier_init(&mbar_epi_done[i], 1);
        }
    }
    __syncthreads();

    if (warp_id == 1) {
        // === MMA WARP ===
        for (int t = 0; t < num_tiles; t++) {
            int buf = t % 2;  // Alternate between buffer halves

            // Wait for epilogue to finish reading this buffer
            if (t >= 2) {
                mbarrier_wait(&mbar_epi_done[buf]);
            }

            // Issue MMA, accumulating into TMEM buffer half
            int tmem_col_offset = buf * 256;
            tcgen05_mma_with_offset(tmem_col_offset,
                                    params.smem_A, params.smem_B);

            // Signal epilogue that this buffer is ready
            mbarrier_arrive(&mbar_mma_done[buf]);
        }

    } else if (warp_id >= 2) {
        // === EPILOGUE WARPS ===
        for (int t = 0; t < num_tiles; t++) {
            int buf = t % 2;

            // Wait for MMA to finish filling this buffer
            mbarrier_wait(&mbar_mma_done[buf]);

            // Read from TMEM buffer half and write to global memory
            int tmem_col_offset = buf * 256;
            epilogue_store(params, t, tmem_col_offset, warp_id);

            // ALL epilogue warps must finish reading TMEM before MMA reuses
            // the buffer. Each epilogue warp arrives on mbar_epi_done;
            // mbar_epi_done is initialized with count = NUM_EPILOGUE_WARPS.
            // MMA warp waits on this mbarrier before writing to this half.
            if (lane_id == 0) {
                mbarrier_arrive(&mbar_epi_done[buf]);
            }
            // mbar_epi_done[buf] fires only after ALL epilogue warps arrive
        }
    }
}
```

## CUTLASS Epilogue Patterns

CUTLASS 4.5.0 provides composable epilogue visitors that fuse arbitrary element-wise operations after GEMM:

```cuda
// CUTLASS SM100 epilogue with fused scale + bias + activation
// Uses the EVT (Epilogue Visitor Tree) pattern

using EpilogueOp = cutlass::epilogue::fusion::LinCombEltAct<
    cutlass::epilogue::thread::ReLU,   // Activation function
    float,                              // Compute type
    float,                              // Scale type
    cutlass::half_t                     // Output type
>;

// The epilogue descriptor tells CUTLASS how to partition work
// across the 14 epilogue warps
using CollectiveEpilogue = cutlass::epilogue::collective::Sm100EpilogueTmaWarpSpecialized<
    cutlass::gemm::TagToStrideC_t<cutlass::layout::RowMajor>,
    cutlass::gemm::TagToStrideC_t<cutlass::layout::RowMajor>,
    EpilogueOp,
    cutlass::gemm::EpilogueDefault  // Default tiling
>;

// In the kernel, the epilogue is invoked after the mainloop:
// epilogue(
//     problem_shape,
//     collective_mainloop.get_accumulator(),  // TMEM reference
//     epilogue_params,                         // scale, bias pointers
//     shared_storage                           // SMEM for TMA stores
// );
```

## Common Fused Epilogue Operations

| Operation | Description | Typical Use |
|-----------|-------------|-------------|
| Scale + Bias | `y = alpha * acc + beta * C` | Standard GEMM epilogue |
| ReLU / GeLU / SiLU | Element-wise activation | MLP layers |
| Quantize | FP32 accumulator to FP8/FP16 | Inference quantization |
| SwiGLU gate | `y = SiLU(gate) * up` | Gated dual GEMM (LLM FFN) |
| Softmax rescale | `y = acc * exp(max_old - max_new)` | Attention epilogue |
| Residual add | `y = acc + residual` | Transformer blocks |

## When to Use

- **All Blackwell GEMMs with non-trivial epilogues**: The 14 epilogue warps are available by default in the warp-specialized model. Fusing operations avoids a separate kernel launch and an extra global memory round-trip.
- **Attention kernels**: The softmax rescaling and output accumulation can be overlapped with the next KV tile's MMA.
- **Quantized inference**: FP32-to-FP8 conversion in the epilogue avoids writing FP32 intermediates to global memory.

## Caveats

- The epilogue can only read TMEM after the MMA for that tile is complete. The double-buffer synchronization is mandatory to prevent reading partial results.
- TMEM-to-register bandwidth is not unlimited. With 14 warps simultaneously reading TMEM, each warp gets a proportional share. Very wide output tiles (large TILE_N) may bottleneck on TMEM read bandwidth.
- Simple epilogues (just store) waste the 14 epilogue warps. For such cases, consider reducing the CTA size or assigning epilogue warps to other work (e.g., next-tile TMA prefetch).
