# Grouped GEMM for MoE

## Overview

Grouped GEMM computes multiple matrix multiplications with variable M dimensions but shared N and K, directly targeting MoE (Mixture of Experts) inference where each expert processes a different number of tokens. This is the most practically important kernel pattern for MoE serving: during inference, the router sends different token counts to each expert, and grouped GEMM batches all expert computations into a single kernel launch.

Grouped GEMM was Problem 4 (heaviest weight: 40%) of the GPU Mode NVFP4 Hackathon, and is also the core of DeepGEMM's MoE support.

## Problem Structure

```
Standard GEMM:  C = A @ B          (single problem)
Grouped GEMM:   C_i = A_i @ B_i    for i in [0, num_groups)

MoE specialization:
  - N and K are FIXED (same expert architecture)
  - Only M varies (different token counts per expert)
  - B_i may be different weight matrices (per-expert weights)

Example (DeepSeek-V3, 256 experts, top-8 routing):
  Group 0: M=47 tokens -> Expert 0 weights [N, K]
  Group 1: M=23 tokens -> Expert 1 weights [N, K]
  ...
  Group 255: M=31 tokens -> Expert 255 weights [N, K]
```

## DeepGEMM Grouped Layouts

DeepGEMM provides three layouts optimized for different MoE phases:

```cpp
// Layout 1: Contiguous (prefill)
// All expert inputs packed sequentially with cumulative offset array
// Memory: [Expert0 (M0 rows)] [Expert1 (M1 rows)] [Expert2 (M2 rows)]...
// Index:  offsets[0]=0         offsets[1]=M0         offsets[2]=M0+M1
struct ContiguousLayout {
    const fp8_t* A;         // Packed input [sum(M_i), K]
    const fp8_t* B;         // Expert weights [num_experts, N, K]
    float* C;               // Packed output [sum(M_i), N]
    const int* offsets;     // Cumulative M offsets [num_experts + 1]
};

// Layout 2: Masked (decode with CUDA graphs)
// Fixed allocation per expert, binary mask for valid tokens
// Compatible with CUDA graph capture (static shapes)
struct MaskedLayout {
    const fp8_t* A;         // [num_experts, M_max, K]
    const fp8_t* B;         // [num_experts, N, K]
    float* C;               // [num_experts, M_max, N]
    const bool* mask;       // [num_experts, M_max] validity flags
};

// Layout 3: K-grouped (weight gradients in training backward)
// Groups along K-axis instead of M-axis
struct KGroupedLayout {
    const fp8_t* A;         // [M, sum(K_i)]
    const fp8_t* B;         // Per-group B matrices with different K
    float* C;               // [M, N]
    const int* k_offsets;   // Cumulative K offsets
};
```

## CUTLASS Grouped GEMM on SM100

```cpp
// CUTLASS schedule for grouped GEMM on Blackwell
using Schedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100;

// PtrArray mode: array of pointers to per-group A, B, C matrices
// TMA handles variable-offset loads via per-group descriptors
// CLC distributes tiles across groups dynamically

using GemmKernel = cutlass::gemm::kernel::GemmGrouped<
    cutlass::gemm::GemmShape<128, 256, 128>,  // Tile shape
    cutlass::arch::Sm100,
    cutlass::float_e2m1_t,    // NVFP4 operand type
    cutlass::float_e2m1_t,
    float,
    cutlass::layout::RowMajor,
    cutlass::layout::ColumnMajor
>;

// Launch: single kernel handles all groups
GemmKernel::Arguments args{
    num_groups,
    problem_sizes,   // [num_groups] array of {M_i, N, K}
    ptr_A, ptr_B, ptr_C,
    scale_factors_A, scale_factors_B
};
GemmKernel kernel;
kernel.run(args, stream);
```

## Tile Scheduling for Variable M

### Static Scheduling (Precomputed)

```cpp
// Precompute tile-to-expert mapping on host before launch
struct TileInfo {
    int expert_id;
    int tile_m_start;  // Local M offset within this expert
    int tile_n_start;
};

std::vector<TileInfo> build_tile_schedule(
    const int* M_per_expert, int num_experts, int N
) {
    std::vector<TileInfo> schedule;
    for (int e = 0; e < num_experts; e++) {
        for (int m = 0; m < M_per_expert[e]; m += BLOCK_M)
            for (int n = 0; n < N; n += BLOCK_N)
                schedule.push_back({e, m, n});
    }
    return schedule;  // Copy to device; blockIdx.x indexes into this
}
```

### Dynamic Scheduling (CLC / Persistent Kernel)

```cpp
// Persistent kernel with atomic tile counter
// Each thread block loops, grabbing tiles until all are processed
__device__ int g_tile_counter = 0;

__global__ void grouped_gemm_persistent(
    const fp8_t** A_ptrs, const fp8_t** B_ptrs, float** C_ptrs,
    const int* M_per_expert, int N, int K,
    const TileInfo* tile_map, int total_tiles
) {
    while (true) {
        int tile_id = atomicAdd(&g_tile_counter, 1);
        if (tile_id >= total_tiles) return;

        TileInfo info = tile_map[tile_id];
        int M_e = M_per_expert[info.expert_id];

        // Effective BLOCK_M may be smaller for the last tile of an expert
        int eff_m = min(BLOCK_M, M_e - info.tile_m_start);

        // TMA load + tcgen05.mma for this tile
        tma_load(A_ptrs[info.expert_id] + info.tile_m_start * K, ...);
        tma_load(B_ptrs[info.expert_id] + info.tile_n_start, ...);
        tcgen05_mma(...);

        // Store result
        store_tile(C_ptrs[info.expert_id] + info.tile_m_start * N
                   + info.tile_n_start, eff_m, BLOCK_N);
    }
}
```

## The Reward Hack

The 1st-place submission to Problem 4 exploited the evaluation harness:

```
Correctness phase: harness clones data -> real kernel runs correctly
Timing phase: harness reuses same objects ->
    Call 1: fires 120-group super-batch (all 15 benchmark problems fused)
    Calls 2-15: detect pre-computed results, skip computation
```

This reported 11.191us (~2us ahead of second place). It led to improvements in the FlashInfer-Bench evaluation methodology for the MLSys 2026 contest.

## When to Use

- MoE inference: dispatch tokens to experts, compute per-expert projections
- Any workload with multiple GEMMs sharing N and K but varying M
- Prefill (contiguous layout) and decode (masked layout for CUDA graph compatibility)

## Caveats

- Expert load imbalance is the primary practical bottleneck (see [tail-effect](../patterns/tail-effect.md))
- Small M per expert causes thin-GEMM inefficiency on tensor cores
- Masked layout wastes compute on padding when M distribution is skewed
- CLC dynamic scheduling adds hardware overhead vs static precomputed schedules
- TMA alignment (128 bytes) constrains minimum tile dimensions

## Sources

- [GPU Mode NVFP4 Hackathon](https://github.com/gpu-mode/reference-kernels)
- [DeepGEMM Grouped GEMM](https://github.com/deepseek-ai/DeepGEMM)
- [Reward Hack Writeup](https://www.gpumode.com/news/reward-hacking-nvfp4)
- [CUTLASS SM100 documentation](https://docs.nvidia.com/cutlass/latest/CHANGELOG.html)
