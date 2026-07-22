# Software Pipelining and Multi-Stage Buffering

## Overview

Software pipelining overlaps data loading (TMA copies from global to shared memory) with computation (tcgen05.mma or wgmma) by maintaining multiple in-flight tile buffers. A circular buffer of 3-5 stages allows the TMA producer to fill stage N+2 while the MMA consumer processes stage N, hiding the global memory latency entirely. This technique is critical for achieving high utilization on both Hopper and Blackwell.

## Pipeline Progression

The tcgen05-tutorial demonstrates the performance impact of pipelining:

```
No pipelining (load, then compute):    695 TFLOPS  (46%)
3-stage pipeline (TMA + MMA overlap):  940 TFLOPS  (62%)
+ warp specialization:                1476 TFLOPS  (98%)
```

The 35% improvement from pipelining alone (695 to 940 TFLOPS) comes from hiding global memory latency behind MMA computation.

## Multi-Stage Circular Buffer Pattern

The fundamental pattern allocates `NUM_STAGES` copies of each SMEM buffer and cycles through them:

```cuda
// 3-stage circular buffer with mbarrier synchronization
// Stages: [0] loading, [1] ready for MMA, [2] being consumed by MMA
#define NUM_STAGES 3

__global__ void __launch_bounds__(512)
pipelined_gemm(const __grid_constant__ GemmParams params)
{
    extern __shared__ char smem[];

    // Circular buffer layout in shared memory
    // Each stage has its own A and B tile buffers
    half* smem_A[NUM_STAGES];
    half* smem_B[NUM_STAGES];
    for (int s = 0; s < NUM_STAGES; s++) {
        smem_A[s] = reinterpret_cast<half*>(
            smem + s * (TILE_A_BYTES + TILE_B_BYTES));
        smem_B[s] = reinterpret_cast<half*>(
            smem + s * (TILE_A_BYTES + TILE_B_BYTES) + TILE_A_BYTES);
    }

    // mbarrier arrays for producer-consumer sync
    __shared__ uint64_t mbar_load_complete[NUM_STAGES];
    __shared__ uint64_t mbar_mma_complete[NUM_STAGES];

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // Initialize barriers
    if (threadIdx.x == 0) {
        for (int s = 0; s < NUM_STAGES; s++) {
            mbarrier_init(&mbar_load_complete[s], 1);
            mbarrier_init(&mbar_mma_complete[s], 1);
        }
    }
    __syncthreads();

    int num_k_tiles = params.K / TILE_K;

    if (warp_id == 0) {
        // ===== TMA PRODUCER =====
        // Prologue: fill the first NUM_STAGES buffers
        for (int s = 0; s < NUM_STAGES && s < num_k_tiles; s++) {
            if (lane_id == 0) {
                // Set expected TX bytes on mbarrier BEFORE issuing TMA.
                // TMA hardware will arrive on the mbarrier when transfer completes.
                uint32_t tx_bytes = TILE_A_BYTES + TILE_B_BYTES;
                mbarrier_arrive_expect_tx(&mbar_load_complete[s], tx_bytes);
                tma_load_tile_A(smem_A[s], params, s, &mbar_load_complete[s]);
                tma_load_tile_B(smem_B[s], params, s, &mbar_load_complete[s]);
                // NOTE: Do NOT manually arrive after TMA issue -- the TMA
                // hardware signals the mbarrier upon transfer completion.
            }
        }

        // Steady state: load stage s while MMA processes stage s-NUM_STAGES
        for (int k = NUM_STAGES; k < num_k_tiles; k++) {
            int stage = k % NUM_STAGES;
            // Wait for MMA to finish with this buffer
            mbarrier_wait(&mbar_mma_complete[stage]);
            if (lane_id == 0) {
                uint32_t tx_bytes = TILE_A_BYTES + TILE_B_BYTES;
                mbarrier_arrive_expect_tx(&mbar_load_complete[stage], tx_bytes);
                tma_load_tile_A(smem_A[stage], params, k, &mbar_load_complete[stage]);
                tma_load_tile_B(smem_B[stage], params, k, &mbar_load_complete[stage]);
            }
        }

    } else if (warp_id == 1) {
        // ===== MMA CONSUMER =====
        for (int k = 0; k < num_k_tiles; k++) {
            int stage = k % NUM_STAGES;
            // Wait for TMA to fill this buffer
            mbarrier_wait(&mbar_load_complete[stage]);

            // Issue MMA on the filled buffer
            if (lane_id == 0) {
                tcgen05_mma(smem_A[stage], smem_B[stage]);
            }
            __syncwarp();

            // Signal that this buffer is free for reuse
            if (lane_id == 0) {
                mbarrier_arrive(&mbar_mma_complete[stage]);
            }
        }
    }
    // Epilogue warps omitted for clarity
}
```

## Stage Count Selection

The optimal number of pipeline stages depends on the ratio of memory latency to compute time per tile:

| Stages | SMEM Usage | Latency Hiding | Best For |
|--------|-----------|----------------|----------|
| 2 | 2x base | Partial | Small tiles, limited SMEM |
| 3 | 3x base | Full for most GEMMs | Standard choice on Blackwell |
| 4-5 | 4-5x base | Full with margin | Large K, high memory latency |
| >5 | Excessive | Diminishing returns | Rarely justified |

The SMEM budget on Blackwell is 228 KB per SM. For a BF16 GEMM with TILE_M=128, TILE_N=256, TILE_K=64:
- A tile: 128 x 64 x 2B = 16 KB
- B tile: 64 x 256 x 2B = 32 KB
- Per stage: 48 KB
- 3 stages: 144 KB (63% of SMEM, leaves room for barriers and epilogue)
- 5 stages: 240 KB (exceeds SMEM capacity; must use smaller tiles)

## mbarrier-Based Synchronization

The mbarrier (memory barrier) is the hardware primitive that makes pipelining efficient. Unlike `__syncthreads()`, mbarrier supports asymmetric producer-consumer synchronization where only the relevant warp participates:

```ptx
// Phase-based mbarrier protocol for 3-stage pipeline
//
// Each mbarrier tracks a "phase" (0 or 1). The producer flips the phase
// on arrive; the consumer waits for the expected phase.

// Producer: arrive on stage %s (flips phase)
mbarrier.arrive.shared.b64  %state, [%mbar_load + %s * 8];

// Consumer: wait for phase %expected_phase on stage %s
// try_wait is non-blocking; the warp can spin-wait or do other work
WAIT_LOOP:
    mbarrier.try_wait.parity.shared.b64  %ready, [%mbar_load + %s * 8], %phase;
    @!%ready bra WAIT_LOOP;

// TMA can also arrive directly on an mbarrier:
// The TMA unit signals completion without CPU thread involvement
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
    [%smem_addr], [%tensor_map, {%coord0, %coord1}], [%mbar_load + %s * 8];
```

The key advantage is that TMA can arrive on an mbarrier autonomously. The producer warp only needs to initiate the TMA; the TMA hardware signals completion directly, removing the producer from the critical path.

## Modular's 5-Stage Implementation

The Modular blog series describes a 5-stage circular buffer reaching 85% of SOTA performance on Blackwell:

```cuda
// Modular-style 5-stage pipeline constants
// Chosen to fully hide B200 HBM latency (~400 cycles)
// while fitting within 228 KB SMEM budget

constexpr int NUM_STAGES = 5;
constexpr int TILE_M = 128;
constexpr int TILE_N = 128;  // Smaller N to fit 5 stages
constexpr int TILE_K = 64;

// Per stage: A (128*64*2=16KB) + B (64*128*2=16KB) = 32 KB
// 5 stages: 160 KB + barriers + metadata < 228 KB

// The pipeline timing diagram for steady state:
//
// Stage:  0         1         2         3         4
// TMA:   [load k5] [done]    [done]    [load k8] [load k9]
// MMA:   [done]    [done]    [comp k7] [done]    [done]
//
// At any point: 1 stage being loaded, 1 being computed, 3 in transit or done
```

## When to Use

- **All memory-bound and compute-bound GEMM kernels**: Pipelining is never harmful and always improves utilization by hiding latency.
- **Attention kernels**: The K-dimension loop in attention benefits from pipelining the KV tile loads.
- **Combined with warp specialization**: Pipelining provides the buffer structure; warp specialization assigns the producer/consumer roles. The two techniques are complementary and almost always used together.

## Caveats

- More stages increase SMEM usage linearly. On Blackwell's fixed 228 KB, this constrains tile size choices.
- Barrier initialization overhead is negligible but must happen before the first TMA. Place init in a `__syncthreads()` block at kernel start.
- Incorrect phase tracking in mbarrier causes deadlocks. The phase alternates with each arrive/wait cycle; off-by-one errors are common during development.
- For very short K dimensions (fewer iterations than stages), the prologue/epilogue overhead may dominate. Guard the loop bounds accordingly.
