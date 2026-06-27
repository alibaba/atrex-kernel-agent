# Ping-Pong Scheduling

## Overview

Ping-pong scheduling alternates two query tiles within a single CTA so the softmax warpgroup never stalls waiting for MMA. Introduced in FlashAttention-4 to exploit Blackwell's asymmetric hardware (2× tensor cores, same SFU count as Hopper).

## Pattern

```cuda
// Two 128-token query tiles per CTA, alternating through the mainloop
// Warpgroup 0: softmax for tile A while MMA runs on tile B
// Warpgroup 1: softmax for tile B while MMA runs on tile A

__global__ void fa4_ping_pong_attn(...) {
    int wg = warp_group_id();

    // TMEM holds accumulators for BOTH tiles
    uint32_t tmem_A = tmem_alloc(256);
    uint32_t tmem_B = tmem_alloc(256);

    for (int k = 0; k < num_kv_tiles; k++) {
        if (wg == 0) {
            // Compute softmax for tile A (previous MMA output)
            softmax_normalize(tmem_A);
            // Issue MMA for tile B next
            tcgen05_mma(Q_B_smem, K_smem[k], tmem_B);
        } else {
            softmax_normalize(tmem_B);
            tcgen05_mma(Q_A_smem, K_smem[k], tmem_A);
        }
        mbarrier_arrive(&ping_pong_sync);
        mbarrier_wait(&ping_pong_sync);
    }
}
```

## Why It Helps on Blackwell

- Tensor core throughput doubled (B200 vs H100) but SFU count unchanged
- Single-tile schedule would leave SFU idle while MMA runs, and vice versa
- Ping-pong keeps both units 100% busy
- FA4 achieves 1605 TFLOPS BF16 (71% utilization) with this pattern

## When To Use

- Compute-bound attention kernels on Blackwell
- Kernels where softmax/epilogue is SFU-heavy
- Not useful on Hopper (balance is different)
