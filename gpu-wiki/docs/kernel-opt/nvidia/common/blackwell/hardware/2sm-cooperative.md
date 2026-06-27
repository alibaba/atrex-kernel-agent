# Two-SM Cooperative MMA

## Overview

Blackwell enables two SMs within a TPC to cooperatively execute a single larger MMA, doubling the effective compute tile size to m256×n256×k16.

## How It Works

```
TPC (Two Processing Clusters)
├── SM 0: CTA 0 -- issues tcgen05.mma with cta_group::2
│   ├── Shared Memory A (rows 0-127)
│   └── TMEM (columns 0-255)
└── SM 1: CTA 1 -- cooperates on same MMA
    ├── Shared Memory A (rows 128-255)
    └── TMEM (columns 256-511)
```

## PTX

```ptx
// 2-SM cooperative MMA
tcgen05.mma.cta_group::2.kind::f16
    [tmem_addr], descA, descB, idescC, idescD, ...;
```

## Requirements
1. **Identical shared memory layouts** across both CTAs
2. `shared::cluster` mbarrier signaling between the two CTAs
3. Both CTAs in the same cluster
4. Each CTA contributes half the M-dimension

## Performance Impact

From tcgen05 tutorial progression:
- 1-SM MMA (m128×n256): 80% of cuBLAS → adding 2-SM: **86%** of cuBLAS
- ~7.5% improvement from doubling the MMA tile size

## When to Use
- Large GEMM problems where M ≥ 256
- Compute-bound kernels where peak FLOPS matters
- Combined with persistent scheduling for maximum throughput

## Related
- [tcgen05-mma](tcgen05-mma.md) -- Base MMA instruction
- [tmem](tmem.md) -- Full TMEM used in 2-SM mode
