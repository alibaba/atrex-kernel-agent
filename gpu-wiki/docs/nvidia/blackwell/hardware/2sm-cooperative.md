# Two-SM Cooperative MMA


**Last updated**: 2026-06-30

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

## Advantages

- **Doubles the effective MMA tile** — M up to 256 (`m256×n256×k16`) per instruction; a single-SM MMA caps at M=128 and reaches only ~50% of tensor-core peak, so 2-SM is the way to approach 100%.
- **Splits the accumulator across two SMs' TMEM** — each CTA holds half (e.g. 128×N), so a tile whose accumulator would not fit one SM's 512 TMEM columns can still run. This is the main reason to go 2-SM for large head-dim / large-N tiles.
- **Halves per-CTA operand SMEM traffic** — with cluster TMA multicast each CTA loads only its half of A/B, cutting redundant DRAM→SMEM movement.

## Overheads / Costs

2-SM is **not free**; weigh these against the advantages:

- **Cross-CTA synchronization tax** — a `shared::cluster` mbarrier / cluster membar plus the leader↔peer handshake is on the critical path every tile. For kernels that are *not* capacity- or throughput-bound, this sync can cost more than the 2-SM speedup (it has been measured at tens of percent of runtime in sync-heavy attention kernels).
- **Mapping constraint** — the pair must be one cluster of 2 mapped to the same TPC (`cluster=(2,1)`); reduces scheduler freedom and can worsen wave quantization for small grids.
- **Leader/peer coordination** — only the even (leader) CTA issues the MMA; TMEM alloc/**dealloc must be symmetric across the pair** (asymmetric ordering can deadlock), and peer-mbarrier addressing must be correct (handled by CUTLASS/CuTeDSL helpers — don't hand-roll).
- **All-or-nothing per kernel** — every tcgen05 op in the kernel must use the same `cta_group`; you cannot mix 1-SM and 2-SM.

## When to Use — decide by *net* cost

Use 2-SM when the **capacity or throughput need outweighs the sync tax**:

- Large compute-bound GEMM (M ≥ 256) where peak FLOPS matters, combined with persistent scheduling.
- Tiles whose accumulator genuinely needs both SMs' TMEM to fit.

Prefer **1-SM** when the tile already fits one SM's TMEM and the kernel is sync- or latency-bound (e.g. an attention tile whose S+O accumulators fit 512 columns at single-stage — see [tmem.md](tmem.md) budget): there the cluster sync tax can dominate and 1-SM is faster. Profile both; choose by measured net cost, not by "tile is large ⇒ 2-SM".

## Related
- [tcgen05-mma](tcgen05-mma.md) -- Base MMA instruction
- [tmem](tmem.md) -- Full TMEM used in 2-SM mode
