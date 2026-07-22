# Software Pipeline Depth Optimization

## Overview

In GPU kernels, the latency of global memory (HBM/GMEM) is much higher than computation latency. **Software pipelining** hides memory latency by using multi-level buffering to overlap memory loads with computation. Pipeline depth (stage count) is a key parameter affecting kernel performance — too few stages cause compute units to idle waiting for data, while too many stages waste SMEM and reduce occupancy.

---

## 1. Why Multi-Stage Pipelining Is Needed

### 1.1 Problem: Memory Latency Is Much Greater Than Computation

| Operation | Typical Latency |
|------|---------|
| HBM Read (H100) | ~400-600 cycles |
| L2 Cache Hit | ~200 cycles |
| SMEM Read | ~20-30 cycles |
| WGMMA 16x256x16 (FP16) | ~16 cycles |
| MFMA 32x32x16 (FP16, MI300X) | ~64 cycles |

With only 1 SMEM buffer, the kernel execution flow is:

```
Time →
[===== Load tile 0 =====][= Compute tile 0 =][===== Load tile 1 =====][= Compute tile 1 =]
                          ↑                                         ↑
                     Compute unit idle                            Compute unit idle
```

### 1.2 Solution: Producer-Consumer Pipeline

Use N SMEM buffers to build a pipeline, allowing DMA (producer) and MMA (consumer) to work in parallel:

```
2-stage pipeline:
Producer: [==load tile 0==][==load tile 1==][==load tile 2==][==load tile 3==]
Consumer:                  [==comp tile 0==][==comp tile 1==][==comp tile 2==]

4-stage pipeline:
Producer: [=ld 0=][=ld 1=][=ld 2=][=ld 3=][=ld 4=][=ld 5=][=ld 6=][=ld 7=]
Consumer:                          [=cp 0=][=cp 1=][=cp 2=][=cp 3=][=cp 4=]
                  ↑ Fill 3 buffers ↑ Start computation, buffer 0 ready
```

Pipeline depth = Number of SMEM buffers = Maximum number of steps the producer can lead the consumer.

### 1.3 Synchronization Mechanism

A pipeline requires two types of synchronization:

1. **Producer → Consumer**: "Data has been written to buffer X" (barrier / mbarrier)
2. **Consumer → Producer**: "Buffer X has been consumededy and can be overwritten" (release)

```
Producer perspective:        Consumer perspective:
  acquire(buf[i])  ←——————  release(buf[i])    "buf[i] available?"
  load(buf[i])
  commit(buf[i])   ——————→  wait(buf[i])       "buf[i] data ready?"
                            compute(buf[i])
                            release(buf[i])
```

Hopper+'s `mbarrier` (asynchronous barrier) natively supports this producer-consumer model, enabling hardware-level pipelining when combined with TMA.

---

## 2. Computing Optimal Pipeline Depth

### 2.1 Basic Formula

```
num_stages >= ceil(T_memory / T_compute) + 1
```

Where:
- `T_memory`: Time to load one tile into SMEM
- `T_compute`: Time to compute one tile
- `+1`: At least one extra buffer is needed seks to hold the tile currently being computed

### 2.2 Estimating T_memory

```
T_memory = tile_bytes / effective_bandwidth

tile_bytes = sizeof(ElementA) * M_tile * K_tile + sizeof(ElementB) * N_tile * K_tile
```

Effective bandwidth needs to consider:
- Theoretical HBM bandwidth (H100: 3.35 TB/s, MI300X: 5.3 TB/s)
- Actual utilization (typically 70-85%, depending on access patterns)
- Multi-CTA bandwidth contention

### 2.3 Estimating T_compute

```
T_compute = total_MMA_ops / MMA_throughput

total_MMA_ops = (M_tile / M_mma) * (N_tile / N_mma) * (K_tile / K_mma)
MMA_throughput = MMA_per_cycle * clock_freq  (but typically calculated using cycles)
```

Latency and throughput of each MMA instruction:

| Instruction | Architecture | Shape | Latency (cycles) | Throughput (per SM per cycle) |
|------|------|------|-------------|----------------------|
| HMMA | Ampere | 16x8x16 | ~32 | 1/warp-group |
| WGMMA | Hopper | 64x256x16 | ~64 | 1/SM |
| UMMA/tcgen05 | Blackwell | 128x256x32 | ~64 | 1/SM |
| MFMA | MI300X | 32x32x16 | ~64 | 4/CU |

### 2.4 Worked Example: H100 GEMM

Parameters:
- Tile: M=128, N=256, K=64
- Data type: FP16 (2 bytes)
- Architecture: H100 (SM90)

**Step 1: tile_bytes**
```
A_tile = 128 * 64 * 2B = 16,384 B = 16 KB
B_tile = 256 * 64 * 2B = 32,768 B = 32 KB
tile_bytes = 48 KB
```**Step 2: T_memory (Rough Estimate)**
```
H100 HBM: 3.35 TB/s, 132 SMs
Per-SM bandwidth ≈ 3350 / 132 ≈ 25.4 GB/s
T_memory ≈ 48 KB / 25.4 GB/s ≈ 1.85 μs ≈ 2960 cycles (at 1.6 GHz)
```

**Step 3: T_compute**
```
WGMMA 64x256x16 (FP16): 1 MMA instruction covers 64x256x16
tile needs: (128/64) * (256/256) * (64/16) = 2 * 1 * 4 = 8 WGMMA instructions
Assume WGMMA pipeline throughput ≈ 16 cycles/op (pipeline throughput, not latency)
T_compute ≈ 8 * 16 = 128 cycles ≈ 0.08 μs
```

**Step 4: num_stages**
```
num_stages >= ceil(2960 / 128) + 1 = 24 + 1 = 25 ???
```

> This number seems very large! In practice we don't need (and can't achieve) the theoretical optimum because:
> 1. Multiple CTAs execute concurrently, and the GPU can switch to other CTAs to hide latency
> 2. SMEM capacity limits the number of stages
> 3. In actual compute-bound kernels, T_compute is larger (the above estimate is optimistic)

In practice, H100 GEMM kernels typically use **4-7 stages**, relying on concurrent CTA execution to hide the remaining latency.

---

## 3. SMEM Budget Constraints

### 3.1 SMEM Cost per Stage

```
stage_footprint = A_tile_bytes + B_tile_bytes + pipeline_overhead

A_tile_bytes = sizeof(A) * M_tile * K_tile
B_tile_bytes = sizeof(B) * N_tile * K_tile
pipeline_overhead ≈ 32-64 bytes  (mbarrier / flag storage)
```

### 3.2 Total SMEM Constraint

```
num_stages * stage_footprint + aux_smem <= available_smem
```

Where `aux_smem` includes:
- Epilogue output tile buffer
- Scale/zero matrices (quantization kernels)
- Softmax row max / row sum (attention kernels)
- Other temporary storage

### 3.3 Available SMEM Capacity

| Architecture | Max SMEM/SM | Notes |
|------|-------------|------|
| Ampere (SM80) | 164 KB | Requires configuring `cudaFuncSetAttribute` to exceed 48 KB |
| Hopper (SM90) | 227 KB (232448 bytes) | TMA uses some SMEM for descriptors |
| Blackwell (SM100) | 227 KB (232448 bytes) | TMEM reduces SMEM requirements |

### 3.4 SMEM vs Occupancy Trade-off

More stages → more SMEM → fewer CTA/SM → lower occupancy:

```Example (H100, FP16, tile 128x256x64):
  3 stages: 3 * 48.1 = 144.3 KB → Can accommodate 1 CTA (227 KB enough)
  4 stages: 4 * 48.1 = 192.4 KB → Can accommodate 1 CTA
  5 stages: 5 * 48.1 = 240.5 KB → Exceeds! Cannot run

  Reduce tile to 128x128x64:
  stage_footprint = 16 KB + 16 KB = 32 KB
  4 stages: 128 KB → Can accommodate 1 CTA
  7 stages: 224 KB → Can accommodate 1 CTA (just enough)```

### 3.5 Sweet Spot Analysis

| Scenario | Typical Stage Count | Reason |
|------|-------------|------|
| GEMM (large tile) | 3-5 | Tight SMEM budget |
| GEMM (small tile) | 5-7 | Small tiles allow more stages |
| GEMM (FP8) | 5-8 | Smaller data type, lower per-stage footprint |
| Attention (FMHA) | 3-5 (KV), 1-2 (Q) | Asymmetric pipeline |
| Convolution | 3-5 | Similar to GEMM |

---

## 4. Automatic Pipeline Depth Calculation

### 4.1 AutoCarveout Mode

In real-world frameworks, the stage count is usually computed automatically. Core algorithm:

```python
def compute_stages(smem_capacity, tile_shape, elem_a, elem_b, carveout_bytes):
    """Calculate maximum stages accommodatable given SMEM capacity"""
    # Bytes per stage
    a_bytes = elem_a.size * tile_shape.M * tile_shape.K
    b_bytes = elem_b.size * tile_shape.N * tile_shape.K
    pipeline_bytes = 64  # mbarrier / flags

    stage_bytes = align_up(a_bytes + b_bytes, 128) + pipeline_bytes

    # Available capacity = Total capacity - kernel extra overhead (epilogue, TMA descriptor, etc.)
    available = align_down(smem_capacity, 128) - align_up(carveout_bytes, 128)

    return available // stage_bytes
```

Key Points:
- `carveout_bytes`: SMEM usage in the kernel outside of the mainloop (epilogue buffers, TMA descriptors, etc.)
- Auto-calculation ensures no hardware limit is exceeded
- Usually there is also a lower bound check: `stages >= 2` (at least double buffering is required)

### 4.2 Typical Stage Counts (from Production Kernels)

**SM90 (Hopper) GEMM:**| Tile MxNxK | Data Type | Auto Stage Count | SMEM Usage |
|-----------|---------|-------------|----------|
| 128x256x64 | FP16 | 4 | ~192 KB |
| 128x128x64 | FP16 | 7 | ~224 KB |
| 256x128x64 | FP16 | 4 | ~192 KB |
| 128x256x64 | FP8 | 7 | ~168 KB |
| 128x128x64 | BF16 | 7 | ~224 KB |

**SM100 (Blackwell) GEMM:**

Blackwell's UMMA instructions use **TMEM** (Tensor Memory, register-level storage within the SM), reducing dependency on SMEM:
- The A matrix can be read directly from TMEM (bypassing the SMEM A buffer)
- Only the B matrix SMEM buffer + TMEM-loaded A buffer is needed
- Result: typical stage count drops to 2-4 (less SMEM pressure, but TMEM itself has capacity limits)

| Tile MxNxK | Data Type | Typical Stage Count |
|-----------|---------|-------------|
| 128x256x128 | FP16 | 2-3 |
| 256x128x128 | FP8 | 3-4 |
| 128x256x64 | BF16 | 3-4 |

---

## 5. Pipelining for Non-GEMM Kernels

### 5.1 Flash Attention — Asymmetric Pipelining

The unique characteristic of Attention: the Q matrix is reused throughout the entire KV loop and does not need frequent reloading:

```
Standard GEMM pipeline (symmetric):
  A buffer: [stage 0][stage 1][stage 2][stage 3]
  B buffer: [stage 0][stage 1][stage 2][stage 3]
  → Same number of stages on both sides

Attention pipeline (asymmetric):
  Q buffer: [stage 0]  ← Load once, reside in SMEM (or RMEM/TMEM)
  K buffer: [stage 0][stage 1][stage 2][stage 3]  ← Multi-stage pipeline
  V buffer: [stage 0][stage 1]                     ← Can have fewer stages

  Q only needs 1 buffer (prologue loads once)
  K needs 3-5 stages (main loop memory-bound part)
  V needs 2-3 stages (overlap with softmax)
```

Key insight of this design:
- Q's access pattern is "load once, use many times," so it is not the pipeline bottleneck
- K is on the critical path and needs the most stages
- V loading can overlap with softmax and P*V computation

### 5.2 Convolution

Convolution can be viewed as implicit GEMM, with the same pipelining model:

```
stage_footprint = activation_tile + filter_tile
               = (M_tile * K_tile + spatial_overhead) * sizeof(T)
               + (N_tile * K_tile) * sizeof(T)
```

Note: expanding the spatial dimensions of convolution may increase the SMEM cost per stage.

### 5.3 General Principle: "Keep the Math Unit Fed"

For any kernel, the goal of pipelining is the same:

```
Goal: MMA/MFMA units always have data to compute

Judgment criteria:
  If T_memory <= (num_stages - 1) * T_compute + T_other_overlap
    → compute-bound (compute unit won't starve)  ✓
  Otherwise
    → memory-bound (compute unit waiting for data)  Need more stages or more concurrent CTAs
```

---

## 6. Tuning Pipeline Depth

### 6.1 Diagnosis: Too Few Stages

Symptoms:
- MMA units stall frequently (waiting for SMEM data to become ready)
- Compute utilization falls below theoretical peak
- Performance improves immediately after increasing stages

smsp__warps_issue_stalled_short_scoreboard    ← MMA waiting for data
smsp__warps_issue_stalled_wait               ← General waiting
l1tex__t_sector_hit_rate                     ← SMEM hit rate (should be close to 100%)
launch__occupancy                             ← Theoretical occupancy
sm__warps_active.avg.pct_of_peak_sustained    ← Actual warp active rate
shared_utilization                            ← SMEM utilization

### 6.3 Tuning Workflow

```
1. Start from auto-calculated stage count (usually the maximum SMEM can accommodate)
2. Gradually reduce stage count, observe performance changes:
   - If performance barely changes → Original stages too many, reducing can improve occupancy
   - If performance drops sharply → Reached minimum necessary stage count
3. Find "knee point": stage count where performance starts to drop significantly
4. Use values near knee point (typically +1 as safety margin)
```

### 6.4 Practical Experience

```
 sweet spot:

GEMM (large tile, FP16):     4-5 stages
GEMM (small tile, FP16):     6-7 stages
GEMM (FP8):                  5-8 stages
Attention (KV pipeline):     3-5 stages
Attention (Q pipeline):      1-2 stages
Reduction / Norm: 2-3 stages (ifpipeline)
```

---

## 7. Interaction Between Pipeline Depth and Other Optimizations

### 7.1 Relationship with Occupancy

```
                Performance
                    ^
                    |        ╭──── Sweet spot
                    |       ╱  ╲
                    |      ╱    ╲
                    |     ╱      ╲
                    |    ╱        ╲
                    |   ╱          ╲── Occupancy decrease
                    |  ╱
                    | ╱── Stall reduction
                    +────────────────→ num_stages
                    2  3  4  5  6  7

Few stages: MMA stall high     Many stages: SMEM full, occupancy low
```### 7.2 Relationship with Warp Specialization

Hopper's warp specialization allows producers and consumers to run in different warp groups:
- Producer warp group: only performs TMA loads
- Consumer warp group: only performs MMA computations
- Independent scheduling, naturally supporting deep pipelining
- May require fewer stages (since the producer and consumer are truly parallel)

### 7.3 Relationship with Persistent Kernel

Persistent kernels process multiple tiles using a single CTA, reducing kernel launch overhead:
- The pipeline can work continuously across tiles (the prologue does not need to re-warm-up for each tile)
- However, the initial warm-up still requires `num_stages - 1` loads

---

## 8. Summary

| Element | Key Points |
|------|---------|
| **What is pipeline depth** | Number of SMEM buffers enabling DMA and MMA to overlap execution |
| **Theoretical formula** | `stages >= ceil(T_mem / T_compute) + 1` |
| **Practical constraints** | SMEM capacity, occupancy trade-offs, using multiple concurrent CTAs to help hide latency |
| **Typical GEMM values** | SM90: 4-7, SM100: 2-4 |
| **Tuning method** | Start from the maximum and decrease, find the knee point |
| **Non-GEMM** | Attention uses asymmetric pipelining; the general principle is "keep the math unit fed" |

---

## Related Documentation

- [Async Copy and Synchronization Primitives](nvidia-ptx-sync-and-async.md) — Synchronization details for mbarrier, cp.async, and TMA
- Occupancy Tuning — Trade-offs between SMEM usage and occupancy
- [Shared Memory Swizzling](smem-swizzling-bank-conflicts.md) — SMEM layout optimization
- [L2 Cache Persistence](../kernel-opt/l2-cache-persistence.md) — Global memory access optimization
