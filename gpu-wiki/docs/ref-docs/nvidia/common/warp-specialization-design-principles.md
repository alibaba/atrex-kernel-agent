# Warp Specialization Design Principles

Warp specialization is a core optimization strategy in modern GPU kernel design. This document explains **when to use, why it works, and how to design** warp specialization kernels, focusing on design decisions at the hardware level rather than specific framework APIs.

> Related documentation: [PTX Synchronization and Asynchronous Operations](nvidia-ptx-sync-and-async.md) | [Occupancy Tuning](../../../kernel-opt/nvidia/common/occupancy-tuning-by-arch.md) | [Register Pressure and Occupancy](register-pressure-warp-occupancy.md)

---

## 1. What is Warp Specialization

### Traditional Mode: Cooperative

All warps execute the same code path and share the same workload:

```
// GEMM: warp load + compute
for (int k = 0; k < K; k += TILE_K) {
 load_tile_A(smem_A, gmem_A); // warp
 load_tile_B(smem_B, gmem_B); // warp
    __syncthreads();
 mma(accum, smem_A, smem_B); // warp
    __syncthreads();
}
```

Problem: Load and compute are **serial**. Tensor Cores are idle during load, and the TMA unit is idle during compute.

### Specialized Mode: Role-based Division

Different warps execute different code paths, utilizing different hardware units concurrently:

```
// Warp Specialization: producer load, consumer compute, row
if (is_producer_warp) {
    for (int k = 0; k < K; k += TILE_K) {
 pipeline.acquire(stage); // wait buffer
 tma_load(smem[stage], gmem, k); // 1 TMA
 pipeline.commit(stage); // notify consumer data
    }
} else {  // consumer warp
    for (int k = 0; k < K; k += TILE_K) {
 pipeline.wait(stage); // waitdata
 wgmma(accum, smem[stage]); // 128 execute Tensor Core MMA
 pipeline.release(stage); // notify producer buffer
    }
}
```

### Why It Works

Modern GPUs have **independent hardware units** for different operations:

| Hardware Unit | Function | Launch Method |
|--------------|----------|---------------|
| TMA Unit (SM90+) | Global memory to shared memory transfer | 1 thread issues, hardware auto-executes |
| Tensor Core (WGMMA/UMMA) | Matrix multiply-accumulate | 128 threads (1 warp group) collaborate |
| CLC Unit (SM100+) | Cluster Launch Control scheduling | 1 thread issues |
| TMEM (SM100+) | Tensor Memory read/write | Coupled with Tensor Core |

Warp specialization enables these hardware units to work **in parallel** rather than taking turns. Theoretically, kernel execution time can be reduced from `T_load + T_compute` to `max(T_load, T_compute)`.

---

## 2. When to Use Warp Specialization (Decision Framework)

### Decision Flowchart

```
 kernel use TMA / asynchronousload？
├── -> use( warp load)
English description
 ├── Compute-bound(compute)->
 │ : warp spec warp producer( 1 TMA),
 │ compute warp
    │
 ├── Memory-bound-> Warp spec
 │ : datacompute, memory latency
    │
 └── -> Warp spec + Pingpong compute epilogue
 : consumer MMA epilogue,
```

### Rules of Thumb

| Condition | Recommended Strategy | Reason |
|-----------|---------------------|--------|
| NCU shows >20% time in memory stall | Warp specialization | Load/compute overlap can eliminate stalls |
| Tensor Core utilization >80% | Cooperative | Already compute-bound, no overlap needed |
| Epilogue takes >15% of time | Pingpong or 3-role | Epilogue can overlap with next MMA round |
| Multiple independent data streams (e.g., X/Delta/B/C in Mamba) | Multi-role | Dedicated warp per TMA stream |
| Simple reduction / elementwise | Cooperative | Role division overhead exceeds benefits |

### By Kernel Type

### By Kernel Type

| Kernel Type | Warp Spec Benefit | Notes |
|------------|-------------------|-------|
| GEMM | Almost always beneficial | TMA load and WGMMA are fully independent |
| Flash Attention | Beneficial | Multiple rounds of GEMM + softmax, load overlap is effective |
| Conv2d / Conv3d | Beneficial | Equivalent to implicit GEMM |
| LayerNorm / Softmax | No benefit | Simple computation, nothing to overlap |
| Elementwise | No benefit | Too little computation, warp spec overhead is greater |
| SSM / Mamba | Significant benefit | Multiple independent data streams, naturally suited for multi-role |

## 3. Role Partitioning Strategies

### 3.1 Two-Role Basic (Hopper)

**Structure**: 1 producer warp group (128 threads) + 1 consumer warp group (128 threads)

```
 0-127: Producer (TMA load A + B)
 128-255: Consumer (WGMMA + Epilogue)
Total: 256 threads = 8 warps
```

**Characteristics**:
- Only **1 thread** in the producer actually issues TMA instructions; the remaining 127 threads are idle
- All 128 threads of the consumer are used for WGMMA
- Suitable for small tiles or simple GEMMs

**Issue**: After the consumer completes MMA, it still needs to perform the epilogue (type conversion, store), while the producer is already idle.

### 3.2 Pingpong (Hopper)

**Structure**: 1 producer warp group (128 threads) + 2 consumer warp groups (128 threads each)

```
 0-127: Producer WG0
  ├── Warp 0: TMA Load (mainloop)
  ├── Warp 1: Scheduler (tile scheduler)
  ├── Warp 2: Epilogue Load (TMA load C)
  └── Warp 3: Auxiliary Load (optional)
 128-255: Consumer0 (MMA -> Epilogue -> MMA -> ...)
 256-383: Consumer1 (Epilogue -> MMA -> Epilogue -> ...)
Total: 384 threads = 12 warps
```

**Key Design**: The two consumers **alternately** execute MMA and epilogue:

```
English description
Consumer0: [MMA tile0] [Epi tile0] [MMA tile2] [Epi tile2] ...
Consumer1: [Epi ----] [MMA tile1] [Epi tile1] [MMA tile3] ...
Producer:  [Load 0] [Load 1] [Load 2] [Load 3] ...
```

When Consumer0 performs MMA, Consumer1 does the epilogue from the previous iteration, and vice versa. This way, the epilogue latency is **completely hidden**.

**Actual Data** (from CUTLASS SM90 implementation):
- `MaxThreadsPerBlock = 384` (must be exactly 384)
- `NumMmaWarpGroups = 2`
- The producer warp is further divided into 4 roles (Mainloop / Scheduler / Epilogue / AuxLoad)
- `MathWarpGroupOrderBarrier` ensures strict alternation between the two consumers

### 3.3 Cooperative Consumption (Hopper Cooperative)

**Structure**: 1 producer warp group + 2 consumer warp groups, but the two consumers **collaborate** on the same tile

```
 0-127: Producer WG
 128-255: Consumer0 ─┐
 256-383: Consumer1 ─┘ Jointly compute 1 tile
Total: 384 threads = 12 warps
```

**Difference from Pingpong**:
- Pingpong: 2 consumers process **different** tiles, alternating MMA/Epilogue
- Cooperative: 2 consumers process the **same** tile, providing 2x MMA throughput

**Applicable Scenarios**:
- Large tiles (>=128x128): `static_assert(size<0>(TileShape{}) >= 128)`
- Requires stream-K splitting: cooperative supports split-K fixup
- When the compute is so large that a single warp group's MMA throughput is insufficient

**Actual Data**:
- `NumMMAThreads = 256` (8 warps, 2 warp groups)
- Requires `TileShape M >= 128`
- Consumer uses `mma_thread_idx = thread_idx % 256` for partitioning

### 3.4 Five-Role (Blackwell SM100)

**Structure**: 5 independent warps, each warp with 1 role

```
Warp 0: MMA          (Tensor Core UMMA computation)
Warp 1: Scheduler    (CLC scheduling, get next tile)
Warp 2: Mainloop Load (TMA load A/B)
Warp 3: Epilogue Load (TMA load C)
Warp 4+: Epilogue     (epilogue computation + TMA store D)
Total: 160 threads = 5 warps
```

**Key Changes**:
- Blackwell's UMMA instruction only requires **1 warp** (32 threads) to issue MMA, no longer needing a warp group
- Accumulators are stored in **TMEM** (Tensor Memory) instead of registers, significantly reducing register pressure
- CLC (Cluster Launch Control) introduces an independent scheduling warp
- Epilogue can consist of multiple warps (determined by `NumEpilogueThreads`)

**Role Determination Logic**:

```cpp
// Blackwell: map role directly using warp index
int warp_idx = canonical_warp_idx_sync();
WarpCategory warp_category;
if (warp_idx < 4) {
    warp_category = WarpCategory(warp_idx);  // MMA=0, Sched=1, Load=2, EpiLoad=3
} else {
    warp_category = WarpCategory::Epilogue;  // Warp 4+ are all epilogue
}
```

### 3.5 Multi-Role (Mamba2 SSD: 7+ Independent Pipelines)

For the Mamba2 SSD kernel, the data flow is far more complex than GEMM, requiring the loading of multiple tensors such as X, Delta, DeltaA, B, C, D, and Z:**7 independent pipelines**: X / Delta / B / C / Cooperate / D / Z, each pipeline has its own mbarrier and stage state.

### How to Choose the Number of Roles

Decision basis: **Number of independent operations** and **matching of hardware resources**.

```
Step 1: List all independent operations in kernel
  For example GEMM: load_A, load_B, compute_MMA, epilogue_load_C, epilogue_store_D

Step 2: Mark hardware unit used by each operation
  load_A/B:       TMA Unit (1 thread to launch)
  compute_MMA:    Tensor Core (128 or 32 threads)
  epilogue_load:  TMA Unit (1 thread to launch)
  epilogue_store: TMA Unit (1 thread to launch)

Step 3: Merge operations using same hardware unit into same warp
  TMA loads can share 1 warp (serial launch, but TMA hardware executes in parallel)
  MMA needs independent warp/warp group

Step 4: Check if register budget allows so many warps
  See register-pressure-warp-occupancy.md
```

---

## 4. Inter-Role Synchronization

### 4.1 Pipeline Model

The synchronization core of warp specialization is the **circular buffer pipeline**:

```
Shared memory buffer:  [Stage 0] [Stage 1] [Stage 2] ... [Stage N-1]

Producer:  acquire(stage) → write data → commit(stage)
Consumer:  wait(stage) → read and compute → release(stage)

Stage state:  Empty → Filling → Full → Draining → Empty
              ↑                                      │
              └──────────────────────────────────────┘
```

Pipeline depth (number of stages) determines how many steps ahead the producer can be relative to the consumer. Typical values:
- GEMM mainloop: 2-7 stages (depending on shared memory size)
- Epilogue load: 2 stages
- CLC scheduler: 2-4 stages

### 4.2 mbarrier (SM90+)

The pipeline uses **mbarrier** (hardware barrier) at the lower level squared to implement producer-consumer synchronization:

```cpp
// Producer: TMA automatically arrives at barrier after completion
tma_load.with(*barrier, mcast_mask);  // TMA hardware auto-arrives after completion

// Consumer: wait for barrier
barrier.try_wait(phase);  // non-blocking check
barrier.wait(phase);      // blocking wait
```

Advantages of mbarrier:
- **Hardware implementation**: ~5 cycles latency, vs ~20 cycles for software barriers
- **TMA integration**: Automatically arrives after TMA completion, no CPU thread intervention required
- **Cross-CTA**: Can synchronize across thread blocks within a cluster
- **Transaction counting**: Can track the number of bytes transferred to ensure data integrity

### 4.3 OrderedSequenceBarrier

Used to ensure execution order **within the same role**. For example, two consumers in pingpong must alternate:

```
OrderedSequenceBarrier<SequenceDepth=2, SequenceLength=2>

Consumer0: wait() → [MMA] → arrive() → wait() → [Epilogue] → arrive()
Consumer1:          wait() → [MMA] → arrive() → wait() → [Epilogue] → arrive()
```

Parameter meanings:
- `SequenceDepth`: How many ordered phases (2 in pingpong: MMA phase and Epilogue phase)
- `SequenceLength`: How many participants (2 in pingpong: Consumer0 and Consumer1)
- `group_id`: The current participant's index (0 or 1)
- `group_size`: How many threads per participant (128 = 1 warp group)

### 4.4 LoadWarpOrderBarrier

Order guarantee within producer warp groups between mainloop load and epilogue load:

```
Mainloop Load warp: [load A/B complete] → arrive()
Epilogue Load warp: wait() → [start loading C]
```

Ensures that epilogue load starts only after the first round of mainloop load is complete, avoiding races during pipeline startup.

### 4.5 Named Barrier (SM100+)

Blackwell uses named barriers to synchronize accumulators between MMA warps and Epilogue warps:

```cpp
// MMA warp: notify epilogue after computation complete
accumulator_pipeline.producer_commit(state);

// Epilogue warp: wait for accumulator ready
accumulator_pipeline.consumer_wait(state);
```

### Synchronization Overhead

| Synchronization Primitive | Latency | Use Case |
|---------------------------|---------|----------|
| `__syncthreads()` | ~20 cycles | Synchronize all threads within CTA (not suitable for warp spec) |
| mbarrier arrive/wait | ~5 cycles | Producer-consumer signaling |
| OrderedSequenceBarrier | ~10 cycles | Ordered execution within the same role |
| Named barrier | ~5 cycles | Synchronize a subset of threads |
| `__syncwarp()` | ~1 cycle | Intra-warp synchronization |

## 5. Load Balancing Between Roles

### 5.1 The Cost of Imbalance

```
Ideal case (perfect overlap):
Producer: [Load 0][Load 1][Load 2][Load 3]
Consumer: --------[MMA 0 ][MMA 1 ][MMA 2 ][MMA 3]
Total time: T_load + T_compute  (pipeline startup + steady state)

Producer too fast:
Producer: [L0][L1][L2][idle][idle][idle]
Consumer: --------[MMA 0    ][MMA 1    ][MMA 2    ]
→ Producer warp resources wasted

Consumer too fast:
Producer: [Load 0        ][Load 1        ][Load 2        ]
Consumer: --------[MMA 0][stall][MMA 1 ][stall][MMA 2 ]
→ Pipeline bubble, Tensor Core utilization drops
```

### 5.2 Balancing Methods

**Adjust pipeline depth**: Deeper pipelines tolerate greater producer/consumer speed differences.

```
Pipeline depth = 2: Producer at most 1 step ahead
Pipeline depth = 7: Producer at most 6 steps ahead

Deeper depth → tolerate more latency fluctuation → but consumes more shared memory
```

**Adjust tile size**: Affects MMA computation time.

```
128x128 tile: MMA computation time long → consumer slow → need deeper pipeline
64x64 tile:   MMA computation time short → consumer fast → 2-3 stages may be enough
```

### 5.3 `elect_one_sync()` Pattern

TMA requires only **1 thread** to issue, leaving the other threads in the producer warp idle:

```cpp
int lane_predicate = cute::elect_one_sync();
if (lane_predicate) {
    // Only 1 thread executes TMA load
    tma_copy(src, dst, barrier);
}
// Remaining 31 threads (or 127 threads, if warp group) idle
```

**Thread Utilization Analysis** (Hopper Pingpong, 384 threads):

| Role | Threads | Active Threads | Utilization |
|------|---------|----------------|-------------|
| Mainloop Load | 32 | 1 | 3.1% |
| Scheduler | 32 | 1 | 3.1% |
| Epilogue Load | 32 | 1 | 3.1% |
| Aux Load | 32 | 0-1 | 0-3.1% |
| Consumer0 (MMA) | 128 | 128 | 100% |
| Consumer1 (MMA) | 128 | 128 | 100% |
| **Total** | 384 | ~259 | **67.4%** |

The low utilization of producer warps is an inherent cost of warp specialization. It is worthwhile because these "wasted" threads enable **full parallelism between TMA and Tensor Core**.

### 5.4 Blackwell Improvements

SM100 reduces MMA to 1 warp (32 threads), resulting in even lower overall utilization, but higher efficiency per thread:

| Role | Threads | Description |
|------|---------|-------------|
| MMA | 32 | UMMA instructions require only 1 warp |
| Scheduler | 32 | CLC scheduling |
| Mainloop Load | 32 | TMA |
| Epilogue Load | 32 | TMA |
| Epilogue | 32+ | Store + post-processing |
| **Total** | 160+ | |

TMEM keeps accumulators out of the register file, allowing more CTAs to coexist on an SM and improving overall throughput.

---

## 6. Historical Evolution

### Volta / Turing (SM70-75): Software Pipelining

```
All warps cooperate:
for (k) {
    // Phase 1: global → register → shared memory
    gmem_load(reg, gmem[k+1]);      // prefetch next round
    __syncthreads();

    // Phase 2: shared memory → register → MMA
    smem_load(frag, smem[k]);
    mma_sync(accum, frag_a, frag_b);

    // Phase 3: register → shared memory
    smem_store(smem, reg);
    __syncthreads();
}
```

Software double-buffering, with all threads participating in all operations. Loads must pass through registers.

### Ampere (SM80): cp.async + Software Pipelining

```
// cp.async bypasses registers, direct GMEM → SMEM
cp_async(smem[stage], gmem[k]);
cp_async_commit_group();
cp_async_wait_group<STAGES-2>();
```

Introduced asynchronous copy, but all warps still collaborate. Pipeline depth = 2 to multiple stages.

### Hopper (SM90): TMA + Hardware mbarrier + Warp Specialization

```
// TMA: 1 thread launches, hardware handles copy automatically
// mbarrier: hardware synchronization, auto-arrives after TMA completion
// setmaxnreg: dynamic register reallocation
// WGMMA: 128-thread warp group MMA

→ True producer-consumer decoupling
→ Three variants: 2-role, pingpong, cooperative
```

The introduction of TMA hardware made warp specialization possible: only 1 thread is needed to issue a TMA, eliminating the need for many threads to participate in data movement.

### Blackwell (SM100): CLC + TMEM + UMMA + 5+ Roles

```
// CLC: hardware tile scheduling
// TMEM: dedicated accumulator memory, frees registers
// UMMA: 1-warp MMA (no longer needs warp group)
// AccumulatorPipeline: async pipeline MMA → Epilogue

→ Finer-grained role division (5 independent warps)
→ Accumulators flow in TMEM, not through registers
→ Higher CTA concurrency
```### Evolution Trends

```
SM70: 1 role type, all warps same       → serial load/compute
SM80: 1 role type + async copy             → partial overlap
SM90: 2-3 role types (warp group granularity)   → complete load/compute overlap
SM100: 5+ role types (single warp granularity)      → one role per hardware unit
Future:  more hardware accelerator units → more roles
```

**Core Insight**: Every time a new hardware autonomous unit is introduced (TMA, CLC, TMEM), it is worth assigning a dedicated warp to it. When the hardware only needs 1 thread to "give orders," the remaining thread resources are an acceptable overhead—because what you get in return is **true parallelism** across multiple hardware units.

---

## 7. Practical Guide

### Rules of Thumb

1. **If your kernel uses TMA, you should consider warp specialization**. TMA only needs 1 thread; there is no reason to waste an entire warp group's worth of it.

2. **Pipeline depth should be at least 2**. 1 stage cannot overlap (when the producer writes stage 0, the consumer has nowhere to read), so 2 stages is the minimum effective configuration.

3. **Give the producer warp fewer registers**. The producer only does TMA launches and address calculations, so 24-40 registers are sufficient. Save those registers for the consumer to use as accumulators.

4. **Do not over-split roles**. Each additional warp role adds 32 threads of overhead and a set of pipeline synchronization. Only split when operations genuinely use different hardware units.

5. **Check pipeline balance with NCU**. Look at the `smem_pipe_*` related stall metrics. If the consumer frequently stalls on `pipeline.wait()`, it means the producer is too slow (or the pipeline is too shallow).

### Further Reading

- Hopper Pingpong in Practice: `docs/kernel-opt/nvidia/common/sm90/hands-on/` directory
- Blackwell Tri-Role in Practice: `docs/kernel-opt/nvidia/common/hands-on/` directory
- Register Budget Design: [Register Pressure and Warp Occupancy](register-pressure-warp-occupancy.md)
- CUTLASS Source Reference: `sm90_gemm_tma_warpspecialized_pingpong.hpp`, `sm100_gemm_tma_warpspecialized.hpp`
