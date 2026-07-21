# Composable Kernel GEMM Pipeline System

CK ck_tile framework's GEMM pipeline architecture—covering 8 pipeline variants (mem/comp_v1-v6/async/preshuffle), multi-stage prefetch formula (MinMemInFlyBytes=32KB), Intrawave vs Interwave scheduling, Stream-K work stealing, tail specialization, MFMA instruction selection and register control, and warp GEMM attribute system.

> **Applicable Architectures**: CDNA3 (gfx942), CDNA4 (gfx950)
> **Source Code Path**: `composable_kernel/include/ck_tile/ops/gemm/`

---

## 1. Pipeline Variant Overview

CK's GEMM pipeline follows the unified naming `gemm_pipeline_ag_bg_cr_<variant>`, where `ag`=A from global, `bg`=B from global, `cr`=C in register.

| Variant | PrefetchStages | PrefillStages | GlobalBufferNum | HotloopUnroll | Core Features |
|------|---------------|--------------|----------------|---------------|---------|
| **mem** | 2-8 (dynamic) | 1 | 1 | 1 | Memory-optimized, MinMemInFlyBytes automatically calculates prefetch depth |
| **comp_v3** | 2 | 1 | 1 | 1 | Compute-optimized, fine-grained `sched_group_barrier` scheduling |
| **comp_v4** | 2 | 1 | 1 | 1 | Compute-optimized variant |
| **comp_v5** | 1 | 1 | 1 | 1 | Preshuffle-specific, no hot loop tail, TailNumber always Empty |
| **comp_v6** | 3 | 1 | 2 | 2 | Deep prefetch + dual global buffer + loop unrolling |
| **comp_async** | 2 | 1 | 1 | 1 | Async DMA (Global→LDS direct transfer), forces double LDS buffer |

### Selection Guide

| Scenario | Recommended Pipeline | Rationale |
|------|-------------|------|
| Large matrices, HBM bandwidth constrained | mem | Automatically adjusts prefetch depth to hide memory latency |
| Small-to-medium matrices, compute constrained | comp_v3 | Fine-grained instruction interleaving maximizes MFMA utilization |
| B matrix preprocessable (inference) | comp_v5 | Preshuffle skips runtime transpose |
| Sufficient registers, deep pipeline needed | comp_v6 | 3-level prefetch + 2x loop unrolling |
| VGPR constrained | comp_async | Async DMA releases VGPR |

---

## 2. Multi-Stage Prefetch and MinMemInFlyBytes

### 2.1 Core Formula

The `mem` pipeline automatically calculates the optimal prefetch depth based on hardware parameters:

```cpp
constexpr index_t MinMemInFlyBytes = 32768;  // 32 KB

// WGP (Workgroup Processor) shared CU
constexpr auto WgpPerCU = max(4 * warp_size / BlockSize, 1);

// prefetchdata
constexpr auto MemBytesPerPrefetch =
    (MPerBlock * sizeof(ADataType) / PackedA +
     NPerBlock * sizeof(BDataType) / PackedB) * KPerBlock;

// memoryprefetch
constexpr auto FullMemBandPrefetchStages =
    ceil_div(MinMemInFlyBytes / WgpPerCU, MemBytesPerPrefetch);

// [2, 8] range
constexpr auto PrefetchStages =
    clamp(FullMemBandPrefetchStages, 2, 8);
```

### 2.2 Formula Interpretation

- **32KB** is the minimum in-flight data volume required to hide global memory latency on MI300X
- **WgpPerCU** accounts for the case where multiple workgroups share the same CU (`4 * 64 / BlockSize`)
- **PackedA/PackedB** are data packing factors; for FP8, PackedA=2 means two FP8 values are packed into one slot
- **clamp(2, 8)** ensures at least 2 prefetch stages (double buffer) and at most 8 (limited by LDS/register resources)

### 2.3 Prefetch Depth Examples

Using `MPerBlock=256, NPerBlock=256, KPerBlock=64, FP16` as an example:

```
MemBytesPerPrefetch = (256*2 + 256*2) * 64 = 65536 bytes = 64 KB
WgpPerCU = max(4*64/256, 1) = 1
FullMemBandPrefetchStages = ceil(32768/1 / 65536) = 1
PrefetchStages = clamp(1, 2, 8) = 2 ( 2 )
```

Using `MPerBlock=64, NPerBlock=64, KPerBlock=32, FP16` as an example:

```
MemBytesPerPrefetch = (64*2 + 64*2) * 32 = 8192 bytes = 8 KB
WgpPerCU = max(4*64/256, 1) = 1
FullMemBandPrefetchStages = ceil(32768 / 8192) = 4
PrefetchStages = clamp(4, 2, 8) = 4 (4 prefetch)
```

---

## 3. Intrawave vs Interwave Scheduling

### 3.1 Scheduling Enumeration

```cpp
enum struct GemmPipelineScheduler {
 Default, // compilationdefault
 Intrawave, // wave
 Interwave, // wave pipeline
};
```

### 3.2 Intrawave Scheduling

**Core Idea**: Within the instruction stream of a single wave, precisely control the interleaving order of different types of instructions through `__builtin_amdgcn_sched_group_barrier`, enabling MFMA, DS, and VMEM instructions to execute in parallel.The Intrawave HotLoop scheduler example for comp_v3:

```cpp
// Stage 1: DS Write MFMA + prefetch buffer_load
__builtin_amdgcn_sched_group_barrier(
    0x100,  // DS write
    ds_write_issue_cycles,
    0);
__builtin_amdgcn_sched_group_barrier(
    0x008,  // MFMA
    mfma_cycles_per_issue,
    0);
__builtin_amdgcn_sched_group_barrier(
    0x020,  // VMEM (buffer_load)
    num_buffer_load_issues,
    0);

// Stage 2: DS Read MFMA
__builtin_amdgcn_sched_group_barrier(
    0x200,  // DS read
    ds_read_issue_cycles,
    0);
__builtin_amdgcn_sched_group_barrier(
    0x008,  // MFMA
    remaining_mfma_cycles,
    0);
```

Key parameter calculation:

| Parameter | Calculation | Typical Value |
|-----------|-------------|---------------|
| MFMA cycles | NPerXDL >= 32 ? 16 : 32 | 16 (32x32 tile) |
| DS read issue cycles | width >= 16 bytes ? 4 : 8 | 4 (b128) |
| DS write issue cycles | width >= 16 bytes ? 4 : 8 | 4 (b128) |

### 3.3 Interwave Scheduling

**Core idea**: Different waves take on data movement and computation roles respectively, achieving overlap through implicit pipelining between waves.

Key difference: Interwave mode **omits the second `block_sync_lds`**.

```cpp
// Interwave pipeline :
// "no second block_sync_lds because it's interwave"
```

The reason is that during cross-wave scheduling, different waves operate on different LDS regions/stages, eliminating the need for a second full block synchronization.

### 3.4 Comparison

| Feature | Intrawave | Interwave |
|---------|-----------|-----------|
| Parallelism granularity | Instruction-level within a single wave | Task-level across multiple waves |
| Synchronization overhead | High (fine-grained barrier) | Low (one fewer sync) |
| Compiler hint | `sched_group_barrier` | No special hint |
| Applicable scenarios | High occupancy, large tiles | Low occupancy, small tiles |
| LDS sync count | 2 per iteration | 1 per iteration |

---

## 4. Tail Specialization

### 4.1 TailNumber Enumeration

```cpp
enum struct TailNumber {
 // (single/double buffer pipeline)
 Odd, // K iterationcount
 Even, // K iterationcount

 // prefetch pipeline ( 8 )
    One, Two, Three, Four, Five, Six, Seven,

 // loop
 Empty, // UnrollStages > PrefetchStages, loopcount UnrollStages
 Full, // UnrollStages <= PrefetchStages, loopcount UnrollStages + PrefetchStages
};
```

### 4.2 Working Principle

When the K dimension is not divisible by `KPerBlock * PrefetchStages`, the last few iterations need special handling. CK generates template specializations for each remainder value at compile time to avoid runtime branching:

```cpp
// mem pipeline tail
if constexpr (TailNum == TailNumber::One) {
 // 1 iteration
    pipeline_tail_one(a_window, b_window, c_reg, ...);
} else if constexpr (TailNum == TailNumber::Two) {
 // 2 iteration
    pipeline_tail_two(a_window, b_window, c_reg, ...);
}
// ... TailNumber::Seven
```

### 4.3 Special Variants

- **comp_v5 (preshuffle)**: TailNumber is always `Empty`, because the prefetch depth is 1 with no residual handling.
- **comp_v6**: Due to `HotloopUnroll=2`, it also needs to handle `Empty` (unroll alignment) and `Full` (prefetch overflow) as two boundary cases.

---

## 5. Stream-K Work Stealing

### 5.1 Concept

Stream-K evenly distributes the GEMM workload by the number of K-dimension iterations across all CTAs, rather than the traditional static allocation by output tiles. This solves the **load imbalance** problem—when the number of output tiles is not divisible by the number of CUs.

### 5.2 Reduction Strategy

```cpp
enum struct StreamKReductionStrategy {
 Atomic, // atomicreduction - CTA direct atomic_add globaloutput
 // Set: separatewrite - CTA independent buffer, coalesced
};
```

### 5.3 Persistent vs Non-Persistent

| Mode | Grid Size | Characteristics |
|------|-----------|-----------------|
| Persistent | `num_cu * occupancy` | CTAs run continuously, acquiring new tiles via global counter |
| Non-Persistent | `dp_tiles + extra_wg` | Data-parallel tiles + extra workgroups handle the Stream-K portion |

### 5.4 Key Interfaces

```cpp
struct StreamKTilePartitionerBase<BlockGemmShape, ReductionStrategy> {
 // CTA iteration
    __device__ auto get_start_iter(index_t block_id) const;

 // CTA iterationboundary [start, end)
    __device__ auto get_iter_boundaries(index_t block_id) const;

 // iteration -> output tile
    __device__ auto get_tile_index(index_t iter) const;

 // reduction output tile
    __device__ auto get_output_tile_index(index_t block_id) const;
};
```## 6. MFMA Instruction Selection

### 6.1 Complete Instruction Table

#### Floating Point (F32 Accumulator)

| Instruction | M | N | K | Data Type | CVec Count | Architecture |
|------|---|---|---|---------|--------|------|
| mfma_f32_16x16x4f32 | 16 | 16 | 4 | FP32 | 4 | gfx908+ |
| mfma_f32_32x32x2f32 | 32 | 32 | 2 | FP32 | 16 | gfx908+ |
| mfma_f32_32x32x8f16 | 32 | 32 | 8 | FP16 | 16 | gfx908+ |
| mfma_f32_16x16x16f16 | 16 | 16 | 16 | FP16 | 4 | gfx908+ |
| mfma_f32_32x32x16f16 | 32 | 32 | 16 | FP16 | 16 | gfx950 |
| mfma_f32_16x16x32f16 | 16 | 16 | 32 | FP16 | 4 | gfx950 |
| mfma_f32_32x32x8bf16 | 32 | 32 | 8 | BF16 | 16 | gfx90a+ |
| mfma_f32_16x16x16bf16 | 16 | 16 | 16 | BF16 | 4 | gfx90a+ |
| mfma_f32_32x32x16bf16 | 32 | 32 | 16 | BF16 | 16 | gfx950 |
| mfma_f32_16x16x32bf16 | 16 | 16 | 32 | BF16 | 4 | gfx950 |

#### FP8 / BF8

| Instruction | M | N | K | Data Type | CVec Count | Architecture |
|------|---|---|---|---------|--------|------|
| mfma_f32_16x16x32_fp8 | 16 | 16 | 32 | FP8/BF8 | 4 | gfx94x |
| mfma_f32_32x32x16_fp8 | 32 | 32 | 16 | FP8/BF8 | 16 | gfx94x |

#### f8f6f4 Scaled (gfx950)

| Instruction | M | N | K | Feature |
|------|---|---|---|------|
| mfma_f32_16x16x128_f8f6f4 | 16 | 16 | 128 | Independent scale factor per 32 elements |
| mfma_f32_32x32x64_f8f6f4 | 32 | 32 | 64 | Independent scale factor per 32 elements |

#### INT8

| Instruction | M | N | K | Architecture |
|------|---|---|---|------|
| mfma_i32_32x32x16_i8 | 32 | 32 | 16 | gfx94x |
| mfma_i32_16x16x32_i8 | 16 | 16 | 32 | gfx94x |
| mfma_i32_16x16x64_i8 | 16 | 16 | 64 | gfx950 |
| mfma_i32_32x32x32_i8 | 32 | 32 | 32 | gfx950 |

#### TF32 Emulation (gfx950)

| Instruction | M | N | K | Implementation |
|------|---|---|---|---------|
| mfma_f32_32x32x16_tf32 | 32 | 32 | 16 | Emulated via 3x BF16 MFMA |
| mfma_f32_16x16x32_tf32 | 16 | 16 | 32 | Emulated via 3x BF16 MFMA |

#### Special Shapes (FP16)

| Instruction | M | N | K | Use Case |
|------|---|---|---|------|
| mfma_f32_4x64x4_f16 | 4 | 64 | 4 | Narrow M scenarios |
| mfma_f32_64x4x4_f16 | 64 | 4 | 4 | Narrow N scenarios |

### 6.2 Warp GEMM Parameter Definitions

Each MFMA impl defines the following compile-time parameters:

| Parameter | Definition |
|------|------|
| `kM, kN, kK` | Logical tile size of the MFMA instruction |
| `kAMBlock, kBNBlock` | Number of blocks for A/B inputs in the M/N dimension |
| `kAMLane, kBNLane` | Lane mapping for A/B inputs in the M/N dimension |
| `kABKLane` | Number of lanes for A/B in the K dimension |
| `kABKPerLane` | Number of elements held per lane in the K dimension |
| `kCMLane, kCNLane` | Lane mapping for C output in the M/N dimension |
| `kCM0PerLane, kCM1PerLane` | Number of staged elements per lane for C output in the M dimension |

---

## 7. Register File Control (WGAttrCtlEnum)

### 7.1 Control Enumeration

```cpp
enum struct WGAttrCtlEnum {
 Default_, // compilation
    Raw_vvv,     // C=VGPR, A=VGPR, B=VGPR
    Raw_vaa,     // C=VGPR, A=AGPR, B=AGPR
    Raw_vav,     // C=VGPR, A=AGPR, B=VGPR
    Raw_vva,     // C=VGPR, A=VGPR, B=AGPR
    Raw_avv,     // C=AGPR, A=VGPR, B=VGPR
};
```

The three letters represent the register file used by **C (accumulator) - A (input) - B (input)**:
- **V** = VGPR (Vector General Purpose Register)
- **A** = AGPR (Accumulator General Purpose Register)

### 7.2 Selection Strategy

| Configuration | Advantages | Disadvantages | Applicable Scenarios |
|------|------|------|---------|
| `vvv` | Most flexible, all operands in VGPR | High VGPR pressure | When VGPR is sufficient |
| `avv` | Place C accumulator in AGPR to free up VGPR | Reading C back requires `v_accvgpr_read` | Large output tiles |
| `vaa` | Place A/B in AGPR to free up VGPR | Loading inputs requires extra moves | Extremely tight VGPR budget |
| `vav` | A in AGPR, B remains in VGPR | Asymmetric | When B needs frequent reuse |

### 7.3 AGPR Features

- AGPR and VGPR are the same size (512 32-bit per CU)
- MFMA instructions can directly read/write AGPR as accumulators (no extra latency)
- Moving AGPR→VGPR requires the `v_accvgpr_read` instruction (1 cycle)
- Using AGPR can **double** the effective register file capacity

---

## 8. Warp GEMM Attribute System

### 8.1 Hierarchy

```
WarpGemmAttributeMfmaImpl // : MFMA
  ↑
WarpGemmAttributeMfma // : TileDistribution
  ↑
WarpGemmAttributeMfmaIterateK // : K iteration (kK *= kKIter)
  ↑
WarpGemmAttributeMfmaTransposed* // : C , Swizzle A/B
```

### 8.2 WarpGemmAttributeMfma

```cpp
template <typename Impl,
          WGAttrNumAccessEnum AttrNumAccessA = Single,
          WGAttrNumAccessEnum AttrNumAccessB = Single>
struct WarpGemmAttributeMfma {
 // Impl, add tile distribution
    // AttrNumAccessA/B: Single=1, Double=2, Quad=4
 // warp GEMM A/B readcount
};
```

### 8.3 IterateK Extension

```cpp
template <typename Impl, index_t kKIter>
struct WarpGemmAttributeMfmaIterateK {
 // K dimensioniteration kKIter
 // kK = Impl::kK * kKIter
 // decrease K loopcount, increaseinstruction-levelrow
};
```

### 8.4 Transpose and Swizzle Variants

| Variant | Function |
|------|------|
| `TransposedCDistribution` | Swap A/B roles, output C with distributed transpose |
| `TransposedCDistribution_SwizzleB` | Transpose + B register swizzle (controlled by SFactor) |
| `IterateK_SwizzleA` | K iteration + A register swizzle |

Swizzle is used to rearrange the data layout in registers, eliminating the need for cross-lane data exchange.

---

## 9. LDS Double Buffering in GEMM

### 9.1 Standard Double Buffer (comp_async)

```cpp
// double buffer
static_assert(DoubleSmemBuffer == true);

// LDS
constexpr auto GetSmemSize() { return 2 * smem_size; }
```

Ping-pong pattern:

```
iteration 0: LDS[0], compute (prefilled data)
iteration 1: LDS[1], LDS[0] compute
iteration 2: LDS[0], LDS[1] compute
...
```

### 9.2 Synchronization Mechanism

| Pipeline | Write Sync | Read Sync |
|----------|---------|---------|
| Standard (mem/comp) | `block_sync_lds()` | `block_sync_lds()` |
| Async (comp_async) | `block_sync_lds_direct_load()` | `block_sync_lds()` |
| Interwave | `block_sync_lds()` (only once) | Implicit inter-wave pipeline |

The async pipeline uses `block_sync_lds_direct_load()` to wait for the DMA engine to complete Global→LDS direct transfer, as opposed to the standard `block_sync_lds()` which waits for VGPR→LDS writes.

---

## 10. Pipeline Implementation Details

### 10.1 mem Pipeline — Adaptive Prefetch

```
[Prefill stage]
  for i in 0..PrefetchStages-1:
 global_prefetch(tile[i]) // asynchronousprefetch
 block_sync_lds // wait LDS write

[Hot Loop]
  for i in PrefetchStages..num_k_tiles-1:
 lds_read(tile[i-PrefetchStages]) // LDS data
 global_prefetch(tile[i]) // prefetchnext
 mfma(a_reg, b_reg, c_reg) // compute
 block_sync_lds // synchronous

[Tail]
 // 1-7 iteration
```

### 10.2 comp_v3 Pipeline — Fine-Grained Scheduling

```
[Prefill]
  global_load(tile[0]) → VGPR
  block_sync_lds()

[Hot Loop] (Intrawave scheduling)
  ┌─── Stage 1 ───────────────────────┐
 │ ds_write(VGPR->LDS) ←-> MFMA │
 │ buffer_load(prefetch) │
  └────────────────────────────────────┘
  block_sync_lds()
  ┌─── Stage 2 ───────────────────────┐
 │ ds_read(LDS->VGPR) ←-> MFMA │
  └────────────────────────────────────┘
  block_sync_lds()

[Tail]
 Odd/Even
```

### 10.3 comp_v5 Pipeline — Preshuffle

The B matrix is preprocessed on the host side into an MFMA-friendly layout, skipping the B transpose in LDS at runtime:- PrefetchStages=1: No multi-level prefetch needed (B is already optimal layout)
- TailNumber always Empty: No remainder handling
- Suitable for inference scenarios (weights can be preshuffled offline)

### 10.4 comp_v6 Pipeline — Deep Pipeline

__GPU_WIZI_TRANSLATION_PLACEHOLDER_000000__

The three-level prefetch and double unrolling of comp_v6 maximize the utilization of hardware execution units (MFMA, VMEM, DS), but require more register resources.

---

## 11. Practical Recommendations

### Pipeline Selection Decision Tree

__GPU_WIZI_TRANSLATION_PLACEHOLDER_000001__

### MFMA Selection Recommendations

| Scenario | Recommended Instruction | Rationale |
|------|---------|------|
| Large matrix FP16 GEMM | 32x32x8 (gfx908+) or 32x32x16 (gfx950) | Highest throughput |
| Small matrix / Low latency | 16x16x16 | Smaller tile reduces padding waste |
| FP8 inference | 32x32x16_fp8 | Doubled K-dimension throughput |
| Mixed precision scaled | 32x32x64_f8f6f4 (gfx950) | Built-in per-32-element scaling |

---

## Related Documents

- [CK Memory Optimization System](ck-memory-optimization.md) — BufferView, XOR preshuffle, TileWindow, LoadStoreTraits
- [AMD MFMA Matrix Core Programming Guide](amd-mfma-matrix-cores.md) — MFMA naming conventions and register layouts
- [LDS Bank Conflict Optimization](../kernel-opt/lds-bank-conflict-optimization.md) — General bank conflict knowledge
- [Occupancy Optimization](../kernel-opt/occupancy-optimization.md) — Relationship between VGPR/AGPR and occupancy
- [Hardware Specification Comparison](../hardware-specs/hardware-comparison-cdna3-cdna4.md) — CDNA3 vs CDNA4 architecture parameters
