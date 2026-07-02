# Deep Dive into CUTLASS FMHA and MLA Implementations

CUTLASS provides reference implementations of Fused Multi-Head Attention (FMHA) for two generations of architectures: Hopper (SM90, example 88) and Blackwell (SM100, example 77), covering forward inference (context / generation), backward training, and Multi-Latent Attention (MLA) inference. This article, based on CUTLASS source code, systematically organizes the core design patterns of FMHA kernels.


**Last updated**: 2026-06-30

## Table of Contents

- [Two-Phase GEMM Structure](#two-phase-gemm-structure)
- [Shared Memory Management](#shared-memory-management)
- [Pipeline Orchestration](#pipeline-orchestration)
- [Online Softmax](#online-softmax)
- [Masking Strategy](#masking-strategy)
- [Attention Variants (MQA/GQA/MLA)](#attention-variants-mqagqamla)
- [Variable Sequence Length](#variable-sequence-length)
- [FP8 KV Cache and Mixed Precision](#fp8-kv-cache-and-mixed-precision)
- [Hopper vs Blackwell FMHA Architecture Differences](#hopper-vs-blackwell-fmha-architecture-differences)
- [Low-Latency GQA (Example 93)](#low-latency-gqa-example-93)
- [Backward Pass](#backward-pass)
- [Reference File Index](#reference-file-index)

---

## Two-Phase GEMM Structure

The core computation of FMHA is a GEMM-Softmax-GEMM fused pipeline:

```
                 QK^T matmul          Softmax          PV matmul
               +-----------+      +-----------+     +-----------+
  Q [M x D] -->| GEMM #1   |--S-->|  Online   |--P->| GEMM #2   |--> O [M x D]
  K [N x D] -->| Q * K^T   |     |  Softmax  |     | P * V     |
               +-----------+      +-----------+     +-----------+
                                                   V [N x D] -->
```

### TileShape Definition

CUTLASS uses `TileShape = Shape<SeqQ_tile, SeqK_tile, HeadDim>` to uniformly describe tiling:

| Parameter | Blackwell Context | Blackwell Gen | Hopper |
|------|------------------|---------------|--------|
| SeqQ (M) | 256 | 128 (NumGroups) | SeqQ |
| SeqK (N) | 128 | 64/128/256 | SeqK |
| HeadDim (D) | 32/64/128 | 128 | 32/64/128/256 |

### QK and PV GEMM Shape Transformations

```cpp
// QK GEMM: (M, N, K) = (SeqQ_tile, SeqK_tile, HeadDim)
using TileShapeQK = Shape<SeqQ_tile/ThreadShape_M, SeqK_tile/ThreadShape_N, HeadDim>;

// PV GEMM: Rearrange dimensions via select<0,2,1>
// (M, N, K) = (SeqQ_tile, HeadDim, SeqK_tile) -- Note K=SeqK_tile
using TileShapePV = select<0,2,1>(TileShapeQK);
```

The P matrix (output of softmax) does not go through SMEM, but is instead used directly in TMEM (Blackwell) or registers (Hopper) as the A operand of GEMM #2.
## Shared Memory Management

### K/V Shared Union

The K and V matrices are temporally mutually exclusive (first load K for QK matmul, then load V for PV matmul), so they share the same block of SMEM:

```cpp
array_aligned<Element, cosize_v<SmemLayoutQ>> smem_q;  // Q allocated independently
    union {
      array_aligned<Element, cosize_v<SmemLayoutK>> smem_k;  // K and V shared
      array_aligned<Element, cosize_v<SmemLayoutV>> smem_v;
    };
```

This union design is completely consistent across both the Hopper and Blackwell implementations.

### Blackwell TMEM Allocation

Blackwell introduces Tensor Memory (TMEM) for MMA accumulators, and the TMEM layout in FMHA is carefully designed with overlapping:

```
TMEM address space:
  S0 [128 cols] | S1 [128 cols] | O0 [128 cols] | O1 [128 cols]
  ^               ^               ^               ^
  |-- QK acc #0   |-- QK acc #1   |-- PV acc #0   |-- PV acc #1
  |-- P0 (32 cols embedded in S0)         |
  |-- V0 (statistics, overlaps with S0)       |-- V1 (statistics, overlaps with S1)
```

The TMEM regions for S and P overlap — S stores FP32 QK accumulation results, and after softmax, they are converted to low precision and written back to the P region (a subset within the S space).

---

## Pipeline Orchestration

### Blackwell: 6 Independent Pipelines

Blackwell FMHA uses 5 warp roles (Softmax0/1, Correction, MMA, Load, Epilogue) and up to 16 warps, decoupled through 6 pipelines:

```
Load ──PipelineQ──> MMA ──PipelineS0──> Softmax0 ──PipelineC0──> Correction ──PipelineE──> Epilogue
     ──PipelineKV──>     ──PipelineS1──> Softmax1 ──PipelineC1──>
                         ──PipelineO───────────────────────────>
```| Pipeline | Type | Stages | Producer | Consumer |
|----------|------|--------|----------|----------|
| PipelineQ | PipelineTmaUmmaAsync | 2 | Load warp | MMA warp |
| PipelineKV | PipelineTmaUmmaAsync | 3-4 | Load warp | MMA warp |
| PipelineS0/S1 | PipelineUmmaAsync | 1 | MMA warp | Softmax0/1 warp |
| PipelineC0/C1 | PipelineAsync | 1 | Softmax0/1 | Correction warp |
| PipelineO | PipelineUmmaAsync | 2 | MMA warp | Correction warp |
| PipelineE | PipelineAsync | 2 | Correction | Epilogue warp |

The **number of KV stages** depends on the data type: FP8 uses 4 stages, FP16/BF16 uses 3 stages.

### Hopper: 2 Pipelines + Warpgroup Coordination

Hopper uses the classic Producer (1 warp group load) + Consumer (2+ warp groups MMA) pattern:

```
Load WG ──PipelineQ (TmaAsync, 2 stages)──> MMA WG0, MMA WG1
         ──PipelineKV (TmaAsync, 5 stages)──> MMA WG0, MMA WG1
```
Hopper relies on `warpgroup_fence_operand` + `warpgroup_arrive/commit_batch` to overlap MMA and softmax, with softmax executed inline within the MMA warp group.

### Blackwell Pingpong Scheduling

Blackwell's MMA warp pingpongs between two S tiles:

```
Timeline:
  Q1*K1->S0 | Q2*K1->S1 | P0*V1->O0 | Q1*K2->S0 | P1*V1->O1 | Q2*K2->S1 | P0*V2->O0 | ...
```

This allows softmax0 and softmax1 to process their respective S tiles in parallel, while the MMA alternates in producing them.

---

## Online Softmax

### Algorithm Overview

FMHA uses online softmax (also known as the FlashAttention algorithm), which computes softmax without requiring two passes:

```
For each new K block:
  1. Compute S = Q * K^T * scale
  2. Update row_max: new_max = max(old_max, max(S))
  3. Compute correction = exp2(scale_log2 * (old_max - new_max))
  4. Rescale: O *= correction, row_sum *= correction
  5. Compute P = exp2(scale_log2 * (S - new_max))
  6. Accumulate: O += P * V, row_sum += sum(P)

Final: O /= row_sum
```

### Blackwell Implementation Details

Blackwell's softmax operates between TMEM and registers:

```cpp
// 1. TMEM_LOAD: Load S from TMEM to registers
copy(tiled_tmem_load, tTMEM_LOADtS, tTMEM_LOADrS);

// 2. Optional: apply_mask (masked iterations only)
Mask{}.apply_mask(tTMEM_LOADrS, tTMEM_LOADcS, problem_size);

// 3. Row-wise max
float row_max = ...;  // 4-way unrolled reduction

// 4. Write old/new row_max to TMEM's V region (for correction warp)
tTMEM_STOREVrS(kIdxOldRowMax) = old_row_max;
tTMEM_STOREVrS(kIdxNewRowMax) = row_max_safe;
copy(tiled_tmem_storev, tTMEM_STOREVrS, tTMEM_STOREVtS);

// 5. scale + exp2 + type conversion (FP32 -> Element)
// Use log2(e) to avoid exp, use faster exp2f instead
float out = scale_log2 * S[i] - row_max * scale_log2;
P[i] = exp2f(out);

// 6. TMEM_STORE: Write P back to TMEM for MMA PV matmul
copy(tiled_tmem_store, tTMEM_STORErS_x4, tTMEM_STOREtS_x4);
```

The **Correction warp** independently performs the O rescale: it reads old/new max, computes `exp2(scale * (old - new))`, then does a TMEM load-multiply-store to rescale the O accumulator in place.

### Numerical Stability

- `row_max_safe = (row_max == -INFINITY) ? 0 : row_max` avoids NaN on empty rows
- Uses `exp2f` instead of `expf` (faster hardware instruction), combined with `scale * log2(e)`
- LSE output: `LSE = log(row_sum) + scale * row_max`

---

## Masking Strategy

CUTLASS defines 4 mask types, all implemented in `collective/fmha_fusion.hpp`:

| Mask Type | Trip Count Optimization | apply_mask Behavior |
|-----------|------------------------|---------------------|
| `NoMask` | All unmasked | No-op |
| `ResidualMask` | Last 1 tile masked when seqK % tileN != 0 | Sets `pos >= seqK` to -INF |
| `CausalMask<kIsQBegin>` | Truncates K tiles by Q row | Sets `q_pos < k_pos` to -INF |
| `CausalForBackwardMask` | Combination causal + residual | Used for BWD pass |### CausalMask Trip Count Optimization

```cpp
// CausalMask::get_trip_count: Skip unnecessary KV tiles using causality
int max_blocks_q = ceil_div((blk_coord_m + 1) * tile_M, tile_N);
return min(max_blocks_k, max_blocks_q);  // Upper triangular part completely skipped

// get_masked_trip_count: 1-2 tile requires mask
return min(trip_count, ceil_div(tile_M, tile_N));
```

This means **most tiles are unmasked**—the main loop does not need per-element checks, and only the last few tiles execute `apply_mask`.

### Dual Q Position Modes

`CausalMask` supports `kIsQBegin=true` (Q at the start of the matrix, common in training) and `kIsQBegin=false` (Q at the end of the matrix, inference cache scenarios), implemented via `offset_q = seqK - seqQ` offsets.

---

## Attention Variants: MQA/GQA/MLA

### CuTe Layout Expression for MQA and GQA

By leveraging CuTe's stride algebra, all attention variants can be expressed without special-case code:

```
MHA:  Q layout = (numHeads : headStride)     KV layout = (numHeads : headStride)
MQA:  Q layout = (numHeads : headStride)     KV layout = (numHeads : 0)          // stride=0 broadcast
GQA:  Q layout = (numHeads : headStride)     KV layout = (numHeads/G, G : headStride, 0)
```

When the KV head stride is 0, all Q heads share the same KV data—this is MQA.

### MLA (Multi-Latent Attention)

Example 77 includes a complete MLA inference kernel (`sm100_fmha_mla_tma_warpspecialized.hpp`), with the following key features:

- **Weight-absorbed regime**: latent head dim = 512, rope head dim = 64
- **2SM mode**: uses the Blackwell 2CTA tensor core (`Allocator2Sm`) to accommodate large accumulators
- **ProblemShape**: `Shape<TileShapeH=128, int(seqK), TileShapeD=(512,64), int(batch)>`
- **Paged KV**: supports TMA loading (page size 128) or `cp.async` (any 2^n page size <= 128)
- **Variable sequence length + paging** combination support

MLA backward uses a special shape of `d=192, d_vo=128`, implemented in `sm100_fmha_bwd_mla_kernel_tma_warpspecialized.hpp`.

---

## Variable Sequence Length

Fixed integer dimensions in the problem shape are replaced with `VariableLength` type:

```cpp
int max_length;              // Maximum length for chunked computation
int* cumulative_seqlen_ptr;  // Cumulative sequence length array [0, len0, len0+len1, ...]
int total_length = -1;

operator int() const { return max_length; }  // Implicitly convert to int for ceil_div etc
```

At runtime, the actual length is obtained based on the batch index:

```cpp
auto logical_shape = apply_variable_length(params.problem_shape, batch_idx);
// VariableLength in shape is replaced with:
//   cumulative_length[batch_idx+1] - cumulative_length[batch_idx]
```

Offset calculations are also automatically adjusted, and the `cumulative_length[batch_idx]` offset is added when writing outputs.

---

## FP8 KV Cache and Mixed Precision

### Forward Pass FP8

- QK GEMM accumulator can be FP32 or FP16 (`ElementAccumulatorQK`)
- Blackwell: `SM100_MMA_F8F6F4_SS` / `SM100_MMA_F8F6F4_TS` instructions
- Supports `scale_q`, `scale_k`, `scale_v` dequantization factors + `inv_scale_o` output quantization
- Softmax scale fusion: `scale_softmax = scale_q * scale_k * (1/sqrt(D))`

### Mixed-Input Decode (CuTeDSL Python)

CuTeDSL examples such as `mixed_input_fmha_decode.py` demonstrate mixed-precision decode with int4/int8 KV cache: Q remains high precision, while KV is loaded in quantized format and dequantized within the GEMM.

---

## Hopper vs Blackwell FMHA Architecture Differences

| Feature | Hopper (SM90, Example 88) | Blackwell (SM100, Example 77) |
|------|--------------------------|-------------------------------|
| **MMA Instruction** | WGMMA SS/RS (warp group) | UMMA SS/TS (single warp) |
| **Accumulator Storage** | Registers | TMEM (Tensor Memory) |
| **Warp Organization** | 3 warp groups (1 load + 2 MMA) | 16 warps (5 roles) |
| **Softmax Location** | Inline in MMA warp group | Dedicated Softmax warp (2 groups × 4 warps) |
| **P Matrix Passing** | Registers (`make_acc_into_op`) | TMEM in-place conversion |
| **Load Method** | TMA (all) | TMA (context) / cp.async (gen) |
| **Pipeline Count** | 2 (Q + KV) | 6 (Q, KV, S0, S1, O, E) |
| **Multi-WG Coordination** | `OrderedSequenceBarrier` | `PipelineUmmaAsync` |
| **Register Allocation** | Automatic per warp group | Manual `warpgroup_reg_set<N>()` |
| **HeadDim Support** | 32, 64, 128, 256 | 32, 64, 128 |### Hopper's GMMA RS Mode

PV GEMM uses RS (Register-Shared) mode — P reads from registers, V reads from SMEM:

```cpp
TiledMmaPV tiled_mma_pv;  // = convert_to_gmma_rs(CollectiveMmaPV::TiledMma)
// Convert P to RS operand format
Tensor acc_qk_fixed = make_acc_into_op<Element>(acc_qk, TiledMmaPV::LayoutA_TV{});
// Execute PV matmul directly as A operand
cute::gemm(tiled_mma_pv, acc_qk_fixed, tOrV, acc_pv);
```

### Blackwell's UMMA TS Mode

PV GEMM uses TS (TMEM-Shared) mode — P reads from TMEM, V reads from SMEM:

```cpp
TiledMMA mma_pv_ts = to_tiled_mma_sm100_ts(mma_pv);  // SS -> TS conversion
// P already in TMEM (softmax writes directly)
gemm_zero_acc(mma_pv_ts, tOrP0, tOrV, tOtO0);  // P0 in TMEM, V in SMEM
```

### Blackwell's Register Budget Management

Blackwell manually allocates register counts for each warp role:

```cpp
// Softmax warps: 192 regs (need many registers for row reduction)
// Correction warps: 96 regs
// MMA/Load/Epilogue: 32-48 regs
// Empty warp: 24 regs (donate registers to other warps)
```

---

## Low-Latency GQA (Example 93)

Example 93 (`tgv_gqa`) is a Blackwell GQA kernel optimized for low batch generation scenarios:

### Design Highlights

- **Cluster flash decoding**: Each cluster (`1x1xMAX_SPLITS`) processes one KV head, with the KV sequence evenly split across CTAs within the cluster
- **7 warp collaboration**: DMA_Q (1), DMA_KV (1), MMA (1), EPILOG (4: softmax + cluster reduction)
- **Cluster reduction**: Exchange fmax/fsum/acc across CTAs via DSMEM (distributed shared memory)
- **Supports**: attention sink + sliding window, flash decoding split, bf16/fp8 KV cache

### Reduction Flow

```
Each CTA independently computes partial attention:
  1. MMA warp: Q*K^T -> S, softmax(S)*V -> partial_O
  2. EPILOG warps: Compute local fmax, fsum

Cross-CTA reduction:
  3. fmax: credux + inter-warp reduction -> store to remote dsmem
  4. fsum: reswizzle through smem -> intra-thread/warp reduction -> dsmem
  5. acc2: Reduction CTAs collect partial_O from each CTA, rescale with fmax/fsum and accumulate
```

---

## Backward Pass

### 5-GEMM Structure

FMHA backward consists of 3 kernels:

1. **SumOdO kernel** (`FmhaKernelBwdSumOdO`): Computes `D[i] = sum_j(O[i,j] * dO[i,j])`
2. **Main BWD kernel** (`Sm100FmhaBwdKernelTmaWarpspecialized`): Core 5-GEMM structure
3. **Convert kernel** (`FmhaKernelBwdConvert`): Converts dQ from FP32 to output precision

The 5 GEMMs of the main BWD kernel:

```
GEMM 1: S = K * Q^T                (recompute attention scores)
GEMM 2: dP = V * dO^T              (compute dP from V and dO)
         P = softmax(S)             (recompute softmax)
         dS = P * (dP - D)          (elementwise: P .* (dP - rowsum(dO*O)))
GEMM 3: dQ += dS^T * K             (accumulate dQ)
GEMM 4: dK += dS * Q               (accumulate dK -- in TMEM)
GEMM 5: dV += P^T * dO             (accumulate dV -- in TMEM)
```

### TMEM Allocation (BWD)

The BWD kernel's TMEM is more compact, leveraging the temporal mutual exclusion of dQ and dP:

```cpp
struct TmemAllocation {
  kDK = 0;                       // dK accumulator
  kDV = kDK + TileShapeDQK;      // dV accumulator (adjacent to dK)
  kDQ = kDV + TileShapeDVO;      // dQ accumulator
  kDP = kDQ;                     // dP shares with dQ (time-mutual)
  kS  = kDQ + max(Q,DQK);        // S (recomputed attention scores)
  kP  = kS;                      // P shares with S
};
```

### BWD Warp Division

```
Warp assignment: 0x12'3333'3333'4444
  Warps  0-3: Reduce (4 warps)
  Warps  4-11: Compute (8 warps, softmax/correction)
  Warp   12: MMA
  Warp   13: Load
  Warp 14-15: Empty (donate regs)
```

## Reference File Index

### Blackwell FMHA (Example 77)

| File | Content |
|------|------|
| `collective/fmha_fusion.hpp` | NoMask, ResidualMask, CausalMask, VariableLength |
| `collective/fmha_common.hpp` | gemm_zero_acc, UMMA SS/TS conversion, warp_uniform |
| `collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp` | Forward main loop (TMA context) |
| `collective/sm100_fmha_gen_mainloop_warpspecialized.hpp` | Generation main loop (cp.async) |
| `collective/sm100_fmha_mla_fwd_mainloop_tma_warpspecialized.hpp` | MLA forward main loop |
| `kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp` | Forward kernel: 5 warp role scheduling |
| `kernel/sm100_fmha_bwd_kernel_tma_warpspecialized.hpp` | Backward kernel: 5-GEMM structure |
| `kernel/sm100_fmha_mla_tma_warpspecialized.hpp` | MLA inference kernel: 2SM mode |

### Hopper FMHA (Example 88)

| File | Content |
|------|------|
| `collective/fmha_collective_tma_warpspecialized.hpp` | Forward main loop: WGMMA SS+RS |
| `collective/fmha_collective_softmax.hpp` | Online softmax: shfl_xor reduction |
| `kernel/fmha_kernel_tma_warpspecialized.hpp` | Kernel layer: Producer/Consumer warp groups |

### Related Documentation

- [CUTLASS GEMM Optimization](cutlass-gemm-optimization.md)
- [SM100 CuTeDSL Programming](../../blackwell/cutedsl/blackwell-cutedsl-sm100.md)
- CUTLASS Programming Model


## Related

- [CuTeDSL API Reference Guide](cutedsl-api-reference-guide.md)
- [CuTeDSL Inline PTX Writing Overview](cutedsl-inline-ptx-patterns.md)
- [CuTeDSL Software Pipeline and Synchronization Patterns](cutedsl-pipeline-patterns.md)
- [CuTeDSL Programming Model](cutedsl-programming-model.md)
- [CUTLASS 3.x Architecture](cutlass-3x-architecture.md)
- [CUTLASS GEMM Optimization Strategy](cutlass-gemm-optimization.md)
- [Community GEMM Optimization Practical Summary](../../../generic/gemm-optimization-guide.md)
- [CUTLASS/CuTe Core Concepts and Layout Algebra](cutlass-cute-fundamentals.md)
