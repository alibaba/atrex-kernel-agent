# CK FMHA Pipelines: Composable Kernel Flash Attention Implementation Details

The Composable Kernel (CK) FMHA (Fused Multi-Head Attention) module implements the complete Flash Attention algorithm, covering forward (multiple pipeline variants) and backward computation. The code is located at `ck_tile/ops/fmha/`, organized in a three-layer structure: **block** (primitives), **pipeline** (core computation flow), **kernel** (launch entry points).

This article is based on CK source code analysis, extracting key designs and optimization techniques.

## Table of Contents

- [Overall Architecture](#overall-architecture)
- [Forward Pipeline Variants](#forward-pipeline-variants-pipeline-enumeration)
  - [qr_ks_vs: Standard Forward](#qr_ks_vs-standard-forward)
  - [qr_ks_vs_async: Asynchronous Copy Variant](#qr_ks_vs_async-async-copy-variant)
  - [splitkv: Long Sequence Sharding](#splitkv-long-sequence-splitting)
  - [pagedkv: Paged KV Cache](#pagedkv-cache-processing)
  - [appendkv: KV Append Write](#appendkv-kv-append-write)
  - [v3: Highly Optimized Variant](#v3-highly-optimized-variant)
  - [batch_prefill: Batch Prefill](#batch_prefill-batch-prefill)
- [Backward Pipeline](#backward-pipeline)
- [Two-Stage GEMM Structure](#two-stage-gemm-structure)
- [Online Softmax Algorithm](#online-softmax-algorithm)
- [Masking System](#masking-system)
- [Attention Variants: Logits Transform and Soft Cap](#attention-variants-logits-transform-and-soft-cap)
- [RoPE Rotary Position Embedding](#rope-rotary-position-encoding)
- [Dropout Support](#dropout-support)
- [PagedKV Cache Processing](#pagedkv-cache-processing)
- [SplitKV Long Sequence Parallelism](#splitkv-long-sequence-parallelism)
- [FP8 Quantization Support](#fp8-quantization-support)
- [Performance Tuning](#performance-tuning)

---

## Overall Architecture

The three-layer structure of CK FMHA:

```
kernel/ # Launch entry + Kargs definition
  fmha_fwd_kernel.hpp           FmhaFwdKernel
  fmha_fwd_splitkv_kernel.hpp   FmhaFwdSplitKVKernel
  fmha_fwd_splitkv_combine_kernel.hpp
  fmha_fwd_pagedkv_kernel.hpp   FmhaFwdPagedKVKernel
  fmha_fwd_appendkv_kernel.hpp  FmhaFwdAppendKVKernel
  fmha_fwd_v3_kernel.hpp        FmhaFwdV3Kernel
  fmha_bwd_kernel.hpp           FmhaBwdKernel
  fmha_batch_prefill_kernel.hpp FmhaBatchPrefillKernel

pipeline/ # corecompute pipeline
 block_fmha_pipeline_qr_ks_vs.hpp standardforward
 block_fmha_pipeline_qr_ks_vs_async.hpp asynchronouscopyforward
 block_fmha_pipeline_qs_ks_vs.hpp Q-in-SMEM forward
 block_fmha_fwd_splitkv_pipeline_*.hpp SplitKV forward
 block_fmha_fwd_pagedkv_pipeline_*.hpp PagedKV forward
  block_fmha_fwd_appendkv_pipeline.hpp      AppendKV
 block_fmha_fwd_v3_pipeline.hpp V3 highoptimization
 block_fmha_fwd_splitkv_combine_pipeline.hpp Combine stage
 block_fmha_bwd_dq_dk_dv_pipeline_*.hpp backward dQ/dK/dV
 block_fmha_bwd_dot_do_o.hpp backward D = dot(dO, O)
 block_fmha_bwd_convert_dq.hpp backward dQ typeconversion

block/ # primitive
  block_masking.hpp             Attention mask (causal/sliding window/custom)
  block_dropout.hpp             Dropout (Philox RNG)
  block_rotary_embedding.hpp    RoPE (interleaved / half-rotated)
 block_position_encoding.hpp ALiBi bit
 page_block_navigator.hpp PagedKV cache
 variants.hpp Attention (standard / logits_soft_cap / composed)
 block_attention_bias_enum.hpp Bias typeenum
 block_attention_quant_scale_enum.hpp quantizationscalingenum
```

The mathematical formula for FMHA forward (defined in kernel file comments):

```
S[seqlen_q, seqlen_k]  = Q[seqlen_q, hdim_q] @ K[seqlen_k, hdim_q]^T
S'[seqlen_q, seqlen_k] = S * scale
S''                     = S' + Bias  (optional)
P[seqlen_q, seqlen_k]  = Softmax(S'')
O[seqlen_q, hdim_v]    = P @ V[seqlen_k, hdim_v]
```

---

## Forward Pipeline Variants### Pipeline Enumeration

```cpp
enum class BlockFmhaPipelineEnum {
    QRKSVS = 0,                    // Q-register, K-SMEM, V-SMEM
 QRKSVS_ASYNC, // same as above + async copy for K
    QSKSVS,                        // Q-SMEM, K-SMEM, V-SMEM
    QRKSVS_ASYNC_TRLOAD,           // async + transpose load (gfx950)
    QRKSVS_ASYNC_TRLOAD_V3,        // V3 pipeline
};
```

### Pipeline Variant Comparison

| Pipeline | Name | Q Location | K/V Loading | PagedKV | SplitKV | Special Features | Target Scenario |
|----------|------|--------|----------|---------|---------|----------|----------|
| `QRKSVS` | `qr` | Register | buffer_load -> LDS | -- | -- | Full feature set | General forward |
| `QRKSVS_ASYNC` | `qr_async` | Register | async copy -> LDS | -- | -- | Double-buffered K | CDNA2+ |
| `QSKSVS` | `qs` | LDS | buffer_load -> LDS | -- | -- | Q also in LDS | Small hdim |
| `SplitKV` | `qr` | Register | buffer_load -> LDS | Optional | **Yes** | Split + combine | Long sequences |
| `PagedKV` | `qr_pagedkv` | Register | page navigate -> LDS | **Yes** | -- | Non-contiguous KV | Inference KV cache |
| `AppendKV` | -- | Register | Direct read/write | Optional | -- | RoPE + KV append | Decode phase |
| `V3` | -- | Register | async + trload | -- | -- | Fine-grained instruction scheduling | gfx950 hdim=128 |
| `BatchPrefill` | `qr_async` | Register | async + page navigate | **Yes** | -- | SGLang page table | Batch prefill |

### qr_ks_vs: Standard Forward

`BlockFmhaPipelineQRKSVS` is the most basic and feature-complete forward pipeline, supporting all feature flags (masking, bias, dropout, logits soft cap, FP8 quantization, sink attention, etc.).

**Naming Convention**: `qr` = Q in Register, `ks` = K in Shared memory (LDS), `vs` = V in Shared memory.

**Core Flow**:

1. **One-time Q Load**: Load the Q tile (M0 x QKHeaddim) from DRAM into registers Madam, reused throughout the entire K sequence iteration
2. **Iterate K Sequence**: Iterate along the seqlen_k dimension with a stride of kN0
3. **Gemm0 (Q x K^T)**: Cache K tiles via LDS, unroll the hdim dimension over multiple rounds of k0_loops
4. **Online Softmax**: Perform online softmax on the S matrix
5. **Gemm1 (P x V)**: Cache V tiles via LDS, accumulate output into O_acc
6. **Finalization**: Normalize O_acc with 1/l

**Dual GEMM LDS Sharing**: K and V share the same LDS space. K uses the `smem_ptr + Q_smem_size` offset, and V starts at `smem_ptr`. Prefetch loading of V (`v_prefetch`) overlaps with Gemm0 execution.

```cpp
// LDS layout
KDataType* k_lds_ptr = static_cast<KDataType*>(
    static_cast<void*>(static_cast<char*>(smem_ptr) +
    Policy::template GetSmemSizeQ<Problem>()));
// V LDS(K )
auto v_lds = make_tensor_view<address_space_enum::lds>(
    reinterpret_cast<VDataType*>(smem_ptr), ...);
```

**K Tile Multi-round Loading and Pipelining**: The hdim dimension is typically larger than the kK0 dimension of a single GEMM, requiring `k0_loops = kQKHeaddim / kK0` iterations. The pipeline overlaps three stages: global load of K data, LDS store, and MFMA execution:

```
loop i:    LDS_sync -> MFMA(k_i) -> LDS_sync -> store(k_{i+1}), load(k_{i+2})
last-1:    MFMA(k_{n-2}) -> store(k_{n-1}) + load(v_prefetch)
last:      MFMA(k_{n-1})
```

### qr_ks_vs_async: Async Copy Variant

`BlockFmhaPipelineQRKSVSAsync` uses CDNA's async copy instruction (`async_copy`) to write K tiles directly from HBM to LDS, bypassing register staging.

- Requires `kPadSeqLenQ == true && kPadHeadDimQ == true && kPadHeadDimV == true` (padding is always used unaffected to align with async copy requirements)
- K double buffering: uses a double-buffer LDS — one buffer for MFMA reads while the other is filled asynchronously
- V still goes through register staging

### splitkv: Long Sequence Splitting

`BlockFmhaFwdSplitKVPipelineQRKSVS` divides seqlen_k into multiple splits, with each workgroup processing a subsequence of K/V.

**Key Differences** (compared to standard forward):
- Outputs `O_acc` and `LSE_acc` to a temporary buffer (rather than the final O)
- Supports `kHasUnevenSplits`: boundary handling when the split length is not evenly divisible
- Obtains the K range each split is responsible for via the mask's `GetTileRangeAlongX(..., num_splits, i_split)`
- Supports `kMergeNumHeadGroupsSeqLenQ` for GQA scenario optimization### pagedkv: Paged KV Cache

`BlockFmhaFwdPagedKVPipelineQRKSVS` handles non-contiguously stored KV cache (PagedAttention mode).

- Implements page table indexing via `PageBlockNavigator`: maps logically contiguous KV sequences to physically scattered page blocks
- Each time the tile window moves, it may cross page boundaries, requiring data pointer updates
- Supports `kDoFp8StaticQuant` for FP8 static quantization on the output

### appendkv: KV Append Write

`BlockFmhaFwdAppendKVPipeline` is responsible for appending new K/V tokens to the KV cache during the inference decode phase.

**Features**:
1. Load Knew/Vnew
2. Optionally apply RoPE to Knew/Q
3. Write to KV cache (supports PagedKV cross-page writes)

```cpp
// Knew, optional RoPE
if constexpr(RotaryEnum != RotaryEmbeddingEnum::NONE) {
    BlockRotaryEmbedding<RotaryEnum>::apply(knew_tile, ...);
}
store_tile(k_dram_block_window, knew_tile);

// PagedKV page
if constexpr(kIsPagedKV) {
    if(k_page_block_navigator.is_cross_block(i_page_block_k, k_dram_block_window)) {
        k_page_block_navigator.move_to_block(...);
        store_tile(k_dram_block_window, knew_tile);
    }
}
```

### v3: Highly Optimized Variant

`BlockFmhaFwdV3Pipeline` is a pipeline deeply optimized for specific hardware.

**Limitations**:
- Only supports `hdim=128` (`kQKHeaddim == 128 && kSubQKHeaddim == 128`)
- Only supports `VLayout = RowMajor`
- Does not support bias, LSE storage, dropout, quantization, skip_min_seqlen_q
- Only supports `GenericAttentionMask`

**Optimization Features**:
- **Fine-grained instruction scheduling**: Uses `CoreLoopScheduler` templates split into masking/non-masking versions, precisely controlling the instruction issue order of MFMA, VALU, TRANS, and SALU via `sched_group_barrier`
- **4-stage scheduling**: Each WaveGroup is divided into Phase 0-3, alternating between GEMM and softmax operations
- **Custom ASM helper functions**: `fma_impl_vsv` forces the use of `v_fma_f32`, while `cvt_pk_fp16_f32` uses `v_cvt_pk_f16_f32`
- **P matrix materialized to LDS**: Extra `kM0 * kN0 * sizeof(PDataType)` LDS space is allocated sessions to store P for Gemm1 read
- **Packed FP32 operations**: Uses packed instructions such as `v_pk_mul_f32`

### batch_prefill: Batch Prefill

`BlockFmhaBatchPrefillPipelineQRKSVSAsync` supports multi-request batch prefill scenarios (e.g., SGLang), combined with async copy and PagedKV:

- Inherits `BlockFmhaPipelineProblem`, adding `kPageBlockSize` parameter
- Supports two KV cache memory layouts: `VECTORIZED_LAYOUT` and `LINEAR_LAYOUT`
- Supports two lookup table formats: `SGLANG_PAGE_TABLE_1D`
- Requires `kIsGroupMode = true` and `VLayout = RowMajor`

---

## Backward Pipeline

### Backward Kernel Composition

CK FMHA backward consists of three kernels:

1. **`block_fmha_bwd_dot_do_o`**: Precomputes `D[i] = dot(dO[i], O[i])`, one scalar per row
2. **`block_fmha_bwd_dq_dk_dv_pipeline`**: Main backward pass, computes dK, dV (output) and dQ (accumulated into fp32 buffer)
3. **`block_fmha_bwd_convert_dq`**: Converts fp32 dQ to the target precision

### Backward Main Pipeline

`BlockFmhaBwdDQDKDVPipelineKRKTRVR` (named `kr_ktr_vr`) uses 5 GEMMs:

| GEMM | Operation | Purpose |
|------|-----------|---------|
| gemm_0 | Q @ K^T | Compute S = Q x K^T |
| gemm_1 | P^T @ dO | Compute dV = P^T x dO |
| gemm_2 | dO @ V^T | Compute dS_part = dO x V^T |
| gemm_3 | dS^T @ Q | Compute dK = dS^T x Q |
| gemm_4 | dS @ K | Compute dQ += dS x K |

**TileFmhaBwdShape** defines the backward tile parameters, with 5 more GEMM-related dimensions than the forward pass:

```cpp
static constexpr index_t kM0 = ...;  // Q seqlen tile
static constexpr index_t kN0 = ...;  // K seqlen tile
static constexpr index_t kK0 = ...;  // gemm0 (Q@K^T) unroll
static constexpr index_t kK1 = ...;  // gemm1 (P^T@dO) unroll
static constexpr index_t kK2 = ...;  // gemm2 (dO@V^T) unroll
static constexpr index_t kK3 = ...;  // gemm3 (dS^T@Q) unroll
static constexpr index_t kK4 = ...;  // gemm4 (dS@K) unroll
```

**Backward iteration direction**: Fixes K/V blocks and iterates along the Q sequence (opposite of forward, which fixes Q and iterates over K). The valid Q range is determined by the mask's `GetTileRangeAlongY()`.**Pipeline Variants**:
- `kr_ktr_vr`: K-register, K-transpose-register, V-register — basic version
- `kr_ktr_vr_iglp`: adds IGLP (instruction group level parallelism) scheduling
- `trload_kr_ktr_vr`: uses transpose load (gfx950)
- `trload_qr_qtr_dor`: transpose load + Q/dO register residency

---

## Two-Stage GEMM Structure

The core of FMHA forward pass is a two-stage GEMM, with each stage executed by a BlockGemm instance provided by the Policy template:

```
┌──────────────────────────────────────────────────────┐
│  Stage 1: GEMM_0 (Q x K^T)                          │
│                                                      │
│  Q_reg [M0, kK0] x K_lds [N0, kK0]^T -> S_acc [M0, N0] │
│ k0_loops = kQKHeaddim / kK0 │
│                                                      │
│ : Q slice from register + K tile from LDS │
│  K tile: DRAM -> register -> LDS -> MFMA              │
└──────────────────────────────────────────────────────┘
                        |
                        v
          ┌─────────────────────────┐
          │  Online Softmax         │
          │  S_acc -> scale/bias    │
          │  -> mask -> exp -> P    │
          └─────────────────────────┘
                        |
                        v
┌──────────────────────────────────────────────────────┐
│  Stage 2: GEMM_1 (P x V)                            │
│                                                      │
│  P_reg [M0, kK1] x V_lds [N1, kK1]^T -> O_acc [M0, N1] │
│ k1_loops = kN0 / kK1 │
│                                                      │
│  V tile: DRAM -> register -> LDS -> MFMA              │
│ (V GEMM_0 executeprefetch) │
└──────────────────────────────────────────────────────┘
```
**Tile Parameter Description** (`TileFmhaShape`):

```cpp
kM0 // Q column tile size ( 64/128)
kN0 // K column tile size ( 64/128)
kK0 // GEMM_0 dimension ( 32)
kN1 // V head_dim tile size
kK1 // GEMM_1 dimension
kQKHeaddim // Q/K head dimension (32~256)
```

**V Layout**: supports both `RowMajor` (seqlen x hdim) and `ColumnMajor` (hdim x seqlen) layouts. RowMajor requires additional register shuffling (`shuffle_tile`) to match the MFMA input layout.

---

## Online Softmax Algorithm

CK implements the classic online softmax (i.e., the core algorithm of Flash Attention), avoiding materializing the full NxN attention matrix:

```
: m = -inf, l = 0, O_acc = 0

for each K tile j:
    S{j} = Q @ K{j}^T                    // GEMM_0
 m_local = rowmax(S{j}) // blockrowmaximum
 m_new = max(m_old, m_local) // globalrowmaximum

 // compute exp
 P{j}[i,k] = exp(S{j}[i,k] - m_new[i]) // or exp2 fast

 // l O_acc(rescale)
 rescale = exp(m_old - m_new) // or exp2
    l = rescale * l + rowsum(P{j})
 O_acc = rescale * O_acc // rescale previous

    O_acc += P{j} @ V{j}                 // GEMM_1

// English comment
O = O_acc / l
```

**Fast exp2 Path**: when `CK_TILE_FMHA_FWD_FAST_EXP2` is enabled, `exp2` is used in place of `exp`, merging the softmax scale into the exponent:

```cpp
// exp2 mode P compute(none bias)
p_compute(i_j_idx) = exp2(scale_s * s[i_j_idx] - row_max);
// where row_max = scale_s * validated_m
```

exp2 is a single-cycle VALU instruction on AMD GPUs (`v_exp_f32`), faster than exp. However, when using bias/ALiBi, the scale needs to be pre-multiplied onto S before adding the bias.**LSE (Log-Sum-Exp) Output**: When `kStoreLSE = true`, store `lse = m + log(l)` for backpropagation:

```cpp
// exp2 mode
lse(i_idx) = m_[i_idx] * scale_s / C_LOG2E + log(l_[i_idx]);
// standardmode
lse(i_idx) = m_[i_idx] + log(l_[i_idx]);
```

**All-Zero Row Handling**: When an entire row is masked out `l == 0`, to avoid division by zero:

```cpp
const auto tmp = l[i_idx] == 0.f ? 0.f : 1 / l[i_idx];
```

---

## Masking System

CK provides three mask implementations, uniformly parameterized via the `(y, x, sink, y_total, x_total)` five-tuple:

### GenericAttentionMask

```cpp
template <bool IsMasking_ = true, bool IsLocal_ = false>
struct GenericAttentionMask;
```

- `IsMasking=false`: No mask, only handles seqlen_k padding
- `IsMasking=true, IsLocal=false`: Causal mask — masks only the upper-right corner
- `IsMasking=true, IsLocal=true`: Sliding window — masks both lower-left and upper-right corners

**Coordinate System**: `(y, x)` defines the diagonal position of the mask:

```
top-left: y = seq_q, x = 1 -> standard causal
bottom-right: y = seq_q, x = seq_k - seq_q + 1
local:        y < seq_q, x < seq_k     → sliding window
no mask:      y = seq_q, x = seq_k
```

Conversion from FlashAttention-style `(left_size, right_size)` window parameters:

```cpp
auto mask_coords = make_generic_attention_mask_coordinates_from_lr_window(
    left_size, right_size, sink_size, seq_q, seq_k, is_top_left);
```

### SimplifiedGenericAttentionMask

Merges `IsLocal` as a runtime check (only mask/nomask two compile variants), reducing code generation at the cost of a few extra instructions in causal mode.

### SimplifiedRatioAttentionMask

Supports GQA merge scenarios (`MergeNumHeadGroupsSeqLenQ`), where the mask's Y-direction stride can be greater than 1.

### Tile-Level Fast Path

`IsEdgeTile()` provides tile-level fast judgment — if the current tile is completely inside or outside the mask, the per-pixel check is skipped:

```cpp
bool need_perpixel_check = mask.IsEdgeTile(
    q_origin.at(number<0>{}), k_origin.at(number<0>{}),
    number<kM0>{}, number<kN0>{});
if(need_perpixel_check) {
    set_tile_if(s_acc, -inf, [&](auto tile_idx) {
        return !variant.LogitsMask(variant_params, batch_idx, row, col, ...);
    });
}
```

### Sink Attention

When `kHasSink = true`, supports "sink tokens" (StreamingLLM), ensuring that several tokens at the beginning of the sequence are always visible:

```cpp
// GetSinkTileRangeAlongX returns (sink_seq_end, start, end)
// sink , causal
```

---

## Attention Variants: Logits Transform and Soft Cap

`variants.hpp` defines a composable attention variant system:

### StandardAttention

Standard attention: `QueryTransform` only multiplies by scale, `LogitsTransform` is an identity transform.

### LogitsSoftCap

Limits the logits range to prevent softmax saturation:

```cpp
// tanh mode
logits_out = logits_soft_cap * tanh(logits * logits_soft_cap_rcp);
// softsign mode(AMD optimization)
logits_out = logits * rcp(1 + abs(logits * logits_soft_cap_rcp));
```

The softsign mode has a specialized inline ASM optimization (`exp2_soft_sign_impl`), keeping softmax_scale in SGPR.

### ComposedAttention

Combines multiple features via bit flags:

```cpp
constexpr uint32_t CUSTOM_MASK     = 1U;
constexpr uint32_t SLIDING_WINDOW  = 2U;
constexpr uint32_t LOGITS_SOFT_CAP = 4U;
constexpr uint32_t ALIBI           = 8U;
```

---

## RoPE Rotary Position Encoding

`BlockRotaryEmbedding` is applied in the AppendKV pipeline, supporting two modes:

```cpp
enum class RotaryEmbeddingEnum {
    NONE         = 0,
 INTERLEAVED = 1, // : (d0, d1), (d2, d3), ...
 HALF_ROTATED = 2, // rotation: (d0, d_{dim/2}), (d1, d_{dim/2+1}), ...
};
```

**Interleaved Mode**:

```cpp
// rotation
new_left  = left * cos - right * sin;
new_right = right * cos + left * sin;
```**Half-rotated Mode**:

```cpp
// English comment
// requiresload "other" half data
new_curr = curr * cos + other * (is_left ? -sin : sin);
```

rotary_dim supports partial dimension rotation (only the first rotary_dim dimensions have RoPE applied, the rest remain unchanged).

---

## Dropout Support

`BlockDropout` uses Philox 4x32 PRNG to generate pseudo-random numbers:

- **Deterministic**: Uses `(seed, offset, batch, head, tile_position)` to determine a unique Philox subsequence, ensuring forward/backward consistency
- **Tile Granularity**: Generates random numbers per 32x32 tile, each tile using an independent subsequence
- **Two-Step Execution**: First generates random bytes, then compares them against a probability threshold cherry dropout

```cpp
// forward: lowthresholdzero out rescale
p_compute(p_idx) = randval[r_idx] <= p_undrop_in_uint8_t
    ? p_compute[p_idx] * rp_undrop
    : PComputeDataType(0);

// backward: use drop
p_compute(p_idx) = randval[r_idx] <= p_undrop_in_uint8_t
    ? p_compute[p_idx]
    : -p_compute[p_idx];
```

**RDNA Compatibility**: The C distribution for gfx11 WMMA differs from MFMA, requiring `PermuteBlockDropoutRandval` for de-interleaving.

---

## PagedKV Cache Processing

`PageBlockNavigator` provides address translation for paged KV cache:

```
logical: [0, 1, 2, ..., seqlen_k-1] x [hdim]
physicalstore: page_table[block_idx] -> physical_blocks + block_stride
```

**Key Operations**:

- `make_tile_window(window_lengths, global_origin)`: Looks up the page table and creates a tile window pointing to the physical page
- `move_tile_window(block_index, tile_window, step)`: Moves the window, potentially across pages, requiring updates to data_ptr and tensor descriptor
- `is_cross_block()`: Detects whether a tile crosses page boundaries
- `prefetch_table_id()`: Prefetches the page table entry for the next page

Two memory layouts:
- **`VECTORIZED_LAYOUT`**: Data within a page is stored aligned by vector size (`page_size % vector_size == 0`)
- **`LINEAR_LAYOUT`**: Data within a page is arranged linearly (supports page_size=1)

`TrivialPageBlockNavigator` is a zero-overhead wrapper for non-paged scenarios.
---

## SplitKV Long-Sequence Parallelism

SplitKV splits seqlen_k into `num_splits` segments, with two kernels working together:

```
Kernel 1: FmhaFwdSplitKVKernel (num_splits workgroup)
  ┌──────────┬──────────┬──────────┐
 │ Split 0 │ Split 1 │ Split 2 │ ... independentcompute
  │ O_acc_0  │ O_acc_1  │ O_acc_2  │
  │ LSE_0    │ LSE_1    │ LSE_2    │
  └──────────┴──────────┴──────────┘
                    |
                    v
Kernel 2: FmhaFwdSplitKVCombineKernel
 split (O_acc, LSE) log-sum-exp coalesced
 output O LSE
```

**Combine Algorithm**:

1. LSE values from all splits are loaded into LDS
2. Compute the global `lse_max = max(LSE_0, LSE_1, ...)`
3. Weight for each split `w_i = exp(LSE_i - lse_max)`
4. Final `O = sum(w_i * O_acc_i) / sum(w_i)`

**`kMaxSplits`**: The maximum number of splits at compile time, determines the LDS size. Supports 4~128 splits.

**GQA Optimization**: `kMergeNumHeadGroupsSeqLenQ` can merge GQA's Q heads into the seqlen_q dimension, using `SimplifiedRatioAttentionMask` to handle non-uniform mask strides.

---

## FP8 Quantization Support

CK FMHA supports multiple quantization modes:

```cpp
enum class BlockAttentionQuantScaleEnum {
 NO_SCALE = 0, // nonequantization
 PERTENSOR = 1, // per-tensor tensor scale
 BLOCKSCALE = 2, // by block granularity scale (K/V shared)
    KV_BLOCKSCALE = 3,   // Q per-tensor, K/V per-page block scale
    MX            = 4,   // Microscaling (e8m0 scale per 32 elements)
};
```

**BLOCKSCALE Mode**: K/V share a single scale value per `block_scale_size_kv` granularity. The GEMM_0 output is multiplied by `k_descale`, and the GEMM_1 output is multiplied by `v_descale`.

**MX (Microscaling) Mode**:
- Q/K/V each have an independent e8m0 scale tensor
- Scale granularity is `kQKScaleGranularity = 32`, i.e., one scale per 32 elements
- The P matrix also needs MX quantization before participating in GEMM_1 (via `cast_tile_mx` conversion)

**FAST_EXP2 + FP8**: When using FP8 block scale, a shift is added in softmax to preserve numerical precision:

```cpp
// OCP FP8: shift = 8.0, FNUZ FP8: shift = 7.0
row_max -= OCP_FP8_SHIFT;
p_compute(i_j_idx) = exp2(scale_s * s[i_j_idx] - row_max);
```

## Performance Tuning

### Occupancy Control

`kBlockPerCu` controls the number of workgroups per CU (compile-time constant or auto-derived):

| hdim | Recommended kBlockPerCu |
|------|--------------------------|
| <= 32 | 2 |
| <= 64 | 3 |
| <= 128 | 2 (reduced to 1 when bias is present) |
| <= 256 | 1 |

### Instruction Scheduling (sched_barrier / sched_group_barrier)

The standard pipeline uses `sched_group_barrier` fine-grained control when `kQKHeaddim == 256`:

```cpp
__builtin_amdgcn_sched_group_barrier(DS_READ, 2, 0); // 2 LDS
__builtin_amdgcn_sched_group_barrier(MFMA, 2, 0); // 2 MFMA
__builtin_amdgcn_sched_group_barrier(DS_READ, 1, 0);
__builtin_amdgcn_sched_group_barrier(MFMA, 2, 0);
__builtin_amdgcn_sched_group_barrier(DS_READ, 1, 0);
__builtin_amdgcn_sched_group_barrier(MFMA, 4, 0);
```

The V3 pipeline goes further by using four-stage `CoreLoopScheduler` to control the interleaving of MFMA/VALU/TRANS/SALU.

### Key Tuning Parameters

| Parameter | Meaning | Typical Values |
|-----------|---------|----------------|
| `kM0` | Q sequence tile | 64, 128 |
| `kN0` | K sequence tile | 64, 128 |
| `kK0` | GEMM_0 unroll | 32 |
| `kK1` | GEMM_1 unroll | 32 |
| `kQKHeaddim` | head dim | 64, 128, 256 |
| `Gemm0BlockWarps` | GEMM_0 warp grid | (4,1,1), (2,2,1) |
| `Gemm1BlockWarps` | GEMM_1 warp grid | (4,1,1), (2,2,1) |
| `IsVLayoutRowMajor` | V layout | true (recommended, required for gfx950) |
| `CK_TILE_FMHA_FWD_FAST_EXP2` | Fast exp2 | 1 (recommended) |

### Early Exit Optimization

In mask scenarios, if the valid K range for the current Q tile is empty, return zero output directly:

```cpp
if constexpr(FmhaMask::IsMasking || kPadSeqLenK) {
    if(num_total_loop <= 0) {
        // store -inf LSE, return zero O_acc
        return o_acc;
    }
}
```

---

## Related Documentation

- [CK GEMM Pipelines](ck-gemm-pipelines.md) — CK general-purpose GEMM pipeline design
- AMD MFMA Instructions — MFMA instruction selection and tuning
- LDS Bank Conflict — LDS swizzle optimization patterns
