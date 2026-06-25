# CK Tile MoE / Norm / Conv / Reduce Operations

In addition to GEMM and Attention, Composable Kernel (CK) Tile provides a rich set of commonly used operator implementations for LLM inference and training, including Fused MoE (sorting + GEMM + activation fusion), TopK Softmax, LayerNorm/RMSNorm (with quantization fusion), Grouped Convolution, im2col, Reduce, and Pooling. All operators follow the CK Tile three-layer architecture of kernel / pipeline / block.

---

## Fused MoE GEMM

`include/ck_tile/ops/fused_moe/` implements the complete Mixture-of-Experts inference pipeline, including token sorting and back-to-back dual GEMM fusion.

### Computation Flow

Fused MoE executes a two-stage GEMM + activation function, corresponding to the Gate/Up + Down structure of the MoE FFN:

```
                    N0 (intermediate)            N1 (hidden)
         +------+------+                    +----------+
    A    | Gate |  Up  |              Down  |          |
  [M,K]  +------+------+  --ACT(SiLU)-->   +----------+  --> Out [M, K]
         |      |      |                    |          |
         +------+------+                    +----------+
         [E, 2N, K]                         [E, K, N]
```

### FusedMoeGemmShape

Defines the tile configuration for both stages:

```cpp
FusedMoeGemmShape<
 BlockTile_0, // stage GEMM: sequence<M, N, K>
 WarpPerBlock_0, // stage warp
 WarpTile_0, // stage warp tile
 BlockTile_1, // stage GEMM: sequence<M, N, K>
 WarpPerBlock_1, // stage warp
 WarpTile_1 // stage warp tile
>
```

Constraints: `Block_M0 == Block_M1` (M dimension must be consistent across both stages); `Block_N0 == Block_K1 || Block_N0/2 == Block_K1` (Gate Only or Gate+Up mode).

### Token Sorting (MoE Sorting)

`moe_sorting_kernel.hpp` implements token-to-expert sorting:

**Inputs**:
- `topk_ids [tokens, topk]` -- top-K expert IDs assigned to each token
- `topk_weight [tokens, topk]` -- corresponding routing weights

**Outputs**:
- `sorted_token_ids [max_num_tokens_padded]` -- token IDs sorted by expert
- `sorted_weight [max_num_tokens_padded]` -- corresponding sorted weights
- `sorted_expert_ids [num_tiles]` -- expert ID corresponding to each tile
- `num_sorted_tiles [1]` -- total number of valid tiles

Token ID encoding: the lower 24 bits store the token_id, and the upper 8 bits store the topk_id, usedרד for indexing per-expert quantization scales in SmoothQuant scenarios.

The sorting implementation uses `max_num_tokens_padded = topk * num_tokens + num_experts * (block_size - 1)` to ensure that the number of tokens per expert is aligned to block_size (Block_M).

### FusedMoeGemmTraits

```cpp
template <bool IsGateOnly_, // true: Gate, false: Gate+Up
 bool UseSmoothQuant_, // use SmoothQuant
 index_t OAtomic_, // 0: noneatomic, 1: atomic pk f16/bf16, 2: atomic f32
 FusedMoeGemmWeightPermuteEnum PermuteEnum_, // weightmode
 bool PadHiddenSize_, // hidden_size requires padding
          bool PadIntermediateSize_,
 bool PipeInterleave_> // pipeline
```

Weight pre-reordering (`b_nr_kr_waveflatten`): reorders the B matrix from `[E, N, K]` to `[E, Nr, Kr, wave_flatten]`, where `wave_flatten = Warp_N * Warp_K`, eliminating runtime LDS bank conflicts.

### Supported Data Types

| Tensor | Type | Description |
|------|------|------|
| A (activation) | FP16/BF16/FP8/BF8 | Input tokens |
| Gate/Up/Down (weight) | FP16/BF16/FP8/BF8 | Pre-reordered weights |
| AScale | FP32 | Token-wise quantization scale |
| GScale / DScale | FP32 | Expert-wise weight scale |
| YSmoothScale | FP32 | Smooth-quant scale for the second stage input |

---

## TopK Softmax

`include/ck_tile/ops/topk_softmax/` implements TopK + Softmax for the MoE routing layer:### Interface

```cpp
struct TopkSoftmaxHostArgs {
    const void* p_input;    // [num_rows, num_experts], router logits
    void* p_output;         // [num_rows, topk], softmax(top-k logits)
 void* p_indices; // [num_rows, topk], expert indices
 index_t num_rows; // token count
 index_t num_experts; // expert total count
 index_t topk; // top-K
 index_t stride_input; // inputrow
 index_t stride_output; // outputrow
};
```

### Implementation Strategy

- **Warp-per-row**: Each warp processes one or more rows (`RowsPerWarp` configuration)
- **Persistent launch**: Uses a persistent kernel when `LaunchType > 0`, with grid size `num_cu * LaunchType`
- Internally implements TopK selection and Softmax normalization via warp-level reduce

---

## LayerNorm2d

`include/ck_tile/ops/layernorm2d/` implements 2D LayerNorm (normalization along the last dimension).

### Features

| Feature | Enum | Description |
|------|------|------|
| Fused Add | `PRE_ADD_STORE` / `PRE_ADD` | Fuses residual addition before normalization (optionally saves the addition result) |
| Fused Quant | `SMOOTH_DYNAMIC_QUANT` / `DYNAMIC_QUANT` | Fuses SmoothQuant or dynamic quantization after normalization |
| X Bias | `ADD_BIAS` | Fuses bias addition |
| Welford | `kWelford` | Uses Welford's algorithm Tess for variance computation (more numerically stable) |
| Two Pass | `kTwoPass` | Uses two-pass scan for large N |
| FastFDiv | `kFastFDiv` | Fast floating-point division |
| Save Mean/InvStd | `kSaveMeanInvStd` | Saves mean and inverse standard deviation (for backpropagation) |

### Inputs and Outputs

```cpp
struct Layernorm2dFwdHostArgs {
 const void* p_x; // [M, N] input
 const void* p_x_residual; // [M, N] input(optional)
 const void* p_sm_scale; // [1, N] smooth scale(optional)
    const void* p_x_bias;     // [1, N] bias
    const void* p_gamma;      // [1, N] gamma
    const void* p_beta;       // [1, N] beta
 void* p_y; // [M, N] output
 void* p_y_residual; // [M, N] output(optional)
 void* p_y_scale; // [M, 1] quantization scale(optional)
 void* p_mean; // [M, 1] (optional)
 void* p_invStd; // [M, 1] standard(optional)
    float epsilon;
};
```

---

## RMSNorm2d

`include/ck_tile/ops/rmsnorm2d/` implements RMSNorm (Root Mean Square Normalization).

### Features

| Feature | Enum | Description |
|------|------|------|
| Fused Add | `PRE_ADD_STORE` / `PRE_ADD` | Same as LayerNorm |
| Fused Quant | `SMOOTH_DYNAMIC_QUANT` / `DYNAMIC_QUANT` | Same as LayerNorm |
| Model Sensitive | `T5_MODEL_LIKE` | T5-style RMSNorm (different value distribution handling) |
| Save InvRms | `kSaveInvRms` | Saves inverse RMS value |
| Save Unquant | `kSaveUnquant` | Saves the result before quantization |
| Two Pass | `kTwoPass` | Two-pass scan for large N |

### Pipeline Variants

- **one_pass**: Single-pass scan, used when N is small
- **two_pass**: First pass computes RMS, second pass normalizes
- **model_sensitive_pass**: Optimized computation path for specific models (e.g., T5)

### Differences from LayerNorm

RMSNorm does not compute the mean, only RMS: `y = x * gamma / sqrt(mean(x^2) + epsilon)`. Therefore:
- No beta parameter
- No mean output
- Lower computational cost, faster inference
- Outputs `p_invRms` instead of `p_invStd`

---

## Grouped Convolution

`include/ck_tile/ops/grouped_convolution/` implements forward/backward computation of grouped convolution.

### Supported Directions

| Kernel | File | Description |
|--------|------|------|
| Forward | `grouped_convolution_forward_kernel.hpp` | Forward convolution |
| Backward Data | `grouped_convolution_backward_data_kernel.hpp` | Data gradient |
| Backward Weight | `grouped_convolution_backward_weight_kernel.hpp` | Weight gradient |

### Implementation

All convolutions are implemented via **conv-to-GEMM transformation**:

```cpp
// forward: TransformConvFwdToGemm
// backwarddata: TransformConvBwdDataToGemm
// backwardweight: TransformConvBwdWeightToGemm
```These transformations map the convolution problem to an equivalent GEMM problem, reusing CK Tile's high-performance GEMM pipeline.

### Layout Support

- Input: `NWGC` (N-Width-Group-Channel)
- Weight: `GKXC` (Group-KernelSize-X-Channel)
- Output: `NWGK` (N-Width-Group-Kernel)

Supports 1D/2D/3D spatial dimensions, controlled via the `NDimSpatial` template parameter.

### Split-K and NumGroupsToMerge

- `k_batch`: Split-K parallelism
- `NumGroupsToMerge`: Merge multiple convolution groups into the same GEMM (not supported when combined with ExplicitGemm)

---

## Image to Column (im2col)

`include/ck_tile/ops/image_to_column/` implements the im2col transformation:

```cpp
// [N, H, W, G, C] [N*Ho*Wo, G, Fh*Fw*C] matrix
// stride, dilation, padding
struct Kargs {
 const void* p_in; // input
 void* p_out; // matrix
    long_index_t G, N, C;   // Group, Batch, Channel
    array<long_index_t, 2> input_spatial_lengths;
    array<long_index_t, 2> filter_spatial_lengths;
    array<long_index_t, 2> output_spatial_lengths;
    array<long_index_t, 2> conv_filter_strides;
    array<long_index_t, 2> conv_filter_dilations;
    array<long_index_t, 2> input_left_pads;
    array<long_index_t, 2> input_right_pads;
};
```

Currently only `NDimSpatial == 2` (2D convolution) is supported. The im2col output can be used directly as the input matrix for GEMM.

---

## Reduce2d

`include/ck_tile/ops/reduce/` implements a generic 2D reduction operation.

### Multi-level Reduction

```
Global Memory → Thread-level Reduce → Warp-level Reduce → Block-level Reduce → Output
```

### Variants

| Kernel | Description |
|--------|-------------|
| `reduce2d_kernel.hpp` | Basic 2D reduction |
| `multi_reduce2d_kernel.hpp` | Multi-target reduction (computes sum, max, etc. simultaneously) |
| `multi_reduce2d_multiblock_kernel.hpp` | Reduction across multiple blocks |
| `multi_reduce2d_threadwise_kernel.hpp` | Thread-level reduction |

### Vectorization Optimization

```cpp
// automaticcomputevectorloadsize
constexpr index_t memory_vector_size = 16 / sizeof(XDataType);
constexpr index_t stride_based_vector_size =
    is_innermost_contiguous ? min(memory_vector_size, thread_tile_vector_size) : 1;
```

For cases where the innermost dimension is contiguous (stride=1), vectorized loads are used; otherwise, it falls back to scalar loads.

Supports input tensors of arbitrary rank, with dimensions to preserve and reduce specified via the `KeptDim` and `ReduceDims` template parameters.

---

## Pooling

`include/ck_tile/ops/pooling/` implements generic pooling operations (Max / Average).

### Interface

```cpp
template <typename TensorShape, typename WindowShape>
struct PoolHostArgs {
    const void* input_ptr;
    void* output_ptr;
 void* output_index_ptr; // Max pooling index output
    TensorShape input_shape;
    TensorShape output_shape;
    TensorShape input_strides;
    TensorShape output_strides;
    WindowShape window_lengths;
    WindowShape window_strides;
    WindowShape window_dilations;
    WindowShape input_left_pads;
    WindowShape input_right_pads;
};
```

Supports arbitrary window sizes, strides, dilations, and padding, consistent with PyTorch's pooling semantics.

---

## Activation Functions

`include/ck_tile/ops/elementwise/unary_element_wise_operation.hpp` defines composable activation functions:

- **PassThrough**: Identity transform
- **Relu**: max(0, x)
- **Gelu**: Gaussian Error Linear Unit
- **FastGelu**: Fast approximate GeLU
- **SiLU / Swish**: x * sigmoid(x), the standard activation for the Gate branch in MoE

These activation functions can be used as GEMM epilogue or MoE intermediate activations, composed into the pipeline via template parameters.

---

## Example Code

CK Tile provides the following examples (`example/ck_tile/` directory):

| Directory | Description |
|-----------|-------------|
| `15_fused_moe/` | Fused MoE (sorting + B2B GEMM + activation) |
| `13_moe_sorting/` | MoE token sorting |
| `09_topk_softmax/` | TopK Softmax router |
| `02_layernorm2d/` | LayerNorm 2D |
| `10_rmsnorm2d/` | RMSNorm 2D |
| `20_grouped_convolution/` | Grouped convolution |
| `04_img2col/` | Image to Column |
| `05_reduce/` | 2D Reduction |
| `36_pooling/` | Pooling |## Related Documentation

- [CK Quantization GEMM and MX Format](ck-quantization-mx.md) -- Detailed implementation of SmoothQuant and Quantized GEMM
- [CK Dispatcher and Kernel Selection](ck-dispatcher-kernel-selection.md) -- Automatic kernel selection mechanism for operators
