# CK Tile Quantized GEMM and MX Format

Composable Kernel (CK) Tile provides a complete quantized GEMM implementation system, covering FP8/FP4 Group Quantization, MX (Microscaling) block-scale format, SmoothQuant, and fused kernels such as fused add+rmsnorm+requantize. All implementations are based on CK Tile's composable pipeline architecture, supporting multiple quantization strategy combinations via C++ template parameterization.

---

## Quantized GEMM Overview

CK Tile's quantized GEMM implementation is located at `include/ck_tile/ops/gemm_quant/`, organized in the standard three-layer structure of kernel / pipeline / block.

### Quantization Type Enumeration

`tile_gemm_quant_traits.hpp` defines five quantization modes:

| QuantType | Meaning | Typical Usage |
|-----------|------|---------|
| `AQuantGrouped` | A matrix grouped quantization only (activation quant) | Dynamic activation quantization |
| `BQuantGrouped` | B matrix grouped quantization only (weight quant) | Weight-only quantization (W8A16, W4A16) |
| `ABQuantGrouped` | Both A and B matrix grouped quantization | FP8 GEMM (W8A8) |
| `RowColQuant` | Row/column mixed quantization | Per-token x per-channel dequant |
| `TensorQuant` | Entire tensor single scale | Per-tensor quantization |

### Quantization Group Size (QuantGroupShape)

Quantization granularity is controlled via `QuantGroupShape<sequence<kM, kN, kK>>`:

```cpp
// 128 K shared scale(per-group along K)
using BQuantGroupSize = QuantGroupShape<sequence<1, 1, 128>>;

// 32 K x 64 N shared scale(block-wise)
using BQuantGroupSize = QuantGroupShape<sequence<1, 64, 32>>;
```

These three dimensions control the quantization group granularity in the M/N/K directions respectively, allowing flexible combinations of different quantization strategies such as per-channel, per-group, and block-wise.

---

## Quantized GEMM Pipeline Variants

### BQuant Pipeline (Weight Quantization)

The `gemm_bquant_pipeline_ag_bg_cr_*.hpp` series implements weight-only quantization:

- **v3**: Complete CompV3 pipeline, supporting async global load + LDS double buffering
- **base**: Basic implementation, suitable for simple tiling scenarios

Data type combination examples:
- A: FP16/BF16 (activation), B: FP8/FP4 (weight), BQ: FP16/FP32 (scale)
- gfx950 supports `pk_fp4_t` packed 4-bit type, capable of leveraging hardware load-with-transpose instructions

### AQuant Pipeline (Activation Quantization)

The `gemm_aquant_pipeline_ag_bg_cr_*.hpp` series:

- Symmetric to BQuant, with quantization scale applied to the A matrix
- Supports `APreshuffleQuant`: pre-shuffle AQ scale to match wave-level distribution

### ABQuant Pipeline (Both-Side Quantization)

The `gemm_abquant_pipeline_ag_bg_cr_*.hpp` series:

- Both A and B have independent group quantization scales
- **eight_waves variant**: 8-wave configuration, suitable for scenarios requiring higher occupancy
- Typically used for FP8 W8A8 GEMM

### Microscale Pipeline

The `gemm_microscale_pipeline_ag_bg_cr_*.hpp` series handles MX format (see below).

### Weight-Preshuffle (WP) Variants

The `gemm_wp_*quant_pipeline_*.hpp` series adds weight preshuffle support on top of the above pipelines:

- Weights are pre-shuffled to `[Nr, Kr, wave_flatten]` layout, eliminating runtime transpose overhead
- Significant performance improvements for FP4/FP8 small-tile scenarios

---

## QuantGemmKernel Core Mechanisms

`gemm_quant_kernel.hpp` is the unified kernel entry point for quantized GEMM, with the following key design aspects:

### Split-K Support

Quantized GEMM fully supports Split-K parallelism, splitting the K dimension across multiple workgroups:

- BQuantGrouped and ABQuantGrouped (non-preshuffle mode) support Split-K
- Automatically computes BQ/AQ K-direction offsets to ensure scale alignment
- Constraint: at least 2 K-tile iterations per batch (required by pipeline prefetch)
- Constraint: KRead must be aligned to `BQuantGroupSize::kK`

### Preshuffle Quant Scale

BQ scale preshuffle (`BPreshuffleQuant`) flow:

1. Group BQ scales by `[N/kN, K/kK]`
2. Flatten into a `[bq_y, N * KPerBlockBQ]` 2D layout
3. Align by block tile + wave tile + warp size
4. Ultimately form an efficient global memory read pattern

### Dequantization Timing

B matrix dequantization timing is controlled via `CastPolicy`:

| CastPolicy | Description | Applicable Scenario |
|------------|------|---------|
| `AfterLDSRead` | Dequantize in registers after LDS read | gfx950 + FP4 (hardware supports load-with-transpose) |
| `BeforeLDSWrite` | Dequantize after VMEM read, before LDS write | Non-gfx950 FP4 (requires manual register transpose) |

## MX (Microscaling) GEMM

The MX format implementation is located at `include/ck_tile/ops/gemm_mx/`, based on `UniversalGemmKernel` extensions.

### MXScalePointer

`scale_pointer.hpp` defines a smart pointer for MX scale, supporting three granularities:

```cpp
// GranularityMN > 0, GranularityK > 0: 2D block scale
MXScalePointer<e8m0_t, 32, 32> // 32x32 block e8m0 scale

// GranularityMN > 0, GranularityK = 0: row/column scale
MXScalePointer<e8m0_t, 1, 0>    // per-row scale

// GranularityMN = -1: none scale(, operator returns 1)
MXScalePointer<e8m0_t, -1>      // disabled
```

### MXGemmKernel

`gemm_mx_kernel.hpp` inherits `UniversalGemmKernel`, with additional handling for:

- **Packed int32 loading of Scale A/B**: Packs the 2M x 2K e8m0_t scale into int32_t, reducing memory transactions
- **XdlPack optimization**: Defaults to `MXdlPack=NXdlPack=KXdlPack=2`, merging scale loads across multiple XDL iterations
- **Ping-pong shared memory**: Allocates separate ping/pong SMEM buffers for the GEMM pipeline
- **Persistent kernel**: Supports persistent launch mode, dynamically computing grid size via `hipOccupancyMaxActiveBlocksPerMultiprocessor`

The scale data type is `e8m0_t` (8-bit exponent-only format), which is the block scale type specified by the MX standard.

### Async Compute Pipeline

`gemm_pipeline_ag_bg_cr_comp_async.hpp` implements the async compute pipeline for MX GEMM, supporting overlap between global loads of A/B matrices and scale loading.

---

## SmoothQuant

`include/ck_tile/ops/smoothquant/` implements SmoothQuant quantization:

### Standard SmoothQuant

Input/output layout:
- Input: `p_x [M, N]` (FP16/BF16) + `p_smscale [1, N]` (FP32 column-wise scale)
- Output: `p_yscale [M, 1]` (FP32 row-wise quant scale) + `p_qy [M, N]` (INT8 quantized result)

Flow: `QY = SaturateCast(X * SmScale / YScale)`

Pipeline variants:
- **one_pass**: Single-pass scan, suitable for small N scenarios
- **two_pass**: Two-pass scan, first pass computes row-wise amax, second pass performs quantization

### MoE SmoothQuant

`moe_smoothquant_kernel.hpp` extends this for MoE scenarios:

- Input adds `p_topk_ids [tokens, topk]`, indicating the expert assigned to each token
- SmoothScale becomes `[experts, hidden_size]` per-expert scale
- Grid configuration: `dim3(topk, ceil(tokens / Block_M), 1)`
- Each block indexes the corresponding expert's smooth scale based on topk_ids

---

## Fused Add + RMSNorm + RDQuant

`include/ck_tile/ops/add_rmsnorm2d_rdquant/` implements a three-in-one fused kernel:

```
X = A + B
Y = RMSNorm(X, gamma, epsilon)
QY = RowwiseDynamicQuant(Y) = SaturateCast(Y / YScale)
```

### Inputs and Outputs

| Tensor | Shape | Type | Description |
|------|-------|------|------|
| `p_a` | [M, N] | FP16/BF16 | Residual input A |
| `p_b` | [M, N] | FP16/BF16 | Residual input B |
| `p_gamma` | [1, N] | FP16/BF16 | RMSNorm gamma |
| `p_x` | [M, N] | FP16/BF16 | Optional output: A+B result (`kSaveX`) |
| `p_yscale` | [M, 1] | FP32 | Output: per-row quant scale |
| `p_qy` | [M, N] | INT8 | Output: quantized result |

### Pipeline Variants

- **one_pass**: Completes add + norm + quant in a single pass, suitable for small N
- **three_pass**: Three-pass scan (compute RMS -> normalize -> quantize), suitable for large N

This fused kernel is critical in LLM inference, combining the transformer block's residual connection + LayerNorm + dynamic quantization into a single kernel launch.

---

## Dequantization Strategy Summary

| Strategy | Scale Shape | QuantType in Code | Applicable Scenarios |
|------|-------------|-------------------|---------|
| Per-tensor | [1] | `TensorQuant` | Coarsest granularity, moderate accuracy |
| Per-token (row) | [M, 1] | `RowColQuant` (AQ) | Dynamic activation quant |
| Per-channel (col) | [1, N] | `RowColQuant` (BQ) | Static weight quant |
| Per-group (K) | [M, K/g] or [K/g, N] | `AQuantGrouped` / `BQuantGrouped` | GPTQ/AWQ style |
| Block-wise | [M/bm, K/bk] | `ABQuantGrouped` | FP8 W8A8 block quant |
| MX block-scale | Defined by GranularityMN/K | MXGemmKernel | OCP MX standard |

## Supported Data Types

| A dtype | B dtype | Scale dtype | Acc dtype | Description |
|---------|---------|------------|-----------|------|
| FP16 | FP8 (E4M3) | FP16/FP32 | FP32 | Weight-only W8A16 |
| FP16 | FP4 (pk_fp4_t) | FP16/FP32 | FP32 | Weight-only W4A16 |
| FP8 | FP8 | FP32 | FP32 | W8A8 |
| BF16 | FP8 | BF16/FP32 | FP32 | Mixed precision |
| FP16/BF16 | INT8 | FP32 | FP32 | SmoothQuant output |

gfx950 (MI355X) specific: `pk_fp4_t` packed 4-bit type, supports hardware load-with-transpose.

---

## Examples

CK Tile provides the following quantization-related examples (`example/ck_tile/` directory):

| Directory | Description |
|------|------|
| `42_mx_gemm/` | MX format GEMM |
| `12_smoothquant/` | SmoothQuant quantization |
| `14_moe_smoothquant/` | MoE SmoothQuant |
| `11_add_rmsnorm2d_rdquant/` | Fused Add + RMSNorm + RDQuant |

---

## Related Documents

- [CK MoE / Norm / Conv Operators](ck-moe-norm-conv.md) -- Scenarios using SmoothQuant in fused MoE
- [CK Dispatcher and Kernel Selection](ck-dispatcher-kernel-selection.md) -- Automatic kernel selection for quantized GEMM
