# NVFP4/MXFP4 Numeric System: PTX, CUTLASS, Triton, and Quantization

Comprehensive analysis of the NVFP4 and MXFP4 numeric systems — covering the mathematical foundations, PTX instruction paths, CUTLASS/Triton operator implementations, and quantization recipes for inference and training.


**Last updated**: 2026-06-30

---

## 1. Background: Why Microscaling FP4 Exists

### 1.1 Why FP4 Instead of INT4

INT8 quantization is mature, but pushing to 4-bit presents two paths: INT4 or FP4. This choice determines whether Microscaling is needed.

**INT4's fundamental problem: linear grid cannot match exponential distributions.** Neural network weights and activations typically follow approximately normal or long-tailed distributions. INT4 uses 16 equally-spaced discrete levels, meaning small-value and large-value intervals get the same quantization step. To cover outliers, the scale must be large, causing relative error to explode for the majority of values.

**FP4's natural advantage: non-uniform grid matches distribution characteristics.** FP4 E2M1 produces values {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6} — a non-uniform grid that is dense near zero and sparse at larger magnitudes, naturally matching neural network distributions.

However, FP4's dynamic range is narrow (only ±6), requiring a per-block scale to cover actual value ranges. Per-tensor or per-channel scaling suffers the same outlier problem as INT4. Microscaling divides tensors into small blocks (e.g., 16 or 32 elements), each with an independent scale, isolating outlier influence.

### 1.2 Microscaling: The Per-Block Scaling Compromise

A Microscaling block is defined by a triplet **(G, E, S)**:
- **G**: Block/group size (e.g., 16 or 32)
- **E**: Element type (e.g., FP4 E2M1, FP6 E3M2)
- **S**: Scale factor type (e.g., UE8M0, UE4M3)

Decoding: `x_i ≈ S · q_i`

Key findings from the Microscaling Data Formats paper:
- **MXFP8**: Near-lossless inference from FP32/BF16; training approaches FP32 with QAT
- **MXFP6**: Approaches FP32 accuracy under simple QAT/PTQ; first sub-8-bit training demonstration
- **MXFP4**: Small accuracy drop in weight quantization training; potential core format for extreme low-precision

### 1.3 MXFP vs NVFP: Two Paths from Standard to Implementation

**OCP MX (Cross-Ecosystem Standard):** Defines MXFP8/MXFP6/MXFP4 as portable logical formats. Hardware acceleration is vendor-specific.

**NVIDIA NVFP4 (Blackwell Implementation):** Deeply integrated into the NVIDIA stack — PTX ISA (`kind::mxf4nvf4`), CUTLASS (`nv_float4_t`), Triton, with native Tensor Core support. Uses two-level scaling: `x_i ≈ s_tensor(FP32) · s_block(E4M3) · q_i`.

**Comparison:**

| Property | MXFP4 | NVFP4 |
|----------|--------|-------|
| Block size (G) | 32 | 16 |
| Element (E) | FP4 E2M1 | FP4 E2M1 |
| Block scale (S) | UE8M0 (power-of-2) | UE4M3 + tensor-level FP32 |
| Storage overhead | ~4.25 bit/element | ~4.5 bit/element |
| Throughput vs FP8 | 2-3× (Blackwell) | 2-3× (Blackwell) |

---

## 2. Numeric Fundamentals: Structure and Error Intuition

### 2.1 Common Foundation: FP4 E2M1

Both formats use FP4 E2M1: 1-bit sign + 2-bit exponent + 1-bit mantissa. Positive values: {0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}. Full set: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}.

This is a highly non-uniform grid — dense in [0.5, 2] and coarse beyond. Without scaling, FP4 E2M1 is too crude and overflow-prone for large models.

### 2.2 MXFP4: Large Block + Power-of-2 Scale

- **Triplet**: (G=32, E=FP4 E2M1, S=UE8M0)
- **Bit budget**: (8 + 32×4) / 32 ≈ 4.25 bit/element
- **Scale quantization**: UE8M0 represents only powers of 2, causing coarse approximation when the ideal scale falls between two powers
- **Outlier sensitivity**: G=32 means a single outlier can hijack the entire block's scale

### 2.3 NVFP4: Small Block + Fractional Scale + Two-Level FP32

- **Triplet**: (G=16, E=FP4 E2M1, S_block=UE4M3, S_tensor=FP32)
- **Bit budget**: (8 + 16×4) / 16 = 4.5 bit/element
- **Scale quantization**: UE4M3 has 3-bit mantissa — MSE ~0.08 vs UE8M0's ~0.72 (order of magnitude improvement)
- **Outlier robustness**: G=16 confines outlier impact to a smaller region
- **Two-level scaling**: FP32 tensor-level scale normalizes the overall dynamic range, preventing block scales from saturating

### 2.4 Error Sources

Total error decomposes into:
1. **Scale quantization error**: |S_real - S| propagates to all elements in the block
2. **Element quantization error**: |x/S - q| depends on FP4 grid and local scale

MXFP4: Scale error dominates (UE8M0 coarseness). NVFP4: Scale error significantly reduced (UE4M3 precision), element error becomes the limiting factor.

---

## 3. PTX Instruction Layer: MXFP4/NVFP4 on Blackwell

### 3.1 Two Tensor Core Paths

| Feature | mma.sync (SM120) | tcgen05.mma (SM100) |
|---------|-------------------|---------------------|
| Data source | Shared Memory | Tensor Memory |
| Scale source | Register selectors | TMEM addresses |
| Register pressure | Higher | Lower |
| Throughput | Standard | Higher |

### 3.2 FP4 Data Packing in PTX

FP4 elements are packed into containers:
- `e2m1x2`: Two FP4 packed into `.b8`
- `e2m1x4`: Four FP4 packed into `.b16`

**Critical difference**: `kind::mxf4` and `kind::mxf4nvf4` use tight packing (no padding). `kind::mxf8f6f4` pads FP4 within the byte.

Scale types:
- **ue8m0**: 8-bit unsigned exponent-only (powers of 2), NaN=0xff. Used by MXFP4.
- **ue4m3**: 7 effective bits (4-bit exponent + 3-bit mantissa), unsigned, with fractional precision. NaN=0x7f. Used by NVFP4.

### 3.3 The `kind` Family: Three Microscaling Semantics

| kind | Element type | Scale type | Default scale_vec | Transpose |
|------|-------------|-----------|-------------------|-----------|
| mxf8f6f4 | Mixed 8/6/4-bit | ue8m0 | 1X | Supported |
| mxf4 | e2m1 | ue8m0 | 2X | Not supported |
| mxf4nvf4 | e2m1 | ue8m0/ue4m3 | Must specify | Not supported |

### 3.4 Block Scaling Math

The `.block_scale` modifier enables: `D = (A · scale_A) · (B · scale_B) + C`

Scale broadcast along K dimension is controlled by `.scale_vec_size`:

| scale_vec_size | scale_A shape | scale_B shape |
|----------------|--------------|--------------|
| 1X | M × 1 | 1 × N |
| 2X | M × 2 | 2 × N |
| 4X | M × 4 | 4 × N |

### 3.5 PTX Instruction Examples

**MXFP4 (mma.sync):**
```
mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale
  .f32.e2m1.e2m1.f32.ue8m0
  {d}, {a}, {b}, {c}, scale_a_data, scale_b_data;
```

**NVFP4 (mma.sync):**
```
mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X
  .f32.e2m1.e2m1.f32.ue4m3
  {d}, {a}, {b}, {c}, scale_a_data, scale_b_data;
```

**MXFP4 (tcgen05, SM100):**
```
tcgen05.mma.cta_group::1.kind::mxf4.scale_vectorsize::block32
  [taddr_d], [taddr_a], bdesc, idesc, [tmem_scaleA], [tmem_scaleB], p;
```

### 3.6 Common Pitfalls

- `mxf4`/`mxf4nvf4` do NOT support Transpose — handle layout before issuing instructions
- `mxf4nvf4` has NO default `scale_vec_size` — must specify explicitly
- Packing differs between `mxf4`/`mxf4nvf4` (tight) and `mxf8f6f4` (padded)
- Scale Data ID for `mxf4`/`mxf4nvf4`: only 0 or 2 (not 0-3)

```bash
# Quick check for FP4 instructions in binary
cuobjdump --dump-sass your_binary | rg -n "mxf4|mxf4nvf4|tcgen05|block_scale"
```

---

## 4. CUTLASS / Triton Operator Implementation

### 4.1 CUTLASS Three-Layer Type System

**Layer 1 — Element types:**
```cpp
float_e2m1_t  // FP4 E2M1
float_e3m2_t  // FP6 E3M2
float_e4m3_t  // FP8 E4M3
```

**Layer 2 — Wrapper types** (bind element + scale):
```cpp
mx_float4_t<float_e2m1_t>   // MXFP4: FP4 + UE8M0
nv_float4_t<float_e2m1_t>   // NVFP4: FP4 + UE4M3
```

**Layer 3 — Instruction kind** (auto-selected from wrapper):

| Wrapper | Scale type | PTX kind |
|---------|-----------|----------|
| mx_float4_t | UE8M0 | mxf4 |
| nv_float4_t | UE4M3 | mxf4nvf4 |

Switching from MXFP4 to NVFP4 requires changing only one template parameter.

### 4.2 Scale Layout: K-Major + 128-Row Blocking

Scale factors are organized as K-major with 128-row blocks (Blk_MN=128, Blk_SF=4), each basic block occupying 512 bytes to match TMEM load alignment.

### 4.3 Architecture-Specific Collective Files

- **SM100** (data center): `sm100_blockscaled_mma_warpspecialized.hpp` — tcgen05.mma
- **SM120** (GeForce): `sm120_blockscaled_mma_tma.hpp` (dense) / `sm120_blockscaled_sparse_mma_tma.hpp` (sparse)

**Common errors:**
- Scale type mismatch between A and B (must be identical)
- VS/scale type mismatch (NVFP4 sparse requires VS=32 + UE4M3)
- Wrong architecture (SM100 cannot use mma.sync-style kinds)

### 4.4 Triton: `tl.dot_scaled`

```python
C = tl.dot_scaled(A, B, scale_a, scale_b, acc=acc)
```

Scale tensors are 2D: `scale_a: [M, K // VEC_SIZE]`, `scale_b: [N, K // VEC_SIZE]`. VEC_SIZE is 16 for NVFP4, 32 for MXFP4.

In global memory, scales use a 5D layout matching TMEM tile access patterns. The kernel reshapes 5D → 2D before feeding to `tl.dot_scaled`.

FP4 packing: 2 elements per byte, requiring physical stride = logical stride / 2.

---

## 5. Quantization Recipes: PTQ, Pretraining, FQT

### 5.1 Inference PTQ: MXINT4-128

Per-block INT4 (G=128) isolates outliers locally. Standard PTQ methods adapt directly:
- **SmoothQuant/AWQ**: Channel-level rescaling, then block-wise quantization
- **GPTQ**: Modified for block-internal updates with Hessian-guided error redistribution

Critical: Apply smoothing BEFORE GPTQ (not after) — GPTQ's Hessian assumes smooth distributions.

### 5.2 NVFP4 Inference: Two-Level Scale Pipeline

Two-level scale structure:
- **Tensor-level FP32**: Determined during calibration (~20 representative samples)
- **Block-level UE4M3**: Computed dynamically per 16-element block at runtime

Sensitive layers (embedding, lm_head, norms) remain at higher precision.

### 5.3 MXFP8 Pretraining

Key findings:
- **Unified E4M3** for weights, activations, AND gradients outperforms mixed E4M3/E5M2
- **Scale-aware rounding** (not naive RTN) prevents systematic bias accumulation
- Result: 8B model / 15T tokens, MXFP8 vs BF16 difference < 0.5%

### 5.4 NVFP4 Pretraining: Four-Component Recipe

1. **Random Hadamard Transform (RHT)**: Reshapes distributions to near-Gaussian, reducing outliers
2. **2D Quantization**: Same quantization strategy for forward and backward paths
3. **Stochastic Rounding**: Unbiased in expectation, prevents long-term training bias
4. **Precision Preservation**: Sensitive layers (embeddings, output head, norms) at higher precision

Result: 12B model / 10T tokens, NVFP4 vs FP8 difference < 1%.

### 5.5 Full-Precision FP4 Training (FQT)

All weights, activations, and gradients at FP4 — the most aggressive approach.

Key techniques:
- **Split rounding**: RTN for forward (stable loss), stochastic for backward (unbiased gradients)
- **OsciReset**: Detects weight oscillation between quantization buckets; periodically realigns
- **OutControl**: Identifies outlier activation channels; handles with higher precision or independent scales
- **Unbiased double-block quantization**: Designs quantization rules so errors cancel in expectation

Gradient norm threshold: when `gradient_norm < √3 × quantization_noise`, switch back to higher precision.

### 5.6 Software Simulation with `microxcaling`

```python
from microxcaling import MxSpecs
specs = MxSpecs(
    w_elem_format='fp4_e2m1',
    a_elem_format='fp4_e2m1',
    block_size=32,  # MXFP4=32, NVFP4-style=16
    scale_bits=8,
    round='nearest',
)
```

Note: microxcaling uses E8M0 scales — not equivalent to NVFP4's E4M3. Useful for trend analysis only.

---

## 6. Summary

MXFP4 and NVFP4 mark a turning point: FP4 enters the era of complete ecosystems with formats, instructions, and recipes.

- **MXFP4**: Cross-ecosystem standard emphasizing portability and unified semantics
- **NVFP4**: Hardware-coupled implementation optimized for Blackwell, deeply integrated across PTX/CUTLASS/Triton

FP4 will likely become the starting point for subsequent ultra-low-precision work — not the endpoint.


## Related

- [Comprehensive Guide to NVIDIA Blackwell Architecture](blackwell-architecture-comprehensive-guide.md)
- [GPGPU Architecture: Blackwell Instruction Analysis](blackwell-architecture-instruction-analysis.md)
- [Blackwell GPGPU Architecture New Features Overview](blackwell-gpgpu-new-features-overview.md)
- [NVIDIA Blackwell Tensor Core Analysis (Part 2): B300](blackwell-tensor-core-analysis-b300.md)
- [NVIDIA Blackwell Tensor Core Analysis (Part 1)](blackwell-tensor-core-analysis-part1.md)
- [PTX Programming Model and Basics](../../common/ptx/ptx-programming-model.md)
- [PTX Core Instruction Set](../../common/ptx/ptx-instruction-set.md)
- [CUTLASS/CuTe Core Concepts and Layout Algebra](../../common/cutedsl/cutlass-cute-fundamentals.md)
