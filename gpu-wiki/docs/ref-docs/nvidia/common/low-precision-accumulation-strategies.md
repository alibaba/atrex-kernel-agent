# Low-Precision Accumulation Strategies

Precision issues and solutions for accumulation in low-precision (FP8/FP4) matrix multiplication: periodic accumulator promotion, chunked quantization, error analysis, and hardware support.

---

## 1. Precision Issues in Low-Precision Arithmetic

### 1.1 Limited Representational Capability of FP8

The FP8 format has two main variants:

| Format | Exponent Bits | Mantissa Bits | Significant Decimal Digits | Dynamic Range |
|--------|---------------|---------------|---------------------------|---------------|
| E4M3 | 4 | 3 | ~1.0 | ±240 |
| E5M2 | 5 | 2 | ~0.6 | ±57344 |
| FP16 | 5 | 10 | ~3.3 | ±65504 |
| BF16 | 8 | 7 | ~2.4 | ±3.4e38 |
| FP32 | 8 | 23 | ~7.2 | ±3.4e38 |

E4M3 has only 3 mantissa bitsSPAN and can accurately represent only about 240 discrete values (including NaN and special values). When two E4M3 numbers are multiplied, the effective precision of the result is only about 6 mantissa bits (the product of two 3-bit mantissas), requiring at least FP16 for lossless storage.

### 1.2 Sources of Accumulation Error

The core of GEMM is the dot product: `C[i][j] = sum(A[i][k] * B[k][j] for k in range(K))`. When K is large:

```
 1 : accum = a0*b0 # = eps1
 2 : accum = (a0*b0 + eps1) + a1*b1 + eps2 # = eps1 + eps2
 N : accum = true_sum + sum(eps_i) # = O(N * ulp)
```

**Problem 1: Rounding error accumulation**. Each FP16 addition introduces up to 1 ULP (Unit in the Last Place) of rounding error. After N accumulations, worst-case error = O(N * 2^(-p)), where p is the number of mantissa bits.

**Problem 2: Catastrophic cancellation**. When a large number in the accumulator is added to a small number, the low-order bits of the small number are discarded. If accum is already 1000.0 (FP16) and a product result of 0.001 is added, the 0.001 is directly rounded away and lost.

**Problem 3: Overflow**. The maximum value of FP16 is 65504. If K = 4096 and each product averages 16.0, the accumulated sum reaches 65536 — overflowing to Inf.

### 1.3 Performance Trade-offs of Accumulator Precision

| Accumulator Type | Precision | Speed | Register Usage |
|------------------|-----------|-------|----------------|
| FP32 accumulator | High (~7.2 decimal digits) | Baseline | 32 bits per element |
| FP16 accumulator | Low (~3.3 decimal digits) | ~2x on some hardware | 16 bits per element |

In Hopper (SM90) WGMMA instructions, one can choose between FP16 or FP32 accumulators. The FP16 accumulator corresponds to the `.scale_d = 0` (clear and re-accumulate) mode. Combined with periodic promotion, it can achieve near-FP32 precision at near-FP16 speed.

---

## 2. Periodic Accumulator Promotion

This is the most critical precision recovery technique in low-precision GEMM.

### 2.1 Basic Principle

The core idea is: instead of using a single FP32 accumulator to run through the entire K dimension, maintain two accumulators:

```
TC_accum: tensorcore( FP16, )
main_accum: register FP32 main(accuracyhigh)

 N MMA :
 1. main_accum += TC_accum ( FP32)
 2. TC_accum = 0 (zero out, )
```

In this way, TC_accum only accumulates N times per "window," with a limited dynamic range, so FP16 precision is sufficient. Long-term accumulation is handled by the FP32 main_accum.

### 2.2 Illustration

```
K dimension (e.g., K=4096, each MMA processes k=16)
|<-- 256 MMA operations ------------------------------------------------->|

Without promotion (pure FP32 accumulator, precise but possibly slow):
[--- FP32 accum accumulates 256 MMA operations -------------------->]

Without promotion (pure FP16 accumulator, fast but inaccurate):
[--- FP16 accum accumulates 256 MMA operations -- overflow/precision disaster --X          ]

Using promotion (interval=4):
[FP16x4][FP16x4][FP16x4][FP16x4]...[FP16x4][residual]
   |       |       |       |            |      |
   +--FP32-+--FP32-+--FP32-+---...-----+--FP32+-> Final result

Each [FP16x4] segment: 4 MMAs accumulated to FP16
Arrow: FP16 result promoted (added) to FP32 main accumulator, then FP16 cleared
```

### 2.3 Promote vs. Scale-Promote

Two promotion modes, depending on whether dequantization is needed:
**Promote (FADD)**: Simple addition
```python
# Pseudocode
main_accum[i] += TC_accum[i]    # FADD instruction
TC_accum[i] = 0
```

Applicable when: inputs are already floating-point (FP8/FP16/BF16), with no additional scaling required.

**Scale-Promote (FFMA)**: Multiply-add (fused dequantization)
```python
# Scalar scaling
main_accum[i] += TC_accum[i] * scale       # FFMA instruction
TC_accum[i] = 0

# Per-element scaling (per-group quantization)
main_accum[i] += TC_accum[i] * scaleA[i] * scaleB[i]
TC_accum[i] = 0
```

Applicable when: inputs are quantized integers (INT4/INT8) that need to be multiplied by a scaling factor to recover floating-point values. Scale-Promote fuses dequantization and promotion into a single FFMA instruction, avoiding a separate dequantization step.

### 2.4 Choosing the Promotion Interval

Promotion interval = the number of MMA operations executed between two promotions.

**Constraints**:
- The interval must be an integer multiple of the number of MMAs per mainloop iteration (otherwise it cannot align to loop boundaries)
- Larger intervals reduce promotion overhead (promotion itself requires extra FADD/FFMA instructions)
- Smaller intervals reduce the dynamic range seen by the FP16 accumulator, yielding better precision**Typical values**:

| Scenario | Recommended Interval | Rationale |
|------|---------|------|
| FP8 × FP8 → FP16 accum | 4 | Accumulation of 4 MMAs will not overflow FP16 |
| FP8 × FP8 → FP32 accum | Not needed | FP32 accumulation is already precise enough |
| INT4 × INT4 → FP16 accum | 2–4 | INT4 product range is small, but accumulation grows quickly |
| FP4 × FP4 → FP16 accum | 4–8 | FP4 value range is extremely small |

**Origin of the default interval = 4**: For FP8 E4M3, a single MMA (16×8×16 shape) produces 128 products and accumulates them into 16×8 = 128 outputs. Each output element accumulates 16 FP8 products. 4 MMAs accumulate 64 products. The maximum FP8 product is approximately 240×240 = 57600, and the sum of 64 such products is ~3.7M, well within the FP32 range, but FP16 may overflow. Therefore, when interval = 4, an FP32 TC accumulator or sufficiently small input values are required.

### 2.5 Residual Handling

When the K dimension is not divisible by the promotion interval, the number of MMAs in the last window is less than the interval. Residuals must be handled explicitly:

```python
# Main loop ends
if mma_count > 0:    # Remaining un-promoted data
    main_accum += TC_accum   # Final promotion
```

In production code, use `__shfl_sync` to ensure that all threads within a warp consistently determine whether promotion is needed, avoiding branch divergence.

---

## 3. When FP16 Accumulation Is Safe

Not all scenarios require FP32 accumulation or promotion techniques. Direct FP16 accumulation is acceptable when the following conditions are met:

### 3.1 Safety Checklist

| Condition | Reason |
|------|------|
| Short K dimension (< 256) | Few accumulation steps; error does not accumulate significantly |
| Inputs pre-scaled to [-1, 1] | Products do not exceed 1.0; accumulation sum grows slowly |
| Inference (not training) | Inference tolerates more precision loss; no gradient backpropagation needed |
| Application tolerates ~1% error | Classification tasks, coarse-grained inference |

### 3.2 Unsafe Scenarios

- **Gradient accumulation during training**: Gradient values span a wide range (up to 6 orders of magnitude); FP16 accumulation causes small gradients to be swallowed, preventing convergence
- **Large GEMM with K > 4096**: Extremely high risk of FP16 overflow
- **Loss scaling in mixed-precision training**: After loss scaling, gradient values are already large; further FP16 accumulation will overflow
- **Attention score accumulation**: Softmax outputs are in (0,1), but K (sequence length) can exceed 128K+

### 3.3 Rule of Thumb

```
if K * max(|A|) * max(|B|) > 32000:    # Approaching half of FP16 max value 65504
    Use FP32 accumulation or promotion
else:
    FP16 accumulation may be safe (but verification recommended)
```

---

## 4. Block-Scaled Quantization

### 4.1 Scale Factor Granularity

The core problem of quantization: how to represent original floating-point values with lower-bit (FP8/FP4/INT4) formats. The answer is scale factors.

```
quantization: q = round(x / scale)
quantization: x_approx = q * scale
```

The granularity of scale factors directly affects precision and storage overhead:

```
Per-Tensor          Per-Channel          Per-Group           Per-Element
                    (along K axis)       (32-128 elements shared)
    1 scale         M or N scales       M*N / group_size     M*N scales
                                         scales

 Precision:  Lowest          Lower                 Higher                 Highest
 Overhead:   Minimum         O(M+N)               O(M*N/G)              O(M*N)
 Speed:      Fastest         Fast                  Slower                 Slowest
```

The most commonly used quantization in inference is **Per-Group** quantization, with group_size typically being 32, 64, or 128.

### 4.2 Dequantization Timing

Performing dequantization (`x = q * scale`) at different points in the GEMM pipeline involves different trade-offs:

**Before MMA (dequantize before MMA)**:
```
load Q_A, Q_B, scale_A, scale_B
A = Q_A * scale_A    # Dequantize in shared memory or registers
B = Q_B * scale_B
C += A * B            # MMA uses high-precision inputs
```
- Pros: MMA sees floating-point values, accumulation precision is high
- Cons: Dequantized data becomes larger (FP16/BF16), doubling shared memory consumption

**After MMA (dequantize after MMA, i.e., Scale-Promote)**:
```
load Q_A, Q_B
TC_accum += Q_A * Q_B   # MMA directly uses quantized values
# Every N MMAs:
main_accum += TC_accum * scale_A * scale_B  # Dequantization fused into promotion
TC_accum = 0
```
- Pros: Shared memory stores only quantized values (saving 2–4×), MMA throughput is highest
- Cons: Requires scale-promote technique; precision of partial sums within the interval is limited

**In Epilogue (dequantize in the epilogue)**:
```
# Entire K loop uses quantized values
C_raw = sum(Q_A * Q_B)
# Final single scaling
C = C_raw * scale_overall
```
- Applicable only to per-tensor quantization; not available for per-group quantization

### 4.3 Microscaling (MX) Format

MX is a block-scaled quantization format standardized by OCP (Open Compute Project):

```
MX Block (e.g., 32 elements):
+--------+-------+-------+-----+-------+
| Shared | elem0 | elem1 | ... | elem31|
| Exponent| Mantissa| Mantissa|     | Mantissa|
| (8-bit)| (p-bit)| (p-bit)|     | (p-bit)|
+--------+-------+-------+-----+-------+

Shared exponent = max(exponent(elem_i)) for entire block
Each element stores only: sign + mantissa (exponent derived from shared exponent)
```MX format variant comparison:

| Format | Element Bit Width | Block Size | Shared Exponent | Total Effective Bits |
|------|---------|-----------|---------|---------|
| MXFP8 (E5M2) | 8 | 32 | 8-bit E8M0 | 2 mantissa + shared exp |
| MXFP8 (E4M3) | 8 | 32 | 8-bit E8M0 | 3 mantissa + shared exp |
| MXFP4 (E2M1) | 4 | 32 | 8-bit E8M0 | 1 mantissa + shared exp |
| NVFP4 (E2M1) | 4 | 16 | 8-bit UE4M3 | 1 mantissa + shared exp |

**Key differences between NVFP4 and MXFP4**:
- NVFP4 uses 16-element blocks (MXFP4 uses 32), providing finer granularity and higher precision
- NVFP4's scale factor is UE4M3 (unsigned FP8), while MXFP4 uses E8M0 (exponent only, no mantissa)
- NVFP4 is NVIDIA Blackwell's native format; MXFP4/MXFP8 are industry standards

### 4.4 Hardware Native Support

Blackwell (SM100) tcgen05 MMA instructions directly support block-scaled operations:
```
tcgen05.mma  ... .scale_a .scale_b   // Hardware automatically reads and applies scale factor

Input: A_quantized (FP4/FP8), B_quantized (FP4/FP8)
      scale_A (per-block), scale_B (per-block)
Output: C (FP32 accumulator) = sum(dequant(A) * dequant(B))
```
Hardware native support means no software-level scale-promote is needed; dequantization is done inside the MMA, with zero additional instruction overhead.

---

## 5. Numerical Error Analysis

### 5.1 Error Bounds

For the accumulation of N FP8 (p mantissa bits) products, using an FP32 accumulator:

```
|| <= N * 2^(-23) * |true_sum| # FP32 ULP
```

Using an FP16 accumulator (no promotion):

```
|| <= N * 2^(-10) * |true_sum| # FP16 ULP , 1000x
```

Using promotion (interval=I, FP16 TC accumulator):

```
|| <= I * 2^(-10) * |partial_sum| # FP16
 + (N/I) * 2^(-23) * |true_sum| # FP32
```

Key insight: promotion reduces N to I (the window size) for the FP16 error, while the FP32 portion's error is inherently small.

### 5.2 Concrete Numerical Example

Scenario: K=4096, FP8 E4M3 inputs, mean ~1.0

```
Without promotion (FP16 accumulator):
  Error upper bound ≈ 4096 * 2^(-10) * 4096 ≈ 16384
  Relative error ≈ 4.0 = 400%  ← Completely unusable

Using promotion (interval=4, FP16 TC accumulator):
  FP16 part: 4 * 2^(-10) * 4 ≈ 0.016
  FP32 part: 1024 * 2^(-23) * 4096 ≈ 0.0005
  Total relative error ≈ 0.4%  ← Acceptable

Without promotion (FP32 accumulator):
  Error upper bound ≈ 4096 * 2^(-23) * 4096 ≈ 0.002
  Relative error ≈ 0.00005%  ← Most accurate
```

### 5.3 Impact of Condition Number

The numerical stability of matrix multiplication is also affected by the condition number:

```
output ≈ cond(A) * cond(B) *
```

For ill-conditioned matrices (e.g., the pre-softmax attention matrix), accumulation errors are amplified. This is why FlashAttention uses an FP32 accumulator with no compromises.

### 5.4 Kahan Compensated Summation

In theory, Kahan summation can reduce the error of N accumulations from O(N * ulp) to O(ulp):

```python
def kahan_sum(values):
    s = 0.0
    c = 0.0        # Compensation term
    for v in values:
        y = v - c
        t = s + y
        c = (t - s) - y    # Recover discarded low-order bits
        s = t
    return s
```

**But it is impractical on GPUs** because:
- Each accumulation requires 4 instructions (instead of 1), reducing throughput to 1/4
- Increases register pressure (extra storage for the compensation term c)
- Tensor core MMA instructions are atomic and cannot have compensation steps inserted
- Periodic promotion achieves a similar effect with 2 instructions (FADD + zero)

---

## 6. Hardware Architecture Support

### 6.1 FP8/FP4 Capability Comparison Across Architectures

```
 FP8 TC FP4 TC TC Block-Scale Promotion
 optionalaccuracy requires
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SM89 (Ada/L40S) FP32 only (1)
SM90 (Hopper/H100) FP16/FP32 (2)
SM100 (Blackwell) FP32  (3)
AMD CDNA3 (MI300X) FP32 only (1)
AMD CDNA4 (MI355X) FP32 (1)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
(1) FP32 , requires promotion
(2) FP16 requires promotion accuracy
(3) block-scale, MMA quantization
```### 6.2 SM90 (Hopper) WGMMA Accumulator Control

The Hopper WGMMA instruction has a `scale_D` parameter:```
D = A * B + scale_D * C

scale_D = 1 (ScaleOut::One):   D = A*B + C   # Accumulation mode
scale_D = 0 (ScaleOut::Zero):  D = A*B       # Clear and restart
```

This `scale_D = 0` is the hardware foundation of the promotion technique:
1. Normal iteration: `scale_D = 1`, accumulate into the TC's FP16 accumulator
2. Promotion iteration: first add the TC accumulator value to the FP32 main accumulator
3. Next iteration: `scale_D = 0`, the TC accumulator starts from zero
### 6.3 SM100 (Blackwell) tcgen05 Block-Scaled MMA

Blackwell introduces hardware-level block-scaled MMA:

```
tcgen05.mma.kind.block_scale_type  D, A, B, idesc, scaleA, scaleB

block_scale_type:
  .block_scale.scale_vec_1x    # 1 scale per 1 column/row
  .block_scale.scale_vec_2x    # 2 scale vectors
```

The hardware performs dequantization inside the MMA, eliminating the need for software promotion. This is especially critical for FP4—FP4 has only 1 mantissa bit, making pure FP4 accumulation meaningless; a scale factor must be used.

---

## 7. Practical Decision Guide

### 7.1 Rules of Thumb

1. **FP8 GEMM, K < 512**: Use FP32 accumulator directly, no promotion needed
2. **FP8 GEMM, K >= 512, Hopper**: Use FP16 TC accumulator + promotion (interval=4), balancing performance and accuracy
3. **FP4 GEMM, any K, Blackwell**: Use hardware block-scaled MMA with zero additional overhead
4. **FP4 GEMM, non-Blackwell**: Dequantize to FP8/FP16 before MMA, then follow the FP8 strategy
5. **Training**: Always use FP32 accumulator with no accuracy compromises
6. **Inference**: Can use FP16 accumulator + promotion, or FP32 directly (depending on the latency vs. accuracy trade-off)

### 7.2 When to Apply

- When developing FP8/FP4 GEMM kernels
- When evaluating the accuracy impact of quantization schemes (W4A4, W8A8, etc.)
- When debugging convergence issues in low-precision training
- When understanding why FlashAttention insists on FP32 accumulation in Transformer inference

### 7.3 Quick Performance vs. Accuracy Evaluation

```python
def should_use_promotion(K, dtype_bits, accum_bits, arch):
    """Rough estimate of whether promotion is needed"""
    if accum_bits == 32:
        return False  # FP32 accumulator is sufficiently precise

    # Worst-case scenario for FP16 accumulator
    max_product = (2 ** dtype_bits) ** 2  # Upper bound of FP8 product
    max_accum = K * max_product

    if max_accum > 30000:  # FP16 max ≈ 65504, leave safety margin
        return True

    return False
```

---

## Further Reading

- [L2 Cache Persistence](../../../kernel-opt/nvidia/common/l2-cache-persistence.md) — L2 cache persistence, which affects cache hits for quantized weights
- [Occupancy Tuning](../../../kernel-opt/nvidia/common/occupancy-tuning-by-arch.md) — promotion increases register pressure, affecting occupancy
- NVIDIA CUTLASS `fp8_accumulation.hpp` — production-grade implementation of the promotion technique
- NVIDIA CUTLASS Example 55 — Hopper mixed-precision GEMM example
- OCP Microscaling Specification — the official standard for the MX format
