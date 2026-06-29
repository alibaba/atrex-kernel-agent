# Software-Emulated Exponential

## Overview

FlashAttention-4 replaces the hardware Special Function Unit (SFU) exponential (`ex2.approx`) with a software-emulated 2^x function that distributes computation across the SM's FMA (fused multiply-add) units. On Blackwell, tensor core throughput doubled compared to Hopper while SFU count remained the same, making the SFU the throughput bottleneck for attention's softmax operation. The software exponential uses Cody-Waite range reduction followed by a Horner-form polynomial evaluation, achieving sufficient accuracy for attention while bypassing the SFU entirely.

## Why SFU Is a Bottleneck on Blackwell

The softmax in attention requires computing `exp(x - max)` for every element of the score matrix. On previous generations, the SFU's `ex2.approx` instruction was fast enough relative to the MMA throughput. On Blackwell:

| Resource | Hopper (SM90) | Blackwell (SM100) | Ratio |
|----------|--------------|-------------------|-------|
| Tensor core TFLOPS (BF16) | ~990 | ~2250 | 2.27x |
| SFU units per SM | 16 | 16 | 1.0x |
| SFU throughput (exp per cycle) | 16 | 16 | 1.0x |
| FMA units per SM | 128 | 128 | 1.0x |

The tensor cores produce 2x more score elements per cycle, but the SFU can only process exp() at the same rate as before. This makes the SFU the bottleneck for any kernel that needs exp() proportional to the number of MMA outputs.

FlashAttention-4's approach: distribute the exp() workload across FMA units (128 per SM) instead of SFU units (16 per SM), achieving 8x the throughput for the exponential computation.

## Cody-Waite Range Reduction

Range reduction transforms the input `x` into a small residual that a polynomial can accurately approximate. The Cody-Waite method splits the input into an integer part (for exact power-of-two scaling) and a fractional part (for polynomial approximation):

```cuda
// Cody-Waite range reduction for 2^x
// Goal: decompose x = n + r where n is integer, r in [-0.5, 0.5]
// Then 2^x = 2^n * 2^r, and 2^r is approximated by polynomial
//
// The Cody-Waite trick: subtract n using two constants (C1 + C2)
// to maintain precision when x is large.
//
// C1 is the nearest representable float to log2(e) with low-order bits zeroed
// C2 is the correction: log2(e) - C1
// This avoids catastrophic cancellation in x - n

__device__ float software_exp2(float x) {
    // Step 1: Range reduction (Cody-Waite)
    // Round x to nearest integer
    float n = rintf(x);
    // High-precision subtraction using two constants
    // C1 and C2 together represent 1.0 in extended precision
    const float C1 = 1.0f;    // Exact in float
    const float C2 = 0.0f;    // Correction term (zero for 2^x, non-zero for e^x)
    // For 2^x, range reduction is simpler: r = x - n
    float r = x - n;          // r in [-0.5, 0.5]

    // Step 2: Polynomial approximation of 2^r via Horner's method
    // Minimax polynomial coefficients for 2^r on [-0.5, 0.5]
    // Degree-4 polynomial: sufficient for ~22 bits of accuracy
    const float c0 = 1.0f;
    const float c1 = 0.6931471805599453f;   // ln(2)
    const float c2 = 0.2402265069591007f;   // ln(2)^2 / 2
    const float c3 = 0.05550410866482158f;  // ln(2)^3 / 6
    const float c4 = 0.009618129107628477f; // ln(2)^4 / 24

    // Horner evaluation: c0 + r*(c1 + r*(c2 + r*(c3 + r*c4)))
    // Each step is one FMA instruction
    float poly = c4;
    poly = fmaf(poly, r, c3);   // FMA 1
    poly = fmaf(poly, r, c2);   // FMA 2
    poly = fmaf(poly, r, c1);   // FMA 3
    poly = fmaf(poly, r, c0);   // FMA 4

    // Step 3: Reconstruct 2^x = 2^n * poly
    // Use integer addition to the float exponent field
    int n_int = (int)n;
    // ldexpf multiplies by 2^n by adjusting the exponent bits
    float result = ldexpf(poly, n_int);

    return result;
}
```

## Distributing Across FMA Units

The key insight is that Horner polynomial evaluation is a chain of FMA operations. With the softmax warp executing on CUDA cores while the MMA warp uses tensor cores, the FMA throughput is fully available:

```cuda
// FlashAttention-4 softmax with software exp2
// Executed by dedicated softmax warpgroups (part of warp specialization)
//
// For each row of the score matrix S[i,:]:
//   1. Find row max: m_new = max(S[i,:])
//   2. Compute exp2((S[i,j] - m_new) * log2(e)) for each j
//   3. Sum for normalization denominator
//   4. Conditionally rescale previous output if max changed

__device__ void softmax_with_software_exp(
    float* scores,       // Input: S[i, 0..N-1] (one row)
    float* output,       // Output: softmax(S[i,:])
    int N,
    float* row_max,      // Running max (for online softmax)
    float* row_sum)      // Running sum
{
    int lane = threadIdx.x % 32;

    // Step 1: Find max across the row (warp reduction)
    float local_max = -INFINITY;
    for (int j = lane; j < N; j += 32) {
        local_max = fmaxf(local_max, scores[j]);
    }
    // Warp-level max reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, offset));
    }
    float m_new = local_max;

    // Step 2: Compute software exp2 and sum
    float local_sum = 0.0f;
    const float LOG2E = 1.4426950408889634f;

    for (int j = lane; j < N; j += 32) {
        float x = (scores[j] - m_new) * LOG2E;
        float exp_val = software_exp2(x);  // 4 FMAs instead of 1 SFU op
        output[j] = exp_val;
        local_sum += exp_val;
    }

    // Warp-level sum reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
    }

    // Step 3: Conditional rescaling (FlashAttention online softmax)
    float m_old = *row_max;
    if (m_new > m_old) {
        // Rescale previous accumulated output
        float scale = software_exp2((m_old - m_new) * LOG2E);
        *row_sum = (*row_sum) * scale + local_sum;
        *row_max = m_new;
        // The output accumulator must also be rescaled by `scale`
    } else {
        float scale = software_exp2((m_new - m_old) * LOG2E);
        *row_sum += local_sum * scale;
        // Rescale current exp values, not the accumulator
    }
}
```

## PTX-Level FMA Chain

At the PTX level, the Horner polynomial compiles to a tight chain of `fma.rn.f32` instructions:

```ptx
// Software exp2 via Horner polynomial in PTX
// Input: %x (float, range-reduced to [-0.5, 0.5])
// Output: %result (float, approximation of 2^x)

.reg .f32 %x, %r, %n, %poly, %result;
.reg .f32 %c0, %c1, %c2, %c3, %c4;

// Load polynomial coefficients
mov.f32 %c0, 0f3F800000;    // 1.0
mov.f32 %c1, 0f3F317218;    // 0.6931471805599453  (ln2)
mov.f32 %c2, 0f3E75FDF0;    // 0.2402265069591007
mov.f32 %c3, 0f3D635847;    // 0.05550410866482158
mov.f32 %c4, 0f3C1D9539;    // 0.009618129107628477

// Range reduction: n = rintf(x), r = x - n
cvt.rni.f32.f32 %n, %x;     // Round to nearest int
sub.f32         %r, %x, %n; // Fractional part

// Horner evaluation: 4 dependent FMAs
//   poly = c4
//   poly = poly * r + c3
//   poly = poly * r + c2
//   poly = poly * r + c1
//   poly = poly * r + c0
mov.f32         %poly, %c4;
fma.rn.f32      %poly, %poly, %r, %c3;  // FMA 1
fma.rn.f32      %poly, %poly, %r, %c2;  // FMA 2
fma.rn.f32      %poly, %poly, %r, %c1;  // FMA 3
fma.rn.f32      %poly, %poly, %r, %c0;  // FMA 4

// Reconstruct: result = poly * 2^n
// Convert n to int and use ex2 scaling via bit manipulation
cvt.rzi.s32.f32 %ni, %n;
ex2.approx.f32  %scale, %n;    // Or use integer exponent manipulation
mul.f32         %result, %poly, %scale;
```

## Accuracy Considerations

The degree-4 polynomial provides approximately 22 bits of mantissa accuracy, which is more than sufficient for attention softmax where:
- The input `x = (S[i,j] - max) * log2(e)` is always non-positive
- The softmax output is normalized, so small absolute errors cancel out
- BF16 output has only 7 mantissa bits anyway

For applications requiring higher accuracy, a degree-6 polynomial (6 FMAs) achieves near-ULP accuracy across the full float range.

## When to Use

- **Attention kernels on Blackwell**: Whenever the SFU is the bottleneck for softmax computation. FlashAttention-4 measured 1.1-1.3x speedup over cuDNN from this technique alone on B200.
- **Any kernel limited by transcendental function throughput**: If profiling shows SFU utilization near 100% while FMA utilization is low, software emulation can rebalance the workload.
- **Not recommended on Hopper**: The SFU-to-MMA throughput ratio is better balanced on SM90. The overhead of 4 FMAs vs 1 SFU instruction is not justified unless the SFU is proven to be the bottleneck.

## B200 vs B300 (Blackwell Ultra) — the key practical difference

Software exp2 is a **B200-targeted mitigation, not a universal Blackwell win**. The bottleneck it addresses is arch-specific:

- **B200 (sm_100):** tensor-core throughput roughly doubled vs Hopper while the SFU/`ex2` path did not, so for softmax-heavy attention the SFU is the bottleneck — software exp2 (exp on FMA units) is a clear win.
- **B300 / Blackwell Ultra (sm_103):** NVIDIA advertises **~2× attention-layer acceleration** vs B200 — the special-function / softmax path is strengthened, so hardware `MUFU.EX2` is much closer to keeping up with the tensor cores. On B300 the SFU exp bottleneck that motivates software exp2 is largely relieved: hardware `ex2.approx` is often adequate, and the extra FMA work of the software path can be **net-negative**. In practice, on B300 it is frequently faster to *disable* the software-exp emulation and use hardware `MUFU.EX2`.

**Rule of thumb:** treat software exp2 as conditional on "is the SFU the proven softmax bottleneck on *this* arch?" — usually yes on B200, often no on B300. Profile `MUFU`/SFU utilization vs FMA utilization on the target part before enabling it, and keep it behind a flag so it can be turned off per architecture.

## Caveats

- The 4-FMA chain has a latency of ~16 cycles (4 dependent FMAs at ~4 cycles each), vs ~20 cycles for SFU `ex2.approx`. Latency is comparable; the win comes from throughput (128 FMA units vs 16 SFU units).
- **Arch-conditional (see above):** on B300/sm_103 the hardware exp path is faster relative to the tensor cores than on B200, so the software path is often unnecessary or slower. Do not enable it unconditionally for "Blackwell".
- Polynomial coefficients are for 2^x on [-0.5, 0.5]. For e^x, multiply the input by log2(e) first.
- The `ldexpf` or exponent bit-manipulation step for 2^n must handle overflow/underflow (very large/small x). In attention, `x <= 0` always holds, so only underflow toward zero is possible.
