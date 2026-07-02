# NVIDIA Blackwell Tensor Core Analysis (Part 2): B300

An analysis of new PTX ISA 8.8 features specific to the B300 (Blackwell Ultra / SM_103) GPU, focusing on the 1.5x FP4 throughput improvement and new attention-related instructions.


**Last updated**: 2026-06-30

---

## 1. Overview

NVIDIA recently released PTX ISA 8.8, which likely contains information about the B300 (Blackwell Ultra) variant. Given that B200 is SM_100 and Thor is SM_102, the newly introduced SM_103 clearly corresponds to B300.

NVIDIA highlights two key improvements for Blackwell Ultra:
1. **1.5x FP4 compute throughput**
2. **"New Attention Instructions"**

The following sections examine the SM_103-exclusive features found in the PTX ISA.

---

## 2. K=96 FP4 Mode (2CTA + SM_103 Only)

When and only when 2CTA mode is enabled on SM_103 hardware, the Tensor Core FP4 mode supports K=96. This feature directly corresponds to the 1.5x FP4 throughput claim:

- H100 FP8 → B200 FP4: 4x improvement
- B300 over B200: additional 1.5x = 6x over H100 FP8

---

## 3. Tensor Memory Load-Reduce (SM_101 and SM_103)

```
// Floating point type load along with reduction
tcgen05.ld.red.sync.aligned.shape3.num.redOp{.abs}{.NaN}.f32 r, redval, [taddr];
tcgen05.ld.red.sync.aligned.shape4.num.redOp{.abs}{.NaN}.f32 r, redval, [taddr], immHalfSplitoff;

// Integer type load along with reduction
tcgen05.ld.red.sync.aligned.shape3.num.redOp.type r, redval, [taddr];
tcgen05.ld.red.sync.aligned.shape4.num.redOp.type r, redval, [taddr], immHalfSplitoff;

.shape3 = { .32x32b }
.shape4 = { .16x32bx2 }
.redOp  = { .min, .max }
.type   = { .u32, .s32 }
```

The tensor memory load-reduce instruction simultaneously loads data from Tensor Memory to the register file while performing a reduction operation along the N dimension. Only `max` or `min` operations are supported.

### 3.1 Purpose

This instruction accelerates **FP32 → block scale type quantization operations**. When loading accumulator values from TMEM, the hardware can compute the row-wise max/min in a single operation, eliminating the need for a separate reduction pass before computing scaling factors.

### 3.2 Relation to "New Attention Instructions"

Whether this constitutes NVIDIA's advertised "New Attention Instructions" is debatable — unless they benchmark NVFP4 attention against FP16 attention to claim the improvement.


## Related

- [Comprehensive Guide to NVIDIA Blackwell Architecture](blackwell-architecture-comprehensive-guide.md)
- [GPGPU Architecture: Blackwell Instruction Analysis](blackwell-architecture-instruction-analysis.md)
- [Blackwell GPGPU Architecture New Features Overview](blackwell-gpgpu-new-features-overview.md)
- [NVIDIA Blackwell Tensor Core Analysis (Part 1)](blackwell-tensor-core-analysis-part1.md)
- [Blackwell Ultra (B300): NVIDIA AI Chip Evolution and Roadmap](blackwell-ultra-b300-chip-evolution.md)
- [Tensor Core from Volta to Blackwell](../../common/tensor-core-volta-to-blackwell.md)
- [PTX Programming Model and Basics](../../common/ptx/ptx-programming-model.md)
- [PTX Core Instruction Set](../../common/ptx/ptx-instruction-set.md)
