# Pitfalls: FlyDSL Flash Attention Backward — Arbitrary Mask Integration on MI308X

Applicability: backend: flydsl; hardware: amd; topic: pitfalls

API-level integration pitfalls encountered while connecting the optimized dQ + dK/dV
backward kernels to an end-to-end benchmark with arbitrary (non-causal) additive masks.

Kernel-level pitfalls (register layout, LDS, scheduling) are in
[attention-backward-dkdv-pitfalls.md](attention-backward-dkdv-pitfalls.md).
Forward mask pitfalls (bit-packed encoding, VT_STRIDE) are in
[flash-attn-pitfalls.md](flash-attn-pitfalls.md).

Companion optimization report:
[cdna3-flash-attn-bwd-bf16-arbitrary-mask-integration.md](../../ref-docs/flydsl/cdna3-flash-attn-bwd-bf16-arbitrary-mask-integration.md).

Reference API code:
[flash_attn_bwd_flydsl.py](../../../../../../reference-kernels/amd/cdna3/flydsl/FlyDSL/flash_attn_bwd_flydsl_mi308x.py).

---

## 1. Passing `seq_len_padded` to kernel after removing F.pad crashes silently with wrong gradients

**Trap**: Kernel was designed with padded tensors (F.pad to multiple of BLOCK_N=32).
To eliminate F.pad overhead, you add OOB guards (arith.select to row 0) in the kernel
and pass unpadded tensors directly. The kernel still needs `seq_len` for its grid
computation (`ceil(seq_len / BLOCK)`). The "obvious" parameter to pass is
`seq_len_padded` (320) since that's what the loop bounds were computed with.

**Result**: All gradients wrong (garbage values), no error raised.

**Why**: The kernel uses `seq_len` to compute memory strides:
`base_ptr = (batch * H + head) * seq_len * D`. If you pass 320 but the actual tensor
stride is 316, every batch/head beyond the first reads from the wrong address.
`ceil(316/32) == ceil(320/32) == 10` — the grid size is identical either way, so the
kernel launches correctly but accesses wrong memory.

**Lesson**: When removing explicit padding, pass the **actual** `seq_len` (tensor
dimension) to the kernel for stride computation. The padded value is only needed for
loop bounds (which are precomputed on host anyway). Always distinguish between
"padded seq_len for tiling" and "actual seq_len for memory addressing."

---

## 2. Double-masking: applying transform to an already-additive mask

**Trap**: Additive mask convention is 0.0 = attend, -1e6 = masked. Code from a
different project used `(mask - 1.0) * 1e6` to convert from {0, 1} binary format.
If the mask is already in additive format, this transform sends 0.0 → -1e6 and
-1e6 → -1e12, effectively masking ALL positions.

**Result**: All attention outputs are NaN or zero; gradients all zero.

**Why**: `(0.0 - 1.0) * 1e6 = -1e6` makes valid positions masked.
`(-1e6 - 1.0) * 1e6 ≈ -1e12` stays masked. Everything is blocked.

**Lesson**: Always verify mask format before applying transforms. Check
`mask.min()` and `mask.max()` — if min ≈ -1e6 and max ≈ 0, it's already additive.
If min ≈ 0 and max ≈ 1, it's binary and needs conversion.

---

## 3. Fully-masked rows produce NaN in fp32 reference, breaking correctness comparison

**Trap**: Sparse masks may have rows where ALL positions are masked (-1e6). When
computing reference gradients via PyTorch autograd on these rows,
`softmax([-1e6, -1e6, ..., -1e6])` produces uniform probabilities (≈ 1/S) rather
than all-zeros, and the backward through softmax produces non-zero gradients.
Meanwhile the kernel (correctly) skips fully-masked rows via loop bounds, producing
zeros. Comparing kernel output vs reference on these rows shows large relative error.

**Result**: dK/dV correctness check fails with relative error > 5% even though the
kernel is correct for all rows that matter to the loss function.

**Why**: In the forward pass, fully-masked rows contribute nothing to the final loss
(their output is multiplied by zero in downstream computations). Their gradients are
"don't care" values. But the fp32 reference computes them anyway because PyTorch
softmax assigns uniform probability to -inf/-1e6 rows, which propagates non-zero
gradients backward.

**Lesson**: For correctness verification with sparse arbitrary masks:
1. Use **causal mask** for full-coverage verification (every row has valid positions).
2. For the actual sparse mask, only compare rows that have at least one attend position:
   `valid_rows = (mask >= -0.5).any(dim=-1)`. Report errors on valid rows only.

---

## 4. OOB guard pattern: `arith.select` to row 0, not to zero vector

**Trap**: When a KV-tile index exceeds `seq_len`, the intuitive guard is to zero
the loaded vector: `vec_safe = select(in_bounds, vec, zero_vec)`. But this requires
materializing a zero vector constant in registers.

**Result**: Works correctly but wastes VGPR for zero constants.

**Why**: A better pattern is to redirect the **index** to row 0:
`row_safe = select(in_bounds, row_idx, 0); vec = load(base + row_safe * stride)`.
The loaded value from row 0 is "wrong" but gets multiplied by zero in the mask
(since the mask bit for OOB positions is 0). This avoids materializing zero vectors.

**Lesson**: For OOB guards in masked attention kernels, redirect the load index to
a valid row (typically row 0) rather than zeroing the loaded data. The bit-packed
mask already ensures OOB contributions are zeroed in the softmax/score computation.
Both patterns are correct; the index-redirect pattern saves VGPR.

---

## 5. aiter CK-tile bias is 2D (S,S) — cannot represent per-batch masks

**Trap**: When benchmarking against aiter's `mha_bwd` with bias, you might expect
to pass the full (B, 1, S, S) mask as bias.

**Result**: RuntimeError from CK kernel: `bias.sizes() == {seqlen_q, seqlen_k}`
assertion fails. Bias must be exactly 2D.

**Why**: aiter's CK-tile backward kernel hardcodes `batch_stride_bias = 0` and
`nhead_stride_bias = 0` — the same 2D (S, S) bias is broadcast identically across
all batches and heads. This is a kernel-level design limitation of CK's fused
backward implementation.

**Lesson**: aiter CK-tile backward with bias only handles the case where all
sequences share the same mask pattern. For per-batch arbitrary masks (common in
training with variable-length packed sequences), aiter cannot be used. This is a
fundamental capability gap — not just a performance difference. FlyDSL's approach
(bit-packed per-batch mask with precomputed loop bounds) supports the general case.

---

## 6. aiter Triton backward does NOT support arbitrary mask despite Triton being capable

**Trap**: "aiter has Triton flash attention kernels; Triton can load arbitrary mask
tensors; therefore aiter's Triton backward should support free masks."

**Result**: All three aiter Triton backward implementations (`mha_fused_bwd`,
`mha_onekernel_bwd`, and the `mha.py` wrapper) raise
`ValueError("Bias is not supported yet in the Triton Backend")` when bias is not None.

**Why**: While Triton can technically load any mask tensor via `tl.load`, aiter's
Triton backward kernels only implement causal masking (`offs_m >= offs_n` diagonal).
Adding arbitrary mask support would require loading the mask tile per block,
applying it to scores, and propagating it through the backward softmax computation —
all of which adds register pressure and memory traffic. The CK-tile path handles
this via C++ template specialization; the Triton path simply hasn't been implemented.

**Lesson**: "Framework X can do Y" ≠ "Library Z built on framework X does Y."
Always check actual implementation, not theoretical capability. For performance
comparisons, verify which code paths are actually exercised.

---

## Speed reference: "use this / not this"

| Situation | Do this | Not this |
|---|---|---|
| Remove F.pad overhead | OOB guards in kernel + pass actual `seq_len` | Pass `seq_len_padded` with unpadded tensors |
| Verify correctness with sparse mask | Compare valid rows only (rows with ≥1 attend position) | Compare all rows including fully-masked ones |
| Check if mask is additive | `mask.min() ≈ -1e6, mask.max() ≈ 0` | Assume format from variable name |
| Benchmark aiter with mask | Use `mha_bwd` with 2D bias directly | Use Triton backend (raises ValueError) |
| OOB loads in masked kernel | Redirect index to valid row (row 0) | Materialize zero vector in VGPR |
| Per-batch arbitrary mask | FlyDSL bit-packed u32 + loop bounds | aiter CK-tile (2D bias only) |
