# SM120 NVFP4 GEMM: CuTeDSL + inline PTX Pitfall Summary


**Last updated**: 2026-06-30

## Objective

Implement a truly runnable NVFP4 `m16n8k64` warp MMA demo on `sm_120a`, with the following constraints:

- Must not use the SM100 `tcgen05` / TMEM approach
- Must be `CuTeDSL + inline PTX`
- The final kernel must call:

```ptx
mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
```

## TL;DR

SM120's NVFP4 path is closer to Ampere/SM80 warp MMA, rather than SM100's `tcgen05`.

A correct minimal implementation can be manually constructed with:

- `ARegisters = uint32_t[4]`
- `BRegisters = uint32_t[2]`
- `SFARegisters = uint32_t[1]`
- `SFBRegisters = uint32_t[1]`
- `CRegisters = float[4]`

Then use inline PTX to directly issue `mma.sync.aligned.kind::mxf4nvf4...m16n8k64...ue4m3`.

See reference code at:

- `reference-kernels/nvidia/blackwell-geforce/cutedsl/cutlass/sm120_nvfp4_inline_ptx_gemm.py`
- `reference-kernels/nvidia/blackwell-geforce/cutedsl/cutlass/test_sm120_nvfp4_inline_ptx_gemm.py`

## The Most Common Pitfalls

### 1. Treating SM120 as SM100

This is the biggest pitfall.

On SM120, do not apply:

- `tcgen05`
- TMEM accumulator
- Descriptor-based A/B operand
- SM100 blockscaled UMMA mental model

Here, SM120 should be understood in terms of warp MMA:

- One warp executes one `m16n8k64`
- A/B/SF are all regular register operands
- The accumulator is in registers

### 2. Mistakenly Assuming A/B Registers Need Bitplane Packing

This is the second biggest pitfall in this implementation.

At first glance, one might easily see:

- `ARegisters = uint32_t[4]`
- `BRegisters = uint32_t[2]`

And then mistakenly assume "4 bitplanes."

For SM120 NVFP4, the correct approach is:

- Pack A's 32 FP4 values in nibble order into 4 `u32`
- Pack B's 16 FP4 values in nibble order into 2 `u32`

Do not expand by bitplane.

In other words:

```text
reg = nibble0 | (nibble1 << 4) | (nibble2 << 8) | ...
```

Rather than:

```text
bitplane0 = b0_0 | (b0_1 << 1) | ...
bitplane1 = b1_0 | (b1_1 << 1) | ...
```

If bitplane packing is used, the results will be completely wrong.

### 3. `flashinfer.nvfp4_quantize(..., layout_128x4)` Scale Is Not Intuitively Row-Major

Although the shape of `a_sf` / `b_sf` is `(128, 4)`, valid data does not exist in every row.

Actual pitfall findings:

- Logical scale row `r`
- Valid physical rows are at `4 * r`

This means that in the current demo, directly accessing:

```python
gSFA_u8[r, :]
```

is wrong.

You should read:

```python
gSFA_u8[4 * r, :]
```

The same applies to `SFB`.

### 4. The Order of the 4 Accumulators in `C` Fragment Is Easy to Get Wrong

This is the last bug that really caused incorrect results.

The actual semantics of `float[4]` are not something to guess casually. For this `m16n8k64` atom, the final confirmed order is:

```text
acc[0] -> (m0, n0)
acc[1] -> (m0, n1)
acc[2] -> (m8, n0)
acc[3] -> (m8, n1)
```

At first, I swapped `acc[1]` and `acc[2]`, and the symptoms were very confusing:

- Even columns were nearly correct
- Odd columns had systematic errors

This symptom looked very much like a `B` register mapping error, but it was actually a `C` store error.

### 5. Don't Rely Solely on `flashinfer.mm_fp4` as the Only Oracle for Small Shapes

When bringing up a single `16x8x64` tile, it's best to use:

```python
dequant_ref = dequant(a_q, a_sf) @ dequant(b_q, b_sf).T
```

as the primary reference.

Reasons:

- For tiny shapes, certain existing backend paths are not necessarily the best debugging oracle
- What you really need to verify is "whether the inline PTX atom aligns with the quantized payload"

So it's recommended to:

1. First align with a dequantized reference
2. Then check the error against a dense BF16/F32 reference

## Final Confirmed Working Mapping

### A Operand

For lane `t`:

```python
lane_group = t % 4
lane_row = t // 4
```

Each lane takes 32 FP4 values of A:

```python
logical_row = lane_row + 8 * v1
logical_k = lane_group * 8 + v0 + 32 * v2
```

Where:

- `v0 in [0, 7]`
- `v1 in [0, 1]`
- `v2 in [0, 1]`

Then pack in nibble order into `u32[4]`.

### B Operand

Each lane takes 16 FP4 values of B:

```python
logical_row = lane_row
logical_k = lane_group * 8 + v0 + 32 * v1
```

Where:

- `v0 in [0, 7]`
- `v1 in [0, 1]`

Then pack in nibble order into `u32[2]`.

### SF Operand

First, compute the logical row:

```python
sfa_row = (t // 4) + (t % 2) * 8
sfb_row = t // 4
```

Then map to the physical row:

```python
sfa_phys_row = 4 * sfa_row
sfb_phys_row = 4 * sfb_row
```

Finally, pack the corresponding 4 `uint8` into a single `uint32` and pass it to PTX.

## Recommended Debugging Order

When bringing up from scratch, we recommend the following order:

1. First, get the inline PTX atom to compile on its own.
2. Then, get the kernel running without illegal memory access errors.
3. Next, verify only the `A/B` payload is correct — don't immediately suspect the scale.
4. Then, confirm the physical layout of the scale.
5. Finally, validate the store order of `CRegisters[4] -> (m, n)` in isolation.

Pay special attention to the last step: don't prematurely attribute all errors to the `B` operand.

## Final Verification

The final minimal demo aligns with the dequantized reference to within numerical noise levels across 10 random seeds:

```text
worst_seed=3
worst_rel=1.414863107e-07
worst_abs=3.814697266e-06
```

## Reference Code

- `reference-kernels/nvidia/blackwell-geforce/cutedsl/cutlass/sm120_nvfp4_inline_ptx_gemm.py`
- `reference-kernels/nvidia/blackwell-geforce/cutedsl/cutlass/test_sm120_nvfp4_inline_ptx_gemm.py`

## Scope

The current reference kernel is:

- A minimal correctness demo of a single `m16n8k64` atom
- Focused on validating the SM120 NVFP4 register contract

It is **not** a complete high-performance multi-tile GEMM.

To extend this into a practical GEMM, the following work remains:

- K-dimension loop
- Multi-tile M/N mapping
- Shared memory staging
- More efficient load/store

> **The complete scale-up optimization journey from v15 to v43 for a persistent multi-stage GEMM** (66 → 581 TFLOPS at
> 4096³, 71% of CUTLASS C++) has been documented in:
> - Report: [sm120-nvfp4-persistent-gemm-pro5000-optimization.md](sm120-nvfp4-persistent-gemm-pro5000-optimization.md)
> - Pitfalls: [nvfp4-gemm-pitfalls.md](pitfalls/nvfp4-gemm-pitfalls.md)
> - Implementation: [sm120_nvfp4_persistent_gemm_pro5000.py](../../../../reference-kernels/nvidia/blackwell-geforce/cutedsl/cutlass/sm120_nvfp4_persistent_gemm_pro5000.py)
>
> Key findings: CUTLASS-style `pack_sf_per_atom` defaults to 8× inflation; cute-DSL 4.4.2's
> `cp.async.bulk` + `PipelineTmaAsync` mbar integration failed — SF must use an independent cp.async pipeline.
- A more systematic epilogue


## Related

- [Stage 3 Closeout — Path-1 fused sigmoid·gate + NVFP4 quant on sm_120](sm120-fused-fa-epilogue-nvfp4-bf16-optimization.md)
- [CuTeDSL Gated DeltaNet Chunk Forward (bf16, Precomputed Neumann) on SM120](sm120-gdn-chunk-fwd-bf16-neumann-optimization.md)
- [SM120 GDN Decode: cp.async + GLOBAL Cache Quick Reference (kernel-opt)](sm120-gdn-decode-cpasync-cache-mode.md)
- [CuteDSL GDN Decode (fp32 state, bf16 q/k/v) on sm_120 — Optimization Journey](sm120-gdn-decode-fp32state-bf16qkv-optimization.md)
- [SM120 INT32 MoE Data-Prep — Optimization Journey](sm120-moe-data-prep-optimization.md)
- [CUTLASS GEMM Optimization Strategy](../../common/cutedsl/cutlass-gemm-optimization.md)
- [Community GEMM Optimization Practical Summary](../../../generic/gemm-optimization-guide.md)
- [PTX Programming Model and Basics](../../common/ptx/ptx-programming-model.md)
