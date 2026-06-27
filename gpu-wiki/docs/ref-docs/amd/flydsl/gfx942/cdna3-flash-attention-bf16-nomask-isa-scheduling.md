# FlyDSL Flash Attention Forward (bf16, no-mask ISA scheduling) on MI308X

Applicability: backend: flydsl; hardware: amd; topic: reference

This note archives the no-mask FlashAttention forward tuning round on
MI308X / gfx942. It is intentionally separate from the existing causal+GQA and
bit-packed-mask FlashAttention records: this kernel is the no-mask D64 BF16
path. The earlier wiki CK/SDPA no-mask number was FP16, so this report does
not use that number as a BF16 CK95 acceptance line.

## Target hardware

- Hardware: AMD MI308X, CDNA3 / gfx942.
- DSL and compiler stack: FlyDSL rebuilt with ISA scheduling improvements
  for this tuning round.
- Kernel: FlashAttention forward, no attention mask, bf16 inputs, fp32 MFMA
  accumulation, fp32 online softmax, bf16 output.
- Tuned shape: `B=1024, H=8, S=316` padded to `S_pad=320`, `D=64`.
- Dtype clarification: the historical CK/SDPA no-mask wiki number
  (`2.94 ms / 71.3 TFLOPS`) is FP16. It is useful context, but not an
  apples-to-apples BF16 target. This archived BF16 result is treated as meeting
  the current BF16 no-mask objective; capture a BF16 CK baseline before making
  a BF16 CK95 comparison.

## Algorithm baseline

The no-mask path inherits the MI308X FlashAttention structure:

- `BLOCK_M=128`, `BLOCK_N=64`, 256 threads / CTA.
- `head_dim=64` is native: `D_CHUNKS=2` and
  `mfma_f32_32x32x8_bf16`; no D64->D128 padding.
- GEMM1 computes `K @ Q^T` so scores stay in the MFMA32 register layout.
- Online softmax is fully register-resident.
- Softmax probabilities feed GEMM2 directly; there is no P roundtrip through
  LDS.
- K and V have separate LDS regions.

The final reference code is
[`flash_attn_func_nomask_mi308x.py`](../../../../../reference-kernels/amd/cdna3/flydsl/FlyDSL/flash_attn_func_nomask_mi308x.py).

## Kernel resource footprint (final)

Authoritative profile:
`profiles/nomask_v6_final_lgkm_sum_default/rocprof/`.

Authoritative ATT:
`profiles/nomask_v6_final_lgkm_sum_default/att/rocprof_att_codeobj4_kernel_body.s`,
MD5 `333eb8ff20e3a0911833b75a2cf86ded`.

| Metric | Final |
|---|---:|
| LDS block size | 16896 B |
| VGPR / AccVGPR / SGPR | 128 / 8 / 112 |
| Scratch | 0 |
| Body instructions | 719 |
| `v_mfma` | 32 |
| `v_exp` | 33 |
| `ds_bpermute` | 2 |
| `s_waitcnt` | 43 |
| `s_barrier` | 2 |
| `s_nop` | 5 |

## Optimization journey

### Baseline before V6

The retained no-mask baseline before this round measured:

| Metric | Value |
|---|---:|
| P50 | 3.503414 ms |
| Throughput | 59.774040 TFLOPS |
| Correctness | rel_err 0.017906 |

The baseline ATT showed the first row reduction used `ds_bpermute` followed
by a full wait:

```asm
s_waitcnt vmcnt(0) lgkmcnt(0)
```

That forced the V global-load stream to drain before the kernel had spent the
independent VALU work available between softmax reduction and V consumption.

### Local CK ISA capture

Real CK ATT was captured with `rocprofv3 --att` for D64 no-mask:

| Local CK runner | Time | Throughput | Use |
|---|---:|---:|---|
| S=316 | 3.746 ms | 55.91 TFLOPS | scheduling reference only |
| S=320 | 3.656 ms | 58.73 TFLOPS | scheduling reference only |

This local CK runner was used only for scheduling evidence. It also did not
reproduce the historical wiki CK number (`2.94 ms / 71.3 TFLOPS`), and that
wiki number is FP16, so neither local CK capture nor the wiki CK number is used
as the BF16 acceptance baseline. The useful cue from CK ATT was wait separation
around the softmax/reduction/V-store boundary rather than exact constants.

### V6 retained change: split LDS reduction wait from VMEM drain

Final default:

```text
FLYDSL_FLASH_ATTN_FUNC_REDUCE_MODE=ds_bpermute_lgkm_sum
```

The retained implementation is asymmetric:

- Rowmax reduction keeps the normal `rocdl.ds_bpermute` path.
- Rowsum reduction emits inline `ds_bpermute_b32` and then
  `s_waitcnt lgkmcnt(0)`.

The important ISA-level effect is not the `ds_bpermute` itself; it is that
the first reduction no longer forces `vmcnt(0)`. The VMEM wait moves later,
after eight independent `v_pk_mul_f32` rescale instructions and just before
`ds_swizzle_b32` consumes the V values.

### Rejected variants

| Variant | P50 | TFLOPS | Verdict |
|---|---:|---:|---|
| `ds_bpermute_lgkm` for both reductions | 3.486033 ms | 60.072068 | helps, not best |
| `ds_bpermute_lgkm_max` | 3.514374 ms | 59.587636 | rowmax-only lgkm path regressed |
| ROCDL rowsum `ds_bpermute` + explicit wait | 3.502474 ms | 59.790083 | did not trigger useful schedule |
| `EARLY_RESCALE_ALL=1` | 3.494454 ms | 59.927313 | extra VALU pressure outweighed overlap |
| `NOMASK_SOFTMAX_BARRIER=0` | 3.492694 ms | 59.957511 | allocation/schedule changed and regressed |

Previous V5 attempts also showed that CK-like constants do not transfer
mechanically: `QK_PREFETCH_DEPTH=3` regressed to `59.312539 TFLOPS`, and
softmax barrier masks `1` / `0x7f` stayed near `57.65` / `57.71 TFLOPS`.

## Final perf vs baseline

| Implementation | P50 | TFLOPS | Relative |
|---|---:|---:|---:|
| Previous retained no-mask best | 3.503414 ms | 59.774040 | 1.000x |
| Final V6 default `ds_bpermute_lgkm_sum` | 3.479213 ms | 60.189822 | 1.00696x |
| Historical CK/SDPA no-mask wiki row | 2.94 ms | 71.3 | FP16; not BF16 target |

Correctness:

- rel_err vs fp32 reference: `0.017906`
- rel_err vs bf16 reference: `0.018016`
- Status: PASS under the `0.02` threshold

The final kernel is recorded as passing the BF16 no-mask objective for this
archive. Do not compute BF16 CK95 from the FP16 CK/SDPA row.

## Remaining optimization headroom

The final schedule removes one avoidable VMEM drain. Further BF16 optimization
headroom remains because:

1. The no-mask D64/S316 shape has limited CTA work for 80 CUs.
2. The final body still has 43 `s_waitcnt` instructions around LDS, VMEM, and
   reduction boundaries.
3. Manual wait splitting is local; it does not give CK's full core-loop
   scheduling control.
4. More CK constants without CK's exact access pattern regressed in FlyDSL.

## What would improve further

- A full CK-like core-loop schedule for the FlyDSL no-mask path, not just local
  wait splitting.
- Inline-asm control for a larger contiguous MFMA/reduction/V-store region if
  the compiler cannot preserve the desired order.
- A reproduced BF16 CK no-mask benchmark on the same runner, so any CK95
  comparison is dtype-matched.
- Shape-specific handling for S=316/S_pad=320 tail behavior if the real CK
  target depends on a specialized no-tail path.

## Sustained recipe

1. Start from the no-mask D64 native path; do not reuse the causal+GQA D128
   assumptions.
2. Keep `QK_PREFETCH_DEPTH=2` unless a fresh profile proves otherwise.
3. Keep the no-mask softmax barrier enabled with mask `0`.
4. Use `ds_bpermute_lgkm_sum`: only rowsum gets inline
   `ds_bpermute_b32` + `s_waitcnt lgkmcnt(0)`.
5. Inspect ATT after every scheduling change. The intended signature is a
   delayed `vmcnt(0)` across the independent `v_pk_mul_f32` rescale window.
6. Validate with both correctness and `rocprofv3 --kernel-trace`; this round's
   useful gain was only 0.7%, so noise control matters.

## Conflict and difference notes

- The existing mask report lists "FlyDSL no mask" as
  `3.88 ms / 54.0 TFLOPS`. This no-mask archive uses rebuilt FlyDSL plus
  later ISA scheduling work, so the final
  `3.479213 ms / 60.189822 TFLOPS` number is not a contradiction.
- The historical wiki CK/SDPA no-mask row is FP16. It is scheduling and context
  evidence only for this BF16 report, not a BF16 CK95 acceptance baseline.
- The local CK ATT captures are slower than the historical wiki CK/SDPA row and
  are scheduling evidence only.
- Pitfall #34 already warns that CK constants do not transfer without CK access
  patterns. This no-mask round reinforces it: depth-3 QK prefetch and CK-like
  barrier masks regressed.

## Related docs

- Reference kernel:
  [`flash_attn_func_nomask_mi308x.py`](../../../../../reference-kernels/amd/cdna3/flydsl/FlyDSL/flash_attn_func_nomask_mi308x.py)
- Related mask journey:
  [cdna3-flash-attention-bf16-mask-optimization.md](cdna3-flash-attention-bf16-mask-optimization.md)
- Related causal+GQA journey:
  [cdna3-flash-attention-bf16-gqa-optimization.md](cdna3-flash-attention-bf16-gqa-optimization.md)
- Pitfalls:
  [flash-attn-pitfalls.md](../../../../pitfalls/amd/flydsl/flash-attn-pitfalls.md)
