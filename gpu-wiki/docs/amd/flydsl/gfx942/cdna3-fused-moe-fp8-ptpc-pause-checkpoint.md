# FlyDSL Fused MoE FP8 PTPC Pause Checkpoint on MI308X

Applicability: backend: flydsl; hardware: amd; topic: reference


**Last updated**: 2026-06-30

This document archives the omoExplore `proj007 task66` pause checkpoint for a
FlyDSL FP8 PTPC Fused MoE two-stage GEMM on AMD MI308X (CDNA3 / gfx942). It is a
continuation map, not a final success report: at the checkpoint, only one of the
fourteen stage/token rows passes the 5 percent gate.

Related code:

- [`reference-kernels/amd/cdna3/flydsl/FlyDSL/moe_fp8_ptpc_mi308x/`](../../../../reference-kernels/amd/cdna3/flydsl/FlyDSL/moe_fp8_ptpc_mi308x)

Pitfalls:

- [`docs/pitfalls/amd/flydsl/fused-moe-fp8-ptpc-pitfalls.md`](../pitfalls/fused-moe-fp8-ptpc-pitfalls.md)

Existing adjacent report:

- [`cdna3-fused-moe-bf16-optimization.md`](cdna3-fused-moe-bf16-optimization.md) covers a different BF16 Fused MoE journey and does not supersede this FP8 PTPC checkpoint.
- [`cdna3-fused-moe-fp8-ptpc-atrex-v2.md`](cdna3-fused-moe-fp8-ptpc-atrex-v2.md) covers the later atrex-open integrated v2 full-pipeline archive. It does not replace this task66 isolated stage checkpoint or its bandwidth `target_us` gate.

## Target Hardware And Scope

```text
hardware: AMD MI308X / CDNA3 / gfx942
framework: FlyDSL
kernel: Fused MoE two-stage GEMM
dtype: FP8 PTPC input, BF16 stage/output activation, F32 scales, I32 metadata
shape: E=512, topk=10, model_dim=4096, inter_dim=256
tokens: 1/16/32/64/128/256/512
```

## Acceptance And Byte Contract

The task66 gate is:

```text
stage1: profile_top_us   <= target_5pct_us
stage2: profile_gemm2_us <= target_5pct_us
target_5pct_us = 1.05 * fp8_loadonly_ref_us_cm2
correctness: passed == 1 and max_delta <= 0.01
```

Byte contract:

- GEMM input reads are FP8.
- Stage outputs and activation outputs are BF16.
- Stage2 includes BF16 output atomic read-modify-write traffic.
- Scale tensors are F32.
- Expert/topk metadata is I32.
- BF16/F16-derived reference rows are invalid for this gate.

## Bandwidth Target Method

The `target_5pct_us` column is generated from a same-byte load-only bandwidth
proxy before comparing kernel timings. It is not a theoretical HBM peak and not a
torch copy number.

Archived harness:

- [`bandwidth_reference.py`](../../../../reference-kernels/amd/cdna3/flydsl/FlyDSL/moe_fp8_ptpc_mi308x/bandwidth_reference.py)

Reproduction command:

```bash
cd reference-kernels/amd/cdna3/flydsl/FlyDSL/moe_fp8_ptpc_mi308x

${PYTHON:-python3} - <<'PY'
import flydsl
PY

${PYTHON:-python3} bandwidth_reference.py \
  --out artifacts/task66_bandwidth_reference.csv \
  --markdown-out artifacts/task66_bandwidth_reference.md
```

The script does not inject machine-local paths and does not depend on
`tensor_tools`; the minimal `flush_cache` and `profile_cuda_kernels` helpers are
inlined in `bandwidth_reference.py`. `flydsl` must be importable from the active
Python environment, or from a `PYTHONPATH` set by the caller before running the
command.

Canonical call shape:

```python
measure_flydsl_load(
    measured_hbm_bytes,
    cache_modifier=2,
    mode="loadonly",
    tiles_per_thread=4,
    warmup=10,
    iters=50,
    cold=True,
)
```

Timing source:

```text
inlined profile_cuda_kernels -> bw_kernel device time
```

`cache_modifier=2` is the task66 target source. `cache_modifier=0`, torch event
timing, torch copy/memcpy fallback, and theoretical peak bandwidth are diagnostic
only. The FlyDSL harness aligns requested bytes to its launch block size for the
load kernel; the gate uses the measured `bw_kernel` device time:

```text
target_us = 1.05 * measured_ref_us
```

Task10 re-ran the 14 target rows with this method on 2026-05-26. The retest
validates the bandwidth target method only; it does not retest the current FP8
PTPC kernel performance.

| Stage | Token | Measured HBM bytes | task66 ref us | retest ref us | task66 target us | retest target us | target delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| stage1 | 1 | 21051392 | 8.164 | 8.312 | 8.572 | 8.727 | +1.81% |
| stage1 | 16 | 288119040 | 65.368 | 65.334 | 68.636 | 68.600 | -0.05% |
| stage1 | 32 | 513188365 | 114.223 | 113.808 | 119.934 | 119.499 | -0.36% |
| stage1 | 64 | 776474938 | 171.382 | 170.966 | 179.951 | 179.514 | -0.24% |
| stage1 | 128 | 996526080 | 219.546 | 218.405 | 230.523 | 229.325 | -0.52% |
| stage1 | 256 | 1069248493 | 235.358 | 234.258 | 247.126 | 245.970 | -0.47% |
| stage1 | 512 | 1090042579 | 245.740 | 252.747 | 258.027 | 265.385 | +2.85% |
| stage2 | 1 | 10866176 | 5.611 | 5.437 | 5.892 | 5.709 | -3.10% |
| stage2 | 16 | 150928525 | 35.613 | 35.535 | 37.394 | 37.312 | -0.22% |
| stage2 | 32 | 259199168 | 59.409 | 59.019 | 62.379 | 61.970 | -0.66% |
| stage2 | 64 | 399025210 | 89.574 | 89.350 | 94.053 | 93.817 | -0.25% |
| stage2 | 128 | 526315059 | 117.419 | 120.774 | 123.290 | 126.813 | +2.86% |
| stage2 | 256 | 583779091 | 130.424 | 129.586 | 136.945 | 136.065 | -0.64% |
| stage2 | 512 | 640056467 | 142.422 | 142.255 | 149.543 | 149.368 | -0.12% |

Summary:

```text
rows: 14
rows within 1% target delta: 10/14
max absolute target delta: 3.10%
```

## Checkpoint State

```text
full-scope gate pass: 1/14
stage1 gate pass: 1/7
stage2 gate pass: 0/7
unfinished cases: 13/14
source state: task64 promoted source + task65 documentation only
retained kernel changes after task65: none
```

## Current Performance Table

| Stage | Token | Current us | Target 5pct us | Need improve us | Gap to target | Theoretical HBM MB | Measured HBM MB | Correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stage1 | 1 | 12.101 | 8.572 | 3.529 | 41.17% | 21.654 | 21.051 | yes |
| stage1 | 16 | 70.919 | 68.636 | 2.283 | 3.33% | 305.665 | 288.119 | yes |
| stage1 | 32 | 126.510 | 119.934 | 6.576 | 5.48% | 544.414 | 513.188 | yes |
| stage1 | 64 | 184.636 | 179.951 | 4.685 | 2.60% | 823.394 | 776.475 | yes |
| stage1 | 128 | 238.236 | 230.523 | 7.713 | 3.35% | 1055.697 | 996.526 | yes |
| stage1 | 256 | 261.414 | 247.126 | 14.288 | 5.78% | 1129.960 | 1069.248 | yes |
| stage1 | 512 | 157.648 | 258.027 | pass | -38.90% | 1144.654 | 1090.043 | yes |
| stage2 | 1 | 6.587 | 5.892 | 0.695 | 11.80% | 10.865 | 10.866 | yes |
| stage2 | 16 | 43.007 | 37.394 | 5.613 | 15.01% | 151.379 | 150.929 | yes |
| stage2 | 32 | 73.563 | 62.379 | 11.184 | 17.93% | 259.987 | 259.199 | yes |
| stage2 | 64 | 116.700 | 94.053 | 22.647 | 24.08% | 400.218 | 399.025 | yes |
| stage2 | 128 | 152.296 | 123.290 | 29.006 | 23.53% | 527.776 | 526.315 | yes |
| stage2 | 256 | 175.388 | 136.945 | 38.443 | 28.07% | 585.082 | 583.779 | yes |
| stage2 | 512 | 193.958 | 149.543 | 44.415 | 29.70% | 649.438 | 640.056 | yes |

## Retained Source State

The archived source package is:

- `reference-kernels/amd/cdna3/flydsl/FlyDSL/moe_fp8_ptpc_mi308x/`

It preserves:

- `moe_kernels.py` for config registration and wrapper dispatch.
- `kernels/moe_gemm_2stage.py` for stage1 and stage2 builders.
- `kernels/mfma_epilogues.py` and `kernels/mfma_preshuffle_pipeline.py` helpers.
- `test_flydsl_moe_fp8_ptpc.py` as the isolated correctness/profile harness.

The source is retained as a checkpoint. It should not be presented as a production-complete full-scope kernel until the remaining 13 rows are revalidated and closed.

## Optimization Journey Summary

### Current promoted source

The task66 checkpoint inherits the task64 promoted source table. Stage1 token512 uses the integrated `hybrid16sort` path and is the only row passing the 5 percent gate.

### Negative probes after task64

Task65 records row-context, transient rowinfo-LDS, low-token stage2, and near-target stage1 probes. All 44 summarized rows are correct, but none pass the 5 percent gate and no source change is retained.

Important non-promotions:

- Stage2 rowctx/rowinfo-LDS reimplementation is correct but not promotable.
- Stage2 low-token tile sweep gives at most tiny local improvement and misses target.
- Stage2 `block_m=32` is negative for tokens 1/16/32.
- Stage1 `FAST_BARRIER` probes do not pass token16/token128.

## Remaining Bottleneck

The primary unresolved bottleneck is stage2 BF16 atomic/output cost. Task40 skip-atomic diagnostics showed that the GEMM body can approach or beat large-token targets when final BF16 atomic/output is removed, but that path is invalid because it does not produce the required output.

The measured HBM bytes are close to theoretical bytes for the current model, so the next useful work should change the stage2 output/atomic structure rather than reopen byte-accounting shortcuts.

## Do Not Reuse As Positive Evidence

- BF16/F16-derived references.
- Any path that changes GEMM input from FP8 or output from BF16.
- Stage2 skip-atomic or no-output diagnostics.
- Existing `block_m=8` fast timings.
- Racy non-atomic output.
- Full `[tokens, topk, model_dim]` intermediate reduce path as currently tested.
- Task65 rowctx/rowinfo-LDS reimplementation.

## Continuation Recipe

1. Revalidate the task66 checkpoint on the target machine and ensure `task66_current_gap_summary.csv` is reproduced before comparing user-visible benefit.
2. Start future work at `task_67_*`.
3. For stage2, reduce BF16 atomic/output cost while preserving the BF16 output contract and avoiding the full intermediate reduce path.
4. For stage1, prioritize near-target rows: tokens16, tokens64, tokens128, then tokens256.
5. Promote source only when correctness, byte contract, and the 5 percent gate all pass.

## Source Provenance

omoExplore source of truth:

- `proj/proj_007_mi308x_fused_moe_opt/tasks/task_66_fp8_ptpc_pause_summary_and_todo.md`
- `proj/proj_007_mi308x_fused_moe_opt/assets/task_66_fp8_ptpc_pause_summary/task66_current_gap_summary.csv`

The archived source package records the exact code tree needed to resume the isolated stage-level harness.


## Related

- [FlyDSL Attention Backward dQ + dK+dV (bf16, Causal Mask) on MI308X (gfx942)](cdna3-attention-backward-dkdv-bf16-causal-mask-optimization.md)
- [MI308 Chunk-GDN Wave-Specialized Megakernel Playbook](cdna3-chunk-gdn-mi308x-wave-specialized-megakernel-optimization.md)
- [FlyDSL Flash Attention (bf16, MHA + GQA) Optimization on MI308X (gfx942)](cdna3-flash-attention-bf16-gqa-optimization.md)
- [FlyDSL Flash Attention Forward (bf16, mask+LSE) on MI308X — V8-V10](cdna3-flash-attention-bf16-mask-lse-optimization.md)
- [FlyDSL Flash Attention bf16 with Free Mask on MI308X (gfx942)](cdna3-flash-attention-bf16-mask-optimization.md)
- [CUTLASS GEMM Optimization Strategy](../../../nvidia/common/cutedsl/cutlass-gemm-optimization.md)
- [Community GEMM Optimization Practical Summary](../../../generic/gemm-optimization-guide.md)
- [Composable Kernel (CK) Architecture Overview](../../common/ck-architecture-overview.md)
- [Document Relationship Diagram](../../../RELATIONS.md)
