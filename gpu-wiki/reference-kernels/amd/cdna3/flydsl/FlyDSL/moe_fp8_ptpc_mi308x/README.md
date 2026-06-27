# MI308X FlyDSL FP8 PTPC Fused MoE Checkpoint

This package archives the omoExplore `proj007 task66` FlyDSL FP8 PTPC Fused MoE
pause checkpoint for AMD MI308X (CDNA3 / gfx942).

It is not a final all-gates-passed implementation. At task66, only stage1 token512
passes the 5 percent acceptance gate. The value of this archive is the retained
source state, byte-accounting contract, invalid-evidence boundaries, and negative
probe map for continuation work.

## Scope

```text
hardware: AMD MI308X / CDNA3 / gfx942
framework: FlyDSL
kernel: Fused MoE two-stage GEMM
dtype: FP8 PTPC input, BF16 stage/output activation, F32 scales, I32 metadata
shape: E=512, topk=10, model_dim=4096, inter_dim=256
tokens: 1/16/32/64/128/256/512
```

## Files

| File | Purpose |
|---|---|
| `moe_kernels.py` | Kernel naming, configuration registration, compile dispatch, and high-level stage1/stage2 wrappers |
| `kernels/moe_gemm_2stage.py` | Stage1 and stage2 FlyDSL MFMA FP8 kernel builders |
| `kernels/mfma_epilogues.py` | Shared default and CShuffle MFMA epilogue helpers |
| `kernels/mfma_preshuffle_pipeline.py` | Shared preshuffle, LDS, and B-tile load helpers |
| `test_flydsl_moe_fp8_ptpc.py` | Correctness and isolated stage timing harness |
| `bandwidth_reference.py` | Task66 FP8 measured-HBM CM2 load-only target_us reproduction harness |

## Acceptance Contract

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

## Bandwidth Target Reference

The `target_5pct_us` values are reproduced with the archived bandwidth harness:

```bash
cd reference-kernels/amd/cdna3/flydsl/FlyDSL/moe_fp8_ptpc_mi308x

${PYTHON:-python3} - <<'PY'
import flydsl
PY

${PYTHON:-python3} bandwidth_reference.py \
  --out task66_bandwidth_reference.csv \
  --markdown-out task66_bandwidth_reference.md
```

The script does not inject machine-local paths and does not depend on
`tensor_tools`; the minimal `flush_cache` and `profile_cuda_kernels` helpers are
inlined in `bandwidth_reference.py`. `flydsl` must be importable from the active
Python environment, or from a `PYTHONPATH` set by the caller before running the
command.

Canonical settings:

```text
bytes: measured_hbm_bytes from task64/task66
mode: loadonly
cache_modifier: 2
tiles_per_thread: 4
warmup: 10
iters: 50
cache state: cold, via inlined flush_cache
timing: inlined profile_cuda_kernels -> bw_kernel device time
target: target_us = 1.05 * measured_ref_us
```

`cache_modifier=0`, torch event timing, torch copy timing, and theoretical peak bandwidth are diagnostic only and must not be used as task66 target_us sources.

## Current Status

```text
full-scope gate pass: 1/14
stage1 gate pass: 1/7
stage2 gate pass: 0/7
retained source state: task64 promoted source + task65 documentation-only negative probes
```

See the optimization checkpoint for the full table:

- [cdna3-fused-moe-fp8-ptpc-pause-checkpoint.md](../../../../../../docs/ref-docs/amd/flydsl/gfx942/cdna3-fused-moe-fp8-ptpc-pause-checkpoint.md)

The later atrex-open integrated v2 full-pipeline archive is a sibling package,
not a replacement for this checkpoint:

- [moe_fp8_ptpc_mi308x_atrex_v2/](../moe_fp8_ptpc_mi308x_atrex_v2/)
- [cdna3-fused-moe-fp8-ptpc-atrex-v2.md](../../../../../../docs/ref-docs/amd/flydsl/gfx942/cdna3-fused-moe-fp8-ptpc-atrex-v2.md)

Pitfalls and invalid evidence boundaries:

- [fused-moe-fp8-ptpc-pitfalls.md](../../../../../../docs/pitfalls/amd/flydsl/fused-moe-fp8-ptpc-pitfalls.md)

## Provenance

Source of truth:

- omoExplore `proj/proj_007_mi308x_fused_moe_opt/tasks/task_66_fp8_ptpc_pause_summary_and_todo.md`
- omoExplore `proj/proj_007_mi308x_fused_moe_opt/assets/task_66_fp8_ptpc_pause_summary/task66_current_gap_summary.csv`

This package intentionally does not replace the generic CDNA MoE reference:

- [`../../../../cdna/flydsl/FlyDSL/moe_gemm_2stage.py`](../../../../cdna/flydsl/FlyDSL/moe_gemm_2stage.py)
