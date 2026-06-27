# MI308X FlyDSL FP8 PTPC Fused MoE atrex-open v2

This package archives the atrex-open integrated FlyDSL v2 FP8 PTPC `fused_moe`
path as a standalone gpu-wiki reference package for AMD MI308X (CDNA3 / gfx942).

It is intentionally a sibling of `../moe_fp8_ptpc_mi308x/`. The sibling package
preserves the older proj007 task66 isolated stage checkpoint and bandwidth
`target_us` method. This package preserves the atrex-open full-pipeline v2
behavior after proj009 task07/task09 integration and trace parity work.

## Scope

```text
hardware: AMD MI308X / CDNA3 / gfx942
framework: FlyDSL
kernel: FP8 PTPC fused_moe v2 full pipeline
shape: E=512, topk=10, model_dim=4096, inter_dim=256
tokens: 1/16/32/64/128/256/512
dtype: FP8 GEMM inputs, BF16 stage/output activations, F32 scales, I32 metadata
```

The package supports only the archived v2 task16 shape. Unsupported shapes are
rejected rather than silently falling back to a different local implementation.

## Files

| File | Purpose |
|---|---|
| `fused_moe_flydsl_fp8.py` | atrex-open v2 full-pipeline wrapper adapted to standalone gpu-wiki imports |
| `moe_kernels.py` | FlyDSL kernel registry, compile dispatch, and stage1/stage2 launch wrappers |
| `kernels/moe_gemm_2stage.py` | Stage1 and stage2 FlyDSL MFMA FP8 kernel builders |
| `kernels/mfma_epilogues.py` | Shared MFMA epilogue helpers |
| `kernels/mfma_preshuffle_pipeline.py` | Shared preshuffle, LDS, and B-tile load helpers |
| `test_fused_moe_atrex_v2.py` | Correctness and profile-parity harness for the archived v2 package |

## Reproduction

Correctness:

```bash
cd reference-kernels/amd/cdna3/flydsl/FlyDSL/moe_fp8_ptpc_mi308x_atrex_v2
PYTHONPATH=..:${AITER_BASE} \
"${PYTHON:-python3}" -m pytest -sv \
  test_fused_moe_atrex_v2.py::test_fused_moe_flydsl_v2_correctness
```

Profile parity:

```bash
cd reference-kernels/amd/cdna3/flydsl/FlyDSL/moe_fp8_ptpc_mi308x_atrex_v2
PYTHONPATH=..:${AITER_BASE} \
"${PYTHON:-python3}" -m pytest -sv \
  test_fused_moe_atrex_v2.py::test_profile_fp8_ptpc_flydsl_v2_vs_aiter
```

Validated on 2026-05-27:

```text
correctness: 7 passed in 8.08s
profile parity: 7 passed in 7.32s
```

Correctness follows the current atrex-open task16 gate: no NaN, same output
shape, and `checkAllclose(..., rtol=1e-2, atol=1e-2)` error ratio within the
atrex-open v2 tolerance. This is not the same gate as the older task66 isolated
stage `max_delta <= 0.01` checkpoint.

## Provenance

Source of truth:

```text
atrex-open path: ${ATREX_OPEN_ROOT}
branch: feature/fused-moe-ptpc
HEAD: c917d1e12f7a8eaf49e3e6f0453dc025173a5239
dirty tracked source: src/triton/fused_moe/fused_moe_flydsl_fp8.py
```

The dirty tracked source only updates the M=1 stream-fence comment and documents
why the stream packet boundary preserves the old `.item()` timing boundary
without host sync or profiler-visible memcpy.

Dependency chain:

```text
proj009 task07: integrated task66 FlyDSL v2 into atrex-open
proj009 task09: v2 trace parity against AITER routing/profile path
proj011 task09: archived task66 pause checkpoint to gpu-wiki
proj011 task10: archived task66 bandwidth target_us reproduction method
proj011 task11: archived this atrex-open v2 full-pipeline parity package
```

## atrex-open Baseline

Baseline command:

```bash
cd ${ATREX_OPEN_ROOT}/op_test
PYTHONPATH=${ATREX_OPEN_ROOT}/python:${ATREX_OPEN_ROOT} \
"${PYTHON:-python3}" -m pytest -sv \
  test_fused_moe.py::test_profile_fp8_ptpc_flydsl_vs_aiter
```

Result: `7 passed in 13.26s`.

| M | routing | quant | stage1 | stage2 | overhead | other | kernel sum | e2e avg | e2e min |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17.7 | 5.1 | 11.6 | 7.9 | 1.5 | 0.0 | 43.9 | 544.9 | 528.9 |
| 16 | 12.2 | 8.3 | 91.4 | 54.6 | 2.4 | 0.0 | 169.0 | 601.0 | 575.2 |
| 32 | 12.0 | 8.7 | 120.9 | 85.3 | 4.3 | 0.0 | 231.2 | 587.6 | 563.8 |
| 64 | 13.1 | 8.4 | 180.5 | 128.6 | 4.6 | 0.0 | 335.2 | 663.4 | 620.2 |
| 128 | 13.9 | 9.3 | 244.4 | 170.6 | 5.2 | 0.0 | 443.4 | 735.1 | 714.8 |
| 256 | 15.8 | 14.4 | 275.1 | 185.5 | 5.7 | 0.0 | 496.4 | 796.2 | 782.6 |
| 512 | 20.1 | 23.5 | 334.2 | 200.0 | 5.9 | 0.0 | 583.8 | 938.0 | 925.4 |

Units are microseconds.

## gpu-wiki Profile

| M | routing | quant | stage1 | stage2 | overhead | other | kernel sum | e2e avg | e2e min |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 18.1 | 5.5 | 11.7 | 8.4 | 1.6 | 0.0 | 45.4 | 530.4 | 512.5 |
| 16 | 12.9 | 8.3 | 85.8 | 48.9 | 2.3 | 0.0 | 158.2 | 557.5 | 539.1 |
| 32 | 12.8 | 8.7 | 119.0 | 80.5 | 4.2 | 0.0 | 225.3 | 564.5 | 544.9 |
| 64 | 13.1 | 8.9 | 179.5 | 116.0 | 4.4 | 0.0 | 321.8 | 602.3 | 587.8 |
| 128 | 14.2 | 9.5 | 238.1 | 152.3 | 5.2 | 0.0 | 419.3 | 699.4 | 681.6 |
| 256 | 15.8 | 12.1 | 262.2 | 167.1 | 5.4 | 0.0 | 462.6 | 738.6 | 726.6 |
| 512 | 19.7 | 18.1 | 315.7 | 189.7 | 5.8 | 0.0 | 549.1 | 880.1 | 864.4 |

## Parity Result

Acceptance compares this gpu-wiki package against the saved atrex-open
full-pipeline baseline, not against the task66 isolated stage `target_us` table.

| M | kernel sum ratio | e2e avg ratio | stage1 ratio | stage2 ratio | e2e limit | pass |
|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 1.034x | 0.973x | 1.009x | 1.063x | 566.2 | yes |
| 16 | 0.936x | 0.928x | 0.939x | 0.896x | 624.0 | yes |
| 32 | 0.974x | 0.961x | 0.984x | 0.944x | 610.2 | yes |
| 64 | 0.960x | 0.908x | 0.994x | 0.902x | 688.3 | yes |
| 128 | 0.946x | 0.951x | 0.974x | 0.893x | 762.2 | yes |
| 256 | 0.932x | 0.928x | 0.953x | 0.901x | 825.1 | yes |
| 512 | 0.941x | 0.938x | 0.945x | 0.948x | 971.1 | yes |

Trace guards:

- No profiler-visible `Memcpy DtoD`, `Memcpy DtoH`, or `Memcpy HtoD` on the v2 path.
- `other == 0.0` for all formal token rows.
- Routing remains AITER-relative and uses the AITER opus sorting path.
- M=1 keeps the stream packet boundary without host sync.

M=1 e2e event timing can jitter. One full run saw `622.0us` e2e average for M=1
and failed the e2e guard, while the kernel trace and kernel sum were aligned.
The immediate M=1 rerun was `526.2us`, and the final full rerun passed at
`530.4us`. Use kernel sum, e2e min, and trace classification to distinguish
pipeline regressions from small-row event jitter.

## Related Docs

- [atrex-open v2 full-pipeline report](../../../../../../docs/ref-docs/amd/flydsl/gfx942/cdna3-fused-moe-fp8-ptpc-atrex-v2.md)
- [task66 pause checkpoint](../../../../../../docs/ref-docs/amd/flydsl/gfx942/cdna3-fused-moe-fp8-ptpc-pause-checkpoint.md)
- [FP8 PTPC Fused MoE pitfalls](../../../../../../docs/pitfalls/amd/flydsl/fused-moe-fp8-ptpc-pitfalls.md)
