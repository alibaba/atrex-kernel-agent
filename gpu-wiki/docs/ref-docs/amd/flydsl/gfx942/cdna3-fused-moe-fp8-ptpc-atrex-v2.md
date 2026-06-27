# FlyDSL Fused MoE FP8 PTPC atrex-open v2 on MI308X

Applicability: backend: flydsl; hardware: amd; topic: reference

This document archives the atrex-open integrated FlyDSL v2 FP8 PTPC `fused_moe`
full pipeline on AMD MI308X (CDNA3 / gfx942). The reference package is:

- [`reference-kernels/amd/cdna3/flydsl/FlyDSL/moe_fp8_ptpc_mi308x_atrex_v2/`](../../../../../reference-kernels/amd/cdna3/flydsl/FlyDSL/moe_fp8_ptpc_mi308x_atrex_v2)

Related context:

- [`cdna3-fused-moe-fp8-ptpc-pause-checkpoint.md`](cdna3-fused-moe-fp8-ptpc-pause-checkpoint.md) archives the older proj007 task66 isolated stage checkpoint and bandwidth gate.
- [`docs/pitfalls/amd/flydsl/fused-moe-fp8-ptpc-pitfalls.md`](../../../../pitfalls/amd/flydsl/fused-moe-fp8-ptpc-pitfalls.md) records task66 and atrex-open v2 migration traps.

## Scope

```text
hardware: AMD MI308X / CDNA3 / gfx942
framework: FlyDSL
kernel: FP8 PTPC fused_moe v2 full pipeline
shape: E=512, topk=10, model_dim=4096, inter_dim=256
tokens: 1/16/32/64/128/256/512
dtype: FP8 GEMM inputs, BF16 stage/output activations, F32 scales, I32 metadata
```

This is not a new Roofline analysis and does not introduce new hardware-spec
derived targets. The acceptance target is direct same-machine profile parity
against atrex-open v2.

## Provenance

```text
atrex-open path: $ATREX_OPEN_ROOT
branch: feature/fused-moe-ptpc
HEAD: c917d1e12f7a8eaf49e3e6f0453dc025173a5239
dirty tracked source: src/triton/fused_moe/fused_moe_flydsl_fp8.py
gpu-wiki path: current gpu-wiki checkout root
gpu-wiki branch: sumu/fused-moe-308x
gpu-wiki base HEAD before task11: a61ec34c28d20eaed97c521e83c5dab69967ebef
```

The `src/triton/...` provenance path is inherited from the historical
repository layout; the archived file is the FlyDSL FP8 source, not an
instruction to implement the kernel in Triton.

The tracked dirty atrex-open source change updates the M=1 stream-fence comment
and documents the packet-boundary intent. It does not change the profiler-visible
kernel list.

Dependency chain:

| Project | Task | Role |
|---|---|---|
| proj009 | task07 | Integrated the task66 FlyDSL v2 source into atrex-open |
| proj009 | task09 | Brought the atrex-open v2 profile trace into parity with AITER routing/trace expectations |
| proj011 | task09 | Archived the task66 pause checkpoint package and report |
| proj011 | task10 | Archived the task66 bandwidth `target_us` reproduction method |
| proj011 | task11 | Archived the atrex-open v2 full-pipeline package and parity evidence |

## Archived Validation Context

The following commands record how this historical reference package was checked
when the archive was produced. They are provenance for the local reference
material, not instructions for running this wiki page.

atrex-open baseline:

```bash
cd "$ATREX_OPEN_ROOT/op_test"
PYTHONPATH="$ATREX_OPEN_ROOT/python:$ATREX_OPEN_ROOT" \
${PYTHON:-python3} -m pytest -sv \
  test_fused_moe.py::test_profile_fp8_ptpc_flydsl_vs_aiter
```

gpu-wiki correctness:

```bash
cd reference-kernels/amd/cdna3/flydsl/FlyDSL/moe_fp8_ptpc_mi308x_atrex_v2
PYTHONPATH="..:$AITER_BASE" \
${PYTHON:-python3} -m pytest -sv \
  test_fused_moe_atrex_v2.py::test_fused_moe_flydsl_v2_correctness
```

gpu-wiki profile parity:

```bash
cd reference-kernels/amd/cdna3/flydsl/FlyDSL/moe_fp8_ptpc_mi308x_atrex_v2
PYTHONPATH="..:$AITER_BASE" \
${PYTHON:-python3} -m pytest -sv \
  test_fused_moe_atrex_v2.py::test_profile_fp8_ptpc_flydsl_v2_vs_aiter
```

Raw logs were stored with the historical archive:

```text
proj/proj_011_gpu_wiki_kernel_archive/assets/task_11_archive_atrex_open_flydsl_v2_perf_parity/atrex_open_baseline_profile.log
proj/proj_011_gpu_wiki_kernel_archive/assets/task_11_archive_atrex_open_flydsl_v2_perf_parity/gpu_wiki_correctness.log
proj/proj_011_gpu_wiki_kernel_archive/assets/task_11_archive_atrex_open_flydsl_v2_perf_parity/gpu_wiki_profile.log
```

## Validation Summary

```text
atrex-open baseline: 7 passed in 13.26s
gpu-wiki correctness: 7 passed in 8.08s
gpu-wiki profile parity: 7 passed in 7.32s
```

Correctness followed the archived atrex-open task16 FP8 PTPC tolerance:
`checkAllclose(..., rtol=1e-2, atol=1e-2)` error ratio must remain within the
atrex-open v2 correctness threshold, with no NaN and matching output shape. This
is distinct from the older task66 isolated stage `max_delta <= 0.01` gate.

## atrex-open Baseline

Full-pipeline profile from atrex-open v2:

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

## gpu-wiki Migrated Profile

Full-pipeline profile from the standalone gpu-wiki package:

| M | routing | quant | stage1 | stage2 | overhead | other | kernel sum | e2e avg | e2e min |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 18.1 | 5.5 | 11.7 | 8.4 | 1.6 | 0.0 | 45.4 | 530.4 | 512.5 |
| 16 | 12.9 | 8.3 | 85.8 | 48.9 | 2.3 | 0.0 | 158.2 | 557.5 | 539.1 |
| 32 | 12.8 | 8.7 | 119.0 | 80.5 | 4.2 | 0.0 | 225.3 | 564.5 | 544.9 |
| 64 | 13.1 | 8.9 | 179.5 | 116.0 | 4.4 | 0.0 | 321.8 | 602.3 | 587.8 |
| 128 | 14.2 | 9.5 | 238.1 | 152.3 | 5.2 | 0.0 | 419.3 | 699.4 | 681.6 |
| 256 | 15.8 | 12.1 | 262.2 | 167.1 | 5.4 | 0.0 | 462.6 | 738.6 | 726.6 |
| 512 | 19.7 | 18.1 | 315.7 | 189.7 | 5.8 | 0.0 | 549.1 | 880.1 | 864.4 |

## Parity Gate

For each token row:

```text
e2e_avg <= atrex_open_e2e_avg * 1.03 + 5us
stage1 <= atrex_open_stage1 * 1.03 + slack
stage2 <= atrex_open_stage2 * 1.03 + slack
slack = 6us if the atrex-open stage baseline is below 20us, otherwise 2us
```

Trace guard:

```text
no profiler-visible Memcpy DtoD / DtoH / HtoD
other == 0.0
routing <= AITER routing * 1.15 + 2us
```

Result:

| M | kernel sum ratio | e2e avg ratio | stage1 ratio | stage2 ratio | e2e limit | pass |
|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 1.034x | 0.973x | 1.009x | 1.063x | 566.2 | yes |
| 16 | 0.936x | 0.928x | 0.939x | 0.896x | 624.0 | yes |
| 32 | 0.974x | 0.961x | 0.984x | 0.944x | 610.2 | yes |
| 64 | 0.960x | 0.908x | 0.994x | 0.902x | 688.3 | yes |
| 128 | 0.946x | 0.951x | 0.974x | 0.893x | 762.2 | yes |
| 256 | 0.932x | 0.928x | 0.953x | 0.901x | 825.1 | yes |
| 512 | 0.941x | 0.938x | 0.945x | 0.948x | 971.1 | yes |

## Migration Details

The standalone package keeps the v2 full-pipeline path but changes imports:

```python
from aiter import ActivationType
from aiter.fused_moe import moe_sorting
from . import moe_kernels as flydsl_v2_kernels
```

`moe_kernels.py` keeps the gpu-wiki standalone dtype provider:

```python
from aiter.utility import dtypes
```

The package keeps AITER routing and quantization dependencies because those are
part of the atrex-open v2 trace-parity contract. It archives the full path, not
only isolated FlyDSL GEMM kernels.

## Distinctions From Adjacent Archives

Do not compare these rows directly to task66 `target_us` rows:

- Task66 is an isolated stage-level checkpoint and bandwidth gate.
- Task10 archives how task66 `target_us` was reproduced with a same-byte FlyDSL
  load-only proxy.
- This task compares full fused_moe pipeline latency, including routing,
  quantization, fill/finalize overhead, and e2e event timing.

The same kernel core appears in both packages, but the validation question is
different. This package answers: "Can the atrex-open v2 full-pipeline behavior be
reproduced from gpu-wiki at parity?"

## M=1 Event Jitter

M=1 has a tiny stage1 launch and can show event-average jitter. During task11,
one full gpu-wiki run reported `622.0us` e2e average for M=1 and failed the e2e
guard, while the classified kernel sum and trace were aligned. The immediate M=1
rerun reported `526.2us`, and the final full rerun reported `530.4us`.

For M=1 regression diagnosis, check in this order:

1. Trace has no memcpy and `other == 0.0`.
2. Stage1/stage2 classified sums remain within the atrex-open limits.
3. e2e min remains aligned.
4. e2e average is rerun only after the trace and kernel sums are clean.

## Review Checklist

- Keep `moe_fp8_ptpc_mi308x/` as the task66 checkpoint package.
- Keep `moe_fp8_ptpc_mi308x_atrex_v2/` as the atrex-open full-pipeline package.
- Do not reintroduce host-sync `.item()` reads on the profile path.
- Preserve the M=1 stream packet boundary without host sync or profiler-visible memcpy.
- Keep the AITER opus sorting trace contract for routing parity.
