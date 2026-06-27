# SM120 RMSNorm-MLP NVFP4 Fusion and PDL Handoff Report

This report summarizes the reusable findings from SM120 RMSNorm, NVFP4 input
quantization, parent MLP, and Programmatic Dependent Launch experiments. It is
written as an English knowledge report and omits project task numbering.

## Scope

- Hardware: RTX PRO 5000 / 4000 Blackwell-GeForce, `sm_120`.
- Runtime: vLLM serving with CUDA Graph replay.
- Boundary:

```text
residual add -> RMSNorm -> MLP input NVFP4 quant -> parent MLP C1 GEMM
```

The goal was to remove or hide handoff traffic before C1. Several routes were
tested: no-PDL fused boundary, whole-A PDL handoff, row-chunk dependent
pipeline, and row-ready C1 wait-cache.

## Programmatic Dependent Launch Facts

CUDA 13 headers expose the relevant device-side APIs on the local SM120 stack:

```cuda
cudaTriggerProgrammaticLaunchCompletion();
cudaGridDependencySynchronize();
```

PDL is an inter-kernel dependency/overlap mechanism. It is not TMEM, not a
single-kernel register/shared-memory pipeline, and not a substitute for a
custom C1 mainloop. If a secondary C1 kernel is launched early, it must wait
before reading `input_fp4` and `input_sf`; otherwise it can race the producer.

## Route Summary

| Route | Correctness | Performance outcome | Decision |
|---|---|---|---|
| No-PDL fused RMSNorm + input quant | Prompt and semantic gates passed | Strict served TTFT signs flipped within noise | Keep as experiment flag only |
| Whole-A PDL handoff | Correctness-safe | Tiny component win, served TTFT regression | Do not default-enable |
| Row-chunk dependent pipeline | Bit-exact | Component timing regressed for small and large M | Do not run served gate |
| C1 row-ready wait-cache | Correctness-clean | Recovered about `1.1 us/layer` but remained operator-negative | Do not run served gate |

## No-PDL Boundary Fusion

The no-PDL route fuses residual/RMSNorm and input NVFP4 quantization before the
parent MLP path. It is correctness-safe, but strict served TTFT did not prove a
stable win. Across prompt lengths, two-order averages stayed below about
`0.3 ms` absolute and changed sign.

The useful interpretation is narrow: the boundary is a safe experiment surface
and a source-map for larger fusion, not a default optimization.

## Whole-A PDL Handoff

Whole-A PDL releases the dependent C1 launch only after the full A payload and
scale have been produced. This is safe, but it does not create tile-level
producer/consumer overlap.

Representative component timing showed small wins at some M values, including
about `10-12 us` at smaller M and only about `1.5 us` at a larger M. Served
streaming TTFT regressed from roughly `260.5 ms` to `262.7 ms`. Under CUDA
Graph, ordinary host launch overhead is already reduced, so the PDL dependency,
producer-side fence/trigger, and scheduling disturbance can erase the component
gain.

## Row-Chunk PDL Pipeline

Row-chunk PDL tries to overlap quantization for one row chunk with C1/C2 work
for another chunk. It was bit-exact but slower. Representative paired medians
regressed by about `23 us`, `11 us`, and `121 us` across small, medium, and
large M cases.

The overhead sources were:

- repeated C1/C2 GEMM launches;
- repeated CUTLASS scheduler/workspace setup;
- lower tile utilization for chunked M;
- extra chunk-output copies;
- synchronization needed for safe temporary storage.

Even chunks that covered the whole M dimension stayed slower, so the issue was
not only chunk granularity.

## C1 Wait-Cache

The row-ready C1 route publishes per-row or per-chunk readiness and makes C1
wait before loading A/input scale. A cached wait avoids repeated waits for
consecutive N tiles covering the same M chunk.

The cache was real but insufficient: paired median delta improved from about
`+9.4 us` to `+8.2 us`, where positive means slower than baseline. The
remaining bottleneck is not just repeated acquire-loads. C1 CTAs can still
occupy SM resources while waiting on unready chunks, and the publication/wait
protocol itself costs time.

## Practical Boundary

Do not promote these PDL routes without a non-negative operator gate on clean
repeated runs. In particular:

- Do not run served TTFT when component timing is already negative.
- Do not rely on whole-A PDL for tile-level overlap.
- Do not launch stock C1 early unless it performs a device-side dependency
  wait before A/input-scale loads.
- Do not call a sub-millisecond served delta a stable optimization unless
  order-controlled prompt matrices keep the same sign.

## Plausible Future Directions

The remaining viable routes require a different C1 dataflow:

- a custom C1 prologue/mainloop that can launch early, do B-independent work,
  and wait immediately before A/input-scale load;
- a row-chunk-aware scheduler that avoids occupying SMs with CTAs spinning on
  unready chunks;
- a single-kernel register/shared-memory publish queue that avoids global
  handoff traffic.

These are new kernel/dataflow problems, not small PDL placement tweaks.

## Related

- Pitfalls:
  [sm120-rmsnorm-mlp-pdl-pitfalls.md](../../../../pitfalls/nvidia/cuda/sm120-rmsnorm-mlp-pdl-pitfalls.md)
- General Blackwell PDL card:
  [pdl-gdc.md](../../../../kernel-opt/nvidia/common/blackwell/hardware/pdl-gdc.md)
- Triton RMSNorm/GDN post-processing report:
  [sm120-fused-rmsnorm-gated-bf16-optimization.md](../../triton/sm120/sm120-fused-rmsnorm-gated-bf16-optimization.md)
