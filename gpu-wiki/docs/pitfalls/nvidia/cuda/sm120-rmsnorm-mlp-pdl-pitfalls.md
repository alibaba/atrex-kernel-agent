# SM120 RMSNorm-MLP PDL Pitfalls

Pitfalls from SM120 RMSNorm, NVFP4 input quantization, C1 GEMM handoff, and
Programmatic Dependent Launch work.

Related:

- Optimization report:
  [sm120-rmsnorm-mlp-pdl-fusion-report.md](../../../ref-docs/nvidia/cuda/sm120/sm120-rmsnorm-mlp-pdl-fusion-report.md)

## 1. Treating PDL as tile-level fusion

**Trap**: Enable Programmatic Dependent Launch and expect the consumer GEMM to
overlap with producer quantization by tile.

**Result**: Whole-A handoff is correctness-safe but only releases C1 after the
complete A payload and scale are ready. It does not overlap C1 mainloop with
row or tile production.

**Why**: PDL controls launch/dependency timing between kernels. It does not
change the stock C1 mainloop into a producer/consumer pipeline.

**Lesson**: Use PDL only when the dependency boundary actually exposes useful
overlap. Otherwise it is a scheduling mechanism with its own overhead.

## 2. Launching stock C1 early without a device-side wait

**Trap**: Launch the secondary CUTLASS C1 kernel early on the same stream and
assume stream ordering or programmatic serialization protects A/input-scale
reads.

**Result**: C1 can race `input_fp4` and `input_sf` unless the kernel waits before
loading them.

**Why**: The dependent launch can start before the producer grid fully
terminates. That is the point of PDL.

**Lesson**: A safely early C1 needs `cudaGridDependencySynchronize()` or an
equivalent device-side wait before A/input-scale loads.

## 3. Promoting a component-level PDL win after served regression

**Trap**: A component harness is a few microseconds faster, so enable the PDL
route by default.

**Result**: Served streaming TTFT regressed even though the component timing was
slightly positive.

**Why**: CUDA Graph replay already reduces host launch gaps. The remaining PDL
fence, trigger, dependency, and graph scheduling effects can exceed the
component-level saving.

**Lesson**: Component timing is not enough. Use served TTFT with order control
before promotion, especially when the component win is tiny.

## 4. Running served TTFT after an operator-negative row pipeline

**Trap**: A row-chunk pipeline is bit-exact, so run the expensive served gate
anyway.

**Result**: Component medians were already slower across the tested M range.
Served testing would only measure a known-negative path with more noise.

**Why**: Correctness and performance gates are separate. Launch repetition,
scheduler setup, tile inefficiency, synchronization, and output copies can make
a correct row pipeline slower.

**Lesson**: Do not run served TTFT while the clean component gate is negative.

## 5. Assuming smaller chunks mean more overlap and better performance

**Trap**: Reduce row-chunk size to expose finer-grained overlap.

**Result**: Finer chunks were much slower because each chunk repeats GEMM launch
and scheduler overhead and reduces tile utilization.

**Why**: Overlap granularity has a fixed cost. If the consumer GEMM is not
chunk-aware, smaller chunks can multiply overhead faster than they expose
overlap.

**Lesson**: Chunking needs a scheduler/dataflow designed for chunks. It is not
just a dependency knob.

## 6. Over-attributing the wait-cache bottleneck to repeated acquire loads

**Trap**: Cache repeated row-ready waits and expect the route to become neutral
or faster.

**Result**: The cache recovered only about `1.1 us/layer`; the paired operator
delta remained slower.

**Why**: CTAs still occupy SM resources while waiting on unready chunks, and
the publish/wait protocol itself adds overhead.

**Lesson**: Wait-cache is a useful diagnostic, not a full solution. A profitable
route needs row-aware scheduling or a different mainloop.

## 7. Calling a sign-flipping TTFT matrix a stable win

**Trap**: One prompt length or one launch order shows a small no-PDL fusion win.

**Result**: Strict two-order averages changed sign by prompt length, with
absolute deltas below a few tenths of a millisecond.

**Why**: The saved traffic is small relative to full served TTFT, and order
noise is comparable to the effect size.

**Lesson**: If strict order-controlled TTFT flips sign, keep the route behind
an experiment flag and do not default-enable it.

## Quick Table

| Use | Avoid |
|---|---|
| Device-side wait before early C1 input reads | early stock C1 with no wait |
| Component gate before served gate | served TTFT for operator-negative paths |
| Larger fusion/dataflow changes | small PDL placement tweaks after no-go |
| Order-controlled prompt matrix | one-order TTFT claims |
| Row-aware C1 scheduler or custom mainloop | repeated chunked CUTLASS launches |
