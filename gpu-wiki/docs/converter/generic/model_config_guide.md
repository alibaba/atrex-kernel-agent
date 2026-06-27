# Model Capability Selection Reference

This page summarizes how to match model capability to GPU-kernel conversion and optimization work. It is knowledge for planning review depth, implementation risk, and validation effort; it is not tied to any particular tool configuration.

## Capability Tiers

| Tier | Suitable Work | Risk Profile |
|---|---|---|
| Fast review | Small doc edits, spelling fixes, simple link checks, one-line code comments | Low risk when the change is local and mechanically verifiable |
| Standard implementation | Straightforward PyTorch to Triton ports, API mapping lookups, simple elementwise kernels, routine benchmark interpretation | Moderate risk; needs syntax, accuracy, and performance validation |
| Deep reasoning | Complex fusion, reductions with numerical stability concerns, layout-heavy Gluon code, multi-stage pipelines, attention kernels | High risk; needs explicit invariants, profile evidence, and comparison against a tuned baseline |
| Specialist review | Hardware-specific ISA/resource diagnosis, roofline analysis, bank-conflict analysis, cross-backend migration | High risk; requires domain-specific evidence such as profiler counters, assembly/resource snapshots, and shape sweeps |

## Complexity Signals

Use a deeper reasoning tier when any of these signals appear:

- The implementation changes memory layout, shared-memory layout, or thread/block mapping.
- The kernel contains reductions, online softmax, recurrent state, or causal masking.
- Correctness depends on numerical stability tricks such as max-subtraction, compensated accumulation, or precision-specific casts.
- Performance depends on hardware-specific instructions such as MFMA, WGMMA, async copy, TMA, or warp/wave specialization.
- The benchmark target is close to a hardware ceiling, so small regressions matter.
- Multiple shapes have different bottlenecks, especially low-grid small-matrix cases.

Fast review is usually enough when the change is limited to prose, links, tables, or a small local code cleanup with no semantic effect.

## Review Perspectives

Different reviews should emphasize different evidence:

| Perspective | Primary Questions | Useful Evidence |
|---|---|---|
| API mapping | Does each source operation map to the correct backend primitive? | Local API tables, backend reference docs, syntax checks |
| Correctness | Does the optimized implementation preserve semantics over representative shapes and dtypes? | Accuracy tests, max-diff summaries, edge-case shape coverage |
| Performance | Does the result improve the right metric for the bottleneck? | Latency, TFLOPS, bandwidth utilization, roofline position |
| Hardware resources | Are occupancy, LDS/shared memory, registers, scratch, and barriers acceptable? | Profiler counters, compiler metadata, ISA snapshots |
| Maintainability | Is the implementation explainable and safe to adapt? | Clear layout comments, minimal special cases, named assumptions |

## Validation Depth

Validation depth should scale with risk:

| Change Type | Minimum Validation |
|---|---|
| Documentation-only changes | Link/path scan and formatting check |
| Simple source transformation | Syntax check, accuracy validation, one representative benchmark |
| Layout or tiling change | Accuracy validation across edge shapes, benchmark across representative shapes, resource check |
| Pipeline or memory hierarchy change | Accuracy validation, benchmark sweep, profiler counters, resource/ISA inspection |
| New fused kernel | Baseline comparison, multi-shape correctness, profiler evidence, rejection log for attempted alternatives |

## Evidence Quality

Prefer evidence that can be reproduced from local references and local benchmark harnesses:

- State the baseline and fusion boundary clearly.
- Use the same shape, dtype, and backend when comparing variants.
- Separate correctness references from performance baselines when they are different implementations.
- Record rejected variants with the measured reason for rejection.
- Preserve enough metadata for later readers to understand hardware scope, backend scope, and applicable shape ranges.

## Practical Selection Rules

1. Use the lightest capability tier that can still explain and verify the change.
2. Increase review depth when the implementation touches layout, synchronization, memory hierarchy, or numerical stability.
3. Treat profiler and benchmark evidence as mandatory for performance claims.
4. Treat model output as a hypothesis until local validation confirms correctness and performance.
