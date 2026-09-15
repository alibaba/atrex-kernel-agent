---
name: ncu-report-skill
description: Interpret NVIDIA Nsight Compute evidence to distinguish launch, memory, compute, synchronization, and imbalance bottlenecks. Use when existing Profile results leave a concrete optimization question, or when analyzing an available raw NCU report.
allowed-tools: "Bash Read"
---

# Nsight Compute evidence

Start with the saved Profile result for the exact Kernel and launch of interest. Profile is optional;
collect additional evidence only when a missing fact could change the next optimization decision.
The injected product, architecture, DSL, and operator contract determine applicability. This Skill
does not prescribe a fixed GPU, profiling quota, standalone harness, or separate report document.

## Read the relevant evidence

- [Diagnosis](references/diagnosis.md): distinguish competing bottleneck hypotheses, including
  occupancy, tails, stalls, tensor utilization, memory traffic, and synchronization.
- [Metric interpretation](references/metrics.md): units, denominators, absent counters, and useful
  metric families. B200-style names are examples, not a portable mandatory collection set.
- [Raw report analysis](references/report-analysis.md): when normalized results are insufficient,
  use the four included analysis helpers on an already available `.ncu-rep`.

GPU requests and their parameters belong to `skills/gpu-measurement/SKILL.md`; saved-result lookup,
source recovery, Journal updates, and terminal reports belong to `skills/runtime-records/SKILL.md`.
Use those interfaces rather than managing profiling jobs, retries, history, or acceptance here.

Distinguish the measured fact from its interpretation. A large counter or NCU estimated speedup is
not proof of a bottleneck or a promised end-to-end gain. State the competing explanations and the
smallest useful check. Verify changes with the prescribed probe-free evaluator; profiler duration
and instrumented timing do not replace its correctness/performance result.
