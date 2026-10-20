---
name: micro-analysis
description: Analyze GPU microarchitecture questions across vendors using focused microkernels, ISA inspection, and controlled A/B experiments when compiler behavior, layout, dependencies, or performance are unclear.
---

# MicroAnalysis

Use this methodology for GPU kernels across vendors and architecture generations. Choose the smallest experiment that distinguishes plausible explanations for the current problem, then revise the hypothesis from the result. The Agent chooses experiments and optimization decisions; reuse existing evidence while its conditions remain applicable.

## Unfamiliar operators and hardware

Establish the operator's semantics, accuracy requirements, representative inputs, and expected work, dataflow, and reuse. Identify the target device, architecture, and compiler/runtime. Treat remembered hardware details as hypotheses until their applicability is established; lane widths, instruction behavior, synchronization, resource limits, and counter meanings can differ across targets.

Obtain hardware knowledge through the existing GPUWiki lookup step, checking the source and target applicability of relevant facts, metrics, and tool guidance. If knowledge is missing, outdated, or unavailable, consult target-vendor documentation, SDK examples, or established implementations through available retrieval tools. This Skill does not maintain hardware tables or collection recipes.

When knowledge still cannot explain the observation, isolate an unknown that changes an implementation choice. Use a small compile or functional probe to establish behavior, a microkernel to measure dependency cost or effective throughput, or an A/B comparison to distinguish implementation effects. Keep unresolved facts explicit: a successful timing run does not establish undocumented synchronization semantics. Record measurements with their input, device, toolchain, and timing scope; distinguish them from specified limits and metric definitions. If the required device or toolchain is unavailable, preserve a runnable experiment and mark the conclusion unmeasured.

## Choose a method for the question

| Question | Useful investigation |
| --- | --- |
| What work or overhead dominates? | Check effective inputs, reuse, task mapping, and timing scope. Separate kernel work from preparation, launch, and combine overhead. |
| What does an instruction or source expression actually do? | Find a valid candidate in target SDK documentation or implementations; inspect actual lowering and verify semantics with a functional probe. |
| Is compute throughput limited by dependencies? | Trace real producer-to-consumer dependencies in the target loop. Compare serial and independent chains, or real consumers and pre-ready inputs, as needed. |
| Is data supply or fragment mapping wrong? | Use element or byte tags to resolve lane, address, matrix orientation, and swizzle mappings. Hold the consumer fixed when comparing valid layouts or load distance. |
| Is synchronization or handoff expensive? | Trace the target's transfer, completion, and consumer protocol. Compare valid protocols or use explicitly ready inputs to isolate waits, uneven arrival, and extra dependencies. |
| Do resource use and parallelism cancel the gain? | Compare actual registers, on-chip storage, spills, task count, and residency conditions. Where useful, control storage capacity separately to distinguish a layout benefit from its residency cost. |

## Validate and interpret

Use these points to guide experiments and judgment; no point-by-point write-up or long report is required. Briefly note relevant observations or working hypotheses and the evidence for or against them.

- **Check actual machine behavior.** When confirming a mechanism, inspect the target loop, instructions, and physical dependencies in the binary actually loaded. Source intent or compilation success does not prove the change survived; fewer instructions need not run faster.
- **Make probes answer one question.** Ensure the intended operations survive optimization, remain inside the measured scope, and affect an observable result. Validate the behavior a functional probe is meant to establish; its checks need not reproduce the full operator. Layout probes should cover production-relevant offsets and boundaries.
- **Keep controls interpretable.** State what A/B is testing and what deliberately changes. Diagnostic comparisons may bypass a module, replace an instruction or data, or change precision and outputs; they need not pass the full operator's correctness check. Inspect collateral changes in actual work, resource use, waits, and scheduling. Interpret the timing difference within that experiment's conditions; it is not automatically the isolated cost of a module or an equivalent-candidate speedup. Split combined changes when needed; individual gains need not add together.
- **Validate candidate speedups separately.** When presenting a change as an optimized operator, verify its required semantics, accuracy, and synchronization, then compare on the same workload and timing scope.
- **Choose measurements for the question.** Instrumentation is optional: use it when relevant and suitable tools are available; otherwise proceed with ordinary timing, ISA inspection, or controlled experiments. If used, confirm claimed speedups with separate uninstrumented timing. For small gains, paired or interleaved runs can check variability. Local probes explain mechanisms; the complete operator call establishes the application benefit.
- **Match conclusions to evidence.** A sampled PC fraction is not a phase duration, capacity is not activity, and local latency is not fully removable operator time. Separate measurements, inferences, and unresolved explanations. Negative results apply to their tested conditions and may be revisited with new evidence.

Keep experiment code and local test/measurement data as the primary evidence, with relevant conditions in code, commands, or logs. Brief observations or working hypotheses with file paths are sufficient; interpretations may be revised or overturned by new evidence in later iterations.

## Working with profiling

If dynamic counters or internal timing would help and suitable tools are available, use the project's selected backend profiling tools or Skills. Their documentation owns collection commands, counter definitions, and report formats; do not transplant those details from another GPU backend.

For ordinary ISA or layout probes, the code and local test data serve as the experiment record. Apply a profiler's evidence/report contract when using that profiler or persisting its evidence.
