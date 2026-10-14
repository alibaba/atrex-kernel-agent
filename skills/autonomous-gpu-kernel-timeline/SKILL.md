---
name: autonomous-gpu-kernel-timeline
description: Let AKA autonomously add, run, inspect, and revise intra-kernel timeline probes for standalone CUDA/inline PTX or CuTe DSL when ordinary benchmark, NSYS, or NCU evidence cannot answer a specific kernel-internal timing question.
---

# Autonomous GPU Kernel Timeline

Use this inside an AKA optimization episode only after a correct runnable kernel and representative
workload exist. Do not trigger it when aggregate kernel timing or ordinary profiler evidence already
answers the question.

## Ownership

- AKA chooses the hypothesis, sites, writer roles, density, boundary semantics, and number of retries.
- The backend records and decodes those choices without silent identity, capacity, or pairing errors.
- The final validator rejects known factual failures; it does not decide whether AKA chose the best
  algorithm boundary or interpretation.

## Route

- Standalone CUDA and inline PTX: read [references/cuda-backend.md](references/cuda-backend.md)
  and use `backends/cuda_backend/atrex_timeline.cuh`.
- CuTe DSL: read [references/iket-quickstart.md](references/iket-quickstart.md) and use IKeT. Do not
  fall back to the CUDA backend.

## Autonomous loop

1. State one falsifiable timing question and begin with the fewest useful sites.
2. Save the current clean `kernel.py` content, then create an instrumented working snapshot in the
   same episode worktree. This snapshot is not a handoff candidate and must not enter promotion or
   stall accounting.
3. Run representative correctness through the immutable evaluator and capture through the campaign
   sandbox. For final evidence, pass the sandbox-owned `.atrex_long_horizon/evaluations.jsonl` as
   `--correctness-evidence`; a model-supplied `--correctness passed` is only an exploration note and
   cannot produce `decision_grade`. Use `scripts/timeline.py` to validate and export the returned
   evidence. If the remote command reads backend files, pass
   `--input skills/autonomous-gpu-kernel-timeline/backends/<backend>` to `tools/sandbox.py`; sync only
   the attempt-specific directory.
4. Read the summary and Perfetto trace. Remove, move, or refine probes and repeat when the evidence
   is insufficient, semantically misplaced, or too intrusive. Name every reported interval as
   `start_site -> end_site` and recompute its numbers from canonical events rather than trusting a
   prose label. A local timeline delta is mechanism evidence, not a substitute for probe-free
   end-to-end ABBA. Reviewer feedback is optional.
5. Before replacing the snapshot, preserve its source or reversible patch and its content hash.
6. Restore a probe-free kernel, implement the optimization, and use the normal evaluator for final
   correctness and performance. Re-instrument the new clean state only when confirmation is useful.

For a final perturbation claim, use `scripts/timeline.py measure` inside one GPU allocation. Give it
baseline/instrumented commands as JSON argv arrays, the exact sources, and materialized binaries when
the compiler exposes them. JIT-only CuTe DSL measurements need not invent a binary artifact.
Each command must perform the requested warmup and iterations, synchronize the device, check the full
representative output, and emit exactly one line of this form:

```text
__ATREX_TIMELINE_SAMPLE__={"latency_ms":1.0,"correctness":"passed","synchronized":true,"workload_identity":"...","device_identity":{"uuid":"..."},"warmup":10,"iterations":100}
```

The helper runs each sample in a fresh process, defaults to ABBA followed by BAAB, rejects workload or
device drift, and writes the raw schedule and samples. Pass that artifact to capture/export with
`--measurement`; `validate` rechecks its hashes and recomputes the medians and overhead.
The sample's `correctness` field rejects a bad timing run but does not replace the immutable evaluator
record required for final evidence.

## Hard boundaries

- Never modify `profile_driver.py`, evaluators, ground truth, or other protected paths.
- Never hand off or promote an instrumented snapshot. Only a probe-free kernel may become the
  episode `candidate_commit == HEAD`.
- Do not combine events from different launches into one apparent execution.
- Construct exactly one recorder per selected owner per launch and reuse it for that owner's events;
  duplicate ownership is a capture failure, not a sampling policy.
- Reject overflow, truncation, invalid owner/site identity, deterministic range mismatch, stale
  source/binary/workload provenance, and unrecomputable numeric claims.
- Never classify final evidence as `decision_grade` without a matching sandbox evaluator record.
- The default low-perturbation target is 10%. A higher-overhead trace may remain diagnostic when it
  is labelled honestly; the threshold is hard only when the task explicitly requires it.
- Do not add fixed site catalogs, fixed coverage quotas, mandatory reviewer approval, or a separate
  orchestration state machine.
