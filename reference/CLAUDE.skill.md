# GPU Kernel Optimizer — Agent Constraints (SKILL.md route)

This file defines hard behavioral constraints for the **in-session** optimization workflow driven by the
`gpu-kernel-optimizer` router (`SKILL.md`) and its `gpu-kernel-profile-optimizer` loop skill. It is auto-loaded
when the agent works inside this workspace.

> This is the **SKILL.md (torch) route**: a specified torch implementation is turned into a high-performance
> kernel and iteratively optimized toward the roofline in a single long session (the profile-optimizer skill
> owns its Stage 1–6 loop, guarded by the installed Stop hook + `gpu-kernel-partial-restart`). The final
> deliverable is `generated_kernel.py` (see `skills/gpu-kernel-output-contract/SKILL.md`). This route is
> independent of the `orchestrator/optimize.py` (SOL-ExecBench) route — there is no `definition.json`,
> `solution.json`, `workload.jsonl`, or `sol-execbench` evaluator here.

## Framework Guidance

- Implement the kernel in the **target DSL** from the workspace `README.md` (`framework`): CuteDSL for
  NVIDIA Hopper/Blackwell, FlyDSL for AMD CDNA. The V0 baseline is already a correct kernel in that DSL.
- `triton` and `gluon` belong to the same framework family (`triton/gluon`); when either is specified, both
  are acceptable implementation targets.
- The `framework` value is the recommended optimization direction. A different DSL or mixed approach is
  allowed if profile evidence shows a better performance path.
- Third-party helper libraries (utility/math libraries) MAY be introduced freely to assist optimization.

## Benchmark and Correctness Integrity

- **The optimization goal is to raise peak utilization toward the roofline** — the `Stop Conditions` in the
  workspace `README.md` (default: TFLOPS/bandwidth peak utilization >= 90% of the sourced hardware peak).
  Judge iterations by the peak-utilization ratios, not by latency alone.
- **Correctness harness is `test_kernel.py`** (written from the PyTorch reference during baseline). Validate
  with `timeout 60 python test_kernel.py` (per-case guard via `TEST_TIMEOUT_SEC`, default 30s). Do NOT edit
  `test_kernel.py` to weaken tolerances, shrink shapes, or otherwise game correctness.
- **Latency is measured with `triton.testing.do_bench`** (p50 median); TFLOPS / bandwidth / peak utilization
  are computed with `tools/compute_utilization.py`. `do_bench` and handwritten timers are timing helpers only —
  they do NOT replace `tools/profile_nvidia.sh` (`ncu`) / `tools/profile_kernel.sh` for identifying bottlenecks.
- **No hacking the measurement.** Do NOT monkey-patch, shadow, or subvert the timing loop, RNG, or comparator
  to make a slower/incorrect kernel look faster or pass. Any speedup must come from a faster kernel on
  arbitrary inputs.

### Real-submission input model (don't overfit to the local test)

The final `generated_kernel.py` is scored by a **hidden evaluator** that supplies **freshly randomized inputs
at freshly allocated addresses on every call** (shapes/dtypes are fixed; values, RNG seed, and tensor pointers
are not). Therefore:

- **Do NOT cache input data / outputs.** Never recognize "I've seen these inputs" and return a precomputed or
  recorded result, or branch on input *values* (checksums, hashes, sentinels, "if input == X return Y").
  `Model.forward` must recompute from its arguments every call.
- **Do NOT cache pointers / addresses.** Never key a code path, plan, autotuned config, or scratch workspace on
  `tensor.data_ptr()` / raw CUDA addresses — they change every call. Key runtime plans on **shape + dtype +
  layout + device** only.
- **Do NOT amortize work across timed iterations** in a way that is only correct because the same inputs repeat.
- **CUDA graphs**: if used, re-bind kernel node parameters from the current tensor `data_ptr`s on each call —
  never replay against capture-time addresses.

### Multi-seed robustness (before accepting a kernel)

A single PASS is not sufficient: because the evaluator reseeds inputs, a kernel that passes on one draw can
fail on another (numerical edge-cases, magnitude-dependent accumulation). Before committing a kernel change,
re-run correctness under **several different random seeds** — extend the seed in `test_kernel.py` if it does
not already vary — and require ALL to PASS. If any seed fails, the kernel is BROKEN: revert
(`git reset --hard HEAD`) and try a different lever.

### Precision margin (don't surf the tolerance line)

Tolerances are safety margins, not optimization targets. If any test's measured error sits close to its
tolerance, STOP and re-review `kernel.py` end-to-end before committing, and confirm the margin is stable across
fresh seeds. A speedup that only works by shrinking the precision margin is not real — revert.

### No multi-stream timing tricks

Do NOT use multiple CUDA streams to overlap independent work and reduce measured latency. Keep all compute on
the default stream; do not create/sync extra streams to parallelize the computation. Legitimate single-stream
optimizations (fusion, tiling, vectorization, lower precision, library primitives) are unaffected.

## Hardware Spec and Profiling Constraints

- **Hardware specs must come from `gpu-wiki`** with a source reference (`<metric>: <value> <unit> <-
  <gpu-wiki>/<path>:<line>`). Never fabricate specs; missing specs are `UNKNOWN (gpu-wiki not found)` and must
  be escalated to the user.
- **Optimization decisions require official profile evidence**: `tools/profile_nvidia.sh` (`ncu`) for NVIDIA,
  `tools/profile_kernel.sh` (rocprofv3/ATT/PMC/ASM) for AMD. Write decisions as `evidence -> inference ->
  action`. Do not edit `kernel.py` without profile evidence attribution.
- Change exactly **one optimization category per iteration** so the result is attributable.
- Correctness must pass before performance conclusions or commits; every accepted iteration is committed with
  git (HEAD = best kernel so far). A regressing iteration reverts and is never committed.
- `memory/v*.json` files with `masked: true` are discarded from active planning.

## Hardware Architecture Constraints

- **blackwell-geforce is NOT blackwell**: `blackwell-geforce` (sm120) and `blackwell` (sm100) are completely
  different architectures. Do NOT conflate them or assume shared optimization strategies.
- **sm103 ≈ sm100 ≠ sm120**: sm103 is similar to sm100 (Blackwell data-center family) but completely different
  from sm120 (Blackwell GeForce / consumer). Prefer sm100/blackwell sources for sm103 — NEVER sm120.

## Workflow References

- Router / global constraints: `SKILL.md` (`gpu-kernel-optimizer`).
- Stage 1 baseline: `agents/gpu-kernel-baseline.md`.
- Stage 2 profile-driven optimization **loop** (owns Stage 6): `skills/gpu-kernel-profile-optimizer/SKILL.md`,
  driving `gpu-kernel-profiler` -> `gpu-kernel-research` -> `kernel-optimize` subagents per iteration.
- Partial restart when no new direction remains: `agents/gpu-kernel-partial-restart.md`.
- Final candidate packaging: `skills/gpu-kernel-output-contract/SKILL.md` -> `generated_kernel.py`.
- Profiling: `tools/profile_nvidia.sh` (NVIDIA), `tools/profile_kernel.sh` (AMD);
  utilization: `tools/compute_utilization.py`; memory: `tools/memory_manager.py`.
