---
name: gpu-kernel-profile-optimizer
description: |
  Profile-driven GPU kernel iterative optimization skill. Use this skill to run a closed loop in a temporary git workspace: profile evidence extraction, evidence-driven search and planning, single-category optimization, validation, memory update, and git commit.
---

# GPU Kernel Profile Optimizer

## When to Use

Use this skill when the user asks to:

- Optimize an existing GPU kernel.
- Continue improving code based on `ncu`, `./tools/profile_kernel.sh`, ATT, PMC, or ASM evidence.
- Use profiling evidence rather than intuition to improve performance.

## Overall Principles

This skill must follow the stage order below. Commands, wiki searches, profiling, validation, records, and commits must be completed inside their corresponding stage.

```text
Stage 1 Profile and evidence extraction
Stage 2 Evidence-driven search and planning
Stage 3 Single-category optimization implementation
Stage 4 Performance, correctness, and quality gate
Stage 5 Memory update and git commit
Stage 6 Stop-condition check or next iteration
```

Constraints:

- Do not skip stages.
- Do not edit code without profile evidence.
- Implement only one optimization category per iteration so the result can be attributed.
- If the quality gate fails, revert to the previous commit, record the failure, and stop the current iteration.
- If stop conditions are not met, do not exit unless the user explicitly stops the workflow.

## Workspace Layout

Maintain this structure:

```text
/tmp/kernel_opt_<name>/
  kernel.py
  reference.py
  README.md
  test_kernel.py
  baseline_report.md
  memory/
    v0.json
    v1.json
  plans/
    v0_plan.md
    v1_plan.md
  profiles/
    v0/
    v1/
```

## Stage 1: Profile and Bottleneck Evidence Extraction

### Goal

Profile the current version with official tools, place outputs in `profiles/v<N>/`, and extract at least one concrete bottleneck evidence item.

### General Requirements

Profile the current `kernel.py` directly. For detailed tool usage, metric interpretation, and troubleshooting, refer to `reference/profile_guide.md`.

```bash
mkdir -p profiles/v<N>
```

Use an independent output directory for every iteration to avoid mixing versions.

### NVIDIA Hopper: ncu

```bash
ncu --set full   --launch-skip <skip>   --launch-count 1   -o profiles/v<N>/ncu   python kernel.py
```

Rules:

- `--launch-skip` skips warmup dispatches.
- `--launch-count 1` captures one stable dispatch.
- Store output as `profiles/v<N>/ncu*`.

Extract at least:

- memory throughput / SOL
- L2 hit rate
- occupancy
- warp stall reasons
- Tensor Core or MM utilization

Use these metrics as Stage 2 search keywords.

### AMD CDNA3/CDNA4: ATT Decoder Setup

ATT profiling depends on the `rocprof-trace-decoder` library from ROCm or the upstream release package.

Before ATT profiling, ensure `rocprofv3` can find the decoder by installing it outside this repository and passing its extracted library directory to ATT commands.

Without the decoder library, `rocprofv3 --att` cannot decode thread-trace binaries and the ATT artifacts are unusable.

### AMD CDNA3/CDNA4: profile_kernel.sh

Use the top-level script instead of writing long `rocprofv3` commands manually:

```bash
bash tools/profile_kernel.sh   kernel.py   --output-dir profiles/v<N>
```

For one data type only:

```bash
bash tools/profile_kernel.sh kernel.py --output-dir profiles/v<N> --pmc-only
bash tools/profile_kernel.sh kernel.py --output-dir profiles/v<N> --att-only
```

For a specific dispatch:

```bash
bash tools/profile_kernel.sh   kernel.py   --output-dir profiles/v<N>   --kernel-regex "<kernel_name>"   --iteration-range 0-0
```

The script collects:

- `ATT` instruction-level trace
- `PMC` hardware counters
- `ASM` assembly

Artifacts:

- `profiles/v<N>/att/`
- `profiles/v<N>/pmc/`
- `profiles/v<N>/kernel.s`

### AMD Assembly Analysis

`profile_kernel.sh` extracts assembly to:

```text
profiles/v<N>/kernel.s
```

It can also extract ASM only:

```bash
bash tools/profile_kernel.sh kernel.py --output-dir profiles/v<N> --asm-only
```

Check assembly for:

- `buffer_load_dword`, `buffer_load_dwordx2`, `buffer_load_dwordx4`
- `ds_read_b32`, `ds_read_b64`, `ds_read_b128`
- `ds_write_b32`, `ds_write_b64`, `ds_write_b128`
- `ds_bpermute`
- `scratch_load`, `scratch_store`
- `vgpr_spill_count`

### Evidence Format

Write conclusions as:

```text
evidence -> inference -> optimization action
```

Examples:

- `PMC shows high SQ_LDS_BANK_CONFLICT` -> `LDS bank conflicts are significant` -> `try a swizzled layout`
- `ASM shows many buffer_load_dword and few dwordx4` -> `global memory vectorization is insufficient` -> `adjust alignment and vector width`
- `ncu shows memory dependency dominates warp stalls` -> `latency hiding is insufficient` -> `try double buffering or software pipelining`

## Stage 2: Evidence-Driven Search and Planning

### Goal

Use Stage 1 profile evidence to extract the current bottleneck, search knowledge sources by priority, and write this iteration's plan to `plans/v<N>_plan.md`. This plan is the only input for Stage 3 implementation.

### Execution: Subagent Required

The main agent must launch a subagent for Stage 2. The main agent must not perform evidence search or write the plan directly.

The subagent reads current profile artifacts, workspace constraints, historical `plans/v*_plan.md`, gpu-wiki, optional reference projects, and public web sources. Once it finds one executable non-duplicate path that matches the current bottleneck, it must write the plan and return. It must not keep broadening the search unnecessarily.

Subagent requirements:

- **Task type**: read-only research plus plan-writing task.
- **Required inputs**: workspace path, version `V<N>`, `README.md`, `memory/` directory, all unmasked `memory/v*.json` files, historical plan paths, `profiles/v<N>/` artifacts, previous `memory/v<N-1>.json` if present, platform, framework, kernel type, and Stop Conditions.
- **Must do**: read all prerequisite files; skip `memory/v*.json` files where `masked: true`; summarize attempted historical methods from unmasked memory files; extract bottlenecks from profile evidence; search gpu-wiki, then reference project, then public web by priority; stop after the first actionable non-duplicate path; write `plans/v<N>_plan.md`.
- **Forbidden**: do not modify `kernel.py`; do not perform Stage 3; do not skip gpu-wiki; do not fabricate specs; do not repeat prior plans; do not read `masked: true` memory files as active data; do not output multiple parallel optimization actions; do not return only a verbal plan.
- **Return**: `plans/v<N>_plan.md` path, evidence summary, search-source summary, the single optimization action, expected impact, risks, and rollback.

### Mandatory Reads per Iteration

Starting from V1, read:

1. `/tmp/kernel_opt_<name>/README.md`
2. `<gpu-wiki>/README.md`
3. All unmasked `/tmp/kernel_opt_<name>/memory/v*.json` files (skip files where `masked: true`)
4. Current `profiles/v<N>/` artifacts
5. Previous `/tmp/kernel_opt_<name>/memory/v<N-1>.json` (if unmasked)
6. Historical `plans/v*_plan.md`

### Search Priority

Search in this strict order:

1. **gpu-wiki first**: search the entire `<gpu-wiki>/` repository, not only `docs/`.
2. **reference project fallback**: use only when gpu-wiki has no new path and `README.md` has `reference-project != none`.
3. **public web fallback**: use only when neither gpu-wiki nor the reference project provides a new actionable path.

Public web findings may provide optimization ideas only. Hardware spec values still require gpu-wiki or explicit user confirmation.

### Plan Format

Follow the format defined in `reference/plan.md`.

## Stage 3: Single-Category Optimization Implementation

Goal: implement exactly one optimization category from `plans/v<N>_plan.md` and keep attribution clear.

Rules:

- Change only one category per iteration, such as vectorized load only, swizzle only, or double buffering only.
- If framework API or operator interface details are needed, use the reference-project fallback in `/tmp/reference-projects/` or public upstream documentation.
- Changes must land in workspace `kernel.py`; auxiliary files may be adjusted only when necessary and must be explained in the report.
- Do not mix unrelated refactors, formatting, or cleanup.
- After editing, immediately run correctness validation through `test_kernel.py` or the validation entry in `kernel.py`.
- Before starting an iteration, create the memory file if it does not exist:

  ```bash
  python tools/memory_manager.py create --workspace /tmp/kernel_opt_<name> --version v<N>
  ```

- Update `memory/v<N>.json` immediately after the edit result is known using the memory manager:

  ```bash
  python tools/memory_manager.py update --workspace /tmp/kernel_opt_<name> --version v<N> \
      --set 'optimization.action_category=<category>' \
      --set 'optimization.action_description=<description>'
  ```

## Stage 4: Performance, Correctness, and Quality Gate

Goal: calculate performance gain, validate correctness, compare ISA target progress, and decide whether the iteration passes.

### Execution: Subagent Required

The main agent must launch a subagent for Stage 4. The main agent must not run validation, measure performance, or write the iteration report directly.

The subagent executes correctness tests with timeout enforcement, measures performance, calculates utilization, compares with the previous iteration, and writes the iteration report. It returns PASS/FAIL and the report path so the main agent can proceed to Stage 5 or revert.

Subagent requirements:

- **Task type**: execution and validation task (may run commands and write reports).
- **Required inputs**: workspace path, version `V<N>`, `kernel.py`, `test_kernel.py`, `memory/v<N>.json`, previous `memory/v<N-1>.json` if present, platform, GPU model, `plans/v<N>_plan.md`.
- **Must do**:
  1. Run correctness validation with timeout guard (see below).
  2. Measure kernel performance (latency, TFLOPS, bandwidth).
  3. Calculate peak utilization using `tools/compute_utilization.py`.
  4. Compare metrics against the previous version.
  5. Evaluate ISA metric progress against targets in `README.md`.
  6. Update `memory/v<N>.json` with performance, correctness, and ISA progress data.
- **Forbidden**: do not modify `kernel.py`; do not perform Stage 3 changes; do not commit; do not skip correctness validation; do not fabricate performance numbers.
- **Return**: quality-gate result (PASS / FAIL / TIMEOUT_FAIL), `memory/v<N>.json` path, performance summary, correctness result, and failure reason if applicable.

### Timeout Guard

All validation runs must enforce a timeout to prevent hanging on compilation errors, infinite loops, or synchronization deadlocks:

```bash
timeout 60 python test_kernel.py   # 60s max per run
```

- Each individual test case must complete within **30 seconds** (configurable via `TEST_TIMEOUT_SEC` env var).
- If a case exceeds the timeout, mark it as `TIMEOUT_FAIL`, kill the process, and record the failure in the iteration report.
- A timeout counts as a quality-gate failure; revert and record the cause.

### Metrics to Record

Record every iteration in `memory/v<N>.json`:

- latency in `us`
- TFLOPS
- bandwidth in `GB/s`
- TFLOPS peak utilization
- bandwidth peak utilization
- delta from previous version
- correctness result
- ISA metric progress

Do not judge by latency alone; use peak-utilization ratios to decide whether the kernel is close to the limit.

### Iteration Data

Update `memory/v<N>.json` with performance, correctness, and ISA progress data following the schema defined in `reference/v_iteration.schema.json`.

### Quality Gate

Pass conditions:

- Correctness validation PASS.
- No unacceptable performance regression, or the regression is clearly explained and supports later optimization.
- No severe ISA regression, such as new spills or a large occupancy drop.

### Failure Handling

If the subagent returns FAIL or TIMEOUT_FAIL, the main agent must:

```bash
git reset --hard HEAD
```

Record the failure reason, skip further planning for this iteration, and do not enter the next iteration. Write the failure into `memory/v<N>.json` under `pitfalls_and_fixes` and `quality_gate`, then commit a revert marker such as `V5: revert V4 (occupancy 25% -> 12%)` when appropriate.

## Stage 5: Memory Update and Git Commit

Goal: finalize `memory/v<N>.json` with quality gate result and git commit hash, then commit.

### Procedure

1. Verify that `memory/v<N>.json` has been updated by Stage 4 with:
   - Performance metrics (TFLOPS, bandwidth, utilization)
   - Correctness result and `rel_err`
   - ISA metric progress
   - All values must come from actual measurements; do not re-measure or fabricate.

2. Update `memory/v<N>.json` using `tools/memory_manager.py`:

   ```bash
   python tools/memory_manager.py update --workspace /tmp/kernel_opt_<name> --version v<N> \
       --set 'quality_gate.result=PASS' \
       --set 'quality_gate.failure_reason=null'
   ```

3. If the quality gate was FAIL or TIMEOUT_FAIL, ensure the failure is recorded in `memory/v<N>.json` under `pitfalls_and_fixes` and `quality_gate`.

4. Commit must include:

- `kernel.py`
- `memory/v<N>.json`
- `plans/v<N>_plan.md`

```bash
git add kernel.py memory/v<N>.json plans/v<N>_plan.md
git commit -m "V<N>: <performance: TFLOPS/Bandwidth(GB/s)> | <optimization summary> (bottleneck: <profile evidence summary>)"
```

5. After commit, update `memory/v<N>.json` with the actual `git_commit_hash` using the memory manager and amend:

```bash
HASH=$(git rev-parse HEAD)
python tools/memory_manager.py update --workspace /tmp/kernel_opt_<name> --version v<N> \
    --set "git_commit_hash=$HASH"
git add memory/v<N>.json
git commit --amend --no-edit
```

Examples:

- `V3: 150 TFLOPS / 1.5 GB/s | XOR16 swizzle layout (bottleneck: ds_read bank conflict stall 12 cycles)`
- `V5: revert V4 (occupancy 25% -> 12%)`

## Stage 6: Stop-Condition Check or Next Iteration

Goal: decide whether to stop or return to Stage 1 for another iteration.

Default stop condition:

```text
utilization reaches >= 90% relative to the same-size baseline
```

If `README.md` or the user specifies a more concrete condition, use the more concrete condition.

When stop conditions are met:

- Output the final performance summary using the memory manager:

  ```bash
  python tools/memory_manager.py summary --workspace /tmp/kernel_opt_<name>
  ```

- Summarize key actions and gains for all versions.
- Preserve `profiles/` so the evidence chain remains auditable.

When stop conditions are not met:

- Return to Stage 1.
- Read the latest unmasked `memory/v<N>.json` and `README.md`.
- Profile the current version, extract the latest bottleneck, then search and plan the next iteration.

## Appendix: Tool-to-Evidence Mapping

- `ncu`: primary NVIDIA profile source for stalls, throughput, occupancy, cache hit rate, and Tensor Core utilization.
- `profile_kernel.sh`: primary AMD profile entry point, collecting ATT, PMC, and ASM for instruction width, spills, and LDS access patterns.
- `kernel.s`: AMD assembly evidence for load/store width, LDS instruction form, scratch operations, and spills.

## Appendix: Prohibited Actions

- Do not skip `<gpu-wiki>/README.md`.
- Do not start the next iteration without reading `README.md` and the latest unmasked `memory/v*.json` files.
- Do not read `memory/v*.json` files where `masked: true` as active iteration data.
- Do not reuse profile artifacts across versions.
- Do not commit performance conclusions without correctness validation.
- Do not record only latency without TFLOPS, bandwidth, and peak-utilization ratios.
- Do not provide unsourced optimization suggestions.
- Do not mix multiple optimization actions in one iteration.
- Do not continue planning after the quality gate fails.
