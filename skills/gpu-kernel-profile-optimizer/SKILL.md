---
name: gpu-kernel-profile-optimizer
description: |
  Profile-driven GPU kernel iterative optimization skill. Use this skill to run a closed loop in a temporary git workspace: profile evidence extraction, evidence-driven search and planning, single-category optimization, validation, memory update, and git commit.
---

# GPU Kernel Profile Optimizer

## When to Use

Use this skill when the user asks to:

- Optimize an existing GPU kernel.
- Continue improving code based on `./tools/profile_nvidia.sh` (`ncu`), `./tools/profile_kernel.sh`, ATT, PMC, or ASM evidence.
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
kernel_opt_<name>/
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

**Execution Context**: All profiling commands must be executed from the workspace root directory (`kernel_opt_<name>/`). The `--output-dir profiles/v<N>` path is relative to this root. Running from a subdirectory will cause profile outputs to land in unexpected locations.

Profile the current `kernel.py` directly. For detailed tool usage, metric interpretation, and troubleshooting, refer to `reference/profile_guide.md`.

```bash
mkdir -p profiles/v<N>
```

Use an independent output directory for every iteration to avoid mixing versions.

### NVIDIA Hopper/Blackwell: profile_nvidia.sh

Use the top-level tool script instead of writing `ncu` commands manually:

```bash
bash tools/profile_nvidia.sh \
  kernel.py \
  --output-dir profiles/v<N> \
  --launch-skip <skip>
```

For source-level stall hotspot analysis (requires the kernel compiled with `-lineinfo`):

```bash
bash tools/profile_nvidia.sh \
  kernel.py \
  --output-dir profiles/v<N> \
  --launch-skip <skip> \
  --source
```

To collect only, without symptom classification:

```bash
bash tools/profile_nvidia.sh \
  kernel.py \
  --output-dir profiles/v<N> \
  --no-classify
```

The script automatically performs these steps:

1. `ncu --set full` collects the `.ncu-rep` binary report.
2. (Optional, `--source`) `ncu --set source` collects source-level stall data.
3. `analyze_reports.py` (bundled in `tools/ncu_helpers/`) parses key metrics into `metrics_key_run.json`.
3b. (Only on `--source`) `source_evidence.py` generates the source-level evidence bundle and indexes it in `source_evidence_manifest.json`. Best-effort, never fatal; the artifacts are a dependency-free Python port of VeloQ's `ncu` verbs onto the same `ncu_report` API, emit a `v1` JSON envelope, and do **not** feed `classify_ncu.py` or change `summary.txt`.
3c. (Optional, `--diff PREV_DIR`) `row_key.py` joins this run's envelopes against a previous run by stable content-derived key and writes `analysis/diff_*.txt`.
4. `classify_ncu.py` classifies symptoms against the 14 NCU diagnosis patterns, producing `summary.txt`.

Artifacts (always):

- `profiles/v<N>/ncu.ncu-rep` — binary report
- `profiles/v<N>/analysis/metrics_key_run.{json,txt}` — key metrics
- `profiles/v<N>/summary.txt` — final summary (metrics + `SYMPTOMS` + `LOCALIZE` + search suggestions)

Artifacts (only with `--source`, indexed by `analysis/source_evidence_manifest.json`):

- `analysis/stall_hotspots_run.txt` — per-line stall hotspots (pcsamp metrics)
- `analysis/disasm_run.{json,txt}` — structured source-correlated SASS (+PTX when `nvdisasm`/`cuobjdump` present)
- `analysis/warp_stalls_{reason,line}_run.{json,txt}` — warp-stall attribution from `timed_warp_samples`
- `analysis/source_metrics_{line,sass}_run.{json,txt}` — per-line / per-SASS metric attribution
- `analysis/diff_*.txt` — only with `--diff`: per-row delta vs a previous run

Extract at least: memory throughput / SOL, L2 hit rate, occupancy, warp stall reasons, and Tensor Core / MMA utilization. The `SYMPTOMS` line in `summary.txt` is controlled vocabulary that feeds directly into the Stage 2 gpu-wiki search (see *Symptom-Driven Retrieval* in `<gpu-wiki>/README.md`). The `LOCALIZE` line names which `--source` evidence file maps each fired symptom to a source line / SASS address — to act on it, rerun with `--source` and open that file (or `source_evidence_manifest.json`). Note `warp_stalls_*` (from `timed_warp_samples`) and `stall_hotspots` (from pcsamp metrics) answer the same "where do warps stall" question from two sources; prefer `warp_stalls_*` and use `stall_hotspots` only to cross-check.

#### Localization rule (mandatory)

The first profile pass runs **without** `--source` (cheap: no second `ncu` collection). Escalate to `--source` only when a localizable symptom actually drives a change:

- **Trigger** — `summary.txt` emits a `LOCALIZE` line (only localizable symptoms produce one; symptoms with no line-level signal, e.g. occupancy, never do) **and** you are about to choose a concrete code change based on that symptom.
- **Required action** — before editing `kernel.py`, re-profile the kernel with `--source`, open the evidence file named on the `LOCALIZE` line (or read `source_evidence_manifest.json`), and pin the change to the specific source line / SASS address it identifies. Do not change a line you have not localized.

This makes the signal — not the agent's discretion — decide when the evidence layer turns on: cheap by default, and the source-level evidence is guaranteed to be read at the moment it drives a code change. When no `LOCALIZE` line is present, no `--source` rerun is needed.

### AMD CDNA3/CDNA4: ATT Decoder Setup

ATT profiling depends on the trace decoder plugin shipped in this skill:

```text
tools/rocprof-trace-decoder-amd-mainline/
```

Before ATT profiling, ensure `rocprofv3` can find the decoder:

```bash
export LD_LIBRARY_PATH=<skill_root>/tools/rocprof-trace-decoder-amd-mainline/releases/linux_glibc_2_28_x86_64:$LD_LIBRARY_PATH
```

Without this path, `rocprofv3 --att` cannot decode thread-trace binaries and the ATT artifacts are unusable.

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

### NVIDIA SASS Analysis with extract_nvidia_asm.py

Beyond `ncu`, NVIDIA kernels also need SASS inspection to confirm tensor core instructions, load/store width, and register spills.

**CuteDSL kernel (recommended flow)**: first collect `.ncu-rep` with `profile_nvidia.sh`, then extract SASS from it:

```bash
# Step 1: collect .ncu-rep (if not already done)
bash tools/profile_nvidia.sh kernel.py --output-dir profiles/v<N>

# Step 2: extract SASS from .ncu-rep and analyze
python tools/extract_nvidia_asm.py \
  --ncu-rep profiles/v<N>/ncu.ncu-rep \
  --check-all --arch sm90
```

This is the most reliable method: ncu's Python API `action.sass_by_pc()` extracts complete SASS directly from the profile report without needing to locate cubin files. It requires the bundled `tools/ncu_helpers/`.

**Triton kernel**: extract directly from the kernel file:

```bash
python tools/extract_nvidia_asm.py \
  kernel.py \
  --output profiles/v<N>/kernel.sass \
  --check-all --arch sm90
```

**Existing cubin / `.so`**:

```bash
python tools/extract_nvidia_asm.py \
  --cubin profiles/v<N>/kernel.cubin \
  --check-all --arch sm90
```

**Existing SASS text**:

```bash
python tools/extract_nvidia_asm.py \
  --asm-file profiles/v<N>/kernel.sass \
  --check-all --arch sm90
```

The `--arch` flag controls the expected instruction set:

- `sm90` (Hopper): expects HMMA / WGMMA / CPASYNC / LDGSTS / LDSM
- `sm100` (Blackwell): expects TCGEN05 / GMMA / TMA / UTMALDG / ULDGSTS / LDSM

This tool helps confirm:

- Whether register spills occur (STL/LDL instructions)
- Whether expected tensor core instructions are present (HMMA/WGMMA/TCGEN05)
- Whether expected async instructions are present (CPASYNC/LDGSTS/TMA)
- Whether load/store width is optimal (LDG.E.128 vs LDG.E)
- Whether scalar fallback occurs (excessive FMUL/FFMA instead of tensor core)
- Instruction classification breakdown (compute / memory / control)

Add `--json` for programmatic consumption.

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

- `summary.txt shows Pattern E (long_scoreboard=4.2)` -> `latency-bound` -> `try cp.async / double buffering`
- `summary.txt shows Pattern A (grid=64 < sm=78)` -> `SM idle` -> `increase split-k or use a persistent kernel`
- `PMC shows high SQ_LDS_BANK_CONFLICT` -> `LDS bank conflicts are significant` -> `try a swizzled layout`
- `ASM shows many buffer_load_dword and few dwordx4` -> `global memory vectorization is insufficient` -> `adjust alignment and vector width`

## Stage 2: Evidence-Driven Search and Planning

### Goal

Use Stage 1 profile evidence to extract the current bottleneck, search knowledge sources by priority, and write this iteration's plan to `plans/v<N>_plan.md`. This plan is the only input for Stage 3 implementation.

### Execution: Subagent Required

The main agent must launch the `gpu-kernel-research` subagent by name for Stage 2. The main agent must not perform evidence search or write the plan directly.

The subagent reads current profile artifacts, workspace constraints, historical `plans/v*_plan.md`, gpu-wiki, optional reference projects, and public web sources. Once it finds one eligible executable path that matches the current bottleneck, it must write the plan and return. In normal mode an untried historical finding is eligible; after three consecutive stalls, a new finding is required. It must not keep broadening the search unnecessarily.

Subagent requirements:

- **Task type**: read-only research plus plan-writing task.
- **Required inputs**: workspace path, version `V<N>`, `README.md`, `memory/` directory, all unmasked `memory/v*.json` files, historical plan paths, `profiles/v<N>/` artifacts, previous `memory/v<N-1>.json` if present, platform, architecture, framework, kernel type, Stop Conditions, and the consecutive reverted/no-improvement `stall_count`.
- **Must do**: read all prerequisite files; skip `memory/v*.json` files where `masked: true`; summarize attempted historical methods from unmasked memory files; extract bottlenecks from profile evidence; search gpu-wiki, then reference project, then public web by priority; stop after the first actionable non-duplicate path; write `plans/v<N>_plan.md`.
- **Forbidden**: do not modify `kernel.py`; do not perform Stage 3; do not skip gpu-wiki; do not fabricate specs; do not repeat prior plans; do not read `masked: true` memory files as active data; do not output multiple parallel optimization actions; do not return only a verbal plan.
- **Return**: `plans/v<N>_plan.md` path, evidence summary, search-source summary, search mode/stall count, the single optimization action, measurable performance expectation, conditional ISA-escalation trigger, risks, and rollback.

### Mandatory Reads per Iteration

Starting from V1, read:

1. `kernel_opt_<name>/README.md`
2. `<gpu-wiki>/README.md`
3. All unmasked `kernel_opt_<name>/memory/v*.json` files (skip files where `masked: true`)
4. Current `profiles/v<N>/` artifacts
5. Previous `kernel_opt_<name>/memory/v<N-1>.json` (if unmasked)
6. Historical `plans/v*_plan.md`

### Knowledge Base Search

Translate Stage 1 profiler symptoms into gpu-wiki search keywords using the
**Symptom-Driven Retrieval (NVIDIA vs AMD)** guidance in `<gpu-wiki>/README.md` —
NVIDIA and AMD use different vocabularies and sub-trees, and that vendor-split
mapping is maintained there, not in this skill. Start with
`python3 <gpu-wiki>/scripts/query.py` using `--arch`, `--vendor`, and, when
applicable, `--dsl`, `--symptom`, `--operator`, and `--section`; then apply the
Search Priority below.

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
- If the change targets a symptom that produced a `LOCALIZE` line in `summary.txt`, do not edit `kernel.py` until you have localized it via a `--source` re-profile (see *Localization rule (mandatory)* in Stage 1); change only the specific line(s) the evidence identifies, not the whole kernel.
- If framework API or operator interface details are needed, search `<gpu-wiki>/reference-kernels/` or clone upstream source to `reference-projects/` first.
- Changes must land in workspace `kernel.py`; auxiliary files may be adjusted only when necessary and must be explained in the report.
- Do not mix unrelated refactors, formatting, or cleanup.
- After editing, immediately run correctness validation through `test_kernel.py` or the validation entry in `kernel.py`.
- Before starting an iteration, create the memory file if it does not exist:

  ```bash
  python tools/memory_manager.py create --workspace kernel_opt_<name> --version v<N>
  ```

- Update `memory/v<N>.json` immediately after the edit result is known using the memory manager:

  ```bash
  python tools/memory_manager.py update --workspace kernel_opt_<name> --version v<N> \
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
   python tools/memory_manager.py update --workspace kernel_opt_<name> --version v<N> \
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
python tools/memory_manager.py update --workspace kernel_opt_<name> --version v<N> \
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
  python tools/memory_manager.py summary --workspace kernel_opt_<name>
  ```

- Summarize key actions and gains for all versions.
- Preserve `profiles/` so the evidence chain remains auditable.

When stop conditions are not met:

- Return to Stage 1.
- Read the latest unmasked `memory/v<N>.json` and `README.md`.
- Profile the current version, extract the latest bottleneck, then search and plan the next iteration.

## Appendix: Tool-to-Evidence Mapping

Different tools provide different layers of evidence and must not be mixed:

- `profile_nvidia.sh` + `classify_ncu.py`: primary NVIDIA profile entry point (wraps `ncu`). Collects the `.ncu-rep`, parses metrics, and classifies symptoms. Artifact: `summary.txt` (symptoms + search suggestions).
- `extract_nvidia_asm.py`: NVIDIA static SASS evidence for tensor core instructions, load/store width, register spills, and scalar fallback.
- `ncu_helpers/source_evidence.py` (VeloQ-ported; run automatically by `profile_nvidia.sh --source`, indexed in `source_evidence_manifest.json`): bundles per-line/per-SASS metric attribution (`source_metrics`), warp-stall attribution (`warp_stalls`), and structured source-correlated SASS (`disasm`). **Independent evidence** — read the `analysis/*_run.json` (v1 envelope) or `.txt` digests to localise *which source line / SASS address* a symptom lives on. They do not change `summary.txt`; the `SYMPTOMS` diagnosis still comes only from `classify_ncu.py`, and its `LOCALIZE` line tells you which of these files to open. Use `ncu_helpers/row_key.py` (or `profile_nvidia.sh --diff`) to compare two iterations row-by-row.
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
