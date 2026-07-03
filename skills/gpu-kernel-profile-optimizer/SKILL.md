---
name: gpu-kernel-profile-optimizer
description: |
  Profile-driven GPU kernel iterative optimization skill. Use this skill to run a closed loop in a temporary git workspace: profile evidence extraction, evidence-driven search and planning, single-category optimization, validation, memory update, and git commit.
---

# GPU Kernel Profile Optimizer

## When to Use

Use this skill when the user asks to:

- Optimize an existing GPU kernel.
- Continue improving code based on `./tools/profile_iter_nvidia.sh` (`ncu`), `./tools/profile_kernel.sh`, ATT, PMC, or ASM evidence.
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
- Each optimization action must have clear profile evidence attribution.
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

**Subagent**: `gpu-kernel-profiler`

### Goal

Profile the current version with official tools, place outputs in `profiles/v<N>/`, and extract at least one concrete bottleneck evidence item.

The main agent must directly launch the `gpu-kernel-profiler` subagent for Stage 1. Do NOT write your own prompt or create an ad-hoc subagent — invoke `gpu-kernel-profiler` by name as a subagent. The main agent must not run profiling commands directly.

Subagent launch instruction:

```
Launch subagent: gpu-kernel-profiler
Task type: execution task
Inputs:
  - workspace_path: <workspace absolute path>
  - version: V<N>
  - platform: <nvidia / amd>
  - kernel_file: kernel.py
  - gpu_wiki_path: <gpu-wiki root path>
  - previous_profiles_dir: <profiles/v<N-1>/ if exists, otherwise omit>
```

The `gpu-kernel-profiler` subagent will autonomously: create `profiles/v<N>/` directory, run platform-specific profiling tools (`profile_iter_nvidia.sh` for NVIDIA or `profile_kernel.sh` for AMD), perform SASS/assembly analysis, and extract structured bottleneck evidence. On NVIDIA, when `previous_profiles_dir` is passed it runs `--diff` and reads back the three-layer cross-round diff (metric / PTX / SASS) so each iteration's change is verified at the IR/ISA layer — this delta is part of the evidence handed to Stage 2.

### Output Received

The profiler subagent returns:

| Field | Usage |
|-------|-------|
| `profiles_dir` | Path to `profiles/v<N>/` directory |
| `summary_path` | Path to `profiles/v<N>/summary.txt` — unified evidence summary for both NVIDIA and AMD |

`summary.txt` contains all extracted evidence: key metrics, `SYMPTOMS`, `LOCALIZE` (if applicable), and search suggestions. Both NVIDIA and AMD platforms produce this file as the single structured output.

### Localization rule (mandatory)

The first profile pass runs **without** `--source` (cheap: no second `ncu` collection). Escalate to `--source` only when a localizable symptom actually drives a change:

- **Trigger** — `summary.txt` emits a `LOCALIZE` line **and** Stage 3 is about to choose a concrete code change based on that symptom.
- **Required action** — before editing `kernel.py` in Stage 3, re-launch `gpu-kernel-profiler` with `--source` mode, open the evidence file named on the `LOCALIZE` line, and pin the change to the specific source line / SASS address it identifies. Do not change a line you have not localized.

### Handoff to Stage 2

After receiving the subagent output:
- Read `summary.txt` and proceed to Stage 2 with the evidence and symptoms as input for research and planning.

## Stage 2: Evidence-Driven Search and Planning

### Goal

Use Stage 1 profile evidence to extract the current bottleneck, search knowledge sources by priority, and write this iteration's plan to `plans/v<N>_plan.md`. This plan is the only input for Stage 3 implementation.

### Execution: Subagent Required

The main agent must directly launch the `gpu-kernel-research` subagent for Stage 2. Do NOT write your own prompt or create an ad-hoc subagent — invoke `gpu-kernel-research` by name as a subagent. The main agent must not perform evidence search or write the plan directly.

**Subagent**: `gpu-kernel-research`

The research subagent owns all search strategy details (progressive three-layer expansion, novelty constraint, layer exhaustion detection). This stage only orchestrates: prepare inputs → launch subagent → receive outputs → hand off to Stage 3.

Subagent launch instruction:

```
Launch subagent: gpu-kernel-research
Task type: research and planning task
Inputs: (see table below)
```

### Input Parameters to Pass

The main agent must provide these parameters when launching the `gpu-kernel-research` subagent:

| Parameter | Source |
|-----------|--------|
| `workspace_path` | Current `kernel_opt_<name>/` absolute path |
| `version` | Current iteration `V<N>` |
| `platform` | From workspace `README.md` (nvidia / amd) |
| `framework` | From workspace `README.md` (triton / cutedsl / flydsl / gluon) |
| `kernel_type` | From workspace `README.md` |
| `profiles_dir` | `profiles/v<N>/` path (Stage 1 output) |
| `memory_dir` | `memory/` directory path |
| `historical_plans` | All `plans/v*_plan.md` paths |
| `stop_conditions` | From workspace `README.md` |
| `gpu_wiki_path` | gpu-wiki root path |

### Output Received

The research subagent returns:

| Field | Usage |
|-------|-------|
| `plan_path` | Written `plans/v<N>_plan.md` — direct input for Stage 3 |
| `evidence_summary` | Bottleneck evidence for iteration report |
| `search_sources` | Sources searched (with new/used annotation) |
| `optimization_actions` | The action(s) to implement in Stage 3 |
| `expected_impact` | Expected performance improvement |
| `risks` | Risk assessment and rollback strategy |

### Handoff to Stage 3

After receiving the subagent output:
- If `plan_path` is returned: proceed to Stage 3 using `plans/v<N>_plan.md` as the implementation spec

## Stage 3: Optimization Implementation

**Subagent**: `kernel-optimize`

### Goal

Implement the optimization actions from `plans/v<N>_plan.md` with clear evidence attribution for each change, validate correctness, and update iteration memory.

The main agent must directly launch the `kernel-optimize` subagent for Stage 3. Do NOT write your own prompt or create an ad-hoc subagent — invoke `kernel-optimize` by name as a subagent. The main agent must not implement optimization changes directly.

Subagent launch instruction:

```
Launch subagent: kernel-optimize
Task type: execution task
Inputs:
  - workspace_path: <workspace absolute path>
  - version: V<N>
  - platform: <nvidia / amd>
  - kernel_file: kernel.py
  - plan_path: plans/v<N>_plan.md
  - profiles_dir: profiles/v<N>/
  - summary_path: profiles/v<N>/summary.txt
  - memory_dir: memory/
  - gpu_wiki_path: <gpu-wiki root path>
```

The `kernel-optimize` subagent will autonomously: validate the plan's evidence attribution, perform localization checks for `LOCALIZE` symptoms (re-profiling with `--source` if needed), implement each optimization action in `kernel.py`, run correctness validation via `test_kernel.py`, and update `memory/v<N>.json` with optimization metadata.

### Output Received

The kernel-optimize subagent returns:

| Field | Usage |
|-------|-------|
| `kernel_file` | Path to modified `kernel.py` |
| `validation_result` | PASS / FAIL from correctness test |
| `memory_file` | Path to updated `memory/v<N>.json` |
| `actions_applied` | List of optimization actions implemented with their evidence attribution |

### Handoff to Stage 4

After receiving the subagent output:
- `validation_result` must be PASS. The `kernel-optimize` subagent is responsible for iteratively fixing correctness failures internally.
- If PASS: proceed to Stage 4 for performance measurement and quality gate.

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

- `profile_iter_nvidia.sh` + `classify_ncu.py`: primary NVIDIA profile entry point (wraps `ncu`). Collects the `.ncu-rep`, parses metrics, and classifies symptoms. Artifact: `summary.txt` (symptoms + search suggestions).
- `extract_nvidia_asm.py`: NVIDIA static SASS evidence for tensor core instructions, load/store width, register spills, and scalar fallback. Also persists each round's raw `kernel.sass` / `kernel.ptx` used by the cross-round text diffs below.
- `ncu_helpers/source_evidence.py` (VeloQ-ported; run automatically by `profile_iter_nvidia.sh --source`, indexed in `source_evidence_manifest.json`): bundles per-line/per-SASS metric attribution (`source_metrics`), warp-stall attribution (`warp_stalls`), and structured source-correlated SASS (`disasm`). **Independent evidence** — read the `analysis/*_run.json` (v1 envelope) or `.txt` digests to localise *which source line / SASS address* a symptom lives on. They do not change `summary.txt`; the `SYMPTOMS` diagnosis still comes only from `classify_ncu.py`, and its `LOCALIZE` line tells you which of these files to open.
- **Cross-round diff (progressive optimization)** — with `profile_iter_nvidia.sh --diff PREV_DIR`, compare two iterations on three layers: `ncu_helpers/row_key.py` for per-row **metric** delta (`diff_*.txt`), `ptx_diff.sh` for the normalized **PTX** instruction-body diff (`diff_ptx.txt` — did the change reach the IR?), and `sass_hist_diff.sh` for the **SASS** instruction-category histogram delta (`diff_sass_hist.txt` — which instruction classes moved?). PTX diff is advisory for JIT frameworks (CuteDSL/Triton); the SASS histogram stays authoritative.
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
- Do not implement optimization actions without corresponding profile evidence.
- Do not continue planning after the quality gate fails.
