---
name: gpu-kernel-profile-optimizer
description: |
  Profile-driven GPU kernel optimization skill. Use this skill to run ONE optimization iteration in a temporary git workspace: profile evidence extraction, evidence-driven search and planning, single-category optimization, validation, memory update, git commit, and handoff. The outer iteration loop is owned by the orchestrator (orchestrator/optimize.py), not by this skill — run one cycle, then exit.
---

# GPU Kernel Profile Optimizer

## When to Use

Use this skill when the user asks to:

- Optimize an existing GPU kernel.
- Continue improving code based on `./tools/profile_nvidia.sh` (`ncu`), `./tools/profile_kernel.sh`, ATT, PMC, or ASM evidence.
- Use profiling evidence rather than intuition to improve performance.

## Overall Principles

This skill runs **exactly ONE iteration** end-to-end, then exits. It does not loop: the orchestrator (`orchestrator/optimize.py`) spawns the next iteration as a separate clean session. Follow the stage order below. Commands, wiki searches, profiling, validation, records, and commits must be completed inside their corresponding stage.

```text
Stage 1 Profile and evidence extraction
Stage 2 Evidence-driven search and planning
Stage 3 Single-category optimization implementation
Stage 4 Performance, correctness, and quality gate
Stage 5 Memory update, git commit, and handoff
```

Constraints:

- Do not skip stages.
- Do not edit code without profile evidence.
- Implement only one optimization category per iteration so the result can be attributed.
- Each optimization action must have clear profile evidence attribution.
- If the quality gate fails, revert to the previous commit, record the failure, and stop the iteration.
- Run exactly one iteration, then exit. Do not start another cycle — the orchestrator decides whether the next iteration runs.

**Shape bucketing — when the evidence calls for it.** Not every kernel needs this: simple / uniform kernels are best kept single-path. But the workload set spans very different scales, and a lever that wins on large shapes (bigger tiles, deeper pipelining) can lose on small ones (launch / occupancy / latency bound). Decide per iteration from evidence — `performance.latency_us_by_shape`, the profile, and prior memory / research: if shapes of different scales are bottlenecked differently, group them into a few **buckets of similar scale** (not one path per shape) and give each bucket its own path — a subkernel, or at least its own tile / block config — behind a dispatcher inside `run()`. This is one attributable category ("shape specialization"), stays within the DPS contract (`run()` is still a single entry point), and per-bucket gains land directly in the geomean goal.

## Workspace Layout

Maintain this structure:

```text
<workspace>/            # the run root == your current working directory
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

The `gpu-kernel-profiler` subagent will autonomously: create `profiles/v<N>/` directory, run platform-specific profiling tools (`profile_nvidia.sh` for NVIDIA or `profile_kernel.sh` for AMD), perform SASS/assembly analysis, and extract structured bottleneck evidence.

### Output Received

The profiler subagent returns:

| Field | Usage |
|-------|-------|
| `profiles_dir` | Path to `profiles/v<N>/` directory |
| `summary_path` | Path to `profiles/v<N>/summary.txt` — unified evidence summary for both NVIDIA and AMD |

`summary.txt` contains all extracted evidence: key metrics, `SYMPTOMS`, `LOCALIZE` (if applicable), and search suggestions. Both NVIDIA and AMD platforms produce this file as the single structured output.

### Localization rule (mandatory)

The `--source` flag is **mandatory** (always included) to ensure source-level localization evidence is produced every run. There is no separate "escalation" step — the profiler subagent always collects source-correlated metrics in a single pass.

- **Evidence usage** — when `summary.txt` emits a `LOCALIZE` line, open the evidence file it names and pin the change to the specific source line / SASS address it identifies. Do not change a line you have not localized.

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
| `workspace_path` | The workspace (run root) absolute path — your current working directory |
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

The `kernel-optimize` subagent will autonomously: validate the plan's evidence attribution, perform localization checks for `LOCALIZE` symptoms (source-level evidence is already available from the `--source` profile), implement each optimization action in `kernel.py`, run correctness validation via `test_kernel.py`, and update `memory/v<N>.json` with optimization metadata.

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

Validation + bench is the SOL-ExecBench harness — run it with an outer hang-backstop:

```bash
timeout 1800 python test_kernel.py --version v<N>   # SOL evaluator over all workloads; writes memory/v<N>.json
```

- The harness runs the real `sol-execbench` evaluator (its own per-eval subprocess timeout applies) over every
  workload in `workload.jsonl` with each workload's own tolerance, then records the metrics below. Do NOT edit it.
- A non-zero exit (some workload failed / hung) counts as a quality-gate failure; revert and record the cause.

### Metrics to Record

The harness writes these into `memory/v<N>.json` — do not hand-fabricate them:

- **`performance.latency_us` = GEOMEAN of per-workload kernel latency — the primary objective (minimize).**
- `performance.latency_us_by_shape` — per-workload latency (keyed by workload `uuid`).
- `performance.speedup_vs_ref_geomean` — geomean speedup vs the reference.
- `correctness.status` / `quality_gate.result` — PASS iff ALL workloads pass.

Judge iterations by the geomean latency (a win = all workloads still PASS AND geomean drops vs HEAD beyond noise).
TFLOPS / bandwidth / peak-utilization (via `tools/compute_utilization.py`) are OPTIONAL enrichment for reasoning
about how close the kernel is to the roofline — record them when useful, but they are not the stop metric.

### Iteration Data

Update `memory/v<N>.json` with performance, correctness, and ISA progress data following the schema defined in `reference/v_iteration.schema.json`.

### Quality Gate

Pass conditions:

- Correctness validation PASS.
- No unacceptable performance regression, or the regression is clearly explained and supports later optimization.
- No severe ISA regression, such as new spills or a large occupancy drop.

### Measurement Reliability Guard

Before accepting a large performance delta (especially regressions > 30%), verify the measurement is trustworthy:

1. Compare `kernel.py` and `test_kernel.py` against the previous committed version (`git diff HEAD -- kernel.py test_kernel.py`). If both are unchanged, the kernel binary and benchmark harness are identical — any large latency change is an environment artifact, not a real regression.
2. Check GPU occupancy: `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader`. If the current GPU shows significant memory usage (> 1GB from other processes) or high utilization from other workloads, the measurement is unreliable.
3. If the GPU is occupied, switch to a free GPU by setting `CUDA_VISIBLE_DEVICES=<free_gpu_id>` and re-run the benchmark.
4. If both files are unchanged but latency differs by more than 30%, do not treat it as a real regression. Re-measure on a confirmed-free GPU before recording.

### Failure Handling

If the subagent returns FAIL or TIMEOUT_FAIL, the main agent must:

```bash
git reset --hard HEAD
```

Record the failure reason, skip further planning for this iteration, and do not enter the next iteration. Write the failure into `memory/v<N>.json` under `pitfalls_and_fixes` and `quality_gate`, then commit a revert marker such as `V5: revert V4 (occupancy 25% -> 12%)` when appropriate.

## Stage 5: Memory Update, Git Commit, and Handoff

Goal: finalize `memory/v<N>.json` with quality gate result and git commit hash, commit, then hand off to the next session.

### Procedure

1. Verify that `memory/v<N>.json` has been updated by Stage 4 with:
   - Performance metrics (TFLOPS, bandwidth, utilization)
   - Correctness result and `rel_err`
   - ISA metric progress
   - All values must come from actual measurements; do not re-measure or fabricate.

2. Update `memory/v<N>.json` using `tools/memory_manager.py`:

   ```bash
   python tools/memory_manager.py update --workspace . --version v<N> \
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
python tools/memory_manager.py update --workspace . --version v<N> \
    --set "git_commit_hash=$HASH"
git add memory/v<N>.json
git commit --amend --no-edit
```

Examples:

- `V3: 150 TFLOPS / 1.5 GB/s | XOR16 swizzle layout (bottleneck: ds_read bank conflict stall 12 cycles)`
- `V5: revert V4 (occupancy 25% -> 12%)`

### Handoff to the next session

Before exiting, record the "word for the next session" in `memory/v<N>.json` — this is how a fresh
clean session (which has none of your conversation context) picks up where you left off:

- **`open_directions`** — up to **3** most-promising untried levers for the next session (fewer if you
  found fewer), including any unfinished-but-promising thread you did not get to. These are *priors, not
  orders*: the next session may pick one or choose a better lever it sees from a fresh profile.

  ```bash
  python tools/memory_manager.py update --workspace . --version v<N> \
      --set 'open_directions=[{"direction":"<lever>","rationale":"<evidence/why promising>"}]'
  ```

- **Dead-ends** — if this iteration regressed or led nowhere, record *why* in `search_log` /
  `pitfalls_and_fixes` so the next session does not retry it.
- **Profile-carry-forward** — if you committed, leave the post-edit profile in `profiles/v<N>/` so the
  next session can reuse it instead of re-profiling.

## End of Iteration

This skill runs one iteration. After Stage 5, **exit** — do not return to Stage 1 and do not start another
cycle. The orchestrator (`orchestrator/optimize.py`) reads `memory/v<N>.json`, decides whether the budget or
stop conditions are met (default: peak utilization >= 90% on a committed, correctness-PASS iteration), and
spawns the next clean session if needed.

Print a one-line status and stop:

```text
v<N>: committed (+X.X%)   |   v<N>: reverted (<reason>)
```

## Appendix: Tool-to-Evidence Mapping

Different tools provide different layers of evidence and must not be mixed:

- `profile_nvidia.sh` + `classify_ncu.py`: primary NVIDIA profile entry point (wraps `ncu`). Collects the `.ncu-rep`, parses metrics, and classifies symptoms. Artifact: `summary.txt` (symptoms + search suggestions).
- `extract_nvidia_asm.py`: NVIDIA static SASS evidence for tensor core instructions, load/store width, register spills, and scalar fallback.
- `ncu_helpers/source_evidence.py` (VeloQ-ported; run automatically by `profile_nvidia.sh --source`, indexed in `source_evidence_manifest.json`): bundles per-line/per-SASS metric attribution (`source_metrics`), warp-stall attribution (`warp_stalls`), and structured source-correlated SASS (`disasm`). **Independent evidence** — read the `analysis/*_run.json` (v1 envelope) or `.txt` digests to localise *which source line / SASS address* a symptom lives on. They do not change `summary.txt`; the `SYMPTOMS` diagnosis still comes only from `classify_ncu.py`, and its `LOCALIZE` line tells you which of these files to open. Use `ncu_helpers/row_key.py` (or `profile_nvidia.sh --diff`) to compare two iterations row-by-row.
- `profile_kernel.sh`: primary AMD profile entry point, collecting ATT, PMC, and ASM for instruction width, spills, and LDS access patterns.
- `kernel.s`: AMD assembly evidence for load/store width, LDS instruction form, scratch operations, and spills.

## Appendix: Prohibited Actions

- Do not skip `<gpu-wiki>/README.md`.
- Do not begin the iteration without reading `README.md` and the latest unmasked `memory/v*.json` files (including the previous `open_directions` and recorded dead-ends).
- Do not loop: run exactly one iteration, then exit. The orchestrator spawns the next session.
- Do not read `memory/v*.json` files where `masked: true` as active iteration data.
- Do not reuse profile artifacts across versions.
- Do not commit performance conclusions without correctness validation.
- Do not record only latency without TFLOPS, bandwidth, and peak-utilization ratios.
- Do not provide unsourced optimization suggestions.
- Do not implement optimization actions without corresponding profile evidence.
- Do not continue planning after the quality gate fails.
