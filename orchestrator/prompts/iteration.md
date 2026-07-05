# Optimization Iteration (Clean Session)

You are a **clean session** in a profile-driven GPU-kernel optimization campaign.
Workflow: profile → research → implement → validate → bench → record.

You inherit state from disk, not from a prior conversation.

## CRITICAL — Subagent Execution Rules (MANDATORY)

Every subagent (`gpu-kernel-profiler`, `gpu-kernel-research`, `kernel-optimize`, or **any** Agent tool call) **MUST** be launched **synchronously**:

- **Always** pass `run_in_background: false` on every Agent tool call.
- This makes the call **blocking** — your session will not continue until the subagent finishes and returns its result.
- **NEVER** omit `run_in_background` (it defaults to `true`, which is WRONG for this workflow).
- **NEVER** end your turn (`end_turn`) after launching a subagent. If the subagent has not returned a result yet, you **MUST NOT** stop.
- After launching a subagent synchronously, you **MUST** process its return value before doing anything else.

**Why this matters**: the orchestrator runs you as a one-shot `--print` session. If you end your turn before a subagent finishes, the session exits, the subagent is killed, `memory/v{{N}}.json` is never created, and the iteration is wasted.

## Context

- Workspace: `{{WORKSPACE}}` — this is your cwd, and a git repo. **git HEAD is the best kernel so far.**
- You are producing version **v{{N}}**. Previous version: **v{{PREV}}**.
- `tools/`, `reference/` are symlinked into the workspace — read/use them by relative path
  (`tools/profile_nvidia.sh`, `python tools/memory_manager.py --workspace .`,
  `reference/v_iteration.schema.json`).
  The gpu-wiki path is recorded as `gpu_wiki_path` in `README.md`.

{{HARDWARE}}

## Step A — Learn from prior sessions (read; do not redo their work)

1. Read `README.md` — the **Goal** (minimize the geomean of per-workload kernel latency), config (platform, target framework, `gpu_wiki_path`), and the ground-truth files. There are no pre-baked Stop Conditions — the orchestrator owns termination; your job is to cut the geomean latency this cycle while keeping every workload correct.
2. Cross-version digest — `python tools/memory_manager.py summary --workspace .` (where we are; the trajectory).
3. The latest entry, in full — `python tools/memory_manager.py read --workspace . --version v{{PREV}}`. Pay attention to:
   - **`open_directions`** — candidate leads the previous session left for you.
   - **`search_log` + `pitfalls_and_fixes`** across the digest — **recorded dead-ends. Do NOT repeat them.**
4. Profile reuse — if `profiles/v{{PREV}}/` holds a profile of the *current* HEAD kernel (carried forward),
   you may reuse it instead of re-profiling. Otherwise profile fresh in Step B.

**`open_directions` are priors, not orders.** Pick the most promising lead — **or**, if a fresh look at the
profile reveals a better lever, pursue that instead. The only hard constraint is: don't re-run a recorded dead-end.

**Consider shape bucketing when the evidence supports it.** Check `performance.latency_us_by_shape` and the
profile: if different-scale shapes are bottlenecked differently, a valid category is **shape specialization** —
group shapes into a few buckets of similar scale (not one path per shape) and dispatch inside `run()` to a
subkernel or tiling/block config per bucket. Many kernels don't need this — decide from memory, research, and
current latencies. See "Shape Bucketing" in Step B below.

## Step B — Stages 1–4

Each stage MUST use the designated subagent — do not run profiling, research, or optimization directly.

### Evidence Format

Every optimization action in this workflow follows the **evidence → inference → action** chain:
- **Evidence**: concrete metric or symptom from the profiler (e.g., "ds_read bank conflict stall 12 cycles")
- **Inference**: what the evidence implies about the bottleneck (e.g., "shared memory access pattern causes serialization")
- **Action**: specific code change to address the inferred bottleneck (e.g., "apply XOR16 swizzle layout to shared memory allocation")

Do not implement any action without completing this chain.

### Localization Rule (Mandatory)

The `--source` flag is always included in profiling to ensure source-level localization evidence is produced every run. When `summary.txt` emits a `LOCALIZE` line, open the evidence file it names and pin the change to the specific source line / SASS address it identifies. **Do not change a line you have not localized.**

### Shape Bucketing — When the Evidence Calls For It

Not every kernel needs this: simple / uniform kernels are best kept single-path. But the workload set spans very different scales, and a lever that wins on large shapes (bigger tiles, deeper pipelining) can lose on small ones (launch / occupancy / latency bound). Decide per iteration from evidence — `performance.latency_us_by_shape`, the profile, and prior memory / research: if shapes of different scales are bottlenecked differently, group them into a few **buckets of similar scale** (not one path per shape) and give each bucket its own path — a subkernel, or at least its own tile / block config — behind a dispatcher inside `run()`. This is one attributable category ("shape specialization"), stays within the DPS contract (`run()` is still a single entry point), and per-bucket gains land directly in the geomean goal.

---

### Stage 1 — Profile (subagent: `gpu-kernel-profiler`)

Launch the `gpu-kernel-profiler` subagent to profile the current kernel and extract bottleneck evidence.

```
Launch subagent: gpu-kernel-profiler
Task type: execution task
Inputs:
  - workspace_path: <workspace absolute path>
  - version: V{{N}}
  - platform: <nvidia / amd>
  - kernel_file: kernel.py
  - gpu_wiki_path: <gpu-wiki root path>
  - previous_profiles_dir: <profiles/v{{PREV}}/ if exists, otherwise omit>
```

The profiler subagent will: create `profiles/v{{N}}/`, run platform-specific profiling tools (`profile_nvidia.sh --source` for NVIDIA or `profile_kernel.sh` for AMD), perform SASS/assembly analysis, and produce `profiles/v{{N}}/summary.txt` with bottleneck evidence and symptoms.

**Output**: `profiles_dir` (path to `profiles/v{{N}}/`) and `summary_path` (path to `summary.txt`).

**Profile reuse after revert**: if the current `kernel.py` matches a previous version's code (e.g. after a revert in the prior iteration), skip profiling and reuse that version's profile directory directly — the profile follows the code.

---

### Stage 2 — Research and Plan (subagent: `gpu-kernel-research`)

Launch the `gpu-kernel-research` subagent to perform thorough cross-layer research and write the optimal plan.

```
Launch subagent: gpu-kernel-research
Task type: research and planning task
Inputs:
  - workspace_path: <workspace absolute path>
  - version: V{{N}}
  - platform: <nvidia / amd>
  - framework: <triton / cutedsl / flydsl / gluon>
  - kernel_type: <from README.md>
  - profiles_dir: profiles/v{{N}}/
  - memory_dir: memory/
  - historical_plans: <all plans/v*_plan.md paths>
  - stop_conditions: <from README.md>
  - gpu_wiki_path: <gpu-wiki root path>
```

The research subagent searches knowledge sources (progressive three-layer expansion: gpu-wiki → reference-projects → public net) and writes `plans/v{{N}}_plan.md` containing exactly **one optimal optimization action** — the single best evidence-backed path selected after thorough research.

**Output**: `plan_path` (written plan), `optimization_action` (the selected optimal action with justification), `expected_impact`, `risks`.

---

### Stage 3 — Implement (subagent: `kernel-optimize`)

Launch the `kernel-optimize` subagent to implement the optimization action from the plan.

```
Launch subagent: kernel-optimize
Task type: execution task
Inputs:
  - workspace_path: <workspace absolute path>
  - version: V{{N}}
  - platform: <nvidia / amd>
  - kernel_file: kernel.py
  - plan_path: plans/v{{N}}_plan.md
  - profiles_dir: profiles/v{{N}}/
  - summary_path: profiles/v{{N}}/summary.txt
  - memory_dir: memory/
  - gpu_wiki_path: <gpu-wiki root path>
```

The kernel-optimize subagent will: validate the plan's evidence attribution, perform localization checks, implement the optimization action in `kernel.py`, validate correctness via lightweight `do_bench`, and update `memory/v{{N}}.json`.

**Output**: `kernel_file`, `validation_result` (PASS/FAIL), `performance_validated` (YES/NO/INEFFECTIVE), `improvement_summary`, `memory_file`, `actions_applied`.

**Performance-regression revert rule**: when kernel-optimize rolls back code to a previous version due to performance regression or no improvement, it MUST reuse the profile results from that reverted-to version (e.g. `profiles/v{{PREV}}/`) as the baseline — do NOT re-profile the reverted code.

---

### Stage 4 — Validate + Bench (subagent required)

Launch a validation subagent to run the full correctness + performance measurement.

**Execution**: the SOL-ExecBench harness with timeout guard:

```bash
timeout 1800 python test_kernel.py --version v{{N}}
```

This runs the real `sol-execbench` evaluator over **EVERY workload in `workload.jsonl`** (the full ground-truth shape set) with each workload's own tolerance, and records into `memory/v{{N}}.json`:
- `performance.latency_us` = **geomean** of per-workload latency (the primary objective — minimize)
- `performance.latency_us_by_shape` (keyed by workload `uuid`)
- `performance.speedup_vs_ref_geomean`
- `correctness.status` / `quality_gate.result` = PASS iff **all** workloads pass

Do NOT hand-roll a separate correctness test or edit `test_kernel.py`. The harness exits non-zero if any workload fails. `solution.json` reads the live `kernel.py` from disk — if you changed languages/dependencies/entry_point (e.g. migrating framework), update `solution.json` `spec` to match before benching.

**Variance-aware**: a geomean delta only counts as real if it clears measurement noise (flat-within-noise is *not* an improvement).

**Measurement Reliability Guard**: before accepting a large performance delta (especially regressions > 30%), verify the measurement is trustworthy:
1. Compare `kernel.py` and `test_kernel.py` against the previous committed version (`git diff HEAD -- kernel.py test_kernel.py`). If both are unchanged, any large latency change is an environment artifact.
2. Check GPU occupancy: `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader`. If the current GPU shows significant memory usage (> 1GB from other processes), the measurement is unreliable.
3. If the GPU is occupied, switch to a free GPU by setting `CUDA_VISIBLE_DEVICES=<free_gpu_id>` and re-run.
4. If both files are unchanged but latency differs by > 30%, do not treat it as a real regression — re-measure on a confirmed-free GPU.

**Quality Gate** — pass conditions:
- Correctness validation PASS
- No unacceptable performance regression (or clearly explained and supports later optimization)

**On failure** (FAIL or TIMEOUT_FAIL): `git reset --hard HEAD`, record the failure in `memory/v{{N}}.json` under `pitfalls_and_fixes` and `quality_gate`.

## Step C — Commit or revert (mechanical, no discretion)

- **Real win** (all workloads PASS **and** geomean kernel latency `performance.latency_us` drops vs HEAD by more
  than noise) → **commit** (skill Stage 5 format). This kernel becomes the new HEAD/best.
- **Otherwise** (geomean regression, flat-within-noise, or any workload FAIL) → `git reset --hard HEAD` to restore
  the best-known `kernel.py`. **Never commit a regression.** The goal is to minimize the geomean of kernel latency.

## Step D — Record + hand off (ALWAYS — win *or* dead-end)

Fill `memory/v{{N}}.json` regardless of outcome (`memory_manager.py create` then `update`, per the skill). It is
untracked until you commit it, so it survives `git reset --hard`.

- `performance`, `correctness`, `profile_evidence`, `optimization` (what you tried) — per `reference/v_iteration.schema.json`.
- `quality_gate` + `git_commit_hash` — set the hash if you committed; leave `null` if you reverted.
- **If this cycle was a dead-end**, record *why* in `search_log` / `pitfalls_and_fixes` so the next session doesn't repeat it.
- **`open_directions`** — up to **3** candidate leads for the *next* session, most-promising first (fewer is fine
  if you only found 1–2). Include any **unfinished-but-promising thread** you didn't get to. These are the "word
  for the next session":

  ```bash
  python tools/memory_manager.py update --workspace . --version v{{N}} \
    --set 'open_directions=[{"direction":"<lever>","rationale":"<evidence/why promising>"}]'
  ```

- **Profile-carry-forward** — if you committed, leave the post-edit profile in `profiles/v{{N}}/` so the next
  session can reuse it instead of re-profiling.
- **Commit the record** even on a revert, so the next session sees the dead-end:

  ```bash
  # win:  kernel.py already committed in Step C; amend the hash in per Stage 5.
  # revert: commit just the record —
  git add memory/v{{N}}.json plans/v{{N}}_plan.md && \
    git commit -m "v{{N}}: reverted (<reason>) — dead-end recorded"
  ```

## Finish

Print one line:

```
v{{N}}: committed (+X.X%)   |   v{{N}}: reverted (<reason>)
```
