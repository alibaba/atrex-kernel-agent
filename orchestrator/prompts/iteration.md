# One optimization iteration (clean session)

You are one **clean session** in a profile-driven GPU-kernel optimization campaign.
Run **exactly one cycle** — profile → pick ONE lever → edit → validate → bench → record — then **exit**.

Hard rules for this session:

- **Do NOT loop.** One cycle, then stop. There is no Stage 6 here — the orchestrator owns the outer loop and decides whether another session runs.
- **Do NOT try to reach the final target** in this session. Just make this one cycle count and hand off cleanly.
- The whole point of a clean session is a fresh context: you inherit state from disk, not from a prior conversation.

## Context

- Workspace: `{{WORKSPACE}}` — this is your cwd, and a git repo. **git HEAD is the best kernel so far.**
- You are producing version **v{{N}}**. Previous version: **v{{PREV}}**.
- `tools/`, `reference/`, `skills/`, `reference-projects/`, and `gpu-wiki/` are symlinked into the workspace — read/use them by relative path
  (`python tools/memory_manager.py --workspace .`, `reference/v_iteration.schema.json`).
- `.claude/skills/ncu-report-skill/` — NVIDIA profiling skill (Stage 1).
- `.claude/skills/KernelWiki/` — kernel optimization knowledge base (Stage 2 L1 search).
- `/humanize:gen-plan` — plan generation plugin (Stage 2, loaded via `--plugin-dir`).

{{HARDWARE}}

This prompt is **self-contained** — it defines all stages (1–4), commit/revert rules, and record format.
Evidence format throughout: `evidence -> inference -> action`. Do Stages 1–4 once, then commit/revert/record and exit.

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
current latencies.

## Step B — One cycle (Stages 1–4)

Execute each stage (1–4) directly in this session. All rules are defined inline below.

### Stage 1 — Profile and Bottleneck Evidence Extraction

**Goal**: Profile the current kernel, place outputs in `profiles/v{{N}}/`, and extract concrete bottleneck evidence.

**Execution**: Follow `.claude/skills/ncu-report-skill/SKILL.md` to complete profiling. Adapt its workflow to this iteration context:

1. **Run directory**: Use `profiles/v{{N}}/` as the run directory (replaces the skill's `profile/<run_name>/` convention). Create subdirs `harness/`, `reports/`, `analysis/` under it.
2. **Harness**: Build a standalone harness (with `-lineinfo`) that invokes `kernel.py`'s entry point with representative workload shapes from `workload.jsonl`. Compile into `profiles/v{{N}}/harness/`.
3. **Collection (initial)**: Run **one** ncu profile with `--set full` (overview metrics + PM sampling), outputs to `profiles/v{{N}}/reports/`. Do NOT collect `--set source` in this initial pass.
4. **Analysis**: Parse with the skill's Python helpers (`.claude/skills/ncu-report-skill/helpers/`), work through the six analysis dimensions, and match to the diagnosis playbook. Write analysis artifacts to `profiles/v{{N}}/analysis/`.
5. **Report**: Write the final report to `profiles/v{{N}}/REPORT.md` per the skill's template — evidence-backed, ranked by expected impact.

   **AMD** — run the AMD profiling path instead:
   ```bash
   bash tools/profile_kernel.sh kernel.py --output-dir profiles/v{{N}}
   ```
   Collects ATT, PMC, and ASM artifacts.

**Localization rule (mandatory)**: The initial collection is `--set full` only — no source-level counters. Escalate to a **second** collection with `--set source --section SourceCounters` only when the REPORT.md identifies a localization-worthy symptom **and** Stage 3 is about to choose a concrete code change based on that symptom. Then pin the change to the specific source line / SASS address the per-line stall analysis identifies.

**Output**: `profiles/v{{N}}/REPORT.md` with bottleneck evidence, diagnosis, and ranked recommendations — this feeds Stage 2.

### Stage 2 — Evidence-Driven Research and Planning

**Goal**: Use Stage 1 evidence to find one optimization path and generate `plans/v{{N}}_plan.md` via `humanize:gen-plan`.

Execution steps:

1. **Mandatory reads**: workspace `README.md`, `gpu-wiki/README.md`, all unmasked `memory/v*.json`, historical `plans/v*_plan.md`.
2. **Parse historical Search Logs** from prior plans to build a used-knowledge set (deduplication reference).
3. **Determine stall count**: Count the number of consecutive most-recent reverted versions (no improvement) from the memory summary. Record as `STALL_COUNT`.
4. **Research strategy** (adaptive based on stall count):
   - **Normal mode** (`STALL_COUNT < 3`): No novelty requirement. Reusing known directions from `open_directions`, prior search findings, or profile evidence is not prohibited.
   - **Forced expansion mode** (`STALL_COUNT >= 3`): The previous directions have failed repeatedly — you MUST expand the search space. Do not limit searches to the current kernel's DSL/language; look at optimization techniques from **other languages or DSLs targeting the same or similar hardware architectures** (e.g., CUDA C++ tricks applicable to Triton, or CuteDSL patterns that inspire Gluon rewrites) and adapt the ideas. Execute the full three-layer progressive search (strict order):
     - **L1 (KernelWiki + gpu-wiki)**: Translate bottleneck diagnoses from `profiles/v{{N}}/REPORT.md` into search keywords. Search `.claude/skills/KernelWiki` first (it is linked in the workspace — use its `scripts/query.py` for semantic search on NVIDIA SM90/SM100). Then navigate `gpu-wiki/docs/`, `gpu-wiki/reference-kernels/` using Symptom-Driven Retrieval guidance in `gpu-wiki/README.md`.
     - **L2 (reference-projects)**: Only if L1 yields no new actionable path. Search relevant modules in `reference-projects/` for implementation patterns.
     - **L3 (public web)**: Only if L1+L2 yield nothing new. Use web search for papers, docs, or community posts.
     - The draft MUST contain at least one `New? = Yes` entry. If all layers produce no new finding, report search space exhaustion and stop — do not fabricate a draft or invoke gen-plan.
5. **Stop early**: Once you find **one viable optimization direction** with supporting evidence, proceed to draft immediately. Do not exhaustively search all layers.
6. **Write draft** to `plans/v{{N}}_draft.md` — a concise summary of:
   - Input Evidence: key metrics and diagnoses from `profiles/v{{N}}/REPORT.md`
   - Search findings: what you found (with Layer, New? annotations) and the chosen optimization direction
   - Constraints: target framework, platform, correctness requirements
   - Stall context: current `STALL_COUNT` and whether forced expansion was triggered
7. **Generate plan** via humanize:
   ```
   /humanize:gen-plan --input plans/v{{N}}_draft.md --output plans/v{{N}}_plan.md --direct
   ```
   This produces a structured plan with acceptance criteria. Use `--direct` to skip convergence rounds (one-shot generation appropriate for a single optimization action within the iteration loop).

**Output**: `plans/v{{N}}_plan.md` (generated by humanize) — the sole input for Stage 3.

### Stage 3 — Optimization Implementation

**Goal**: Implement exactly one optimization category from `plans/v{{N}}_plan.md` with clear evidence attribution.

Execution steps:

1. **Framework learning** (if needed): If the plan references framework APIs or operator interfaces you're unfamiliar with, search `gpu-wiki/reference-kernels/` or reference-projects first.
2. **Localization check**: If the change targets a symptom with a `LOCALIZE` line, ensure you have re-profiled with `--source` (see Stage 1 localization rule). Change only the specific line(s) the evidence identifies.
3. **Implement** the optimization action in `kernel.py`:
   - Change only one category per iteration (e.g., vectorized load only, swizzle only, double buffering only).
   - Each change must have clear evidence attribution (`evidence -> inference -> action`).
   - Do not mix unrelated refactors, formatting, or cleanup.
4. **Correctness validation** — immediately after editing:
   ```bash
   python test_kernel.py
   ```
   If validation fails, iteratively fix until it passes. Do not proceed to Stage 4 with broken correctness.
5. **Create/update memory**:
   ```bash
   python tools/memory_manager.py create --workspace . --version v{{N}}
   python tools/memory_manager.py update --workspace . --version v{{N}} \
       --set 'optimization.action_category=<category>' \
       --set 'optimization.action_description=<description>'
   ```

**Output**: Modified `kernel.py` with correctness PASS, updated `memory/v{{N}}.json`.

### Stage 4 — Validate + Bench

**Goal**: Run the SOL-ExecBench harness for full performance measurement and quality gate.

Execution steps:

1. **Run the benchmark harness** with timeout guard:
   ```bash
   timeout 1800 python test_kernel.py --version v{{N}}
   ```
   This runs the real `sol-execbench` evaluator over **every workload in `workload.jsonl`** with each workload's own tolerance. Do NOT edit `test_kernel.py` or hand-roll a separate test.

2. **Metrics recorded** (by the harness into `memory/v{{N}}.json`):
   - `performance.latency_us` = **geomean** of per-workload kernel latency (primary objective: minimize)
   - `performance.latency_us_by_shape` — per-workload latency keyed by workload `uuid`
   - `performance.speedup_vs_ref_geomean` — geomean speedup vs reference
   - `correctness.status` / `quality_gate.result` — PASS iff ALL workloads pass

3. **Measurement reliability guard**: Before accepting a large delta (especially regressions >30%), verify GPU is not occupied by other processes. Switch to a free GPU via `CUDA_VISIBLE_DEVICES` if needed and re-measure.

4. **Quality gate**: PASS requires correctness PASS + geomean latency drop vs HEAD beyond measurement noise. A flat-within-noise result is NOT an improvement.

5. If `solution.json` needs updating (e.g. framework migration changed dependencies/entry_point), update it before benching.

**Output**: `memory/v{{N}}.json` with full performance data and quality gate result.

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
  # win:  kernel.py already committed in Step C; amend the hash into memory.
  # revert: commit just the record —
  git add memory/v{{N}}.json plans/v{{N}}_draft.md plans/v{{N}}_plan.md && \
    git commit -m "v{{N}}: reverted (<reason>) — dead-end recorded"
  ```

## Finish

Print one line and stop — do **not** start another cycle:

```
v{{N}}: committed (+X.X%)   |   v{{N}}: reverted (<reason>)
```
