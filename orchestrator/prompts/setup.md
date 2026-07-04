# Campaign setup (clean session, run once)

You are the **setup session** for a profile-driven GPU-kernel optimization campaign.
The orchestrator drives the optimization loop after you; your job is to produce the **V0 baseline** and stop.

The workspace already exists at your cwd (`{{WORKSPACE}}`) — the orchestrator created it
(directory structure, git, `kernel.py`, and `CLAUDE.md` are in place). **Do NOT re-create the workspace.**

Environment (resolve all paths against your cwd = the workspace):
- `tools/`, `reference/`, and `skills/` are symlinked into the workspace — read/use them by relative path
  (e.g. `tools/profile_nvidia.sh`, `python tools/memory_manager.py --workspace .`,
  `reference/v_iteration.schema.json`, `skills/gpu-kernel-profile-optimizer/SKILL.md`).
- Subagents are launched **by name** (e.g. the `gpu-kernel-baseline` subagent) — the orchestrator has
  made them discoverable, so you do not read them by a workspace path.
- The gpu-wiki knowledge base is at `{{GPU_WIKI}}` — record this as `gpu_wiki_path` in `README.md` and resolve every `<gpu-wiki>/...` reference to it.

{{HARDWARE}}

## CRITICAL — Subagent Execution Rules (MANDATORY)

Every subagent (`gpu-kernel-baseline`, `gpu-kernel-profiler`, `gpu-kernel-research`, `kernel-optimize`, or **any** Agent tool call) **MUST** be launched **synchronously**:

- **Always** pass `run_in_background: false` on every Agent tool call.
- This makes the call **blocking** — your session will not continue until the subagent finishes and returns its result.
- **NEVER** omit `run_in_background` (it defaults to `true`, which is WRONG for this workflow).
- **NEVER** end your turn (`end_turn`) after launching a subagent. If the subagent has not returned a result yet, you **MUST NOT** stop — keep waiting or send a follow-up.
- After launching a subagent synchronously, you **MUST** process its return value before doing anything else (including stopping).

**Why this matters**: the orchestrator runs you as a one-shot `--print` session. If you end your turn before a subagent finishes, the session exits, the subagent is killed, and the entire optimization campaign aborts. There is no second chance unless the orchestrator retries — and retries waste the token budget.

---

Do the following, in order, but only through baseline:

1. **Step 0 — Hardware specs + Roofline.** Source
   every hardware spec from `{{GPU_WIKI}}/` (**no fabrication** — every spec value must cite a gpu-wiki path),
   do the Roofline analysis, compute absolute targets (`hardware peak * 90%`), and write `Hardware Spec`,
   the Roofline analysis, and `Stop Conditions` into the workspace `README.md`.
2. **Write `README.md`** — static config from the parameters below + Step 0 outputs (use `reference/README.md` as the template).
3. **Stage 1 — Baseline.** Launch the `gpu-kernel-baseline` subagent (by name): implement `kernel.py` + `test_kernel.py`,
   validate correctness and baseline performance, write `baseline_report.md`, write `memory/v0.json` (via
   `tools/memory_manager.py`), and `git commit` ("V0: baseline kernel").
   **`test_kernel.py` MUST bench every shape in the workspace `shapes.json`** (the full ground-truth set, keyed
   by integer sid) — not a hand-picked subset. This shape set + harness is the immutable per-campaign benchmark
   methodology (see `reference/CLAUDE.md` "Benchmark Harness Integrity"); later iterations reuse it unchanged.
   Record `performance.latency_us_by_shape` (all sids) and `latency_us` (their mean). Do **not** compute a
   priority here — the orchestrator derives the (anchor-weighted) priority from this per-shape latency.

Then **STOP**. Do **NOT** enter Stage 2 / any optimization iteration — the orchestrator spawns those as
separate clean sessions. Exit once `memory/v0.json` exists and the baseline is committed.

## Parameters

- platform: `{{PLATFORM}}`
- framework: `{{FRAMEWORK}}`
- kernel_demo: `{{KERNEL_DEMO}}` (already copied to `kernel.py`)
- additional_notes: `{{NOTES}}`
