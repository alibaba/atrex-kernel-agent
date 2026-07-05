# Campaign setup (clean session, run once)

You are the **setup session** for a profile-driven GPU-kernel optimization campaign.
The orchestrator drives the optimization loop after you; your job is to produce the **V0 baseline** and stop.

The workspace already exists at your cwd (`{{WORKSPACE}}`) — it was created by the orchestrator
(`workspace_init.sh` already ran: directory structure, git, and `kernel.py` are in place).
**Do NOT re-run `workspace_init.sh`.**

Environment (resolve all paths against your cwd = the workspace):
- `tools/`, `reference/`, `skills/`, `reference-projects/`, and `gpu-wiki/` are symlinked into the workspace — read/use them by relative path
  (e.g. `python tools/memory_manager.py --workspace .`, `reference/v_iteration.schema.json`).
- `.claude/skills/ncu-report-skill/` — NVIDIA profiling skill.
- `.claude/skills/KernelWiki/` — kernel optimization knowledge base.

{{HARDWARE}}
Do the following, in order, but only through baseline:

1. **Step 0 — Hardware specs + Roofline.** Source
   every hardware spec from `gpu-wiki/` (**no fabrication** — every spec value must cite a gpu-wiki path),
   do the Roofline analysis, compute absolute targets (`hardware peak * 90%`), and write `Hardware Spec`,
   the Roofline analysis, and `Stop Conditions` into the workspace `README.md`.
2. **Write `README.md`** — static config from the parameters below + Step 0 outputs (use `reference/README.md` as the template).
3. **Stage 1 — Baseline.** Launch the `gpu-kernel-baseline` subagent (by name). You may spawn it in the background, but **you MUST wait for it to complete before you exit**: implement `kernel.py` + `test_kernel.py`,
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
