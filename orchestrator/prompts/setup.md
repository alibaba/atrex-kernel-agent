# Campaign setup (clean session, run once)

You are the **setup session** for a profile-driven GPU-kernel optimization campaign.
The orchestrator drives the optimization loop after you; your job is to produce the **V0 baseline** and stop.

The workspace already exists at your cwd (`{{WORKSPACE}}`) — it was created by the orchestrator
(`workspace_init.sh` already ran: directory structure, git, and `kernel.py` are in place).
**Do NOT re-run `workspace_init.sh`.**

Environment (resolve all paths against your cwd = the workspace):
- `tools/`, `reference/`, and `skills/` are symlinked into the workspace — read/use them by relative path
  (e.g. `tools/profile_nvidia.sh`, `python tools/memory_manager.py --workspace .`,
  `reference/v_iteration.schema.json`, `skills/gpu-kernel-profile-optimizer/SKILL.md`).
- The gpu-wiki knowledge base is at `{{GPU_WIKI}}` — record this as `gpu_wiki_path` in `README.md` and resolve every `<gpu-wiki>/...` reference to it.

{{HARDWARE}}
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
   Record `performance.latency_us_by_shape` (all sids), `latency_us` (their mean), and `priority_ms`
   (mean over sids of `max(0, latency_ms - roofline.json.shapes[sid].sol_time_ms)`).

Then **STOP**. Do **NOT** enter Stage 2 / any optimization iteration — the orchestrator spawns those as
separate clean sessions. Exit once `memory/v0.json` exists and the baseline is committed.

## Parameters

- platform: `{{PLATFORM}}`
- framework: `{{FRAMEWORK}}`
- kernel_demo: `{{KERNEL_DEMO}}` (already copied to `kernel.py`)
- additional_notes: `{{NOTES}}`
