# Triton → Gluon conversion (clean session, convert-only)

You are one **clean session** whose ONLY job is to convert the current Triton `kernel.py` to **Gluon**.
This is a framework lowering, **not** an optimization: preserve the algorithm, tiling, and block sizes.
Gluon is a lower-level DSL, so the *following* sessions will optimize deeper — your job is to hand them
a correct Gluon kernel. Do **one** conversion, then exit.

## Context
- Workspace: `{{WORKSPACE}}` — your cwd, a git repo. **git HEAD is the best Triton kernel so far.**
- You are producing version **v{{N}}** (previous: v{{PREV}}).
- `tools/`, `reference/`, `skills/` are symlinked in; gpu-wiki is at `{{GPU_WIKI}}`.

{{HARDWARE}}

## Read first (one file, then source on demand)
Read the single conversion sheet for the **real arch above** — do not read the others:
- `sm_100` / `sm_103` (Blackwell data-center, B200/B300) → `{{GPU_WIKI}}/docs/converter/nvidia/blackwell.md`
- `sm_90` (Hopper) → `{{GPU_WIKI}}/docs/converter/nvidia/hopper.md`
- `gfx94*` (CDNA3) → `{{GPU_WIKI}}/docs/converter/amd/cdna3.md`; `gfx95*` (CDNA4) → `{{GPU_WIKI}}/docs/converter/amd/cdna4.md`

The sheet gives the Triton→Gluon API map, the critical pitfalls, and **pointers to the exact local
Triton source** (`reference-projects/triton`). Open that source only for the construct you are
converting — do not re-derive the whole API.

## Do (delegate to the `gpu-kernel-convert` subagent)
Launch the **`gpu-kernel-convert`** subagent with: workspace, version v{{N}}, the arch, the sheet path,
and `kernel.py`. It must:
0. **Learn from prior attempts** — this may be a RETRY. Read `memory/v*.json` entries with
   `optimization.action_category="triton_to_gluon_conversion"` (`python tools/memory_manager.py read --workspace .`);
   do NOT repeat a lowering a previous attempt recorded as failed or >5% slower — take a different approach.
1. Extract real layouts from the current kernel: `python tools/extract_ttgir.py <driver>.py -o v{{N}}.ttgir` (confirm the target arch). Never fabricate layouts.
2. Rewrite `kernel.py` to Gluon per the sheet — **same algorithm/tiling**, no new optimizations. If entry/deps change, keep `solution.json` in sync (Gluon still `languages:["triton"]`).
3. Validate: `python -c "import kernel"` then `python test_kernel.py --version v{{N}}` — the real evaluator over EVERY workload with its own tolerance.

## Commit gate — correctness AND performance parity (±5%)
A direct translation must be **as fast as the Triton kernel it came from** — same algorithm, same
work. Commit only if BOTH hold:
1. **All workloads PASS** (correctness).
2. **Geomean kernel latency is within +5% of the Triton HEAD** you converted from (compare v{{N}}'s
   `performance.latency_us` against v{{PREV}}'s in memory). Faster is fine; **>5% slower means the
   conversion is defective** (missed async/TMA copy, wrong layout, register round-trip instead of
   direct SMEM/TMEM) — fix it, don't accept it.
- Both hold → `git commit -m "v{{N}}: triton->gluon conversion (no opt, perf parity)"`. This Gluon kernel becomes HEAD and unlocks the deeper Gluon phase.
- Correctness FAIL or >5% slower after fix attempts → `git reset --hard HEAD` (restore the Triton kernel), record why in `memory/v{{N}}.json` (`pitfalls_and_fixes`), and exit.

The orchestrator **mechanically re-checks this parity** and will revert a convert that committed a
>5%-slower kernel — so do not commit a slow translation hoping it slips through.

**Always write `memory/v{{N}}.json`** — success OR failure — with
`optimization.action_category="triton_to_gluon_conversion"`, and on failure the exact cause (compile
error / correctness mismatch / which construct made it >5% slower) in `pitfalls_and_fixes`. **Commit the
record even on revert** (`git add memory/v{{N}}.json plans/ && git commit -m "v{{N}}: conversion reverted (<reason>)"`)
so the next conversion attempt — after Triton plateaus again — learns from it. Then print one line and stop:
```
v{{N}}: converted to gluon (PASS, <±X%> vs triton)   |   v{{N}}: conversion reverted (<reason>)
```
