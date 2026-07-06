# Triton → Gluon conversion (clean session, convert-only)

You are one **clean session** whose ONLY job is to convert the current Triton `kernel.py` to **Gluon**.
This is a framework lowering, **not** an optimization: preserve the algorithm, tiling, and block sizes.
Gluon is a lower-level DSL, so the *following* sessions will optimize deeper — your job is to hand them
a correct Gluon kernel. Do **one** conversion, then exit.

**Do the conversion yourself, in THIS session — do not delegate it to a subagent.** The steps below are
your work directly: read the sheet, extract TTGIR, rewrite `kernel.py`, validate, and commit or revert.
(You may still spawn a short-lived helper for a narrow diagnostic — e.g. a profiler to explain a >5%
regression — but the conversion itself is yours, and you must finish it before you exit. A session that
launches work and exits early produces no v{{N}} and wastes the whole attempt.)

## Context
- Workspace: `{{WORKSPACE}}` — your cwd, a git repo. **git HEAD is the best Triton kernel so far.**
- You are producing version **v{{N}}** (previous: v{{PREV}}).
- `tools/`, `reference/`, `skills/` are symlinked in; gpu-wiki is at `{{GPU_WIKI}}`.

{{HARDWARE}}

## Step 0 — Learn from prior attempts (this may be a RETRY)
Read `memory/v*.json` entries with `optimization.action_category="triton_to_gluon_conversion"`
(`python tools/memory_manager.py read --workspace .`). The orchestrator re-issues conversion each time
Triton re-plateaus, so earlier attempts may have failed or come out >5% slower — their
`pitfalls_and_fixes` tell you which lowering to avoid. Take a **different** approach this time; do not
repeat a recorded dead-end.

## Step 1 — Read the ONE conversion sheet for the real arch above
Read only the sheet for the **real arch** — do not read the others:
- `sm_100` / `sm_103` (Blackwell data-center, B200/B300) → `{{GPU_WIKI}}/docs/converter/nvidia/blackwell.md`
- `sm_90` (Hopper) → `{{GPU_WIKI}}/docs/converter/nvidia/hopper.md`
- `gfx94*` (CDNA3) → `{{GPU_WIKI}}/docs/converter/amd/cdna3.md`; `gfx95*` (CDNA4) → `{{GPU_WIKI}}/docs/converter/amd/cdna4.md`

The sheet gives the Triton→Gluon API map, the critical pitfalls, and **pointers to the exact local
Triton source** (`reference-projects/triton`). Open that source only for the construct you are
converting — do not re-derive the whole API or read other arches' sheets.

## Step 2 — Extract TTGIR FIRST, before writing any Gluon
`python tools/extract_ttgir.py <driver>.py -o v{{N}}.ttgir` (the driver must launch the kernel; the
kernel's `__main__` profiling block works). Confirm the target matches the real arch (e.g. `cuda:100`).
The Gluon kernel's layouts **must be the real `#blocked`/`#shared`/`#tmem` layouts from THIS kernel's
TTGIR** — never fabricate them, and never lift the reference example's layouts/shapes (use the example
for code *structure*, not its concrete layouts). Do not draft Gluon before this dump exists.

## Step 3 — Rewrite `kernel.py` → Gluon (same algorithm, no optimization)
Map loads → the arch's async/TMA + barrier pattern, matmuls → the arch's MMA + accumulator-residency
pattern, and reproduce the original `num_stages` (nothing more). Keep the DPS `run(...)` signature
(inputs then outputs). Change **only** `kernel.py`; if languages/deps/entry change, keep `solution.json`
in sync (Gluon stays `languages:["triton"]`). Do **not** add tiling changes, split-K, fusion, or new
pipelining — those are for later sessions.

## Step 4 — Validate (the SOL harness is the only gate)
`python -c "import kernel"` (compiles) → `timeout 1800 python test_kernel.py --version v{{N}}` (real
evaluator, every workload, per-workload tolerance). Iterate on compile/correctness fixes only.

## Commit gate — correctness AND performance parity (±5%)
A direct translation preserves the algorithm and the work, so the Gluon kernel must be **as fast as the
Triton kernel it came from**. Commit only if BOTH hold:
1. **All workloads PASS** (correctness), and
2. **Geomean kernel latency within +5% of the Triton HEAD** you converted from — compare v{{N}}'s
   `performance.latency_us` against v{{PREV}}'s in memory. Faster is fine; **>5% slower means the
   conversion is defective** (missed async/TMA copy, wrong/fabricated layout forcing extra conversions,
   accumulator not resident in TMEM/registers, or a missing pipeline the Triton had) — **fix the specific
   construct and re-measure; do not accept it.**

- Both hold → `git commit -m "v{{N}}: triton->gluon conversion (no opt, perf parity)"`. This Gluon kernel
  becomes HEAD and unlocks the deeper Gluon phase.
- Correctness FAIL or >5% slower after fix attempts → `git reset --hard HEAD` (restore the Triton kernel),
  record why in `memory/v{{N}}.json` (`pitfalls_and_fixes`), and exit.

The orchestrator **mechanically re-checks this parity** and will revert a convert that committed a
>5%-slower kernel — so do not commit a slow translation hoping it slips through.

## Constraints
- Convert only — no optimization, no algorithm change, no shape bucketing.
- Never fabricate layouts; extract from TTGIR or derive via the arch's documented helpers.
- Never use another vendor's Gluon APIs (e.g. `gl.amd.*` on NVIDIA → unregistered-dialect crash).
- Do not edit `test_kernel.py`, `definition.json`, `reference.py`, or `workload.jsonl`.

## Record + finish (ALWAYS — success OR failure)
**Always write `memory/v{{N}}.json`** with `optimization.action_category="triton_to_gluon_conversion"`,
and on failure the exact cause (compile error / correctness mismatch / which construct made it >5% slower)
in `pitfalls_and_fixes`. **Commit the record even on revert**
(`git add memory/v{{N}}.json plans/ && git commit -m "v{{N}}: conversion reverted (<reason>)"`) so the
next conversion attempt — after Triton plateaus again — learns from it. Then print one line and stop:
```
v{{N}}: converted to gluon (PASS, <±X%> vs triton)   |   v{{N}}: conversion reverted (<reason>)
```
