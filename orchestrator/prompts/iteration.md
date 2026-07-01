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
- `tools/`, `reference/`, and `skills/` are symlinked into the workspace — read/use them by relative path
  (`tools/profile_nvidia.sh`, `python tools/memory_manager.py --workspace .`,
  `reference/v_iteration.schema.json`, `skills/gpu-kernel-profile-optimizer/SKILL.md`).
  The gpu-wiki path is recorded as `gpu_wiki_path` in `README.md`.

{{HARDWARE}}

Read and follow **`skills/gpu-kernel-profile-optimizer/SKILL.md`** for the *mechanics* — profiling commands,
evidence format (`evidence -> inference -> action`), the localization rule, plan format, the Stage 4 quality
gate, and the Stage 5 commit format. This prompt **overrides its loop/stop behavior**: do Stages 1–5 once,
skip Stage 6, then exit. Honor that skill's subagent requirements for Stage 2 (planning) and Stage 4 (validation).

## Step A — Learn from prior sessions (read; do not redo their work)

1. Read `README.md` — config, `Hardware Spec`, and `Stop Conditions`.
2. Cross-version digest — `python tools/memory_manager.py summary --workspace .` (where we are; the trajectory).
3. The latest entry, in full — `python tools/memory_manager.py read --workspace . --version v{{PREV}}`. Pay attention to:
   - **`open_directions`** — candidate leads the previous session left for you.
   - **`search_log` + `pitfalls_and_fixes`** across the digest — **recorded dead-ends. Do NOT repeat them.**
4. Profile reuse — if `profiles/v{{PREV}}/` holds a profile of the *current* HEAD kernel (carried forward),
   you may reuse it instead of re-profiling. Otherwise profile fresh in Step B.

**`open_directions` are priors, not orders.** Pick the most promising lead — **or**, if a fresh look at the
profile reveals a better lever, pursue that instead. The only hard constraint is: don't re-run a recorded dead-end.

## Step B — One cycle

1. **Profile** the current `kernel.py` into `profiles/v{{N}}/` (skill Stage 1). Extract ≥1 concrete bottleneck.
2. **Pick exactly ONE optimization lever** (skill Stage 2 → `plans/v{{N}}_plan.md`). One category only, so the result is attributable.
3. **Edit** `kernel.py` — apply that single change, nothing else (skill Stage 3).
4. **Validate + bench** (skill Stage 4): correctness first (with the timeout guard), then measure
   latency / TFLOPS / bandwidth / peak-utilization. Bench must be **variance-aware** — a delta only counts as
   real if it clears measurement noise (best-of-N or delta > noise band; flat-within-noise is *not* an improvement).

## Step C — Commit or revert (mechanical, no discretion)

- **Real win** (correctness PASS **and** speedup clears noise vs v{{PREV}}) → **commit** (skill Stage 5 format).
  This kernel becomes the new HEAD/best.
- **Otherwise** (regression, flat-within-noise, or correctness FAIL) → `git reset --hard HEAD` to restore the
  best-known `kernel.py`. **Never commit a regression.**

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

Print one line and stop — do **not** start another cycle:

```
v{{N}}: committed (+X.X%)   |   v{{N}}: reverted (<reason>)
```
