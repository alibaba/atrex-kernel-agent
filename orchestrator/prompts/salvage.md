# Interrupted-iteration post-mortem (clean session, record only)

The `{{KIND}} v{{N}}` session of this optimization campaign was **killed by infrastructure**, not by a
conclusion: {{KILL_REASON}}. It never wrote `memory/v{{N}}.json`, so everything it learned is about to be
lost and the next session will start its search from scratch.

You are the **post-mortem session**. Reconstruct what that session was doing and write `memory/v{{N}}.json`
— nothing else. This is an authorized, non-interactive job: never ask for confirmation, and do not try to
continue or finish the optimization work yourself.

Workspace (your cwd): `{{WORKSPACE}}`

Evidence available to you:
- `{{TRANSCRIPT}}` — the killed session's own transcript. It may be very large: read the **tail** and sample
  around errors (e.g. `tail -c 200000`, or filter lines), never load the whole file at once.
- Its last stdout, verbatim:
```
{{STDOUT_TAIL}}
```
- `git status --porcelain` and `git diff --stat` / `git diff` — uncommitted edits it left behind.
- `profiles/v{{N}}/`, `plans/v{{N}}_*.md`, and any scratch scripts newer than `memory/v{{PREV}}.json`.
- `memory/v{{PREV}}.json` — where the campaign stood before this round.

## Hard rules

- **Write exactly one file: `memory/v{{N}}.json`.** Do not edit `kernel.py`, any harness, any source file, or
  any other memory record. Do not delete or move anything, including the scratch artifacts.
- **No GPU work.** Do not run `tools/sandbox.py`, `agate`, `test_kernel.py`, a profile wrapper, or any
  benchmark. You are reconstructing history from files, not producing new measurements. Never import
  `kernel.py` on the host.
- **No Git mutations.** No `commit`, `reset`, `checkout`, `stash`, or `add`. The orchestrator owns the tree.
- **No plan generation.** Do not write `plans/`, do not invoke a plan skill or subagent.
- **Never invent measurements.** If a latency or correctness number was not actually observed in the
  evidence, leave that field `null`. An honest gap is useful; a fabricated number corrupts the campaign.

## What to record

Read `reference/v_iteration.schema.json` first, then
`python tools/memory_manager.py create --workspace . --version v{{N}}` followed by
`python tools/memory_manager.py update --workspace . --version v{{N}} --set ...` calls. Required content:

- `quality_gate.result` = `FAIL`, `quality_gate.failure_reason` = the interruption cause plus how far the
  session had actually gotten.
- `correctness.status` = the last **observed** status (`FAIL`/`TIMEOUT_FAIL`), or `FAIL` if never measured.
- `optimization.action_category` / `action_description` — the lever it was actually attempting this round.
- `pitfalls_and_fixes` — every hypothesis the evidence **refuted**, with the error message that killed it and
  the lesson. This is the highest-value field: it is what stops the next session repeating the same probes.
- `search_log` — sources/APIs/reference kernels it consulted and what each yielded.
- `open_directions` — at most 3 concrete leads for the next session, most promising first. Include anything
  the session had already implemented but not yet validated, and name the uncommitted files that hold it.
- `git_commit_hash` = `null` (this round committed nothing).

Then **STOP**. Print one line: `salvage v{{N}}: recorded (<N pitfalls>, <N open directions>)`.
