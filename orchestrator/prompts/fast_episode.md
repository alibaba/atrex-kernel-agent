# Fast kernel optimization episode {{EPISODE}}

Run one deliberately lightweight optimization episode containing exactly {{FAST_TRIALS}}
consecutive trials in this Git-free Episode workspace. Every trial repeats **plan -> implement ->
evaluator**. Optimize for short turnaround; do not expand any trial into the normal
profile/research/ABBA loop. External review remains part of every plan. Do not terminate after an
early success: complete all {{FAST_TRIALS}} trials unless infrastructure failure or missing
authority makes the episode `blocked`.

The supervisor owns the incumbent branch, canonical memory, acceptance, and squash promotion. You
edit `kernel.py` and submit structured evidence through Runtime tools.

## Context

- Workspace: `{{WORKSPACE}}`
- Canonical version produced by the supervisor: `v{{VERSION}}`
- Platform: `{{PLATFORM}}`
- Framework: `{{FRAMEWORK}}`
- Additional constraints: {{NOTES}}
- `tools/`, `reference/`, `skills/`, `reference-projects/`, and `gpu-wiki/` are linked into the worktree.
{{AGENT_RUNTIME}}

{{RESUME_DIRECTIVE}}

## Journal interface

Use `skills/runtime-records/SKILL.md`: register Directions, record Experiments with returned
Gateway Record IDs, and submit `episode-report`. Keep the prescribed planning/research,
Wiki attribution and Phase Marker workflow. Git, private Journals, handoffs and acceptance are
Supervisor-owned. Do not run Git commands or search for Git metadata.

Never edit evaluator or ground-truth files, including `test_kernel.py`, `profile_driver.py`,
`definition.json`, `reference.py`, `workload.jsonl`, `input.py`, `shapes.json`,
`agent_problem.json`, `metadata.json`, `roofline.json`, `CLAUDE.md`, or `README.md`. Do not write
canonical `memory/vN.json`; the supervisor writes and commits one for every terminal episode.

{{MODE_POLICY}}

{{EVALUATOR}}

{{HARDWARE}}

{{SANDBOX}}

## Fast-mode boundaries

- Do not profile. Never use `--kind profile`, `ncu`, `rocprofv3`, a profile wrapper, or create
  profile artifacts.
- Do not start a separate research phase or the full `gpu-kernel-episode-loop` skill. Every trial's
  planning phase must use its configured `gen-plan` flow and whichever Codex/Qoder reviews are
  enabled for fast episodes; keep their evidence bounded to `kernel.py`, the public operator
  contract, prior trial journal evidence, and canonical `memory/v*.json`.
- Do not run multi-seed validation and do not run or simulate incumbent/candidate ABBA.
- Never run GPU/JIT code on the host. Static source inspection is allowed; the official evaluator
  command below is the only GPU execution route and must run once per trial. Do not route
  import/API probes or additional benchmarks through `tools/sandbox.py`.
- Never install or build dependencies. Never start, stop, restart, signal, replace, or mutate the
  shared sandbox executor or its jobs. Treat infrastructure failure as `blocked`.

## Required flow

At episode start, run the required bounded GPU Wiki query once using the campaign's exact operator
identifier rather than paraphrasing it. Additional targeted queries are allowed later when new
evidence creates a materially different question:

```bash
python3 gpu-wiki/tools/query_nl.py "Target hardware {{PLATFORM}}, DSL {{FRAMEWORK}}. Optimize operator {{OPERATOR}} and retrieve techniques and pitfalls." --brief
```

Read only directly applicable returned records. Preserve the response's top-level `query_id` and
the `wiki_id` from each record actually considered. Trial 1 must record this query as either
`declared` (with the materially used/rejected record rows) or `no_material_use`; it must not claim
`not_queried`. The query front door binds the product to its recorded architecture and returns the
product spec alongside operator knowledge; do not paraphrase the exact command into a bridge-agent
request. Later trials reuse that query id only when they actually reconsider the response.

Keep telemetry usable for per-step timing even though each trial is short. Repeat these phase
markers for every trial, with at most one phase active. Use `planning`, `implementation`, and
`benchmark` respectively; never emit `profile` or `research` markers:

```bash
python3 tools/iteration_trace.py phase-start <planning|implementation|benchmark>
python3 tools/iteration_trace.py phase-end <planning|implementation|benchmark>
```

At episode start, save the incumbent bytes in `scratch/incumbent.py` and record its canonical `performance_score`, latency, and
kernel as the initial `best_score`, `best_latency`, and `best_kernel`. The score is the
optimization objective and higher is better; latency remains diagnostic evidence. Each trial starts
from the best passing kernel found so far, not automatically from the immediately preceding trial. A
failed or lower-scoring trial must not contaminate the next trial.

### 1. Plan — repeat for trials 1 through {{FAST_TRIALS}}

Before planning a trial, restore `kernel.py` from the saved `best_kernel` bytes when the previous trial was not kept.
Read that kernel, recent canonical memory, and the structured results of earlier trials in this
episode. Pick one small, coherent implementation change that is not a verbatim repeat of a failed
trial. Write the trial's unique draft with its hypothesis, exact code change, expected effect, and
rollback condition. Then run the matching backend-native generator below. Every invocation obtains
the enabled external Codex/Qoder reviews, records disabled reviewer statuses, and writes a unique
synthesized plan. Use these exact per-trial paths:

{{FAST_TRIAL_PLAN_PATHS}}

The backend-native generator pattern is below. For trial `N`, replace its displayed draft and plan
paths with the corresponding unique paths above, while retaining direct/no-discussion mode:

{{PLAN_GENERATOR}}

For trial `N`, use only the `Trial N` generator and the resulting
`plans/v{{VERSION}}_trialN_plan.md`. Read it before editing and implement only its final bounded
direction. A reviewer disabled by episode configuration or by the campaign's availability probe
must be recorded as disabled; do not replace it with an ad-hoc research phase. Do not collect
profile data.

### 2. Implement — once per trial

Edit only `kernel.py`. Keep the change focused. You may statically inspect source and repair obvious
syntax or logic defects before evaluation, but do not launch exploratory GPU commands. Each trial is
one attributable candidate; do not combine unrelated optimizations merely to fill the
{{FAST_TRIALS}}-trial budget.

### 3. Evaluator — exactly once per trial

Run one official full-workload base-seed evaluator. The Supervisor snapshots the exact submitted
source and starts independent policy review in parallel; do not submit Git commits or reviewer files:

```bash
{{FAST_EVALUATOR_COMMAND}}
```

Do not pass `--multi-seed`, do not run ABBA, and do not rerun the evaluator inside the same trial. A
compile/correctness failure consumes that trial; a repair must be the next reviewed
plan -> implement -> evaluator trial. The sandbox records every result with the exact `kernel.py`
hash.

Immediately after each evaluator, append one structured journal experiment for that trial. Record
the reviewed plan, implementation, correctness, `performance_score` and latency when available,
comparison with `best_score`, and the decision to keep or reject the trial. If this trial queried GPU Wiki,
copy the response's top-level `query_id` and each used record's own emitted canonical `wiki_id`
exactly; never reconstruct them from response mapping keys or prose. Add one `wiki_usage` row for
each returned Wiki record that was materially used or explicitly evaluated;
classify it as `applied`, `partially_applied`, `reference_only`, or `rejected`. Every experiment must
set `wiki_usage_status`: use `declared` with a non-empty `wiki_usage`, `no_material_use` when Wiki was
queried but no returned knowledge materially influenced the trial, or `not_queried` when no Wiki query
occurred. For `declared` and `no_material_use`, also set `wiki_query_ids` to every Wiki query considered
by the trial. For `not_queried`, omit both `wiki_query_ids` and `wiki_usage`. Record the evaluator's returned `gateway_record_id`, not hand-transcribed counters.
Use `record-experiment` as documented in the Runtime Records Skill, including the reviewed plan
path and your keep/reject analysis. One research Direction can contain multiple trial Experiments.

If the result passes and its `performance_score` exceeds `best_score`, update `best_score`, `best_latency`, and `best_kernel`. Otherwise keep the prior best and restore it before
planning the next trial. Continue
until {{FAST_TRIALS}} evaluator results and {{FAST_TRIALS}} journal experiments exist. Only
infrastructure failure or missing authority may end early as `blocked`; a bad candidate is evidence
for the next trial, not an early terminal `pivot`.

After trial {{FAST_TRIALS}}, select the highest-scoring passing strict improvement over the canonical
incumbent. Restore its exact previously evaluated bytes from your scratch copy or `kernel-read`.
Do not run an additional evaluator: the Supervisor matches the selected bytes to their record. If no trial produced a passing strict
improvement, finish as `pivot`.

Wrap every journal append and final journal/handoff publication in the `recording` telemetry phase.

## Framework escalation state

{{CONVERSION_DIRECTIVE}}

When conversion is mandatory, trial 1 must produce a Gluon kernel and all later trials must
remain Gluon. Preserve the incumbent algorithm, tiling, signatures, and evaluator behavior during
the conversion trial; later trials may make bounded reviewed optimizations. A passing candidate may
be handed off when its evaluator latency is plausibly within 5% of the incumbent; the supervisor
enforces conversion parity without adding ABBA to this fast episode.

## Terminal contract

Close all active Directions, then submit a request file through:

```bash
python3 tools/sandbox.py --kind episode-report --request-file scratch/episode-report.json
```

For a candidate, leave its exact measured bytes in `kernel.py`:

```json
{"status":"candidate_ready","summary":"What changed and what evidence supports it","selected_experiment_id":"experiment_<returned-id>"}
```

The selected Experiment must be from this Episode and cite a passing, standard full-workload
Evaluate of these bytes. The Supervisor creates the candidate commit, verifies it and decides
promotion. Do not send `candidate_commit`, run Git, or write a handoff/Journal projection yourself.
An accepted report is not a promotion decision.

For an exhausted direction, use `{"status":"pivot","summary":"Evidence-backed conclusion"}`.
For infrastructure/authority blockers, use
`{"status":"blocked","summary":"What prevented progress","blocker":"Concrete missing capability"}`.
Fast candidate/pivot reports require the configured trial minimums; blocked is exempt.
Full pivot/blocked may have no Experiments only when no Direction needs closing.

Use `skills/runtime-records/references/journal.md` for complete fields and repair instructions.
For PPU, optional `accepted_ppu_diagnostics` must still satisfy the mounted profiler Skill.
A rejected report leaves the Episode open: correct the fields or prerequisites identified in the
error and resubmit. An identical accepted report can be replayed to repair publication.
Stop only after successful handoff; chat text alone is not a handoff.
