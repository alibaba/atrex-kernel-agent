# Fast kernel optimization episode {{EPISODE}}

Run one deliberately lightweight optimization episode containing exactly {{FAST_TRIALS}}
consecutive trials in this isolated Git worktree. Every trial repeats **plan -> implement ->
evaluator**. Optimize for short turnaround; do not expand any trial into the normal
profile/research/ABBA loop. External review remains part of every plan. Do not terminate after an
early success: complete all {{FAST_TRIALS}} trials unless infrastructure failure or missing
authority makes the episode `blocked`.

The supervisor owns the incumbent branch, canonical memory, acceptance, and squash promotion. You
own only this episode branch, its final `kernel.py`, journal, and terminal handoff.

## Context

- Workspace: `{{WORKSPACE}}`
- Canonical version produced by the supervisor: `v{{VERSION}}`
- Platform: `{{PLATFORM}}`
- Framework: `{{FRAMEWORK}}`
- Incumbent commit: `{{BASE_COMMIT}}`
- Episode branch: `{{EPISODE_BRANCH}}`
- Journal: `{{JOURNAL_PATH}}`
- Handoff: `{{HANDOFF_PATH}}`
- Additional constraints: {{NOTES}}
- `tools/`, `reference/`, `skills/`, `reference-projects/`, and `gpu-wiki/` are linked into the worktree.
{{AGENT_RUNTIME}}

{{RESUME_DIRECTIVE}}

Never switch branches, push, merge, rebase, or alter refs. Every commit must contain only
`kernel.py`. Plans, journals, and handoffs are ignored episode evidence and must never be added to
Git. Never edit evaluator or ground-truth files, including `test_kernel.py`, `profile_driver.py`,
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

Keep telemetry usable for per-step timing even though each trial is short. Repeat these phase
markers for every trial, with at most one phase active. Use `planning`, `implementation`, and
`benchmark` respectively; never emit `profile` or `research` markers:

```bash
python3 tools/iteration_trace.py phase-start <planning|implementation|benchmark>
python3 tools/iteration_trace.py phase-end <planning|implementation|benchmark>
```

At episode start, record the incumbent `HEAD`, its canonical `performance_score`, latency, and
kernel as the initial `best_commit`, `best_score`, `best_latency`, and `best_kernel`. The score is the
optimization objective and higher is better; latency remains diagnostic evidence. Each trial starts
from the best passing kernel found so far, not automatically from the immediately preceding trial. A
failed or lower-scoring trial must not contaminate the next trial.

### 1. Plan — repeat for trials 1 through {{FAST_TRIALS}}

Before planning a trial, restore `kernel.py` from `best_commit` when the previous trial was not kept.
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

After the trial edit, commit only `kernel.py`, then atomically publish that trial's exact candidate
commit for the supervisor's independent policy reviewer. Publishing starts policy review in parallel
with the evaluator; do not wait for the reviewer:

```bash
git add -- kernel.py
git commit -m "v{{VERSION}} trial N: fast kernel candidate"
candidate_commit=$(git rev-parse HEAD)
printf '{"schema_version":1,"candidate_commit":"%s"}\n' "$candidate_commit" \
  > .atrex_long_horizon/policy_review_request.json.tmp
mv .atrex_long_horizon/policy_review_request.json.tmp \
  .atrex_long_horizon/policy_review_request.json
```

Immediately run one official full-workload base-seed evaluator:

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
by the trial. For `not_queried`, omit both `wiki_query_ids` and `wiki_usage`. Also record the evaluator
outcome in structured `evaluation`; copy its kernel hash and latency
when emitted. Wiki telemetry validation is fail-open: malformed rows
are omitted and reported in `wiki_usage_errors`, while the experiment still records normally:

```bash
{{JOURNAL_COMMAND}} append --path {{JOURNAL_PATH_SHELL}} \
  --experiment-json '{"name":"fast trial N: plan -> implement -> evaluator","hypothesis":"...","change":"...","evidence":"official base-seed evaluator result or blocker","result":"...","evaluation":{"correctness":"pass|fail|unknown","performance":"improved|not_improved|unknown","latency_us":null,"kernel_hash":"<evaluator-kernel-hash-or-empty>"},"decision":"keep_as_best | reject_and_continue | blocked","wiki_usage_status":"declared","wiki_query_ids":["<emitted-query-id>"],"wiki_usage":[{"query_id":"<emitted-query-id>","wiki_id":"<emitted-canonical-wiki-id>","disposition":"reference_only","use":"decision or code change influenced by the record","evidence":"observable evidence for this disposition"}]}'
```

If the result passes and its `performance_score` exceeds `best_score`, update `best_commit`,
`best_score`, `best_latency`, and `best_kernel`. Otherwise keep the prior best and restore it before
planning the next trial. Continue
until {{FAST_TRIALS}} evaluator results and {{FAST_TRIALS}} journal experiments exist. Only
infrastructure failure or missing authority may end early as `blocked`; a bad candidate is evidence
for the next trial, not an early terminal `pivot`.

After trial {{FAST_TRIALS}}, select the highest-scoring passing strict improvement over the canonical
incumbent. If that best kernel is not the current `HEAD`, restore its exact previously evaluated bytes,
commit only `kernel.py` as `v{{VERSION}}: select best fast candidate`, and atomically publish that
selection commit to the same policy-review request path. Do not run an additional evaluator: the supervisor
matches the selected bytes to their recorded evaluator hash. If no trial produced a passing strict
improvement, finish as `pivot`.

Wrap every journal append and final journal/handoff publication in the `recording` telemetry phase.

## Framework escalation state

{{CONVERSION_DIRECTIVE}}

When conversion is mandatory, trial 1 must produce a committed Gluon kernel and all later trials must
remain Gluon. Preserve the incumbent algorithm, tiling, signatures, and evaluator behavior during
the conversion trial; later trials may make bounded reviewed optimizations. A passing candidate may
be handed off when its evaluator latency is plausibly within 5% of the incumbent; the supervisor
enforces conversion parity without adding ABBA to this fast episode.

## Terminal contract

Reach exactly one state:

1. `candidate_ready`: all {{FAST_TRIALS}} trials are recorded, the selected best evaluator
   result passes and matches the final kernel bytes, the selected candidate is committed, and the
   worktree `kernel.py` matches that commit. Protected files must be unchanged; other uncommitted
   intermediate artifacts may remain in the worktree.
2. `pivot`: all {{FAST_TRIALS}} trials are recorded and none produced a passing strict
   improvement; keep the incumbent.
3. `blocked`: infrastructure or missing authority prevents the required flow.

For `candidate_ready`, use the exact selected candidate commit and finalize the journal after it:

```bash
candidate_commit=$(git rev-parse HEAD)
{{JOURNAL_COMMAND}} finalize --path {{JOURNAL_PATH_SHELL}} --state candidate_ready \
  --candidate-commit "$candidate_commit" \
  --outcome-json '{"summary":"...","next_directions":["..."],"selected_experiment_index":N}'
```

`selected_experiment_index` is the one-based journal experiment index whose evaluated kernel bytes
were selected for handoff. It is required for `candidate_ready`; do not point it at a rejected trial.

For `pivot`, finalize only after {{FAST_TRIALS}} trial experiments and
{{FAST_TRIALS}} evaluator results. For `blocked`, finalize immediately with the completed trial
evidence available, appending a blocker experiment first if no trial experiment exists yet. Omit
`--candidate-commit` for both. Every terminal state requires at least one experiment and a non-empty
outcome summary.

Only after finalizing, atomically publish the control handoff by writing complete JSON to
`{{HANDOFF_PATH}}.tmp` and renaming it to `{{HANDOFF_PATH}}`:

```json
{
  "status": "candidate_ready | pivot | blocked",
  "candidate_commit": "required only for candidate_ready",
  "last_trial_commit": "optional checkpoint for pivot or blocked"
}
```

Chat text is not a handoff. The supervisor rejects non-blocked fast handoffs with fewer than
{{FAST_TRIALS}} journal experiments or evaluator results. Do not claim a speedup merely to
terminate; an evidence-backed {{FAST_TRIALS}}-trial `pivot` is a valid fast outcome.
