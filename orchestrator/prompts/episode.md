# Kernel optimization episode {{EPISODE}}

Own one complete engineering direction in this isolated Git worktree. Continue through as many
profile, research, plan, edit, compile, correctness, benchmark, autotune, and repair cycles as the
direction needs. Do not stop after one edit, one failed compile, or one benchmark while a concrete
next engineering step remains.

The supervisor owns the incumbent branch, authoritative ABBA verification, canonical memory, and
final squash promotion. You own only this episode branch and its structured evidence.

## Context

- Workspace: `{{WORKSPACE}}`
- Canonical version produced by the supervisor: `v{{VERSION}}`
- Platform: `{{PLATFORM}}`
- Framework: `{{FRAMEWORK}}`
- Production incumbent commit: `{{BASE_COMMIT}}`
- Episode starting commit: `{{DEVELOPMENT_BASE_COMMIT}}`
- Episode branch: `{{EPISODE_BRANCH}}`
- Journal: `{{JOURNAL_PATH}}`
- Handoff: `{{HANDOFF_PATH}}`
- Additional constraints: {{NOTES}}
- `tools/`, `reference/`, `skills/`, `reference-projects/`, and `gpu-wiki/` are linked into the worktree.
{{AGENT_RUNTIME}}

{{RESUME_DIRECTIVE}}

Never switch branches, push, merge, rebase, or alter refs. Private checkpoint commits on the episode
branch are allowed, but every commit must contain only `kernel.py`. Plans, profiles, discussion
transcripts, journals, and handoffs are ignored episode evidence: write them normally but never add
them to Git. Never edit evaluator or ground-truth files, including `test_kernel.py`,
`profile_driver.py`, `definition.json`, `reference.py`, `workload.jsonl`, `input.py`, `shapes.json`,
`agent_problem.json`, `metadata.json`, `roofline.json`, `CLAUDE.md`, or `README.md`. For generalized
Atrex-Bench tasks, do not search outside the workspace for the source operator directory or hidden
evaluator files. Do not write canonical `memory/vN.json`;
the supervisor creates it after terminal validation.

{{MODE_POLICY}}

{{EVALUATOR}}

{{HARDWARE}}

{{SANDBOX}}

## Non-negotiable execution boundary

- Never run `python test_kernel.py`, `python kernel.py`, or import GPU/JIT kernel packages directly
  on the host. Route every compile, correctness, benchmark, and profiling command through
  `python tools/sandbox.py ... --`.
- Never start, stop, restart, signal, replace, or mutate the shared gateway service, its screen
  session, state directory, database, log, or jobs. Report infrastructure failure instead.
- Never install or build dependencies with pip, uv, conda, setup.py, ninja, cmake, or package-manager
  commands. Use only the immutable campaign environment.
- Static source inspection is allowed. Imports or probes that may initialize CUDA/ROCm/JIT code must
  run through the sandbox.

## Framework escalation state

{{CONVERSION_DIRECTIVE}}

When conversion is mandatory, treat the whole episode as a Triton-to-Gluon lowering direction:

1. Query only for the conversion record matching the authoritative runtime architecture. In the
   natural-language request, name the true product `{{PLATFORM}}` and copy the authoritative runtime
   architecture exactly from the injected Hardware ground-truth block; then
   request both the full product specification and the matching Triton-to-Gluon conversion guidance:
   ```bash
   python3 gpu-wiki/tools/query_nl.py "The true target product is {{PLATFORM}} and the authoritative
     runtime architecture is <exact value from Hardware ground truth>. Return the full product specification and only the matching
     Triton-to-Gluon conversion guidance." --brief
   ```
   The expected conversion record is `nvidia.blackwell.any.converter.blackwell` for `sm_100`/`sm_103`,
   `nvidia.hopper.any.converter.hopper` for `sm_90`, `amd.cdna3.any.converter.cdna3` for `gfx94*`, or
   `amd.cdna4.any.converter.cdna4` for `gfx95*`. Do not use a sibling architecture's conversion record.
2. Extract TTGIR before writing Gluon and derive layouts from the real kernel; never fabricate them.
3. Preserve algorithm, tiling, signatures, and evaluator behavior. Fix compile/correctness/parity
   defects inside this episode rather than handing off the first translation attempt.
4. A terminal candidate must be committed Gluon, correctness-passing in development, and plausibly
   within 5% of the incumbent. The supervisor independently enforces parity.

## Prior iteration state

No recent-episode summary is injected into this prompt. Reconstruct prior outcomes exclusively from
the canonical `memory/v*.json` records in the workspace. Treat those records as evidence, not orders,
and do not repeat a rejected direction unless new evidence or a materially different implementation
changes the expected result. Detailed within-episode journals remain archived under
`.atrex_long_horizon/episodes/` and are not part of the inherited prompt context.

## Multi-episode architectural initiative

{{STAGED_REWRITE_DIRECTIVE}}

A staged checkpoint is an engineering continuation point, never a production candidate. It may be
neutral or slower than the incumbent because it establishes a prerequisite such as a new data
layout, pipeline, loader, or synchronization model. It must still be a coherent `kernel.py` state
that compiles through the official sandbox and proves one stage-specific architectural advancement.
An initiative must state the incumbent limitation it is escaping, the material architectural delta,
the measurable final success criterion, and the evidence that would abort the initiative. These four
parts remain stable across continuation stages; if the hypothesis is falsified, pivot instead of
preserving a checkpoint. A compile-only refactor, parameter sweep, or renamed incumbent path is not
an architectural advancement. Do not weaken final correctness, policy, or performance requirements
to preserve a stage.
The supervisor keeps the production incumbent unchanged and restores the staged kernel only into the
next isolated episode worktree.

## Wiki attribution contract

GPU Wiki query responses emit a top-level `query_id`, and every returned record emits its own
canonical `wiki_id` in `store::record` form. Copy those fields exactly; never reconstruct either
value from a response mapping key or from prose. Whenever a returned record
materially influences an experiment or is explicitly evaluated and rejected, add `wiki_usage` to
that experiment's journal append. Each row must contain the response's emitted `query_id`, an
actually returned record's emitted `wiki_id`, a disposition of `applied`, `partially_applied`,
`reference_only`, or `rejected`, plus a
short `use` and observable `evidence`. Preserve repeated use in separate experiments; do not dedupe
across the episode. Every experiment must set `wiki_usage_status` to `declared` with non-empty usage,
`no_material_use` when Wiki was queried without attributable use, or `not_queried` when it was not
queried. For `declared` and `no_material_use`, include `wiki_query_ids` with every Wiki query considered
by the experiment; omit it for `not_queried`. Record `evaluation.correctness`, `evaluation.performance`, optional evaluator latency/hash,
and an explicit decision so attribution can be joined to the experiment outcome.
Malformed Wiki telemetry is diagnostic only: the journal drops bad rows into `wiki_usage_errors`
without invalidating the optimization experiment or its terminal handoff.

## Engineering loop

`skills/gpu-kernel-episode-loop/SKILL.md` defines the binding evidence loop for this episode:
reconstruct the incumbent, profile and localize, research progressively, plan one coherent direction,
implement and repair, validate development correctness and performance, record every decisive
experiment, and mark the phase telemetry. **Read that file now and execute its loop**; it is a
requirement, not background reading.

Bind its placeholders to this episode:

| Skill placeholder | This episode |
| --- | --- |
| `<PROFILE_DIR>` | `profiles/episode_{{EPISODE}}` |
| `<PLAN_DRAFT>` | `plans/v{{VERSION}}_draft.md` |
| `<PLAN_FILE>` | `plans/v{{VERSION}}_plan.md` |
| `<JOURNAL_CLI>` | `{{JOURNAL_COMMAND}}` |
| `<JOURNAL_PATH>` | `{{JOURNAL_PATH_SHELL}}` |

`<PLAN_GENERATOR>` is the backend-native plan generator for this session:

{{PLAN_GENERATOR}}

As soon as one coherent candidate passes the full development correctness check and has credible
performance evidence, publish the terminal handoff. Do not hold a promotable candidate while pursuing
secondary tweaks; those belong to a later episode and version.

## Terminal contract

Reach exactly one evidence-backed terminal state:

1. `candidate_ready`: a mature candidate is committed, the worktree `kernel.py` matches that exact
   commit, protected files are unchanged, and development correctness/performance supports
   independent verification. Uncommitted intermediate artifacts may remain in the worktree.
2. `staged_ready`: an enabled architectural initiative completed one coherent prerequisite stage,
   the checkpoint compiles and satisfies its declared stage gate, but the initiative is not yet
   eligible for production verification. A temporary performance regression is allowed here.
3. `pivot`: the engineering direction is exhausted and a fresh episode should pursue another one.
4. `blocked`: infrastructure or missing authority prevents meaningful progress.

For `candidate_ready`, append the final evidence, commit only `kernel.py`, then finalize the journal.
The candidate commit must be the episode `HEAD`, and its complete diff from the incumbent must name
exactly `kernel.py`:

```bash
git add -- kernel.py
git commit -m "v{{VERSION}}: kernel candidate"
candidate_commit=$(git rev-parse HEAD)
{{JOURNAL_COMMAND}} finalize --path {{JOURNAL_PATH_SHELL}} --state candidate_ready \
  --candidate-commit "$candidate_commit" \
  --outcome-json '{"summary":"...","next_directions":["..."],"selected_experiment_index":N}'
```

Use the one-based journal index of the experiment selected for handoff. Its structured evaluation
must pass correctness and its decision must be `promote` or `keep_as_best`.

For an enabled `staged_ready`, append the stage evidence, commit only `kernel.py`, and finalize after
that exact commit. A new initiative starts at stage 1; continuation stages keep the same
`initiative_id` and increment `stage` by exactly one. The escape contract fields must remain exact
across continuation stages. `stage_gate.compile` and `stage_gate.advancement` must both be `pass`;
`scope` states the bounded functional or structural invariant proven by this stage, and `evidence`
identifies the official sandbox result. Merely compiling unchanged or locally retuned architecture
does not satisfy `advancement`.

```bash
git add -- kernel.py
git commit -m "v{{VERSION}}: staged architectural checkpoint"
checkpoint_commit=$(git rev-parse HEAD)
{{JOURNAL_COMMAND}} finalize --path {{JOURNAL_PATH_SHELL}} --state staged_ready \
  --checkpoint-commit "$checkpoint_commit" \
  --outcome-json '{"summary":"...","next_directions":["..."],"initiative_id":"...","stage":1,"next_stage":"...","escape_hypothesis":"incumbent limitation that local tuning cannot remove","architectural_delta":"materially different dataflow/layout/pipeline/synchronization/communication design","final_success_criterion":"measurable final correctness and performance gate","abort_criterion":"evidence that falsifies the initiative","stage_gate":{"compile":"pass","advancement":"pass","scope":"bounded invariant proven by this stage","evidence":"official sandbox result"}}'
```

For `pivot` or `blocked`, finalize with that state and omit both commit arguments. The journal must
contain at least one structured experiment and a non-empty outcome summary. A valid `pivot` abandons
the active staged initiative; a `blocked` handoff preserves its last accepted checkpoint.

Only after finalizing, atomically publish the control handoff by writing complete JSON to
`{{HANDOFF_PATH}}.tmp` and renaming it to `{{HANDOFF_PATH}}`:

```json
{
  "status": "candidate_ready | staged_ready | pivot | blocked",
  "candidate_commit": "required only for candidate_ready",
  "checkpoint_commit": "required only for staged_ready",
  "last_trial_commit": "optional checkpoint for pivot or blocked"
}
```

Chat text is not a handoff. A missing or invalid handoff causes bounded same-session recovery. Do not
claim a speedup merely to terminate; a well-supported pivot is a valid outcome.
