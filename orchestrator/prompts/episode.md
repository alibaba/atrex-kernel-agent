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
- Incumbent commit: `{{BASE_COMMIT}}`
- Episode branch: `{{EPISODE_BRANCH}}`
- Journal: `{{JOURNAL_PATH}}`
- Handoff: `{{HANDOFF_PATH}}`
- Additional constraints: {{NOTES}}
- `tools/`, `reference/`, `skills/`, `reference-projects/`, and `gpu-wiki/` are linked into the worktree.
{{AGENT_RUNTIME}}

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

1. Read only the conversion sheet matching the authoritative runtime architecture:
   - `sm_100`/`sm_103`: `gpu-wiki/docs/nvidia/blackwell/converter/blackwell.md`
   - `sm_90`: `gpu-wiki/docs/nvidia/hopper/converter/hopper.md`
   - `gfx94*`: `gpu-wiki/docs/amd/cdna3/converter/cdna3.md`
   - `gfx95*`: `gpu-wiki/docs/amd/cdna4/converter/cdna4.md`
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

1. `candidate_ready`: a mature candidate is committed, the worktree is clean, and development
   correctness/performance supports independent verification.
2. `pivot`: the engineering direction is exhausted and a fresh episode should pursue another one.
3. `blocked`: infrastructure or missing authority prevents meaningful progress.

For `candidate_ready`, append the final evidence, commit only `kernel.py`, then finalize the journal.
The candidate commit must be the episode `HEAD`, and its complete diff from the incumbent must name
exactly `kernel.py`:

```bash
git add -- kernel.py
git commit -m "v{{VERSION}}: kernel candidate"
candidate_commit=$(git rev-parse HEAD)
{{JOURNAL_COMMAND}} finalize --path {{JOURNAL_PATH_SHELL}} --state candidate_ready \
  --candidate-commit "$candidate_commit" \
  --outcome-json '{"summary":"...","next_directions":["..."]}'
```

For `pivot` or `blocked`, finalize with that state and omit `--candidate-commit`. The journal must
contain at least one structured experiment and a non-empty outcome summary.

Only after finalizing, atomically publish the control handoff by writing complete JSON to
`{{HANDOFF_PATH}}.tmp` and renaming it to `{{HANDOFF_PATH}}`:

```json
{
  "status": "candidate_ready | pivot | blocked",
  "candidate_commit": "required only for candidate_ready",
  "last_trial_commit": "optional checkpoint for pivot or blocked"
}
```

Chat text is not a handoff. A missing or invalid handoff causes bounded same-session recovery. Do not
claim a speedup merely to terminate; a well-supported pivot is a valid outcome.
