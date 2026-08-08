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
branch are allowed. Never edit evaluator or ground-truth files, including `test_kernel.py`,
`definition.json`, `reference.py`, `workload.jsonl`, `input.py`, `shapes.json`, `metadata.json`,
`roofline.json`, `CLAUDE.md`, or `README.md`. Do not write canonical `memory/vN.json`; the supervisor
creates it after terminal validation.

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

## Prior episode evidence

```json
{{HISTORY}}
```

Historical attempts are evidence, not orders. Do not repeat a rejected direction unless new evidence
or a materially different implementation changes the expected result.

## Structured telemetry

Telemetry is best-effort and must not block engineering work. Mark phase boundaries with standalone
commands and keep at most one phase active:

```bash
python tools/iteration_trace.py phase-start <profile|research|planning|implementation|correctness|benchmark|recording>
python tools/iteration_trace.py phase-end <profile|research|planning|implementation|correctness|benchmark|recording>
python tools/iteration_trace.py source-read <gpu_wiki|reference_projects|workspace|public_web> <safe-relative-reference>
```

Never put credentials, private URL parameters, absolute user paths, raw tool output, or transcript
text into telemetry.

## Engineering loop

Repeat the following evidence loop until the direction yields a mature candidate or is exhausted.
As soon as one coherent candidate passes the full development correctness check and has credible
performance evidence, publish the terminal handoff. Do not hold a promotable candidate while pursuing
secondary tweaks; those belong to a later episode and version.

### 1. Reconstruct the incumbent and choose a hypothesis

Read the workspace goal, unmasked `memory/v*.json`, prior plans/profiles, and the prior episode
evidence above. Identify attempted dead ends and open directions. Start with one falsifiable
hypothesis tied to the current bottleneck.

### 2. Profile and localize

Reuse a profile only when it matches the current committed kernel. Otherwise profile through the
sandbox using the vendor-appropriate tooling:

```bash
# NVIDIA
python tools/sandbox.py --kind profile --sync profiles/episode_{{EPISODE}} -- \
  bash tools/profile_nvidia.sh kernel.py --output-dir profiles/episode_{{EPISODE}} --source

# AMD
python tools/sandbox.py --kind profile --sync profiles/episode_{{EPISODE}} -- \
  bash tools/profile_kernel.sh kernel.py --output-dir profiles/episode_{{EPISODE}}
```

Extract a concrete bottleneck and source-level target. Use PTX/SASS/TTGIR inspection when compiler
lowering or instruction selection is part of the hypothesis. Do not make speculative optimization
changes before obtaining usable evidence.

### 3. Research progressively

Search in this order and stop when one actionable direction is supported:

1. `gpu-wiki/docs/` and `gpu-wiki/reference-kernels/`, filtered by real architecture, vendor, DSL,
   operator, and profiler symptom.
2. `reference-projects/` only when the local wiki is insufficient.
3. Public primary sources only when local sources do not answer the question.

After repeated rejected episodes, expand across DSLs targeting the same architecture instead of
repeating local parameter tweaks. Record source paths and the evidence-to-action chain.

### 4. Plan a coherent direction

Write or update `plans/v{{VERSION}}_draft.md` with profile evidence, research findings, concrete
edits, risks, rollback points, and measurable acceptance criteria. Then use the backend-native plan
generator:

{{PLAN_GENERATOR}}

The episode may contain multiple related experiments, but they must advance one coherent engineering
direction. Checkpoint useful intermediate states so failed sub-steps can be reverted without losing
the whole direction.

### 5. Implement and repair

Modify only candidate source/metadata files allowed by policy. Compile and probe through the sandbox.
On compile or correctness failure, diagnose and repair while the direction remains viable. Do not
publish an intermediate checkpoint as a candidate.

### 6. Development correctness and performance

Use the immutable evaluator for development measurements:

```bash
python tools/sandbox.py --kind run --no-sync -- \
  python test_kernel.py --version vlong --no-memory
python tools/sandbox.py --kind run --no-sync -- \
  python test_kernel.py --version vlong --multi-seed 5 --no-memory
```

All workloads and all additional seeds must pass. Never depend on tensor values, pointer identity,
cached outputs, evaluator ordering, or hidden workload IDs. Shape/dtype/layout dispatch is allowed.
Repeated development measurements are not promotion authority; the supervisor reruns incumbent and
candidate in one ABBA allocation.

### 7. Record every decisive experiment immediately

Immediately after each decisive experiment, append it to the single episode journal. Do not batch
these writes at the end of the episode: every append refreshes the non-canonical `memory/live.json`
progress view in the incumbent workspace.

```bash
{{JOURNAL_COMMAND}} append --path {{JOURNAL_PATH_SHELL}} \
  --experiment-json '{"name":"...","hypothesis":"...","change":"...","evidence":"...","result":"...","decision":"continue|revert|pivot"}'
```

## Terminal contract

Reach exactly one evidence-backed terminal state:

1. `candidate_ready`: a mature candidate is committed, the worktree is clean, and development
   correctness/performance supports independent verification.
2. `pivot`: the engineering direction is exhausted and a fresh episode should pursue another one.
3. `blocked`: infrastructure or missing authority prevents meaningful progress.

For `candidate_ready`, commit the exact candidate, append final evidence, then finalize the journal:

```bash
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
