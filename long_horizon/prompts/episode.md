# Long-horizon kernel exploration episode {{EPISODE}}

You own one complete engineering direction in this single coding-agent session. Continue through as
many profile, research, edit, compile, correctness, benchmark, autotune, and repair cycles as the
direction needs. Do not stop after one edit, one failed compile, or one benchmark while a concrete
next engineering step remains.

## Ownership boundary

You own the inner engineering loop and may make private Git checkpoint commits on the isolated
episode branch. The external supervisor exclusively owns the incumbent branch, authoritative ABBA
verification, and final squash promotion.

- Workspace: `{{WORKSPACE}}`
- Main-compatible campaign version: `v{{VERSION}}`
- Platform: `{{PLATFORM}}`
- Framework: `{{FRAMEWORK}}`
- Incumbent commit: `{{BASE_COMMIT}}`
- Episode branch: `{{EPISODE_BRANCH}}`
- Journal: `{{JOURNAL_PATH}}`
- Handoff: `{{HANDOFF_PATH}}`
- Additional constraints: {{NOTES}}

Never switch branches, push, merge, rebase, or alter refs. Do not edit evaluator/ground-truth files,
including `test_kernel.py`, `definition.json`, `reference.py`, `workload.jsonl`, `input.py`,
`shapes.json`, `metadata.json`, `roofline.json`, `CLAUDE.md`, or `README.md`. Do not write canonical
`memory/vN.json`; the supervisor creates it only after independent verification.

{{MODE_POLICY}}

{{EVALUATOR}}

{{HARDWARE}}

{{SANDBOX}}

## Framework escalation state

{{CONVERSION_DIRECTIVE}}

## Prior episode evidence

```json
{{HISTORY}}
```

Historical attempts are evidence, not orders. Do not repeat a rejected direction unless new evidence
or a materially different implementation makes it worthwhile.

## Development loop and journal

Use the current immutable evaluator for development measurements. Repeated development measurements
are not promotion authority; the supervisor reruns incumbent and candidate in one ABBA allocation.
Inside this mode, ignore any generic sandbox-directive sentence that says to update `memory/v<N>.json`;
run with `--no-memory` and record findings only in the episode journal. A typical full development run is:

```bash
python tools/sandbox.py --kind run --no-sync -- \
  python test_kernel.py --version vlong --no-memory
```

Record every decisive experiment before terminal handoff:

```bash
{{JOURNAL_COMMAND}} append --path {{JOURNAL_PATH_SHELL}} \
  --experiment-json '{"name":"...","hypothesis":"...","change":"...","evidence":"...","result":"...","decision":"continue|revert|pivot"}'
```

The entire episode uses this one journal. Git checkpoints preserve intermediate source states. Keep
temporary regressions only when they are useful steps toward a coherent larger rewrite.

## Terminal contract

Reach exactly one evidence-backed terminal state:

1. `candidate_ready`: a mature candidate is committed, the worktree is clean, and development
   correctness/performance supports independent verification.
2. `pivot`: the current direction is exhausted and a fresh context should pursue another direction.
3. `blocked`: infrastructure or missing authority prevents meaningful progress.

For `candidate_ready`, first commit the exact candidate. Then append final experiment evidence and
finalize the journal using that exact commit:

```bash
candidate_commit=$(git rev-parse HEAD)
{{JOURNAL_COMMAND}} finalize --path {{JOURNAL_PATH_SHELL}} --state candidate_ready \
  --candidate-commit "$candidate_commit" \
  --outcome-json '{"summary":"...","next_directions":["..."]}'
```

For `pivot` or `blocked`, finalize with the corresponding state and omit `--candidate-commit`.
The journal must contain at least one structured experiment and a non-empty outcome summary.

Only after finalizing, atomically publish the small control handoff by writing complete JSON to
`{{HANDOFF_PATH}}.tmp` and renaming it to `{{HANDOFF_PATH}}`:

```json
{
  "status": "candidate_ready | pivot | blocked",
  "candidate_commit": "required only for candidate_ready",
  "last_trial_commit": "optional checkpoint for pivot or blocked"
}
```

Chat text is not a handoff. A missing or invalid file causes the supervisor to resume this same
session. Do not claim a speedup merely to terminate; a well-supported pivot is a valid outcome.

## Inherited current-main optimization playbook

The playbook below is rendered directly from the latest `orchestrator/prompts/iteration.md`; follow
its profiling, research, implementation, sandbox, correctness, multi-seed, benchmark, and evidence
requirements. This long-horizon overlay changes only four iteration mechanics:

1. repeat its engineering cycle as many times as useful instead of stopping after one cycle;
2. write the structured episode journal instead of canonical `memory/v{{VERSION}}.json`;
3. private checkpoint commits are allowed on the isolated episode branch;
4. finish with the atomic terminal handoff above rather than main's single-iteration exit format.

All evaluator, framework policy, dependency, sandbox, profiling, and correctness rules below remain
authoritative.

---

{{MAIN_ITERATION_PLAYBOOK}}
