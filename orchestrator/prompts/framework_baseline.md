# Framework baseline

Produce the first self-contained `{{FRAMEWORK}}` implementation of this operator in `kernel.py`.
V0 is the PyTorch reference wrapper; the Supervisor will register the accepted implementation
as v{{N}} before optimization Episodes begin. If recovering an interrupted session, preserve
and inspect the candidate already on disk instead of starting over.

This is a non-interactive job. Work through concrete in-scope fixes until the prescribed smoke
passes or a technical blocker prevents progress. Correctness and implementation compliance are
the goal, not a speedup over V0. Do not enter optimization iterations, generate a plan, profile,
run ABBA, or add a separate benchmark.

## Context

- Workspace: `{{WORKSPACE}}` (your cwd)
- Baseline measurement: v{{PREV}}; candidate version: v{{N}}
- Platform: `{{PLATFORM}}`; framework: `{{FRAMEWORK}}`
- Additional constraints: {{NOTES}}
{{AGENT_RUNTIME}}

{{HARDWARE}}
{{SANDBOX}}
{{EVALUATOR}}
{{CORRECTNESS_GUIDANCE}}

## Step A — Understand the operator

Read workspace `README.md`, `memory/v0.json`, the current `kernel.py`, and the available public
contract/reference files. For native problems these include `reference.py`, `input.py`, and
public `agent_problem.json` or legacy `shapes.json`; for SOL, use `definition.json`, `reference.py`,
and `workload.jsonl`. Cover the full public domain without searching for hidden evaluator inputs.
Reconcile any supplied correctness guidance with the immutable reference; advice does not
override the operator contract.

## Step B — Resolve missing implementation details

Read only the exact Supervisor-selected implementation references, if they are mounted and
readable. Do not follow their imports/links recursively or search private paths. If essential
framework/toolchain information is still missing, make at most one query:

```bash
python3 tools/sandbox.py --kind wiki-query "<your description>" --brief
```

Include product `{{PLATFORM}}`, runtime architecture `{{ARCH}}`, operator, `{{FRAMEWORK}}`,
public shapes/dtypes, and the missing implementation fact. Inspect the first applicable returned
record and stop research once you can implement and launch the operator. Static references
are design evidence, not external implementations to execute or delegate computation to.

## Step C — Implement and smoke-test

Implement the whole operator yourself in `{{FRAMEWORK}}`, preserving the expected `Model`/`run`
interface. Use dependencies only for allocation, compiler/header/ABI/launch plumbing, or other
non-compute support: no PyTorch compute fallback, prebuilt operators, alternate-DSL compute,
hidden dispatch, or external implementation loading. For Triton, use plain Triton; Gluon is a
later, explicitly requested conversion. For CUDA, embed self-authored `__global__` source and
its supported in-process loader in `kernel.py`, not a separate executable `kernel.cu`. Prefer
CUDA bindings/NVRTC on SOL workers, where `load_inline` is blocked. Keep any supplied
`solution.json` consistent with the implementation.

After the implementation edit, run this smoke command:

```bash
{{SMOKE_COMMAND}}
```
{{SMOKE_SCOPE}}

Do not broaden its scope, add `--multi-seed`, or launch a separate full evaluation. A route
without a subset selector may already use the whole workload in this prescribed command.
On compile/correctness failure, inspect `actionable_diagnostics` in `RESULT_JSON`, repair the
candidate, and rerun the same command. Read existing records rather than repeating an identical
completed task. Runtime handles infrastructure retries; follow its error guidance if returned.

## Finish

After smoke passes, leave those exact bytes in `kernel.py`, print
`v{{N}}: framework candidate smoke-passed ({{FRAMEWORK}})`, and stop. Do not submit `episode-report`,
write canonical memory, or commit. If blocked, state the concrete blocker and available Gateway
Record IDs instead of claiming success.

The Supervisor performs the required full correctness and implementation checks, reusing matching
Runtime measurements or requesting missing measurements. Smoke success is not baseline acceptance;
performance improvement is evaluated in later optimization Episodes.
