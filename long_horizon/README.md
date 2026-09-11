# Episode supervisor internals

This package implements the native optimization engine used by
`orchestrator/optimize.py`. It is not a separate command-line entry point.

Each canonical optimization version is explored in an isolated Git branch and worktree. A coding
agent may run multiple related profile/research/edit/validate cycles, preserve private checkpoint
commits, and finally publish one structured handoff: `candidate_ready`, `pivot`, or `blocked`.

The supervisor validates the journal and candidate commit, checks production policy, and evaluates
incumbent and candidate in an exact same-allocation ABBA schedule. A strict correctness-passing
improvement is squash-promoted to the incumbent; every other outcome records canonical
`memory/vN.json` evidence without changing the incumbent kernel.

An episode candidate commit contains only `kernel.py`. Plans, profiles, planner discussions,
journals, and handoffs stay uncommitted and are copied into the episode archive before the isolated
worktree is removed.

Runtime state lives under `.atrex_long_horizon/` in generated campaign workspaces. Public options
such as `--handoff-resumes`, `--verify-repeats`, `--verify-run-timeout`, and
`--min-improvement-pct` are parsed directly by `orchestrator/optimize.py`.

Each active episode also exposes ignored `memory/live.json`. It is initialized immediately and
atomically refreshed after every journal append, but it never participates in version selection or
promotion; `memory/vN.json` remains the canonical supervisor-owned record.

Every canonical record carries a compact copy of all structured experiments already persisted in
the episode journal. If the supervisor is terminated while an episode is active, the next startup
resumes the registered episode worktree in place, including its source edits, checkpoints, journal,
plans, profiles, and generated intermediate files. If that worktree is missing or no longer matches
the recorded branch and baseline, recovery falls back to archiving it and recording an
`interrupted` `memory/vN.json`. Recovery remains idempotent across repeated termination.

Claude and Codex can resume the same session to repair an incomplete handoff; Qoder and Pi use a
single long invocation. Codex token deltas and marker ordering are read incrementally from the
resumable native rollout. Available invocation components must reconcile with cumulative rollout and
`turn.completed` totals before attribution. Reconciled events may form one phase interval across a
resume boundary. If ledger observation fails, consecutive cumulative stdout usage still supplies a
non-duplicated invocation total while phase attribution degrades fail-closed.

## Module responsibilities

- `campaign.py`: episode budgets, recovery, terminal-state processing, and promotion decisions.
- `git_episode.py`: private branch/worktree lifecycle, protected-path checks, squash promotion,
  and canonical outcome commits.
- `session.py`: one long coding-agent invocation plus bounded same-thread recovery for Claude and
  Codex.
- `journal.py` and `protocol.py`: atomic journal/handoff I/O and terminal validation.
- `verifier.py` and `remote_abba.py`: one-allocation incumbent/candidate ABBA execution.
- `store.py` and `telemetry.py`: restart state, archived attempts, and best-effort episode metrics.

`.atrex_long_horizon/state.json` and `active_episode.json` are restart state, while each
`episodes/eNNNN/` directory archives the prompt, journal-derived attempt, worktree snapshot,
verification payload, and telemetry available for that episode. These files are intentionally
excluded from campaign commits. Accepted evidence is also written to committed
`memory/long_horizon_eNNNN.json` and the canonical `memory/v<N>.json`.

At normal completion, Python failure, `SIGINT`, `SIGTERM`, or `SIGHUP`, the orchestrator writes an ignored
`trace-retention-manifest.json`. It declares only the candidate sources,
canonical memory, episode journals and evaluations, Wiki query events, and
compact profiler reports needed for offline knowledge extraction. It explicitly
does not declare coding-agent session JSONL, stdout/stderr logs, temporary
worktrees, caches, or bulk profiler captures. The manifest also binds the
consumer-supplied `platform`, resolved `arch`, and `sandbox_hardware` to the run;
these values are deployment evidence that cannot be reconstructed reliably from
the workspace name or from a locally scoped `solution.json`. A deployment hook
may use this manifest as producer evidence, but remains responsible for path
validation, secret scanning, archive limits, transport, and retry.
`SIGKILL` cannot be handled by Python; a completion hook must treat a missing
manifest after forcible termination as an interrupted/incomplete run.

### Production numerical safety

Production V1, resume and fast/full candidates use the same numerical gate,
independent of framework/dependency review. The gate works with the Atrex
`input.py` / `reference.py` / `shapes.json` contract. Attention, GEMM, norm and
other/fused operators share one runner, schedule and reviewer; the engine has no
operator-name dispatch, fixed tensor names, TP-specific path or error-file parsing.

An operator may supply private `numerical_suite.json`. Otherwise a read-only
contract-author session builds 3–6 complementary cases from trusted reference,
input and public contract files, without candidate source. The result is cached in
`<private-reference>/.atrex_numerical/<contract-digest>/`; changed reference/input/
shapes or author instructions invalidate it. Examples for attention, GEMM and norm
are in `reference/numerical_suites/`. They illustrate mathematical risks, not fixed
ABI names or universally valid numeric ranges. Contract construction errors block
certification instead of guessing valid inputs. The candidate's independent
numerical reviewer still checks the generated suite's relevance and sufficiency.

Each case declares every input's constructor or structural-preservation reason.
Uniform, log-uniform, sparse, alternating, constant, axis-wise ramp, near-constant
and packed-byte constructors generate values independently of the ordinary input
samples. Tensor shape, dtype and strides are preserved. Masks, lengths, indices,
packed formats and coupled metadata must remain contract-valid. Examples cover
softmax saturation/near-equal logits, GEMM cancellation and sparse operands, and
low-variance/epsilon-sensitive normalization. Custom legal risks use the same schema.

`--numerical-gate auto` selects one of two depths: **light** for SSH or a
loopback gateway, **thorough** for remote agate. An explicit gateway profile selects
remote agate; otherwise URL, `AGATE_URL`, then agate's config/default endpoint
resolve the destination. Override depth with `--numerical-gate light|thorough`.

Default `coverage: "compact"` separates numerical stress from ordinary full-shape
correctness/ABBA:

| Depth | Representative shapes per risk | Seeds | Four risks, single GPU | Twelve risks, TP2 |
| --- | --- | --- | --- | --- |
| light | Largest + rotating shape (up to 2) | 1; first risk gets 2 | 10 probes | 52 rank-probes |
| thorough | Smallest + largest + rotating middle (up to 3) | 2 per risk | 24 probes | 144 rank-probes |

Counts assume nine available shapes and exclude ordinary full-shape checks. The
previous all-shapes/two-seeds product cost 72 and 432 probes respectively. Workload
size is a heuristic based on numeric input parameters, not guaranteed dispatch-path
coverage. A case's private `shape_ids` pins required regression/dispatch witnesses.
`coverage: "exhaustive"` explicitly requests all shapes × all configured seeds.
Every required rank runs each probe; distributed rank coverage is never reduced.

Remote agate starts **all independent distribution case workers concurrently**, with one
allocation per case and no local worker cap (a four-case recipe means four jobs;
the twelve-case TP2 suite means twelve jobs). Gateway quotas and queue admission
control actual GPU concurrency. Local sandbox processes share an endpoint-specific
admission lock around upload and `--no-wait` submission, preventing ingress overload.
The lock is released as soon as the job is accepted, before any result polling;
accepted GPU jobs remain parallel. Shapes/seeds within a case reuse that allocation.
SSH and loopback gateways execute cases serially. This scheduling choice applies
even with a manual depth override. The supervisor includes the sandbox's queue
allowance in its wall timeout; worker execution remains capped at 600 seconds.
A failed case cancels sibling jobs through sandbox cleanup. Successful results are
assembled in suite order regardless of completion order.

`--correctness-only` skips timing even with one seed. Receipts, selected-shape digest
and expected seed/rank counts must match the supervisor's exact plan. Missing
evidence, a failed probe or a reviewer rejection blocks promotion. Evidence/cache
identity includes depth, so a light pass cannot satisfy a thorough gate.

A numerically failing production HEAD blocks normal resume. To repair it in an
optimization episode, explicitly pass `--repair-numerical-head`. Dependency/framework
review must pass before repair admission; the numerical gate still runs and records
its rejection. The episode receives the failure and treats that HEAD as uncertified
comparison evidence. Every candidate still passes the complete production and ABBA
gates before promotion. The repair directive expires when canonical HEAD changes.
Large candidates can use `--production-review-timeout SECONDS` to give the independent
dependency reviewer more time; the default is 600 seconds. A timeout never counts as
an approval.

For native Atrex workspaces, the supervisor bundles its current transport adapter
and installs it only inside the temporary GPU allocation. Original input/harness
files are restored in `finally`. Custom evaluators explicitly declare an
`evaluator_command` with compatible `--shape-id`, `--multi-seed`,
`--correctness-only` and RESULT_JSON support. Optional operator-owned
`evaluator_files` bind adapters into the evidence digest. No guessed TP filenames
or custom metric paths exist in the shared driver. Numerical metrics travel in
RESULT_JSON; each operator retains its own official comparator and tolerances.

After dynamic probes pass, a separate read-only numerical reviewer checks domain,
precision/reductions, nonlinear/quantization math, routing/boundaries and coverage.
It sees the compact coverage policy and must identify concrete untested risks,
not demand a Cartesian product automatically. A sampled or exhaustive finite suite
is empirical evidence, never a proof over all finite values. Exact hidden shapes
and metadata are withheld from this reviewer. Numerical receipts and aggregate
metrics are returned through the sanitized sandbox protocol.

Evidence is saved under `verification_artifacts/.atrex_long_horizon_verify/`.
The in-process certificate cache binds candidate, suite, trusted reference/input/
shapes, evaluator, transport, driver, reviewer prompt and public contract bytes.
Restart revalidates a candidate. No tolerance is relaxed to make a test pass.

## Infrastructure recovery during validation

Confirmed GPU transport outages and structured review-service errors pause the current validation
step. After each failure, the supervisor waits 30 minutes before retrying the same
step. Recovery is established by a real validation request, rather than a health
endpoint alone. There is no retry-count limit and no episode, rejection, or stall
increment while waiting. The candidate, journal, and handoff remain in place.

Each step stores its status, failure category, retry count, and next retry time under
`.atrex_long_horizon/infrastructure/`. Step identities include the candidate or
contract digest, and a restarted supervisor honors the recorded retry deadline.
The independent numerical cases retain their configured concurrency; a failed case
waits and retries without rerunning successful cases in the same supervisor process.
An interrupted ABBA batch repeats its complete A/B/B/A schedule in one allocation;
partial timing samples are never combined across allocations.

Explicit numerical mismatches, compilation or kernel execution failures, policy
rejections, invalid evidence, and insufficient speedup remain validation failures.
They are not treated as transport outages. Gateway infrastructure categories remain
visible through a fixed marker while hidden evaluator details stay private.

Reviewer exits without a structured service error fail validation. A reviewer execution
timeout gets one retry in a fresh isolated session with restored evidence, for dependency
review, numerical review and suite authoring alike. Timeout counts are persisted per
evidence digest and reviewer configuration; a second timeout blocks validation with an
explicit reviewer-infrastructure diagnosis. Restarting cannot reset the limit. Changing
the evidence or reviewer timeout permits a new bounded attempt. Successful completion
clears consecutive timeout counts. Only top-level CLI service errors (`overloaded_error`,
`rate_limit_error`, `service_unavailable_error`) enter the 30-minute retry loop.
`--production-review-timeout` and `--numerical-review-timeout` independently bound
the dependency and numerical reviewers (both default to 600 seconds).

Sandbox infrastructure errors use exit code 75 plus an exact stderr marker;
remote command text cannot declare this category. Agate upload/nonblocking
submission holds its admission lock for at most 600 seconds, limited further by
the remaining wait budget. A submission deadline releases the lock and returns
the infrastructure signal so subsequent jobs can submit.
