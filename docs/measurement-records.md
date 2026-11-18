# Measurement records and execution recovery

Light stores measurement facts outside Agent drafts and recognizes identical tasks across
Sessions and Episodes. The Experiment service owns source/request identity, private records,
deduplication, and checkpoints. Evaluation owns plan/result semantics; Execution owns physical
jobs. Research revisions and exact-commit submission review are separate central stores.

## Lifecycle

1. Freeze the complete candidate and trusted evaluator inputs.
2. Compute task identity and acquire its private reservation.
3. Reuse validated completed evidence, or wait inside Runtime for an identical active
   request and return its normal result with the same Gateway Record ID.
4. Dispatch a new job or reconcile its durable accepted ID.
5. Retain raw outcomes and complete ABBA batch checkpoints.
6. Validate the canonical result, seal its public projection, and publish the record.

Current owners are `aka.experiment` for measurement lifecycle and recovery,
`aka.evaluation` for preparation/interpretation, and `aka.execution` for remote jobs.
The Agent ToolHost owns scoped transport; Workflow handlers supply request policy.

## Records and identity

Campaign records live under `<campaign-workspace>/measurements/`.

`reviews/<submission-id|direction-id|experiment-id>/` stores independent Review
state in this same private store. `home/` retains native CLI resume files and
`workspace/` is the stable reviewer project path. Each `vN/` contains `result.json`,
`conversation.jsonl`, `token-usage.json`, and all attempt captures. These files are
not Gateway Records and are never projected to the optimization Agent. Revisions
resume the same reviewer but retain separate audit files; failed calls remain
auditable and their tokens count as Review usage, separately from the main Agent.
The store includes a private identity key, Kernel snapshots, Gateway records with frozen
inputs and repetition/job checkpoints, and task reservation indexes.

The initial V0 full Evaluate is recorded here immediately, using the same
request/source/result integrity envelope as later Evaluates. Its visibility is
Supervisor-private and it is used only as reference evidence; it is not an
accepted submission and does not enter the Pareto frontier or winner selection.
New Campaigns have no separate V0/baseline store. Pre-MeasurementStore V0 files
are read only for legacy migration.

- `kernel_commit_id` is the full Git commit ID and binds the exact committed candidate artifact. For multi-file layouts this includes
  the complete serialized source snapshot, headers, and build settings, not only `kernel.py`.
  The Measurement Store also archives and verifies the serialized source bytes. A private file
  named `kernel.py` may contain the snapshot encoding; use the artifact reader instead of
  assuming Python text.
- A `gateway-...` ID identifies one logical measurement and its public result. ABBA binds
  both candidate and incumbent artifacts.
- Task identity additionally binds evaluator/input bytes, operation, target/transport settings,
  options, and repetition policy. Version labels and output locations do not justify a new sample.
- Custom launcher values use a campaign-private keyed fingerprint. Retain `identity.key`
  during restoration. Requests using custom launcher controls are not ordinary reusable cache hits.
- Env and Wiki queries are not Kernel measurements.

Completed evidence is integrity-checked before reuse. A temporary storage failure preserves
the reservation and stops dispatch; it does not justify another GPU allocation.
Moving the campaign root changes its storage scope. No automatic retention cleanup is provided.

## Agate submission recovery

HTTP 429/502/503/504 during unified Job submission are retried inside the Runtime
using the same endpoint, complete wire body and explicit idempotency key. This
includes HTML proxy errors on Dev/Command. The Experiment `retry_delays` and
`retry_budget_s` configuration also controls these retries: defaults are 5 and
10 seconds (three total attempts), with a 120-second recovery window. An empty
delay list disables resubmission. `Retry-After` is respected; neither recovery
nor collection extends the original request deadline.

Every explicit `error.error_class=infra` response is automatically retryable, including
`logs_unavailable` and `dev_submission_ambiguous`. This applies across Evaluate/ABBA,
Dev/Command, Profile, Check, Disassemble, Env and upload preparation. The explicit
infrastructure class overrides backend `retry_advice=never`, `replay_policy=never`
and `execution_state=unknown`; no inference is made from words in stdout/stderr.
Confirmed failed Jobs are resubmitted with a **new idempotency key**. Before a Job ID
is confirmed, HTTP retries retain the existing key. An accepted Job whose GET fails
transiently is queried again rather than replaced solely because that read failed.
The original command may still run when an ambiguous Job is replaced.

HTTP admission and failed-Job resubmission share one attempt counter, delays and
recovery window; they do not multiply retry budgets. Exhaustion returns the last
original failure. Env and upload preparation use the same configured policy within
their request deadline. Cancellation and non-infrastructure candidate/validation
errors do not trigger this override. An empty retry schedule still disables retries.

Every HTTP attempt and its original error remains in the private execution
archive. A later accepted Job for the identical key and wire request resolves
earlier admission uncertainty for result qualification without erasing those
failed attempts. Each ambiguous Dev resubmission retains its own Job ID, key and
raw response in the same execution archive. The successful replacement result
does not assert that the original command stopped. Query retries continue to use
the Job ID of their own submission attempt.

## Agent interface

Run these commands in an active managed draft, using IDs returned by the campaign:

```bash
python3 tools/sandbox.py --kind evaluate --no-sync
# Read records/*.json, root PROGRESS.md, and the corresponding local/shared Git commits.
```

GPU calls and projected Record files contain `operation`, `status`,
`result`, `gateway_record_id`, and `kernel_commit_id` when source-bound.
ABBA is `operation: evaluate` with a comparison and `baseline_kernel_commit_id`.
Parse the complete JSON, not private executor marker lines.

Initial failed operations retain their nonzero execution outcome. Dev preserves bounded
probe stdout/stderr and process exit code inside `result`.
Reading projected files and Git submits no GPU work and does not recreate old profiler artifacts.
ABBA accepts `--baseline-commit`; Runtime validates reachability and exports the source privately.
Root `PROGRESS.md` returns observed
per-shape minima from eligible passing full Evaluations, not a measured composite kernel.

In pooled search, each lineage sees its local Records, Records cited by the Direction/Experiment
revisions currently visible to that lineage, and passing standard full Evaluates. Accepted submissions
and selected winners therefore expose their complete Evaluate evidence automatically. There is no
Agent operation for publishing arbitrary local Records; all trajectories read the same Campaign
Measurement Store rather than receiving copied records.

Only Evaluate requests, including ABBA, are deduplicated. Command, Dev, Profile, Check and
Disassemble always execute independently, even for the same commit and identical parameters or
concurrent callers. Each invocation receives its own Gateway Record and execution archive;
previous non-Evaluate cache entries are not reused. This does not disable retries within a call.

For Evaluate, the first normal terminal response for an exact request is reusable, regardless of
correctness, compilation success or available performance fields. Subsequent identical calls return
that response without another Agate task, with `reused_gateway_record_id` pointing to the original
Record. This is a reuse notice, not an error or a passing correctness verdict. Infrastructure,
Gateway, transport and request-admission failures are not cached; a later call may retry. All failed
attempts remain archived. An in-progress duplicate blocks inside Runtime until the original result is archived,
then receives that result and a Record projection in its own workspace. Concurrent callers share
failures as well as successes; they do not each dispatch a replacement for a non-cacheable failure.
A later independent Evaluate retry still follows the normal cache and recovery policy.

The waiting caller retains its own request deadline and cancellation. Expiry returns an explicit
`measurement_wait_timeout`; cancellation ends only that caller's wait. Neither cancels the
original request or submits another GPU job. If the original owner is interrupted before archival,
the next owner resumes the same reservation and its existing execution checkpoints.
Do not change labels or delete indexes to force a new random sample.

## Failure and recovery policy

| Evidence | Current behavior |
| --- | --- |
| Proven not submitted, transient infrastructure failure | Experiment retries at most twice, after 5 and 10 seconds, within a 120-second recovery admission window |
| Accepted job ID, collection interrupted | Query the original job; a read timeout does not authorize another submission |
| Explicit `error.error_class=infra` | Automatically retry within the configured budget, overriding backend `never`; a confirmed failed Job gets a new key, and duplicate execution is possible |
| Unknown submission without a job ID | Retain pending state; replay only with an explicitly verified stable-key capability |
| Confirmed failed or terminated job without an infrastructure error | Preserve the result; candidate errors do not trigger infrastructure retries |
| Candidate correctness/compilation failure | Preserve the rejection; source repair creates a different task |
| Cancelled, malformed, contradictory or incomplete result | Preserve uncertainty and job identity; do not record a passing verdict |

Formal and development measurements share the Experiment retry policy. The 120-second
window starts at the first infrastructure failure and controls retry admission, not GPU
execution or normal queue waiting. Retry counts and origin persist for recorded steps.
Workflow does not repeat failed GPU operations around this budget. Review-service outages
have their separate Workflow policy.

Default client waiting adds no independent total deadline. Explicit
`ATREX_SANDBOX_QUEUE_WAIT_GRACE` bounds client waiting at allocation plus grace; its expiry
retains the original job identity and does not establish remote termination. Backend queue,
attempt and execution limits still apply. `gateway.retry_delays` is retired; configure
`experiment.retry_delays` and `experiment.retry_budget_s`. An empty delay list disables
submission retries. `gateway.warmup_iters` is also retired: set `--warmup` on the Evaluation
request. Old retry-ledger formats are not automatically migrated.

A terminal `command_timeout` follows its explicit error class. Infrastructure-class errors
can recover; candidate/unknown errors do not become infrastructure merely because they mention
a timeout. OOM, illegal memory access, or numerical mismatch does not prove device unavailability.

Agate HTTP rejections and terminal errors are returned with their original error fields
and HTTP status. Runtime adds no recovery instructions and does not reclassify the error
for the Agent. Retry classification remains private; after retries are exhausted the
current attempt's original error is returned. A later success does not replay an earlier
attempt's failure. Invalid local endpoint/client configuration remains a Light error
before dispatch. No Eval-to-Dev fallback is introduced.

Cancellation, liveness, and enclosing deadlines bound every retry. Complete ABBA batches
must have the exact schedule and coverage; unrelated standalone measurements cannot reconstruct
a comparison. Unavailable responses retain raw evidence but cannot satisfy a new cache check.
Normal terminal responses remain cacheable even when their correctness or performance fields are
incomplete; that does not qualify them for submission or winner selection.

## Optional repeated measurement

The default is one measurement. Set the Evaluation configuration explicitly for three, retaining Light comparison/batching:

```json
{"api_version":1,"patches":[{"id":"evaluation","config":{"measurement_repetitions":3,"comparison_method":"native-abba","shape_batch_size":1}}]}
```

Three repetitions apply to full Evaluate and whole ABBA comparisons. Per-shape medians are
aggregated into summaries; ABBA sides are aggregated separately. Correctness-only Evaluate,
Profile, Dev, Check, and Disassemble remain single-operation requests. An incomplete or
rejected repetition cannot become a successful median.

Per-Shape numerical checks are preserved in `result.correctness_by_shape` in both
the Agent receipt and the Record. A single measurement exposes the native status
and cases directly; repeated measurements retain those entries in a `runs` list.
Each case retains `outputs`, `mutated_inputs` (declared in-place results), and
`unexpected_mutations` (checks for unintended input changes). Passing and failing
checks keep the original error, numerical errors, mismatch counts, element counts,
and complete tolerance thresholds returned by Agate.

This repetition count is separate from native five-case correctness and ABBA's even
measurement count per side. Standard native Evaluate uses five correctness cases;
matching correctness evidence can be reused by trusted
acceptance without relaxing full-measurement or ABBA identity.

SOL uses its full-workload contract with `sol_execbench_coverage_v1` in task identity.
Earlier records remain readable but cannot substitute for a new contract. SOL source layout
must be explicit, and its ordinary evaluation support does not imply Light ABBA support.

## Public feedback and limits

Runtime requests have independent identities and execution deadlines; a pending Gateway
call or Claim Review does not hold a Session-wide lock. Research operations use isolated
validation snapshots. Revisions of the same Direction, Experiment, or Submission serialize
until the preceding request completes (or the caller cancels); they do not inherit the
60-second dispatch queue limit. Publication holds only a short projection lock. Wiki keeps
its separate configured concurrency budget and explicitly mounts the selected Python/CLI
installation read-only when filesystem isolation is enabled.

Agent Evaluate and ABBA receipts omit internal `evaluator` / `eval_id` fields,
Gateway polling observations (`gateway-job-wait`), and empty diagnostic lists.
Execution audits retain the polling observations and original responses. The public
Gateway Record ID and path remain available for tracing; correctness, performance,
nonempty actionable diagnostics, and original upstream errors remain visible.

Response reuse requires a normal terminal Evaluate response, not a boolean correctness verdict
or complete performance data. Runtime records response availability separately from measurement
qualification. Candidate rejections can be reused; infrastructure and Gateway failures cannot.
`submit-kernel` still requires passing full Evaluate evidence and complete performance results.

Feedback retains actionable compiler, runner, rank-local, numerical, and tolerance details.
Private dimensions/credentials are redacted; failing-case seeds are exposed repair metadata.
The Runtime client preserves complete provider errors and Dev/Command streams rather than
applying a response-length truncation. This consumes memory proportional to the response.
The complete record plus newline must fit 32 MiB before publication.

See [evaluator outcomes](evaluator-outcomes/README.md) for exact cache, diagnostic, and
objective-score versus latency-ratio semantics, and
[submissions](design.md#optimization-and-submissions) for accepted-winner eligibility.
