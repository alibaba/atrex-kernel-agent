# Measurement records and execution recovery

Light stores measurement facts outside Agent drafts and recognizes identical tasks across
Sessions and Episodes. The Experiment service owns source/request identity, private records,
deduplication, and checkpoints. Evaluation owns plan/result semantics; Execution owns physical
jobs. Journal annotations and promotion decisions are separate responsibilities.

## Lifecycle

1. Freeze the complete candidate and trusted evaluator inputs.
2. Compute task identity and acquire its private reservation.
3. Return an existing record reference for an Agent duplicate, or let trusted acceptance
   reuse compatible completed evidence.
4. Dispatch a new job or reconcile its durable accepted ID.
5. Retain raw outcomes and complete ABBA batch checkpoints.
6. Validate the canonical result, seal its public projection, and publish the record.

Current owners are `aka.experiment` for measurement lifecycle and recovery,
`aka.evaluation` for preparation/interpretation, and `aka.execution` for remote jobs.
The Agent ToolHost owns scoped transport; Workflow handlers supply request policy.

## Records and identity

Campaign records live under
`<workspace-parent>/.atrex-supervisor-runtime/<workspace-key>/measurements/`.
The store includes a private identity key, Kernel snapshots, Gateway records with frozen
inputs and repetition/job checkpoints, and task reservation indexes.

- A `kernel-...` ID binds the exact candidate artifact. For multi-file layouts this includes
  the complete serialized source snapshot, headers, and build settings, not only `kernel.py`.
  Historical single-file IDs retain their interpretation. A private file named `kernel.py`
  may contain the snapshot encoding; use the artifact reader instead of assuming Python text.
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

## Agent interface

Run these commands in an active managed draft, using IDs returned by the campaign:

```bash
python3 tools/sandbox.py --kind run --no-sync
python3 tools/sandbox.py --kind record-read --record-id gateway-11111111111111111111111111111111
python3 tools/sandbox.py --kind kernel-records --kernel-id kernel-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
python3 tools/sandbox.py --kind kernel-read --kernel-id kernel-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa --output-path scratch/previous
python3 tools/sandbox.py --kind kernel-pareto-frontier
```

GPU calls and `record-read` return one JSON object containing `operation`, `status`,
`result`, `gateway_record_id`, and `kernel_id` when source-bound.
ABBA is `operation: evaluate` with a comparison and baseline Kernel ID.
Parse the complete JSON, not private executor marker lines.

A successful read exits zero even when the stored evaluator rejected the source.
Initial failed operations retain their nonzero execution outcome. Dev preserves bounded
probe stdout/stderr and process exit code inside `result`.
Record reads submit no GPU work and do not recreate old profiler artifacts.

`kernel-read` restores a file for `single_file` and a directory for multi-file artifacts.
Use that same layout for `--baseline-path`. `kernel-records` lists metadata; a listed
ID still needs full integrity validation when read. The Pareto query returns observed
per-shape minima from eligible passing full Evaluations, not a measured composite kernel.

An Agent duplicate returns `duplicate_gateway_task` with an existing record reference
and recovery guidance. In-progress duplicates instruct the caller to wait.
Do not change labels or delete indexes to force a new random sample.

## Failure and recovery policy

| Evidence | Current behavior |
| --- | --- |
| Proven not submitted, transient infrastructure failure | Experiment retries at most twice, after 5 and 10 seconds, within a 120-second recovery admission window |
| Accepted job ID, collection interrupted | Query the original job; a read timeout does not authorize another submission |
| Unknown submission without a job ID | Retain pending state; replay only with an explicitly verified stable-key capability |
| Confirmed failed or terminated job | Preserve the result; remeasurement or backend change requires explicit opt-in |
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

Known-not-submitted candidate source rejection produces `candidate_source_rejected` and
repair guidance, without an Eval-to-Dev fallback. Invalid endpoint/client configuration is
an operator failure before dispatch.

Cancellation, liveness, and enclosing deadlines bound every retry. Complete ABBA batches
must have the exact schedule and coverage; unrelated standalone measurements cannot reconstruct
a comparison. Unavailable or incomplete historical records retain raw evidence but cannot
satisfy a new completed-result cache check; completed candidate rejections remain cacheable.

## Optional repeated measurement

The default is one measurement. Set the Evaluation configuration explicitly for three, retaining Light comparison/batching:

```json
{"api_version":1,"patches":[{"id":"evaluation","config":{"measurement_repetitions":3,"comparison_method":"native-abba","shape_batch_size":1}}]}
```

Three repetitions apply to full Evaluate and whole ABBA comparisons. Per-shape medians are
aggregated into summaries; ABBA sides are aggregated separately. Correctness-only Evaluate,
Profile, Dev, Check, and Disassemble remain single-operation requests. An incomplete or
rejected repetition cannot become a successful median.

This repetition count is separate from native six-case correctness and ABBA's even
measurement count per side. Standard native Evaluate defaults to six correctness cases;
targeted smoke tests default to one. Matching correctness evidence can be reused by trusted
acceptance without relaxing full-measurement or ABBA identity.

SOL uses its full-workload contract with `sol_execbench_coverage_v1` in task identity.
Earlier records remain readable but cannot substitute for a new contract. SOL source layout
must be explicit, and its ordinary evaluation support does not imply Light ABBA support.

## Public feedback and limits

Canonical publication/reuse requires a completed boolean verdict and known outcome.
A successful transport with a failed or missing evaluator receipt is not a successful evaluation.
Completed candidate rejection remains useful evidence; infrastructure unavailability does not.

Feedback retains actionable compiler, runner, rank-local, numerical, and tolerance details.
Private dimensions/credentials are redacted; failing-case seeds are exposed repair metadata.
The diagnostic response limit is 8 MiB of ASCII-escaped JSON, with explicit omissions.
The complete record plus newline must fit 32 MiB before publication.

See [evaluator outcomes](evaluator-outcomes/README.md) for exact cache, diagnostic, and
objective-score versus latency-ratio semantics, and
[Supervisor promotion](supervisor-promotion.md) for acceptance reuse.
