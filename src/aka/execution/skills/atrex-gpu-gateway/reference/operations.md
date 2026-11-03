# Operations and privileged capabilities

Read this page only for self-hosting, capacity management, diagnosis or raw dev.
Ordinary GPU jobs use the main CLI workflow. See [capacity-control.md](capacity-control.md) for capacity commands and handoff procedures.

## Raw dev

`agate dev` is a privileged raw shell, disabled by default. Its AK must be in `auth.dev_allowed_aks`.
The deployment must establish network/execution isolation for target clusters.
An unset `supported_kinds` allows every kind, including dev. Prefer `check`, `eval`, `profile`, `disassemble` or `env` whenever they cover the task.

```bash
agate dev 'python run.py' --gpu H20 --file run.py=./run.py --intent custom_harness
```

Raw dev requires recycling; `--no-recycle` is rejected.
asicli confirms recycling through a change in container PID 1 generation; do not enable backends without equivalent isolation.
String blocking is defense in depth, not a replacement for network policies or RunD/Kata isolation.
Command bodies are excluded from ordinary request logs, audit and Hologres.
Allow/deny records retain identity, digest, byte count and bounded risk labels.
Corporate browser login does not replace dev AK/SK authentication or its dedicated ACL.

Confirmed-terminal retryable infrastructure failures receive bounded `max_attempts` retries, defaulting to the initial attempt plus one retry;
commands must tolerate repeated execution. Submission disconnection/slow acknowledgment checks the original receipt/identity for at most 60 seconds;
unknown outcomes are not re-executed. For `dev_submission_ambiguous`, inspect external side effects before deciding to create a new job.
An explicitly empty ACL returns `403 dev_disabled`; an AK outside it returns `403 dev_acl_denied`;
all matching clusters disabling dev returns `400 dev_cluster_disabled`.
Default job timeout is 300 seconds; the default deployment permits adjustment to 1800 seconds, subject to command policy,
with an absolute code limit of 10800 seconds.

## Capacity, queues and failover

A scale-up target is not ready GPU capacity. Keep polling the original job while waiting for actual supply.
Use conservative quotas for unknown/stale capacity; node specifications do not replace online supply,
and reachability alone does not establish capacity readiness.
The default unfinished-GPU-unit budget per pool is roughly two waves of online capacity; an ordinary single AK receives roughly half.
Multi-GPU jobs count GPUs, and zero-GPU jobs consume at least one unit. Administrators may hot-update budgets;
capacity-cache TTL does not determine queue duration.

Unexecuted jobs may switch among compatible pools matched at submission, retaining hardware, GPU-count, placement, kind and OSS constraints.
If the original pool is disabled/removed without a compatible alternative, return `infra/resource_pool_unavailable`.
Temporary capacity failure/full destination queues retain the original deadline and wait.
Unknown submissions are not replayed; executed dev commands are not re-executed because of pool switching.
New pools or pools with changed execution configuration do not automatically become candidates for old jobs.
Healthy instances may take over failed instances while retaining `job_id`, idempotency key and remote attempt.
Only explicit administrator confirmation that the original cluster was destroyed permits closure as `backend_cluster_destroyed`;
re-execution requires a new request.

Follow the capacity-control reference for Ray floors, cross-environment ownership and compatibility leases.
Pause gateway capacity writes before modifying Ray directly. Gateway readiness does not wait for every cluster registration;
unready pools independently stop new routes/admission.

## Self-hosting and observations

The server uses a Rust coordinator without a Python fallback. Single-instance SQLite and multi-instance PostgreSQL share the same job entry point.
For SQLite upgrades, stop the old owner, back up the database, provide a matching Rust binary and remove old `off/passive/active` settings.
PG takeover uses job claim/epoch. Cross-database migration requires stopping old writers and offline verification, without runtime-state dual writes.
PG rolling upgrades enable new runtime semantics after old members leave; rollback must support the same protocol.
Temporary runtime-database outage pauses admission; after lease/durable-state recovery, continue original jobs.
Do not replay write transactions with unknown outcomes. `coordinator_capability_unsupported` requires deployment-side capability checks,
not resubmission under a different idempotency key.

For a local GPU, deploy the gateway on that GPU node with `python -m app.main` and configure a `local` cluster;
clients still connect over HTTP. Local execution uses gateway-user privileges without Pod/container isolation.
Client and server remain separate installations.

Holo observations and operations are independent of runtime state.
The gateway delivers the three observation fact categories by default; only explicit
`observation_mirror.delivery_mode=external_cdc` delegates delivery to external CDC, retaining PG persistence and pausing source cleanup.
Switching requires acceptance of the external path, capacity budget and safe cleanup.
Rollback must handle historical backlog, not merely disable mirroring.
PG writes, claim/ack, maintenance and Holo delivery failures are reported separately; maintenance failure is not job execution failure.

Homepage job composition/timings use the terminal-time window across all environments, including cancellations;
they are not submission counts or currently running counts. Reliability failure rate counts infrastructure failures only, with timeouts reported separately.
Unavailable ledgers/stale caches retain unknown/stale markers.
`/ops` shows authorized instance health and abnormal runtime state; interrupted sampling neither means lease expiry nor directly rewrites runtime state.
SQL cancellation affects only the original request. In `GET /dashboard/api/stats`,
`business_runtime.http_runtime.workers[].state_store.query_cancellations` is a process-local cumulative count.
Interpret freshness alongside worker `age_s`; it resets on restart and is unavailable for SQLite.

DingTalk monitoring frequency, thresholds, windows, switches and delivery settings support Diamond hot updates.
Parameter edits retain events; disabling a category clears that category's events. Re-enabling traffic detection warms its baseline again.
Cleanup alerts are disabled by default. Do not sum readings from multiple instances of the PG shared projection.

## TaskModule preview

TaskModule is a self-hosted preview with no `agate` subcommand; use `atrex-gateway module validate|test` and `/v1/modules`.
Online preview is disabled by default and still requires a trusted AK plus cluster opt-in for `module`/`runtime_classes`.
The SQLite registry is memory-only; the PG registry is shared/durable. It is not a general production artifact service.
Registration/reads return file digests by default; input/source contents stay out of request logs.
The schema is a bounded subset rejecting `$ref`, regex and recursive composition keywords. OSS attachments are unsupported.
Terminal JSON is limited to 256 KiB; use declared artifacts for larger content.
Ray stage admission requires native Dashboard/JobDetails visibility to be operator-only because the driver receives a callback bearer.
Keep the feature disabled if that control-plane boundary cannot be enforced.
