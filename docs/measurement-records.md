# Measurement records and execution recovery

GPU work can finish even when the Agent connection, local publication or Supervisor process fails. Without a durable association between the exact inputs and the returned result, retrying can spend another GPU allocation, while reusing a cancelled or incomplete response can turn an infrastructure failure into permanent Candidate rejection.

The Campaign Runtime now retains measurement facts outside Agent workspaces and recognizes identical tasks across Sessions and Episodes. This is the fourth part of the simplified AKA migration: it changes measurement storage and reliability, not the optimization workflow. Setup, Framework Baseline, Fast/Full Episodes, Agent Git operations, Journal/report formats and independent acceptance remain intact. Runtime Journal/Report and Supervisor-owned promotion are separate changes.

## Lifecycle

```mermaid
flowchart TD
    A["Agent: tools/sandbox.py"] --> S["measurements.execute<br/>Snapshot exact inputs + compute task identity"]
    S --> L["MeasurementStore.reserve<br/>Private per-task file lock"]
    L -->|Verified completed record| D["duplicate_gateway_task + Gateway Record ID"]
    D --> R["record-read<br/>Same saved public response; no GPU call"]
    L -->|New or interrupted request| J["gateway_jobs.execute_job"]
    J -->|Known job ID| P["Poll the existing job"]
    J -->|New submission| N["Persist submitting; submit; persist accepted ID"]
    N --> P
    P -->|Confirmed terminal infrastructure failure| B["Backoff then submit a new job"]
    B --> N
    P -->|Complete outcome| C["Persist raw response<br/>ABBA: checkpoint each complete batch"]
    C --> M["measurements.aggregate<br/>One result or per-Shape median"]
    M --> F["MeasurementTask.finish<br/>Persist and validate public result before publication"]
    F --> A
    P -->|Interrupted polling| I["Keep accepted ID for operator-approved recovery"]
    N -->|Submission outcome unknown; no ID| U["Keep uncertain reservation<br/>Require operator reconciliation"]
    F -->|Storage temporarily unreadable| U
```

These boundaries are implemented in `supervisor/measurements.py`, `measurement_records.py`, `gateway_jobs.py` and `abba_checkpoints.py`. `orchestrator/supervisor_runtime.py` owns authorization, private storage and the enclosing deadline. The Agent client does not retry HTTP requests automatically.

## Records and identity

For a Campaign workspace configured as `/work/campaign`, storage is under `/work/.atrex-supervisor-runtime/<workspace-key>/measurements/`, not inside an Episode worktree:

```text
measurements/
├── kernels/kernel-<32 hex>/kernel.py
├── records/gateway-<32 hex>/
│   ├── request.json                  # Semantic request, input/evaluator hashes
│   ├── inputs/                       # Exact Kernel, contract and submitted local inputs
│   ├── repetition-1/
│   │   ├── executor.json             # Bounded raw executor streams
│   │   ├── jobs/<request-hash>/       # Accepted ID, raw outcomes and retry state
│   │   └── abba-batches/             # Completed physical ABBA batch checkpoints
│   └── record.json                   # Bound request + Kernel identities + public response
└── tasks/<request-hash>.json          # Dedup/resume index; separate lock file
```

The configured Campaign root, not an individual Episode path, scopes lookups and deduplication. Moving that root changes its storage key. A library-created Runtime without `RuntimeConfig.workspace` uses temporary storage and does not promise restart recovery. Bubblewrap hides Supervisor storage; native mode still has the operator user's filesystem authority. No retention cleanup is automatic. Private records can contain hidden Shapes and inputs and must not be uploaded as Agent artifacts.

- `kernel-…` is deterministic from the exact `kernel.py` bytes. Identical bytes retain the same ID across Episodes and stores. The underlying SHA-256 remains an internal integrity check, not the Agent-facing ID. Auxiliary source/input files are also part of the task identity, not of this single-file Kernel ID.
- `gateway-…` is a fresh UUID for a new logical request. It binds the Kernel IDs, operation, request digest and stored public result. ABBA records bind both candidate and baseline IDs.
- The task identity includes source/input bytes, operation, measurement options, hardware/transport settings, evaluator code hashes and the Supervisor repetition policy. Changed source, inputs or measurement conditions create a different task. Episode version labels, workspace paths and output publication paths do not make a new measurement.
- Evaluate, ABBA, Profile, Dev, Check and Disassemble are recorded. `env` and Wiki remain queries with their existing diagnostics; they are not Kernel measurements. Standalone operator/verifier execution retains its existing interface and has no Campaign record-query API.

Task reservations are protected by OS file locks, released on process exit. A second live owner cannot dispatch an identical request. Completed records are integrity-checked before reuse. Old/malformed IDs, missing records and identity/digest mismatches invalidate only the stale index, allowing a new reservation while retaining the old evidence. Temporary storage errors instead stop dispatch and preserve the reference; they do not justify duplicate GPU work. Existing PR3 diagnostic files are not imported as measurements.

## Agent interface

Existing result markers and CLI exit codes remain. Structured result markers gain `gateway_record_id` and `kernel_id`; ABBA also gains `baseline_kernel_id`. Arbitrary Dev stdout need not be evaluator output: a separate `[sandbox] RECORD_JSON=` line carries the operation, status and IDs. A Dev probe before Kernel creation has a Gateway Record but no Kernel ID. Raw Gateway envelopes and private inputs never become the record-read result.

```bash
python3 tools/sandbox.py --kind run --no-sync
python3 tools/sandbox.py --kind record-read --record-id gateway-0123456789abcdef0123456789abcdef
python3 tools/sandbox.py --kind kernel-read --kernel-id kernel-0123456789abcdef0123456789abcdef --output-path scratch/previous.py
python3 tools/sandbox.py --kind kernel-records --kernel-id kernel-0123456789abcdef0123456789abcdef
```

`record-read` returns exactly the saved Agent-visible `exit_code`, `stdout` and `stderr`, including a failed measurement's exit code. It does not run the GPU again or republish old Profile files. `kernel-read` writes only to a regular path inside `scratch/` and returns a success message. `kernel-records` lists Gateway Record IDs, operation and timestamp, not all result bodies. IDs in examples are placeholders; use IDs returned in the current Campaign.

An identical completed request is rejected with exit code 2:

```json
{"error":{"code":"duplicate_gateway_task","message":"Identical Gateway task is already recorded","gateway_record_id":"gateway-0123456789abcdef0123456789abcdef","next_action":"Read the recorded result: python3 tools/sandbox.py --kind record-read --record-id gateway-0123456789abcdef0123456789abcdef"}}
```

The duplicate response is not a Kernel failure. Read the result instead of changing version labels or resubmitting to obtain another random sample. Requests whose first execution is still running return a duplicate notice without a readable record ID and should wait for the original request.

## Failure and recovery policy

| Evidence | Action |
| --- | --- |
| Accepted job ID, polling interrupted | Retain the ID; recovery polls that same job before considering another submission |
| Terminal `error_class=infra` or `failure_origin=infrastructure` | New job after 5, 10, 20, 40, then 60-second backoff, within request budgets |
| `logs_unavailable`, including `backend_state=succeeded` | Terminal infrastructure failure: resubmit a new job, not repeated get on the failed ID |
| `command_timeout` or cancelled-before-result without an error | At most one additional submission; never a permanent Candidate-failure cache entry |
| Explicit Candidate/compilation/correctness/validation rejection | Preserve the rejection and deduplicate; repairing inputs/source creates a new task |
| Empty, malformed, unclassified or incomplete result | Not a reusable measurement; preserve available evidence |
| POST interrupted before an accepted ID is durably saved | Outcome unknown; retain reservation and require operator reconciliation, never blindly resubmit |

Retryable failures do not bypass the existing request deadline, revocation or process cleanup. A queued request rejected before any dispatch retains PR3's safe-to-retry 429. A later repetition timing out before its own dispatch is **not** a claim that the overall request submitted no job: it follows the post-dispatch 503 path. Agents should not convert transport/unknown-outcome failures into retry loops. After operator inspection, an unchanged request can recover known job IDs and completed ABBA batches. A no-ID uncertain submission cannot be safely repaired by deleting its index blindly.

Each complete ABBA batch is persisted before the next batch starts. The checkpoint requires the exact schedule, expected Shape coverage, valid latencies for successful runs and explicit correctness results for rejected runs. Incomplete runs never become checkpoints. Recovery keeps completed batches in the same logical record/repetition; it does not reconstruct an ABBA comparison from unrelated standalone Evaluate measurements.

## Optional repeated measurement

The default remains **one** measurement, preserving the upstream measurement policy. Operators may set `ATREX_AKA_MEASUREMENT_REPETITIONS=3` before starting the Campaign. Only `1` and `3` are accepted; the Agent cannot set this policy through request environment flags.

Three repeats apply to full Evaluate and whole ABBA comparisons. The Runtime takes each Shape's median over the three completed measurements, then recomputes geometric/arithmetic means. ABBA aggregates candidate and baseline separately and recomputes their speedup. Correctness-only Evaluate, Profile, Dev, Check and Disassemble remain single-operation requests. A rejected or incomplete repetition cannot become a successful median. Legacy evaluation logs receive the aggregate used in the Agent response, not just the final physical repetition.

This increases GPU cost and wall time; repeats are sequential and do not multiply the enclosing deadline automatically. For repeated or multi-batch work, configure `ATREX_AKA_REQUEST_TIMEOUT_SECONDS` for the whole operation. The repetition count enters the dedup key, so switching policy does not silently reuse a result measured with another count. Dynamic changes in the remote environment behind the same target name cannot be inferred from a local request hash; use a new Campaign/storage scope when intentionally requalifying such an environment.

## Validation and rollback

Regression fixtures run outside the repository, per project convention. They exercise a local fake Gateway and real Supervisor subprocess/HTTP paths: all six measurement operations, exact source/result reads, cross-Session dedup, immutable identity checks, corrupt-index recovery, transient storage failure, accepted-ID polling, new-job infrastructure retry, cancellation/timeout budgets, ABBA checkpoints, default single execution and per-Shape median aggregation. Prior Runtime authorization, hidden-result projection, deadline and publication-failure checks remain applicable. These checks are not production GPU queue-time evidence or a real-model optimization-quality benchmark.

For operator smoke checks, run an ordinary evaluation in a test Campaign, read its returned Record ID, issue the identical request again and confirm that the Gateway job count does not increase. Change Kernel source and confirm a new Record ID. For repeated mode, set the policy to 3 before Campaign startup and inspect the three private repetition directories. Test interruption only against a controlled Gateway; retain the accepted job IDs before restarting.

Rollback: stop the Campaign, preserve private records for investigation, and return to the PR3/previous `aka-light` revision before resuming. Old Journal/numbered-memory/Kernel commit schemas are unchanged. Setting repetitions to 1 disables repeated measurement, not deduplication or recording; there is no switch that grants Agents direct Gateway credentials. Storage growth, changed behavior for duplicate calls and retry cost are deliberate trade-offs of this feature.
