# Jobs, attachments and troubleshooting

## Job identity and waiting

`eval` submits and waits for a terminal state by default. `--no-wait` returns only `job_id`; later run `agate get JOB_ID --wait`.
Raw `agate submit payload.json` does not wait by default; add `--wait`.
`execution_phase` distinguishes gateway queueing, backend submission/queueing, execution and collection;
`phase_elapsed_ms` measures the current phase. Cancellation acceptance does not confirm remote termination; keep querying the original `job_id`.

| Timeout option | Scope |
|---|---|
| `--job-timeout` | Worker job budget, subject to command and server configuration |
| `--wait-timeout` | CLI budget for waiting on terminal state |
| `--http-timeout` | Individual HTTP request budget |

Supported options vary by subcommand; inspect `--help`. Deprecated `--timeout` sets all three.
Gateway queueing is capped at two hours by default; shorter existing budgets still apply, and restarts/retries do not reset the start time.
`timeout/gateway_queue_timeout` means gateway queue expiry; `exclusive_lane_queue_timeout` means an exclusive profile job has not reached the backend.

Execution timeout returns `failed` and `finished_at`. `cleanup_pending=true` means remote termination still needs confirmation;
reservations remain during bounded cleanup. Missing receipts are archived immediately; other timeout cleanup lasts at most 60 seconds.
`execution_quarantined=true` with `cleanup_pending=false` means unknown execution is archived with its original identity,
and the original remote is no longer accessed automatically. A 404 does not prove non-execution.
Do not change identity and resubmit because of timeout, cancellation or disconnection.
Confirmed remote terminal results still finish within the collection budget; disabling the original pool does not discard an already-collected path.

## Rate limits, retries and permissions

HTTP 429 `gateway_overloaded`, `gpu_queue_full` and `tenant_gpu_queue_full` create no job.
The client honors `Retry-After` and uses exponential backoff with jitter. The 30-second cap applies only to extra randomized backoff,
not the server's requested wait. With insufficient budget, return the original error; callers may retry the same request/key within their business deadline.
One call-level idempotency key is generated per logical submission. Cross-process deduplication requires an explicitly supplied stable key.
Throttle batches by unfinished GPU units for the same AK; coordinate the aggregate across processes sharing an AK, not merely HTTP concurrency.

`503 coordinator_busy` may arise while backfilling historical demand; it creates no job and does not trigger the SDK's automatic 429 retry.
For `503 cluster_configuration_changed`, wait for configuration synchronization and retry with the same key.
`403 cluster_ak_denied` requires administrator authorization or another environment; do not retry it as overload.
Administrator identity/employee ID does not replace AK permissions. Permission changes do not cancel existing jobs;
queries, cancellation and idempotent replay retain the original request's permissions. Cross-tenant queries return 404.

`503 job_record_unreadable` with `details.scope=job` affects only that record; preserve its identity and wait for repair while other jobs remain queryable.
For `owner_unavailable` or disconnection, also confirm the original submission first.
`ipc_connection_limit` indicates IPC saturation. `ipc_operation_failed`/`ipc_response_invalid` indicate business/encoding errors;
retain the trace ID for operations rather than treating every error as an unreachable owner.

Ordinary reads and unified idempotent submissions recover from transient transport failure at most twice by default,
sharing the original HTTP budget, body and idempotency key.
Uploads, cancellation, legacy submission endpoints, forced probing and permanent DNS/TLS-validation errors do not recover automatically.
SDK `TransportError` has `status=None`; safe details include reason, phase, errno and attempt count.
Use `Client(..., transport_retries=0)` to observe raw stability. Successful recovery does not erase earlier failures.

## Dependencies and environment variables

```bash
agate eval --gpu H20 --candidate my_kernel.py --reference-dir data/rms_norm -r requirements.txt
agate eval --gpu H20 --candidate my_kernel.py --reference-dir data/rms_norm -e TORCH_LOGS=+dynamo
```

Repeat `-r` for pip specifications or HTTPS wheels. HTTP/file/git+/editable/local-path inputs are rejected.
Default `freeze_installed` reuses satisfied base packages and installs only missing/unsatisfied dependencies.
`no_deps` installs only listed packages, so list missing transitive dependencies yourself.
Installation is isolated per job and supported for both eval harnesses, profile, check and disassemble.
Wheels must be reachable from the target GPU region. Installation failure returns `deps_install_failed`, with tool errors in `details.logs_tail`.
`-e/--env-var` follows the server's prefix allowlist, such as `TRITON_`/`TORCH_`; it is not arbitrary environment access.

## OSS attachments

`eval`, SOL, `profile`, `check` and `disassemble` accept repeatable `--oss-file DEST=LOCALPATH` and `--oss-output PATH`.
Client and worker access OSS directly; bytes do not pass through gateway JSON.
Outputs expose temporary URLs in `artifacts[]` without automatic download. Directories/globs are unsupported; package them into a regular file first.
The final `job_id` is reserved before upload and printed even after upload/submission failure, enabling cleanup under the same prefix.
Attachments cannot overwrite gateway-validated candidate, solution or driver files; conflicts are rejected before queueing.
`dev` does not support these attachment options; it uses `--file`.

`oss_unavailable` means the authorized resources lack OSS.
`400 kind_not_allowed` means every matching resource prohibits the command; choose another GPU type/command rather than retrying as an OSS failure.
Upload preparation and submission both enforce AK, command and placement constraints.

## Error attribution and evidence

- Fix the candidate for `code/*` and the request for `validation/*`. Interpret `infra/*` and `timeout/*` using execution evidence and remaining budget.
  In particular, do not rerun a dev command with external side effects merely because an error is classified as infrastructure.
- Triton OutOfResources and ptxas compilation errors are attributed to the candidate; `details.failure_rule` identifies the evidence rule.
  Cases such as `ptxas-blackwell` return `code/codegen_error`; retain concrete assembly diagnostics first.
- For atrex-bench environment/version errors, `next step:` distinguishes request pins from image configuration.
  Use `agate update` for an outdated client. Package-index timeout may return `infra/deps_install_timeout`.
- With no independent result/structured terminal evidence and inaccessible logs, return `infra/logs_unavailable`.
  A persisted native failure retains its original reason and adds `details.output_unavailable_reason=logs_unavailable`.
  Readable logs with no identifiable cause produce `infra/no_result`; retained remote diagnostic records expose `details.gc_deferred=true`.
- Candidate exceptions before profiling return `code/code_execution_failed`.
  Successful-result notes print real newlines to stderr without changing stdout JSON.
  `eval_result_diagnostic=unsupported_verdict_layout` means only an unknown structure, not proven code failure.
- SOL nvcc/TileLang compilation failure before the first trace returns `code/code_execution_failed` with `failure_rule=sol_native_compile_error`.
  `details.execution_diagnostic` retains bounded exception summaries, exit status and byte counts.
  `scan_complete=false` means incomplete evidence; absent fields may indicate an older worker. Valid traces take precedence.
  `details.result_payload_state` distinguishes empty, unparseable, missing-marker and truncated payloads.
- An explicit SOL candidate segmentation fault returns `code/code_execution_failed`, preserving `sol_native_crash` and `segmentation_fault`.
  It is not retried automatically or counted as a cluster failure. Native abort remains `infra/native_crash` with `failure_origin=unknown`.
  Exit 137/139 alone proves neither OOM nor candidate error; without other evidence it remains `no_result`.
