---
name: gpu-measurement
description: Run Kernel evaluation, ABBA comparison, profiling and GPU probes through the session's Supervisor HTTP Runtime.
---

# GPU measurement

Use `python3 tools/sandbox.py`. The Supervisor chooses the Gateway/SSH endpoint,
hardware, credentials and timeout, snapshots the current workspace, and supplies
the evaluator and private inputs. Do not set endpoint/credential flags or run GPU
code locally. Every remote job starts fresh; no remote filesystem is retained.

Follow the current session's measurement scope and finish procedure. Framework
Baseline permits only its prescribed smoke; optimization Episodes may use the
operations below. Measurements and comparisons do not grant promotion authority.

Read [requests.md](references/requests.md) for request examples. Use
`--kind OPERATION --help` for full CLI parameters. Evaluate keeps
`[test_kernel] RESULT_JSON=`; Profile returns `[sandbox] PROFILE_JSON=`.
Numbers in those results are measured facts; interpretation is your responsibility.

Measurements return a `gateway_record_id` and `kernel_id`. Identical tasks across
Episodes are rejected with `duplicate_gateway_task` and the previous Record ID:
read it using `--kind record-read --record-id ID` instead of resubmitting.
Version labels do not request a fresh measurement. See the record-query examples
in [requests.md](references/requests.md). Record reads return the saved result,
not another GPU execution; a saved rejection still has a nonzero exit code.

Missing/revoked capability or transport failure is an infrastructure blocker.
Do not bypass the Runtime or blindly retry an operation whose outcome is unknown.
Gateway retry policy is owned by the Supervisor; do not add automatic client-side retry loops.
If the Runtime explicitly returns `repairable: true` with `error.code: "request_not_started"`,
this request submitted no job: wait at least `retry_after_seconds`, then retry unchanged
with backoff. This permission does not apply to transport failures or unknown outcomes.
