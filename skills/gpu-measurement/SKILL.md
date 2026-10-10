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

Follow the injected evaluator contract. SOL-ExecBench uses `--kind run --no-sync`
for the official full workload; do not add typed `--mode` or top-level seed controls.

Read [requests.md](references/requests.md) for request examples. Use
`--kind OPERATION --help` for full CLI parameters. Evaluate keeps
`[test_kernel] RESULT_JSON=`; Typed Profile returns `[sandbox] PROFILE_JSON=`.
Numbers in those results are measured facts; interpretation is your responsibility.

`memory_sol_pct` describes the busiest memory subsystem, not DRAM bandwidth.
Inspect `dram_throughput_pct`, `traffic` and requested counters to distinguish
DRAM, cache and memory-pipeline activity. `bound`, `dominant_bound` and
`weighted_sol_pct` are coarse hints, not proof of the actual limiting unit or
algorithmic optimality. Do not use `1 / SOL` as an algorithm speedup ceiling;
changing data reuse or work performed can change the relevant resource demand.

Measurements return a `gateway_record_id` and `kernel_id`. Identical tasks across
Episodes are rejected with `duplicate_gateway_task` and the previous Record ID:
read it using `--kind record-read --record-id ID` instead of resubmitting.
Version labels do not request a fresh measurement. See the record-query examples
in [requests.md](references/requests.md). Record reads return the saved result,
not another GPU execution; a saved rejection still has a nonzero exit code.

`SOURCE_ERROR_JSON` with `error.code: "candidate_source_rejected"` is a source
repair, not an infrastructure blocker. Fix the listed forbidden imports,
attribute accesses or string literals in `kernel.py`, then rerun the operation.
No GPU job was submitted by that rejected request. Do not switch to Dev to bypass
the source validator.

`GATEWAY_ERROR_JSON` preserves Agate's original `error` fields, including its
`error_class`, `reason`, `message` and `details`, with credential/path redaction
and output limits. Check the actual error before deciding what to do: compiler,
import and missing-file errors require repairing your code or declared inputs,
not waiting for the Gateway. Custom Dev probes retain their stdout/stderr even
on failure. Profile, Check, Disassemble and ABBA may also include errors in their
result markers. These diagnostics are not proof that another submission is safe;
an unknown outcome still requires reconciliation. Read the saved Record rather
than resubmitting an identical failed task.

Missing/revoked capability or transport failure is an infrastructure blocker.
Do not bypass the Runtime or blindly retry an operation whose outcome is unknown.
Gateway retry policy is owned by the Supervisor; do not add automatic client-side retry loops.
If the Runtime explicitly returns `repairable: true` with `error.code: "request_not_started"`,
this request submitted no job: wait at least `retry_after_seconds`, then retry unchanged
with backoff. This permission does not apply to transport failures or unknown outcomes.
