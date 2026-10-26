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

Read [requests.md](references/requests.md) to choose an operation, prepare its inputs,
and interpret its result or failure; it includes executable CLI examples. Use
`--kind OPERATION --help` for full CLI parameters. Measurements return one JSON
object with `operation`, `status`, `result` and saved IDs. ABBA is `operation: evaluate`
with `result.comparison`. Read metrics and diagnostics inside `result`, not stdout markers.
Numbers in those results are measured facts; interpretation is your responsibility.

For each call, identify the uncertainty it will resolve and the observation that would
support or contradict your hypothesis. Do not promote a plausible explanation to a fact
because one aggregate metric looks consistent with it. No separate plan document is needed.

Follow `CLAUDE.md`'s foreground/wait rules. Runtime requests block; the session Prompt states
whether the Supervisor also pauses the Agent tree and excludes that wait from active time.
Do not wrap commands in shell timeouts, background them, or edit the candidate while a call is pending.
After completion, inspect the JSON result before choosing the next action.

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
not another GPU execution. A successful read exits zero even when the saved
operation failed; inspect `status` and `result.correct`.
An identical call that is still executing returns `gateway_task_running` immediately
after local input identity checks. Wait for the original call; do not submit it again
or try `record-read` before a completed Record ID exists. `remaining_seconds` counts
down the original Runtime request deadline, not a guaranteed Agate completion time;
the original call may return a timeout after cleanup. A null value means no estimate.

`result.error.code: "candidate_source_rejected"` is a source
repair, not an infrastructure blocker. Fix the listed forbidden imports,
attribute accesses or string literals in `kernel.py`, then rerun the operation.
No GPU job was submitted by that rejected request. Do not switch to Dev to bypass
the source validator.

`result.error` preserves Agate's original error fields, including its
`error_class`, `reason`, `message` and `details`, with credential/path redaction
and output limits. Check the actual error before deciding what to do: compiler,
import and missing-file errors require repairing your code or declared inputs,
not waiting for the Gateway. Custom Dev probes retain `result.stdout`/`result.stderr`
even on failure. Profile, Check, Disassemble and ABBA may also include errors in
`result`. These diagnostics are not proof that another submission is safe;
an unknown outcome still requires reconciliation. Read the saved Record rather
than resubmitting an identical failed task.

For a CUDA extension on Torch `2.9.0+cu130` / CUDA `13.0`, a compiler error in
`ATen/core/List_inl.h:201` about a missing `typename` can be a C++17 toolchain
compatibility issue, even though that header already contains `typename`.
The same minimal extension was verified to compile with C++20 on L20N/sm_120.
If this exact signature and environment match, try `-std=c++20` in both
`extra_cflags` and `extra_cuda_cflags` in your extension build, then rerun Check
or Evaluate. Do not patch installed Torch headers, change the GPU environment,
or classify this as a Gateway outage. This is a conditional workaround, not a
requirement to change the language standard for every Kernel.

Evaluate's `diagnostics` identify the failing stage, opaque Shape, case and output,
preserving the upstream error and available numerical differences. Inspect these
before changing the Kernel. `performance_by_shape` carries timing statistics and
Kernel events when supplied by Agate; neither field reveals evaluator inputs.
`feedback_truncation` signals omitted detail. ABBA attaches diagnostics to the
affected baseline or candidate side.

The Supervisor retries explicitly classified infrastructure failures with backoff.
Lost-log infrastructure failures submit a replacement job; polling outages keep
the already accepted Job ID. Known device-unavailable results are retried with a
bounded budget and do not become negative correctness evidence. Cancellation,
unknown failures and Candidate errors do not by themselves authorize resubmission.
If recovery reaches its deadline/budget, follow `next_action` and report the blocker;
do not turn infrastructure failure into a Kernel correctness conclusion.

Missing/revoked capability or transport failure is an infrastructure blocker.
Do not bypass the Runtime or blindly retry an operation whose outcome is unknown.
Gateway retry policy is owned by the Supervisor; do not add automatic client-side retry loops.
If the Runtime explicitly returns `repairable: true` with `error.code: "request_not_started"`,
this request submitted no job: wait at least `retry_after_seconds`, then retry unchanged
with backoff. This permission does not apply to transport failures or unknown outcomes.
