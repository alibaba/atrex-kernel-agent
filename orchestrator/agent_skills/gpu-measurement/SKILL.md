---
name: gpu-measurement
description: Measure and diagnose Kernels on the remote GPU through the workspace sandbox.py client. Use when selecting an evaluation, comparison, profile, compile check, disassembly, or custom probe and forming its request.
allowed-tools: "Bash Read"
---

# GPU measurement

Run commands from the Agent workspace. `python3 tools/sandbox.py` sends authenticated
HTTP requests to the Supervisor; credentials and endpoints are already configured.
The Supervisor packages inputs, records exact sources/results, retries infrastructure
failures, and returns a compact result. Do not implement those responsibilities yourself.

## Choose the operation

| Need | `--kind` | Meaning |
| --- | --- | --- |
| Correctness and latency | `run` | Standard evaluator over the current `kernel.py` |
| Matched comparison | `run --baseline-path scratch/baseline.py` | Same-allocation ABBA against saved source |
| GPU bottleneck evidence | `profile` | Profiler facts, not end-to-end acceptance latency |
| Import, construction, launch or sanitizer diagnostic | `check` | Does not establish numerical correctness |
| Compiled GPU assembly | `disassemble` | Diagnostic output, not a performance result |
| A custom GPU experiment | `dev` | Execute an explicitly supplied probe |
| Environment capabilities | `env` | Inspect selectable environments; does not change the task target |

Use only operations permitted by the current phase. Making Profile or ABBA available
does not require using it. For parameters not shown here, query
`python3 tools/sandbox.py --kind OPERATION --help` rather than guessing flags.

## Measure and diagnose

```bash
python3 tools/sandbox.py --kind run --mode full --no-sync
python3 tools/sandbox.py --kind profile --profile-level sol --no-sync
python3 tools/sandbox.py --kind check --no-sync
python3 tools/sandbox.py --kind disassemble --format auto --no-sync
python3 tools/sandbox.py --kind env
```

These typed operations need no local evaluation or profiling driver. `kernel.py` is
selected automatically. Keep its measured contents unchanged until the request completes.
Standard full Run and ABBA tasks are deduplicated by exact task identity across this Episode
and visible earlier Episodes of the same Campaign. The Supervisor runs a new task three times
and aggregates per-Shape medians. A duplicate returns `duplicate_gateway_task` with the prior
`gateway_record_id` without launching another job; use `record-read` to retrieve that result.
Shape IDs are opaque labels; they do not disclose the hidden evaluation inputs.

For custom inputs, correctness-only runs, ABBA, focused profiling, dependencies, or Dev
probes, read [specialized requests](references/requests.md) only when needed.

## Failure handling

- Invalid arguments or rejected input: use the error and operation's `--help` to correct
  the request; do not repeat it unchanged.
- Compilation or correctness failure: inspect the returned diagnostic and repair the Kernel.
- Infrastructure retrying: let the Supervisor finish; do not create a competing submission loop.
- Unavailable Runtime or a terminal infrastructure blocker: preserve the diagnostic and
  report the blocker. Never repair services, obtain Agate credentials, or bypass this client.

Keep request/probe files in `scratch/` or reusable scripts in writable `tools/`. Local
source inspection is allowed; GPU execution and JIT-capable imports belong in Gateway jobs.
