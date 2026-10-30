# Read saved measurement evidence

All commands run through `python3 tools/sandbox.py` in the Session workspace.

| ID | Meaning | Use |
| --- | --- | --- |
| `gateway-…` | One saved measurement operation | Measured evidence, result reader, direct report citation when research recording is disabled |
| `kernel-…` | Exact Kernel source; equal bytes have equal IDs | Source reader, measurement index |
| `direction_…` | Research hypothesis and lifecycle | Direction list/load/update |
| `experiment_…` | Interpretation tied to real measurement IDs | Experiment list/load, closure, report |

Agate Job IDs cannot replace these IDs.

```bash
python3 tools/sandbox.py --kind record-read --record-id gateway-11111111111111111111111111111111
python3 tools/sandbox.py --kind kernel-records --kernel-id kernel-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
python3 tools/sandbox.py --kind kernel-read --kernel-id kernel-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa --output-path scratch/previous.py
```

`record-read` prints one JSON object, the same public result returned by a new measurement:
`operation`, `status`, `result`, `gateway_record_id`, and `kernel_id` when Kernel-bound.
ABBA uses `operation: evaluate` with `result.comparison` and `baseline_kernel_id`.
Other operation names are `profile`, `dev`, `check`, and `disassemble`.
Parse the whole stdout as JSON; do not look for legacy executor markers.
Dev preserves probe text in `result.stdout`/`result.stderr` and its process `exit_code`.
Its Kernel is the workspace snapshot, not proof the probe measured it.

The read exits zero when evidence was retrieved, regardless of the stored outcome.
Inspect `status` (`completed`, `failed`, `cancelled`) and, for Evaluate, `result.correct`.
An evaluator can complete with a rejected Kernel. Missing or corrupt records still fail the query.
Old stream-only records are converted in memory, without changing their stored evidence or
submitting GPU work. Private request/options/input hashes are never included.

`kernel-records` lists Record ID, operation and timestamp, not result bodies. `kernel-read`
writes source to the requested scratch file and acknowledges it. Read that file before restoring it.

Use returned IDs, not these illustrative examples. Shape IDs are opaque. Profile duration and
Evaluate latency are different measurements; a diagnostic cannot establish untested performance.
When adopting an older Kernel, cite its saved measurement in a current-Episode Experiment if
research recording is enabled; otherwise cite it directly in `episode-report`.

## Per-Shape historical best

```bash
python3 tools/sandbox.py --kind kernel-pareto-frontier
```

This read-only query takes no IDs and submits no GPU work. It returns one JSON object:
`best_by_shape` maps each opaque Shape ID to `kernel_id`, `gateway_record_id`, and `latency_us`;
`winner_kernel_ids` lists the distinct winning sources; `shape_count` counts returned Shapes;
`evaluation_count` counts eligible full Evaluations, not distinct Kernels. With no eligible history,
the map and list are empty and both counts are zero. Use `kernel-read` and `record-read` on returned IDs.

Only completed, passing standard full Evaluations in this Campaign count, across Episodes,
including saved baseline/controller full Evaluations. Custom-input, Shape-subset, correctness-only,
ABBA, Profile and Dev results do not count. Failed, incomplete and invalid records are excluded.
The query remains available with research recording disabled. Equal latencies break ties by
Kernel ID, then Record ID. These are observed minima, not a statistically confirmed bound or a
single Kernel that achieves every minimum; combining implementations requires a new full Evaluate.

## Errors

Journal validation errors return `ok:false`, `repairable:true`, and `error` with
`code`, `message`, `next_action`; field errors also list missing/unexpected/allowed fields.
Fix the indicated request or lifecycle state and call again. This is not permission to replay an
unknown-outcome mutation. Missing/corrupt private storage and post-write publication failures are
infrastructure errors, not input mistakes.

- Duplicate measurement: read `error.gateway_record_id`; do not submit another GPU task.
- Direction conflict/limit (research recording enabled): finish the active hypothesis before starting another; propose ideas
  for later Episodes without starting a fourth Direction.
- Report rejected: verify exact source and measured evidence. With research recording enabled,
  check the selected Experiment and close active Directions; otherwise check `gateway_record_ids`.
  Correct the report and resubmit. No successful finalization occurred; the Supervisor owns Commit creation.
- Report publication failed: after storage recovery, resubmit the identical report; the durable
  report is reused. Do not rerun GPU work or create another Commit.
- Unknown transport outcome: list/load records or ask the operator before resubmitting a write.
