# Read saved measurement evidence

Runtime projects every visible result as `records/<gateway_record_id>.json`. Read and search those
flat files normally. Exact Kernel source is ordinary reachable Git history.

Runtime measurement calls print a compact receipt rather than the complete result. Env is the
exception: it prints its redacted Agate response directly and creates no Record. Query a full saved
measurement Record locally without another GPU request. Use ordinary read/search tools or a short
local JSON query; for example:

```bash
RECORD="records/${GATEWAY_RECORD_ID}.json"
python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["result"]["correct"])' "$RECORD"
python3 -c 'import json,sys; print(sorted(json.load(open(sys.argv[1]))["result"]))' "$RECORD"
```

Read large text with bounded line or byte ranges. Do not use `cat`, a full `json.dumps`, or an
equivalent command that copies a complete Record, per-Shape table, source, stdout, or stderr into
the Session. A receipt with stream output includes `stdout_location` and/or `stderr_location`, whose
`path` names this Record and whose `field` is `result.stdout` or `result.stderr`.

| ID | Meaning | Use |
| --- | --- | --- |
| `gateway-…` | One saved measurement operation | Measured evidence, result reader, direct report citation when the Experiment module is disabled |
| full Git commit ID | Exact committed Kernel source; exposed as `kernel_commit_id` | Source reader, measurement index, peer Git history |
| `direction_…` | Optimization objective and lifecycle | `PROGRESS.md`, `directions/<status>/*.json`, Direction updates |
| `experiment_…` | Concrete test and interpretation tied to measurement IDs | `experiments/<experiment_id>/experiment.json`, closure and report |

Agate Job IDs cannot replace these IDs.

Each projected file contains the same public result returned by a new measurement:
`operation`, `status`, `result`, `gateway_record_id`, and `kernel_commit_id` when Kernel-bound.
ABBA uses `operation: evaluate` with `result.comparison`, `candidate_kernel_commit_id`, and
`baseline_kernel_commit_id`.
Other operation names are `profile`, `dev`, `check`, and `disassemble`.
Parse a measurement command's whole stdout as one compact receipt. Env stdout is the direct
redacted Agate capability response and is not a receipt.
Dev preserves probe text in `result.stdout`/`result.stderr` and its process `exit_code`.
Its Kernel is the workspace snapshot, not proof the probe measured it.

Inspect `status` (`completed`, `failed`, `cancelled`) and, for Evaluate, `result.correct`.
An evaluator can complete with a rejected Kernel. Private request/options/input hashes are never
included. Search the flat files for a commit ID to enumerate its measurements. Use Git to inspect,
branch from, restore or compare the corresponding source.

Use returned IDs, not these illustrative examples. Shape parameters are public through
`shapes.json` only in exposed mode; in hidden mode every Shape ID is opaque. Profile duration and
Evaluate latency are different measurements; a diagnostic cannot establish untested performance.
When checking out an older Kernel, reuse a visible Experiment whose measured decision has a `supported` Review and whose saved full Evaluate
still matches that exact commit, or record a new Experiment when the interpretation or evidence has
changed. Source-policy review is requested separately with `submit-kernel`.

## Per-Shape accepted-submission Pareto

```bash
# Read root PROGRESS.md for the Research Board and per-Shape frontier.
```

The Pareto section of Runtime-managed root `PROGRESS.md` takes no IDs and submits no GPU work.
It includes only commits accepted by `submit-kernel` that also have eligible full Evaluate evidence.
`best_by_shape` maps each Shape ID to
`kernel_commit_id`, `gateway_record_id`, and `latency_us`;
`winner_kernel_commit_ids` lists the distinct winning sources; `shape_count` counts returned Shapes;
`evaluation_count` counts eligible full Evaluations, not distinct Kernels. With no eligible history,
the map and list are empty and both counts are zero. Open the cited Record file and Git commit directly.

Only completed, passing standard full Evaluations for accepted submissions in this Campaign count,
across Episodes and imported peer lineages. An Evaluate without an accepted submission remains in
`records/` but does not enter Pareto. Custom-input, Shape-subset, correctness-only, ABBA, Profile and
Dev results do not count. Failed, incomplete and invalid records are excluded.
The projection remains available with research recording disabled. Equal latencies break ties by
Kernel commit ID, then Record ID. These are observed minima, not a statistically confirmed bound or a
single Kernel that achieves every minimum; combining implementations requires a new full Evaluate.

## Errors

Research/report validation errors return `ok:false`, `repairable:true`, and `error` with
`code`, `message`, `next_action`; field errors also list missing/unexpected/allowed fields.
Fix the indicated request or lifecycle state and call again. This is not permission to replay an
unknown-outcome mutation. Missing/corrupt private storage and post-write publication failures are
infrastructure errors, not input mistakes.

- Duplicate measurement: read `error.gateway_record_id`; do not submit another GPU task.
- Kernel submission needs evaluation: inspect `evaluation_evidence`; run a standard full Evaluate
  for the exact commit, then submit it again. A failed saved Evaluate must be repaired in a new
  commit and remeasured.
- Kernel source review rejected: inspect `diagnostics`, repair the final-source policy violation in
  a new commit, run its full Evaluate, and submit that exact commit. Rejection never changes Git or
  measurement records.
- Unknown transport outcome: inspect `PROGRESS.md` and the projected research files, or ask the
  operator before resubmitting a write.
