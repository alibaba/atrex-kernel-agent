# Read saved measurement evidence

All commands run through `python3 tools/sandbox.py` in the Session workspace.

| ID | Meaning | Use |
| --- | --- | --- |
| `gateway-…` | One saved measurement operation | Experiment evidence, result reader |
| `kernel-…` | Exact Kernel source; equal bytes have equal IDs | Source reader, measurement index |
| `direction_…` | Research hypothesis and lifecycle | Direction list/load/update |
| `experiment_…` | Interpretation tied to real measurement IDs | Experiment list/load, closure, report |

Agate Job IDs cannot replace these IDs.

```bash
python3 tools/sandbox.py --kind record-read --record-id gateway-11111111111111111111111111111111
python3 tools/sandbox.py --kind kernel-records --kernel-id kernel-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
python3 tools/sandbox.py --kind kernel-read --kernel-id kernel-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa --output-path scratch/previous.py
```

`record-read` returns exactly the stored Agent-visible stdout, stderr and exit code, not a new
wrapper or raw Gateway envelope. A failed measurement retains its failure exit code. Locate the
operation's marker line instead of parsing the entire stdout as JSON:

| Operation | Result marker |
| --- | --- |
| Evaluate | `[test_kernel] RESULT_JSON=` |
| ABBA | `[sandbox] ABBA_JSON=` |
| Profile | `[sandbox] PROFILE_JSON=` |
| Check | `[sandbox] CHECK_JSON=` |
| Disassemble | `[sandbox] DISASSEMBLE_JSON=` |
| Arbitrary Dev | Probe stdout plus `[sandbox] RECORD_JSON=` |

Markers carry `gateway_record_id` and, when Kernel-bound, `kernel_id`. ABBA additionally has
`baseline_kernel_id`. Dev's Kernel is the workspace snapshot, not proof the probe measured it.
`kernel-records` lists Record ID, operation and timestamp, not result bodies. `kernel-read`
writes source to the requested scratch file and acknowledges it. Read that file before restoring it.

Use returned IDs, not these illustrative examples. Shape IDs are opaque. Profile duration and
Evaluate latency are different measurements; a diagnostic cannot establish untested performance.
When adopting an older Kernel, record a current-Episode Experiment citing its saved measurement.

## Errors

Journal validation errors return `ok:false`, `repairable:true`, and `error` with
`code`, `message`, `next_action`; field errors also list missing/unexpected/allowed fields.
Fix the indicated request or lifecycle state and call again. This is not permission to replay an
unknown-outcome mutation. Missing/corrupt private storage and post-write publication failures are
infrastructure errors, not input mistakes.

- Duplicate measurement: read `error.gateway_record_id`; do not submit another GPU task.
- Direction conflict/limit: finish the active hypothesis before starting another; propose ideas
  for later Episodes without starting a fourth Direction.
- Report rejected: verify exact source, selected Experiment and existing candidate Commit, close
  active Directions, then resubmit. No successful finalization occurred.
- Report publication failed: after storage recovery, resubmit the identical report; the durable
  report is reused. Do not rerun GPU work or create another Commit.
- Unknown transport outcome: list/load records or ask the operator before resubmitting a write.
