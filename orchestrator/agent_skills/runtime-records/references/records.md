# Saved GPU results and exact Kernel source

## IDs and output formats

- `gateway_record_id` identifies one persisted operation result. Use it for `record-read` and
  Experiment `gateway_record_ids`.
- `kernel_id` identifies stored exact Kernel source. Use it for source recovery or to discover
  the Gateway operations associated with that source. A Kernel can have several Gateway records.
- `direction_id` and `experiment_id` are returned by Journal writes, then used by list/load and reports.
- These IDs are not interchangeable with remote Agate Job IDs. Examples below use illustrative IDs.

Successful Journal calls print plain JSON. GPU/history calls usually print a prefixed JSON line:

| Operation | Prefix before the JSON object |
| --- | --- |
| ordinary `run` | `[test_kernel] RESULT_JSON=` |
| `run --baseline-path ...` | `[sandbox] ABBA_JSON=` |
| `profile` | `[sandbox] PROFILE_JSON=` |
| `check` | `[sandbox] CHECK_JSON=` |
| `disassemble` | `[sandbox] DISASSEMBLY_JSON=` |
| `env` | `[sandbox] ENV_JSON=` |
| `record-read` (all views) | `[sandbox] RECORD_JSON=` |
| custom `dev` | Probe output plus `[sandbox] DEV_RECORD_JSON=` |

If parsing captured output, locate the appropriate prefixed line and decode the text after `=`;
do not parse all stdout as one JSON document. Errors may instead be plain JSON on stdout or stderr;
check both and the exit status. `[test_kernel]` is a result label, not a file to run or recreate.

For example, a successful ordinary Run can return:

```text
[test_kernel] RESULT_JSON={"all_pass":true,"latency_us_geomean":10.0,"latency_us_arith_mean":10.0,"latency_us_by_shape":{"0":10.0},"failures":[],"actionable_diagnostics":[],"gateway_record_id":"gateway-100-111111111111","kernel_id":"kernel-100-aaaaaaaaaaaa"}
```

Fields such as `performance_score`, `mode`, and `input_scope` depend on the operation and evaluator.
Shape IDs are opaque; do not infer hidden inputs. An ordinary measurement's latency is not a Profile
kernel duration. Check or Dev success only establishes what that diagnostic actually tested.

## Read a Gateway result

```bash
python3 tools/sandbox.py --kind record-read --record-id gateway-100-111111111111
```

The JSON after `[sandbox] RECORD_JSON=` is:

```json
{
  "record_type": "gateway_result",
  "gateway_record_id": "gateway-100-111111111111",
  "operation": "evaluate",
  "status": "completed",
  "kernel_id": "kernel-100-aaaaaaaaaaaa",
  "result": {
    "all_pass": true,
    "latency_us_geomean": 10.0,
    "latency_us_arith_mean": 10.0,
    "latency_us_by_shape": {"0": 10.0},
    "failures": [],
    "actionable_diagnostics": []
  }
}
```

Unlike the immediate Run, measurements are under `result`; identity and operation are outside it.
The reader supports `evaluate`, `same_allocation_abba`, `profile`, `dev`, `check`, and `disassemble`
records without another GPU job. A successful read means the record was found, not that its Kernel
passed: inspect `result.all_pass`, diagnostic success fields, or `result.error`, as applicable.
Failed records can be read too. This is a bounded Agent-facing projection, not raw private logs.

For ABBA, top-level `kernels.incumbent.kernel_id` and `kernels.candidate.kernel_id` distinguish the
two sources; top-level `kernel_id` is the candidate. For Dev, the Kernel identity is only the workspace
snapshot when the probe ran, not proof the command evaluated it. Probe output remains in its result.

## Discover a Kernel's Gateway records

```bash
python3 tools/sandbox.py --kind record-read --record-id kernel-100-aaaaaaaaaaaa --view gateway-records
```

JSON after the record prefix:

```json
{
  "record_type": "kernel_gateway_records",
  "kernel_id": "kernel-100-aaaaaaaaaaaa",
  "gateway_records": [
    {"gateway_record_id":"gateway-100-111111111111","operation":"evaluate","status":"completed","role":"subject"}
  ]
}
```

This is an index, not all result contents. Read the relevant Gateway IDs separately. `role` is
`subject` for ordinary operations, `incumbent`/`candidate` for ABBA, or `workspace_snapshot` for Dev.

## Recover exact Kernel source

```bash
python3 tools/sandbox.py --kind record-read --record-id kernel-100-aaaaaaaaaaaa --view source --output-path scratch/previous.py
```

JSON after the record prefix:

```json
{"record_type":"kernel_source","status":"written","kernel_id":"kernel-100-aaaaaaaaaaaa","file":"scratch/previous.py","size_bytes":25}
```

`size_bytes` depends on the stored source. Read the written file; source is not printed inline.
Only a workspace-relative path below `scratch/` is allowed. This copies the recorded source, not
the Agent's current mutable Kernel. Inspect it before copying it into `kernel.py`; do not use Git.
When adopting source in a later Episode, record a new current-Episode Experiment citing the existing
measurement before selecting that Experiment for handoff.

## Error recovery

Validation failures are compact objects, for example:

```json
{
  "ok": false,
  "repairable": true,
  "error": {
    "code": "invalid_fields",
    "message": "Episode Report fields do not match the request format.",
    "missing_fields": ["summary"],
    "unexpected_fields": ["decision"],
    "allowed_fields": ["accepted_ppu_diagnostics", "blocker", "selected_experiment_id", "status", "summary"],
    "next_action": "Add missing_fields and remove unexpected_fields in the request JSON, then resubmit. Read the session instructions for field types and meanings."
  }
}
```

Consult the Journal reference for the field definitions, fix the request, and call again.
`repairable: true` means the input/state can be corrected, not that an unchanged replay is safe.

- `duplicate_gateway_task`: use `error.gateway_record_id` with `record-read`; do not submit a new
  measurement. If no prior ID is available, the first request is still running: wait for its result.
- Direction conflict/limit: load the indicated Direction, follow the lifecycle table, and close or
  defer the active exploration before switching. You can still propose future ideas after reaching
  the three-Direction advancement limit.
- Candidate compile/correctness failure: read `failures` and `actionable_diagnostics`, repair source,
  then measure the changed candidate. Failure evidence may be cited in Experiments.
- Infrastructure failure: preserve `error_class`, `reason`, and `message`; the Supervisor owns retries.
  If a terminal blocker remains, report it rather than modifying credentials/services or installing dependencies.
- Unknown outcome (`runtime_failure`, transport interruption): a write or GPU job may already have
  taken effect. Inspect available Journal/Kernel records before resubmission; do not blindly replay.
  Use an `error_id` if provided to identify the failure to the operator.

Never treat missing results as a measured regression. A repaired `episode-report` may be submitted
again after a validation rejection; after an accepted handoff, stop instead of appending more work.
