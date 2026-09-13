# Journal and worked Episode example

This example links one exploration to its measured Kernel and terminal report. It is not a mandatory
optimization strategy or permission to run operations forbidden by the phase. Create request files
under `scratch/`; replace the illustrative IDs below with the values returned in your session.

## Propose a Direction

Save as `scratch/direction.json`:

```json
{
  "action": "propose",
  "name": "Fuse the elementwise stages",
  "hypothesis": "One GPU launch can reduce intermediate memory traffic",
  "rationale": "The current source materializes an intermediate tensor",
  "plan": ["Implement one fused path", "Check correctness and compare with the incumbent"],
  "success_criteria": ["All cases pass", "Performance improves under the required evaluator"],
  "stop_conditions": ["No gain after a bounded implementation repair"]
}
```

```bash
python3 tools/sandbox.py --kind update-direction --request-file scratch/direction.json
```

Response:

```json
{"status":"recorded","direction_id":"direction_11111111111111111111111111111111"}
```

All shown fields are required. `name`, `hypothesis`, and `rationale` are non-empty strings;
`plan`, `success_criteria`, and `stop_conditions` are lists of non-empty strings. A proposal records
the idea but does not start it. Propose any number of useful, nonduplicate directions.

## Start or resume exploration

Save as `scratch/start.json`:

```json
{
  "action": "start",
  "direction_id": "direction_11111111111111111111111111111111",
  "analysis": "Begin implementing the fused path"
}
```

```bash
python3 tools/sandbox.py --kind update-direction --request-file scratch/start.json
```

Response is `{"status":"recorded","direction_id":"..."}`. Lifecycle updates accept exactly these
three fields. At most one Direction can be `in_progress`; at most three distinct Directions can be
started in an Episode, including inherited ones. Resuming an already-started deferred Direction in
the same Episode does not consume another slot.

| Action | Allowed prior status | New status | Extra condition |
| --- | --- | --- | --- |
| `start` | `proposed`, `deferred` | `in_progress` | No other active Direction; advancement limit |
| `complete` | `in_progress` | `completed` | A supporting Experiment in this Episode |
| `abandon` | `in_progress` | `abandoned` | A supporting Experiment in this Episode |
| `block` | `in_progress` | `blocked` | Explain the blocker in `analysis` |
| `defer` | `in_progress` | `deferred` | Explain why exploration is paused |

Completed, abandoned, and blocked Directions cannot be restarted; propose a derived hypothesis if
new evidence warrants revisiting one. Only deferred Directions can resume.

## Measure and record an Experiment

Edit `kernel.py`, then run the evaluator prescribed by the current phase. For an ordinary full
evaluation where no additional options are required:

```bash
python3 tools/sandbox.py --kind run --mode full --no-sync
```

The response contains `gateway_record_id` and `kernel_id`; see [result formats](records.md).
Use the Gateway ID in `scratch/experiment.json`:

```json
{
  "direction_id": "direction_11111111111111111111111111111111",
  "name": "Fused candidate measurement",
  "hypothesis": "Removing the intermediate reduces traffic",
  "change": "Replaced the two stages with one fused implementation",
  "gateway_record_ids": ["gateway-100-111111111111"],
  "evidence": "Full evaluation passed; per-Shape latency is recorded in the cited result",
  "analysis": "Keep this candidate for independent verification; the result supports the hypothesis",
  "action": "keep_after"
}
```

```bash
python3 tools/sandbox.py --kind record-experiment --request-file scratch/experiment.json
```

Response:

```json
{"status":"recorded","experiment_id":"experiment_22222222222222222222222222222222"}
```

All fields are required; textual fields must be non-empty. The Direction must be `in_progress`.
`gateway_record_ids` is a duplicate-free list of visible Gateway IDs: Evaluate, ABBA, Profile, Dev,
Check, and Disassemble are all valid evidence. It may span different Kernels; order does not mean
before/after. Do not submit Kernel IDs, Job IDs, digests, or result payloads instead of these IDs.
When asserting improvement over another measurement, cite that result too and explain the comparison.

| Experiment `action` | Meaning |
| --- | --- |
| `baseline` | First measured anchor; at most once per Episode, not a request to register V0 |
| `keep_after` | Keep the tested implementation |
| `restore_before` | Restore an earlier implementation |
| `adopt` | Reuse an existing implementation |
| `abandon_direction` | Explain a dead end; may have no Gateway records if no GPU operation ran |

Only `abandon_direction` permits an empty record list. An Experiment action records your decision;
it does not copy source or update Direction lifecycle. Restore source yourself through the Kernel
reader when needed, and call `update-direction` separately to close the Direction.

## Close the Direction

Save as `scratch/close.json`:

```json
{
  "action": "complete",
  "direction_id": "direction_11111111111111111111111111111111",
  "analysis": "The measured candidate is ready for verification"
}
```

```bash
python3 tools/sandbox.py --kind update-direction --request-file scratch/close.json
```

Response is `{"status":"recorded","direction_id":"..."}`. Use `abandon`, `block`, or `defer`
instead when that describes the outcome. No in-progress Direction may remain at terminal handoff;
proposed/deferred Directions may remain for later exploration.

## Candidate report

Leave the selected measurement's exact source in `kernel.py`. Save as `scratch/episode-report.json`:

```json
{
  "status": "candidate_ready",
  "summary": "The fused implementation passes evaluation and is ready for independent verification",
  "selected_experiment_id": "experiment_22222222222222222222222222222222"
}
```

```bash
python3 tools/sandbox.py --kind episode-report --request-file scratch/episode-report.json
```

Response:

```json
{"status":"accepted","message":"Report accepted; Supervisor committed the measured Kernel for verification"}
```

The selected Experiment must belong to this Episode, use `baseline`, `keep_after`, or `adopt`, and
cite a passing Evaluate record for byte-identical current `kernel.py`. Profile/Dev/Check/Disassemble
or ABBA alone cannot satisfy this report binding. Fulfil the phase's full validation requirements;
custom-input or correctness-only diagnostics are not substitutes for standard performance evidence.
Do not submit Commit IDs or write canonical `memory/vN.json`. `accepted` confirms handoff, not that
the candidate has won the Supervisor's independent gates. Stop after successful handoff.

## Pivot report

If exploration found no candidate to advance, use:

```json
{"status":"pivot","summary":"The explored change did not improve the incumbent; try a different direction"}
```

At least one current-Episode Experiment is required. Omit `selected_experiment_id` and `blocker`.
Save to the same report path and call `episode-report` as above; the response is
`{"status":"accepted","message":"Report accepted and recorded"}`.

## Blocked report

If infrastructure or missing authority prevents progress:

```json
{"status":"blocked","summary":"Cannot obtain the required measurement","blocker":"The Runtime reports that the GPU service is unavailable"}
```

Both text fields must be non-empty; omit `selected_experiment_id`. A blocked report can be submitted
without an Experiment if no experiment was possible; first block/defer any in-progress Direction.
Use the same report command and acknowledgement as for pivot. For all statuses, malformed reports
can be corrected and resubmitted; do not mistake a rejected report for a completed Episode.

The only other optional report field is `accepted_ppu_diagnostics`; use its schema in the mounted
`ppu-acu-joint-profile` Skill when applicable. Do not invent additional fields.

## List and load history

```bash
python3 tools/sandbox.py --kind list-directions --output-path scratch/directions.json
python3 tools/sandbox.py --kind list-experiments --output-path scratch/experiments.json
```

Each command returns an acknowledgement, not the index itself:

```json
{"status":"written","file":"scratch/experiments.json","count":1}
```

Read the named local file. It contains `{"directions":[...]}` or `{"experiments":[...]}` across the
current and visible earlier Episodes, not just this Episode. A Direction index entry has
`direction_id`, `name`, `status`, and any declared ancestry. An Experiment index entry has
`experiment_id`, `name`, `hypothesis`, `change`, `gateway_record_ids`, `evidence`, `analysis`, `action`.
These exports do not update automatically; invoke list again when you need a fresh snapshot.

```bash
python3 tools/sandbox.py --kind load-direction --record-id direction_11111111111111111111111111111111
python3 tools/sandbox.py --kind load-experiment --record-id experiment_22222222222222222222222222222222
```

Load returns the record directly as JSON, not a file or `result` wrapper. A loaded Direction includes
the proposal, current `status`/`analysis`, timestamps, declared ancestry, and automatically linked
`supporting_experiment_ids`. A loaded Experiment contains its submitted fields plus `experiment_id`
and `recorded_at`; no `sequence`. Use its `gateway_record_ids` with `record-read` for measurements
and exact Kernel identity. Subsequent reads of a Direction include newly recorded supporting Experiments.

## Derived Directions

Reuse the same ID for an unchanged deferred hypothesis. A proposal for a revised hypothesis may add
`relationship`: `retry`, `refinement`, `reimplementation`, `correction`, `port`, or `combination`, plus
`derived_from_direction_ids` and/or `derived_from_experiment_ids` from visible history. Each list
allows at most 32 unique IDs. Explain the derivation in `rationale`.

A relationship needs a parent; `combination` needs two distinct parent Directions, directly or via
Experiments. A `correction` may set `supersedes_direction_id` to one parent; this does not change that
parent's lifecycle. Ancestry is immutable; lifecycle updates must not include these extra fields.
