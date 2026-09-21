# Journal and worked Episode example

Use Runtime Journal in a controller-managed Episode; do not mix legacy local append/finalize commands with these tools. Plans and conclusions belong in Directions and Experiments, not mandatory separate files. Every report must satisfy the lifecycle and evidence checks; `blocked` also requires a nonempty blocker. Framework Baseline follows its own finish procedure.

This example links one exploration to its measured Kernel and terminal report. It is not a mandatory
optimization strategy or permission to run operations forbidden by the phase. Create request files
under `scratch/`; replace the illustrative IDs below with the values returned in your session.

Request files may contain at most 2,093,056 bytes (2 MiB minus 4 KiB for the envelope).
The complete encoded HTTP body must also fit within 2 MiB. If either limit is exceeded,
shorten text or lists and retry; the client sends no request on a size rejection.

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

Response is `{"status":"recorded","direction_id":"..."}`. `start` accepts exactly these three
fields; closure actions also require `hypothesis_status` and `supporting_experiment_ids` below.
At most one Direction can be `in_progress`; at most three distinct Directions can be
started in an Episode, including inherited ones. Restarting an already-started Direction in
the same Episode does not consume another slot.

| Action | Allowed prior status | New status | Extra condition |
| --- | --- | --- | --- |
| `start` | `proposed` or closed | `in_progress` | No other active Direction; advancement limit |
| `complete` | `in_progress` or closed | `completed` | Explicit evidence selection and hypothesis assessment |
| `abandon` | `in_progress` or closed | `abandoned` | Explicit evidence selection and hypothesis assessment |
| `block` | `in_progress` or closed | `blocked` | Evidence-backed blocker; assessment can remain unresolved |
| `defer` | `in_progress` or closed | `deferred` | Evidence-backed pause; assessment can remain unresolved |

Closed means completed, abandoned, blocked, or deferred. Restarting resets the current assessment
to unresolved and clears selected support; prior events remain immutable. Reuse the ID only for the
same hypothesis; propose a derived Direction when the hypothesis changes.

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
  "gateway_record_ids": ["gateway-11111111111111111111111111111111"],
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

All shown fields are required; textual fields must be non-empty. Existing optional `wiki_usage`, `wiki_query_ids`, and `wiki_usage_status` attribution fields remain supported with the legacy schema; malformed attribution is diagnostic, not an acceptance gate. The Direction must be in progress or closed;
not merely proposed. Late records can attach already obtained evidence after closure, even while
another Direction is active. This does not restart exploration or change previous closure support.
`gateway_record_ids` is a non-empty, duplicate-free list of visible Kernel-bound Gateway IDs: Evaluate, ABBA, Profile, Dev,
Check, and Disassemble are all valid evidence. The record must belong to this Campaign; corrupted private records are infrastructure blockers. It may span different Kernels; order does not mean
before/after. Do not submit Kernel IDs, Job IDs, digests, or result payloads instead of these IDs.
When asserting improvement over another measurement, cite that result too and explain the comparison.

| Experiment `action` | Meaning |
| --- | --- |
| `baseline` | First measured anchor; at most once per Episode, not a request to register V0 |
| `keep_after` | Keep the tested implementation |
| `restore_before` | Restore an earlier implementation |
| `adopt` | Reuse an existing implementation |
| `abandon_direction` | Document an investigation or blocker using actual Gateway records; no performance claim is required |

No action permits an empty record list. Failed diagnostics may document an unresolved blocker;
Env/Health/Wiki responses, unbound Dev outputs and transport errors without a saved Gateway Record
are not Experiment evidence. An Experiment action records your decision;
it does not copy source or update Direction lifecycle. Restore source yourself through the Kernel
reader when needed, and call `update-direction` separately to close the Direction.

## Close the Direction

Save as `scratch/close.json`:

```json
{
  "action": "complete",
  "direction_id": "direction_11111111111111111111111111111111",
  "analysis": "The measured candidate is ready for verification",
  "hypothesis_status": "supported",
  "supporting_experiment_ids": ["experiment_22222222222222222222222222222222"]
}
```

```bash
python3 tools/sandbox.py --kind update-direction --request-file scratch/close.json
```

Response is `{"status":"recorded","direction_id":"..."}`. Use `abandon`, `block`, or `defer`
instead when that describes the outcome. All four closure actions require exactly these five fields.
Select 1–32 unique Experiment IDs from this Direction's visible history. `hypothesis_status` means:

- `unresolved`: evidence is insufficient to settle the hypothesis; use this for untested claims or infrastructure failures.
- `supported`: the cited evidence supports your hypothesis.
- `refuted`: the cited evidence contradicts your hypothesis.

Lifecycle does not imply a verdict: completing work is not proving a hypothesis, and abandoning it
is not refuting it. For supported/refuted, every selected Experiment must cite at least one completed
Gateway observation. A completed correctness failure can refute a correctness claim; a failed job
cannot settle a performance claim. Runtime verifies records and Kernel bindings, not the scientific
truth or relevance of your interpretation. Do not use unrelated measurements to justify a claim.

`associated_experiment_ids` includes all linked Experiments; `supporting_experiment_ids` is only the
latest explicit closure selection. Late records extend associations without rewriting that selection.
To revise a conclusion, explicitly submit another closure with its assessment and selected evidence.
No in-progress Direction may remain at terminal handoff;
proposed/deferred Directions may remain for later exploration.

## Candidate report

Leave the selected measurement's exact source in `kernel.py`. The Supervisor reuses matching six-case correctness evidence, or performs the missing check, before sealing that source into a candidate Commit. Do not run Git or supply a Commit ID. Save as `scratch/episode-report.json`:

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
or ABBA alone cannot satisfy this report binding; custom-input or correctness-only diagnostics are
not substitutes for standard performance evidence. Before accepting `candidate_ready`, the Supervisor
reuses a matching successful full or correctness-only Evaluate with the base case plus five additional
seeds. Ordinary Atrex-Bench Evaluate already checks these six cases. If evidence is missing or does
not match the complete contract, the Supervisor performs a correctness-only check without timing.
You need not submit or cite an extra check yourself. `candidate_correctness_failed` returns a Record
ID and repair instructions: fix the Kernel, update its full Evaluate/Experiment evidence and resubmit.
No report is finalized or Supervisor candidate commit created on failure. Infrastructure/unknown
outcomes are blockers, not evidence that the Kernel is incorrect. Pivot/blocked reports do not trigger
this check; replaying an accepted report republishes handoff without rerunning it.
Do not submit `candidate_commit` in a managed Episode. Do not write canonical `memory/vN.json`. `accepted` confirms handoff, not that
the candidate has won the Supervisor's independent gates. Stop after successful handoff.

## Pivot report

If exploration found no candidate to advance, use:

```json
{"status":"pivot","summary":"The explored change did not improve the incumbent; try a different direction"}
```

Omit `selected_experiment_id`, `candidate_commit`, and `blocker`. Journals may be empty if no Direction needs closing; do not invent experiments to satisfy a count.
Save to the same report path and call `episode-report` as above; the response is
`{"status":"accepted","message":"Report accepted and recorded"}`.

## Blocked report

If infrastructure or missing authority prevents progress:

```json
{"status":"blocked","summary":"Cannot obtain the required measurement","blocker":"The Runtime reports that the GPU service is unavailable"}
```

Both text fields must be non-empty; omit `selected_experiment_id` and `candidate_commit`. A blocked report can be submitted
without an Experiment only if no Direction needs closing. Otherwise record actual Kernel-bound
diagnostic evidence, then block/defer with an unresolved assessment. If no Gateway Record exists,
closure remains blocked and normal session recovery handles the failure; never fabricate evidence.
Use the same report command and acknowledgement as for pivot. For all statuses, malformed reports
can be corrected and resubmitted; do not mistake a rejected report for a completed Episode.

For a report validation error (`repairable:true`), use `error.message` and `error.next_action`
to fix the indicated fields or prerequisites, then call `episode-report` again in this Session.
Field errors also identify missing/unexpected fields and allowed fields. This does not apply to
`journal_protocol_conflict`: finish that Episode with the legacy commands in its prompt instead.

The other optional report field is `accepted_ppu_diagnostics`; use its schema in the mounted
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
`direction_id`, `name`, `status`, `hypothesis_status`, and any declared ancestry. An Experiment index entry has
`experiment_id`, `name`, `hypothesis`, `change`, `gateway_record_ids`, `evidence`, `analysis`, `action`.
These exports do not update automatically; invoke list again when you need a fresh snapshot.

```bash
python3 tools/sandbox.py --kind load-direction --record-id direction_11111111111111111111111111111111
python3 tools/sandbox.py --kind load-experiment --record-id experiment_22222222222222222222222222222222
```

Load returns the record directly as JSON, not a file or `result` wrapper. A loaded Direction includes
the proposal, current `status`/`analysis`/`hypothesis_status`, timestamps, declared ancestry,
all `associated_experiment_ids`, and explicitly selected `supporting_experiment_ids`.
Old closures without an explicit assessment read as unresolved; their old automatic support is
association only. Old unmeasured notes remain readable but cannot support new closures.
A loaded Experiment contains its submitted fields plus `experiment_id`
and `recorded_at`; no `sequence`. Use its `gateway_record_ids` with `record-read` for measurements
and exact Kernel identity. Subsequent reads of a Direction include newly associated Experiments.

## Derived Directions

Reuse the same ID for an unchanged hypothesis; use `start` to resume a closed Direction. A new
implementation of that hypothesis is an Experiment, not a new ancestry kind. For a changed
hypothesis, a proposal may add one of these `relationship` values:

- `refinement`: narrow or extend a parent hypothesis, such as focusing a tiling idea on register pressure.
- `correction`: revise a mistaken parent hypothesis while preserving the original record.
- `combination`: combine ideas from two or more distinct parent Directions, such as fusion and layout changes.

Add `derived_from_direction_ids` and/or `derived_from_experiment_ids` from visible history. Each list
allows at most 32 unique IDs. Explain the derivation in `rationale`; these labels do not establish
that a hypothesis is correct. Cross-Campaign/DSL porting is not supported by this Journal.

A relationship needs a parent; `combination` needs two distinct parent Directions, directly or via
Experiments. A `correction` may set `supersedes_direction_id` to one parent; this does not change that
parent's lifecycle. Ancestry is immutable; lifecycle updates must not include these extra fields.
