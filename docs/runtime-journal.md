# Runtime Journal and Episode Reports

The Light Journal stores terminal Episode reports and optional Agent-authored research
annotations. Measurement records are independent Supervisor-owned facts. Research prose
references those facts; it does not become authoritative correctness or performance evidence.

## Scope and compatibility

The Workflow registers each Episode with a trusted workspace, number, base commit, and branch.
Only authorized optimizer Sessions can write its Journal. Baseline and auxiliary reviewers
have their own interfaces. Agents cannot select a private Journal path or another campaign.

Direction and Experiment recording are enabled by default and can be selected independently.
Reports and automatic measurement records remain enabled in every mode. Configure the enabled
modules with:

```json
{"api_version":1,"patches":[{"id":"light_journal","config":{"tool_modules":["directions","experiments"]}}]}
```

Pass the patch before `--` in `aka optimize --repo-root ... --patch ... -- ...`.
Use only `directions`, only `experiments`, both, or an empty list for report-only mode. The setting
is frozen per Episode. Older Journals without the setting retain enabled behavior; resume them with
their original configuration. Legacy local append/finalize APIs belong to
the Legacy workflow and must not be mixed with Light reports.

## Report-only workflow

After editing the permitted candidate files and obtaining a standard full evaluation, save
a report under scratch. Replace the illustrative ID with a real measurement from this campaign:

```json
{
  "status": "candidate_ready",
  "handoff_reason": "direction_terminal",
  "summary": "Full evaluation passed for the selected candidate.",
  "gateway_record_ids": ["gateway-11111111111111111111111111111111"]
}
```

```bash
python3 tools/sandbox.py --kind episode-report --request-file scratch/report.json
```

At least one citation must be a passing standard full Evaluate for the exact current
candidate. Custom-input, shape-subset, correctness-only, Profile, or ABBA alone cannot
satisfy that full-evaluation report binding. Supplemental acceptance checks and native
ABBA promotion remain separate.

Every report requires `handoff_reason`: `direction_terminal`, `cross_trajectory_share`, or
`blocked`. The second is available only in pooled search, and `blocked` must pair with a blocked
status and meaningful blocker. Invalid fields or evidence return repair guidance and leave the
Episode open. A successful report response means the report was accepted, not that the kernel was
promoted.

## Direction and Experiment recording

When their respective modules are enabled, Directions describe durable optimization objectives,
applicability, strategy plans, lifecycle, and ancestry; Experiments describe concrete test intent
and interpretations that may cite Gateway records or supporting files.
`gateway_record_ids` and `references` are independent optional arrays; either may be omitted or
empty. Runtime validates cited records and privately archives non-empty references, rewriting them
to `experiments/<experiment_id>/references/...` paths.
A candidate report selects an Experiment with `selected_experiment_id` when the Experiment module
is enabled, or cites `gateway_record_ids` directly when it is disabled. An ordinary terminal handoff
may preserve a local `in_progress` Direction for a later Episode; a blocked handoff requires every
local active Direction to be closed with `block`.

| Command | Purpose |
| --- | --- |
| `update-direction` | Propose, start, revise, complete, abandon, block, or defer an optimization program |
| `record-experiment` | Attach analysis and actual measurement IDs |
| `episode-report` | Submit the terminal outcome |

Write operations use `--request-file` and run through
`python3 tools/sandbox.py --kind ...`. Runtime continuously projects visible research into root
`PROGRESS.md`, `directions/<status>/*.json`, `experiments/*/`, and flat
`records/<gateway_record_id>.json`; inspect those files with ordinary file tools. Disabled research
writes return an unavailable contract, while measurement records remain projected independently.

An Episode may start any number of Directions and may keep multiple local Directions active when
the work genuinely overlaps. With multiple active Directions, mutations and Experiments identify
their Direction explicitly. Closing a Direction requires an explicit assessment; supporting Experiment
IDs are required only where the enabled module contract and claimed conclusion require them.
Ancestry and Wiki attribution are validated claims, not proofs of benefit.
See the [packaged Journal reference](../src/aka/light/workspace/resources/skills/runtime-records/references/journal.md)
for the enabled research schema and
[record queries](../src/aka/light/workspace/resources/skills/runtime-records/references/records.md).
Edit only source files permitted by the selected layout. New native tasks use the unified editable
`kernel.py + impl/` source tree; `kernel.py` is the evaluator entry point rather than a fixed adapter.

## Lifecycle and failure paths

`aka.light.runtime.server` authenticates and dispatches.
`aka.light.workflow.journal_service` coordinates reporting, validation, and publication;
`aka.light.workflow.journal` validates and persists the private Journal.
CandidateWorkspace performs sealing, while Acceptance and Workflow own verification and promotion.

The report transaction validates citations, runs required candidate checks, seals exact source,
persists the report, then publishes controller compatibility state and handoff.
Later draft edits cannot change the seal. Identical accepted reports are idempotent and can
repair interrupted publication; conflicting terminal mutations are rejected.

Mandatory native correctness requires five cases and full coverage under matching source,
input, evaluator, tolerance, and environment identity. Compatible full or correctness-only
records may satisfy that correctness check. SOL uses its complete trusted workload contract,
without native-only seed controls; Light ABBA still requires native Agate Atrex-Bench.

| Failure | Response |
| --- | --- |
| Invalid fields, citation, lifecycle, or measured candidate rejection | Repairable feedback; correct the request/source and resubmit |
| Supplemental validation incomplete | Report remains unaccepted; preserve evidence for repair or a blocked outcome |
| Private storage or post-write publication failure | Infrastructure error; reconcile durable state before retrying |
| Unknown transport outcome | Inspect existing records/report; do not blindly repeat mutations |
| Protocol conflict | Continue the original workflow/configuration; changing arguments cannot switch protocols |

The controller verifies its private report and source binding during completion and recovery.
An Agent-writable projection cannot authorize completion or manufacture missing evidence.
Generated memory is optional and does not control report durability.

## Storage, limits and rollback

Private Journals live at
`<Supervisor scope>/journals/episodes/eNNNNNNNN/journal.json`.
The HTTP body is limited to 2 MiB; request files to 2,093,056 bytes, reserving envelope space.
Journals and workspace projections are bounded to 16 MiB, with smaller per-field limits.

Workspace projections include the current Episode and finalized earlier Episodes in the same
campaign, including rejected explorations. They exclude unfinished/future/unrelated Episodes.
The process-local derived index and workspace files are rebuilt on restart; private Journal files
remain authoritative.

Bubblewrap hides the store from the agent. Native mode retains same-UID authority.
Preserve Journals and measurement records when restoring a matching deployment.
For the complete acceptance boundary, see [handoff and promotion](supervisor-promotion.md).
