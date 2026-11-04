# Runtime Journal and Episode Reports

The Light Journal stores terminal Episode reports and optional Agent-authored research
annotations. Measurement records are independent Supervisor-owned facts. Research prose
references those facts; it does not become authoritative correctness or performance evidence.

## Scope and compatibility

The Workflow registers each Episode with a trusted workspace, number, base commit, and branch.
Only authorized optimizer Sessions can write its Journal. Baseline and auxiliary reviewers
have their own interfaces. Agents cannot select a private Journal path or another campaign.

**Research recording is disabled by default.** Reports and automatic measurement records remain
enabled. To add Direction/Experiment annotations:

```json
{"api_version":1,"patches":[{"id":"workflow","config":{"startup":true,"research_records":true}}]}
```

Pass it with `--patch /path/to/options.json` in the native launch argument list.
The setting is frozen per Episode. Older Journals without the setting retain enabled behavior;
resume them with their original configuration. Legacy local append/finalize APIs belong to
the Legacy workflow and must not be mixed with Light reports.

## Default report-only workflow

After editing the permitted candidate files and obtaining a standard full evaluation, save
a report under scratch. Replace the illustrative ID with a real measurement from this campaign:

```json
{
  "status": "candidate_ready",
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

`pivot` and `blocked` do not require invented research records. A blocked report needs
a meaningful blocker. Invalid fields or evidence return repair guidance and leave the
Episode open. A successful report response means the report was accepted, not that the
kernel was promoted.

## Optional research recording

With `research_records: true`, Directions describe hypotheses, plans, lifecycle, and ancestry;
Experiments describe changes and interpretations tied to Gateway records.
A candidate report selects an Experiment with `selected_experiment_id`.
No Direction may remain in progress at handoff.

| Command | Purpose |
| --- | --- |
| `update-direction` | Propose, start, complete, abandon, block, or defer a hypothesis |
| `record-experiment` | Attach analysis and actual measurement IDs |
| `list-directions` / `list-experiments` | Export a visible-history index |
| `load-direction` / `load-experiment` | Read an existing research record |
| `episode-report` | Submit the terminal outcome |

Write operations use `--request-file`. List operations use `--output-path scratch/...`;
load operations use `--record-id`. All run through `python3 tools/sandbox.py --kind ...`.
Disabled research writes return an unavailable contract; read-only queries can still inspect
earlier recorded Episodes.

At most three distinct Directions can be started in one Episode, and only one can be active
at a time. Closing a Direction requires an explicit assessment and supporting Experiment IDs.
Ancestry and Wiki attribution are validated claims, not proofs of benefit.
See the [packaged Journal reference](../src/aka/resources/skills/runtime-records/references/journal.md)
for the enabled research schema and
[record queries](../src/aka/resources/skills/runtime-records/references/records.md).
Edit only source files permitted by the selected layout; a default source-tree adapter is fixed.

## Lifecycle and failure paths

`aka.agent.tools.host` authenticates and dispatches scoped tool calls.
`aka.workflow.light.host.journal` coordinates reporting, validation, and publication;
`aka.workflow.light.journal` validates and persists the private Journal.
CandidateWorkspace performs sealing, while Acceptance and Workflow own verification and promotion.

The report transaction validates citations, runs required candidate checks, seals exact source,
persists the report, then publishes controller compatibility state and handoff.
Later draft edits cannot change the seal. Identical accepted reports are idempotent and can
repair interrupted publication; conflicting terminal mutations are rejected.

Mandatory native correctness requires six cases and full coverage under matching source,
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
Journals and exported indexes are bounded to 16 MiB, with smaller per-field limits.

History queries include the current Episode and finalized earlier Episodes in the same
campaign, including rejected explorations. They exclude unfinished/future/unrelated Episodes.
The process-local derived index is rebuilt on restart; private Journal files remain authoritative.

Bubblewrap hides the store from the agent. Native mode retains same-UID authority.
Preserve Journals and measurement records when restoring a matching deployment.
For the complete acceptance boundary, see [handoff and promotion](supervisor-promotion.md).
