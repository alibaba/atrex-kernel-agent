---
name: runtime-records
description: Use sandbox.py to track research Directions, record Experiments, retrieve saved GPU results and Kernel source, and submit an Episode report. Read when using historical evidence, updating the Journal, or preparing terminal handoff.
allowed-tools: "Bash Read"
---

# Runtime records

Run `python3 tools/sandbox.py` from the Agent workspace. The Supervisor owns the persistent
Journal, exact Kernel snapshots, and measurement records; do not search for or edit backing files.
This Skill is always mounted. It does not change the current phase's permitted operations or finish
procedure: a framework-baseline session still uses its injected smoke-and-exit procedure.

## Use the records while working

- Before exploring a hypothesis, inspect relevant history, then propose/start a Direction or resume
  an existing deferred one. A Direction is a research avenue, not each individual code edit.
- Record each decisive experiment promptly, citing the actual `gateway_record_ids`. Record measured
  facts in `evidence` and your interpretation in `analysis`; earlier interpretations may be wrong.
- Reuse saved measurements and exact source instead of repeating an identical completed task.
  Changing the probe or measurement question can be a different task; do not change it merely to
  evade deduplication.
- Before `episode-report`, close all in-progress Directions and select the Experiment supporting
  the exact candidate left in `kernel.py`. An accepted report is a handoff, not final Kernel promotion.

The usual linkage is Direction → Experiment → Gateway Record → Kernel source. Gateway results also
exist without an Experiment: measurement persistence is automatic, interpretation is Agent-authored.
Use returned IDs, never invented IDs or a remote Agate Job ID in their place.

## Read only the details needed

- [Journal and worked Episode example](references/journal.md): Direction lifecycle, all seven
  Journal/report commands, request and response examples, historical indexes, and alternate terminal states.
- [Saved results and source](references/records.md): `record-read`, immediate versus historical result
  formats, Kernel lookup, exact source recovery, and error handling.
- For new GPU requests or custom-input/Dev examples, use `skills/gpu-measurement/SKILL.md`.
  For Wiki retrieval, use `skills/KernelWiki/SKILL.md`.

Keep request JSON and exported indexes under `scratch/`. New Episodes start with an empty scratch
directory; same-Episode recovery preserves it. Successful Journal writes are immediately durable
and queryable in later visible Episodes. List exports are snapshots: rerun the list command to refresh
one after new writes. Do not replace durable Journal entries with scratch notes.

If a request fails, follow `error.next_action`; correct rejected input before retrying. If the outcome
is unknown, inspect available records before resubmitting a write or GPU job. Infrastructure repair
belongs to the operator, not the Agent. See the error section in the saved-results reference.
