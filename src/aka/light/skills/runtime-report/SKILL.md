---
name: runtime-records
description: Query saved measurement evidence and submit an Episode Report without Direction or Experiment recording.
allowed-tools: "Bash Read"
---

# Runtime records — report-only mode

Use `python3 tools/sandbox.py` inside a controller-registered Episode. Direction and research
Experiment recording are disabled; do not use `update-direction` or `record-experiment`.
This does not disable automatic measurement persistence, duplicate detection, correctness checks,
or independent ABBA promotion. Framework Baseline has its own smoke-and-exit procedure.

Inspect saved measurement evidence and optional canonical `memory/v*.json` summaries when present.
Implement and measure a candidate, then leave its exact source in place (`kernel.py` or the
contracted source tree from `WORKSPACE.md`) and cite real Gateway Record IDs in `episode-report`.
No synthetic Direction, Experiment, separate plan/profile document, or Git operation is required.

- [Episode report and examples](references/journal.md): direct evidence selection and repair.
- [Saved results and source](references/records.md): measurement readers, per-Shape historical best, and errors.
- Use `skills/gpu-measurement/SKILL.md` for new measurements.

Read-only Direction/Experiment history queries remain available for earlier recorded Episodes;
they do not create records or change the current policy. Never invent IDs or measured results.
Reuse saved results rather than repeating identical tasks. Accepted reports can be replayed to
repair publication, but different terminal reports are rejected. Report acceptance is not promotion.
