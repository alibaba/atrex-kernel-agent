---
name: runtime-records
description: Record research Directions and Experiments, query saved evidence, and submit a validated Episode Report through the Supervisor.
allowed-tools: "Bash Read"
---

# Runtime records

Use `python3 tools/sandbox.py` inside a controller-registered Long Horizon Episode.
Setup and Framework Baseline keep their existing finish procedures. This Skill does not replace
the Episode's planning, profiling, Fast/Full, Wiki attribution, Phase Marker or verification policy.

Choose one Journal interface before recording experiments. Existing local Journal commands remain
supported. If using Runtime Journal, replace only local append/finalize and manual handoff with
the commands here; do not mix the two Journal protocols. Git stays Agent-owned: commit only
`kernel.py` before reporting a candidate, and supply that Commit ID. The Supervisor still decides
acceptance independently.

1. Inspect prior evidence, propose a research Direction, then start it before exploration.
2. Record decisive Experiments promptly, linking actual `gateway_record_ids`; separate measured
   `evidence` from your `analysis`. Measurements are automatically saved; interpretation is not.
3. Close the Direction with an explicit hypothesis assessment and supporting Experiment IDs.
4. Prepare the report incrementally. Close all active Directions before `episode-report`.
   Correct validation errors and resubmit. An accepted report is a handoff, not promotion.

The link is Direction → Experiment → Gateway Record → exact Kernel. Propose as many useful
nonduplicate ideas as needed; start at most three distinct Directions per Episode and only one at
a time. List exports are snapshots under `scratch/`; rerun list to refresh. Current and finalized
earlier Runtime Journals are visible through the tools, including unsuccessful explorations.
Legacy Journals are not automatically converted into Direction/Experiment IDs.

- [Journal requests and worked example](references/journal.md): seven Journal/report commands.
- [Saved results and source](references/records.md): existing measurement readers and errors.
- For new measurements use `skills/gpu-measurement/SKILL.md`; for knowledge queries use
  `skills/KernelWiki/SKILL.md`.

Never invent IDs or measured results. Read an existing result instead of repeating an identical
task. If a transport failure leaves a mutation's outcome unknown, inspect records before retrying.
Identical accepted reports can be resubmitted to repair handoff publication; different terminal
reports and further Journal mutations are rejected.
