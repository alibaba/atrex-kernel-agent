---
name: runtime-records
description: Inspect saved evidence and review exact Kernel commits without Direction or Experiment recording.
allowed-tools: "Bash Read"
---

# Runtime records — report-only mode

Direction and Experiment recording are disabled; automatic Gateway Records and exact-commit source review remain enabled. Start with root `PROGRESS.md` and relevant `records/*.json`. Do not call Direction or Experiment operations or invent their IDs.
Read or query relevant `records/<gateway_record_id>.json` files directly, selecting only needed
fields or bounded text; do not print complete Records or long streams into the Session.

Use local Git for candidate history. Before every Kernel-bound Runtime request, commit the complete SourceProvider candidate and leave the worktree clean; the full HEAD is its `kernel_commit_id`. Continue bounded edit/measurement/revert-or-keep cycles while the selected path has a concrete high-value next step. One useful checkpoint alone is not a stop condition.

Use `submit-kernel` when you judge an exact reachable Agent Git commit ready to serve as a final implementation for the current mode. It must already have a passing standard full Evaluate with complete per-Shape performance data. Submit when ready, not mechanically after every Evaluate. Fast permits declared reproducible compute libraries; Performance requires self-authored core compute with Agent-selected DSL; Ultra additionally enforces its configured DSL. Follow [the exact submission schema](references/journal.md) and [saved result/source semantics](references/records.md).

End an optimization Episode by ending the Session normally. No Episode Report is required.
