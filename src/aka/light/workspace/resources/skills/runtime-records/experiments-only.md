---
name: runtime-records
description: Record research Experiments and review exact Kernel commits; Direction recording is disabled.
allowed-tools: "Bash Read"
---

# Runtime Experiment records

Experiment recording is enabled; Direction commands are disabled. Start with root `PROGRESS.md` and `experiments/index.json`, then open only relevant `experiments/<experiment_id>/experiment.json` and `records/*.json`. These Runtime projections are read-only.
Read or query relevant `records/<gateway_record_id>.json` files directly, selecting only needed
fields or bounded text; do not print complete Records or long streams into the Session.

Record each decisive measurement or research conclusion with `name`, concrete `intent`, and bounded `interpretation`; omit `direction_id`. `gateway_record_ids` and archived `references` are independent optional lists. Keep execution facts in Records and source facts in Git rather than copying them into analysis. Continue bounded edit/measurement/revert-or-keep cycles while useful work remains.

Revise only `local_writable` Experiments by resubmitting their required fields and projected ID.
Runtime increments `revision`, reruns Review, and preserves omitted optional evidence. Extend
`peer_read_only` evidence with a new local Experiment.

Use `submit-kernel` when you judge an exact reachable Agent Git commit ready to serve as a final implementation for the current mode; it must already have a passing standard full Evaluate with complete per-Shape performance. Submit when ready, not mechanically after every Evaluate. Follow [the exact Experiment and submission schemas](references/journal.md) and [saved result/source semantics](references/records.md). Never invent IDs or measured results.
