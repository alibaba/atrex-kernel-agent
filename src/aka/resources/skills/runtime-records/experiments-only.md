---
name: runtime-records
description: Record research Experiments and review exact Kernel commits; Direction recording is disabled.
allowed-tools: "Bash Read"
---

# Runtime Experiment records

Experiment recording is enabled; Direction commands are disabled. Start with root `PROGRESS.md`, then open only relevant `research-library/experiments/<experiment_id>/vN/experiment.json` and `records/*.json`; the greatest visible `vN` is current. These Runtime projections are read-only.
Read or query relevant `records/<gateway_record_id>.json` files directly, selecting only needed
fields or bounded text; do not print complete Records or long streams into the Session.

Related candidate trials may continue before recording one Experiment at a decision checkpoint, when switching paths, or when sharing findings. Record observations, their measured scope, and the next action; the mechanism may remain unknown. Cite relevant results, including regressions that limit the conclusion. Runtime retains each measurement and its source identity. A disposable probe that changes no decision need not be recorded. Use `name`, concrete `intent`, and bounded `interpretation`; omit `direction_id`. `gateway_record_ids` and archived `references` are independent optional lists. Continue high-value candidate trials while useful work remains.

A partial Review does not require immediate repair or resubmission. Use observations within `scope_safe_interpretation`; unresolved claims remain uncertain. `scientific_next_check` is advisory and may say no additional check is needed. Perform it or revise only when the uncertainty affects the next implementation choice. Do not repeat measurements or submissions merely to obtain `supported`; an unknown mechanism does not prevent a supported observation and next trial.

Revise only `local_writable` Experiments by resubmitting their required fields and projected ID.
Runtime increments `revision`, reruns Review, and preserves omitted optional evidence. Extend
`peer_read_only` evidence with a new local Experiment.

Use `submit-kernel` when you judge an exact reachable Agent Git commit ready to serve as a final implementation for the current mode; it must already have a passing standard full Evaluate with complete per-Shape performance. Submit when ready, not mechanically after every Evaluate. Follow [the exact Experiment and submission schemas](references/journal.md) and [saved result/source semantics](references/records.md). Never invent IDs or measured results.
