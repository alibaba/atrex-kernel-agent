---
name: runtime-records
description: Manage research Directions and review exact Kernel commits; Experiment recording is disabled.
allowed-tools: "Bash Read"
---

# Runtime Direction records

Direction recording is enabled; Experiment commands are disabled. Start with root `PROGRESS.md`, then open only relevant `research-library/directions/<direction_id>/vN/direction.json` and `records/*.json`; the greatest visible `vN` is current. These Runtime projections are read-only.
Read or query relevant `records/<gateway_record_id>.json` files directly, selecting only needed
fields or bounded text; do not print complete Records or long streams into the Session.

A Direction is a durable optimization program around a consequential bottleneck, workload regime,
or architecture objective, not Runtime/evaluator maintenance. It should span multiple candidates or
probes; a single implementation, parameter choice, capability probe, or mechanism test remains a
bounded step under an existing Direction. Check for duplicates and rank by expected benefit and
applicability. Multiple local Directions may be `in_progress`; the most recently started or prioritized one is the focus. Choose executable work from the Local work queue; restart paused work explicitly. The Shared direction pool is discovery-only: absorb compatible peer ideas into one local derivative and cite every source ID in `derived_from_direction_ids`. Continue bounded probe/edit/measurement/revert-or-keep cycles while useful work remains.

Use `update-direction` and `prioritize-direction` for mutations. Routine execution of plan steps does not require a Runtime mutation; use `action=revise` only when the claim or remaining plan changes materially. Omit `supporting_experiment_ids`; Experiment-derived genealogy is unavailable. Claim-bearing Direction requests may cite their own measurement evidence with `gateway_record_ids`, allowing the independent Review to inspect the corresponding Records. Use `submit-kernel` when you judge an exact Agent Git commit ready to serve as a final implementation for the current mode; it must already have a passing standard full Evaluate with complete per-Shape performance. Submit when ready, not mechanically after every Evaluate. Follow [the exact Direction and submission schemas](references/journal.md) and [saved result/source semantics](references/records.md). Never invent IDs or measured results.

Lifecycle is `proposed --start--> in_progress`. Only `in_progress` may `complete`, `abandon`,
`defer`, or `block`; restart `deferred` or `blocked` work with `start`. `completed` and `abandoned`
are terminal. In this Direction-only mode, closure evidence uses `gateway_record_ids`, including for
`abandon`; do not use `supporting_experiment_ids`.

Revise a proposed or active local Direction with `action=revise`, its `direction_id`, and the complete
revised claim and plan. Restart a deferred or blocked Direction first. Completed and abandoned
Directions are terminal; continue them through a new derived Direction. Runtime keeps the ID,
increments `revision`, reruns Review, and preserves lifecycle status and genealogy. Shared peer
Directions are read-only.
