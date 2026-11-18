---
name: runtime-records
description: Maintain enabled research records and review exact Kernel commits through Runtime.
allowed-tools: "Bash Read"
---

# Runtime records

Use `python3 tools/sandbox.py`; Help lists the enabled operations. Runtime owns immutable
measurements and commit review; the Agent owns optimization decisions and interpretations.
Independent requests may run concurrently. Collect the completed receipt before using its result
or revising the same identity.

Read `PROGRESS.md`, then only the research versions and bounded Record fields needed for the next
decision. Gateway Records are measurement authority. The greatest visible `vN` is current.
`research-library/` and `records/` are read-only projections; change research only through Runtime.
Revise only `local_writable` identities. Extend peer evidence with a new local record.
Synchronous peers see revisions at Epoch boundaries; asynchronous peers see revisions immediately.

## Directions

A Direction is a durable program around a bottleneck, workload regime, or architecture objective,
spanning multiple candidates. Check duplicates and rank by expected benefit. Choose executable
work from the Local work queue; absorb Shared pool ideas into local derivatives with source IDs.
`propose` requires an objective and plan, not prior measurements. It runs admission Review.
`start` changes `proposed`, `deferred`, or `blocked` to `in_progress` without another Review.
It accepts `action`, `direction_id`, and optional `analysis`. Multiple Directions may be active;
starting sets focus, while queue reordering does not.

Use `revise` when the objective, plan, or interim `analysis` changes materially. Keep work active
while high-value branches remain. Only `in_progress` may `complete`, `abandon`, `defer`, or `block`.
`complete` and `abandon` require supported closure Review; partial closure leaves work active.
Completed and abandoned Directions are terminal; continue through a derivative.
See [Direction requests](references/journal.md#propose-and-select-a-direction).

## Experiments

Record at a decision boundary: when you use a measurement to keep/discard a candidate, select a
strategy, rule out a branch, or choose the next implementation or probe. Record the observations,
measured scope, and next action before pursuing that action. Combine completed trials informing
this decision; do not wait for another probe, a complete Kernel, full Evaluate, or mechanism proof.
A disposable probe changing no decision need not be recorded. An unknown mechanism is acceptable.

Write `scratch/experiment.json` with `name`, `intent`, `interpretation`, and optional
`gateway_record_ids` and `references`, then run:

```bash
python3 tools/sandbox.py --kind record-experiment --request-file scratch/experiment.json
```

Cite only evidence supporting the bounded observation, including regressions. Env and unbound
probe Records may establish environment facts or probe behavior, not full-Kernel performance.
Later evidence updating the same conclusion uses the returned `experiment_id` and required fields;
a different question gets a new Experiment. References are archived by Runtime.
A supported Experiment does not imply that an optimization program is complete.

<!-- directions:start -->
An Experiment can exist before a Direction. Omitted `direction_id` is inferred only when exactly
one local Direction is active; otherwise it stays unassociated. A revision can add a local
`direction_id` later; an existing association cannot change. A proposed Direction may own an
Experiment. Do not delay recording to create or review a Direction first.
<!-- directions:end -->

See [Experiment requests and examples](references/journal.md#record-an-experiment).

## Research review

Review returns inline. `archive_status` distinguishes storage from `claim_review_quality` and
`evidence_eligible`. Use partial observations within `scope_safe_interpretation` for further trials.
`scientific_next_check` is advisory: perform it only when the uncertainty affects the next choice.
Do not repeat probes or submissions merely to obtain `supported`. Unsupported submissions have no
archive path; unavailable Review is retried once. Archived versions remain visible with their
actual quality. Full Review receipts and conversations are Supervisor-private.

## Commit review

Use `submit-kernel` when an exact commit is ready for the selected final-source policy and already
has passing standard full-Evaluate evidence with complete per-Shape performance data. Submission
does not require research records. Fast permits declared reproducible compute libraries;
Performance requires self-authored core compute; Ultra also enforces the configured DSL.
See [commit review requests](references/journal.md#review-an-exact-kernel-commit).

Use Agent Git for checkpoints and peer refs such as `atrex/T2`. See [saved results](references/records.md)
and `.agents/skills/gpu-measurement/SKILL.md` for measurements.
