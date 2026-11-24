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

## Select the next attempt

You are responsible for both choices: which measured implementation to retain in your Git,
and which implementation or investigation to explore next. Runtime's accepted best is a measured
reference, not a command to keep modifying that candidate. After a meaningful attempt yields
results, a key expectation is missed, or you are about to switch paths, compare continuing an
untested branch, revisiting an earlier branch under a changed implementation or measurement,
and exploring another promising idea. Use the relevant accumulated evidence and its limits.
Choose for plausible full-workload benefit and what the trial can resolve. A `supported` review
label describes evidential support, not speed, importance, or search priority; do not rank by it.
A high-upside but uncertain rewrite may merit the smallest end-to-end trial without proving its
benefit first. Complexity or risk alone is not evidence that it cannot help.

Briefly state the next attempt, the evidence behind the choice, what remains untested, and what
the attempt will distinguish in your normal next-action explanation. Skip alternatives that are
irrelevant or already resolved; reuse the assessment until evidence changes. Do not perform this
comparison after every Command. It requires no separate selector, report, Runtime call, Review,
or research lifecycle transition. <!-- directions:start -->
New attempts under the same high-level idea remain within the same Direction; switching to a
materially different idea can use another Direction under the existing lifecycle.
<!-- directions:end -->
<!-- experiments:start -->
Use related Experiments' `implementation`, `expectations`, and `interpretation` to compare
attempts. A materially different experimental approach gets a new Experiment, not a replacement
of earlier attempts. These fields inform selection; missing predictions do not prevent action.
<!-- experiments:end -->

## Directions

Before choosing the next implementation, read related visible Directions' `analysis`, remaining
plans, and cited evidence, including paused or terminal attempts relevant to the same bottleneck.
Compare what was tried, what the measurements actually rule out, and what remains untested.
Use the current visible versions and cite relevant IDs in the normal next-action explanation;
explain how the next candidate differs. Reuse that assessment until new evidence changes it.
A Direction summary alone does not establish that every implementation of its strategy failed.
Correctness or infrastructure failures and cost estimates do not establish a performance limit.
Missing evidence means untested. This readback requires no extra record, revision, Review, or
lifecycle transition.

A Direction is a durable program around a bottleneck, workload regime, or architecture objective,
spanning multiple candidates. Check duplicates and rank by expected benefit. Choose executable
work from the Local work queue; absorb Shared pool ideas into local derivatives with source IDs.
Establish or reuse a Direction once the high-level idea is clear; exploratory probes or a minimal
implementation may come first. Recording is not a prerequisite for implementation or measurement.
Update its assessment, branches, or lifecycle only when those actually change.
`propose` requires an objective and plan, not prior measurements. It runs admission Review.
`start` changes `proposed`, `deferred`, or `blocked` to `in_progress` without another Review.
It accepts `action`, `direction_id`, and optional `analysis`. Multiple Directions may be active;
starting sets focus, while queue reordering does not.

Keep the same Direction while the core optimization idea and applicable problem remain unchanged. Its revisions update the current assessment and remaining branches. Create a new Direction when the main optimization principle or target problem changes, not merely when a new implementation is tried. A successful Experiment adds a viable branch; it does not by itself complete the Direction.

Use `revise` when the plan or interim `analysis` changes materially within that idea. Keep work active
while high-value branches remain. Only `in_progress` may `complete`, `abandon`, `defer`, or `block`.
`complete` and `abandon` require supported closure Review; partial closure leaves work active.
Completed and abandoned Directions are terminal; continue through a derivative.
See [Direction requests](references/journal.md#propose-and-select-a-direction).

## Experiments

Experiments include probes, microbenchmarks, correctness tests, profiling, and implementation trials.
Use prior Experiments to choose the next candidate or investigation. Search the visible library by bottleneck,
workload regime, objective, and mechanism aliases; include unassociated Experiments. Read current
visible versions and only the older versions or bounded Record fields needed to clarify the decision.
<!-- directions:start -->
Start with Experiments linked to the relevant Direction, then check related attempts outside it.
<!-- directions:end -->
In the normal next-action explanation, briefly connect **Tried** (implementation and measured scope),
**Ruled out** (the narrow exclusion actually supported), and **Still untested** (remaining variants,
regimes, or combinations), citing relevant IDs. Explain how the next candidate differs. Reuse this
assessment until new evidence changes it; no extra record, revision, Review, or lifecycle transition
is required for this readback. Do not reread the entire library after every tool call.
A correctness or infrastructure failure is not a performance result; a cost estimate is not an
end-to-end measurement. A failed implementation does not disprove its strategy. Missing evidence
means untested. Compare promising untested branches before repeating a measured variant; repeat
when implementation, setting, or an unresolved measurement warrants it.

Record an Experiment when a concrete implementation, probe, or test has enough evidence to
assess the attempt. Combine its related debugging, parameter trials, and measurements. As you
conclude an attempt and prepare to switch to a materially different approach, promptly save a
short record of the method, expectations, observations, remaining uncertainty, and next choice;
do not defer that record until the next attempt produces results. Recording is not a prerequisite
for continuing implementation, measurement, or Kernel submission. Do not add verification solely
to finish the record, wait for mechanism proof, or revise a Direction merely because an Experiment
was recorded.
A disposable probe changing no decision need not be recorded. An unknown mechanism is acceptable.

Use optional `implementation` to describe the concrete implementation, probe, or test method
and its difference from prior attempts. Use optional `expectations` for scoped comparisons:
each row contains non-empty `scope`, `prediction`, `observation`, and `assessment`
(`met`, `missed`, or `unresolved`). Keep `intent` for the experimental question and
`interpretation` for remaining hypotheses and the next action.
Distinguish expectations that were met, missed, or remain unresolved; mixed Shape outcomes stay
separate. Keep observations separate from causal explanations: an observed regression can be
established while its cause remains unknown. Label suspected causes as hypotheses, not findings.
Compilation or infrastructure failure leaves a performance prediction unresolved, not missed.
Do not run extra mechanism probes merely to fill in an explanation. Revisions retain the original
prediction and earlier outcome; append updated evidence and assessment rather than rewriting the
prediction to match the result. Cite the relevant Records or archived notes. For verifiable advance predictions, pass `--predictions '[{"scope":"long sequences","prediction":"Lower latency"}]'` to Evaluate or Command (before the command `--` separator). The optional request field `predictions` is a JSON array of `{"scope":"long sequences","prediction":"Lower latency"}`. Runtime freezes it before new execution and returns `frozen_predictions` with the Gateway Record. Cite the Record in `gateway_record_ids`; Runtime automatically brings in all its frozen predictions. Do not supply an index or repeat the forecast. Missing observations remain `unresolved`; explain the actual outcome in `interpretation`. Optional expectation rows with unambiguous matching scope/prediction may update observations and assessments. Frozen forecasts are retained on revision; Runtime never infers results or matches ambiguous identical forecasts across Records. Reused Evaluate results retain only their original forecasts; this call's new predictions are not registered. Local notes alone cannot establish execution order. Unbound expectations are marked `retrospective`. An exploratory
test may have no directional prediction. These optional fields require no extra Runtime call and add no Review or lifecycle gate.
Omitting them on revision retains the previous values. An explicit `expectations` array replaces
the rows in the new version (`[]` clears only unbound rows); frozen predictions must remain, and earlier immutable versions stay archived.
Preserve the original predictions when updating observations and assessments. Source identity
and measured performance remain in Gateway Records; do not duplicate their complete tables.

Write `scratch/experiment.json` with `name`, `intent`, `interpretation`, and optional
`implementation`, `expectations`, `gateway_record_ids`, and `references`, then run:

```bash
python3 tools/sandbox.py --kind record-experiment --request-file scratch/experiment.json
```

Cite only evidence supporting the bounded observation, including regressions. Env and unbound
probe Records may establish environment facts or probe behavior, not full-Kernel performance.
An Experiment is one concrete investigative attempt: a candidate implementation, diagnostic probe, microbenchmark, correctness test, profiling experiment, or controlled comparison. It records what was tried and what the observations establish. Keep its identity while the specific question and experimental approach remain unchanged. Use revision for repeat measurements, corrections, explanation, or repairs within that attempt. Create a new Experiment when the experimental question or method, implementation strategy, key dataflow, or combination changes materially, even within the same Direction. Several commits, parameter trials, and diagnostic Records may belong to one attempt; one commit or tool call does not require one Experiment. A complete Kernel or full Evaluate is not required. Preserve earlier attempts rather than replacing them with the latest conclusion.
Use the returned `experiment_id` and required fields for revisions. References are archived by Runtime.
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

<!-- directions:start -->
Read `research-library/directions/<direction_id>/attempts.md` when choosing the next attempt under a Direction. Runtime automatically groups all linked, visible Experiments using their latest visible revisions, including failures and unresolved predictions; it does not filter by `supporting_experiment_ids` or Review quality. The table shows method differences, predictions, observations, assessments and interpretation. “Not recorded” is missing information, not evidence that a branch was tested or ruled out. Use the linked version JSON for records and attachments. This read-only view follows the same peer visibility watermark as the Research Library and requires no extra record or lifecycle action.
<!-- directions:end -->

<!-- experiments:start -->
Use optional Experiment `next_step` to record the chosen next action separately from the interpretation:
```json
"next_step": {
  "action": "Implement V-split while preserving input reuse",
  "reason": "The previous trial repeated input reads; partitioning with reuse remains untested"
}
```
`action` says what to do; `reason` explains why this is the next useful attempt. Both are nonempty strings when provided. Omit `next_step` on revision to preserve it; use `null` to clear an obsolete plan. This is an editable plan, not a new tool, mandatory report, Review requirement, or execution gate. A concrete next attempt normally stays under the existing Direction; create another Direction only when the high-level idea changes. The Direction attempt view displays action and reason separately; Runtime does not infer them from `interpretation` or execute them automatically.
<!-- experiments:end -->

<!-- directions:start -->
### Where to record the next step

- Another implementation, parameter choice, dataflow variant, or probe under the same high-level idea: keep the current Direction. Once the attempt has useful results, record an Experiment; do not create a Direction for every next action.
- Additional measurements or a correction to the same attempt: revise its Experiment. A materially different approach is a new Experiment under the same Direction when the high-level idea remains unchanged.
- A different primary optimization principle or target problem: create a new Direction.
- A distinct high-level idea worth exploring later: record a proposed Direction (or defer it using the supported lifecycle operation), so it does not survive only in conversation or another record's future-work text. Do not start it merely to document it.

When Experiment recording is enabled, use `next_step.action` and `next_step.reason` for the concrete selected action and rationale. Keep remaining same-idea branches in the Direction's plan or assessment. These are recording guidelines, not prerequisites for implementation or evaluation; do not turn each parameter trial into a separate record.
<!-- directions:end -->
