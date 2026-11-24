## Experiment records

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
or research lifecycle transition. Keep related attempts discoverable by their shared optimization idea; no Direction is needed.


Find relevant visible Experiments by their name, intent, and interpretation, including mechanism aliases; group them by the current bottleneck or objective rather than requiring a Direction.

When choosing the next implementation, revisit the visible evidence for the same bottleneck, workload regime, or architecture objective, including related attempts under other names. Use the current visible versions; consult older versions or bounded Gateway Record fields only to resolve a relevant ambiguity. In the normal next-action explanation, briefly connect: **Tried** (implementation and measured scope), **Ruled out** (only what those measurements actually exclude), and **Still untested** (remaining variants, regimes, or combinations). Cite the relevant record IDs and explain how the next candidate differs from earlier attempts. Reuse this assessment while it remains current; do not reread the library after every tool call.

A failed candidate does not rule out its whole strategy. Separate correctness or infrastructure failures from measured performance regressions, and distinguish cost-model estimates from end-to-end measurements. Missing evidence means untested, not disproven. Compare promising untested branches before repeating a measured variant; a repeat should address a changed implementation, setting, or unresolved measurement. This is a decision aid, not a new report: no extra record, revision, Review, or lifecycle transition is required merely to summarize prior evidence.

Experiment recording is enabled; Direction recording is disabled. Read only relevant `research-library/experiments/<experiment_id>/vN/experiment.json` files; the greatest visible `vN` is current. Revise only `local_writable` Experiments; extend peer evidence with a new local Experiment. Record an Experiment when a concrete implementation, probe, or test has enough evidence to
assess the attempt. Combine its related debugging, parameter trials, and measurements. As you
conclude an attempt and prepare to switch to a materially different approach, promptly save a
short record of the method, expectations, observations, remaining uncertainty, and next choice;
do not defer that record until the next attempt produces results. Recording is not a prerequisite
for continuing implementation, measurement, or Kernel submission. Do not add verification solely
to finish the record, wait for mechanism proof, or revise a Direction merely because an Experiment
was recorded.

An Experiment is one concrete investigative attempt: a candidate implementation, diagnostic probe, microbenchmark, correctness test, profiling experiment, or controlled comparison. It records what was tried and what the observations establish. Keep its identity while the specific question and experimental approach remain unchanged. Use revision for repeat measurements, corrections, explanation, or repairs within that attempt. Create a new Experiment when the experimental question or method, implementation strategy, key dataflow, or combination changes materially, even within the same Direction. Several commits, parameter trials, and diagnostic Records may belong to one attempt; one commit or tool call does not require one Experiment. A complete Kernel or full Evaluate is not required. Preserve earlier attempts rather than replacing them with the latest conclusion. Not every Command or Evaluate needs an Experiment. Record observations, their measured scope, and the next action; the mechanism may remain unknown. Cite relevant results, including regressions that limit the conclusion. Runtime retains each measurement and its source identity; a disposable probe that changes no decision need not be recorded. Use `name`, concrete `intent`, and bounded `interpretation`; omit `direction_id`. `gateway_record_ids` and `references` are independent optional lists. Runtime validates Records and archives referenced workspace material beside that immutable version.

Records cited by an Experiment currently visible to this lineage and passing standard full-Evaluate Records are visible here; accepted submissions and winners therefore carry their complete Evaluate evidence. Other local Records remain private.

Each submission returns an inline independent `review`. `archive_status`, `claim_review_quality`, and `evidence_eligible` distinguish storage from established support. Partial records may guide further trials within `scope_safe_interpretation`; unresolved claims remain uncertain. `scientific_next_check` is advisory and may say no additional check is needed. Perform it or revise only when the uncertainty affects the next implementation choice; do not seek `supported` through irrelevant mechanism probes or repeated submissions. Every archived revision enters `research-library/` with its actual quality. Synchronous peers see the revision frozen at the last Epoch boundary; asynchronous peers see revisions immediately. Unsupported submissions have no `archive_path`. Runtime retries an unavailable Reviewer once before archiving with `claim_review_quality=review_unavailable`. Runtime returns the full Review once; compact fields remain projected while full receipts and conversations are Supervisor-private. Revise local Experiments by stable `experiment_id`.

For example, an output-coalescing investigation can include a store-profile Experiment, a layout microbenchmark, an end-to-end implementation comparison, and a boundary-correctness test. Repeating the same microbenchmark revises it; a different test question or materially different method creates another Experiment. These are illustrative options, not required stages.

## Compare predictions with observations

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

## Record the decision

After using a measurement to choose the next action, write `scratch/experiment.json`:

```json
{
  "name": "MMA probe selects the next candidate",
  "intent": "Compare MMA paths in the tested configuration",
  "gateway_record_ids": [
    "gateway-<actual-record-id>"
  ],
  "interpretation": "Test this path in a full candidate. The cause of the throughput difference, other Shapes, and end-to-end latency remain unverified.",
  "implementation": "Run a controlled throughput and output-check probe against the comparator",
  "expectations": [
    {
      "scope": "Tested probe configuration",
      "prediction": "Higher throughput than the comparator (retrospective reconstruction; no advance prediction was recorded)",
      "observation": "Output check passed and measured throughput improved",
      "assessment": "met"
    }
  ]
}
```

Use the actual observation and Record ID, then run:

```bash
python3 tools/sandbox.py --kind record-experiment --request-file scratch/experiment.json
```

Save the concluded attempt promptly without waiting for the next attempt's results; this command
is not a gate on continuing work. To supplement or correct the same attempt, submit its required
fields with the returned `experiment_id`.

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
