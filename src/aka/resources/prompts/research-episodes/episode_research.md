## Direction and Experiment records

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
or research lifecycle transition. A new concrete attempt under the same high-level idea remains
under the same Direction when that module is enabled.


Start from the relevant Direction assessment and plan, then read its linked Experiments and other visible Experiments addressing the same bottleneck, including unassociated ones. Do not treat a Direction summary as a substitute for the observations behind it.

When choosing the next implementation, revisit the visible evidence for the same bottleneck, workload regime, or architecture objective, including related attempts under other names. Use the current visible versions; consult older versions or bounded Gateway Record fields only to resolve a relevant ambiguity. In the normal next-action explanation, briefly connect: **Tried** (implementation and measured scope), **Ruled out** (only what those measurements actually exclude), and **Still untested** (remaining variants, regimes, or combinations). Cite the relevant record IDs and explain how the next candidate differs from earlier attempts. Reuse this assessment while it remains current; do not reread the library after every tool call.

A failed candidate does not rule out its whole strategy. Separate correctness or infrastructure failures from measured performance regressions, and distinguish cost-model estimates from end-to-end measurements. Missing evidence means untested, not disproven. Compare promising untested branches before repeating a measured variant; a repeat should address a changed implementation, setting, or unresolved measurement. This is a decision aid, not a new report: no extra record, revision, Review, or lifecycle transition is required merely to summarize prior evidence.

Both modules are enabled. Start with `PROGRESS.md`; open only the research files needed for the next decision. A Direction is a durable optimization program organized around a bottleneck, workload regime, or architecture objective and should normally span multiple candidates and Experiments. A single implementation, parameter sweep, capability probe, or mechanism test is an Experiment, not a new Direction. Reuse matching Directions and check duplicates before proposing. Choose executable work from the Local work queue, explicitly restart paused work, and treat the Shared direction pool as discovery-only. Absorb compatible peer ideas into a local derivative citing every source ID. Genealogy does not inherit source, evidence, status, or priority.

Keep the same Direction while the core optimization idea and applicable problem remain unchanged. Its revisions update the current assessment and remaining branches. Create a new Direction when the main optimization principle or target problem changes, not merely when a new implementation is tried. A successful Experiment adds a viable branch; it does not by itself complete the Direction.

Establish or reuse a Direction once the high-level idea is clear. Exploratory probes and a minimal implementation may precede proposal or start; research bookkeeping does not gate implementation or measurement. Update the Direction only when its high-level assessment, remaining branches, or actual work state changes. Lifecycle is explicit: `propose` creates `proposed`; call `start` with `direction_id` to enter `in_progress` (`analysis` is optional; omitting it preserves the current assessment); only an `in_progress` Direction may `complete`, `abandon`, `defer`, or `block`. Restart `deferred` or `blocked` work with `start`. `completed` and `abandoned` are terminal, so continued work needs a derived Direction.

Starting a Direction makes it the focus. Queue reordering does not change focus. An Experiment may be recorded before any Direction exists. If `direction_id` is omitted, Runtime infers it only when exactly one local Direction is active; otherwise it stays unassociated. Later, revise the same Experiment with a local `direction_id` to associate it. An existing association cannot be changed. Proposing or starting a Direction requires no Experiment. A proposal can state the objective and plan without measured numeric claims; do not wait for another probe to assemble a complete evidence narrative. A local proposed Direction may own an Experiment when a decision-changing measurement already exists; new research can start immediately after proposal. Do not put measured numeric claims only in Direction prose. Inspect and continue imported peer history through remote-tracking refs such as `atrex/T2` with normal Git commands.

An Experiment records observations, their measured scope, and the next action; the mechanism may remain unknown. Record an Experiment when a concrete implementation, probe, or test has enough evidence to
assess the attempt. Combine its related debugging, parameter trials, and measurements. As you
conclude an attempt and prepare to switch to a materially different approach, promptly save a
short record of the method, expectations, observations, remaining uncertainty, and next choice;
do not defer that record until the next attempt produces results. Recording is not a prerequisite
for continuing implementation, measurement, or Kernel submission. Do not add verification solely
to finish the record, wait for mechanism proof, or revise a Direction merely because an Experiment
was recorded.

An Experiment is one concrete investigative attempt: a candidate implementation, diagnostic probe, microbenchmark, correctness test, profiling experiment, or controlled comparison. It records what was tried and what the observations establish. Keep its identity while the specific question and experimental approach remain unchanged. Use revision for repeat measurements, corrections, explanation, or repairs within that attempt. Create a new Experiment when the experimental question or method, implementation strategy, key dataflow, or combination changes materially, even within the same Direction. Several commits, parameter trials, and diagnostic Records may belong to one attempt; one commit or tool call does not require one Experiment. A complete Kernel or full Evaluate is not required. Preserve earlier attempts rather than replacing them with the latest conclusion. Not every Command or Evaluate needs an Experiment. Cite all relevant Gateway Records or archived references, including regressions that limit the conclusion. Runtime retains each measurement and its source identity. Use concrete `intent` and bounded `interpretation`; a disposable probe that changes no decision need not become an Experiment.

Records cited by a Direction/Experiment currently visible to this lineage and passing standard full-Evaluate Records are visible here; accepted submissions and winners therefore carry their complete Evaluate evidence. Other local Records remain private.

A supported Experiment updates the evidence for its Direction; it does not imply that the Direction is complete. Compare the remaining branches and continue the highest-value candidate trial. Keep the Direction active while high-value variants, parameter families, workload regimes, or combinations remain unexplored. A Direction revision is not required after each Experiment; use `revise` when its assessment or remaining plan changes materially. Optional `analysis` records interim findings while keeping the Direction active; omitting it retains the previous value. Create a derived Direction only when the optimization objective or applicability changes materially.

Close a Direction only when its optimization program is genuinely completed or exhausted, strategically deferred, concretely blocked, or reasonably abandoned. Cite relevant `supporting_experiment_ids` when available; a lifecycle decision may omit them when no Experiment directly establishes it.

`start`, `defer`, and `block` allow `partially_supported`. Only `complete` and `abandon` require `supported` Review of that terminal decision; otherwise the Direction remains `in_progress` and the Review is returned for correction.

Direction `propose` runs independent admission Review and writes immutable `v1`; no Experiment or prior measurement is required. Every `revise`, Direction closure, and Experiment submission is reviewed. `start` changes lifecycle without Review and retains the prior review quality. Each archived update writes the next immutable `vN`; Experiments also start at `v1`. Accepted versions enter the Campaign Research Library. The workspace `research-library/` is its read-only, visibility-filtered copy: synchronous search freezes peer visibility at each Epoch boundary, while asynchronous search exposes new peer revisions immediately. Only the original owner may revise an identity; continue a peer idea through a new local derived record. Review labels evidence strength, not visibility. `archive_status` reports storage, `claim_review_quality` reports review state, and `evidence_eligible` marks established support. Partial records may guide further trials within `scope_safe_interpretation`; unresolved claims remain uncertain. `scientific_next_check` is advisory and may say no additional check is needed. Perform it or revise only when the uncertainty affects the next implementation choice or a lifecycle decision; do not seek `supported` through irrelevant mechanism probes or repeated submissions. `review_unavailable` records remain visible; Runtime retries an unavailable Reviewer once. Unsupported mutations have no `archive_path`. Runtime returns the full Review once, then projects compact fields; full receipts and conversations remain Supervisor-private. Follow the installed Skill for exact fields.

For example, one output-coalescing Direction can accumulate a store-profile Experiment, a layout microbenchmark, an end-to-end implementation comparison, and a boundary-correctness test. Repeating the same microbenchmark revises it; a different test question or materially different method creates another Experiment. These are illustrative options, not required stages.

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
Supply `direction_id` when you intend a specific local Direction; do not delay an unassociated Experiment to create one.

Claim Review is disabled by default unless explicitly enabled for this Campaign. If Runtime returns `review_skipped: true`, it was skipped. The record is archived with `claim_review_quality=review_not_requested` and `evidence_eligible=false`; this is not a supported verdict. Recording, revisions, sharing and lifecycle operations remain available.

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
