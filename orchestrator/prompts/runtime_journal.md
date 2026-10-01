## Runtime Journal and handoff

Use `update-direction` before exploring a Direction. Advance at most three Directions per Episode,
with only one `in_progress` at a time; additional proposals may remain for later. Put plans in
`plan`, `success_criteria`, and `stop_conditions`. Use `record-experiment` after decisive results,
citing Gateway Record IDs in `gateway_record_ids` and separating `evidence` from `analysis`.
Reuse saved results and exact source when applicable; separate plan/profile documents are not required.

Close every in-progress Direction before `episode-report`. Each closure must explicitly select
`supporting_experiment_ids` and declare `hypothesis_status=unresolved|supported|refuted`.
Lifecycle is not a hypothesis verdict. Every Experiment needs at least one real Kernel-bound Gateway
Record; failed diagnostics may document an unresolved blocker, but supported/refuted requires a
completed observation in every selected Experiment. Runtime validates bindings, not causal relevance.
Never invent evidence to close a Direction. Late Experiments update `associated_experiment_ids`
without reopening a Direction or changing its selected support. For a candidate, select a current-Episode
Experiment citing a passing Evaluate for the exact source left in `kernel.py`;
Profile or ABBA alone is insufficient for this handoff. Prepare records while working, not only at
termination. Correct rejected reports and submit again; stop after acceptance.

Chat text is not a handoff. Submit JSON in the `runtime-records` Skill's format with
`python3 tools/sandbox.py --kind episode-report --request-file scratch/episode-report.json`.
