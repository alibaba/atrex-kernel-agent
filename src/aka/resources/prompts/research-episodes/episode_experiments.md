## Experiment records

Experiment recording is enabled; Direction recording is disabled. Read only relevant `research-library/experiments/<experiment_id>/vN/experiment.json` files; the greatest visible `vN` is current. Revise only `local_writable` Experiments; extend peer evidence with a new local Experiment. Record an Experiment at a decision boundary: when you use measured results to keep or discard a candidate, select a parameter or workload strategy, rule out a branch, or switch paths. Record the observations and chosen next action before pursuing that action. Combine completed trials that inform this same decision; once the decision is made, do not postpone recording for the next probe, a complete implementation, full Evaluate, mechanism proof, or Direction closure. Revise the Experiment when later evidence updates the same conclusion; use a new Experiment for a different question. Not every Command or Evaluate needs an Experiment. Record observations, their measured scope, and the next action; the mechanism may remain unknown. Cite relevant results, including regressions that limit the conclusion. Runtime retains each measurement and its source identity; a disposable probe that changes no decision need not be recorded. Use `name`, concrete `intent`, and bounded `interpretation`; omit `direction_id`. `gateway_record_ids` and `references` are independent optional lists. Runtime validates Records and archives referenced workspace material beside that immutable version.

Records cited by an Experiment currently visible to this lineage and passing standard full-Evaluate Records are visible here; accepted submissions and winners therefore carry their complete Evaluate evidence. Other local Records remain private.

Each submission returns an inline independent `review`. `archive_status`, `claim_review_quality`, and `evidence_eligible` distinguish storage from established support. Partial records may guide further trials within `scope_safe_interpretation`; unresolved claims remain uncertain. `scientific_next_check` is advisory and may say no additional check is needed. Perform it or revise only when the uncertainty affects the next implementation choice; do not seek `supported` through irrelevant mechanism probes or repeated submissions. Every archived revision enters `research-library/` with its actual quality. Synchronous peers see the revision frozen at the last Epoch boundary; asynchronous peers see revisions immediately. Unsupported submissions have no `archive_path`. Runtime retries an unavailable Reviewer once before archiving with `claim_review_quality=review_unavailable`. Runtime returns the full Review once; compact fields remain projected while full receipts and conversations are Supervisor-private. Revise local Experiments by stable `experiment_id`.

## Record the decision

After using a measurement to choose the next action, write `scratch/experiment.json`:

```json
{
  "name": "MMA probe selects the next candidate",
  "intent": "Choose the compute path for the next implementation",
  "gateway_record_ids": ["gateway-<actual-record-id>"],
  "interpretation": "The tested configuration passed the probe check. Next implement a candidate using it; full-Kernel correctness and performance remain untested."
}
```

Use the actual observation and Record ID, then run:

```bash
python3 tools/sandbox.py --kind record-experiment --request-file scratch/experiment.json
```

Do this before the next implementation or probe that relies on that decision. To update the same
conclusion later, submit the required fields with the returned `experiment_id`.
