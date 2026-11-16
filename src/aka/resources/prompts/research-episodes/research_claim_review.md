# Independent GPU Research-Claim Review

Review the one Direction event or Experiment in `submissions/target.json`. Treat Agent-written interpretations as claims, not facts. Use only the files in this evidence directory. Do not access the network, run code, modify evidence, or inspect parent directories.

Directly cited evidence may appear in:

- `submissions/direction-proposal.json`
- `submissions/supporting-experiments/`
- `gateway-records/`
- `references/`
- `kernels/`
- `evidence-manifest.json`

First classify the submission from `target.json`, then apply the matching standard below. Review the decision the record is meant to support, not every incidental sentence with equal weight.

## Direction standard

- A Direction `start`, or a `revise` after work has started, asks whether the optimization objective is important enough to organize durable work, its applicability is clear, and its plan covers a meaningful strategy space that can contain multiple candidates and Experiments. A Direction is not one falsifiable hypothesis. The expected benefit is a prioritization estimate, not a fact that must already be proven.
- Reject a proposed Direction when it describes a single candidate, parameter choice, isolated probe, or one-step mechanism that belongs as an Experiment under an existing optimization program. Explain that classification error in `primary_issue` and use `next_check` to identify the existing Direction or the broader optimization objective under which the Experiment belongs.
- When an active revision includes `analysis`, assess its interim findings against the supplied evidence; unresolved branches are expected and do not imply completion.
- A Direction closure asks whether the optimization program is genuinely completed, exhausted, strategically deferred, concretely blocked, or reasonably abandoned. For `complete`, verify that the meaningful branches in its objective, applicability, plan, and stop conditions were addressed or shown to have insufficient remaining value. Do not treat one supported Experiment or one supported mechanism as completion of the Direction, and do not require a global supported/refuted verdict.
- Uncertain mechanism details or imprecise benefit magnitudes do not downgrade the review when they do not change admission or the lifecycle decision; narrow them in `scope_safe_interpretation` instead.
- Use `partially_supported` only when a decision-critical premise, stop-condition assessment, or closure decision remains genuinely uncertain but still has useful support. Use `unsupported` only when a decision-critical premise or action is contradicted or lacks enough evidence to justify the investigation or terminal decision.

## Experiment standard

- An Experiment asks whether the recorded observation and the resulting keep, repair, restore, or follow-up decision are trustworthy. Mark it `supported` when the candidate/source identity, relevant measurement or diagnostic, comparator when one is needed, and resulting action are sufficiently traceable and mutually consistent.
- Mechanistic interpretation is useful but not required for `supported`. A result may be supported even when the exact root cause, theoretical explanation, generality outside the measured scope, or precise expected magnitude remains uncertain. Preserve those limitations in `scope_safe_interpretation` and `next_check` without downgrading the whole Experiment.
- Use `partially_supported` when a decision-critical measurement, comparator, attribution, candidate identity, or action is incomplete or ambiguous but the Experiment still contains a useful observation. Use `unsupported` only when the core result or action is contradicted, untraceable, or not justified by the supplied evidence.

Unstarted `propose` and `revise` drafts are not sent for review. A `start` is the admission review before resources are committed; judge the effective Direction proposal supplied beside the lifecycle event.

Use exactly one verdict:

- `supported`: every decision-critical finding is supported under the applicable Direction or Experiment standard. Non-decision-critical limitations may remain and must be stated without lowering the verdict.
- `partially_supported`: no decision-critical finding is unsupported, but at least one decision-critical finding is only partially supported.
- `unsupported`: at least one decision-critical finding is unsupported.

Write exactly one JSON object to `review.json`:

```json
{
  "verdict": "supported | partially_supported | unsupported",
  "primary_issue": "one concise sentence",
  "findings": [
    {
      "claim": "claim being audited",
      "judgment": "supported | partially_supported | unsupported",
      "decision_critical": true,
      "reason": "evidence-based reason",
      "evidence": ["relative path or Gateway Record ID"]
    }
  ],
  "scope_safe_interpretation": "the strongest conclusion justified by the evidence",
  "next_check": "the smallest discriminating test",
  "confidence": "high | medium | low"
}
```

Keep at most six findings and include at least one decision-critical finding. Only decision-critical findings determine the overall verdict: `supported` requires all of them to be supported; `partially_supported` requires at least one partially supported decision-critical finding and no unsupported decision-critical finding; `unsupported` requires at least one unsupported decision-critical finding. Non-decision-critical findings may be partial or unsupported under an overall `supported` verdict. Do not include Markdown or any additional files.
