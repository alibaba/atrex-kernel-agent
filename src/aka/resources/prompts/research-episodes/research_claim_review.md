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

Gateway Records need not bind a Kernel. Env can support environment facts, and self-contained
probes or failed requests can support the observations they actually contain. Do not reject a
record solely for lacking a Kernel binding. Require candidate/source identity only when the
claim depends on a particular implementation; environment facts alone do not prove its
correctness or performance.

Distinguish evidence for trying a candidate from evidence for declaring a mechanism established. Ask: if the mechanism explanation were wrong, would the cited observations still justify the proposed next experiment? If yes, keep the uncertainty explicit without making it decision-critical. Require stronger support when a mechanism claim is used to exclude a high-upside direction, generalize beyond measured workloads, or declare an optimization program exhausted. Correctness, measurement validity, and source/comparator identity still require evidence.

## Direction standard

- A Direction `propose` or `revise` asks whether the optimization objective is important enough to organize durable work, its applicability is clear, and its plan covers a meaningful strategy space that can contain multiple candidates and Experiments. A Direction is not one falsifiable hypothesis. The expected benefit is a prioritization estimate, not a fact that must already be proven.
- Reject a proposed Direction when it describes a single candidate, parameter choice, isolated probe, or one-step mechanism that belongs as an Experiment under an existing optimization program. Explain that classification error in `primary_issue` and use `next_check` to identify the existing Direction or the broader optimization objective under which the Experiment belongs.
- Admission does not require an Experiment, a Gateway Record, or a prior measurement. Judge whether the proposed investigation is worthwhile; do not demand proof of the expected benefit. If the proposal asserts an already measured result, check its cited evidence separately from the prospective plan.
- Keep the same Direction while the core optimization idea and applicable problem remain unchanged. Its revisions update the current assessment and remaining branches. Create a new Direction when the main optimization principle or target problem changes, not merely when a new implementation is tried. A successful Experiment adds a viable branch; it does not by itself complete the Direction.
- When a revision includes `analysis`, assess its interim findings against the supplied evidence; unresolved branches are expected and do not imply completion.
- A Direction closure asks whether the optimization program is genuinely completed, exhausted, strategically deferred, concretely blocked, or reasonably abandoned. For `complete`, verify that the meaningful branches in its objective, applicability, plan, and stop conditions were addressed or shown to have insufficient remaining value. Do not treat one supported Experiment or one supported mechanism as completion of the Direction, and do not require a global supported/refuted verdict.
- Uncertain mechanism details or imprecise benefit magnitudes do not downgrade the review when they do not change admission or the lifecycle decision; narrow them in `scope_safe_interpretation` instead.
- Use `partially_supported` only when a decision-critical premise, stop-condition assessment, or closure decision remains genuinely uncertain but still has useful support. Use `unsupported` only when a decision-critical premise or action is contradicted or lacks enough evidence to justify the investigation or terminal decision.

## Experiment standard

Use `implementation` for the concrete experiment method and `expectations` for scoped
`prediction`, `observation`, and `assessment` rows; `interpretation` explains implications and
uncertain causes. When predictions are present, check their comparison with observations, not whether the
prediction succeeded. A missed prediction can be a fully supported Experiment. Preserve mixed
Shape outcomes and distinguish observed effects from causal hypotheses. Compilation or
infrastructure failures leave performance predictions unresolved. Do not require mechanism
proof, a fabricated advance prediction, or another probe merely to complete this comparison.
A prediction reconstructed after measurement must not be presented as a prior forecast. During
revision, use supplied earlier evidence to preserve the original prediction and outcome; new
observations may change the assessment without rewriting history. These rules guide the existing
review; they add no required fields, review calls, or lifecycle gates.

An Experiment is one concrete investigative attempt: a candidate implementation, diagnostic probe, microbenchmark, correctness test, profiling experiment, or controlled comparison. It records what was tried and what the observations establish. Keep its identity while the specific question and experimental approach remain unchanged. Use revision for repeat measurements, corrections, explanation, or repairs within that attempt. Create a new Experiment when the experimental question or method, implementation strategy, key dataflow, or combination changes materially, even within the same Direction. Several commits, parameter trials, and diagnostic Records may belong to one attempt; one commit or tool call does not require one Experiment. A complete Kernel or full Evaluate is not required. Preserve earlier attempts rather than replacing them with the latest conclusion. Do not require one commit per Experiment or an extra Review to choose its identity.

- An Experiment asks whether the recorded observation and the resulting keep, repair, restore, or follow-up decision are trustworthy. Mark it `supported` when the relevant measurement or diagnostic, any candidate/source identity or comparator needed for the claim, and resulting action are sufficiently traceable and mutually consistent.
- An Experiment may summarize several related candidates, repairs, or parameter trials within the same experimental question and approach. Distinct experimental methods or implementation strategies belong to separate attempts under the same high-level Direction. Review the observations, their measured scope, and the proposed next action; do not require one record per candidate or a complete mechanism explanation. For example, a traceable observation that V-split=4 helps measured long sequences but regresses short ones can support testing workload dispatch while the cause remains unknown. It does not establish benefits outside that scope.
- Mechanistic interpretation is useful but not required for `supported`. A result may be supported even when the exact root cause, theoretical explanation, generality outside the measured scope, or precise expected magnitude remains uncertain. Preserve those limitations in `scope_safe_interpretation` without downgrading the whole Experiment or requiring mechanism-isolating probes that would not change the next decision.
- Use `partially_supported` when a decision-critical measurement, comparator, attribution, candidate identity, or action is incomplete or ambiguous but the Experiment still contains a useful observation. Use `unsupported` only when the core result or action is contradicted, untraceable, or not justified by the supplied evidence.

`propose` runs admission Review; every `revise` reviews the effective Direction supplied beside the revision event. `start` only changes lifecycle and is not sent for review.

Runtime accepts partial support for starting or pausing work (`defer`/`block`). Only `complete`
and `abandon` require `supported` Review of the terminal decision. Preserve a partial verdict
when that decision remains uncertain; do not upgrade it merely to permit the transition.

Use `next_check` only for a check that could change the next implementation choice or resolve a premise needed for the submitted lifecycle decision. Otherwise write `No additional check needed for the next decision.` A narrower interpretation may resolve an overclaim without new measurement. Do not manufacture follow-up work merely to improve a review label; a partial review can preserve useful observations while the Agent continues a justified high-value trial.

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
  "next_check": "a decision-relevant check, or No additional check needed for the next decision.",
  "confidence": "high | medium | low"
}
```

Keep at most six findings and include at least one decision-critical finding. Only decision-critical findings determine the overall verdict: `supported` requires all of them to be supported; `partially_supported` requires at least one partially supported decision-critical finding and no unsupported decision-critical finding; `unsupported` requires at least one unsupported decision-critical finding. Non-decision-critical findings may be partial or unsupported under an overall `supported` verdict. Do not include Markdown or any additional files.
