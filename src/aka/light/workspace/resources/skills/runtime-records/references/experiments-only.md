# Experiment-only requests

The Experiment module is enabled and the Direction module is disabled. Runtime maintains
`PROGRESS.md`, immutable `research-library/experiments/<experiment_id>/vN/experiment.json`
versions, and flat `records/<gateway_record_id>.json`. The greatest visible `vN` is current. Read
these generated files directly; do not edit them.

## Record an Experiment

Record a decision-useful interpretation after a real probe or measurement:

```json
{
  "name": "KSPLIT=2 complete-workload measurement",
  "intent": "Determine whether split KV pays for its reduction cost",
  "interpretation": "Long-KV regimes improved while short-KV remains unresolved",
  "gateway_record_ids": ["gateway-0123456789abcdef0123456789abcdef"],
  "references": ["scratch/ksplit-analysis.md", "profiles/ksplit2"]
}
```

Submit with `--kind record-experiment --request-file scratch/experiment.json`. The required fields
are `name`, `intent`, and `interpretation`; omit `direction_id`. `gateway_record_ids` and
`references` are independent optional arrays and may be empty. Runtime verifies every cited Record.
Execution facts, correctness, latency, source identity, and failures remain in root `records/`; do
not restate them as interpretation.

To correct or extend an existing local Experiment, submit the same request with its projected
`experiment_id`. Runtime keeps the stable ID, increments `revision`, reviews the revised claim, and
writes the next Agent-visible immutable version. Omitted `gateway_record_ids`, `references`, and
Wiki attribution are preserved; explicit empty arrays clear the corresponding optional evidence.
The Direction association cannot change. Shared peer Experiments are read-only, while every local
submission and Review remains in the private Supervisor audit history.

Each nonempty reference must already exist as a safe workspace-relative regular file or directory.
Runtime accepts at most 32 roots, 256 files, 16 MiB per file, and 64 MiB total, archives the bytes,
and rewrites the visible paths below `experiments/<id>/references/`. A missing or unsafe path rejects
the request without creating an Experiment.

When GPU Wiki knowledge materially affects the work, optional attribution uses
`wiki_usage_status`, `wiki_query_ids`, and `wiki_usage`. Each usage row contains the returned
`query_id`, `disposition`, and `wiki_id` or `cite`. Valid dispositions are `applied`,
`partially_applied`, `reference_only`, and `rejected`. Do not invent identifiers.

## Adopt an exact measured Kernel

Imported peer source history is available under refs such as
`atrex/T2`. Keep or commit local work before switching, inspect with
`git log`/`git diff`, and use `git switch -c <branch> <ref>` to continue a peer commit. A Git checkout
alone is not correctness or performance evidence; evaluate the resulting clean HEAD.

## Review an exact Kernel commit

Submit `{"kernel_commit_id":"<full reachable Agent Git commit>"}` through `--kind submit-kernel`.
First run a standard full Evaluate for that exact commit. Submission verifies the saved passing
result and checks the source contract and selected final-source policy. It neither selects an
Experiment nor launches Evaluate/ABBA, promotion, rollback, or handoff. Keep correctness and
performance claims in Experiment-linked Records. End the Session normally when useful work is done.
