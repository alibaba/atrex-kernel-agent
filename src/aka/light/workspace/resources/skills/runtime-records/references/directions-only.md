# Direction-only requests

The Direction module is enabled and the Experiment module is disabled. Runtime maintains
`PROGRESS.md`, `directions/<status>/<direction_id>.json`, and flat
`records/<gateway_record_id>.json`. Read these generated files directly; do not edit them.

## Propose, select, and progress work

A Direction is a durable optimization program around a bottleneck, workload regime, or architecture
objective and should contain several implementation variants or probes. A single implementation,
parameter choice, capability probe, or mechanism test remains a bounded step inside an existing
Direction rather than becoming a new Direction. Runtime/evaluator maintenance,
infrastructure operation, and unrelated documentation are outside the Direction queue.

Write mutation requests below `scratch/`. A proposal requires:

```json
{
  "action": "propose",
  "name": "Decode parallelism and KV-traffic architecture",
  "change_scope": "major_redesign",
  "objective": "Remove repeated KV work and expose enough parallelism across decode regimes",
  "applicability": "Short-query, long-KV and GQA-dominated workloads",
  "expected_benefit": "Reduce decode-side full-workload latency",
  "plan": ["Measure GQA packing", "Explore split-KV and combine families"]
}
```

`stop_conditions` is an optional array of non-empty strings, not one string. Optional genealogy uses `relationship` plus
`derived_from_direction_ids`; Experiment-derived genealogy is unavailable because this mode does
not expose Experiments. Relationship values are
`refinement`, `correction`, `combination`, `alternative`, `reimplementation`, or
`contradiction`. All cited IDs must already be visible. Refinement, correction, alternative,
reimplementation, and contradiction require exactly one effective parent; combination requires at
least two. Reuse an existing Direction ID while its objective and applicability are unchanged. Genealogy is descriptive:
the child does not inherit parent status, plan, evidence, Kernel source or priority. Check out or
branch from the relevant imported Git ref separately when needed. Runtime checks parent visibility, arity and cycles, not
the scientific correctness of the relationship label.

```bash
python3 tools/sandbox.py --kind update-direction --request-file scratch/direction.json
```

Before starting work, read `PROGRESS.md` and all relevant Direction files. Prefer expected impact,
including the value of the Kernel path an investigation could unlock, over implementation
convenience; high-value major redesigns should not be abandoned merely because they are difficult.

```json
{"action":"start","direction_id":"direction_11111111111111111111111111111111","analysis":"Highest expected impact and a bounded first probe"}
```

A trajectory may own multiple local active Directions. Keep concurrent work intentional, address
updates by Direction ID, and leave new ideas proposed until work actually starts. Reorder the durable queue with:

```json
{"direction_id":"direction_11111111111111111111111111111111","before_direction_id":"direction_22222222222222222222222222222222","reason":"Highest-information structural probe"}
```

Use `after_direction_id` instead, or omit both positions to move it to the end. Submit durable queue
changes through `prioritize-direction`. The `plan` belongs to the Agent: Runtime neither records
completed plan steps nor chooses the next step. Revise the Direction only when its objective, scope, or
remaining plan changes materially.

## Close or preserve a Direction

Lifecycle is `proposed --start--> in_progress`. Only `in_progress` may `complete`, `abandon`,
`defer`, or `block`; restart `deferred` or `blocked` work with `start`. `completed` and `abandoned`
are terminal and continued work requires a derived Direction.

```json
{
  "action": "defer",
  "direction_id": "direction_11111111111111111111111111111111",
  "analysis": "The capability probe is blocked by the current compiler",
  "reopen_condition": "The target compiler exposes the required instruction"
}
```

`complete`, `abandon`, `block`, and `defer` use the same core fields. Do not provide Experiment
support fields; in this mode their optional evidence field is `gateway_record_ids`. There is no
`hypothesis_status`; lifecycle state records whether the optimization program is active, paused,
completed, or abandoned. `defer` and `block` require `reopen_condition`. Use `defer` for a strategic pause and `block`
only for a concrete technical or external dependency that prevents progress. Continue multiple bounded
probe/candidate/measurement cycles while useful work remains; do not hand off the first promotable
checkpoint.

## Continue imported peer history

Peer source history is already imported under refs such as
`refs/atrex/shared/T2/e0001/head`. Keep or commit local work before switching, then use normal Git:

```bash
git log --oneline --decorate refs/atrex/shared/T2/e0001/head
git diff HEAD refs/atrex/shared/T2/e0001/head
git switch -c explore/t2-e0001 refs/atrex/shared/T2/e0001/head
```

Switching source does not establish correctness or performance; evaluate the resulting clean HEAD.

## Review an exact Kernel commit

Submit `{"kernel_commit_id":"<full reachable Agent Git commit>"}` through `--kind submit-kernel`.
First run a standard full Evaluate for that exact commit. Submission verifies the saved passing
result and checks the source contract and selected final-source policy. It does not consume
Direction evidence or launch Evaluate/ABBA, promotion, rollback, or handoff. Directions may remain
active across Episode boundaries. End the Session normally when useful work is done.
